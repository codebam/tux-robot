import { Bot, Context, webhookCallback } from 'grammy';
import { WorkflowEntrypoint, WorkflowEvent, WorkflowStep } from 'cloudflare:workers';
import { HistoryManager, getBalance } from './lib/history_manager.js';
import { markdownToHtml, fetchTool, wikipediaTool, createTavilySearchTool } from './lib/utils.js';
import { streamAiResponseToTelegram } from './lib/ai.js';

export interface Environment {
	SECRET_TELEGRAM_API_TOKEN: string;
	AI: Ai;
	R2: R2Bucket;
	CONVERSATION_HISTORY: KVNamespace;
	AI_WORKFLOW: Workflow;
	TAVILY_API_KEY: string;
}

export interface Task {
	type: 'code' | 'message' | 'business_message' | 'photo' | 'gen_photo' | 'voice' | 'tool_call';
	updateId?: number;
	updateType?: string;
	guestQueryId?: string;
	businessConnectionId?: string;
	prompt: string;
	userId?: number | string;
	senderId?: number;
	chatId?: string;
	threadId?: number;
	messageId?: number;
	history?: { role: string; content: string }[];
	modelId?: string;
	fileId?: string;
	systemPrompt?: string;
	telegramToken?: string;
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	tools?: any[];
	stream?: boolean;
}

type MyContext = Context & {
	env: Environment;
	executionCtx: ExecutionContext;
};

export const SYSTEM_PROMPTS = {
	TUX_ROBOT:
		'You are a friendly assistant named TuxRobot. You have access to an HTTP fetch tool and a web search tool. If a user asks you to get data from an API, look up a profile, or visit a website, you MUST execute the fetch tool yourself to get the data. You can perform web searches using the `search` tool. DO NOT write code for the user to run; just get the data and summarize it. If the user replies with only a single word, sticker, or emoji, respond with no more than one short paragraph. Always keep replies below 4096 characters.',
	BUSINESS_MODE:
		'You are {owner_name}. You are replying to a message from a customer, friend, or contact on Telegram. Be helpful, natural, and concise. Do not mention you are an AI or a bot. Respond exactly as {owner_name}. Do not use any generic placeholders like [Business Name], [Your Name], or similar in your response.'
};

export const AVAILABLE_MODELS: Record<
	string,
	{ id: string; cost: number; supportsTools?: boolean }
> = {
	hermes: { id: '@hf/nousresearch/hermes-2-pro-mistral-7b', cost: 5, supportsTools: true },
	gemma4: { id: '@cf/google/gemma-4-26b-a4b-it', cost: 10, supportsTools: true },
	'google/gemini-3-flash': { id: 'google/gemini-3-flash', cost: 15, supportsTools: true },
	'google/gemini-3.1-flash-lite': {
		id: 'google/gemini-3.1-flash-lite',
		cost: 10,
		supportsTools: true
	},
	'google/gemini-3.1-pro': { id: 'google/gemini-3.1-pro', cost: 80, supportsTools: true },
	'kimi-k2.6': { id: '@cf/moonshotai/kimi-k2.6', cost: 40, supportsTools: true },
	'glm-4.7-flash': { id: '@cf/zai-org/glm-4.7-flash', cost: 10, supportsTools: true },
	'llama-3.3-70b': {
		id: '@cf/meta/llama-3.3-70b-instruct-fp8-fast',
		cost: 40,
		supportsTools: true
	},
	'deepseek-r1-32b': {
		id: '@cf/deepseek-ai/deepseek-r1-distill-qwen-32b',
		cost: 60,
		supportsTools: false
	},
	'nemotron-3': { id: '@cf/nvidia/nemotron-3-120b-a12b', cost: 100, supportsTools: true }
};

async function getBusinessOwnerData(
	ctx: MyContext,
	connectionId: string
): Promise<{ id: number; name: string; username?: string } | null> {
	let ownerData = await ctx.env.CONVERSATION_HISTORY.get<{ id: number; name: string; username?: string }>(
		`business_connection:${connectionId}`,
		'json'
	);
	if (ownerData && ownerData.username !== undefined) {
		console.log(`[getBusinessOwnerData] Cache HIT for connection ${connectionId}:`, JSON.stringify(ownerData));
	} else {
		console.log(`[getBusinessOwnerData] Cache MISS or stale entry for connection ${connectionId}. Fetching from Telegram API...`);
		try {
			const response = await fetch(`https://api.telegram.org/bot${ctx.env.SECRET_TELEGRAM_API_TOKEN}/getBusinessConnection?business_connection_id=${connectionId}`);
			console.log(`[getBusinessOwnerData] Telegram API response status: ${response.status}`);
			if (response.status === 200) {
				const json = (await response.json()) as {
					ok: boolean;
					result?: {
						user?: { first_name: string; username?: string; id: number };
						user_chat_id?: number;
					};
				};
				console.log(`[getBusinessOwnerData] Telegram API returned JSON:`, JSON.stringify(json));
				if (json.ok && json.result) {
					const id = json.result.user?.id || json.result.user_chat_id;
					const name = json.result.user?.first_name || 'the business owner';
					const username = json.result.user?.username;
					if (id) {
						ownerData = { id, name, username };
						console.log(`[getBusinessOwnerData] Successfully resolved owner: id=${id}, name=${name}, username=${username ?? ''}. Caching in KV...`);
						await ctx.env.CONVERSATION_HISTORY.put(
							`active_connection:${id}`,
							connectionId
						);
						await ctx.env.CONVERSATION_HISTORY.put(
							`business_connection:${connectionId}`,
							JSON.stringify(ownerData)
						);
					} else {
						console.error(`[getBusinessOwnerData] Failed to resolve owner ID from result:`, JSON.stringify(json.result));
					}
				} else {
					console.error(`[getBusinessOwnerData] Telegram API returned ok=false or missing result:`, JSON.stringify(json));
				}
			} else {
				console.error(`[getBusinessOwnerData] Telegram API call failed. Status: ${response.status}`);
			}
		} catch (e) {
			console.error('[getBusinessOwnerData] Failed to fetch business connection:', e);
		}
	}
	return ownerData;
}

async function chargeStars(
	ctx: MyContext,
	task: Task,
	amountOverride?: number
) {
	const historyManager = new HistoryManager(ctx.env.CONVERSATION_HISTORY);
	let userId: number | string | undefined = ctx.from?.id;
	let billingUserId = ctx.from?.id;

	if (ctx.update.business_message) {
		const connectionId = ctx.update.business_message?.business_connection_id;
		const customerId = ctx.update.business_message?.chat.id;
		if (connectionId && customerId) {
			userId = `business:${connectionId}:${customerId}`;
			const ownerData = await getBusinessOwnerData(ctx, connectionId);
			if (ownerData?.id) {
				billingUserId = ownerData.id;
			}
		}
	}

	if (!userId || userId === ctx.me.id) {
		console.log(`Skipping chargeStars: userId=${userId}, botId=${ctx.me.id}`);
		return;
	}

	task.userId = userId;
	task.senderId = ctx.from?.id;
	task.chatId = ctx.chat?.id.toString();
	task.updateId = ctx.update.update_id;
	task.messageId = ctx.message?.message_id ?? ctx.update.business_message?.message_id;
	task.updateType = Object.keys(ctx.update).find((k) => k !== 'update_id');
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	task.guestQueryId = (ctx.update as any).guest_message?.guest_query_id;
	task.businessConnectionId = ctx.update.business_message?.business_connection_id?.toString();
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	task.threadId = ctx.message?.message_thread_id ?? (ctx.update as any).guest_message?.message_thread_id;
	
	const balanceKey = `balance:${String(billingUserId)}`;
	const balance = await getBalance(billingUserId || 0, ctx.env.CONVERSATION_HISTORY);

	const modelPreference =
		(await ctx.env.CONVERSATION_HISTORY.get<string>(`model:${String(billingUserId)}`)) ?? 'gemma4';
	const modelConfig = AVAILABLE_MODELS[modelPreference] ?? AVAILABLE_MODELS.gemma4;

	if (task.type === 'tool_call' && !modelConfig.supportsTools) {
		task.modelId = AVAILABLE_MODELS.gemma4.id;
	} else {
		task.modelId = modelConfig.id;
	}

	const amount = amountOverride ?? modelConfig.cost;

	if (balance >= amount) {
		await ctx.replyWithChatAction('typing');
		await ctx.env.CONVERSATION_HISTORY.put(balanceKey, JSON.stringify(balance - amount));
		task.telegramToken = ctx.env.SECRET_TELEGRAM_API_TOKEN;

		if (ctx.update.business_message) {
			if (!task.systemPrompt) {
				task.systemPrompt = SYSTEM_PROMPTS.BUSINESS_MODE;
			}
		} else {
			const customPrompt = await ctx.env.CONVERSATION_HISTORY.get(`prompt:${String(userId)}`);
			if (customPrompt) {
				task.systemPrompt = customPrompt;
			} else if (!task.systemPrompt) {
				task.systemPrompt = SYSTEM_PROMPTS.TUX_ROBOT;
			}
		}

		if (!task.history && userId) {
			task.history = await historyManager.getHistory(userId, task.threadId);
		}

		ctx.executionCtx.waitUntil(
			ctx.env.AI_WORKFLOW.create({ params: task }).catch(console.error)
		);
	} else {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		if (ctx.update.business_message || (ctx.update as any).guest_message) {
			await ctx.reply(
				'Insufficient balance. Please go to direct messages and use /load to top up your Stars.'
			);
		} else {
			const taskId = crypto.randomUUID();
			await ctx.env.CONVERSATION_HISTORY.put(`task:${taskId}`, JSON.stringify(task), {
				expirationTtl: 3600
			});
			await ctx.replyWithInvoice(
				'AI Generation',
				'Charge for AI message generation',
				taskId,
				'XTR',
				[{ label: 'Stars', amount }]
			);
		}
	}
}

function setupBot(bot: Bot<MyContext>, env: Environment, executionCtx: ExecutionContext) {
	bot.use(async (ctx, next) => {
		ctx.env = env;
		ctx.executionCtx = executionCtx;
		await next();
	});

	bot.use(async (ctx, next) => {
		const token = ctx.env.SECRET_TELEGRAM_API_TOKEN;
		const botTtl = (await ctx.env.CONVERSATION_HISTORY.get<number>(`ttl:${token.slice(0, 10)}`, 'json')) ?? 2;

		const isSelf = ctx.from?.id === ctx.me.id;
		const counterKey = `ttl_counter:${ctx.chat?.id}:${token.slice(0, 10)}`;

		if (isSelf) {
			const count = (await ctx.env.CONVERSATION_HISTORY.get<number>(counterKey, 'json')) ?? 0;
			if (count >= botTtl) {
				console.log(`TTL exceeded for chat ${ctx.chat?.id}. Blocking update.`);
				return;
			}
			await ctx.env.CONVERSATION_HISTORY.put(counterKey, JSON.stringify(count + 1), {
				expirationTtl: 3600,
			});
		} else {
			await ctx.env.CONVERSATION_HISTORY.delete(counterKey);
		}
		await next();
	});

	bot.command('start', async (ctx) => {
		await ctx.reply(
			'Welcome! Here are my commands:\n' +
				'/balance - Check your current Star balance\n' +
				'/load <amount> - Top up your balance with Telegram Stars\n' +
				'/photo <prompt> - Generate an image (100 Stars)\n' +
				'/model <name> - Switch AI model and see costs\n' +
				'/ttl <1-5> - Set the TTL for bot-to-bot responses\n' +
				'/code <prompt> - Generate code snippets\n' +
				'/prompt <"prompt"> - Set your custom system prompt (use "" or reset to clear)\n' +
				'/facts <"facts"> - Set facts about yourself for business mode (use "" or reset to clear)\n' +
				'/request <prompt> - Make arbitrary API requests (uses fetch tool)\n' +
				'<prompt> - Generate text (may use tools if supported by model)\n' +
				'Send a voice note - Transform your bot into a voice assistant (+20 Stars)\n' +
				'/clear - Clear your conversation history\n\n' +
				'New users start with 200 free credits!\n\n' +
				'Click the button below to open the Web App!',
			{
				reply_markup: {
					inline_keyboard: [[{ text: 'Open Web App', web_app: { url: 'https://tux-robot.codebam.ca' } }]],
				},
			},
		);
	});

	bot.command('balance', async (ctx) => {
		if (ctx.from) {
			const balance = await getBalance(ctx.from.id, ctx.env.CONVERSATION_HISTORY);
			await ctx.reply(`Your current balance is ${String(balance)} Stars.`);
		}
	});

	bot.command('load', async (ctx) => {
		const amount = parseInt(ctx.match || '0');
		if (isNaN(amount) || amount <= 0 || amount > 1000) {
			await ctx.reply('Please specify an amount between 1 and 1000 Stars. Example: /load 100');
		} else {
			await ctx.replyWithInvoice('Stars Top-up', `Purchase ${String(amount)} Stars`, `load:${String(amount)}`, 'XTR', [
				{ label: 'Stars', amount },
			]);
		}
	});

	bot.command('clear', async (ctx) => {
		if (ctx.from) {
			const historyManager = new HistoryManager(ctx.env.CONVERSATION_HISTORY);
			let historyUserId: number | string = ctx.from.id;
			if (ctx.update.business_message) {
				const connectionId = ctx.update.business_message?.business_connection_id;
				const customerId = ctx.update.business_message?.chat.id;
				if (connectionId && customerId) {
					historyUserId = `business:${connectionId}:${customerId}`;
				}
			}
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			const threadId = ctx.message?.message_thread_id ?? (ctx.update as any).guest_message?.message_thread_id;
			await historyManager.clearHistory(historyUserId, threadId);
			await ctx.reply('History cleared');
		}
	});

	bot.command('code', async (ctx) => {
		const prompt = ctx.match;
		if (prompt) {
			await chargeStars(ctx, { type: 'code', prompt });
		}
	});

	bot.command('ttl', async (ctx) => {
		const newTtl = parseInt(ctx.match || '0');
		const token = ctx.env.SECRET_TELEGRAM_API_TOKEN;
		if (newTtl >= 1 && newTtl <= 5) {
			await ctx.env.CONVERSATION_HISTORY.put(`ttl:${token.slice(0, 10)}`, JSON.stringify(newTtl));
			await ctx.reply(`TTL set to ${newTtl}`);
		} else {
			const currentTtl = (await ctx.env.CONVERSATION_HISTORY.get<number>(`ttl:${token.slice(0, 10)}`, 'json')) ?? 2;
			await ctx.reply(`Invalid TTL. Please use a value between 1 and 5. Current TTL: ${currentTtl}`);
		}
	});

	bot.command('model', async (ctx) => {
		if (ctx.from) {
			const modelKey = `model:${String(ctx.from.id)}`;
			const selectedModel = ctx.match?.toLowerCase();
			if (selectedModel) {
				if (selectedModel in AVAILABLE_MODELS) {
					await ctx.env.CONVERSATION_HISTORY.put(modelKey, selectedModel);
					await ctx.reply(`Model updated to <b>${selectedModel}</b>.`, { parse_mode: 'HTML' });
				} else {
					await ctx.reply(`Invalid model. Available models:\n${Object.keys(AVAILABLE_MODELS).join('\n')}`);
				}
			} else {
				const currentModel = (await ctx.env.CONVERSATION_HISTORY.get<string>(modelKey)) ?? 'gemma4';
				await ctx.reply(
					`Current model: <b>${currentModel}</b>\n\n` +
						`Available models:\n` +
						Object.entries(AVAILABLE_MODELS)
							.map(([name, cfg]) => `- <code>${name}</code> (${String(cfg.cost)} Stars)`)
							.join('\n'),
					{ parse_mode: 'HTML' },
				);
			}
		}
	});

	bot.command('prompt', async (ctx) => {
		if (ctx.from) {
			let promptValue = ctx.match.trim();
			if (promptValue === 'reset' || promptValue === '""' || promptValue === "''" || promptValue === '') {
				await ctx.env.CONVERSATION_HISTORY.delete(`prompt:${String(ctx.from.id)}`);
				await ctx.reply('System prompt reset to default.');
			} else {
				if (
					(promptValue.startsWith('"') && promptValue.endsWith('"')) ||
					(promptValue.startsWith("'") && promptValue.endsWith("'"))
				) {
					promptValue = promptValue.substring(1, promptValue.length - 1);
				}
				await ctx.env.CONVERSATION_HISTORY.put(`prompt:${String(ctx.from.id)}`, promptValue);
				await ctx.reply(`System prompt updated to:\n\n${promptValue}`);
			}
		}
	});

	bot.command('facts', async (ctx) => {
		if (ctx.from) {
			let factsValue = ctx.match.trim();
			const userId = ctx.from.id;
			if (factsValue === 'reset' || factsValue === '""' || factsValue === "''" || factsValue === '') {
				await ctx.env.CONVERSATION_HISTORY.delete(`business_facts:${String(userId)}`);
				const connectionId = await ctx.env.CONVERSATION_HISTORY.get(`active_connection:${userId}`);
				if (connectionId) {
					const ownerData = await ctx.env.CONVERSATION_HISTORY.get<{ id: number; name: string; username?: string }>(
						`business_connection:${connectionId}`,
						'json',
					);
					if (ownerData) {
						if (ownerData.username) {
							await ctx.env.CONVERSATION_HISTORY.delete(`business_facts:${ownerData.username}`);
						}
						if (ownerData.name) {
							await ctx.env.CONVERSATION_HISTORY.delete(`business_facts:${ownerData.name}`);
						}
					}
				}
				await ctx.reply('Business facts cleared.');
			} else {
				if (
					(factsValue.startsWith('"') && factsValue.endsWith('"')) ||
					(factsValue.startsWith("'") && factsValue.endsWith("'"))
				) {
					factsValue = factsValue.substring(1, factsValue.length - 1);
				}
				await ctx.env.CONVERSATION_HISTORY.put(`business_facts:${String(userId)}`, factsValue);
				const connectionId = await ctx.env.CONVERSATION_HISTORY.get(`active_connection:${userId}`);
				if (connectionId) {
					const ownerData = await ctx.env.CONVERSATION_HISTORY.get<{ id: number; name: string; username?: string }>(
						`business_connection:${connectionId}`,
						'json',
					);
					if (ownerData) {
						if (ownerData.username) {
							await ctx.env.CONVERSATION_HISTORY.put(`business_facts:${ownerData.username}`, factsValue);
						}
						if (ownerData.name) {
							await ctx.env.CONVERSATION_HISTORY.put(`business_facts:${ownerData.name}`, factsValue);
						}
					}
				}
				await ctx.reply(`Business facts updated to:\n\n${factsValue}`);
			}
		}
	});

	bot.command('request', async (ctx) => {
		const prompt = ctx.match;
		if (!prompt) {
			await ctx.reply('Please provide a request. Example: /request what is the weather in San Francisco?');
			return;
		}
		await chargeStars(ctx, { type: 'tool_call', prompt, tools: [fetchTool, wikipediaTool] });
	});

	bot.on('message:document', async (ctx) => {
		const fileId = ctx.message.document.file_id;
		const file = await ctx.api.getFile(fileId);
		const fileUrl = `https://api.telegram.org/file/bot${ctx.env.SECRET_TELEGRAM_API_TOKEN}/${file.file_path}`;
		const fileResponse = await fetch(fileUrl);
		const id = crypto.randomUUID().slice(0, 5);
		await ctx.env.R2.put(id, await fileResponse.arrayBuffer());
		await ctx.reply(`https://r2.seanbehan.ca/${id}`);
	});

	bot.on('pre_checkout_query', async (ctx) => {
		await ctx.answerPreCheckoutQuery(true);
	});

	bot.on('message:successful_payment', async (ctx) => {
		const payment = ctx.message.successful_payment;
		const payload = payment.invoice_payload;
		const userId = ctx.from?.id;
		if (!userId) return;

		if (payload.startsWith('load:')) {
			const amount = parseInt(payload.split(':')[1]);
			const balanceKey = `balance:${String(userId)}`;
			const balance = (await ctx.env.CONVERSATION_HISTORY.get<number>(balanceKey, 'json')) ?? 0;
			await ctx.env.CONVERSATION_HISTORY.put(balanceKey, JSON.stringify(balance + amount));
			await ctx.reply(`Successfully loaded ${String(amount)} Stars! New balance: ${String(balance + amount)} Stars.`);
			return;
		}

		const taskId = payload;
		const task = await ctx.env.CONVERSATION_HISTORY.get<Task>(`task:${taskId}`, 'json');
		if (!task) {
			await ctx.reply('Error: Task not found');
			return;
		}
		task.telegramToken = ctx.env.SECRET_TELEGRAM_API_TOKEN;
		ctx.executionCtx.waitUntil(ctx.env.AI_WORKFLOW.create({ params: task }).catch(console.error));
		await ctx.env.CONVERSATION_HISTORY.delete(`task:${taskId}`);
	});

	bot.on('message:photo', async (ctx) => {
		const photo = ctx.message.photo;
		const fileId = photo[photo.length - 1].file_id;
		const prompt = ctx.message.caption ?? 'Please describe this image';
		await chargeStars(ctx, { type: 'photo', prompt, fileId }, 10);
	});

	bot.on('message:voice', async (ctx) => {
		const fileId = ctx.message.voice.file_id;
		await chargeStars(ctx, { type: 'voice', prompt: '', fileId });
	});

	bot.on('inline_query', async (ctx) => {
		const query = ctx.inlineQuery.query;
		if (!query.endsWith('.') && !query.endsWith('?')) {
			await ctx.answerInlineQuery([
				{
					type: 'article',
					id: 'complete_sentence',
					title: 'Please complete your sentence',
					input_message_content: {
						message_text: 'End your sentence with a period (.) or question mark (?) to get an AI response',
						parse_mode: 'HTML',
					},
				},
			]);
			return;
		}
		const messages = [
			{ role: 'system', content: SYSTEM_PROMPTS.TUX_ROBOT },
			{ role: 'user', content: query },
		];
		try {
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			const rawResponse = await ctx.env.AI.run('@cf/meta/llama-3.2-11b-vision-instruct' as any, {
				messages,
				max_completion_tokens: 100,
			});
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			const aiResponse = rawResponse as any;
			if (aiResponse.response) {
				await ctx.answerInlineQuery([
					{
						type: 'article',
						id: 'ai_response',
						title: 'AI Response',
						input_message_content: {
							message_text: await markdownToHtml(aiResponse.response),
							parse_mode: 'HTML',
						},
					},
				]);
			}
		} catch {
			/* ignore */
		}
	});

	bot.on('business_connection', async (ctx) => {
		const connection = ctx.update.business_connection;
		if (connection) {
			const ownerName = connection.user.first_name;
			const username = connection.user.username;
			const ownerId = connection.user.id;
			await ctx.env.CONVERSATION_HISTORY.put(`active_connection:${ownerId}`, connection.id);
			await ctx.env.CONVERSATION_HISTORY.put(
				`business_connection:${connection.id}`,
				JSON.stringify({
					id: ownerId,
					name: ownerName || 'the business owner',
					username: username,
				}),
			);
		}
	});

	bot.on('business_message', async (ctx) => {
		await ctx.replyWithChatAction('typing');
		const businessMessage = ctx.update.business_message!;
		const photo = businessMessage.photo;
		const fileId = photo ? photo[photo.length - 1].file_id : '';
		let prompt = businessMessage.text ?? businessMessage.caption ?? '';
		if (businessMessage.reply_to_message) {
			const reply = businessMessage.reply_to_message;
			const replyText = reply.text ?? reply.caption ?? '';
			if (replyText) {
				prompt = `Context of the message I am replying to: "${replyText}"\n\nMy message: ${prompt}`;
			}
		}

		let ownerName = 'the business owner';
		let ownerId: number | undefined;
		let username: string | undefined;
		const connectionId = businessMessage.business_connection_id;
		if (connectionId) {
			const ownerData = await getBusinessOwnerData(ctx, connectionId);
			if (ownerData) {
				ownerName = ownerData.name;
				ownerId = ownerData.id;
				username = ownerData.username;
			}
		}

		let systemPrompt = SYSTEM_PROMPTS.BUSINESS_MODE.replaceAll('{owner_name}', ownerName);
		let facts: string | null = null;
		if (ownerId) {
			facts = await ctx.env.CONVERSATION_HISTORY.get(`business_facts:${String(ownerId)}`);
		}
		if (!facts && username) {
			facts = await ctx.env.CONVERSATION_HISTORY.get(`business_facts:${username}`);
		}
		if (!facts && ownerName && ownerName !== 'the business owner') {
			facts = await ctx.env.CONVERSATION_HISTORY.get(`business_facts:${ownerName}`);
		}

		if (facts) {
			systemPrompt += `\n\nHere are some facts about yourself (${ownerName}) that you should keep in mind and use to answer accurately if relevant:\n${facts}`;
		}

		await chargeStars(ctx, {
			type: 'business_message',
			prompt,
			fileId,
			systemPrompt,
		});
	});

	bot.on('message:text', async (ctx) => {
		// Guest message logic
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		const guestMessage = (ctx.update as any).guest_message;
		if (guestMessage) {
			let prompt = guestMessage.text?.toString() ?? '';
			const token = ctx.env.SECRET_TELEGRAM_API_TOKEN;
			let botUsername = await ctx.env.CONVERSATION_HISTORY.get(`bot_username:${token.slice(0, 10)}`);
			if (!botUsername) {
				const me = await ctx.api.getMe();
				botUsername = me.username;
				await ctx.env.CONVERSATION_HISTORY.put(`bot_username:${token.slice(0, 10)}`, botUsername, {
					expirationTtl: 86400,
				});
			}
			const isMentioned = guestMessage.entities?.some(
				(e: { type: string; offset: number; length: number }) =>
					e.type === 'mention' && prompt.substring(e.offset, e.offset + e.length).toLowerCase() === `@${botUsername?.toLowerCase()}`,
			);
			if (!isMentioned) return;

			if (guestMessage.reply_to_message) {
				const reply = guestMessage.reply_to_message;
				const replyText = reply.text ?? reply.caption ?? '';
				if (replyText) {
					prompt = `Context of the message I am replying to: "${replyText}"\n\nMy message: ${prompt}`;
				}
			}
			await chargeStars(ctx, { type: 'message', prompt });
			return;
		}

		// Regular message logic
		let prompt = ctx.message.text;
		if (ctx.message.reply_to_message) {
			const reply = ctx.message.reply_to_message;
			const replyText = reply.text ?? reply.caption ?? '';
			if (replyText) {
				prompt = `Context of the message I am replying to: "${replyText}"\n\nMy message: ${prompt}`;
			}
		}
		await chargeStars(ctx, { type: 'message', prompt });
	});
}

export class AIWorkflow extends WorkflowEntrypoint<Environment, any> {
	async run(event: WorkflowEvent<any>, step: WorkflowStep): Promise<void> {
		const task = event.payload;
		const env = this.env;

		const config = await step.do('Initialize Context', async () => {
			const messages = [
				{ role: 'system', content: task.systemPrompt || 'You are a helpful assistant.' },
				...(task.history || []),
				{ role: 'user', content: task.prompt },
			];
			const modelId = task.modelId || '@cf/meta/llama-3.1-8b-instruct-fp8';
			return { messages, modelId };
		});

		let stepResult: { content: string; toolExecutions?: any[] };

		try {
			stepResult = await step.do('Stream AI Response', async () => {
				const toolExecutions: any[] = [];

				const wrapTool = (originalTool: any) => {
					return {
						...originalTool,
						function: async (args: any) => {
							try {
								const result = await originalTool.function(args);
								const resultStr = String(result);
								toolExecutions.push({
									tool: originalTool.name,
									arguments: args,
									status: 'success',
									output: resultStr.length > 1000 ? resultStr.slice(0, 1000) + '... (truncated)' : resultStr,
								});
								return result;
							} catch (error) {
								toolExecutions.push({
									tool: originalTool.name,
									arguments: args,
									status: 'error',
									error: String(error),
								});
								throw error;
							}
						},
					};
				};

				const wrappedFetch = wrapTool(fetchTool);
				const wrappedWikipedia = wrapTool(wikipediaTool);
				const tavilyTool = createTavilySearchTool(env.TAVILY_API_KEY);
				const wrappedTavily = wrapTool(tavilyTool);

				const botInstance = new Bot<MyContext>(env.SECRET_TELEGRAM_API_TOKEN);

				const responseContent = await streamAiResponseToTelegram(
					{
						env,
						api: botInstance.api,
						reply: (text: string, options: any) => botInstance.api.sendMessage(task.chatId, text, options),
					},
					env.AI,
					config.modelId,
					config.messages,
					task,
					[wrappedFetch, wrappedWikipedia, wrappedTavily],
				);

				return {
					content: responseContent,
					toolExecutions: toolExecutions.length > 0 ? toolExecutions : undefined,
				};
			});
		} catch (e) {
			console.error('[Workflow] Error during streaming step execution:', e);
			throw e;
		}

		if (task.userId && stepResult.content) {
			await step.do('Save Conversation History', async () => {
				const historyManager = new HistoryManager(env.CONVERSATION_HISTORY);
				await historyManager.addMessage(task.userId, task.prompt, stepResult.content, task.threadId);
			});
		}
	}
}

export default {
	async fetch(request: Request, env: Environment, executionCtx: ExecutionContext): Promise<Response> {
		const bot = new Bot<MyContext>(env.SECRET_TELEGRAM_API_TOKEN);
		setupBot(bot, env, executionCtx);

		if (request.method === 'GET') {
			const url = new URL(request.url);
			if (url.searchParams.get('command') === 'set') {
				const token = env.SECRET_TELEGRAM_API_TOKEN;
				const webhookUrl = `${url.origin}${url.pathname}`;
				const telegramUrl = `https://api.telegram.org/bot${token}/setWebhook`;

				const params = new URLSearchParams({
					url: webhookUrl,
					max_connections: '40',
					allowed_updates: JSON.stringify([
						'message',
						'edited_message',
						'callback_query',
						'inline_query',
						'guest_message',
						'business_message',
						'business_connection',
						'pre_checkout_query',
					]),
					drop_pending_updates: 'true',
				});

				const res = await fetch(`${telegramUrl}?${params.toString()}`);
				return new Response(JSON.stringify(await res.json()), {
					headers: { 'Content-Type': 'application/json' },
					status: res.status,
				});
			}
		}
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		return (webhookCallback(bot, 'cloudflare-mod', {
			onTimeout: 'return',
		}) as any)(request, env, executionCtx);
	},
};
