import { Bot, Api, Context, webhookCallback, session, GrammyError, HttpError, InputFile } from 'grammy';
import { WorkflowEntrypoint, WorkflowStep, WorkflowEvent } from 'cloudflare:workers';
import { autoRetry } from '@grammyjs/auto-retry';
import { stream, type StreamFlavor } from '@grammyjs/stream';
import { CommandGroup, type CommandsFlavor } from '@grammyjs/commands';
import { conversations, createConversation, type Conversation, type ConversationFlavor } from '@grammyjs/conversations';
import { KvAdapter } from '@grammyjs/storage-cloudflare';
import type { KVNamespace as CfKVNamespace } from '@cloudflare/workers-types';
import { HistoryManager, getBalance, markdownToMarkdownV2, SYSTEM_PROMPTS, AVAILABLE_MODELS, type Task, type Environment, type GeminiPart, type ChatMessage } from '@codebam/shared';
import { fetchTool, wikipediaTool, createTavilySearchTool, createSandboxTool } from './lib/utils.js';
import { createTelegramFileReaderTool, createTelegramFileSearchTool } from './lib/documentTool.js';
import { streamAiResponseToTelegram, customRunWithTools } from './lib/ai.js';

export { Sandbox } from '@cloudflare/sandbox';

function arrayBufferToBase64(buffer: ArrayBuffer): string {
	let binary = '';
	const bytes = new Uint8Array(buffer);
	const len = bytes.byteLength;
	for (let i = 0; i < len; i++) {
		binary += String.fromCharCode(bytes[i]);
	}
	return btoa(binary);
}

type BaseContext = CommandsFlavor &
	Context & {
		env: Environment;
		executionCtx: ExecutionContext;
		session: Record<string, unknown>;
	};

type MyContext = StreamFlavor<BaseContext & ConversationFlavor<BaseContext>>;

type MyConversation = Conversation<MyContext, MyContext>;

async function getBusinessOwnerData(
	api: Api,
	env: Environment,
	connectionId: string
): Promise<{ id: number; name: string; username?: string } | null> {
	let ownerData = await env.CONVERSATION_HISTORY.get<{ id: number; name: string; username?: string }>(
		`business_connection:${connectionId}`,
		'json'
	);
	if (ownerData) {
		console.log(`[getBusinessOwnerData] Cache HIT for connection ${connectionId}:`, JSON.stringify(ownerData));
	} else {
		console.log(`[getBusinessOwnerData] Cache MISS or stale entry for connection ${connectionId}. Fetching from Telegram API...`);
		try {
			const result = await api.getBusinessConnection(connectionId);
			const id = result.user?.id || result.user_chat_id;
			const name = result.user?.first_name || 'the business owner';
			const username = result.user?.username;
			if (id) {
				ownerData = { id, name, username };
				console.log(`[getBusinessOwnerData] Successfully resolved owner: id=${id}, name=${name}, username=${username ?? ''}. Caching in KV...`);
				await env.CONVERSATION_HISTORY.put(
					`active_connection:${id}`,
					connectionId
				);
				await env.CONVERSATION_HISTORY.put(
					`business_connection:${connectionId}`,
					JSON.stringify(ownerData)
				);
			} else {
				console.error(`[getBusinessOwnerData] Failed to resolve owner ID from result:`, JSON.stringify(result));
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

	if (ctx.has('business_message')) {
		const bizMsg = ctx.businessMessage;
		const connectionId = bizMsg.business_connection_id;
		const customerId = bizMsg.chat.id;
		if (connectionId && customerId) {
			userId = `business:${connectionId}:${customerId}`;
			const ownerData = await getBusinessOwnerData(ctx.api, ctx.env, connectionId);
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
	task.senderId = ctx.from?.id || ctx.update.guest_message?.from?.id;
	task.chatId = ctx.chatId?.toString() || ctx.update.guest_message?.chat?.id?.toString();
	task.updateId = ctx.update.update_id;
	task.messageId = ctx.msg?.message_id || ctx.update.guest_message?.message_id;
	task.updateType = Object.keys(ctx.update).find((k) => k !== 'update_id');
	task.guestQueryId = ctx.update.guest_message?.guest_query_id;
	task.businessConnectionId = ctx.businessMessage?.business_connection_id?.toString();
	task.threadId = ctx.msg?.message_thread_id || ctx.update.guest_message?.message_thread_id;
	
	const balanceKey = `balance:${String(billingUserId)}`;
	const balance = await getBalance(billingUserId || 0, ctx.env.CONVERSATION_HISTORY);

	const modelPreference =
		(await ctx.env.CONVERSATION_HISTORY.get<string>(`model:${String(billingUserId)}`)) ?? 'glm-4.7-flash';
	const modelConfig = AVAILABLE_MODELS[modelPreference] ?? AVAILABLE_MODELS['glm-4.7-flash'];

	if (task.type === 'tool_call' && !modelConfig.supportsTools) {
		task.modelId = AVAILABLE_MODELS['glm-4.7-flash'].id;
	} else if ((task.type === 'photo' || task.geminiParts?.some((p) => p.inlineData)) && !modelConfig.supportsVision) {
		task.modelId = AVAILABLE_MODELS['google/gemini-3.1-flash-lite'].id;
	} else {
		task.modelId = modelConfig.id;
	}

	const amount = amountOverride ?? modelConfig.cost;

	if (balance >= amount) {
		try {
			await ctx.replyWithChatAction('typing', {
				business_connection_id: ctx.businessMessage?.business_connection_id,
			});
		} catch (e) {
			console.log('[chargeStars] Failed to send chat action (likely not a member):', e);
		}
		await ctx.env.CONVERSATION_HISTORY.put(balanceKey, JSON.stringify(balance - amount));
		task.telegramToken = ctx.env.SECRET_TELEGRAM_API_TOKEN;

		if (ctx.has('business_message')) {
			if (!task.systemPrompt) {
				let prompt = SYSTEM_PROMPTS.BUSINESS_MODE;
				const connectionId = ctx.businessMessage.business_connection_id;
				if (connectionId) {
					const ownerData = await getBusinessOwnerData(ctx.api, ctx.env, connectionId);
					if (ownerData) {
						prompt = prompt.replace(/{owner_name}/g, ownerData.name);
						const facts = await ctx.env.CONVERSATION_HISTORY.get(`business_facts:${String(ownerData.id)}`);
						if (facts) {
							prompt += `\n\nHere are some facts about you:\n${facts}`;
						}
					}
				}
				task.systemPrompt = prompt;
			}
		}
 else {
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

		await ctx.env.STREAM_WORKFLOW.create({
			params: task,
		});
	} else {
		if (ctx.has('business_message') || ctx.has('guest_message')) {
			await ctx.reply(
				'Insufficient balance. Please go to direct messages and use /load to top up your Stars.',
				{
					business_connection_id: ctx.businessMessage?.business_connection_id,
					reply_parameters: { message_id: ctx.msgId },
				}
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

export function createChatConversation(env: Environment, executionCtx: ExecutionContext) {
	return async function chatConversation(conversation: MyConversation, ctx: MyContext) {
		ctx.env = env;
		ctx.executionCtx = executionCtx;
		let userId: number | string = ctx.from!.id;
		const threadId = ctx.msg?.message_thread_id;

		if (ctx.has('business_message')) {
			const bizMsg = ctx.businessMessage;
			const connectionId = bizMsg.business_connection_id;
			const customerId = bizMsg.chat.id;
			if (connectionId && customerId) {
				userId = `business:${connectionId}:${customerId}`;
			}
		}

		// Initialize history from KV if it exists, otherwise start fresh
		const history = (await conversation.external(async () => {
			const historyManager = new HistoryManager(env.CONVERSATION_HISTORY);
			return await historyManager.getHistory(userId, threadId);
		})) || [];

		while (true) {
			let prompt = ctx.msg?.text || ctx.msg?.caption || '';

			const geminiParts: GeminiPart[] = [];
			const photo = ctx.msg?.photo;
			if (photo) {
				const largestPhoto = photo[photo.length - 1];
				const file = await conversation.external(() => ctx.api.getFile(largestPhoto.file_id));
				const fileUrl = `https://api.telegram.org/file/bot${env.SECRET_TELEGRAM_API_TOKEN}/${file.file_path}`;
				const base64Data = await conversation.external(async () => {
					const fileRes = await fetch(fileUrl);
					if (fileRes.ok) {
						const arrayBuffer = await fileRes.arrayBuffer();
						return arrayBufferToBase64(arrayBuffer);
					}
					return null;
				});
				if (base64Data) {
					geminiParts.push({
						inlineData: {
							mimeType: 'image/jpeg',
							data: base64Data
						}
					});
				}
				if (!prompt) {
					prompt = 'Please describe this image';
				}
			}

			let billingUserId = ctx.from?.id;
			let ownerData: { id: number; name: string; username?: string } | null = null;
			if (ctx.has('business_message')) {
				const connectionId = ctx.businessMessage.business_connection_id;
				if (connectionId) {
					ownerData = await conversation.external(() => getBusinessOwnerData(ctx.api, env, connectionId));
					if (ownerData?.id) {
						billingUserId = ownerData.id;
					}
				}
			}

			const { balance, modelPreference } = await conversation.external(async () => {
				const b = await getBalance(billingUserId || 0, env.CONVERSATION_HISTORY);
				const mp = (await env.CONVERSATION_HISTORY.get<string>(`model:${String(billingUserId)}`)) ?? 'glm-4.7-flash';
				return { balance: b, modelPreference: mp };
			});

			const modelConfig = AVAILABLE_MODELS[modelPreference] ?? AVAILABLE_MODELS['glm-4.7-flash'];

			if (ctx.msg?.document) {
				const doc = ctx.msg.document;
				prompt = `[Uploaded Document: Name="${doc.file_name || 'document'}", MIME="${doc.mime_type || ''}", FileID="${doc.file_id}"]\n\n${prompt || 'Please process this document.'}`;
			}

			const replyToMessage = ctx.msg?.reply_to_message;
			if (replyToMessage) {
				if (replyToMessage.document) {
					const replyDoc = replyToMessage.document;
					const replyText = replyToMessage.caption || '';
					prompt = `Context of the uploaded document I am replying to: Name="${replyDoc.file_name || 'document'}", MIME="${replyDoc.mime_type || ''}", FileID="${replyDoc.file_id}"${replyText ? ` with caption "${replyText}"` : ''}\n\nMy message: ${prompt}`;
				} else {
					const replyText = replyToMessage.text || replyToMessage.caption || '';
					if (replyText) {
						prompt = `Context of the message I am replying to: "${replyText}"\n\nMy message: ${prompt}`;
					}
				}
			}

			if (prompt) {
				if (geminiParts.some((p) => p.inlineData) && !modelConfig.supportsVision) {
					await ctx.reply(
						`⚠️ Your current model (*${modelPreference}*) does not support vision/images.\n\n` +
							`Please switch to a vision-enabled model using:\n` +
							`- \`/model kimi-k2.6\` (40 Stars)\n` +
							`- \`/model gemma4\` (10 Stars)\n` +
							`- \`/model google/gemini-3.1-flash-lite\` (10 Stars)\n` +
							`- \`/model llama-3.2-vision\` (10 Stars)\n` +
							`- \`/model google/gemini-3.1-pro\` (80 Stars)`,
						{ parse_mode: 'MarkdownV2' }
					);
					ctx = await conversation.wait();
					continue;
				}

				const amount = modelConfig.cost;

				if (balance >= amount) {
					try {
						await ctx.replyWithChatAction('typing', {
							business_connection_id: ctx.businessMessage?.business_connection_id,
						});
					} catch (e) {
						console.error('[chatConversation] Failed to send chat action:', e);
					}
					const balanceKey = `balance:${String(billingUserId)}`;
					await conversation.external(() => env.CONVERSATION_HISTORY.put(balanceKey, JSON.stringify(balance - amount)));

					const systemPrompt = await conversation.external(async () => {
						if (ctx.has('business_message')) {
							let prompt = SYSTEM_PROMPTS.BUSINESS_MODE;
							if (ownerData) {
								prompt = prompt.replace(/{owner_name}/g, ownerData.name);
								const facts = await env.CONVERSATION_HISTORY.get(`business_facts:${String(ownerData.id)}`);
								if (facts) {
									prompt += `\n\nHere are some facts about you:\n${facts}`;
								}
							}
							return prompt;
						}
						const isFileTask = !!ctx.msg?.document || !!(replyToMessage && replyToMessage.document);
						if (isFileTask) {
							return 'You are a helpful assistant running on Telegram. Ensure your responses are formatted using supported Telegram MarkdownV2.';
						}
						const customPrompt = await env.CONVERSATION_HISTORY.get(`prompt:${String(userId)}`);
						return customPrompt || SYSTEM_PROMPTS.TUX_ROBOT;
					});

					const userMessage: ChatMessage = { role: 'user', content: prompt };
					if (geminiParts.length > 0) {
						userMessage.geminiParts = [
							{ text: prompt || 'Please describe this image' },
							...geminiParts
						];
					}

					const modelId = modelConfig.id;

					await conversation.external(async () => {
						await ctx.env.STREAM_WORKFLOW.create({
							params: {
								type: ctx.has('business_message') ? 'business_message' : 'message',
								updateType: ctx.has('business_message') ? 'business_message' : 'message',
								prompt,
								chatId: ctx.chatId?.toString(),
								threadId,
								businessConnectionId: ctx.businessMessage?.business_connection_id?.toString(),
								messageId: ctx.msgId,
								userId: String(userId),
								systemPrompt,
								history,
								telegramToken: env.SECRET_TELEGRAM_API_TOKEN,
								modelId,
							},
						});
					});
				} else {
					// Handle insufficient balance (omitted full logic for brevity, can call ctx.replyWithInvoice)
					await ctx.reply('Insufficient balance. Please top up your Stars.');
					break;
				}
			}

			ctx = (await conversation.wait()) as MyContext;
			ctx.env = env;
			ctx.executionCtx = executionCtx;
		}
	};
}

function setupBot(bot: Bot<MyContext>, env: Environment, executionCtx: ExecutionContext) {
	bot.use(async (ctx, next) => {
		const updateType = Object.keys(ctx.update).find(k => k !== 'update_id');
		console.log(`[Grammy-Update] Received update: ${ctx.update.update_id}, Type: ${updateType}`);
		ctx.env = env;
		ctx.executionCtx = executionCtx;
		try {
			await next();
			console.log(`[Grammy-Update] Finished handling update ${ctx.update.update_id}`);
		} catch (e) {
			console.error(`[Grammy-Update] Error handling update ${ctx.update.update_id}:`, e);
			if (e instanceof GrammyError) {
				console.error(`[Grammy-Error-Detail] Method: ${e.method}, Error Code: ${e.error_code}, Description: ${e.description}`);
			} else if (e instanceof HttpError) {
				console.error(`[Grammy-Error-Detail] HTTP network connection error contacting Telegram API.`, e);
			} else if (e instanceof Error) {
				console.error(`[Grammy-Error-Detail] Stack trace:\n${e.stack}`);
			}
			throw e;
		}
	});

	bot.api.config.use(autoRetry());
	bot.api.config.use(async (prev, method, payload, signal) => {
		console.log(`[Grammy-API] Request: ${method}, Payload:`, JSON.stringify(payload));
		try {
			const res = await prev(method, payload, signal);
			console.log(`[Grammy-API] Success: ${method}`);
			return res;
		} catch (e) {
			console.error(`[Grammy-API] Error in ${method}:`, e);
			throw e;
		}
	});

	bot.use(stream());

	bot.use(
		session({
			initial: () => ({}),
			storage: new KvAdapter(env.CONVERSATION_HISTORY as unknown as CfKVNamespace),
		}),
	);
	bot.use(conversations());
	bot.use(createConversation(createChatConversation(env, executionCtx), 'chatConversation'));

	const commands = new CommandGroup<MyContext>();

	commands.command('start', 'Welcome message and command list', async (ctx) => {
		await ctx.reply(
			'Welcome! Here are my commands:\n' +
				'/balance - Check your current Star balance\n' +
				'/load <amount> - Top up your balance with Telegram Stars\n' +
				'/photo <prompt> - Generate an image (100 Stars)\n' +
				'/model <name> - Switch AI model and see costs\n' +
				'/prompt <"prompt"> - Set your custom system prompt (use "" or reset to clear)\n' +
				'/facts <"facts"> - Set facts about yourself for business mode (use "" or reset to clear)\n' +
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

	commands.command('balance', 'Check your current Star balance', async (ctx) => {
		const balance = await getBalance(ctx.from?.id || 0, ctx.env.CONVERSATION_HISTORY);
		await ctx.reply(`Your current balance is ${String(balance)} Stars.`);
	});

	commands.command('load', 'Top up your balance with Telegram Stars', async (ctx) => {
		const amount = parseInt(ctx.match || '0');
		if (isNaN(amount) || amount <= 0 || amount > 1000) {
			await ctx.reply('Please specify an amount between 1 and 1000 Stars. Example: /load 100');
		} else {
			await ctx.replyWithInvoice('Stars Top-up', `Purchase ${String(amount)} Stars`, `load:${String(amount)}`, 'XTR', [
				{ label: 'Stars', amount },
			]);
		}
	});

	commands.command('photo', 'Generate an image', async (ctx) => {
		const prompt = ctx.match;
		if (prompt) {
			await chargeStars(ctx, { type: 'gen_photo', prompt }, 100);
		} else {
			await ctx.reply('Please provide a prompt for the photo. Example: /photo a futuristic city');
		}
	});

	commands.command('clear', 'Clear your conversation history', async (ctx) => {
		const historyManager = new HistoryManager(ctx.env.CONVERSATION_HISTORY);
		let historyUserId: number | string = ctx.from?.id || 0;
		if (ctx.update.business_message) {
			const bizMsg = ctx.update.business_message;
			const connectionId = bizMsg.business_connection_id;
			const customerId = bizMsg.chat.id;
			if (connectionId && customerId) {
				historyUserId = `business:${connectionId}:${customerId}`;
			}
		}
		const threadId = ctx.msg?.message_thread_id || ctx.update.guest_message?.message_thread_id;
		await historyManager.clearHistory(historyUserId, threadId);
		await ctx.reply('History cleared');
	});

	commands.command('model', 'Switch AI model and see costs', async (ctx) => {
		const modelKey = `model:${String(ctx.from?.id)}`;
		const selectedModel = ctx.match?.toLowerCase();
		if (selectedModel) {
			if (selectedModel in AVAILABLE_MODELS) {
				await ctx.env.CONVERSATION_HISTORY.put(modelKey, selectedModel);
				await ctx.reply(`Model updated to *${selectedModel}*.`, { parse_mode: 'MarkdownV2' });
			} else {
				await ctx.reply(`Invalid model. Available models:\n${Object.keys(AVAILABLE_MODELS).join('\n')}`);
			}
		} else {
			const currentModel = (await ctx.env.CONVERSATION_HISTORY.get<string>(modelKey)) ?? 'glm-4.7-flash';
			await ctx.reply(
				`Current model: *${currentModel}*\n\n` +
					`Available models:\n` +
					Object.entries(AVAILABLE_MODELS)
						.map(([name, cfg]) => `- \`${name}\` (${String(cfg.cost)} Stars)`)
						.join('\n'),
				{ parse_mode: 'MarkdownV2' },
			);
		}
	});

	commands.command('prompt', 'Set your custom system prompt', async (ctx) => {
		let promptValue = ctx.match.trim();
		const userId = String(ctx.from?.id);
		if (promptValue === 'reset' || promptValue === '""' || promptValue === "''" || promptValue === '') {
			await ctx.env.CONVERSATION_HISTORY.delete(`prompt:${userId}`);
			await ctx.reply(`System prompt reset to default:\n\n${SYSTEM_PROMPTS.TUX_ROBOT}`);
		} else {
			if (
				(promptValue.startsWith('"') && promptValue.endsWith('"')) ||
				(promptValue.startsWith("'") && promptValue.endsWith("'"))
			) {
				promptValue = promptValue.substring(1, promptValue.length - 1);
			}
			await ctx.env.CONVERSATION_HISTORY.put(`prompt:${userId}`, promptValue);
			await ctx.reply(`System prompt updated to:\n\n${promptValue}`);
		}
	});

	commands.command('facts', 'Set facts about yourself for business mode', async (ctx) => {
		let factsValue = ctx.match.trim();
		const userId = ctx.from?.id;
		if (!userId) return;

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
	});

	bot.use(commands);

	bot.on('message:document', async (ctx) => {
		await ctx.conversation.enter('chatConversation');
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
		await ctx.env.STREAM_WORKFLOW.create({
			params: task,
		});
		await ctx.env.CONVERSATION_HISTORY.delete(`task:${taskId}`);
	});

	bot.on('message:photo', async (ctx) => {
		await ctx.conversation.enter('chatConversation');
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
						parse_mode: 'MarkdownV2',
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
			const rawResponse = await ctx.env.AI.run('@cf/meta/llama-3.2-11b-vision-instruct', {
				messages,
				max_completion_tokens: 100,
			});
			const aiResponse = rawResponse as { response?: string };
			if (aiResponse.response) {
				await ctx.answerInlineQuery([
					{
						type: 'article',
						id: 'ai_response',
						title: 'AI Response',
						input_message_content: {
							message_text: await markdownToMarkdownV2(aiResponse.response),
							parse_mode: 'MarkdownV2',
						},
					},
				]);
			}
		} catch {
			/* ignore */
		}
	});

	bot.on('business_connection', async (ctx) => {
		const connection = ctx.businessConnection;
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
	});

	bot.on('business_message', async (ctx) => {
		await ctx.conversation.enter('chatConversation');
	});

	bot.on('message:text', async (ctx) => {
		await ctx.conversation.enter('chatConversation');
	});

	bot.on('guest_message', async (ctx) => {
		const guestMessage = ctx.update.guest_message!;
		let prompt = guestMessage.text?.toString() ?? '';
		const token = ctx.env.SECRET_TELEGRAM_API_TOKEN;
		let botUsername = await ctx.env.CONVERSATION_HISTORY.get(`bot_username:${token.slice(0, 10)}`);
		if (!botUsername) {
			botUsername = ctx.me.username;
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

		if (prompt.includes('/start') && guestMessage.guest_query_id) {
			try {
				await ctx.api.answerGuestQuery(guestMessage.guest_query_id, {
					type: 'article',
					id: crypto.randomUUID(),
					title: 'Welcome',
					input_message_content: {
						message_text:
							'Welcome! Here are my commands:\n' +
							'/balance - Check your current Star balance\n' +
							'/load <amount> - Top up your balance with Telegram Stars\n' +
							'/photo <prompt> - Generate an image (100 Stars)\n' +
							'/model <name> - Switch AI model and see costs\n' +
							'/prompt <"prompt"> - Set your custom system prompt (use "" or reset to clear)\n' +
							'/facts <"facts"> - Set facts about yourself for business mode (use "" or reset to clear)\n' +
							'<prompt> - Generate text (may use tools if supported by model)\n' +
							'Send a voice note - Transform your bot into a voice assistant (+20 Stars)\n' +
							'/clear - Clear your conversation history\n\n' +
							'New users start with 200 free credits!\n\n' +
							'Click the button below to open the Web App!',
					},
					reply_markup: {
						inline_keyboard: [
							[{ text: 'Open Web App', url: 'https://tux-robot.codebam.ca' }],
						],
					},
				});
			} catch (e) {
				console.error('[guest_message] Failed to answer guest query:', e);
			}
			return;
		}

		await chargeStars(ctx, { type: 'message', prompt });
	});

	bot.catch((err) => {
		const ctx = err.ctx;
		const e = err.error;
		console.error(`[Grammy-Error] Error while handling update ${ctx?.update?.update_id}:`);
		if (e instanceof GrammyError) {
			console.error(`[Grammy-Error-Detail] Method: ${e.method}, Error Code: ${e.error_code}, Description: ${e.description}`);
		} else if (e instanceof HttpError) {
			console.error(`[Grammy-Error-Detail] HTTP network connection error contacting Telegram API.`, e);
		} else if (e instanceof Error) {
			console.error(`[Grammy-Error-Detail] Unknown error:`, e.message);
			console.error(`[Grammy-Error-Detail] Stack trace:\n${e.stack}`);
		} else {
			console.error(`[Grammy-Error-Detail] Unknown error object:`, e);
		}
	});
}

async function processTask(task: Task, env: Environment): Promise<void> {
	try {
		const token = env.SECRET_TELEGRAM_API_TOKEN;
		if (!token) {
			throw new Error('SECRET_TELEGRAM_API_TOKEN is empty in processTask');
		}
		const botInstance = new Bot<MyContext>(token);
		botInstance.api.config.use(autoRetry());

		if (task.type === 'voice' && task.fileId) {
			try {
				const file = await botInstance.api.getFile(task.fileId);
				const fileUrl = `https://api.telegram.org/file/bot${token}/${file.file_path}`;
				const fileRes = await fetch(fileUrl);
				if (fileRes.ok) {
					const audioData = await fileRes.arrayBuffer();
					const transcription = (await env.AI.run('@cf/openai/whisper', {
						audio: [...new Uint8Array(audioData)],
					})) as { text: string };
					if (transcription.text) {
						task.prompt = transcription.text;
						task.type = 'message';
					}
				}
			} catch (e) {
				console.error('[processTask] Failed to transcribe voice:', e);
				await botInstance.api.sendMessage(task.chatId!, 'Failed to transcribe voice message.', {
					business_connection_id: task.businessConnectionId,
					reply_parameters: { message_id: task.messageId! },
				});
				return;
			}
		}

		if (task.type === 'gen_photo') {
			try {
				const response = (await env.AI.run('@cf/black-forest-labs/flux-1-schnell', {
					prompt: task.prompt,
				})) as { image: string };

				if (response.image) {
					const binaryString = atob(response.image);
					const bytes = new Uint8Array(binaryString.length);
					for (let i = 0; i < binaryString.length; i++) {
						bytes[i] = binaryString.charCodeAt(i);
					}
					await botInstance.api.sendPhoto(task.chatId!, new InputFile(bytes, 'photo.png'), {
						business_connection_id: task.businessConnectionId,
						reply_parameters: { message_id: task.messageId! },
					});
					return;
				}
			} catch (e) {
				console.error('[processTask] Failed to generate photo:', e);
				await botInstance.api.sendMessage(task.chatId!, 'Failed to generate photo.', {
					business_connection_id: task.businessConnectionId,
					reply_parameters: { message_id: task.messageId! },
				});
				return;
			}
		}

		const userMessage: ChatMessage = { role: 'user', content: task.prompt };
		if (task.type === 'photo' && task.fileId) {
			try {
				const file = await botInstance.api.getFile(task.fileId);
				const fileUrl = `https://api.telegram.org/file/bot${token}/${file.file_path}`;
				const fileRes = await fetch(fileUrl);
				if (fileRes.ok) {
					const arrayBuffer = await fileRes.arrayBuffer();
					const base64Data = arrayBufferToBase64(arrayBuffer);
					userMessage.geminiParts = [
						{ text: task.prompt || 'Please describe this image' },
						{
							inlineData: {
								mimeType: 'image/jpeg',
								data: base64Data
							}
						}
					];
				}
			} catch (e) {
				console.error('[processTask] Failed to download image for vision task:', e);
			}
		}

		const messages = [
			{ role: 'system', content: task.systemPrompt || 'You are a helpful assistant.' },
			...(task.history || []),
			userMessage,
		];
		const modelId = task.modelId || '@cf/meta/llama-3.1-8b-instruct-fp8';

		const responseContent = await streamAiResponseToTelegram(
			{
				env,
				api: botInstance.api,
			},
			env.AI,
			modelId,
			messages,
			task,
			[
				fetchTool,
				wikipediaTool,
				createTavilySearchTool(env.TAVILY_API_KEY || ''),
				createSandboxTool(env.Sandbox, String(task.userId)),
				createTelegramFileReaderTool(env, env.Sandbox, String(task.userId), messages, modelId),
				createTelegramFileSearchTool(env, modelId),
			],
		);

		if (task.userId && responseContent) {
			const historyManager = new HistoryManager(env.CONVERSATION_HISTORY);
			await historyManager.addMessage(task.userId, task.prompt, responseContent, task.threadId);
		}
	} catch (e) {
		console.error('[processTask] Error processing task:', e);
	}
}

export class BotWorkflow extends WorkflowEntrypoint<Environment, Task> {
	async run(event: WorkflowEvent<Task>, step: WorkflowStep) {
		const task = event.payload;

		if (
			task.type === 'message' ||
			task.type === 'business_message' ||
			task.type === 'tool_call' ||
			task.type === 'photo' ||
			task.type === 'voice' ||
			task.type === 'gen_photo'
		) {
			await step.do('process task', async () => {
				await processTask(task, this.env);
			});
		}
	}
}

export default {
	BotWorkflow,
	async queue(_batch: MessageBatch<any>, _env: Environment): Promise<void> {
		console.log('[Queue] Dummy handler (Queues are being removed)');
	},
	async fetch(request: Request, env: Environment, executionCtx: ExecutionContext): Promise<Response> {
		const url = new URL(request.url);
		const xSource = request.headers.get('x-source');
		const xPassword = request.headers.get('x-password');
		console.log(`[Fetch] Incoming request: ${request.method} ${url.href} (hostname: ${url.hostname}, source: ${xSource})`);

		if (url.hostname === 'workflow.local' || url.pathname === '/workflow' || xSource === 'webapp') {
			if (xPassword !== env.SECRET_TELEGRAM_API_TOKEN) {
				return new Response('Unauthorized', { status: 401 });
			}
			console.log('[Fetch] Matches task endpoint, processing task...');
			const task = (await request.json()) as Task;
			console.log(`[Fetch] Task type: ${task.type}, prompt: ${task.prompt}, stream: ${task.stream}`);

			const userMessage: ChatMessage = { role: 'user', content: task.prompt };
			if (task.type === 'photo' && task.fileId) {
				try {
					const api = new Bot<MyContext>(env.SECRET_TELEGRAM_API_TOKEN).api;
					const file = await api.getFile(task.fileId);
					if (file.file_path) {
						const fileUrl = `https://api.telegram.org/file/bot${env.SECRET_TELEGRAM_API_TOKEN}/${file.file_path}`;
						const fileRes = await fetch(fileUrl);
						if (fileRes.ok) {
							const arrayBuffer = await fileRes.arrayBuffer();
							const base64Data = arrayBufferToBase64(arrayBuffer);
							userMessage.geminiParts = [
								{ text: task.prompt || 'Please describe this image' },
								{
									inlineData: {
										mimeType: 'image/jpeg',
										data: base64Data
									}
								}
							];
						}
					}
				} catch (e) {
					console.error('[Fetch] Failed to download image for vision task:', e);
				}
			}

			const messages = [
				{ role: 'system', content: task.systemPrompt || 'You are a helpful assistant.' },
				...(task.history || []),
				userMessage,
			];

			const tools = [
				fetchTool,
				wikipediaTool,
				createTavilySearchTool(env.TAVILY_API_KEY || ''),
				createSandboxTool(env.Sandbox, String(task.userId)),
				createTelegramFileReaderTool(env, env.Sandbox, String(task.userId), messages, task.modelId || '@cf/meta/llama-3.1-8b-instruct-fp8'),
				createTelegramFileSearchTool(env, task.modelId || '@cf/meta/llama-3.1-8b-instruct-fp8'),
			];

			const aiResponse = await customRunWithTools(
				env.AI,
				task.modelId || '@cf/meta/llama-3.1-8b-instruct-fp8',
				{ messages, tools: task.type === 'tool_call' ? tools : [] },
				{ streamFinalResponse: task.stream || false },
			);

			console.log(`[Fetch] aiResponse type: ${typeof aiResponse}, constructor: ${aiResponse && typeof aiResponse === 'object' ? aiResponse.constructor?.name : 'unknown'}`);

			let stream: ReadableStream | null = null;
			if (aiResponse instanceof ReadableStream) {
				stream = aiResponse;
			} else if (aiResponse && typeof aiResponse === 'object' && 'body' in aiResponse && aiResponse.body instanceof ReadableStream) {
				stream = aiResponse.body;
			} else if (aiResponse && typeof aiResponse === 'object' && 'getReader' in aiResponse && typeof aiResponse.getReader === 'function') {
				stream = aiResponse as unknown as ReadableStream;
			}

			if (task.stream && stream) {
				console.log(`[Fetch] Returning streaming response. Locked: ${stream.locked}`);
				return new Response(stream, {
					headers: { 'Content-Type': 'text/event-stream' },
				});
			}

			console.log('[Fetch] Returning JSON response');
			return new Response(JSON.stringify(aiResponse), {
				headers: { 'Content-Type': 'application/json' },
			});
		}

		const token = env.SECRET_TELEGRAM_API_TOKEN;
		if (!token) {
			console.error('[Fetch] SECRET_TELEGRAM_API_TOKEN is empty or undefined!');
			return new Response('Error: SECRET_TELEGRAM_API_TOKEN is missing', { status: 500 });
		}
		const bot = new Bot<MyContext>(token);
		setupBot(bot, env, executionCtx);

		if (request.method === 'POST') {
			const clone = request.clone();
			try {
				const body = await clone.json();
				console.log('[Fetch] Incoming Update:', JSON.stringify(body));
			} catch (e) {
				console.error('[Fetch] Error parsing update body:', e);
			}
		}

		if (request.method === 'GET') {
			const url = new URL(request.url);
			if (url.searchParams.get('command') === 'set') {
				const token = env.SECRET_TELEGRAM_API_TOKEN;
				const webhookUrl = `${url.origin}${url.pathname}`;
				const api = new Bot<MyContext>(token).api;

				try {
					const result = await api.setWebhook(webhookUrl, {
						max_connections: 40,
						allowed_updates: [
							'message',
							'edited_message',
							'callback_query',
							'inline_query',
							'guest_message',
							'business_message',
							'business_connection',
							'pre_checkout_query',
						],
						drop_pending_updates: true,
					});
					return new Response(JSON.stringify({ ok: result }), {
						headers: { 'Content-Type': 'application/json' },
					});
				} catch (e: any) {
					return new Response(JSON.stringify({ ok: false, error: e.message }), {
						headers: { 'Content-Type': 'application/json' },
						status: 500,
					});
				}
			}
		}
		try {
			return await webhookCallback(bot, 'cloudflare-mod', {
				onTimeout: 'return',
			})(request);
		} catch (e: any) {
			console.error('[Fetch-Webhook-Error] Error during webhook update handling:', e);
			if (e instanceof GrammyError) {
				console.error(`[Fetch-Webhook-Error] GrammyError: ${e.method}, Error Code: ${e.error_code}, Description: ${e.description}`);
			} else if (e instanceof HttpError) {
				console.error(`[Fetch-Webhook-Error] HttpError: Could not contact Telegram API.`, e);
			} else if (e instanceof Error) {
				console.error(`[Fetch-Webhook-Error] Stack trace:\n${e.stack}`);
			}
			return new Response('Internal Webhook Error', { status: 500 });
		}
		},
		};
