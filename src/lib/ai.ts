import type { Api } from 'grammy';
import type { StreamContextExtension } from '@grammyjs/stream';
import {
	markdownToMarkdownV2,
	AVAILABLE_MODELS,
	extractText,
	extractThinking,
	extractReasoning,
	stripThinking,
	sanitizeMarkdownV2,
	THINK_TAGS,
	type AiResponse,
	type ChatMessage,
	type GeminiPart,
	type NormalizedToolCall,
	type RawToolCall,
	type Task,
	type Tool,
} from '@codebam/shared';

/**
 * Stream-aware filter: feed chunks of generated text in order and receive
 * the same text with think-tag content removed, tolerating tags split across
 * chunks.
 */
export function createThinkFilter() {
	let inside = false;
	let buffer = '';
	let emittedAny = false;
	const flush = (chunk: string) => {
		const out = emittedAny ? chunk : chunk.replace(/^\s+/, '');
		if (out) emittedAny = true;
		return out;
	};
	return {
		push(chunk: string): string {
			let result = '';
			let current = buffer + chunk;
			buffer = '';

			while (current.length > 0) {
				if (inside) {
					const closeMatch = current.match(/<\/([a-z]+)>/i);
					if (closeMatch && THINK_TAGS.includes(closeMatch[1].toLowerCase())) {
						const closeIdx = closeMatch.index!;
						current = current.slice(closeIdx + closeMatch[0].length);
						inside = false;
					} else {
						// Look for partial close tag
						const lastLt = current.lastIndexOf('<');
						if (lastLt !== -1 && lastLt > current.length - 10) {
							result += ''; // Still inside
							buffer = current.slice(lastLt);
							return result;
						}
						return '';
					}
				} else {
					const openMatch = current.match(/<([a-z]+)(?:\s[^>]*)?>/i);
					if (openMatch && THINK_TAGS.includes(openMatch[1].toLowerCase())) {
						const openIdx = openMatch.index!;
						result += flush(current.slice(0, openIdx));
						current = current.slice(openIdx + openMatch[0].length);
						inside = true;
					} else {
						const lastLt = current.lastIndexOf('<');
						if (lastLt !== -1 && lastLt > current.length - 20) {
							const partial = current.slice(lastLt);
							const isPotential = THINK_TAGS.some(tag => {
								const openTag = `<${tag}`;
								const closeTag = `</${tag}`;
								return openTag.startsWith(partial) || closeTag.startsWith(partial);
							});
							if (isPotential) {
								result += flush(current.slice(0, lastLt));
								buffer = partial;
								return result;
							}
						}
						result += flush(current);
						current = '';
					}
				}
			}
			return result;
		},
		end(): string {
			if (inside) {
				inside = false;
				buffer = '';
				return '';
			}
			const tail = buffer;
			buffer = '';
			return flush(tail);
		}
	};
}

interface AiRunner {
	run(model: string, inputs: Record<string, unknown>): Promise<AiResponse | ReadableStream | Response>;
}

function createMockStream(text: string): ReadableStream {
	const encoder = new TextEncoder();
	return new ReadableStream({
		start(controller) {
			const payload = `data: ${JSON.stringify({ response: text })}\n\ndata: [DONE]\n\n`;
			controller.enqueue(encoder.encode(payload));
			controller.close();
		}
	});
}

/**
 * Custom runner that supports tool calls across different AI models.
 */
export async function customRunWithTools(
	ai: AiRunner,
	model: string,
	input: { messages: ChatMessage[]; tools?: Tool[] },
	config: { streamFinalResponse: boolean }
): Promise<AiResponse | ReadableStream> {
	console.log(`[customRunWithTools] Model: ${model}, Tools: ${input.tools?.length || 0}, Stream: ${config.streamFinalResponse}`);
	
	const modelConfig = Object.values(AVAILABLE_MODELS).find((cfg) => cfg.id === model);
	const supportsVision = modelConfig?.supportsVision || false;
	const isGemini = model.includes('google/gemini');

	const messages: ChatMessage[] = input.messages.map((m) => {
		if (m.geminiParts) {
			if (!supportsVision) {
				const textParts = m.geminiParts.filter(p => !p.inlineData);
				const firstText = textParts.find(p => p.text)?.text || m.content;
				return {
					...m,
					content: firstText,
					geminiParts: undefined
				};
			} else if (!isGemini) {
				const hasImage = m.geminiParts.some(p => p.inlineData);
				if (hasImage) {
					const contentParts: any[] = [];
					const textPart = m.geminiParts.find(p => p.text);
					if (textPart && textPart.text) {
						contentParts.push({ type: 'text', text: textPart.text });
					} else if (m.content) {
						contentParts.push({ type: 'text', text: m.content });
					}

					for (const part of m.geminiParts) {
						if (part.inlineData) {
							contentParts.push({
								type: 'image_url',
								image_url: {
									url: `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`
								}
							});
						}
					}
					return {
						...m,
						content: contentParts as any,
						geminiParts: undefined
					};
				}
			}
		}
		return { ...m };
	});

	const tools = input.tools || [];

	const cfTools = tools.map((t) => ({
		name: t.name,
		description: t.description,
		parameters: t.parameters,
	}));

	const runModel = async (msgs: ChatMessage[], stream: boolean, omitTools = false) => {
		console.log(`[customRunWithTools] runModel starting. Stream: ${stream}, Model: ${model}, OmitTools: ${omitTools}`);
		console.log(`[customRunWithTools] msgs passed to Cloudflare:`, JSON.stringify(msgs));
		try {
			if (isGemini) {
				const systemMessage = msgs.find((m) => m.role === 'system');
				const otherMessages = msgs.filter((m) => m.role !== 'system');
				const geminiInput: Record<string, unknown> = {
					contents: otherMessages.map((m) => {
						// For Gemini, we must preserve the exact parts array from previous turns
						if (m.geminiParts) {
							return {
								role: m.role === 'assistant' ? 'model' : 'user',
								parts: m.geminiParts,
							};
						}

						const role = m.role === 'assistant' ? 'model' : 'user';
						const parts: GeminiPart[] = [];
						if (m.content) parts.push({ text: m.content });
						return { role, parts };
					}),
					tools: (!omitTools && cfTools.length > 0) ? [{ functionDeclarations: cfTools }] : undefined,
					generationConfig: { maxOutputTokens: 65536 },
					stream,
				};
				if (systemMessage?.content) {
					geminiInput.system_instruction = {
						parts: [{ text: systemMessage.content }],
					};
				}
				return await ai.run(model, geminiInput);
			}

			const options: Record<string, unknown> = {
				messages: msgs.map((m) => {
					// Ensure content is not null (use empty string if empty/null)
					const cleanMessage: any = { ...m };
					if (cleanMessage.role === 'assistant' && cleanMessage.tool_calls && !cleanMessage.content) {
						cleanMessage.content = null;
					} else if (cleanMessage.content === null || cleanMessage.content === undefined) {
						cleanMessage.content = '';
					}
					// Remove internal geminiParts before sending to CF
					delete cleanMessage.geminiParts;
					return cleanMessage;
				}),
				tools: (!omitTools && cfTools.length > 0) ? cfTools.map((t) => ({ type: 'function', function: t })) : undefined,
				tool_choice: (!omitTools && cfTools.length > 0) ? 'auto' : undefined,
				max_tokens: 65536,
				max_completion_tokens: 65536,
				stream,
			};

			console.log(`[customRunWithTools] Calling ai.run with options:`, JSON.stringify({
				...options,
				messages: (options.messages as any[]).map(m => ({ role: m.role, contentLength: m.content?.length, keys: Object.keys(m) })),
				tools: options.tools ? (options.tools as any[]).map(t => t.function?.name || t.name) : undefined
			}));

			const result = await ai.run(model, options);
			const isStream = result && (typeof (result as any).getReader === 'function' || (typeof (result as any).body?.getReader === 'function'));
			console.log(`[customRunWithTools] ai.run returned. Type: ${typeof result}, isStream: ${isStream}, Keys: ${result && typeof result === 'object' ? Object.keys(result).join(', ') : 'none'}`);
			
			if (stream && !isStream) {
				if (result && typeof result === 'object' && 'body' in result && result.body && typeof (result.body as any).getReader === 'function') {
					return result.body;
				}
			}
			return result;
		} catch (e) {
			console.error(`[customRunWithTools] ai.run failed for model ${model}:`, e);
			throw e;
		}
	};

	let turn = 0;
	while (turn < 5) {
		const isFinalTurn = turn > 0 || cfTools.length === 0;
		const shouldStream = isFinalTurn ? config.streamFinalResponse : false;
		console.log(`[customRunWithTools] Turn:${turn} ShouldStream:${shouldStream} FinalTurn:${isFinalTurn}`);
		const response = await runModel(messages, shouldStream, turn > 0);

		if (shouldStream || (response && typeof (response as any).getReader === 'function')) {
			return response as ReadableStream;
		}

		const aiRes = response as AiResponse;
		let toolCalls: RawToolCall[] = [];
		let geminiParts: GeminiPart[] = [];

		if (aiRes?.tool_calls) {
			toolCalls = [...aiRes.tool_calls];
		} else if (aiRes?.choices?.[0]?.message?.tool_calls) {
			toolCalls = [...aiRes.choices[0].message.tool_calls];
		} else if (isGemini && aiRes?.candidates?.[0]?.content?.parts) {
			geminiParts = aiRes.candidates[0].content.parts;
			// Extract Gemini function calls
			for (const part of geminiParts) {
				if (part.functionCall) {
					toolCalls.push({
						id: `call_${Math.random().toString(36).substring(2, 9)}`,
						type: 'function',
						function: {
							name: part.functionCall.name,
							arguments: part.functionCall.args,
						},
					});
				}
			}
		}

		const responseText = extractText(aiRes);
		console.log(`[customRunWithTools] Turn:${turn} Tools:${toolCalls.length} TextLen:${responseText.length}`);

		if (toolCalls.length > 0) {
			const normalizedToolCalls: NormalizedToolCall[] = toolCalls.map((call, index) => {
				let name = call.name || call.function?.name || '';
				if (name.startsWith('functions.')) {
					name = name.substring(10);
				}
				let args = call.arguments ?? call.function?.arguments ?? '';
				if (typeof args !== 'string') {
					try {
						args = JSON.stringify(args);
					} catch {
						args = '{}';
					}
				}
				return {
					id: call.id || `call_${Math.random().toString(36).substring(2, 9)}_${index}`,
					type: 'function',
					function: { name, arguments: args },
				};
			});

			const assistantMessage: any = {
				role: 'assistant',
				content: responseText || null,
				tool_calls: normalizedToolCalls,
			};
			if (isGemini) {
				assistantMessage.geminiParts = geminiParts;
			}
			messages.push(assistantMessage);

			for (const call of normalizedToolCalls) {
				const toolName = call.function.name;
				const toolId = call.id;
				const toolArgsString = call.function.arguments;
				const tool = tools.find((t) => t.name === toolName);

				if (tool && tool.function) {
					try {
						let parsedArgs: unknown;
						try {
							parsedArgs = JSON.parse(toolArgsString);
						} catch {
							parsedArgs = toolArgsString;
						}
						const result = await tool.function(parsedArgs);
						const content = typeof result === 'string' ? result : JSON.stringify(result);

						const toolMessage: ChatMessage = { role: 'tool', tool_call_id: toolId, name: toolName, content };
						if (isGemini) {
							toolMessage.geminiParts = [
								{
									functionResponse: {
										name: toolName,
										response: { content },
									},
								},
							];
						}
						messages.push(toolMessage);
					} catch (e) {
						console.error(`[customRunWithTools] Tool execution failed: ${toolName}`, e);
						const content = `Error: ${String(e)}`;
						const toolMessage: ChatMessage = { role: 'tool', tool_call_id: toolId, name: toolName, content };
						if (isGemini) {
							toolMessage.geminiParts = [
								{
									functionResponse: {
										name: toolName,
										response: { content },
									},
								},
							];
						}
						messages.push(toolMessage);
					}
				} else {
					const content = 'Tool not found';
					const toolMessage: ChatMessage = { role: 'tool', tool_call_id: toolId, name: toolName, content };
					if (isGemini) {
						toolMessage.geminiParts = [
							{
								functionResponse: {
									name: toolName,
									response: { content },
								},
							},
						];
					}
					messages.push(toolMessage);
				}
			}
		} else {
			if (config.streamFinalResponse) {
				console.log(`[customRunWithTools] Creating mock stream from successful response. Text length: ${responseText.length}`);
				return createMockStream(responseText);
			}
			return aiRes;
		}
		turn++;
	}

	console.log('[customRunWithTools] Maximum turns reached.');
	const finalResponse = await runModel(messages, config.streamFinalResponse, true);
	if (config.streamFinalResponse && finalResponse instanceof ReadableStream) {
		return finalResponse;
	}
	const finalText = extractText(finalResponse as AiResponse);
	if (config.streamFinalResponse && finalText) {
		return createMockStream(finalText);
	}
	return finalResponse as AiResponse;
}

export async function sendMessageDraft(api: Api, data: Record<string, any>, retries = 3) {
	const textLen = typeof data.text === 'string' ? data.text.length : 0;
	for (let i = 0; i < retries; i++) {
		try {
			await (api.raw as any).sendMessageDraft(data);
			console.log(`[sendMessageDraft] Success. Len: ${textLen}`);
			return;
		} catch (e: any) {
			console.error(`[sendMessageDraft] Attempt ${i + 1} Failed. Len: ${textLen}, Error:`, e);
			if (e.error_code === 429) {
				const retryAfter = (e.parameters?.retry_after || 1) * 1000;
				await new Promise(resolve => setTimeout(resolve, retryAfter));
				continue;
			}
			if (e.error_code >= 500) {
				await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
				continue;
			}
			break;
		}
	}
}

export interface StreamChunk {
	type: 'content' | 'thinking' | 'reasoning';
	text: string;
}

/**
 * Get AI stream for a model.
 */
async function* runStream(ai: AiRunner, model: string, messages: ChatMessage[], tools: Tool[] = [], onStatusUpdate?: (status: 'Thinking' | 'Reasoning') => void): AsyncGenerator<StreamChunk, void, unknown> {
	const startTime = Date.now();
	console.log(`[runStream] Starting stream for model: ${model} at ${new Date(startTime).toISOString()}`);
	yield { type: 'content', text: '' }; // Yield immediately to satisfy TTFT and unblock UI

	const response = await customRunWithTools(ai, model, { messages, tools }, { streamFinalResponse: true });
	const filter = createThinkFilter();

	if (response && (response as any).role === 'tool') {
		console.log('[runStream] customRunWithTools returned a tool response unexpectedly');
		return;
	}

	if (response && typeof (response as any).getReader === 'function') {
		const reader = (response as ReadableStream).getReader();
		const decoder = new TextDecoder();
		let buffer = '';
		let chunkCount = 0;
		let streamFinished = false;

		let pendingContent = '';
		let pendingThinking = '';
		let pendingReasoning = '';
		let lastYieldTime = Date.now();

		const flushPending = async function* (): AsyncGenerator<StreamChunk, void, unknown> {
			if (pendingThinking) {
				yield { type: 'thinking', text: pendingThinking };
				pendingThinking = '';
			}
			if (pendingReasoning) {
				yield { type: 'reasoning', text: pendingReasoning };
				pendingReasoning = '';
			}
			if (pendingContent) {
				yield { type: 'content', text: pendingContent };
				pendingContent = '';
			}
			lastYieldTime = Date.now();
		};

		try {
			while (!streamFinished) {
				let result;
				try {
					result = await reader.read();
				} catch (readErr) {
					console.error(`[runStream] Reader Error at chunk ${chunkCount}:`, readErr);
					break;
				}

				const { done, value } = result;
				if (done) {
					console.log(`[runStream] Stream Done. Chunks:${chunkCount} Duration:${Date.now() - startTime}ms`);
					break;
				}
				chunkCount++;
				try {
					buffer += decoder.decode(value, { stream: true });
				} catch (decodeErr) {
					console.error(`[runStream] Decode Error at chunk ${chunkCount}:`, decodeErr);
					continue;
				}
				
				// Batch lines for processing
				if (!buffer.includes('\n') && buffer.length < 2048) {
					continue;
				}

				const lines = buffer.split('\n');
				buffer = lines.pop() || '';

				for (const line of lines) {
					const trimmed = line.trim();
					if (!trimmed) continue;
					if (trimmed.startsWith('data: ')) {
						const data = trimmed.substring(6);
						if (data === '[DONE]') {
							console.log('[runStream] [DONE] signal received');
							streamFinished = true;
							break;
						}
						try {
							const parsed = JSON.parse(data);
							
							const thinking = extractThinking(parsed);
							if (thinking) {
								onStatusUpdate?.('Thinking');
								pendingThinking += thinking;
							}

							const reasoning = extractReasoning(parsed);
							if (reasoning) {
								onStatusUpdate?.('Reasoning');
								pendingReasoning += reasoning;
							}

							const text = extractText(parsed);
							const choice = parsed.choices?.[0];
							const finishReason = choice?.finish_reason || parsed.candidates?.[0]?.finish_reason || '';
							
							if (chunkCount % 50 === 0 || finishReason) {
								console.log(`[runStream] C:${chunkCount} L:${text.length} FR:${finishReason} hasT:${!!text} hasR:${!!reasoning}`);
							}

							if (text) {
								const filtered = filter.push(text);
								if (filtered) pendingContent += filtered;
							}
						} catch (e) {
							console.error(`[runStream] JSON ERR:${chunkCount} line:"${trimmed.slice(0, 40)}..."`, e);
						}
					} else if (trimmed.startsWith('{')) {
						try {
							const parsed = JSON.parse(trimmed);
							const thinking = extractThinking(parsed);
							if (thinking) {
								onStatusUpdate?.('Thinking');
								pendingThinking += thinking;
							}

							const reasoning = extractReasoning(parsed);
							if (reasoning) {
								onStatusUpdate?.('Reasoning');
								pendingReasoning += reasoning;
							}

							const text = extractText(parsed);
							if (chunkCount % 50 === 0) {
								console.log(`[runStream] NC-C:${chunkCount} L:${text.length} hasT:${!!text}`);
							}
							if (text) {
								const filtered = filter.push(text);
								if (filtered) pendingContent += filtered;
							}
						} catch (e) {
							// Not valid JSON after all, ignore
						}
					}
				}

				// Yield in batches every 500ms or if pending content is large
				if (Date.now() - lastYieldTime > 500 || pendingContent.length > 512 || pendingThinking.length > 512 || pendingReasoning.length > 512) {
					yield* flushPending();
				}
			}
			// Final flush
			yield* flushPending();
		} catch (outerErr) {
			console.error(`[runStream] Fatal Loop Error:`, outerErr);
		} finally {
			try {
				reader.releaseLock();
			} catch {
				// Safety release
			}
		}
		const tail = filter.end();
		if (tail) yield { type: 'content', text: tail };
		console.log(`[runStream] Finished. TotalChunks:${chunkCount} BufferLength:${buffer.length} Duration:${Date.now() - startTime}ms`);
	} else {
		console.log(`[runStream] Response is not a stream. Type: ${typeof response}`);
		const fullText = extractText(response as AiResponse, true);
		const thinking = extractThinking(response as AiResponse);
		const reasoning = extractReasoning(response as AiResponse);
		const content = stripThinking(fullText);
		if (thinking) yield { type: 'thinking', text: thinking };
		if (reasoning) yield { type: 'reasoning', text: reasoning };
		yield { type: 'content', text: content };
	}
}

/**
 * Get AI stream for a model.
 */
export async function* getAiStream(ai: AiRunner, model: string, messages: ChatMessage[], tools: Tool[] = [], onStatusUpdate?: (status: 'Thinking' | 'Reasoning') => void): AsyncGenerator<StreamChunk, void, unknown> {
	console.log(`[getAiStream] Running model: ${model}`);
	yield* runStream(ai, model, messages, tools, onStatusUpdate);
}

export interface StreamCtx {
	env: { SECRET_TELEGRAM_API_TOKEN: string };
	api: Api;
	replyWithStream?: StreamContextExtension['replyWithStream'];
}

async function formatTelegramMessage(content: string, thinking?: string, reasoning?: string, skipMarkdown = false): Promise<string> {
	let message = '';
	if (thinking) {
		const trimmedThinking = thinking.trim().replace(/\n\n$/, '\n');
		const sanitizedThinking = sanitizeMarkdownV2(trimmedThinking);
		// For blockquotes in MarkdownV2, each line must start with '>'. 
		// We ensure there is a space after '>' for better compatibility and to avoid parsing issues.
		message += sanitizedThinking.split('\n').map(line => `> ${line}`).join('\n') + '\n\n';
	}
	if (reasoning) {
		const sanitizedReasoning = sanitizeMarkdownV2(reasoning.trim());
		message += sanitizedReasoning.split('\n').map(line => `> ${line}`).join('\n') + '\n\n';
	}
	if (skipMarkdown) {
		message += sanitizeMarkdownV2(content);
	} else {
		message += await markdownToMarkdownV2(content);
	}
	return message;
}

/**
 * Stream AI response to Telegram, with periodic updates to avoid rate limits.
 */
export async function streamAiResponseToTelegram(
	ctx: StreamCtx,
	ai: AiRunner,
	modelId: string,
	messages: ChatMessage[],
	task: Task,
	tools: Tool[] = []
): Promise<string> {
	const draftId = task.updateId || Date.now();

	console.log(`[streamAiResponseToTelegram] Starting for task: ${task.type}, updateType: ${task.updateType}, model: ${modelId}`);

	if (task.updateType !== 'guest_message' && task.updateType !== 'business_message') {
		await sendMessageDraft(ctx.api, {
			chat_id: task.chatId,
			text: '> Thinking',
			parse_mode: 'MarkdownV2',
			message_thread_id: task.threadId,
			business_connection_id: task.businessConnectionId,
			draft_id: draftId,
		});
	}

	let streamContent = '';
	let thinkingContent = '';
	let reasoningContent = '';
	let hasSeenReasoning = false;

	const lastUpdate = { time: Date.now() };
	try {
		for await (const chunk of getAiStream(ai, modelId, messages, tools, (status) => {
			if (status === 'Reasoning') hasSeenReasoning = true;
		})) {
			if (chunk.type === 'thinking') {
				thinkingContent += chunk.text;
			} else if (chunk.type === 'reasoning') {
				reasoningContent += chunk.text;
				hasSeenReasoning = true;
			} else {
				streamContent += chunk.text;
			}
			const now = Date.now();
			if (
				task.updateType !== 'guest_message' &&
				task.updateType !== 'business_message' &&
				now - lastUpdate.time > 5000
			) {
				const text = (streamContent.trim() || reasoningContent.trim() || thinkingContent.trim())
					? await formatTelegramMessage(streamContent + (streamContent ? '...' : ''), thinkingContent + (thinkingContent && !streamContent ? '...' : ''), reasoningContent + (reasoningContent && !streamContent ? '...' : ''), true)
					: (hasSeenReasoning ? '> Reasoning' : '> Thinking');

				await sendMessageDraft(ctx.api, {
					chat_id: task.chatId,
					text,
					parse_mode: 'MarkdownV2',
					message_thread_id: task.threadId,
					business_connection_id: task.businessConnectionId,
					draft_id: draftId,
				});
				console.log(`[streamAiResponseToTelegram] Draft Task. StreamL:${streamContent.length} ThinkL:${thinkingContent.length} ReasonL:${reasoningContent.length} TotalL:${text.length}`);
				lastUpdate.time = now;
			}
		}
	} catch (e) {
		console.error(`[streamAiResponseToTelegram] Loop Error:`, e);
	}
	console.log(`[streamAiResponseToTelegram] Stream finished. ContentLength: ${streamContent.length}`);

	const TEXT_LIMIT = 3800;
	if (streamContent.length > TEXT_LIMIT) {
		console.log(`[streamAiResponseToTelegram] Limit Reached. ${streamContent.length} > ${TEXT_LIMIT}`);
		streamContent = streamContent.slice(0, TEXT_LIMIT) + '\n\n[Truncated due to Telegram length limit]';
	}

	if (streamContent.trim() || reasoningContent.trim() || thinkingContent.trim()) {
		const finalMessage = await formatTelegramMessage(streamContent, thinkingContent, reasoningContent);
		console.log(`[streamAiResponseToTelegram] Final Message Len: ${finalMessage.length} (Content: ${streamContent.length}, Think: ${thinkingContent.length}, Reason: ${reasoningContent.length})`);
		if (task.updateType === 'guest_message') {
			if (task.guestQueryId) {
				console.log('[streamAiResponseToTelegram] answerGuestQuery');
				await ctx.api
					.answerGuestQuery(task.guestQueryId, {
						type: 'article',
						id: crypto.randomUUID(),
						title: streamContent.slice(0, 64),
						input_message_content: {
							message_text: finalMessage,
							parse_mode: 'MarkdownV2',
						},
					})
					.catch((e: unknown) => console.error('[streamAiResponseToTelegram] Guest Error:', e));
			} else {
				console.warn('[streamAiResponseToTelegram] no guestQueryId for guest_message');
			}
		} else if (task.updateType === 'business_message') {
			console.log('[streamAiResponseToTelegram] sendMessage (business)');
			await ctx.api
				.sendMessage(task.chatId!, finalMessage, {
					parse_mode: 'MarkdownV2',
					message_thread_id: task.threadId,
					business_connection_id: task.businessConnectionId,
					reply_to_message_id: task.messageId,
				})
				.catch((e: unknown) => console.error('[streamAiResponseToTelegram] Business Message Error:', e));
		} else {
			console.log('[streamAiResponseToTelegram] final sendMessageDraft and sendMessage');
			await sendMessageDraft(ctx.api, {
				chat_id: task.chatId,
				text: finalMessage,
				parse_mode: 'MarkdownV2',
				message_thread_id: task.threadId,
				business_connection_id: task.businessConnectionId,
				draft_id: draftId,
				finish: true,
			});
			if (task.chatId) {
				await ctx.api.sendMessage(task.chatId, finalMessage, {
					parse_mode: 'MarkdownV2',
					message_thread_id: task.threadId,
					business_connection_id: task.businessConnectionId,
					reply_parameters: task.messageId ? { message_id: task.messageId } : undefined,
				}).catch((e: unknown) => console.error('[streamAiResponseToTelegram] Final Message Error:', e));
			}
		}
	} else {
		console.warn('[streamAiResponseToTelegram] No content to send.');
	}

	return streamContent;
}
