/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { Genkit, ToolRequest, ToolRequestPart, ToolResponse } from 'genkit';
import { logger } from 'genkit/logging';
import {
  GenerateRequest,
  GenerateResponseData,
  GenerationCommonConfigSchema,
  MessageData,
  ToolDefinition,
  getBasicUsageStats,
} from 'genkit/model';
import { GenkitPlugin, genkitPlugin } from 'genkit/plugin';
import { defineOllamaEmbedder } from './embeddings.js';
import {
  ApiType,
  Message,
  ModelDefinition,
  OllamaTool,
  OllamaToolCall,
  RequestHeaders,
  type OllamaPluginParams,
} from './types.js';

export { type OllamaPluginParams };

const ANY_JSON_SCHEMA: Record<string, any> = {
  $schema: 'http://json-schema.org/draft-07/schema#',
};

export function ollama(params: OllamaPluginParams): GenkitPlugin {
  return genkitPlugin('ollama', async (ai: Genkit) => {
    const serverAddress = params.serverAddress;
    params.models?.map((model) =>
      ollamaModel(ai, model, serverAddress, params.requestHeaders)
    );
    params.embedders?.map((model) =>
      defineOllamaEmbedder(ai, {
        name: model.name,
        modelName: model.name,
        dimensions: model.dimensions,
        options: params,
      })
    );
  });
}

function ollamaModel(
  ai: Genkit,
  model: ModelDefinition,
  serverAddress: string,
  requestHeaders?: RequestHeaders
) {
  return ai.defineModel(
    {
      name: `ollama/${model.name}`,
      label: `Ollama - ${model.name}`,
      configSchema: GenerationCommonConfigSchema,
      supports: {
        multiturn: !model.type || model.type === 'chat',
        systemRole: true,
        tools: model.supports?.tools,
      },
    },
    async (input, streamingCallback) => {
      const options: Record<string, any> = {};
      if (input.config?.temperature !== undefined) {
        options.temperature = input.config.temperature;
      }
      if (input.config?.topP !== undefined) {
        options.top_p = input.config.topP;
      }
      if (input.config?.topK !== undefined) {
        options.top_k = input.config.topK;
      }
      if (input.config?.stopSequences !== undefined) {
        options.stop = input.config.stopSequences.join('');
      }
      if (input.config?.maxOutputTokens !== undefined) {
        options.num_predict = input.config.maxOutputTokens;
      }
      const type = model.type ?? 'chat';
      const request = toOllamaRequest(
        model.name,
        input,
        options,
        type,
        !!streamingCallback
      );
      logger.debug(request, `ollama request (${type})`);

      const extraHeaders = requestHeaders
        ? typeof requestHeaders === 'function'
          ? await requestHeaders(
              {
                serverAddress,
                model,
              },
              input
            )
          : requestHeaders
        : {};

      let res;
      try {
        res = await fetch(
          serverAddress + (type === 'chat' ? '/api/chat' : '/api/generate'),
          {
            method: 'POST',
            body: JSON.stringify(request),
            headers: {
              'Content-Type': 'application/json',
              ...extraHeaders,
            },
          }
        );
      } catch (e) {
        const cause = (e as any).cause;
        if (
          cause &&
          cause instanceof Error &&
          cause.message?.includes('ECONNREFUSED')
        ) {
          cause.message += '. Make sure the Ollama server is running.';
          throw cause;
        }
        throw e;
      }
      if (!res.body) {
        throw new Error('Response has no body');
      }

      let message: MessageData;

      if (streamingCallback) {
        const reader = res.body.getReader();
        const textDecoder = new TextDecoder();
        let textResponse = '';
        for await (const chunk of readChunks(reader)) {
          const chunkText = textDecoder.decode(chunk);
          const json = JSON.parse(chunkText);
          const message = parseMessage(json, type);
          streamingCallback({
            index: 0,
            content: message.content,
          });
          textResponse += message.content[0].text;
        }
        message = {
          role: 'model',
          content: [
            {
              text: textResponse,
            },
          ],
        };
      } else {
        const txtBody = await res.text();
        const json = JSON.parse(txtBody);
        logger.debug(txtBody, 'ollama raw response');

        message = parseMessage(json, type);
      }

      return {
        message,
        usage: getBasicUsageStats(input.messages, message),
        finishReason: 'stop',
      } as GenerateResponseData;
    }
  );
}

function parseMessage(response: any, type: ApiType): MessageData {
  if (response.error) {
    throw new Error(response.error);
  }
  if (type === 'chat') {
    // Tool calling is available only on the 'chat' API, not on 'generate'
    // https://github.com/ollama/ollama/blob/main/docs/api.md#chat-request-with-tools
    if (response.message.tool_calls && response.message.tool_calls.length > 0) {
      return {
        role: toGenkitRole(response.message.role),
        content: toGenkitToolRequest(response.message.tool_calls),
      };
    } else {
      return {
        role: toGenkitRole(response.message.role),
        content: [
          {
            text: response.message.content,
          },
        ],
      };
    }
  } else {
    return {
      role: 'model',
      content: [
        {
          text: response.response,
        },
      ],
    };
  }
}

function toOllamaRequest(
  name: string,
  input: GenerateRequest,
  options: Record<string, any>,
  type: ApiType,
  stream: boolean
) {
  const request: any = {
    model: name,
    options,
    stream,
    tools: input.tools?.filter(isValidOllamaTool).map(toOllamaTool),
  };
  if (type === 'chat') {
    const messages: Message[] = [];
    input.messages.forEach((m) => {
      let messageText = '';
      const role = toOllamaRole(m.role);
      const images: string[] = [];
      const toolRequests: ToolRequest[] = [];
      const toolResponses: ToolResponse[] = [];
      m.content.forEach((c) => {
        if (c.text) {
          messageText += c.text;
        }
        if (c.media) {
          images.push(c.media.url);
        }
        if (c.toolRequest) {
          toolRequests.push(c.toolRequest);
        }
        if (c.toolResponse) {
          toolResponses.push(c.toolResponse);
        }
      });
      // Add tool responses, if any.
      toolResponses.forEach((t) => {
        messages.push({
          role,
          content:
            typeof t.output === 'string' ? t.output : JSON.stringify(t.output),
        });
      });
      messages.push({
        role: role,
        content: toolRequests.length > 0 ? '' : messageText,
        images: images.length > 0 ? images : undefined,
        tool_calls:
          toolRequests.length > 0 ? toOllamaToolCall(toolRequests) : undefined,
      });
    });
    request.messages = messages;
  } else {
    request.prompt = getPrompt(input);
    request.system = getSystemMessage(input);
  }
  return request;
}

function toOllamaRole(role) {
  if (role === 'model') {
    return 'assistant';
  }
  return role; // everything else seems to match
}

function toGenkitRole(role) {
  if (role === 'assistant') {
    return 'model';
  }
  return role; // everything else seems to match
}

function toOllamaTool(tool: ToolDefinition): OllamaTool {
  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.inputSchema ?? ANY_JSON_SCHEMA,
    },
  };
}

function toOllamaToolCall(toolRequests: ToolRequest[]): OllamaToolCall[] {
  return toolRequests.map((t) => ({
    function: {
      name: t.name,
      // This should be safe since we already filtered tools that don't accept
      // objects
      arguments: t.input as Record<string, any>,
    },
  }));
}

function toGenkitToolRequest(tool_calls: OllamaToolCall[]): ToolRequestPart[] {
  return tool_calls.map((t) => ({
    toolRequest: {
      name: t.function.name,
      ref: t.function.index ? t.function.index.toString() : undefined,
      input: t.function.arguments,
    },
  }));
}

function readChunks(reader) {
  return {
    async *[Symbol.asyncIterator]() {
      let readResult = await reader.read();
      while (!readResult.done) {
        yield readResult.value;
        readResult = await reader.read();
      }
    },
  };
}

function getPrompt(input: GenerateRequest): string {
  return input.messages
    .filter((m) => m.role !== 'system')
    .map((m) => m.content.map((c) => c.text).join())
    .join();
}

function getSystemMessage(input: GenerateRequest): string {
  return input.messages
    .filter((m) => m.role === 'system')
    .map((m) => m.content.map((c) => c.text).join())
    .join();
}

function isValidOllamaTool(tool: ToolDefinition): boolean {
  if (tool.inputSchema?.type !== 'object') {
    throw new Error(
      `Unsupported tool: '${tool.name}'. Ollama only supports tools with object inputs`
    );
  }
  return true;
}
