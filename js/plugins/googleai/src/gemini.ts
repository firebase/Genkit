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

import {
  CandidateData,
  defineModel,
  GenerationCommonConfigSchema,
  getBasicUsageStats,
  MediaPart,
  MessageData,
  ModelAction,
  ModelMiddleware,
  modelRef,
  ModelReference,
  Part,
  ToolDefinitionSchema,
  ToolRequestPart,
  ToolResponsePart,
} from '@genkit-ai/ai/model';
import {
  downloadRequestMedia,
  simulateSystemPrompt,
} from '@genkit-ai/ai/model/middleware';
import { GENKIT_CLIENT_HEADER } from '@genkit-ai/core';
import {
  FunctionCallPart,
  FunctionDeclaration,
  FunctionDeclarationSchemaType,
  FunctionResponsePart,
  GenerateContentCandidate as GeminiCandidate,
  Content as GeminiMessage,
  Part as GeminiPart,
  GenerateContentResponse,
  GenerationConfig,
  GoogleGenerativeAI,
  InlineDataPart,
  RequestOptions,
  StartChatParams,
} from '@google/generative-ai';
import process from 'process';
import z from 'zod';

const SafetySettingsSchema = z.object({
  category: z.enum([
    'HARM_CATEGORY_UNSPECIFIED',
    'HARM_CATEGORY_HATE_SPEECH',
    'HARM_CATEGORY_SEXUALLY_EXPLICIT',
    'HARM_CATEGORY_HARASSMENT',
    'HARM_CATEGORY_DANGEROUS_CONTENT',
  ]),
  threshold: z.enum([
    'BLOCK_LOW_AND_ABOVE',
    'BLOCK_MEDIUM_AND_ABOVE',
    'BLOCK_ONLY_HIGH',
    'BLOCK_NONE',
  ]),
});

const GeminiConfigSchema = GenerationCommonConfigSchema.extend({
  safetySettings: z.array(SafetySettingsSchema).optional(),
});

export const geminiPro = modelRef({
  name: 'googleai/gemini-pro',
  info: {
    label: 'Google AI - Gemini Pro',
    versions: ['gemini-1.0-pro', 'gemini-1.0-pro-latest', 'gemini-1.0-pro-001'],
    supports: {
      multiturn: true,
      media: false,
      tools: true,
      systemRole: true,
    },
  },
  configSchema: GeminiConfigSchema,
});

export const geminiProVision = modelRef({
  name: 'googleai/gemini-pro-vision',
  info: {
    label: 'Google AI - Gemini Pro Vision',
    // none declared on https://ai.google.dev/models/gemini#model-variations
    versions: [],
    supports: {
      multiturn: true,
      media: true,
      tools: false,
      systemRole: false,
    },
  },
  configSchema: GeminiConfigSchema,
});

export const gemini15Pro = modelRef({
  name: 'googleai/gemini-1.5-pro-latest',
  info: {
    label: 'Google AI - Gemini 1.5 Pro',
    supports: {
      multiturn: true,
      media: true,
      tools: true,
      systemRole: true,
      output: ['text', 'json'],
    },
    versions: ['gemini-1.5-pro-001'],
  },
  configSchema: GeminiConfigSchema,
});

export const gemini15Flash = modelRef({
  name: 'googleai/gemini-1.5-flash-latest',
  info: {
    label: 'Google AI - Gemini 1.5 Flash',
    supports: {
      multiturn: true,
      media: true,
      tools: true,
      systemRole: true,
      output: ['text', 'json'],
    },
    versions: ['gemini-1.5-flash-001'],
  },
  configSchema: GeminiConfigSchema,
});

export const geminiUltra = modelRef({
  name: 'googleai/gemini-ultra',
  info: {
    label: 'Google AI - Gemini Ultra',
    versions: [],
    supports: {
      multiturn: true,
      media: false,
      tools: true,
      systemRole: true,
    },
  },
  configSchema: GeminiConfigSchema,
});

export const SUPPORTED_V1_MODELS: Record<
  string,
  ModelReference<z.ZodTypeAny>
> = {
  'gemini-pro': geminiPro,
  'gemini-pro-vision': geminiProVision,
  // 'gemini-ultra': geminiUltra,
};

export const SUPPORTED_V15_MODELS: Record<
  string,
  ModelReference<z.ZodTypeAny>
> = {
  'gemini-1.5-pro-latest': gemini15Pro,
  'gemini-1.5-flash-latest': gemini15Flash,
};

const SUPPORTED_MODELS = {
  ...SUPPORTED_V1_MODELS,
  ...SUPPORTED_V15_MODELS,
};

function toGeminiRole(
  role: MessageData['role'],
  model?: ModelReference<z.ZodTypeAny>
): string {
  switch (role) {
    case 'user':
      return 'user';
    case 'model':
      return 'model';
    case 'system':
      if (model && SUPPORTED_V15_MODELS[model.name]) {
        // We should have already pulled out the supported system messages,
        // anything remaining is unsupported; throw an error.
        throw new Error(
          'system role is only supported for a single message in the first position'
        );
      } else {
        throw new Error('system role is not supported');
      }
    case 'tool':
      return 'function';
    default:
      return 'user';
  }
}

function convertSchemaProperty(property) {
  if (!property) {
    return null;
  }
  if (property.type === 'object') {
    const nestedProperties = {};
    Object.keys(property.properties).forEach((key) => {
      nestedProperties[key] = convertSchemaProperty(property.properties[key]);
    });
    return {
      type: FunctionDeclarationSchemaType.OBJECT,
      properties: nestedProperties,
      required: property.required,
    };
  } else if (property.type === 'array') {
    return {
      type: FunctionDeclarationSchemaType.ARRAY,
      items: convertSchemaProperty(property.items),
    };
  } else {
    return {
      type: FunctionDeclarationSchemaType[property.type.toUpperCase()],
    };
  }
}

function toGeminiTool(
  tool: z.infer<typeof ToolDefinitionSchema>
): FunctionDeclaration {
  const declaration: FunctionDeclaration = {
    name: tool.name.replace(/\//g, '__'), // Gemini throws on '/' in tool name
    description: tool.description,
    parameters: convertSchemaProperty(tool.inputSchema),
  };
  return declaration;
}

function toInlineData(part: MediaPart): InlineDataPart {
  const dataUrl = part.media.url;
  const b64Data = dataUrl.substring(dataUrl.indexOf(',')! + 1);
  const contentType =
    part.media.contentType ||
    dataUrl.substring(dataUrl.indexOf(':')! + 1, dataUrl.indexOf(';'));
  return { inlineData: { mimeType: contentType, data: b64Data } };
}

function fromInlineData(inlinePart: InlineDataPart): MediaPart {
  // Check if the required properties exist
  if (
    !inlinePart.inlineData ||
    !inlinePart.inlineData.hasOwnProperty('mimeType') ||
    !inlinePart.inlineData.hasOwnProperty('data')
  ) {
    throw new Error('Invalid InlineDataPart: missing required properties');
  }
  const { mimeType, data } = inlinePart.inlineData;
  // Combine data and mimeType into a data URL
  const dataUrl = `data:${mimeType};base64,${data}`;
  return {
    media: {
      url: dataUrl,
      contentType: mimeType,
    },
  };
}

function toFunctionCall(part: ToolRequestPart): FunctionCallPart {
  if (!part?.toolRequest?.input) {
    throw Error('Invalid ToolRequestPart: input was missing.');
  }
  return {
    functionCall: {
      name: part.toolRequest.name,
      args: part.toolRequest.input,
    },
  };
}

function fromFunctionCall(part: FunctionCallPart): ToolRequestPart {
  if (!part.functionCall) {
    throw Error('Invalid FunctionCallPart');
  }
  return {
    toolRequest: {
      name: part.functionCall.name,
      input: part.functionCall.args,
    },
  };
}

function toFunctionResponse(part: ToolResponsePart): FunctionResponsePart {
  if (!part?.toolResponse?.output) {
    throw Error('Invalid ToolResponsePart: output was missing.');
  }
  return {
    functionResponse: {
      name: part.toolResponse.name,
      response: {
        name: part.toolResponse.name,
        content: part.toolResponse.output,
      },
    },
  };
}

function fromFunctionResponse(part: FunctionResponsePart): ToolResponsePart {
  if (!part.functionResponse) {
    throw new Error('Invalid FunctionResponsePart.');
  }
  return {
    toolResponse: {
      name: part.functionResponse.name.replace(/__/g, '/'), // restore slashes
      output: part.functionResponse.response,
    },
  };
}

function toGeminiPart(part: Part): GeminiPart {
  if (part.text !== undefined) return { text: part.text };
  if (part.media) return toInlineData(part);
  if (part.toolRequest) return toFunctionCall(part);
  if (part.toolResponse) return toFunctionResponse(part);
  throw new Error('Unsupported Part type');
}

function fromGeminiPart(part: GeminiPart, jsonMode: boolean): Part {
  if (jsonMode && part.text !== undefined) {
    return { data: JSON.parse(part.text) };
  }
  if (part.text !== undefined) return { text: part.text };
  if (part.inlineData) return fromInlineData(part);
  if (part.functionCall) return fromFunctionCall(part);
  if (part.functionResponse) return fromFunctionResponse(part);
  throw new Error('Unsupported GeminiPart type');
}

export function toGeminiMessage(
  message: MessageData,
  model?: ModelReference<z.ZodTypeAny>
): GeminiMessage {
  return {
    role: toGeminiRole(message.role, model),
    parts: message.content.map(toGeminiPart),
  };
}

export function toGeminiSystemInstruction(message: MessageData): GeminiMessage {
  return {
    role: 'user',
    parts: message.content.map(toGeminiPart),
  };
}

function fromGeminiFinishReason(
  reason: GeminiCandidate['finishReason']
): CandidateData['finishReason'] {
  if (!reason) return 'unknown';
  switch (reason) {
    case 'STOP':
      return 'stop';
    case 'MAX_TOKENS':
      return 'length';
    case 'SAFETY': // blocked for safety
    case 'RECITATION': // blocked for reciting training data
      return 'blocked';
    default:
      return 'unknown';
  }
}

export function fromGeminiCandidate(
  candidate: GeminiCandidate,
  jsonMode: boolean = false
): CandidateData {
  return {
    index: candidate.index || 0, // reasonable default?
    message: {
      role: 'model',
      content: (candidate.content?.parts || []).map((part) =>
        fromGeminiPart(part, jsonMode)
      ),
    },
    finishReason: fromGeminiFinishReason(candidate.finishReason),
    finishMessage: candidate.finishMessage,
    custom: {
      safetyRatings: candidate.safetyRatings,
      citationMetadata: candidate.citationMetadata,
    },
  };
}

/**
 *
 */
export function googleAIModel(
  name: string,
  apiKey?: string,
  apiVersion?: string,
  baseUrl?: string
): ModelAction {
  const modelName = `googleai/${name}`;

  if (!apiKey) {
    apiKey = process.env.GOOGLE_GENAI_API_KEY || process.env.GOOGLE_API_KEY;
  }
  if (!apiKey) {
    throw new Error(
      'Please pass in the API key or set the GOOGLE_GENAI_API_KEY or GOOGLE_API_KEY environment variable.\n' +
        'For more details see https://firebase.google.com/docs/genkit/plugins/google-genai'
    );
  }

  const model: ModelReference<z.ZodTypeAny> = SUPPORTED_MODELS[name];
  if (!model) throw new Error(`Unsupported model: ${name}`);

  const middleware: ModelMiddleware[] = [];
  if (SUPPORTED_V1_MODELS[name]) {
    middleware.push(simulateSystemPrompt());
  }
  if (model?.info?.supports?.media) {
    // the gemini api doesn't support downloading media from http(s)
    middleware.push(downloadRequestMedia({ maxBytes: 1024 * 1024 * 10 }));
  }

  return defineModel(
    {
      name: modelName,
      ...model.info,
      configSchema: model.configSchema,
      use: middleware,
    },
    async (request, streamingCallback) => {
      const options: RequestOptions = { apiClient: GENKIT_CLIENT_HEADER };
      if (apiVersion) {
        options.apiVersion = apiVersion;
      }
      if (apiVersion) {
        options.baseUrl = baseUrl;
      }
      const client = new GoogleGenerativeAI(apiKey!).getGenerativeModel(
        {
          model: request.config?.version || model.version || name,
        },
        options
      );

      // make a copy so that modifying the request will not produce side-effects
      const messages = [...request.messages];
      if (messages.length === 0) throw new Error('No messages provided.');

      // Gemini does not support messages with role system and instead expects
      // systemInstructions to be provided as a separate input. The first
      // message detected with role=system will be used for systemInstructions.
      // Any additional system messages may be considered to be "exceptional".
      let systemInstruction: GeminiMessage | undefined = undefined;
      if (SUPPORTED_V15_MODELS[name]) {
        const systemMessage = messages.find((m) => m.role === 'system');
        if (systemMessage) {
          messages.splice(messages.indexOf(systemMessage), 1);
          systemInstruction = toGeminiSystemInstruction(systemMessage);
        }
      }

      const strictlyTypedGenerationConfig: GenerationConfig = {
        candidateCount: request.candidates || undefined,
        temperature: request.config?.temperature,
        maxOutputTokens: request.config?.maxOutputTokens,
        topK: request.config?.topK,
        topP: request.config?.topP,
        stopSequences: request.config?.stopSequences,
        responseMimeType:
          request.output?.format === 'json' || request.output?.schema
            ? 'application/json'
            : undefined,
      };

      const chatRequest = {
        systemInstruction,
        tools: request.tools?.length
          ? [{ functionDeclarations: request.tools?.map(toGeminiTool) }]
          : [],
        history: messages
          .slice(0, -1)
          .map((message) => toGeminiMessage(message, model)),
        generationConfig: strictlyTypedGenerationConfig,
        safetySettings: request.config?.safetySettings,
      } as StartChatParams;
      const msg = toGeminiMessage(messages[messages.length - 1], model);

      const jsonMode =
        request.output?.format === 'json' || !!request.output?.schema;
      const fromJSONModeScopedGeminiCandidate = (
        candidate: GeminiCandidate
      ) => {
        return fromGeminiCandidate(candidate, jsonMode);
      };

      if (streamingCallback) {
        const result = await client
          .startChat(chatRequest)
          .sendMessageStream(msg.parts);
        for await (const item of result.stream) {
          (item as GenerateContentResponse).candidates?.forEach((candidate) => {
            const c = fromJSONModeScopedGeminiCandidate(candidate);
            streamingCallback({
              index: c.index,
              content: c.message.content,
            });
          });
        }
        const response = await result.response;
        if (!response.candidates?.length) {
          throw new Error('No valid candidates returned.');
        }
        return {
          candidates:
            response.candidates?.map(fromJSONModeScopedGeminiCandidate) || [],
          custom: response,
        };
      } else {
        const result = await client
          .startChat(chatRequest)
          .sendMessage(msg.parts);
        if (!result.response.candidates?.length)
          throw new Error('No valid candidates returned.');
        const responseCandidates =
          result.response.candidates?.map(fromJSONModeScopedGeminiCandidate) ||
          [];
        return {
          candidates: responseCandidates,
          custom: result.response,
          usage: {
            ...getBasicUsageStats(request.messages, responseCandidates),
            inputTokens: result.response.usageMetadata?.promptTokenCount,
            outputTokens: result.response.usageMetadata?.candidatesTokenCount,
            totalTokens: result.response.usageMetadata?.totalTokenCount,
          },
        };
      }
    }
  );
}
