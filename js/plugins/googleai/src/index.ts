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
  embedderActionMetadata,
  embedderRef,
  modelActionMetadata,
  type ActionMetadata,
  type EmbedderReference,
  type ModelReference,
  type z,
} from 'genkit';
import { logger } from 'genkit/logging';
import { modelRef } from 'genkit/model';
import { genkitPluginV2, type GenkitPluginV2 } from 'genkit/plugin';
import type { ActionType } from 'genkit/registry';
import { getApiKeyFromEnvVar } from './common.js';
import {
  SUPPORTED_MODELS as EMBEDDER_MODELS,
  GeminiEmbeddingConfigSchema,
  defineGoogleAIEmbedder,
  geminiEmbedding001,
  textEmbedding004,
  textEmbeddingGecko001,
  type GeminiEmbeddingConfig,
} from './embedder.js';
import {
  GeminiConfigSchema,
  SUPPORTED_GEMINI_MODELS,
  defineGoogleAIModel,
  gemini,
  gemini10Pro,
  gemini15Flash,
  gemini15Flash8b,
  gemini15Pro,
  gemini20Flash,
  gemini20FlashExp,
  gemini20FlashLite,
  gemini20ProExp0205,
  gemini25FlashLite,
  gemini25FlashPreview0417,
  gemini25ProExp0325,
  gemini25ProPreview0325,
  type GeminiConfig,
  type GeminiVersionString,
} from './gemini.js';
import {
  GENERIC_IMAGEN_INFO,
  ImagenConfigSchema,
  defineImagenModel,
  type KNOWN_IMAGEN_MODELS,
} from './imagen.js';
import { listModels } from './list-models.js';
import {
  GENERIC_VEO_INFO,
  KNOWN_VEO_MODELS,
  VeoConfigSchema,
  defineVeoModel,
} from './veo.js';
export {
  gemini,
  gemini10Pro,
  gemini15Flash,
  gemini15Flash8b,
  gemini15Pro,
  gemini20Flash,
  gemini20FlashExp,
  gemini20FlashLite,
  gemini20ProExp0205,
  gemini25FlashLite,
  gemini25FlashPreview0417,
  gemini25ProExp0325,
  gemini25ProPreview0325,
  geminiEmbedding001,
  textEmbedding004,
  textEmbeddingGecko001,
  type GeminiConfig,
  type GeminiVersionString,
};

export interface PluginOptions {
  /**
   * Provide the API key to use to authenticate with the Gemini API. By
   * default, an API key must be provided explicitly here or through the
   * `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variables.
   *
   * If `false` is explicitly passed, the plugin will be configured to
   * expect an `apiKey` option to be provided to the model config at
   * call time.
   **/
  apiKey?: string | false;
  apiVersion?: string | string[];
  baseUrl?: string;
  models?: (
    | ModelReference</** @ignore */ typeof GeminiConfigSchema>
    | string
  )[];
  experimental_debugTraces?: boolean;
}

async function listActions(options?: PluginOptions): Promise<ActionMetadata[]> {
  const apiKey = options?.apiKey || getApiKeyFromEnvVar();
  if (!apiKey) {
    // If API key is not configured we don't want to error, just return empty.
    // In practice it will never actually reach this point without the API key,
    // plugin initializer will fail before that.
    logger.error(
      'Pass in the API key or set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable.'
    );
    return [];
  }

  const models = await listModels(
    options?.baseUrl || 'https://generativelanguage.googleapis.com',
    apiKey
  );

  return [
    // Imagen
    ...models
      .filter(
        (m) =>
          m.supportedGenerationMethods.includes('predict') &&
          m.name.includes('imagen')
      )
      // Filter out deprecated
      .filter((m) => !m.description || !m.description.includes('deprecated'))
      .map((m) => {
        const imagenModelName = m.name.split('/').at(-1)!;

        return modelActionMetadata({
          name: imagenModelName,
          info: { ...GENERIC_IMAGEN_INFO },
          configSchema: ImagenConfigSchema,
        });
      }),
    // Veo
    ...models
      .filter(
        (m) =>
          m.supportedGenerationMethods.includes('predictLongRunning') &&
          m.name.includes('veo')
      )
      // Filter out deprecated
      .filter((m) => !m.description || !m.description.includes('deprecated'))
      .map((m) => {
        const veoModelName = m.name.split('/').at(-1)!;

        return modelActionMetadata({
          name: veoModelName,
          info: { ...GENERIC_VEO_INFO },
          configSchema: VeoConfigSchema,
          background: true,
        });
      }),
    // Models
    ...models
      .filter((m) => m.supportedGenerationMethods.includes('generateContent'))
      // Filter out deprecated
      .filter((m) => !m.description || !m.description.includes('deprecated'))
      .map((m) => {
        const modelName = m.name.startsWith('models/')
          ? m.name.substring('models/'.length)
          : m.name;
        const ref = gemini(modelName);

        return modelActionMetadata({
          name: modelName,
          info: ref.info,
          configSchema: GeminiConfigSchema,
        });
      }),
    // Embedders
    ...models
      .filter((m) => m.supportedGenerationMethods.includes('embedContent'))
      // Filter out deprecated
      .filter((m) => !m.description || !m.description.includes('deprecated'))
      .map((m) => {
        const embedderName = m.name.startsWith('models/')
          ? m.name.substring('models/'.length)
          : m.name;

        return embedderActionMetadata({
          name: embedderName,
          configSchema: GeminiEmbeddingConfigSchema,
          info: {
            dimensions: 768,
            label: `Google Gen AI - ${embedderName}`,
            supports: {
              input: ['text'],
            },
          },
        });
      }),
  ];
}

/**
 * Google Gemini Developer API plugin.
 */
export function googleAIPlugin(options?: PluginOptions): GenkitPluginV2 {
  let listActionsCache;
  return genkitPluginV2({
    name: 'googleai',
    async init() {
      const actions: any[] = [];
      const apiVersions = Array.isArray(options?.apiVersion)
        ? options!.apiVersion
        : options?.apiVersion
          ? [options.apiVersion]
          : ['v1'];

      if (apiVersions.includes('v1beta')) {
        Object.keys(SUPPORTED_GEMINI_MODELS).forEach((name) => {
          actions.push(
            defineGoogleAIModel({
              name: name,
              apiKey: options?.apiKey,
              apiVersion: 'v1beta',
              baseUrl: options?.baseUrl,
              debugTraces: options?.experimental_debugTraces,
            })
          );
        });
      }

      if (apiVersions.includes('v1')) {
        Object.keys(SUPPORTED_GEMINI_MODELS).forEach((name) => {
          actions.push(
            defineGoogleAIModel({
              name: name,
              apiKey: options?.apiKey,
              baseUrl: options?.baseUrl,
              debugTraces: options?.experimental_debugTraces,
            })
          );
        });
        Object.keys(EMBEDDER_MODELS).forEach((name) => {
          actions.push(
            defineGoogleAIEmbedder(name, {
              apiKey: options?.apiKey,
            })
          );
        });
      }

      if (options?.models) {
        for (const modelOrRef of options.models) {
          const modelName =
            typeof modelOrRef === 'string'
              ? modelOrRef
              : modelOrRef.name.split('/').pop()!;
          const ref =
            typeof modelOrRef === 'string' ? gemini(modelOrRef) : modelOrRef;
          actions.push(
            defineGoogleAIModel({
              name: modelName,
              apiKey: options?.apiKey,
              baseUrl: options?.baseUrl,
              info: { ...ref.info, label: `Google AI - ${modelName}` },
              debugTraces: options?.experimental_debugTraces,
            })
          );
        }
      }
      return actions;
    },
    async resolve(actionType: ActionType, actionName: string) {
      // In v2, actionName is already unprefixed
      if (actionType === 'embedder') {
        return defineGoogleAIEmbedder(actionName, {
          apiKey: options?.apiKey,
        });
      } else if (actionName.startsWith('veo')) {
        if (actionType === 'background-model') {
          return defineVeoModel(actionName, options?.apiKey);
        }
      } else if (actionType === 'model') {
        if (actionName.startsWith('imagen')) {
          return defineImagenModel(actionName, options?.apiKey);
        }
        // Fallback dynamic Gemini model
        return defineGoogleAIModel({
          name: actionName,
          apiKey: options?.apiKey,
          baseUrl: options?.baseUrl,
          debugTraces: options?.experimental_debugTraces,
        });
      }
      return undefined;
    },
    async list() {
      if (listActionsCache) return listActionsCache;
      listActionsCache = await listActions(options);
      return listActionsCache;
    },
  });
}

export type GoogleAIPlugin = {
  (params?: PluginOptions): GenkitPluginV2;
  model(
    name: keyof typeof SUPPORTED_GEMINI_MODELS | (`gemini-${string}` & {}),
    config?: z.infer<typeof GeminiConfigSchema>
  ): ModelReference<typeof GeminiConfigSchema>;
  model(
    name: KNOWN_IMAGEN_MODELS | (`imagen${string}` & {}),
    config?: z.infer<typeof ImagenConfigSchema>
  ): ModelReference<typeof ImagenConfigSchema>;
  model(
    name: KNOWN_VEO_MODELS | (`veo${string}` & {}),
    config?: z.infer<typeof VeoConfigSchema>
  ): ModelReference<typeof VeoConfigSchema>;
  model(name: string, config?: any): ModelReference<z.ZodTypeAny>;
  embedder(
    name: string,
    config?: GeminiEmbeddingConfig
  ): EmbedderReference<typeof GeminiEmbeddingConfigSchema>;
};

/**
 * Google Gemini Developer API plugin.
 */
export const googleAI = googleAIPlugin as GoogleAIPlugin;
// provide generic implementation for the model function overloads.
(googleAI as any).model = (
  name: string,
  config?: any
): ModelReference<z.ZodTypeAny> => {
  if (name.startsWith('imagen')) {
    return modelRef({
      name: name,
      config,
      configSchema: ImagenConfigSchema,
    });
  }
  if (name.startsWith('veo')) {
    return modelRef({
      name: name,
      config,
      configSchema: VeoConfigSchema,
    });
  }
  return modelRef({
    name: name,
    config,
    configSchema: GeminiConfigSchema,
  });
};
googleAI.embedder = (
  name: string,
  config?: GeminiEmbeddingConfig
): EmbedderReference<typeof GeminiEmbeddingConfigSchema> => {
  return embedderRef({
    name: name,
    config,
    configSchema: GeminiEmbeddingConfigSchema,
  });
};

export default googleAI;
