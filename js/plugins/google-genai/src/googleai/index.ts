/**
 * Copyright 2025 Google LLC
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

import { ActionMetadata, EmbedderReference, ModelReference, z } from 'genkit';
import { GenkitPluginV2, genkitPluginV2 } from 'genkit/plugin';
import type { ActionType } from 'genkit/registry';

// These are namespaced because they all intentionally have
// functions of the same name with the same arguments.
// (All exports from these files are used here)
import { listModels } from './client.js';
import * as embedder from './embedder.js';
import * as gemini from './gemini.js';
import * as imagen from './imagen.js';
import { GoogleAIPluginOptions } from './types.js';
import { getApiKeyFromEnvVar } from './utils.js';
import * as veo from './veo.js';

export { type EmbeddingConfig } from './embedder.js';
export { type GeminiConfig, type GeminiTtsConfig } from './gemini.js';
export { type ImagenConfig } from './imagen.js';
export { type GoogleAIPluginOptions };

/**
 * Google Gemini Developer API plugin.
 */
export function googleAIPlugin(
  options?: GoogleAIPluginOptions
): GenkitPluginV2 {
  // Cache for list function - per plugin instance
  let listCache: ActionMetadata[] | null = null;

  return genkitPluginV2({
    name: 'googleai',
    init: async () => {
      return [
        ...imagen.defineKnownModels(options),
        ...gemini.defineKnownModels(options),
        ...veo.defineKnownModels(options),
        ...embedder.defineKnownModels(options),
      ];
    },
    resolve: async (actionType: ActionType, name: string) => {
      // Extract the model name without the plugin prefix
      const modelName = name.replace(/^googleai\//, '');

      if (actionType === 'model') {
        if (veo.isVeoModelName(modelName)) {
          return veo.defineModel(modelName, options);
        }
        if (imagen.isImagenModelName(modelName)) {
          return imagen.defineModel(modelName, options);
        }
        // For gemini, tts, gemma, and unknown model families
        return gemini.defineModel(modelName, options);
      } else if (actionType === 'embedder') {
        return embedder.defineEmbedder(modelName, options);
      } else if (actionType === 'background-model') {
        if (veo.isVeoModelName(modelName)) {
          return veo.defineModel(modelName, options);
        }
      }

      return undefined;
    },
    list: async () => {
      // Return cached result if available
      if (listCache !== null) {
        return listCache;
      }

      const apiKey = options?.apiKey || getApiKeyFromEnvVar();

      if (!apiKey) {
        return [];
      }

      try {
        const models = await listModels(apiKey, options);
        const result = [
          ...imagen.listActions(models),
          ...gemini.listActions(models),
          ...veo.listActions(models),
          ...embedder.listActions(models),
        ];

        // Cache the result
        listCache = result;
        return result;
      } catch (error) {
        // Handle API errors gracefully by returning empty array
        console.warn('GoogleAI plugin: Failed to fetch models list:', error);
        return [];
      }
    },
  });
}

export type GoogleAIPlugin = {
  (pluginOptions?: GoogleAIPluginOptions): GenkitPluginV2;
  createModelRef(
    name: gemini.KnownGemmaModels | (gemini.GemmaModelName & {}),
    config: gemini.GemmaConfig
  ): ModelReference<gemini.GemmaConfigSchemaType>;
  createModelRef(
    name: gemini.KnownTtsModels | (gemini.TTSModelName & {}),
    config: gemini.GeminiTtsConfig
  ): ModelReference<gemini.GeminiTtsConfigSchemaType>;
  createModelRef(
    name: gemini.KnownGeminiModels | (gemini.GeminiModelName & {}),
    config?: gemini.GeminiConfig
  ): ModelReference<gemini.GeminiConfigSchemaType>;
  createModelRef(
    name: imagen.KnownModels | (imagen.ImagenModelName & {}),
    config?: imagen.ImagenConfig
  ): ModelReference<imagen.ImagenConfigSchemaType>;
  createModelRef(
    name: veo.KnownModels | (veo.VeoModelName & {}),
    config?: veo.VeoConfig
  ): ModelReference<veo.VeoConfigSchemaType>;
  createModelRef(name: string, config?: any): ModelReference<z.ZodTypeAny>;

  embedder(
    name: string,
    config?: embedder.EmbeddingConfig
  ): EmbedderReference<embedder.EmbeddingConfigSchemaType>;
};

/**
 * Google Gemini Developer API plugin.
 */
export const googleAI = googleAIPlugin as GoogleAIPlugin;
(googleAI as any).createModelRef = (
  name: string,
  config?: any
): ModelReference<z.ZodTypeAny> => {
  if (veo.isVeoModelName(name)) {
    return veo.createModelRef(name, config);
  }
  if (imagen.isImagenModelName(name)) {
    return imagen.createModelRef(name, config);
  }
  // gemma, tts, gemini and unknown model families.
  return gemini.createModelRef(name, config);
};
googleAI.embedder = (
  name: string,
  config?: embedder.EmbeddingConfig
): EmbedderReference<embedder.EmbeddingConfigSchemaType> => {
  return embedder.createModelRef(name, config);
};

export default googleAI;
