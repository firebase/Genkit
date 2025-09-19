/**
 * @license
 *
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

/**
 * @module /
 */

import { EmbedderReference, ModelReference, z } from 'genkit';
import { GenkitPluginV2, genkitPluginV2 } from 'genkit/plugin';

import * as embedder from './embedder.js';
import * as gemini from './gemini.js';
import * as imagen from './imagen.js';
import * as lyria from './lyria.js';
import * as veo from './veo.js';

import { ActionType } from 'genkit/registry';
import { listModels } from './client.js';
import { VertexPluginOptions } from './types.js';
import { getDerivedOptions } from './utils.js';

export { type EmbeddingConfig } from './embedder.js';
export { type GeminiConfig } from './gemini.js';
export { type ImagenConfig } from './imagen.js';
export { type LyriaConfig } from './lyria.js';
export { type VertexPluginOptions } from './types.js';
export { type VeoConfig } from './veo.js';

/**
 * Add Google Cloud Vertex AI to Genkit. Includes Gemini and Imagen models and text embedder.
 */
function vertexAIPlugin(options?: VertexPluginOptions): GenkitPluginV2 {
  // Cache for list function - per plugin instance
  let listCache: any[] | null = null;

  return genkitPluginV2({
    name: 'vertexai',
    init: async () => {
      const clientOptions = await getDerivedOptions(options);
      return [
        ...veo.defineKnownModels(clientOptions, options),
        ...imagen.defineKnownModels(clientOptions, options),
        ...lyria.defineKnownModels(clientOptions, options),
        ...gemini.defineKnownModels(clientOptions, options),
        ...embedder.defineKnownModels(clientOptions, options),
      ];
    },
    resolve: async (actionType: ActionType, name: string) => {
      const clientOptions = await getDerivedOptions(options);
      if (actionType === 'model') {
        return gemini.defineModel(name, clientOptions);
      } else if (actionType === 'embedder') {
        return embedder.defineEmbedder(name, clientOptions);
      } else if (actionType === 'background-model') {
        return veo.defineModel(name, clientOptions);
      }
      return undefined;
    },
    list: async () => {
      // Return cached result if available
      if (listCache !== null) {
        return listCache;
      }

      try {
        // Allow undefined options for testing, but try to get derived options
        const clientOptions = await getDerivedOptions(options);
        const models = await listModels(clientOptions);
        const result = [
          ...imagen.listActions(models),
          ...lyria.listActions(models),
          ...veo.listActions(models),
          ...gemini.listActions(models),
          // Note: embedders are excluded from list() for VertexAI plugin
        ];

        // Cache the result
        listCache = result;
        return result;
      } catch (error) {
        // Handle errors gracefully by returning empty array
        console.warn('VertexAI plugin: Failed to fetch models list:', error);
        return [];
      }
    },
  });
}

export type VertexAIPlugin = {
  (pluginOptions?: VertexPluginOptions): GenkitPluginV2;
  createModelRef(
    name: gemini.KnownModels | (gemini.GeminiModelName & {}),
    config?: gemini.GeminiConfig
  ): ModelReference<gemini.GeminiConfigSchemaType>;
  createModelRef(
    name: imagen.KnownModels | (imagen.ImagenModelName & {}),
    config?: imagen.ImagenConfig
  ): ModelReference<imagen.ImagenConfigSchemaType>;
  createModelRef(
    name: lyria.KnownModels | (lyria.LyriaModelName & {}),
    config: lyria.LyriaConfig
  ): ModelReference<lyria.LyriaConfigSchemaType>;
  createModelRef(
    name: veo.KnownModels | (veo.VeoModelName & {}),
    config: veo.VeoConfig
  ): ModelReference<veo.VeoConfigSchemaType>;
  createModelRef(name: string, config?: any): ModelReference<z.ZodTypeAny>;

  embedder(
    name: string,
    config?: embedder.EmbeddingConfig
  ): EmbedderReference<embedder.EmbeddingConfigSchemaType>;
};

/**
 * Google Cloud Vertex AI plugin for Genkit.
 * Includes Gemini and Imagen models and text embedder.
 */
export const vertexAI = vertexAIPlugin as VertexAIPlugin;
// provide generic implementation for the model function overloads.
(vertexAI as any).createModelRef = (
  name: string,
  config?: any
): ModelReference<z.ZodTypeAny> => {
  if (imagen.isImagenModelName(name)) {
    return imagen.createModelRef(name, config);
  }
  if (lyria.isLyriaModelName(name)) {
    return lyria.createModelRef(name, config);
  }
  if (veo.isVeoModelName(name)) {
    return veo.createModelRef(name, config);
  }
  // gemini and unknown model families
  return gemini.createModelRef(name, config);
};
vertexAI.embedder = (
  name: string,
  config?: embedder.EmbeddingConfig
): EmbedderReference<embedder.EmbeddingConfigSchemaType> => {
  return embedder.createModelRef(name, config);
};

export default vertexAI;
