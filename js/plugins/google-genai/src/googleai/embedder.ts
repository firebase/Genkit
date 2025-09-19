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

import {
  ActionMetadata,
  EmbedderAction,
  embedderActionMetadata,
  EmbedderInfo,
  EmbedderReference,
  z,
} from 'genkit';
import { embedderRef } from 'genkit/embedder';
import { embedder } from 'genkit/plugin';
import { embedContent } from './client.js';
import {
  EmbedContentRequest,
  GoogleAIPluginOptions,
  Model,
  TaskTypeSchema,
} from './types.js';
import {
  calculateApiKey,
  checkApiKey,
  checkModelName,
  extractVersion,
} from './utils.js';

export const EmbeddingConfigSchema = z
  .object({
    /** Override the API key provided at plugin initialization. */
    apiKey: z.string().optional(),
    /**
     * The `task_type` parameter is defined as the intended downstream application to help the model
     * produce better quality embeddings.
     **/
    taskType: TaskTypeSchema.optional(),
    title: z.string().optional(),
    version: z.string().optional(),
    /**
     * The `outputDimensionality` parameter allows you to specify the dimensionality of the embedding output.
     * By default, the model generates embeddings with 768 dimensions. Models such as
     * `text-embedding-004`, `text-embedding-005`, and `text-multilingual-embedding-002`
     * allow the output dimensionality to be adjusted between 1 and 768.
     * By selecting a smaller output dimensionality, users can save memory and storage space, leading to more efficient computations.
     **/
    outputDimensionality: z.number().min(1).optional(),
  })
  .passthrough();
export type EmbeddingConfigSchemaType = typeof EmbeddingConfigSchema;
export type EmbeddingConfig = z.infer<EmbeddingConfigSchemaType>;

// This contains all the embedder config schema types
type ConfigSchemaType = EmbeddingConfigSchemaType;

function commonRef(
  name: string,
  info?: EmbedderInfo,
  configSchema: ConfigSchemaType = EmbeddingConfigSchema
): EmbedderReference<ConfigSchemaType> {
  return embedderRef({
    name: `googleai/${name}`,
    configSchema,
    info: info ?? {
      dimensions: 768,
      supports: {
        input: ['text'],
      },
    },
  });
}

const GENERIC_MODEL = commonRef('embedder');

const KNOWN_MODELS = {
  'text-embedding-004': commonRef('text-embedding-004'),
  'gemini-embedding-001': commonRef('gemini-embedding-001'),
};
export type KnownModels = keyof typeof KNOWN_MODELS; // For autocomplete

export function createModelRef(
  version: string,
  config: EmbeddingConfig = {}
): EmbedderReference<ConfigSchemaType> {
  const name = checkModelName(version);
  return embedderRef({
    name: `googleai/${name}`,
    config,
    configSchema: GENERIC_MODEL.configSchema,
    info: {
      ...GENERIC_MODEL.info,
    },
  });
}

export function listActions(models: Model[]): ActionMetadata[] {
  return (
    models
      .filter((m) => m.supportedGenerationMethods.includes('embedContent'))
      // Filter out deprecated
      .filter((m) => !m.description || !m.description.includes('deprecated'))
      .map((m) => {
        const ref = createModelRef(m.name);
        return embedderActionMetadata({
          name: ref.name,
          info: ref.info,
          configSchema: ref.configSchema,
        });
      })
  );
}

export function defineKnownModels(
  options?: GoogleAIPluginOptions
): EmbedderAction[] {
  return Object.keys(KNOWN_MODELS).map((name) => defineEmbedder(name, options));
}

export function defineEmbedder(
  name: string,
  pluginOptions?: GoogleAIPluginOptions
): EmbedderAction {
  checkApiKey(pluginOptions?.apiKey);
  const ref = createModelRef(name);

  return embedder(
    {
      name: ref.name,
      configSchema: ref.configSchema,
      info: ref.info,
    },
    async (request, _) => {
      const embedApiKey = calculateApiKey(
        pluginOptions?.apiKey,
        request.options?.apiKey
      );
      const embedVersion = request.options?.version || extractVersion(ref);

      const embeddings = await Promise.all(
        request.input.map(async (doc) => {
          const response = await embedContent(embedApiKey, embedVersion, {
            taskType: request.options?.taskType,
            title: request.options?.title,
            content: {
              role: '',
              parts: [{ text: doc.text }],
            },
            outputDimensionality: request.options?.outputDimensionality,
          } as EmbedContentRequest);
          const values = response.embedding.values;
          return { embedding: values };
        })
      );
      return { embeddings };
    }
  );
}

export const TEST_ONLY = { KNOWN_MODELS };
