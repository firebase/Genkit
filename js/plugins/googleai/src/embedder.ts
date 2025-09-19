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
  GoogleGenerativeAI,
  type EmbedContentRequest,
} from '@google/generative-ai';
import {
  GenkitError,
  z,
  type EmbedderAction,
  type EmbedderReference,
} from 'genkit';
import { embedderRef } from 'genkit/embedder';
import { embedder } from 'genkit/plugin';
import { getApiKeyFromEnvVar } from './common.js';
import type { PluginOptions } from './index.js';

export const TaskTypeSchema = z.enum([
  'RETRIEVAL_DOCUMENT',
  'RETRIEVAL_QUERY',
  'SEMANTIC_SIMILARITY',
  'CLASSIFICATION',
  'CLUSTERING',
]);
export type TaskType = z.infer<typeof TaskTypeSchema>;

export const GeminiEmbeddingConfigSchema = z.object({
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
  outputDimensionality: z.number().min(1).max(768).optional(),
});

export type GeminiEmbeddingConfig = z.infer<typeof GeminiEmbeddingConfigSchema>;

export const textEmbeddingGecko001 = embedderRef({
  name: 'googleai/embedding-001',
  configSchema: GeminiEmbeddingConfigSchema,
  info: {
    dimensions: 768,
    label: 'Google Gen AI - Text Embedding Gecko (Legacy)',
    supports: {
      input: ['text'],
    },
  },
});

export const textEmbedding004 = embedderRef({
  name: 'googleai/text-embedding-004',
  configSchema: GeminiEmbeddingConfigSchema,
  info: {
    dimensions: 768,
    label: 'Google Gen AI - Text Embedding 001',
    supports: {
      input: ['text'],
    },
  },
});

export const geminiEmbedding001 = embedderRef({
  name: 'googleai/gemini-embedding-001',
  configSchema: GeminiEmbeddingConfigSchema,
  info: {
    dimensions: 768,
    label: 'Google Gen AI - Gemini Embedding 001',
    supports: {
      input: ['text'],
    },
  },
});

export const SUPPORTED_MODELS = {
  'embedding-001': textEmbeddingGecko001,
  'text-embedding-004': textEmbedding004,
  'gemini-embedding-001': geminiEmbedding001,
};

export function defineGoogleAIEmbedder(
  name: string,
  pluginOptions: PluginOptions
): EmbedderAction<any> {
  let apiKey: string | undefined;
  // DO NOT throw if {apiKey: false} was supplied to options.
  if (pluginOptions.apiKey !== false) {
    apiKey = pluginOptions?.apiKey || getApiKeyFromEnvVar();
    if (!apiKey)
      throw new Error(
        'Please pass in the API key or set either GEMINI_API_KEY or GOOGLE_API_KEY environment variable.\n' +
          'For more details see https://genkit.dev/docs/plugins/google-genai'
      );
  }
  // In v2, plugin internals use UNPREFIXED action names.
  const actionName = name;

  const embedderReference: EmbedderReference =
    SUPPORTED_MODELS[actionName] ??
    embedderRef({
      name: actionName,
      configSchema: GeminiEmbeddingConfigSchema,
      info: {
        dimensions: 768,
        label: `Google AI - ${actionName}`,
        supports: {
          input: ['text', 'image', 'video'],
        },
      },
    });
  return embedder(
    {
      name: actionName,
      configSchema: GeminiEmbeddingConfigSchema,
      info: embedderReference.info!,
    },
    async ({ input, options }) => {
      if (pluginOptions.apiKey === false && !options?.apiKey) {
        throw new GenkitError({
          status: 'INVALID_ARGUMENT',
          message:
            'GoogleAI plugin was initialized with {apiKey: false} but no apiKey configuration was passed at call time.',
        });
      }
      const client = new GoogleGenerativeAI(
        options?.apiKey || apiKey!
      ).getGenerativeModel({
        model:
          options?.version ||
          embedderReference.config?.version ||
          embedderReference.version ||
          actionName,
      });
      const embeddings = await Promise.all(
        input.map(async (doc) => {
          const response = await client.embedContent({
            taskType: options?.taskType,
            title: options?.title,
            content: {
              role: '',
              parts: [{ text: doc.text }],
            },
            outputDimensionality: options?.outputDimensionality,
          } as EmbedContentRequest);
          const values = response.embedding.values;
          return { embedding: values };
        })
      );
      return { embeddings };
    }
  );
}
