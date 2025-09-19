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
  modelActionMetadata,
  modelRef,
  ModelReference,
  z,
} from 'genkit';
import { ModelAction, ModelInfo } from 'genkit/model';
import { model } from 'genkit/plugin';
import { lyriaPredict } from './client.js';
import { fromLyriaResponse, toLyriaPredictRequest } from './converters.js';
import { ClientOptions, Model, VertexPluginOptions } from './types.js';
import { checkModelName, extractVersion } from './utils.js';

export const LyriaConfigSchema = z
  .object({
    negativePrompt: z
      .string()
      .optional()
      .describe(
        'Optional. A description of what to exclude from the generated audio.'
      ),
    seed: z
      .number()
      .optional()
      .describe(
        'Optional. A seed for deterministic generation. If provided, the model will attempt to produce the same audio given the same prompt and other parameters. Cannot be used with sample_count in the same request.'
      ),
    sampleCount: z
      .number()
      .optional()
      .describe(
        'Optional. The number of audio samples to generate. Default is 1 if not specified and seed is not used. Cannot be used with seed in the same request.'
      ),
  })
  .passthrough();
export type LyriaConfigSchemaType = typeof LyriaConfigSchema;
export type LyriaConfig = z.infer<LyriaConfigSchemaType>;

type ConfigSchemaType = LyriaConfigSchemaType;

function commonRef(
  name: string,
  info?: ModelInfo,
  configSchema: ConfigSchemaType = LyriaConfigSchema
): ModelReference<ConfigSchemaType> {
  return modelRef({
    name: `vertexai/${name}`,
    configSchema,
    info: info ?? {
      supports: {
        media: true,
        multiturn: false,
        tools: false,
        systemRole: false,
        output: ['media'],
      },
    },
  });
}

const GENERIC_MODEL = commonRef('lyria');

const KNOWN_MODELS = {
  'lyria-002': commonRef('lyria-002'),
} as const;
export type KnownModels = keyof typeof KNOWN_MODELS; // For autocorrect
export type LyriaModelName = `lyria-${string}`;
export function isLyriaModelName(value?: string): value is LyriaModelName {
  return !!value?.startsWith('lyria-');
}

export function createModelRef(
  version: string,
  config: LyriaConfig = {}
): ModelReference<ConfigSchemaType> {
  const name = checkModelName(version);
  return modelRef({
    name: `vertexai/${name}`,
    config,
    configSchema: LyriaConfigSchema,
    info: { ...GENERIC_MODEL.info },
  });
}

export function listActions(models: Model[]): ActionMetadata[] {
  return models
    .filter((m: Model) => isLyriaModelName(m.name))
    .map((m: Model) => {
      const ref = createModelRef(m.name);
      return modelActionMetadata({
        name: ref.name,
        info: ref.info,
        configSchema: ref.configSchema,
      });
    });
}

export function defineKnownModels(
  clientOptions: ClientOptions,
  pluginOptions?: VertexPluginOptions
): ModelAction[] {
  return Object.keys(KNOWN_MODELS).map((name) =>
    defineModel(name, clientOptions, pluginOptions)
  );
}

export function defineModel(
  name: string,
  clientOptions: ClientOptions,
  pluginOptions?: VertexPluginOptions
): ModelAction {
  const ref = createModelRef(name);

  return model(
    {
      name: ref.name,
      ...ref.info,
      configSchema: ref.configSchema,
    },
    async (request, { abortSignal }) => {
      const clientOpt = { ...clientOptions, signal: abortSignal };
      const lyriaPredictRequest = toLyriaPredictRequest(request);

      const response = await lyriaPredict(
        extractVersion(ref),
        lyriaPredictRequest,
        clientOpt
      );

      if (!response.predictions || response.predictions.length == 0) {
        throw new Error(
          'Model returned no predictions. Possibly due to content filters.'
        );
      }

      return fromLyriaResponse(response, request);
    }
  );
}

export const TEST_ONLY = {
  GENERIC_MODEL,
  KNOWN_MODELS,
};
