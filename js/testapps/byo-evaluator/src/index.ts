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
import { EvaluatorAction } from '@genkit-ai/ai';
import { ModelReference } from '@genkit-ai/ai/model';
import { configureGenkit, genkitPlugin, PluginProvider } from '@genkit-ai/core';
import { dotprompt } from '@genkit-ai/dotprompt';
import { firebase } from '@genkit-ai/firebase';
import { geminiPro, googleAI } from '@genkit-ai/googleai';
import * as z from 'zod';
import {
  PERMISSIVE_SAFETY_SETTINGS,
  URL_REGEX,
  US_PHONE_REGEX,
} from './constants.js';
import {
  createDeliciousnessEvaluator,
  DELICIOUSNESS,
} from './deliciousness/deliciousness_evaluator.js';
import {
  createFunninessEvaluator,
  FUNNINESS,
} from './funniness/funniness_evaluator.js';
import { createPiiEvaluator, PII_DETECTION } from './pii/pii_evaluator.js';
import {
  createRegexEvaluators,
  isRegexMetric,
  regexMatcher,
  RegexMetric,
} from './regex/regex_evaluator.js';

configureGenkit({
  plugins: [
    dotprompt(),
    firebase(),
    googleAI({ apiVersion: ['v1', 'v1beta'] }),
    byoEval({
      judge: geminiPro,
      judgeConfig: PERMISSIVE_SAFETY_SETTINGS,
      metrics: [
        // regexMatcher will register an evaluator with a name in the format
        // byo/regex_match_{suffix}. In this case, byo/regex_match_url
        regexMatcher('url', URL_REGEX),
        // byo/regex_match_us_phone
        regexMatcher('us_phone', US_PHONE_REGEX),
        PII_DETECTION,
        DELICIOUSNESS,
        FUNNINESS,
      ],
    }),
  ],
  flowStateStore: 'firebase',
  traceStore: 'firebase',
  enableTracingAndMetrics: true,
  logLevel: 'debug',
});

/**
 * Generic metric definition with flexible configuration.
 */
export interface ByoMetric {
  name: string;
}

/**
 * Plugin option definition for the PII evaluator
 */
export interface PluginOptions<ModelCustomOptions extends z.ZodTypeAny> {
  judge: ModelReference<ModelCustomOptions>;
  judgeConfig?: z.infer<ModelCustomOptions>;
  metrics?: Array<ByoMetric>;
}

/**
 * Configurable Eval plugin that provides different kinds of custom evaluators.
 */
export function byoEval<ModelCustomOptions extends z.ZodTypeAny>(
  params: PluginOptions<ModelCustomOptions>
): PluginProvider {
  // Define the new plugin
  const plugin = genkitPlugin(
    'byo',
    async (params: PluginOptions<ModelCustomOptions>) => {
      const { judge, judgeConfig, metrics } = params;
      if (!metrics) {
        throw new Error(`Found no configured metrics.`);
      }
      const regexMetrics = metrics?.filter((metric) => isRegexMetric(metric));
      const hasPiiMetric = metrics?.includes(PII_DETECTION);
      const hasFunninessMetric = metrics?.includes(FUNNINESS);
      const hasDelicousnessMetric = metrics?.includes(DELICIOUSNESS);

      let evaluators: EvaluatorAction[] = [];

      if (regexMetrics) {
        evaluators = [...createRegexEvaluators(regexMetrics as RegexMetric[])];
      }

      if (hasPiiMetric) {
        evaluators.push(createPiiEvaluator(judge, judgeConfig));
      }

      if (hasFunninessMetric) {
        evaluators.push(createFunninessEvaluator(judge, judgeConfig));
      }

      if (hasDelicousnessMetric) {
        evaluators.push(createDeliciousnessEvaluator(judge, judgeConfig));
      }

      return { evaluators };
    }
  );

  // create the plugin with the passed params
  return plugin(params);
}

export default byoEval;
