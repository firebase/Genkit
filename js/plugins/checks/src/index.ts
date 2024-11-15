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

import { Genkit } from 'genkit';
import { logger } from 'genkit/logging';
import { GenkitPlugin, genkitPlugin } from 'genkit/plugin';
import { GoogleAuth, GoogleAuthOptions } from 'google-auth-library';
import {
  ChecksEvaluationMetric,
  ChecksEvaluationMetricType,
  checksEvaluators,
} from './evaluation.js';
export { ChecksEvaluationMetricType as ChecksEvaluationMetricType };

export interface PluginOptions {
  /** The Google Cloud project id to call. Must have quota for the Checks API. */
  projectId?: string;
  /** Provide custom authentication configuration for connecting to Checks API. */
  googleAuth?: GoogleAuthOptions;
  /** Configure Checks evaluators. */
  evaluation?: {
    metrics: ChecksEvaluationMetric[];
  };
}

const CLOUD_PLATFROM_OAUTH_SCOPE =
  'https://www.googleapis.com/auth/cloud-platform';

const CHECKS_OAUTH_SCOPE = 'https://www.googleapis.com/auth/checks';

/**
 * Add Google Checks evaluators.
 */
export function checks(options?: PluginOptions): GenkitPlugin {
  return genkitPlugin('checks', async (ai: Genkit) => {
    let authClient: GoogleAuth;
    let authOptions = options?.googleAuth;

    // Allow customers to pass in cloud credentials from environment variables
    // following: https://github.com/googleapis/google-auth-library-nodejs?tab=readme-ov-file#loading-credentials-from-environment-variables
    if (process.env.GCLOUD_SERVICE_ACCOUNT_CREDS) {
      const serviceAccountCreds = JSON.parse(
        process.env.GCLOUD_SERVICE_ACCOUNT_CREDS
      );
      authOptions = {
        credentials: serviceAccountCreds,
        scopes: [CLOUD_PLATFROM_OAUTH_SCOPE, CHECKS_OAUTH_SCOPE],
      };
      authClient = new GoogleAuth(authOptions);
    } else {
      authClient = new GoogleAuth(
        authOptions ?? {
          scopes: [CLOUD_PLATFROM_OAUTH_SCOPE, CHECKS_OAUTH_SCOPE],
        }
      );
    }

    const client = await authClient.getClient();
    if (client.quotaProjectId) {
      logger.warn(
        `Checks Evaluator: Your Google cloud authentication has a default quota project(${client.quotaProjectId}) associated with it which will overrid the projectId in your Checks plugin config(${options?.projectId}).`
      );
    }

    const projectId = options?.projectId || (await authClient.getProjectId());

    if (!projectId) {
      throw new Error(
        `Checks Plugin is missing the 'projectId' configuration. Please set the 'GCLOUD_PROJECT' environment variable or explicitly pass 'projectId' into genkit config.`
      );
    }

    const metrics =
      options?.evaluation && options.evaluation.metrics.length > 0
        ? options.evaluation.metrics
        : [];
    checksEvaluators(ai, authClient, metrics, projectId);
  });
}

export default checks;
