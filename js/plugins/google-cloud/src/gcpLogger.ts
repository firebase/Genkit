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

import { LoggerConfig } from '@genkit-ai/core';
import { LoggingWinston } from '@google-cloud/logging-winston';
import { PluginOptions } from './index.js';

/**
 * Provides a {LoggerConfig} for exporting Genkit debug logs to GCP Cloud
 * logs.
 */
export class GcpLogger implements LoggerConfig {
  private readonly options: PluginOptions;

  constructor(options?: PluginOptions) {
    this.options = options || {};
  }

  async getLogger(env: string) {
    // Dynamically importing winston here more strictly controls
    // the import order relative to registering instrumentation
    // with OpenTelemetry. Incorrect import order will trigger
    // an internal OT warning and will result in logs not being
    // associated with correct spans/traces.
    const winston = await import('winston');
    const format = this.shouldExport(env)
      ? { format: winston.format.json() }
      : {
          format: winston.format.printf((info): string => {
            return `[${info.level}] ${info.message}`;
          }),
        };

    const transport = this.shouldExport(env)
      ? new LoggingWinston({
          projectId: this.options.projectId,
          labels: { module: 'genkit' },
          prefix: 'genkit',
          logName: 'genkit_log',
        })
      : new winston.transports.Console();

    return winston.createLogger({
      transports: [transport],
      ...format,
    });
  }

  private shouldExport(env: string) {
    return this.options.forceDevExport || env !== 'dev';
  }
}
