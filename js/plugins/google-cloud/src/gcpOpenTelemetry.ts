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
  MetricExporter,
  type ExporterOptions,
} from '@google-cloud/opentelemetry-cloud-monitoring-exporter';
import { TraceExporter } from '@google-cloud/opentelemetry-cloud-trace-exporter';
import { GcpDetectorSync } from '@google-cloud/opentelemetry-resource-util';
import { SpanStatusCode, TraceFlags, type Span } from '@opentelemetry/api';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
import { type ExportResult } from '@opentelemetry/core';
import type { Instrumentation } from '@opentelemetry/instrumentation';
import { PinoInstrumentation } from '@opentelemetry/instrumentation-pino';
import { WinstonInstrumentation } from '@opentelemetry/instrumentation-winston';
import { Resource } from '@opentelemetry/resources';
import {
  AggregationTemporality,
  DefaultAggregation,
  ExponentialHistogramAggregation,
  InMemoryMetricExporter,
  InstrumentType,
  PeriodicExportingMetricReader,
  type PushMetricExporter,
  type ResourceMetrics,
} from '@opentelemetry/sdk-metrics';
import type { NodeSDKConfiguration } from '@opentelemetry/sdk-node';
import {
  BatchSpanProcessor,
  InMemorySpanExporter,
  type ReadableSpan,
  type SpanExporter,
} from '@opentelemetry/sdk-trace-base';
import { GENKIT_VERSION } from 'genkit';
import { logger } from 'genkit/logging';
import { actionTelemetry } from './telemetry/action.js';
import { engagementTelemetry } from './telemetry/engagement.js';
import { featuresTelemetry } from './telemetry/feature.js';
import { generateTelemetry } from './telemetry/generate.js';
import { pathsTelemetry } from './telemetry/path.js';
import type { GcpTelemetryConfig } from './types.js';
import {
  metricsDenied,
  metricsDeniedHelpText,
  tracingDenied,
  tracingDeniedHelpText,
} from './utils.js';

let metricExporter: PushMetricExporter;
let spanProcessor: BatchSpanProcessor;
let spanExporter: AdjustingTraceExporter;

/**
 * Provides a {TelemetryConfig} for exporting OpenTelemetry data (Traces,
 * Metrics, and Logs) to the Google Cloud Operations Suite.
 */
export class GcpOpenTelemetry {
  private readonly config: GcpTelemetryConfig;
  private readonly resource: Resource;

  constructor(config: GcpTelemetryConfig) {
    this.config = config;
    this.resource = new Resource({ type: 'global' }).merge(
      new GcpDetectorSync().detect()
    );
  }

  /**
   * Log hook for writing trace and span metadata to log messages in the format
   * required by GCP.
   */
  private gcpTraceLogHook = (span: Span, record: any) => {
    const spanContext = span.spanContext();
    const isSampled = !!(spanContext.traceFlags & TraceFlags.SAMPLED);
    const projectId = this.config.projectId;

    record['logging.googleapis.com/trace'] ??=
      `projects/${projectId}/traces/${spanContext.traceId}`;
    record['logging.googleapis.com/trace_sampled'] ??= isSampled ? '1' : '0';
    record['logging.googleapis.com/spanId'] ??= spanContext.spanId;

    // Clear out the duplicate trace and span information in the log metadata.
    // These will be incorrect for logs written during span export time since
    // the logs are written after the span has fully executed. Those logs are
    // explicitly tied to the correct span in createCommonLogAttributes in
    // utils.ts.
    delete record['span_id'];
    delete record['trace_id'];
    delete record['trace_flags'];
  };

  async getConfig(): Promise<Partial<NodeSDKConfiguration>> {
    spanProcessor = new BatchSpanProcessor(await this.createSpanExporter());
    return {
      resource: this.resource,
      spanProcessor: spanProcessor,
      sampler: this.config.sampler,
      instrumentations: this.getInstrumentations(),
      metricReader: await this.createMetricReader(),
    };
  }

  private async createSpanExporter(): Promise<SpanExporter> {
    spanExporter = new AdjustingTraceExporter(
      this.shouldExportTraces()
        ? new TraceExporter({
            // provided projectId should take precedence over env vars, etc
            projectId: this.config.projectId,
            // creds for non-GCP environments, in lieu of using ADC.
            credentials: this.config.credentials,
          })
        : new InMemorySpanExporter(),
      this.config.exportInputAndOutput,
      this.config.projectId,
      getErrorHandler(
        (err) => {
          return tracingDenied(err);
        },
        await tracingDeniedHelpText()
      )
    );
    return spanExporter;
  }

  /**
   * Creates a {MetricReader} for pushing metrics out to GCP via OpenTelemetry.
   */
  private async createMetricReader(): Promise<PeriodicExportingMetricReader> {
    metricExporter = await this.buildMetricExporter();
    return new PeriodicExportingMetricReader({
      exportIntervalMillis: this.config.metricExportIntervalMillis,
      exportTimeoutMillis: this.config.metricExportTimeoutMillis,
      exporter: metricExporter,
    });
  }

  /** Gets all open telemetry instrumentations as configured by the plugin. */
  private getInstrumentations() {
    let instrumentations: Instrumentation[] = [];

    if (this.config.autoInstrumentation) {
      instrumentations = getNodeAutoInstrumentations(
        this.config.autoInstrumentationConfig
      );
    }

    return instrumentations
      .concat(this.getDefaultLoggingInstrumentations())
      .concat(this.config.instrumentations ?? []);
  }

  private shouldExportTraces(): boolean {
    return this.config.export && !this.config.disableTraces;
  }

  private shouldExportMetrics(): boolean {
    return this.config.export && !this.config.disableMetrics;
  }

  /** Always configure the Pino and Winston instrumentations */
  private getDefaultLoggingInstrumentations(): Instrumentation[] {
    return [
      new WinstonInstrumentation({ logHook: this.gcpTraceLogHook }),
      new PinoInstrumentation({ logHook: this.gcpTraceLogHook }),
    ];
  }

  private async buildMetricExporter(): Promise<PushMetricExporter> {
    const exporter: PushMetricExporter = this.shouldExportMetrics()
      ? new MetricExporterWrapper(
          {
            userAgent: {
              product: 'genkit',
              version: GENKIT_VERSION,
            },
            // provided projectId should take precedence over env vars, etc
            projectId: this.config.projectId,
            // creds for non-GCP environments, in lieu of using ADC.
            credentials: this.config.credentials,
          },
          getErrorHandler(
            (err) => {
              return metricsDenied(err);
            },
            await metricsDeniedHelpText()
          )
        )
      : new InMemoryMetricExporter(AggregationTemporality.DELTA);
    return exporter;
  }
}

/**
 * Rewrites the export method to include an error handler which logs
 * helpful information about how to set up metrics/telemetry in GCP.
 */
class MetricExporterWrapper extends MetricExporter {
  private promise = new Promise<void>((resolve) => resolve());

  constructor(
    options?: ExporterOptions,
    private errorHandler?: (error: Error) => void
  ) {
    super(options);
  }

  async export(
    metrics: ResourceMetrics,
    resultCallback: (result: ExportResult) => void
  ): Promise<void> {
    await this.promise;
    this.modifyStartTimes(metrics);
    this.promise = new Promise<void>((resolve) => {
      super.export(metrics, (result) => {
        try {
          if (this.errorHandler && result.error) {
            this.errorHandler(result.error);
          }
          resultCallback(result);
        } finally {
          resolve();
        }
      });
    });
  }

  selectAggregation(instrumentType: InstrumentType) {
    if (instrumentType === InstrumentType.HISTOGRAM) {
      return new ExponentialHistogramAggregation();
    }
    return new DefaultAggregation();
  }

  selectAggregationTemporality(instrumentType: InstrumentType) {
    return AggregationTemporality.DELTA;
  }

  /**
   * Modify the start times of each data point to ensure no
   * overlap with previous exports.
   *
   * Cloud metrics do not support delta metrics for custom metrics
   * and will convert any DELTA aggregations to CUMULATIVE ones on
   * export. There is implicit overlap in the start/end times that
   * the Metric reader is sending -- the end_time of the previous
   * export will become the start_time of the current export. The
   * overlap in times means that only one of those records will
   * persist and the other will be overwritten. This
   * method adds a thousandth of a second to ensure discrete export
   * timeframes.
   */
  private modifyStartTimes(metrics: ResourceMetrics): void {
    metrics.scopeMetrics.forEach((scopeMetric) => {
      scopeMetric.metrics.forEach((metric) => {
        metric.dataPoints.forEach((dataPoint) => {
          dataPoint.startTime[1] = dataPoint.startTime[1] + 1_000_000;
        });
      });
    });
  }

  async shutdown(): Promise<void> {
    return await this.forceFlush();
  }

  async forceFlush(): Promise<void> {
    await this.promise;
  }
}

/**
 * Adjusts spans before exporting to GCP. Redacts model input
 * and output content, and augments span attributes before sending to GCP.
 */
class AdjustingTraceExporter implements SpanExporter {
  constructor(
    private exporter: SpanExporter,
    private logInputAndOutput: boolean,
    private projectId?: string,
    private errorHandler?: (error: Error) => void
  ) {}

  export(
    spans: ReadableSpan[],
    resultCallback: (result: ExportResult) => void
  ): void {
    this.exporter?.export(this.adjust(spans), (result) => {
      if (this.errorHandler && result.error) {
        this.errorHandler(result.error);
      }
      resultCallback(result);
    });
  }

  shutdown(): Promise<void> {
    return this.exporter?.shutdown();
  }

  getExporter(): SpanExporter {
    return this.exporter;
  }

  forceFlush(): Promise<void> {
    if (this.exporter?.forceFlush) {
      return this.exporter.forceFlush();
    }
    return Promise.resolve();
  }

  private adjust(spans: ReadableSpan[]): ReadableSpan[] {
    return spans.map((span) => {
      this.tickTelemetry(span);

      span = this.redactInputOutput(span);
      span = this.markErrorSpanAsError(span);
      span = this.markFailedSpan(span);
      span = this.markGenkitFeature(span);
      span = this.markGenkitModel(span);
      span = this.normalizeLabels(span);
      return span;
    });
  }

  private tickTelemetry(span: ReadableSpan) {
    const attributes = span.attributes;
    if (!Object.keys(attributes).includes('genkit:type')) {
      return;
    }

    const type = attributes['genkit:type'] as string;
    const subtype = attributes['genkit:metadata:subtype'] as string;
    const isRoot = !!span.attributes['genkit:isRoot'];

    pathsTelemetry.tick(span, this.logInputAndOutput, this.projectId);
    if (isRoot) {
      // Report top level feature request and latency only for root spans
      // Log input to and output from to the feature
      featuresTelemetry.tick(span, this.logInputAndOutput, this.projectId);
      // Set root status explicitly
      span.attributes['genkit:rootState'] = span.attributes['genkit:state'];
    } else if (type === 'action' && subtype === 'model') {
      // Report generate metrics () for all model actions
      generateTelemetry.tick(span, this.logInputAndOutput, this.projectId);
    } else if (type === 'userEngagement') {
      // Report user acceptance and feedback metrics
      engagementTelemetry.tick(span, this.logInputAndOutput, this.projectId);
    } else {
      // Report request and latency metrics for all actions
      actionTelemetry.tick(span, this.logInputAndOutput, this.projectId);
    }
  }

  private redactInputOutput(span: ReadableSpan): ReadableSpan {
    const hasInput = 'genkit:input' in span.attributes;
    const hasOutput = 'genkit:output' in span.attributes;

    return !hasInput && !hasOutput
      ? span
      : {
          ...span,
          spanContext: span.spanContext,
          attributes: {
            ...span.attributes,
            'genkit:input': '<redacted>',
            'genkit:output': '<redacted>',
          },
        };
  }

  // This is a workaround for GCP Trace to mark a span with a red
  // exclamation mark indicating that it is an error.
  private markErrorSpanAsError(span: ReadableSpan): ReadableSpan {
    return span.status.code !== SpanStatusCode.ERROR
      ? span
      : {
          ...span,
          spanContext: span.spanContext,
          attributes: {
            ...span.attributes,
            '/http/status_code': '599',
          },
        };
  }

  private normalizeLabels(span: ReadableSpan): ReadableSpan {
    const normalized = {} as Record<string, any>;
    for (const [key, value] of Object.entries(span.attributes)) {
      normalized[key.replace(/\:/g, '/')] = value;
    }
    return {
      ...span,
      spanContext: span.spanContext,
      attributes: normalized,
    };
  }

  private markFailedSpan(span: ReadableSpan): ReadableSpan {
    if (span.attributes['genkit:isFailureSource']) {
      span.attributes['genkit:failedSpan'] = span.attributes['genkit:name'];
      span.attributes['genkit:failedPath'] = span.attributes['genkit:path'];
    }
    return span;
  }

  private markGenkitFeature(span: ReadableSpan): ReadableSpan {
    if (span.attributes['genkit:isRoot'] && !!span.attributes['genkit:name']) {
      span.attributes['genkit:feature'] = span.attributes['genkit:name'];
    }
    return span;
  }

  private markGenkitModel(span: ReadableSpan): ReadableSpan {
    if (
      span.attributes['genkit:metadata:subtype'] === 'model' &&
      !!span.attributes['genkit:name']
    ) {
      span.attributes['genkit:model'] = span.attributes['genkit:name'];
    }
    return span;
  }
}

function getErrorHandler(
  shouldLogFn: (err: Error) => boolean,
  helpText: string
): (err: Error) => void {
  // only log the first time
  let instructionsLogged = false;

  return (err) => {
    // Use the defaultLogger so that logs don't get swallowed by the open
    // telemetry exporter
    const defaultLogger = logger.defaultLogger;
    if (err && shouldLogFn(err)) {
      if (!instructionsLogged) {
        instructionsLogged = true;
        defaultLogger.error(
          `Unable to send telemetry to Google Cloud: ${err.message}\n\n${helpText}\n`
        );
      }
    } else if (err) {
      defaultLogger.error(`Unable to send telemetry to Google Cloud: ${err}`);
    }
  };
}

/** @hidden */
export function __getMetricExporterForTesting(): InMemoryMetricExporter {
  return metricExporter as InMemoryMetricExporter;
}

/** @hidden */
export function __getSpanExporterForTesting(): InMemorySpanExporter {
  return spanExporter.getExporter() as InMemorySpanExporter;
}

/** @hidden */
export function __forceFlushSpansForTesting() {
  spanProcessor.forceFlush();
}
