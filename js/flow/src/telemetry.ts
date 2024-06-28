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

import { GENKIT_VERSION } from '@genkit-ai/core';
import { logger } from '@genkit-ai/core/logging';
import {
  internalMetricNamespaceWrap,
  MetricCounter,
  MetricHistogram,
} from '@genkit-ai/core/metrics';
import {
  PathMetadata,
  spanMetadataAls,
  toDisplayPath,
  traceMetadataAls,
} from '@genkit-ai/core/tracing';
import { ValueType } from '@opentelemetry/api';
import express from 'express';

/**
 * Wraps the declared metrics in a Genkit-specific, internal namespace.
 */
const _N = internalMetricNamespaceWrap.bind(null, 'flow');

const flowCounter = new MetricCounter(_N('requests'), {
  description: 'Counts calls to genkit flows.',
  valueType: ValueType.INT,
});

const pathCounter = new MetricCounter(_N('path/requests'), {
  description: 'Tracks unique flow paths per flow.',
  valueType: ValueType.INT,
});

const pathLatencies = new MetricHistogram(_N('path/latency'), {
  description: 'Latencies per flow path.',
  ValueType: ValueType.DOUBLE,
  unit: 'ms',
});

const flowLatencies = new MetricHistogram(_N('latency'), {
  description: 'Latencies when calling Genkit flows.',
  valueType: ValueType.DOUBLE,
  unit: 'ms',
});

export function recordError(err: any) {
  const qualifiedPath = spanMetadataAls?.getStore()?.path || '';
  const path = toDisplayPath(qualifiedPath);
  logger.logStructuredError(`Error[${path}, ${err.name}]`, {
    path,
    qualifiedPath,
    name: err.name,
    message: err.message,
    stack: err.stack,
    source: 'ts',
    sourceVersion: GENKIT_VERSION,
  });
}

export function writeFlowSuccess(flowName: string, latencyMs: number) {
  const dimensions = {
    name: flowName,
    status: 'success',
    source: 'ts',
    sourceVersion: GENKIT_VERSION,
  };
  flowCounter.add(1, dimensions);
  flowLatencies.record(latencyMs, dimensions);

  const paths = traceMetadataAls.getStore()?.paths || new Set<PathMetadata>();
  if (paths) {
    const pathDimensions = {
      flowName: flowName,
      status: 'success',
      source: 'ts',
      sourceVersion: GENKIT_VERSION,
    };
    const relevantPaths = Array.from(paths).filter((meta) =>
      meta.path.includes(flowName)
    );

    logger.logStructured(`Paths[${flowName}]`, {
      flowName: flowName,
      paths: relevantPaths.map((p) => p.path),
    });

    relevantPaths.forEach((p) => {
      pathCounter.add(1, {
        ...pathDimensions,
        path: p.path,
      });

      pathLatencies.record(p.latency, {
        ...pathDimensions,
        path: p.path,
      });
    });
  }
}

export function writeFlowFailure(
  flowName: string,
  latencyMs: number,
  err: any
) {
  const dimensions = {
    name: flowName,
    status: 'failure',
    source: 'ts',
    sourceVersion: GENKIT_VERSION,
    error: err.name,
  };
  flowCounter.add(1, dimensions);
  flowLatencies.record(latencyMs, dimensions);

  const allQualifiedPaths =
    traceMetadataAls.getStore()?.paths || new Set<PathMetadata>();
  if (allQualifiedPaths) {
    const qualifiedFailPath = spanMetadataAls?.getStore()?.path || '';
    const failPath = toDisplayPath(qualifiedFailPath);
    const relevantPaths = Array.from(allQualifiedPaths).filter(
      (meta) => meta.path.includes(flowName) && meta.path !== qualifiedFailPath
    );

    logger.logStructured(`Paths[${flowName}]`, {
      flowName: flowName,
      paths: relevantPaths.map((p) => toDisplayPath(p.path)),
    });

    const pathDimensions = {
      flowName: flowName,
      source: 'ts',
      sourceVersion: GENKIT_VERSION,
    };
    // All paths that have succeeded need to be tracked as succeeded.
    relevantPaths.forEach((p) => {
      const path = toDisplayPath(p.path);
      pathCounter.add(1, {
        ...pathDimensions,
        status: 'success',
        path: p.path,
      });

      pathLatencies.record(p.latency, {
        ...pathDimensions,
        status: 'success',
        path: p.path,
      });
    });

    pathCounter.add(1, {
      ...pathDimensions,
      status: 'failure',
      error: err.name,
      path: qualifiedFailPath,
    });

    pathLatencies.record(latencyMs, {
      ...pathDimensions,
      status: 'failure',
      error: err.name,
      path: qualifiedFailPath,
    });
  }
}

export function logRequest(flowName: string, req: express.Request) {
  const qualifiedPath = spanMetadataAls?.getStore()?.path || '';
  const path = toDisplayPath(qualifiedPath);
  logger.logStructured(`Request[${flowName}]`, {
    flowName: flowName,
    headers: {
      ...req.headers,
      authorization: '<redacted>',
    },
    params: req.params,
    body: req.body,
    query: req.query,
    originalUrl: req.originalUrl,
    path,
    qualifiedPath,
    source: 'ts',
    sourceVersion: GENKIT_VERSION,
  });
}

export function logResponse(flowName: string, respCode: number, respBody: any) {
  const qualifiedPath = spanMetadataAls?.getStore()?.path || '';
  const path = toDisplayPath(qualifiedPath);
  logger.logStructured(`Response[${flowName}]`, {
    flowName: flowName,
    path,
    qualifiedPath,
    code: respCode,
    body: respBody,
    source: 'ts',
    sourceVersion: GENKIT_VERSION,
  });
}
