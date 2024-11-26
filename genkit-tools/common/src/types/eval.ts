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

import { z } from 'zod';
import { CustomAnySchema, JSONSchema7Schema } from './action';
import {
  CreateDatasetRequest,
  ListEvalKeysRequest,
  ListEvalKeysResponse,
  UpdateDatasetRequest,
} from './apis';

/**
 * This file defines schema and types that are used by the Eval store.
 */

/**
 * Structured input for inference part of evaluation
 */
export const EvalInferenceStructuredInputSchema = z.object({
  samples: z.array(
    z.object({
      testCaseId: z.string().optional(),
      input: z.any(),
      reference: z.any().optional(),
    })
  ),
});
export type EvalInferenceStructuredInput = z.infer<
  typeof EvalInferenceStructuredInputSchema
>;

/**
 * A set of samples that is ready for inference.
 *
 * This should be used in user-facing surfaces (CLI/API inputs) to accommodate various user input formats. For internal wire-transfer/storage, prefer {@link Dataset}.
 */
export const EvalInferenceInputSchema = z.union([
  z.array(z.any()),
  EvalInferenceStructuredInputSchema,
]);
export type EvalInferenceInput = z.infer<typeof EvalInferenceInputSchema>;

/**
 * Represents a Dataset, to be used for bulk-inference / evaluation. This is a more optionated form of EvalInferenceInput, which guarantees testCaseId for each test sample.
 */
export const DatasetSchema = z.array(
  z.object({
    testCaseId: z.string(),
    input: z.any(),
    reference: z.any().optional(),
  })
);
export type Dataset = z.infer<typeof DatasetSchema>;

/**
 * A record that is ready for evaluation.
 *
 * TODO: consider renaming.
 */
export const EvalInputSchema = z.object({
  testCaseId: z.string(),
  input: z.any(),
  output: z.any(),
  error: z.string().optional(),
  context: z.array(z.string()).optional(),
  reference: z.any().optional(),
  traceIds: z.array(z.string()),
});
export type EvalInput = z.infer<typeof EvalInputSchema>;

export const EvalMetricSchema = z.object({
  evaluator: z.string(),
  scoreId: z.string().optional(),
  score: z.union([z.number(), z.string(), z.boolean()]).optional(),
  rationale: z.string().optional(),
  error: z.string().optional(),
  traceId: z.string().optional(),
  spanId: z.string().optional(),
});
export type EvalMetric = z.infer<typeof EvalMetricSchema>;

/**
 * The scored result of an evaluation of an individual inference.
 *
 * This should be an extension of the input with the evaluation metrics.
 */
export const EvalResultSchema = EvalInputSchema.extend({
  metrics: z.array(EvalMetricSchema).optional(),
});
export type EvalResult = z.infer<typeof EvalResultSchema>;

/**
 * A unique identifier for an Evaluation Run.
 */
export const EvalRunKeySchema = z.object({
  actionRef: z.string().optional(),
  datasetId: z.string().optional(),
  datasetVersion: z.number().optional(),
  evalRunId: z.string(),
  createdAt: z.string(),
});
export type EvalRunKey = z.infer<typeof EvalRunKeySchema>;
export const EvalKeyAugmentsSchema = EvalRunKeySchema.pick({
  datasetId: true,
  datasetVersion: true,
  actionRef: true,
});
export type EvalKeyAugments = z.infer<typeof EvalKeyAugmentsSchema>;

/**
 * A container for the results of evaluation over a batch of test cases.
 */
export const EvalRunSchema = z.object({
  key: EvalRunKeySchema,
  results: z.array(EvalResultSchema),
  metricsMetadata: z
    .record(
      z.string(),
      z.object({
        displayName: z.string(),
        definition: z.string(),
      })
    )
    .optional(),
});
export type EvalRun = z.infer<typeof EvalRunSchema>;

/**
 * Eval store persistence interface.
 */
export interface EvalStore {
  /**
   * Persist an EvalRun
   * @param evalRun the result of evaluating a collection of test cases.
   */
  save(evalRun: EvalRun): Promise<void>;

  /**
   * Load a single EvalRun from storage
   * @param evalRunId the ID of the EvalRun
   */
  load(evalRunId: string): Promise<EvalRun | undefined>;

  /**
   * List the keys of all EvalRuns from storage
   * @param query (optional) filter criteria for the result list
   */
  list(query?: ListEvalKeysRequest): Promise<ListEvalKeysResponse>;
}

export const DatasetSchemaSchema = z
  .record(z.string(), CustomAnySchema)
  .describe('Schema objects related to the dataset.')
  .nullish();

/**
 * Metadata for Dataset objects containing version, create and update time, etc.
 */
export const DatasetMetadataSchema = z.object({
  /** unique. user-provided or auto-generated */
  datasetId: z.string(),
  size: z.number(),
  schema: DatasetSchemaSchema.optional(),
  targetAction: z.string().optional(),
  /** 1 for v1, 2 for v2, etc */
  version: z.number(),
  createTime: z.string(),
  updateTime: z.string(),
});
export type DatasetMetadata = z.infer<typeof DatasetMetadataSchema>;

/**
 * Eval dataset store persistence interface.
 */
export interface DatasetStore {
  /**
   * Create new dataset with the given data
   * @param req create requeest with the data
   * @returns dataset metadata
   */
  createDataset(req: CreateDatasetRequest): Promise<DatasetMetadata>;

  /**
   * Update dataset
   * @param req update requeest with new data
   * @returns updated dataset metadata
   */
  updateDataset(req: UpdateDatasetRequest): Promise<DatasetMetadata>;

  /**
   * Get existing dataset
   * @param datasetId the ID of the dataset
   * @returns dataset ready for inference
   */
  getDataset(datasetId: string): Promise<Dataset>;

  /**
   * List all existing datasets
   * @returns array of dataset metadata objects
   */
  listDatasets(): Promise<DatasetMetadata[]>;

  /**
   * Delete existing dataset
   * @param datasetId the ID of the dataset
   */
  deleteDataset(datasetId: string): Promise<void>;
}
