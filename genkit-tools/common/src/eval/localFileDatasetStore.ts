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

import crypto from 'crypto';
import fs from 'fs';
import { readFile, rm, writeFile } from 'fs/promises';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { CreateDatasetRequest, UpdateDatasetRequest } from '../types/apis';
import {
  Dataset,
  DatasetMetadata,
  DatasetStore,
  EvalFlowInputSchema,
} from '../types/eval';
import { logger } from '../utils/logger';

/**
 * A local, file-based DatasetStore implementation.
 */
export class LocalFileDatasetStore implements DatasetStore {
  private readonly storeRoot;
  private readonly indexFile;
  private readonly INDEX_DELIMITER = '\n';
  private static cachedDatasetStore: LocalFileDatasetStore | null = null;

  private constructor(storeRoot: string) {
    this.storeRoot = storeRoot;
    this.indexFile = this.getIndexFilePath();
    fs.mkdirSync(this.storeRoot, { recursive: true });
    if (!fs.existsSync(this.indexFile)) {
      fs.writeFileSync(path.resolve(this.indexFile), '');
    }
    logger.info(
      `Initialized local file dataset store at root: ${this.storeRoot}`
    );
  }

  static getDatasetStore() {
    if (!this.cachedDatasetStore) {
      this.cachedDatasetStore = new LocalFileDatasetStore(
        this.generateRootPath()
      );
    }
    return this.cachedDatasetStore;
  }

  static reset() {
    this.cachedDatasetStore = null;
  }

  async createDataset(req: CreateDatasetRequest): Promise<DatasetMetadata> {
    return this.createDatasetInternal(req.data, req.displayName);
  }

  private async createDatasetInternal(
    data: Dataset,
    displayName?: string
  ): Promise<DatasetMetadata> {
    const datasetId = this.generateDatasetId();
    const filePath = path.resolve(
      this.storeRoot,
      this.generateFileName(datasetId)
    );

    if (fs.existsSync(filePath)) {
      logger.error(`Dataset already exists at ` + filePath);
      throw new Error(
        `Create dataset failed: file already exists at {$filePath}`
      );
    }

    logger.info(`Saving Dataset to ` + filePath);
    await writeFile(filePath, JSON.stringify(data));

    const now = new Date().toString();
    const metadata = {
      datasetId,
      size: Array.isArray(data) ? data.length : data.samples.length,
      version: 1,
      displayName: displayName,
      createTime: now,
      updateTime: now,
    };

    let metadataMap = await this.getMetadataMap();
    metadataMap[datasetId] = metadata;

    logger.debug(
      `Saving DatasetMetadata for ID ${datasetId} to ` +
        path.resolve(this.indexFile)
    );

    await writeFile(path.resolve(this.indexFile), JSON.stringify(metadataMap));
    return metadata;
  }

  async updateDataset(req: UpdateDatasetRequest): Promise<DatasetMetadata> {
    const datasetId = req.datasetId;
    const filePath = path.resolve(
      this.storeRoot,
      this.generateFileName(datasetId)
    );
    if (!fs.existsSync(filePath)) {
      throw new Error(`Update dataset failed: dataset not found`);
    }

    let metadataMap = await this.getMetadataMap();
    const prevMetadata = metadataMap[datasetId];
    if (!prevMetadata) {
      throw new Error(`Update dataset failed: dataset metadata not found`);
    }

    logger.info(`Updating Dataset at ` + filePath);
    await writeFile(filePath, JSON.stringify(req.patch));

    const now = new Date().toString();
    const newMetadata = {
      datasetId: datasetId,
      size: Array.isArray(req.patch)
        ? req.patch.length
        : req.patch.samples.length,
      version: prevMetadata.version + 1,
      displayName: req.displayName,
      createTime: prevMetadata.createTime,
      updateTime: now,
    };

    logger.debug(
      `Updating DatasetMetadata for ID ${datasetId} at ` +
        path.resolve(this.indexFile)
    );
    // Replace the metadata object in the metadata map
    metadataMap[datasetId] = newMetadata;
    await writeFile(path.resolve(this.indexFile), JSON.stringify(metadataMap));

    return newMetadata;
  }

  async getDataset(datasetId: string): Promise<Dataset> {
    const filePath = path.resolve(
      this.storeRoot,
      this.generateFileName(datasetId)
    );
    if (!fs.existsSync(filePath)) {
      throw new Error(`Dataset not found for dataset ID {$id}`);
    }
    return await readFile(filePath, 'utf8').then((data) =>
      EvalFlowInputSchema.parse(JSON.parse(data))
    );
  }

  async listDatasets(): Promise<DatasetMetadata[]> {
    return this.getMetadataMap().then((metadataMap) => {
      let metadatas = [];

      for (var key in metadataMap) {
        metadatas.push(metadataMap[key]);
      }
      return metadatas;
    });
  }

  async deleteDataset(datasetId: string): Promise<void> {
    const filePath = path.resolve(
      this.storeRoot,
      this.generateFileName(datasetId)
    );
    await rm(filePath);

    let metadataMap = await this.getMetadataMap();
    delete metadataMap[datasetId];

    logger.debug(
      `Deleting DatasetMetadata for ID ${datasetId} in ` +
        path.resolve(this.indexFile)
    );
    await writeFile(path.resolve(this.indexFile), JSON.stringify(metadataMap));
  }

  private static generateRootPath(): string {
    const rootHash = crypto
      .createHash('md5')
      .update(process.cwd() || 'unknown')
      .digest('hex');
    return path.resolve(process.cwd(), `.genkit/${rootHash}/datasets`);
  }

  private generateDatasetId(): string {
    return uuidv4();
  }

  private generateFileName(datasetId: string): string {
    return `${datasetId}.json`;
  }

  private getIndexFilePath(): string {
    return path.resolve(this.storeRoot, 'index.json');
  }

  private async getMetadataMap(): Promise<any> {
    if (!fs.existsSync(this.indexFile)) {
      return Promise.resolve({} as any);
    }
    return await readFile(path.resolve(this.indexFile), 'utf8').then((data) =>
      JSON.parse(data)
    );
  }
}
