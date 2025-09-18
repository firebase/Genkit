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

import { Runtime } from '@genkit-ai/tools-common/manager';
import { logger } from '@genkit-ai/tools-common/utils';
import { existsSync, readFileSync } from 'fs';
import { lstat, mkdir, readlink, symlink, writeFile } from 'fs/promises';
import * as path from 'path';
import { AIToolConfigResult, AIToolModule, InitConfigOptions } from '../types';
import { GENKIT_PROMPT_PATH, initGenkitFile } from '../utils';

// GEMINI specific paths
const GEMINI_DIR = '.gemini';
const GEMINI_SETTINGS_PATH = path.join(GEMINI_DIR, 'settings.json');
const GENKIT_MD_SYMLINK_PATH = path.join(GEMINI_DIR, GENKIT_PROMPT_PATH);

// GENKIT specific constants
const GENKIT_MCP_CONFIG = {
  command: 'genkit',
  args: ['mcp', '--no-update-notification'],
  cwd: '.',
  timeout: 30000,
  trust: false,
  excludeTools: [
    'run_shell_command(genkit start)',
    'run_shell_command(npx genkit start)',
  ],
};

/** Configuration module for Gemini CLI */
export const gemini: AIToolModule = {
  name: 'gemini',
  displayName: 'Gemini CLI',

  /**
   * Configures the Gemini CLI extension for Genkit.
   */
  async configure(
    runtime: Runtime,
    options?: InitConfigOptions
  ): Promise<AIToolConfigResult> {
    logger.info('Installing as part of GEMINI.md');
    return await installInMdFile(runtime);
  },
};

async function installInMdFile(runtime: Runtime): Promise<AIToolConfigResult> {
  const files: AIToolConfigResult['files'] = [];
  // Part 1: Generate GENKIT.md file.

  logger.info('Installing the Genkit MCP server for Gemini CLI');
  // Handle MCP configuration - merge with existing if present
  let existingConfig: any = {};
  let settingsUpdated = false;
  try {
    const fileExists = existsSync(GEMINI_SETTINGS_PATH);
    if (fileExists) {
      existingConfig = JSON.parse(readFileSync(GEMINI_SETTINGS_PATH, 'utf-8'));
    } else {
      await mkdir(GEMINI_DIR, { recursive: true });
    }
  } catch (e) {
    // File doesn't exist or is invalid JSON, start fresh
  }

  // Check if genkit server already exists
  if (!existingConfig.mcpServers?.genkit) {
    if (!existingConfig.mcpServers) {
      existingConfig.mcpServers = {};
    }
    existingConfig.mcpServers.genkit = GENKIT_MCP_CONFIG;

    if (existingConfig.contextFileName) {
      const contextFiles = Array.isArray(existingConfig.contextFileName)
        ? [...existingConfig.contextFileName]
        : [existingConfig.contextFileName];
      if (!contextFiles.includes('GENKIT.md')) {
        contextFiles.push('GENKIT.md');
      }
      existingConfig.contextFileName = contextFiles;
    } else {
      existingConfig.contextFileName = 'GENKIT.md';
    }
    await writeFile(
      GEMINI_SETTINGS_PATH,
      JSON.stringify(existingConfig, null, 2)
    );
    settingsUpdated = true;
  }
  files.push({ path: GEMINI_SETTINGS_PATH, updated: settingsUpdated });

  // Copy GENKIT.md file
  logger.info('Copying the GENKIT.md file...');
  const baseResult = await initGenkitFile(runtime);
  files.push({ path: GENKIT_PROMPT_PATH, updated: baseResult.updated });

  logger.info('Adding a symlink for GENKIT.md in the .gemini folder');
  // Check if symlink already exists
  try {
    const stats = await lstat(GENKIT_MD_SYMLINK_PATH);
    if (stats.isSymbolicLink()) {
      // Verify the symlink points to the correct target
      const linkTarget = await readlink(GENKIT_MD_SYMLINK_PATH);
      const expectedTarget = path.relative(GEMINI_DIR, GENKIT_PROMPT_PATH);

      if (linkTarget === expectedTarget || linkTarget === GENKIT_PROMPT_PATH) {
        logger.info(
          'Symlink already exists and points to the correct location'
        );
        files.push({ path: GENKIT_MD_SYMLINK_PATH, updated: false });
      } else {
        logger.warn(
          'Symlink exists but points to wrong location. Please remove this file and try again if this was not intentional.'
        );
        files.push({ path: GENKIT_MD_SYMLINK_PATH, updated: false });
      }
    } else {
      // File exists but is not a symlink
      logger.warn(
        `${GENKIT_MD_SYMLINK_PATH} exists but is not a symlink, skipping symlink creation`
      );
      files.push({ path: GENKIT_MD_SYMLINK_PATH, updated: false });
    }
  } catch (error: any) {
    if (error.code === 'ENOENT') {
      // Symlink doesn't exist, create it
      await symlink(
        path.relative(GEMINI_DIR, GENKIT_PROMPT_PATH),
        GENKIT_MD_SYMLINK_PATH
      );
      files.push({ path: GENKIT_MD_SYMLINK_PATH, updated: true });
    } else {
      // Some other error occurred
      logger.error('Error checking symlink:', error);
      throw error;
    }
  }

  return { files };
}
