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

import { RuntimeManager } from '@genkit-ai/tools-common/manager';
import { logger } from '@genkit-ai/tools-common/utils';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { defineDocsTool } from '../mcp/docs';
import { defineFlowTools } from './flows';
import { defineTraceTools } from './trace';
import { defineUsageGuideTool } from './usage';

export async function startMcpServer(manager: RuntimeManager) {
  const server = new McpServer({
    name: 'Genkit MCP',
    version: '0.0.2',
  });

  await defineDocsTool(server);
  await defineUsageGuideTool(server);
  defineFlowTools(server, manager);
  defineTraceTools(server, manager);

  return new Promise(async (resolve) => {
    const transport = new StdioServerTransport();
    transport.onclose = () => {
      resolve(undefined);
    };
    await server.connect(transport);
    logger.info('Genkit MCP Server running on stdio');
  });
}
