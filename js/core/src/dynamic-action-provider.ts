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

import type * as z from 'zod';
import { Action, ActionMetadata, defineAction } from './action.js';
import { ActionType, Registry } from './registry.js';

type DapValue = {
  [K in ActionType]?: Action<z.ZodTypeAny, z.ZodTypeAny, z.ZodTypeAny>[];
};

class SimpleCache {
  private value: DapValue | undefined;
  private expiresAt: number | undefined;
  private ttlMillis: number;
  private dap: DapAction;
  private dapFn: DapFn;

  constructor(dap: DapAction, config: DapConfig, dapFn: DapFn) {
    this.dap = dap;
    this.dapFn = dapFn;
    this.ttlMillis = config.cacheConfig?.ttlMillis || 3 * 1000;
  }

  async getOrFetch(): Promise<DapValue> {
    if (
      !this.value ||
      !this.expiresAt ||
      this.ttlMillis < 0 ||
      Date.now() > this.expiresAt
    ) {
      // Get a new value
      this.value = await this.dapFn(); // this returns the actual actions
      this.expiresAt = Date.now() + this.ttlMillis;

      // Also run the action
      this.dap.run(this.value); // This returns metadata and shows up in dev UI
    }
    return this.value;
  }
}

export interface DynamicRegistry {
  __cache: SimpleCache;
  getAction(
    actionType: string,
    actionName: string
  ): Promise<Action<z.ZodTypeAny, z.ZodTypeAny, z.ZodTypeAny>>;
  listActions(
    actionType: string,
    actionName: string
  ): Promise<Action<z.ZodTypeAny, z.ZodTypeAny, z.ZodTypeAny>[]>;
}

export type DapAction = Action<z.ZodTypeAny, z.ZodTypeAny, z.ZodTypeAny> &
  DynamicRegistry & {
    __action: {
      metadata: {
        type: 'dynamic-action-provider';
      };
    };
  };

export function isDynamicActionProvider(
  obj: Action<z.ZodTypeAny, z.ZodTypeAny>
): obj is DapAction {
  return obj.__action?.metadata?.type == 'dynamic-action-provider';
}

export interface DapConfig {
  name: string;
  description?: string;
  cacheConfig?: {
    // Negative = no caching
    // Zero or undefined = default (3000 milliseconds)
    // Positive number = how many milliseconds the cache is valid for
    ttlMillis: number;
  };
  metadata?: Record<string, any>;
}

export type DapFn = () => Promise<DapValue>;
export type DapMetadata = {
  [K in ActionType]?: ActionMetadata[];
};

function transformDapValue(value: DapValue): DapMetadata {
  const metadata: DapMetadata = {};
  for (const key of Object.keys(value)) {
    metadata[key] = value[key].map((a) => {
      return a.__action;
    });
  }
  return metadata;
}

export function defineDynamicActionProvider(
  registry: Registry,
  config: DapConfig | string,
  fn: DapFn
): DapAction {
  let cfg: DapConfig;
  if (typeof config == 'string') {
    cfg = { name: config };
  } else {
    cfg = { ...config };
  }
  const a = defineAction(
    registry,
    {
      ...cfg,
      actionType: 'dynamic-action-provider',
      metadata: { ...(cfg.metadata || {}), type: 'dynamic-action-provider' },
    },
    async (i, _options) => {
      // The actions are retrieved and saved in a cache and then passed in here.
      // We run this action to return the metadata for the actions only.
      // We pass the actions in here to prevent duplicate calls to the mcp
      // and also so we are guaranteed the same actions since there is only a
      // single call to mcp client/host.
      return transformDapValue(i);
    }
  );
  implementDap(a as DapAction, cfg, fn);
  return a as DapAction;
}

function implementDap(dap: DapAction, config: DapConfig, dapFn: DapFn) {
  dap.__cache = new SimpleCache(dap, config, dapFn);

  dap.getAction = async (actionType: string, actionName: string) => {
    let result = await dap.__cache.getOrFetch();
    if (result[actionType]) {
      const action = result[actionType].find(
        (t) => t.__action.name == actionName
      );
      return action;
    }
    return undefined;
  };

  dap.listActions = async (actionType: string, actionName: string) => {
    let result = await dap.__cache.getOrFetch();
    if (result[actionType] && actionName == '*') {
      return result[actionType];
    }
    const action = result[actionType].find(
      (t) => t.__action.name == actionName
    );
    if (action) {
      return [action];
    }
    return [];
  };
}
