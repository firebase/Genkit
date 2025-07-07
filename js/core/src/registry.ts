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

import { Dotprompt } from 'dotprompt';
import { AsyncLocalStorage } from 'node:async_hooks';
import type * as z from 'zod';
import {
  runOutsideActionRuntimeContext,
  type Action,
  type ActionMetadata,
} from './action.js';
import { GenkitError } from './error.js';
import { logger } from './logging.js';
import type { PluginProvider } from './plugin.js';
import { toJsonSchema, type JSONSchema } from './schema.js';

export type AsyncProvider<T> = () => Promise<T>;

/**
 * Type of a runnable action.
 */
export type ActionType =
  | 'custom'
  | 'embedder'
  | 'evaluator'
  | 'executable-prompt'
  | 'flow'
  | 'indexer'
  | 'model'
  | 'prompt'
  | 'reranker'
  | 'retriever'
  | 'tool'
  | 'util';

/**
 * A schema is either a Zod schema or a JSON schema.
 */
export interface Schema {
  schema?: z.ZodTypeAny;
  jsonSchema?: JSONSchema;
}

function parsePluginName(registryKey: string) {
  const tokens = registryKey.split('/');
  if (tokens.length >= 4) {
    return tokens[2];
  }
  return undefined;
}

interface ParsedRegistryKey {
  actionType: ActionType;
  pluginName?: string;
  actionName: string;
}

/**
 * Parses the registry key into key parts as per the key format convention. Ex:
 *  - /model/googleai/gemini-2.0-flash
 *  - /prompt/my-plugin/folder/my-prompt
 *  - /util/generate
 */
export function parseRegistryKey(
  registryKey: string
): ParsedRegistryKey | undefined {
  const tokens = registryKey.split('/');
  if (tokens.length < 3) {
    // Invalid key format
    return undefined;
  }
  // ex: /model/googleai/gemini-2.0-flash or /prompt/my-plugin/folder/my-prompt
  if (tokens.length >= 4) {
    return {
      actionType: tokens[1] as ActionType,
      pluginName: tokens[2],
      actionName: tokens.slice(3).join('/'),
    };
  }
  // ex: /util/generate
  return {
    actionType: tokens[1] as ActionType,
    actionName: tokens[2],
  };
}

export type ActionsRecord = Record<string, Action<z.ZodTypeAny, z.ZodTypeAny>>;
export type ActionMetadataRecord = Record<string, ActionMetadata>;

/**
 * The registry is used to store and lookup actions, trace stores, flow state stores, plugins, and schemas.
 */
export class Registry {
  private actionsById: Record<
    string,
    | Action<z.ZodTypeAny, z.ZodTypeAny>
    | PromiseLike<Action<z.ZodTypeAny, z.ZodTypeAny>>
  > = {};
  private pluginsByName: Record<string, PluginProvider> = {};
  private schemasByName: Record<string, Schema> = {};
  private valueByTypeAndName: Record<string, Record<string, any>> = {};
  private allPluginsInitialized = false;
  public apiStability: 'stable' | 'beta' = 'stable';

  readonly asyncStore: AsyncStore;
  readonly dotprompt: Dotprompt;
  readonly parent?: Registry;

  constructor(parent?: Registry) {
    if (parent) {
      this.parent = parent;
      this.apiStability = parent?.apiStability;
      this.asyncStore = parent.asyncStore;
      this.dotprompt = parent.dotprompt;
    } else {
      this.asyncStore = new AsyncStore();
      this.dotprompt = new Dotprompt({
        schemaResolver: async (name) => {
          const resolvedSchema = await this.lookupSchema(name);
          if (!resolvedSchema) {
            throw new GenkitError({
              message: `Schema '${name}' not found`,
              status: 'NOT_FOUND',
            });
          }
          return toJsonSchema(resolvedSchema);
        },
      });
    }
  }

  /**
   * Creates a new registry overlaid onto the provided registry.
   * @param parent The parent registry.
   * @returns The new overlaid registry.
   */
  static withParent(parent: Registry) {
    return new Registry(parent);
  }

  /**
   * Looks up an action in the registry.
   * @param key The key of the action to lookup.
   * @returns The action.
   */
  async lookupAction<
    I extends z.ZodTypeAny,
    O extends z.ZodTypeAny,
    R extends Action<I, O>,
  >(key: string): Promise<R> {
    // We always try to initialize the plugin first.
    const parsedKey = parseRegistryKey(key);
    if (parsedKey?.pluginName && this.pluginsByName[parsedKey.pluginName]) {
      await this.initializePlugin(parsedKey.pluginName);

      // If we don't see the key in the registry, we try to resolve
      // the action with the dynamic resolver. If it exists, it will
      // register the action in the registry.
      if (!this.actionsById[key]) {
        await this.resolvePluginAction(
          parsedKey.pluginName,
          parsedKey.actionType,
          parsedKey.actionName
        );
      }
    }
    return (
      ((await this.actionsById[key]) as R) || this.parent?.lookupAction(key)
    );
  }

  /**
   * Registers an action in the registry.
   * @param type The type of the action to register.
   * @param action The action to register.
   */
  registerAction<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
    type: ActionType,
    action: Action<I, O>
  ) {
    if (type !== action.__action.actionType) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `action type (${type}) does not match type on action (${action.__action.actionType})`,
      });
    }
    const key = `/${type}/${action.__action.name}`;
    logger.debug(`registering ${key}`);
    if (this.actionsById.hasOwnProperty(key)) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `Failed to register ${key} action because it's already in the registry.`,
      });
    }
    this.actionsById[key] = action;
  }

  /**
   * Registers an action promise in the registry.
   */
  registerActionAsync<I extends z.ZodTypeAny, O extends z.ZodTypeAny>(
    type: ActionType,
    name: string,
    action: PromiseLike<Action<I, O>>
  ) {
    const key = `/${type}/${name}`;
    logger.debug(`registering ${key} (async)`);
    if (this.actionsById.hasOwnProperty(key)) {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `Failed to register ${key} action because it's already in the registry.`,
      });
    }
    this.actionsById[key] = action;
  }

  /**
   * Returns all actions that have been registered in the registry.
   * @returns All actions in the registry as a map of <key, action>.
   */
  async listActions(): Promise<ActionsRecord> {
    await this.initializeAllPlugins();
    const actions: Record<string, Action<z.ZodTypeAny, z.ZodTypeAny>> = {};
    await Promise.all(
      Object.entries(this.actionsById).map(async ([key, action]) => {
        actions[key] = await action;
      })
    );
    return {
      ...(await this.parent?.listActions()),
      ...actions,
    };
  }

  /**
   * Returns all actions that are resolvable by plugins as well as those that are already
   * in the registry.
   *
   * NOTE: this method should not be used in latency sensitive code paths.
   * It may rely on "admin" API calls such as "list models", which may cause increased cold start latency.
   *
   * @returns All resolvable action metadata as a map of <key, action metadata>.
   */
  async listResolvableActions(): Promise<ActionMetadataRecord> {
    const resolvableActions = {} as ActionMetadataRecord;
    // We listActions for all plugins in parallel.
    await Promise.all(
      Object.entries(this.pluginsByName).map(async ([pluginName, plugin]) => {
        if (plugin.listActions) {
          try {
            (await plugin.listActions()).forEach((meta) => {
              if (!meta.name) {
                throw new GenkitError({
                  status: 'INVALID_ARGUMENT',
                  message: `Invalid metadata when listing actions from ${pluginName} - name required`,
                });
              }
              if (!meta.actionType) {
                throw new GenkitError({
                  status: 'INVALID_ARGUMENT',
                  message: `Invalid metadata when listing actions from ${pluginName} - actionType required`,
                });
              }
              resolvableActions[`/${meta.actionType}/${meta.name}`] = meta;
            });
          } catch (e) {
            logger.error(`Error listing actions for ${pluginName}\n`, e);
          }
        }
      })
    );
    // Also add actions that are already registered.
    for (const [key, action] of Object.entries(await this.listActions())) {
      resolvableActions[key] = action.__action;
    }
    return {
      ...(await this.parent?.listResolvableActions()),
      ...resolvableActions,
    };
  }

  /**
   * Initializes all plugins in the registry.
   */
  async initializeAllPlugins() {
    if (this.allPluginsInitialized) {
      return;
    }
    for (const pluginName of Object.keys(this.pluginsByName)) {
      await this.initializePlugin(pluginName);
    }
    this.allPluginsInitialized = true;
  }

  /**
   * Registers a plugin provider. This plugin must be initialized before it can be used by calling {@link initializePlugin} or {@link initializeAllPlugins}.
   * @param name The name of the plugin to register.
   * @param provider The plugin provider.
   */
  registerPluginProvider(name: string, provider: PluginProvider) {
    if (this.pluginsByName[name]) {
      throw new Error(`Plugin ${name} already registered`);
    }
    this.allPluginsInitialized = false;
    let cached;
    let isInitialized = false;
    this.pluginsByName[name] = {
      name: provider.name,
      initializer: () => {
        if (!isInitialized) {
          cached = provider.initializer();
          isInitialized = true;
        }
        return cached;
      },
      resolver: async (actionType: ActionType, actionName: string) => {
        if (provider.resolver) {
          await provider.resolver(actionType, actionName);
        }
      },
      listActions: async () => {
        if (provider.listActions) {
          return await provider.listActions();
        }
        return [];
      },
    };
  }

  /**
   * Looks up a plugin.
   * @param name The name of the plugin to lookup.
   * @returns The plugin provider.
   */
  lookupPlugin(name: string): PluginProvider | undefined {
    return this.pluginsByName[name] || this.parent?.lookupPlugin(name);
  }

  /**
   * Resolves a new Action dynamically by registering it.
   * @param pluginName The name of the plugin
   * @param actionType The type of the action
   * @param actionName The name of the action
   * @returns
   */
  async resolvePluginAction(
    pluginName: string,
    actionType: ActionType,
    actionName: string
  ) {
    const plugin = this.pluginsByName[pluginName];
    if (plugin) {
      return await runOutsideActionRuntimeContext(this, async () => {
        if (plugin.resolver) {
          await plugin.resolver(actionType, actionName);
        }
      });
    }
  }

  /**
   * Initializes a plugin already registered with {@link registerPluginProvider}.
   * @param name The name of the plugin to initialize.
   * @returns The plugin.
   */
  async initializePlugin(name: string) {
    if (this.pluginsByName[name]) {
      return await runOutsideActionRuntimeContext(this, () =>
        this.pluginsByName[name].initializer()
      );
    }
  }

  /**
   * Registers a schema.
   * @param name The name of the schema to register.
   * @param data The schema to register (either a Zod schema or a JSON schema).
   */
  registerSchema(name: string, data: Schema) {
    if (this.schemasByName[name]) {
      throw new Error(`Schema ${name} already registered`);
    }
    this.schemasByName[name] = data;
  }

  registerValue(type: string, name: string, value: any) {
    if (!this.valueByTypeAndName[type]) {
      this.valueByTypeAndName[type] = {};
    }
    this.valueByTypeAndName[type][name] = value;
  }

  async lookupValue<T = unknown>(
    type: string,
    key: string
  ): Promise<T | undefined> {
    const pluginName = parsePluginName(key);
    if (!this.valueByTypeAndName[type]?.[key] && pluginName) {
      await this.initializePlugin(pluginName);
    }
    return (
      (this.valueByTypeAndName[type]?.[key] as T) ||
      this.parent?.lookupValue(type, key)
    );
  }

  async listValues<T>(type: string): Promise<Record<string, T>> {
    await this.initializeAllPlugins();
    return {
      ...((await this.parent?.listValues(type)) || {}),
      ...(this.valueByTypeAndName[type] || {}),
    } as Record<string, T>;
  }

  /**
   * Looks up a schema.
   * @param name The name of the schema to lookup.
   * @returns The schema.
   */
  lookupSchema(name: string): Schema | undefined {
    return this.schemasByName[name] || this.parent?.lookupSchema(name);
  }
}

/**
 * Manages AsyncLocalStorage instances in a single place.
 */
export class AsyncStore {
  private asls: Record<string, AsyncLocalStorage<any>> = {};

  getStore<T>(key: string): T | undefined {
    return this.asls[key]?.getStore();
  }

  run<T, R>(key: string, store: T, callback: () => R): R {
    if (!this.asls[key]) {
      this.asls[key] = new AsyncLocalStorage<T>();
    }
    return this.asls[key].run(store, callback);
  }
}

/**
 * An object that has a reference to Genkit Registry.
 */
export interface HasRegistry {
  get registry(): Registry;
}
