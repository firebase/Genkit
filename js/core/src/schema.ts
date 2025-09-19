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

import Ajv, { type ErrorObject, type JSONSchemaType } from 'ajv';
import addFormats from 'ajv-formats';
import { z } from 'zod';
import zodToJsonSchema from 'zod-to-json-schema';
import { GenkitError } from './error.js';
import type { Registry } from './registry.js';
const ajv = new Ajv();
addFormats(ajv);

export { z }; // provide a consistent zod to use throughout genkit

/**
 * JSON schema.
 */
export type JSONSchema = JSONSchemaType<any> | any;

const jsonSchemas = new WeakMap<z.ZodTypeAny, JSONSchema>();
const validators = new WeakMap<JSONSchema, ReturnType<typeof ajv.compile>>();

/**
 * Wrapper object for various ways schema can be provided.
 */
export interface ProvidedSchema {
  jsonSchema?: JSONSchema;
  schema?: z.ZodTypeAny;
}

/**
 * Schema validation error.
 */
export class ValidationError extends GenkitError {
  constructor({
    data,
    errors,
    schema,
  }: {
    data: any;
    errors: ValidationErrorDetail[];
    schema: JSONSchema;
  }) {
    super({
      status: 'INVALID_ARGUMENT',
      message: `Schema validation failed. Parse Errors:\n\n${errors.map((e) => `- ${e.path}: ${e.message}`).join('\n')}\n\nProvided data:\n\n${JSON.stringify(data, null, 2)}\n\nRequired JSON schema:\n\n${JSON.stringify(schema, null, 2)}`,
      detail: { errors, schema },
    });
  }
}

/**
 * Converts a Zod schema into a JSON schema, utilizing an in-memory cache for known objects.
 * @param options Provide a json schema and/or zod schema. JSON schema has priority.
 * @returns A JSON schema.
 */
export function toJsonSchema({
  jsonSchema,
  schema,
}: ProvidedSchema): JSONSchema | undefined {
  // if neither jsonSchema or schema is present return undefined
  if (!jsonSchema && !schema) return null;
  if (jsonSchema) return jsonSchema;
  if (jsonSchemas.has(schema!)) return jsonSchemas.get(schema!)!;
  const outSchema = zodToJsonSchema(schema!, {
    $refStrategy: 'none',
    removeAdditionalStrategy: 'strict',
  });
  jsonSchemas.set(schema!, outSchema as JSONSchema);
  return outSchema as JSONSchema;
}

/**
 * Schema validation error details.
 */
export interface ValidationErrorDetail {
  path: string;
  message: string;
}

function toErrorDetail(error: ErrorObject): ValidationErrorDetail {
  return {
    path: error.instancePath.substring(1).replace(/\//g, '.') || '(root)',
    message: error.message!,
  };
}

/**
 * Validation response.
 */
export type ValidationResponse =
  | { valid: true; errors: never }
  | { valid: false; errors: ErrorObject[] };

/**
 * Validates the provided data against the provided schema.
 */
export function validateSchema(
  data: unknown,
  options: ProvidedSchema
): { valid: boolean; errors?: any[]; schema: JSONSchema } {
  const toValidate = toJsonSchema(options);
  if (!toValidate) {
    return { valid: true, schema: toValidate };
  }
  const validator = validators.get(toValidate) || ajv.compile(toValidate);
  const valid = validator(data) as boolean;
  const errors = validator.errors?.map((e) => e);
  return { valid, errors: errors?.map(toErrorDetail), schema: toValidate };
}

/**
 * Parses raw data object against the provided schema.
 */
export function parseSchema<T = unknown>(
  data: unknown,
  options: ProvidedSchema
): T {
  const { valid, errors, schema } = validateSchema(data, options);
  if (!valid) {
    throw new ValidationError({ data, errors: errors!, schema });
  }
  return data as T;
}

/**
 * Registers provided schema as a named schema object in the Genkit registry.
 *
 * @hidden
 */
export function defineSchema<T extends z.ZodTypeAny>(
  registry: Registry,
  name: string,
  schema: T
): T {
  registry.registerSchema(name, { schema });
  return schema;
}

/**
 * Registers provided JSON schema as a named schema object in the Genkit registry.
 *
 * @hidden
 */
export function defineJsonSchema(
  registry: Registry,
  name: string,
  jsonSchema: JSONSchema
) {
  registry.registerSchema(name, { jsonSchema });
  return jsonSchema;
}
