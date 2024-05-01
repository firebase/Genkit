import { z } from 'zod';

// NOTE: Keep this file in sync with genkit-tools/src/types/flow.ts!
// Eventually tools will be source of truth for these types (by generating a
// JSON schema) but until then this file must be manually kept in sync

export interface TraceQuery {
  limit?: number;
  continuationToken?: string;
}
export interface TraceQueryResponse {
  traces: TraceData[];
  continuationToken?: string;
}

export interface TraceStore {
  save(traceId, trace: TraceData): Promise<void>;
  load(traceId: string): Promise<TraceData | undefined>;
  list(query?: TraceQuery): Promise<TraceQueryResponse>;
}

export const SpanMetadataSchema = z.object({
  name: z.string(),
  state: z.enum(['success', 'error']).optional(),
  input: z.any().optional(),
  output: z.any().optional(),
  isRoot: z.boolean().optional(),
  metadata: z.record(z.string(), z.string()).optional(),
});
export type SpanMetadata = z.infer<typeof SpanMetadataSchema>;

export const SpanStatusSchema = z.object({
  code: z.number(),
  message: z.string().optional(),
});

export const TimeEventSchema = z.object({
  time: z.number(),
  annotation: z.object({
    attributes: z.record(z.string(), z.any()),
    description: z.string(),
  }),
});

export const SpanContextSchema = z.object({
  traceId: z.string(),
  spanId: z.string(),
  isRemote: z.boolean().optional(),
  traceFlags: z.number(),
});

export const LinkSchema = z.object({
  context: SpanContextSchema.optional(),
  attributes: z.record(z.string(), z.any()).optional(),
  droppedAttributesCount: z.number().optional(),
});

export const InstrumentationLibrarySchema = z.object({
  name: z.string().readonly(),
  version: z.string().optional().readonly(),
  schemaUrl: z.string().optional().readonly(),
});

export const SpanDataSchema = z.object({
  spanId: z.string(),
  traceId: z.string(),
  parentSpanId: z.string().optional(),
  startTime: z.number(),
  endTime: z.number(),
  attributes: z.record(z.string(), z.any()),
  displayName: z.string(),
  links: z.array(LinkSchema).optional(),
  instrumentationLibrary: InstrumentationLibrarySchema,
  spanKind: z.string(),
  sameProcessAsParentSpan: z.object({ value: z.boolean() }).optional(),
  status: SpanStatusSchema.optional(),
  timeEvents: z
    .object({
      timeEvent: z.array(TimeEventSchema),
    })
    .optional(),
});
export type SpanData = z.infer<typeof SpanDataSchema>;

export const TraceDataSchema = z.object({
  displayName: z.string().optional(),
  startTime: z.number().optional(),
  endTime: z.number().optional(),
  spans: z.record(z.string(), SpanDataSchema),
});

export type TraceData = z.infer<typeof TraceDataSchema>;
