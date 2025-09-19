// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal/base"
)

// Model represents a model that can generate content based on a request.
type Model interface {
	// Name returns the registry name of the model.
	Name() string
	// Generate applies the [Model] to provided request, handling tool requests and handles streaming.
	Generate(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error)
	// Register registers the model with the given registry.
	Register(r api.Registry)
}

// ModelArg is the interface for model arguments. It can either be the retriever action itself or a reference to be looked up.
type ModelArg interface {
	Name() string
}

// ModelRef is a struct to hold model name and configuration.
type ModelRef struct {
	name   string
	config any
}

// ToolConfig handles configuration around tool calls during generation.
type ToolConfig struct {
	MaxTurns           int  // Maximum number of tool call iterations before erroring.
	ReturnToolRequests bool // Whether to return tool requests instead of making the tool calls and continuing the generation.
}

// ModelFunc is a streaming function that takes in a ModelRequest and generates a ModelResponse, optionally streaming ModelResponseChunks.
type ModelFunc = core.StreamingFunc[*ModelRequest, *ModelResponse, *ModelResponseChunk]

// ModelStreamCallback is a stream callback of a ModelAction.
type ModelStreamCallback = func(context.Context, *ModelResponseChunk) error

// ModelMiddleware is middleware for model generate requests that takes in a ModelFunc, does something, then returns another ModelFunc.
type ModelMiddleware = core.Middleware[*ModelRequest, *ModelResponse, *ModelResponseChunk]

// model is an action with functions specific to model generation such as Generate().
type model struct {
	core.ActionDef[*ModelRequest, *ModelResponse, *ModelResponseChunk]
}

// generateAction is the type for a utility model generation action that takes in a GenerateActionOptions instead of a ModelRequest.
type generateAction = core.ActionDef[*GenerateActionOptions, *ModelResponse, *ModelResponseChunk]

// result is a generic struct for parallel operation results with index, value, and error.
type result[T any] struct {
	index int
	value T
	err   error
}

// resumeOptionOutput is the return type for resolveResumeOption.
type resumeOptionOutput struct {
	revisedRequest      *GenerateActionOptions
	interruptedResponse *ModelResponse
	toolMessage         *Message
}

// resumedToolRequestOutput is the return type for resolveResumedToolRequest.
type resumedToolRequestOutput struct {
	toolRequest  *Part
	toolResponse *Part
	interrupt    *Part
}

// ModelOptions represents the configuration options for a model.
type ModelOptions struct {
	// ConfigSchema is the JSON schema for the model's config.
	ConfigSchema map[string]any `json:"configSchema,omitempty"`
	// Label is a user-friendly name for the model.
	Label string `json:"label,omitempty"`
	// Stage indicates the maturity stage of the model.
	Stage ModelStage `json:"stage,omitempty"`
	// Supports defines the capabilities of the model.
	Supports *ModelSupports `json:"supports,omitempty"`
	// Versions lists the available versions of the model.
	Versions []string `json:"versions,omitempty"`
}

// DefineGenerateAction defines a utility generate action.
func DefineGenerateAction(ctx context.Context, r api.Registry) *generateAction {
	return (*generateAction)(core.DefineStreamingAction(r, "generate", api.ActionTypeUtil, nil, nil,
		func(ctx context.Context, actionOpts *GenerateActionOptions, cb ModelStreamCallback) (resp *ModelResponse, err error) {
			logger.FromContext(ctx).Debug("GenerateAction",
				"input", fmt.Sprintf("%#v", actionOpts))
			defer func() {
				logger.FromContext(ctx).Debug("GenerateAction",
					"output", fmt.Sprintf("%#v", resp),
					"err", err)
			}()

			return GenerateWithRequest(ctx, r, actionOpts, nil, cb)
		}))
}

// NewModel creates a new [Model].
func NewModel(name string, opts *ModelOptions, fn ModelFunc) Model {
	if name == "" {
		panic("ai.NewModel: name is required")
	}

	if opts == nil {
		opts = &ModelOptions{
			Label: name,
		}
	}
	if opts.Supports == nil {
		opts.Supports = &ModelSupports{}
	}

	metadata := map[string]any{
		"type": api.ActionTypeModel,
		"model": map[string]any{
			"label": opts.Label,
			"supports": map[string]any{
				"media":       opts.Supports.Media,
				"context":     opts.Supports.Context,
				"multiturn":   opts.Supports.Multiturn,
				"systemRole":  opts.Supports.SystemRole,
				"tools":       opts.Supports.Tools,
				"toolChoice":  opts.Supports.ToolChoice,
				"constrained": opts.Supports.Constrained,
				"output":      opts.Supports.Output,
				"contentType": opts.Supports.ContentType,
				"longRunning": opts.Supports.LongRunning,
			},
			"versions":      opts.Versions,
			"stage":         opts.Stage,
			"customOptions": opts.ConfigSchema,
		},
	}

	inputSchema := core.InferSchemaMap(ModelRequest{})
	if inputSchema != nil && opts.ConfigSchema != nil {
		if props, ok := inputSchema["properties"].(map[string]any); ok {
			props["config"] = opts.ConfigSchema
		}
	}

	mws := []ModelMiddleware{
		simulateSystemPrompt(opts, nil),
		augmentWithContext(opts, nil),
		validateSupport(name, opts),
		addAutomaticTelemetry(),
	}
	fn = core.ChainMiddleware(mws...)(fn)

	return &model{
		ActionDef: *core.NewStreamingAction(name, api.ActionTypeModel, metadata, inputSchema, fn),
	}
}

// DefineModel creates a new [Model] and registers it.
func DefineModel(r api.Registry, name string, opts *ModelOptions, fn ModelFunc) Model {
	m := NewModel(name, opts, fn)
	m.Register(r)
	return m
}

// LookupModel looks up a [Model] registered by [DefineModel].
// It will try to resolve the model dynamically if the model is not found.
// It returns nil if the model was not resolved.
func LookupModel(r api.Registry, name string) Model {
	action := core.ResolveActionFor[*ModelRequest, *ModelResponse, *ModelResponseChunk](r, api.ActionTypeModel, name)
	if action == nil {
		return nil
	}
	return &model{
		ActionDef: *action,
	}
}

// GenerateWithRequest is the central generation implementation for ai.Generate(), prompt.Execute(), and the GenerateAction direct call.
func GenerateWithRequest(ctx context.Context, r api.Registry, opts *GenerateActionOptions, mw []ModelMiddleware, cb ModelStreamCallback) (*ModelResponse, error) {
	if opts.Model == "" {
		if defaultModel, ok := r.LookupValue(api.DefaultModelKey).(string); ok && defaultModel != "" {
			opts.Model = defaultModel
		}
		if opts.Model == "" {
			return nil, core.NewError(core.INVALID_ARGUMENT, "ai.GenerateWithRequest: model is required")
		}
	}

	m := LookupModel(r, opts.Model)

	if m == nil {
		return nil, core.NewError(core.NOT_FOUND, "ai.GenerateWithRequest: model %q not found", opts.Model)
	}

	bgAction := LookupBackgroundModel(r, opts.Model)
	resumeOutput, err := handleResumeOption(ctx, r, opts)
	if err != nil {
		return nil, err
	}

	if resumeOutput.interruptedResponse != nil {
		return nil, core.NewError(core.FAILED_PRECONDITION,
			"One or more tools triggered an interrupt during a restarted execution.")
	}

	opts = resumeOutput.revisedRequest

	if resumeOutput.toolMessage != nil && cb != nil {
		err := cb(ctx, &ModelResponseChunk{
			Content: resumeOutput.toolMessage.Content,
			Role:    RoleTool,
		})
		if err != nil {
			return nil, fmt.Errorf("streaming callback failed for resumed tool message: %w", err)
		}
	}

	toolDefMap := make(map[string]*ToolDefinition)
	for _, t := range opts.Tools {
		if _, ok := toolDefMap[t]; ok {
			return nil, core.NewError(core.INVALID_ARGUMENT, "ai.GenerateWithRequest: duplicate tool %q", t)
		}

		tool := LookupTool(r, t)
		if tool == nil {
			return nil, core.NewError(core.NOT_FOUND, "ai.GenerateWithRequest: tool %q not found", t)
		}

		toolDefMap[t] = tool.Definition()
	}
	toolDefs := make([]*ToolDefinition, 0, len(toolDefMap))
	for _, t := range toolDefMap {
		toolDefs = append(toolDefs, t)
	}

	maxTurns := opts.MaxTurns
	if maxTurns < 0 {
		return nil, core.NewError(core.INVALID_ARGUMENT, "ai.GenerateWithRequest: max turns must be greater than 0, got %d", maxTurns)
	}
	if maxTurns == 0 {
		maxTurns = 5 // Default max turns.
	}

	var outputCfg ModelOutputConfig
	var formatHandler FormatHandler

	if opts.Output != nil {
		formatter, err := resolveFormat(r, opts.Output.JsonSchema, opts.Output.Format)
		if err != nil {
			return nil, err
		}

		formatHandler, err = formatter.Handler(opts.Output.JsonSchema)
		if err != nil {
			return nil, err
		}
		outputCfg = formatHandler.Config()

		// Native constrained output is enabled only when the user has
		// requested it, the model supports it, and there's a JSON schema.
		outputCfg.Constrained = opts.Output.JsonSchema != nil &&
			opts.Output.Constrained && m.(*model).supportsConstrained(len(toolDefs) > 0)

		// Add schema instructions to prompt when not using native constraints.
		// This is a no-op for unstructured output requests.
		if !outputCfg.Constrained {
			instructions := ""
			if opts.Output.Instructions != nil {
				instructions = *opts.Output.Instructions
			} else {
				instructions = formatHandler.Instructions()
			}
			if instructions != "" {
				opts.Messages = injectInstructions(opts.Messages, instructions)
			}

			// This is optional to make the output config internally consistent.
			outputCfg.Schema = nil
		}
	}

	req := &ModelRequest{
		Messages:   opts.Messages,
		Config:     opts.Config,
		Docs:       opts.Docs,
		ToolChoice: opts.ToolChoice,
		Tools:      toolDefs,
		Output:     &outputCfg,
	}

	var fn ModelFunc
	if bgAction != nil {
		// Create a wrapper function that calls the background model but returns a ModelResponse with operation
		fn = func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			op, err := bgAction.StartOperation(ctx, req)
			if err != nil {
				return nil, err
			}

			// Return response with operation
			operationMap := map[string]any{
				"action": op.Action,
				"id":     op.ID,
				"done":   op.Done,
			}
			if op.Output != nil {
				operationMap["output"] = op.Output
			}
			if op.Error != nil {
				operationMap["error"] = map[string]any{
					"message": op.Error.Error(),
				}
			}
			if op.Metadata != nil {
				operationMap["metadata"] = op.Metadata
			}

			return &ModelResponse{
				Operation: operationMap,
				Request:   req,
			}, nil
		}
	} else {
		fn = m.Generate
	}

	fn = core.ChainMiddleware(mw...)(fn)

	// Inline recursive helper function that captures variables from parent scope.
	var generate func(context.Context, *ModelRequest, int) (*ModelResponse, error)

	generate = func(ctx context.Context, req *ModelRequest, currentTurn int) (*ModelResponse, error) {
		spanMetadata := &tracing.SpanMetadata{
			Name:    "generate",
			Type:    "util",
			Subtype: "util",
		}

		return tracing.RunInNewSpan(ctx, spanMetadata, req, func(ctx context.Context, req *ModelRequest) (*ModelResponse, error) {
			resp, err := fn(ctx, req, cb)
			if err != nil {
				return nil, err
			}

			// If this is a long-running operation response, return it immediately without further processing
			if resp.Operation != nil {
				return resp, nil
			}

			if formatHandler != nil {
				resp.Message, err = formatHandler.ParseMessage(resp.Message)
				if err != nil {
					logger.FromContext(ctx).Debug("model failed to generate output matching expected schema", "error", err.Error())
					return nil, core.NewError(core.INTERNAL, "model failed to generate output matching expected schema: %v", err)
				}
			}

			if len(resp.ToolRequests()) == 0 || opts.ReturnToolRequests {
				return resp, nil
			}

			if currentTurn+1 > maxTurns {
				return nil, core.NewError(core.ABORTED, "exceeded maximum tool call iterations (%d)", maxTurns)
			}

			newReq, interruptMsg, err := handleToolRequests(ctx, r, req, resp, cb)
			if err != nil {
				return nil, err
			}
			if interruptMsg != nil {
				resp.FinishReason = "interrupted"
				resp.FinishMessage = "One or more tool calls resulted in interrupts."
				resp.Message = interruptMsg
				return resp, nil
			}
			if newReq == nil {
				return resp, nil
			}

			return generate(ctx, newReq, currentTurn+1)
		})
	}

	return generate(ctx, req, 0)
}

// Generate generates a model response based on the provided options.
func Generate(ctx context.Context, r api.Registry, opts ...GenerateOption) (*ModelResponse, error) {
	genOpts := &generateOptions{}
	for _, opt := range opts {
		if err := opt.applyGenerate(genOpts); err != nil {
			return nil, core.NewError(core.INVALID_ARGUMENT, "ai.Generate: error applying options: %v", err)
		}
	}

	var modelName string
	if genOpts.Model != nil {
		modelName = genOpts.Model.Name()
	}

	toolNames, dynamicTools, err := resolveUniqueTools(r, genOpts.Tools)
	if err != nil {
		return nil, err
	}

	if len(dynamicTools) > 0 {
		if !r.IsChild() {
			r = r.NewChild()
		}
		for _, t := range dynamicTools {
			t.Register(r)
		}
	}

	if len(genOpts.Resources) > 0 {
		if !r.IsChild() {
			r = r.NewChild()
		}
		for _, res := range genOpts.Resources {
			res.(*resource).Register(r)
		}
	}

	messages := []*Message{}
	if genOpts.SystemFn != nil {
		system, err := genOpts.SystemFn(ctx, nil)
		if err != nil {
			return nil, err
		}

		messages = append(messages, NewSystemTextMessage(system))
	}
	if genOpts.MessagesFn != nil {
		msgs, err := genOpts.MessagesFn(ctx, nil)
		if err != nil {
			return nil, err
		}

		messages = append(messages, msgs...)
	}
	if genOpts.PromptFn != nil {
		prompt, err := genOpts.PromptFn(ctx, nil)
		if err != nil {
			return nil, err
		}

		messages = append(messages, NewUserTextMessage(prompt))
	}

	if modelRef, ok := genOpts.Model.(ModelRef); ok && genOpts.Config == nil {
		genOpts.Config = modelRef.Config()
	}

	respondParts := []*toolResponsePart{}
	for _, part := range genOpts.RespondParts {
		if !part.IsToolResponse() {
			return nil, core.NewError(core.INVALID_ARGUMENT, "ai.Generate: respond part is not a tool response")
		}

		respondParts = append(respondParts, &toolResponsePart{
			ToolResponse: part.ToolResponse,
			Metadata:     part.Metadata,
		})
	}

	restartParts := []*toolRequestPart{}
	for _, part := range genOpts.RestartParts {
		if !part.IsToolRequest() {
			return nil, core.NewError(core.INVALID_ARGUMENT, "ai.Generate: restart part is not a tool request")
		}

		restartParts = append(restartParts, &toolRequestPart{
			ToolRequest: part.ToolRequest,
			Metadata:    part.Metadata,
		})
	}

	actionOpts := &GenerateActionOptions{
		Model:              modelName,
		Messages:           messages,
		Tools:              toolNames,
		MaxTurns:           genOpts.MaxTurns,
		Config:             genOpts.Config,
		ToolChoice:         genOpts.ToolChoice,
		Docs:               genOpts.Documents,
		ReturnToolRequests: genOpts.ReturnToolRequests != nil && *genOpts.ReturnToolRequests,
		Output: &GenerateActionOutputConfig{
			JsonSchema:   genOpts.OutputSchema,
			Format:       genOpts.OutputFormat,
			Instructions: genOpts.OutputInstructions,
			Constrained:  !genOpts.CustomConstrained,
		},
	}

	if len(respondParts) > 0 || len(restartParts) > 0 {
		actionOpts.Resume = &GenerateActionResume{
			Respond: respondParts,
			Restart: restartParts,
		}
	}

	// Process resources in messages
	processedMessages, err := processResources(ctx, r, messages)
	if err != nil {
		return nil, core.NewError(core.INTERNAL, "ai.Generate: error processing resources: %v", err)
	}
	actionOpts.Messages = processedMessages

	return GenerateWithRequest(ctx, r, actionOpts, genOpts.Middleware, genOpts.Stream)
}

// GenerateText run generate request for this model. Returns generated text only.
func GenerateText(ctx context.Context, r api.Registry, opts ...GenerateOption) (string, error) {
	res, err := Generate(ctx, r, opts...)
	if err != nil {
		return "", err
	}

	return res.Text(), nil
}

// Generate run generate request for this model. Returns ModelResponse struct.
// TODO: Stream GenerateData with partial JSON
func GenerateData[Out any](ctx context.Context, r api.Registry, opts ...GenerateOption) (*Out, *ModelResponse, error) {
	var value Out
	opts = append(opts, WithOutputType(value))

	resp, err := Generate(ctx, r, opts...)
	if err != nil {
		return nil, nil, err
	}

	err = resp.Output(&value)
	if err != nil {
		return nil, nil, err
	}

	return &value, resp, nil
}

// Generate applies the [Action] to provided request.
func (m *model) Generate(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
	if m == nil {
		return nil, core.NewError(core.INVALID_ARGUMENT, "Model.Generate: generate called on a nil model; check that all models are defined")
	}

	return m.ActionDef.Run(ctx, req, cb)
}

// supportsConstrained returns whether the model supports constrained output.
func (m *model) supportsConstrained(hasTools bool) bool {
	if m == nil {
		return false
	}

	metadata := m.ActionDef.Desc().Metadata
	if metadata == nil {
		return false
	}

	modelMeta, ok := metadata["model"].(map[string]any)
	if !ok {
		return false
	}

	supportsMeta, ok := modelMeta["supports"].(map[string]any)
	if !ok {
		return false
	}

	constrained, ok := supportsMeta["constrained"].(ConstrainedSupport)
	if !ok {
		return false
	}

	if constrained == "" ||
		constrained == ConstrainedSupportNone ||
		(constrained == ConstrainedSupportNoTools && hasTools) {
		return false
	}

	return true
}

// clone creates a deep copy of the provided object using JSON marshaling and unmarshaling.
func clone[T any](obj *T) *T {
	if obj == nil {
		return nil
	}

	bytes, err := json.Marshal(obj)
	if err != nil {
		panic(fmt.Sprintf("clone: failed to marshal object: %v", err))
	}

	var newObj T
	if err := json.Unmarshal(bytes, &newObj); err != nil {
		panic(fmt.Sprintf("clone: failed to unmarshal object: %v", err))
	}

	return &newObj
}

// handleToolRequests processes any tool requests in the response, returning
// either a new request to continue the conversation or nil if no tool requests
// need handling.
func handleToolRequests(ctx context.Context, r api.Registry, req *ModelRequest, resp *ModelResponse, cb ModelStreamCallback) (*ModelRequest, *Message, error) {
	toolCount := 0
	if resp.Message != nil {
		for _, part := range resp.Message.Content {
			if part.IsToolRequest() {
				toolCount++
			}
		}
	}

	if toolCount == 0 {
		return nil, nil, nil
	}

	resultChan := make(chan result[any])
	toolMsg := &Message{Role: RoleTool}
	revisedMsg := clone(resp.Message)

	for i, part := range revisedMsg.Content {
		if !part.IsToolRequest() {
			continue
		}

		go func(idx int, p *Part) {
			toolReq := p.ToolRequest
			tool := LookupTool(r, p.ToolRequest.Name)
			if tool == nil {
				resultChan <- result[any]{idx, nil, core.NewError(core.NOT_FOUND, "tool %q not found", toolReq.Name)}
				return
			}

			output, err := tool.RunRaw(ctx, toolReq.Input)
			if err != nil {
				var tie *toolInterruptError
				if errors.As(err, &tie) {
					logger.FromContext(ctx).Debug("tool %q triggered an interrupt: %v", toolReq.Name, tie.Metadata)

					newPart := clone(p)
					if newPart.Metadata == nil {
						newPart.Metadata = make(map[string]any)
					}
					if tie.Metadata != nil {
						newPart.Metadata["interrupt"] = tie.Metadata
					} else {
						newPart.Metadata["interrupt"] = true
					}

					revisedMsg.Content[idx] = newPart

					resultChan <- result[any]{idx, nil, tie}
					return
				}

				resultChan <- result[any]{idx, nil, core.NewError(core.INTERNAL, "tool %q failed: %v", toolReq.Name, err)}
				return
			}

			newPart := clone(p)
			if newPart.Metadata == nil {
				newPart.Metadata = make(map[string]any)
			}
			newPart.Metadata["pendingOutput"] = output
			revisedMsg.Content[idx] = newPart

			resultChan <- result[any]{idx, output, nil}
		}(i, part)
	}

	var toolResps []*Part
	hasInterrupts := false
	for range toolCount {
		res := <-resultChan
		if res.err != nil {
			var tie *toolInterruptError
			if errors.As(res.err, &tie) {
				hasInterrupts = true
				continue
			}

			return nil, nil, res.err
		}

		toolReq := revisedMsg.Content[res.index].ToolRequest
		toolResps = append(toolResps, NewToolResponsePart(&ToolResponse{
			Name:   toolReq.Name,
			Ref:    toolReq.Ref,
			Output: res.value,
		}))
	}

	if hasInterrupts {
		return nil, revisedMsg, nil
	}

	toolMsg.Content = toolResps

	if cb != nil {
		err := cb(ctx, &ModelResponseChunk{
			Content: toolMsg.Content,
			Role:    RoleTool,
		})
		if err != nil {
			return nil, nil, fmt.Errorf("streaming callback failed: %w", err)
		}
	}

	newReq := req
	newReq.Messages = append(slices.Clone(req.Messages), resp.Message, toolMsg)

	return newReq, nil, nil
}

// Text returns the contents of the first candidate in a
// [ModelResponse] as a string. It returns an empty string if there
// are no candidates or if the candidate has no message.
func (gr *ModelResponse) Text() string {
	if gr.Message == nil {
		return ""
	}
	return gr.Message.Text()
}

// History returns messages from the request combined with the response message
// to represent the conversation history.
func (mr *ModelResponse) History() []*Message {
	if mr.Message == nil {
		return mr.Request.Messages
	}
	return append(mr.Request.Messages, mr.Message)
}

// Reasoning concatenates all reasoning parts present in the message
func (mr *ModelResponse) Reasoning() string {
	var sb strings.Builder
	if mr.Message == nil {
		return ""
	}

	for _, p := range mr.Message.Content {
		if !p.IsReasoning() {
			continue
		}
		sb.WriteString(p.Text)
	}
	return sb.String()
}

// Output unmarshals structured JSON output into the provided
// struct pointer.
func (mr *ModelResponse) Output(v any) error {
	j := base.ExtractJSONFromMarkdown(mr.Text())
	if j == "" {
		return errors.New("unable to parse JSON from response text")
	}
	return json.Unmarshal([]byte(j), v)
}

// ToolRequests returns the tool requests from the response.
func (mr *ModelResponse) ToolRequests() []*ToolRequest {
	toolReqs := []*ToolRequest{}
	if mr.Message == nil {
		return toolReqs
	}
	for _, part := range mr.Message.Content {
		if part.IsToolRequest() {
			toolReqs = append(toolReqs, part.ToolRequest)
		}
	}
	return toolReqs
}

// Interrupts returns the interrupted tool request parts from the response.
func (mr *ModelResponse) Interrupts() []*Part {
	parts := []*Part{}
	if mr.Message == nil {
		return parts
	}
	for _, part := range mr.Message.Content {
		if part.IsInterrupt() {
			parts = append(parts, part)
		}
	}
	return parts
}

// Text returns the text content of the [ModelResponseChunk]
// as a string. It returns an error if there is no Content
// in the response chunk.
func (c *ModelResponseChunk) Text() string {
	if len(c.Content) == 0 {
		return ""
	}
	if len(c.Content) == 1 {
		return c.Content[0].Text
	}
	var sb strings.Builder
	for _, p := range c.Content {
		if p.IsText() || p.IsData() {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

// Text returns the contents of a [Message] as a string. It
// returns an empty string if the message has no content.
// If you want to get reasoning from the message, use Reasoning() instead.
func (m *Message) Text() string {
	if m == nil {
		return ""
	}
	if len(m.Content) == 0 {
		return ""
	}
	if len(m.Content) == 1 {
		return m.Content[0].Text
	}
	var sb strings.Builder
	for _, p := range m.Content {
		if p.IsText() || p.IsData() {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

// NewModelRef creates a new ModelRef with the given name and configuration.
func NewModelRef(name string, config any) ModelRef {
	return ModelRef{name: name, config: config}
}

// Name returns the name of the model.
func (m ModelRef) Name() string {
	return m.name
}

// Config returns the configuration to use by default for this model.
func (m ModelRef) Config() any {
	return m.config
}

// handleResumedToolRequest resolves a tool request from a previous, interrupted model turn,
// when generation is being resumed. It determines the outcome of the tool request based on
// pending output, or explicit 'respond' or 'restart' directives in the resume options.
func handleResumedToolRequest(ctx context.Context, r api.Registry, genOpts *GenerateActionOptions, p *Part) (*resumedToolRequestOutput, error) {
	if p == nil || !p.IsToolRequest() {
		return nil, core.NewError(core.INVALID_ARGUMENT, "handleResumedToolRequest: part is not a tool request")
	}

	if pendingOutputVal, ok := p.Metadata["pendingOutput"]; ok {
		newReqPart := clone(p)
		delete(newReqPart.Metadata, "pendingOutput")

		newRespPart := NewResponseForToolRequest(p, pendingOutputVal)
		newRespPart.Metadata = map[string]any{"source": "pending"}

		return &resumedToolRequestOutput{
			toolRequest:  newReqPart,
			toolResponse: newRespPart,
		}, nil
	}

	if genOpts.Resume != nil {
		toolReq := p.ToolRequest

		for _, respondPart := range genOpts.Resume.Respond {
			if respondPart.ToolResponse != nil &&
				respondPart.ToolResponse.Name == toolReq.Name &&
				respondPart.ToolResponse.Ref == toolReq.Ref {
				newToolReq := clone(p)
				if interruptVal, ok := newToolReq.Metadata["interrupt"]; ok {
					delete(newToolReq.Metadata, "interrupt")
					newToolReq.Metadata["resolvedInterrupt"] = interruptVal
				}

				tool := LookupTool(r, toolReq.Name)
				if tool == nil {
					return nil, core.NewError(core.NOT_FOUND, "handleResumedToolRequest: tool %q not found", toolReq.Name)
				}

				toolDef := tool.Definition()
				if len(toolDef.OutputSchema) > 0 {
					outputBytes, err := json.Marshal(respondPart.ToolResponse.Output)
					if err != nil {
						return nil, core.NewError(core.INVALID_ARGUMENT, "handleResumedToolRequest: failed to marshal tool output for validation: %v", err)
					}

					schemaBytes, err := json.Marshal(toolDef.OutputSchema)
					if err != nil {
						return nil, core.NewError(core.INTERNAL, "handleResumedToolRequest: tool %q has invalid output schema: %v", toolReq.Name, err)
					}

					if err := base.ValidateRaw(outputBytes, schemaBytes); err != nil {
						return nil, core.NewError(core.INVALID_ARGUMENT, "handleResumedToolRequest: tool %q output validation failed: %v", toolReq.Name, err)
					}
				}

				newToolResp := NewToolResponsePart(respondPart.ToolResponse)
				newToolResp.Metadata = respondPart.Metadata

				return &resumedToolRequestOutput{
					toolRequest:  newToolReq,
					toolResponse: newToolResp,
				}, nil
			}
		}

		for _, restartPart := range genOpts.Resume.Restart {
			if restartPart.ToolRequest != nil &&
				restartPart.ToolRequest.Name == toolReq.Name &&
				restartPart.ToolRequest.Ref == toolReq.Ref {
				tool := LookupTool(r, restartPart.ToolRequest.Name)
				if tool == nil {
					return nil, core.NewError(core.NOT_FOUND, "handleResumedToolRequest: tool %q not found", restartPart.ToolRequest.Name)
				}

				resumedCtx := ctx
				if resumedVal, ok := restartPart.Metadata["resumed"]; ok {
					// TODO: Better handling here or in tools.go.
					switch resumedVal := resumedVal.(type) {
					case map[string]any:
						resumedCtx = resumedCtxKey.NewContext(resumedCtx, resumedVal)
					case bool:
						if resumedVal {
							resumedCtx = resumedCtxKey.NewContext(resumedCtx, map[string]any{})
						}
					}
				}
				if originalInputVal, ok := restartPart.Metadata["originalInput"]; ok {
					resumedCtx = origInputCtxKey.NewContext(resumedCtx, originalInputVal)
				}

				output, err := tool.RunRaw(resumedCtx, restartPart.ToolRequest.Input)
				if err != nil {
					var tie *toolInterruptError
					if errors.As(err, &tie) {
						logger.FromContext(ctx).Debug("tool %q triggered an interrupt: %v", restartPart.ToolRequest.Name, tie.Metadata)

						interruptPart := clone(p)
						if interruptPart.Metadata == nil {
							interruptPart.Metadata = make(map[string]any)
						}
						interruptPart.Metadata["interrupt"] = tie.Metadata

						return &resumedToolRequestOutput{
							interrupt: interruptPart,
						}, nil
					}

					return nil, core.NewError(core.INTERNAL, "tool %q failed: %v", restartPart.ToolRequest.Name, err)
				}

				newToolReq := clone(p)
				if interruptVal, ok := newToolReq.Metadata["interrupt"]; ok {
					delete(newToolReq.Metadata, "interrupt")
					newToolReq.Metadata["resolvedInterrupt"] = interruptVal
				}

				newToolResp := NewToolResponsePart(&ToolResponse{
					Name:   restartPart.ToolRequest.Name,
					Ref:    restartPart.ToolRequest.Ref,
					Output: output,
				})

				return &resumedToolRequestOutput{
					toolRequest:  newToolReq,
					toolResponse: newToolResp,
				}, nil
			}
		}
	}

	refStr := p.ToolRequest.Name
	if p.ToolRequest.Ref != "" {
		refStr = "#" + p.ToolRequest.Ref
	}
	return nil, core.NewError(core.INVALID_ARGUMENT, fmt.Sprintf("unresolved tool request %q was not handled by the Resume argument; you must supply Respond or Restart directives, or ensure there is pending output from a previous tool call", refStr))
}

// handleResumeOption amends message history to handle `resume` arguments.
// It returns the amended history.
func handleResumeOption(ctx context.Context, r api.Registry, genOpts *GenerateActionOptions) (*resumeOptionOutput, error) {
	if genOpts.Resume == nil || (len(genOpts.Resume.Respond) == 0 && len(genOpts.Resume.Restart) == 0) {
		return &resumeOptionOutput{revisedRequest: genOpts}, nil
	}

	toolDefMap := make(map[string]*ToolDefinition)
	for _, t := range genOpts.Tools {
		tool := LookupTool(r, t)
		if tool == nil {
			return nil, core.NewError(core.NOT_FOUND, "handleResumeOption: tool %q not found", t)
		}
		toolDefMap[t] = tool.Definition()
	}

	messages := genOpts.Messages
	if len(messages) == 0 {
		return nil, core.NewError(core.FAILED_PRECONDITION, "handleResumeOption: cannot resume generation with no messages")
	}
	lastMessage := messages[len(messages)-1]

	if lastMessage.Role != RoleModel || !slices.ContainsFunc(lastMessage.Content, func(p *Part) bool { return p.IsToolRequest() }) {
		return nil, core.NewError(core.FAILED_PRECONDITION, "handleResumeOption: cannot resume generation unless the last message is by a model with at least one tool request")
	}

	resultChan := make(chan result[*resumedToolRequestOutput])
	newContent := make([]*Part, len(lastMessage.Content))
	toolReqCount := 0

	for i, part := range lastMessage.Content {
		if !part.IsToolRequest() {
			newContent[i] = part
			continue
		}
		toolReqCount++

		go func(idx int, p *Part) {
			output, err := handleResumedToolRequest(ctx, r, genOpts, p)
			resultChan <- result[*resumedToolRequestOutput]{
				index: idx,
				value: output,
				err:   err,
			}
		}(i, part)
	}

	var toolResps []*Part
	interrupted := false

	for range toolReqCount {
		res := <-resultChan
		if res.err != nil {
			return nil, fmt.Errorf("handleResumeOption: failed to resolve resumed tool request: %w", res.err)
		}

		if res.value.interrupt != nil {
			interrupted = true
			newContent[res.index] = res.value.interrupt
		} else {
			toolResps = append(toolResps, res.value.toolResponse)
			newContent[res.index] = res.value.toolRequest
		}
	}

	lastMessage.Content = newContent

	if interrupted {
		return &resumeOptionOutput{
			interruptedResponse: &ModelResponse{
				Message:       lastMessage,
				FinishReason:  "interrupted",
				FinishMessage: "One or more tools triggered interrupts while resuming generation. The model was not called.",
			},
		}, nil
	}

	if len(toolResps) != toolReqCount {
		return nil, core.NewError(core.FAILED_PRECONDITION, fmt.Sprintf("handleResumeOption: Expected %d tool responses but resolved to %d.", toolReqCount, len(toolResps)))
	}

	toolMessage := &Message{
		Role:    RoleTool,
		Content: toolResps,
		Metadata: map[string]any{
			"resumed": true,
		},
	}
	if genOpts.Resume.Metadata != nil {
		toolMessage.Metadata["resumed"] = genOpts.Resume.Metadata
	}
	revisedMessages := append(slices.Clone(messages), toolMessage)

	return &resumeOptionOutput{
		revisedRequest: &GenerateActionOptions{
			Model:              genOpts.Model,
			Messages:           revisedMessages,
			Tools:              genOpts.Tools,
			MaxTurns:           genOpts.MaxTurns,
			Config:             genOpts.Config,
			ToolChoice:         genOpts.ToolChoice,
			Docs:               genOpts.Docs,
			ReturnToolRequests: genOpts.ReturnToolRequests,
			Output:             genOpts.Output,
		},
		toolMessage: toolMessage,
	}, nil
}

// addAutomaticTelemetry creates middleware that automatically measures latency and calculates character and media counts.
func addAutomaticTelemetry() ModelMiddleware {
	return func(fn ModelFunc) ModelFunc {
		return func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
			startTime := time.Now()

			// Call the underlying model function
			resp, err := fn(ctx, req, cb)
			if err != nil {
				return nil, err
			}

			// Calculate latency
			latencyMs := float64(time.Since(startTime).Nanoseconds()) / 1e6
			if resp.LatencyMs == 0 {
				resp.LatencyMs = latencyMs
			}

			if resp.Usage == nil {
				resp.Usage = &GenerationUsage{}
			}
			if resp.Usage.InputCharacters == 0 {
				resp.Usage.InputCharacters = countInputCharacters(req)
			}
			if resp.Usage.OutputCharacters == 0 {
				resp.Usage.OutputCharacters = countOutputCharacters(resp)
			}
			if resp.Usage.InputImages == 0 {
				resp.Usage.InputImages = countInputParts(req, func(part *Part) bool { return part.IsImage() })
			}
			if resp.Usage.OutputImages == 0 {
				resp.Usage.OutputImages = countOutputParts(resp, func(part *Part) bool { return part.IsImage() })
			}
			if resp.Usage.InputVideos == 0 {
				resp.Usage.InputVideos = countInputParts(req, func(part *Part) bool { return part.IsVideo() })
			}
			if resp.Usage.OutputVideos == 0 {
				resp.Usage.OutputVideos = countOutputParts(resp, func(part *Part) bool { return part.IsVideo() })
			}
			if resp.Usage.InputAudioFiles == 0 {
				resp.Usage.InputAudioFiles = countInputParts(req, func(part *Part) bool { return part.IsAudio() })
			}
			if resp.Usage.OutputAudioFiles == 0 {
				resp.Usage.OutputAudioFiles = countOutputParts(resp, func(part *Part) bool { return part.IsAudio() })
			}

			return resp, nil
		}
	}
}

// countInputParts counts parts in the input request that match the given predicate.
func countInputParts(req *ModelRequest, predicate func(*Part) bool) int {
	if req == nil {
		return 0
	}

	count := 0
	for _, msg := range req.Messages {
		if msg == nil {
			continue
		}
		for _, part := range msg.Content {
			if part != nil && predicate(part) {
				count++
			}
		}
	}
	return count
}

// countInputCharacters counts the total characters in the input request.
func countInputCharacters(req *ModelRequest) int {
	if req == nil {
		return 0
	}

	total := 0
	for _, msg := range req.Messages {
		if msg == nil {
			continue
		}
		for _, part := range msg.Content {
			if part != nil && part.Text != "" {
				total += len(part.Text)
			}
		}
	}
	return total
}

// countOutputParts counts parts in the output response that match the given predicate.
func countOutputParts(resp *ModelResponse, predicate func(*Part) bool) int {
	if resp == nil || resp.Message == nil {
		return 0
	}

	count := 0
	for _, part := range resp.Message.Content {
		if part != nil && predicate(part) {
			count++
		}
	}
	return count
}

// countOutputCharacters counts the total characters in the output response.
func countOutputCharacters(resp *ModelResponse) int {
	if resp == nil || resp.Message == nil {
		return 0
	}

	total := 0
	for _, part := range resp.Message.Content {
		if part != nil && part.Text != "" {
			total += len(part.Text)
		}
	}
	return total
}

// processResources processes messages to replace resource parts with actual content.
func processResources(ctx context.Context, r api.Registry, messages []*Message) ([]*Message, error) {
	processedMessages := make([]*Message, len(messages))
	for i, msg := range messages {
		processedContent := []*Part{}

		for _, part := range msg.Content {
			if part.IsResource() {
				// Find and execute the matching resource
				resourceParts, err := executeResourcePart(ctx, r, part.Resource.Uri)
				if err != nil {
					return nil, fmt.Errorf("failed to process resource %q: %w", part.Resource, err)
				}
				// Replace resource part with content parts
				processedContent = append(processedContent, resourceParts...)
			} else {
				// Keep non-resource parts as-is
				processedContent = append(processedContent, part)
			}
		}

		processedMessages[i] = &Message{
			Role:     msg.Role,
			Content:  processedContent,
			Metadata: msg.Metadata,
		}
	}

	return processedMessages, nil
}

// executeResourcePart finds and executes a resource, returning the content parts.
func executeResourcePart(ctx context.Context, r api.Registry, resourceURI string) ([]*Part, error) {
	resource, input, err := FindMatchingResource(r, resourceURI)
	if err != nil {
		return nil, err
	}

	output, err := resource.Execute(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to execute resource %q: %w", resourceURI, err)
	}

	return output.Content, nil
}
