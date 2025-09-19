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
	"maps"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/base"
)

var resumedCtxKey = base.NewContextKey[map[string]any]()
var origInputCtxKey = base.NewContextKey[any]()

// ToolFunc is the function type for tool implementations.
type ToolFunc[In, Out any] = func(ctx *ToolContext, input In) (Out, error)

// ToolRef is a reference to a tool.
type ToolRef interface {
	Name() string
}

// ToolName is a distinct type for a tool name.
// It is meant to be passed where a ToolRef is expected but no Tool is had.
type ToolName string

// Name returns the name of the tool.
func (t ToolName) Name() string {
	return (string)(t)
}

// tool is an action with functions specific to tools.
// It embeds [core.Action] instead of [core.ActionDef] like other primitives
// because the inputs/outputs can vary and the tool is only meant to be called
// with JSON input anyway.
type tool struct {
	api.Action
}

// Tool represents a tool that can be called by a model.
type Tool interface {
	// Name returns the name of the tool.
	Name() string
	// Definition returns the definition for this tool to be passed to models.
	Definition() *ToolDefinition
	// RunRaw runs this tool using the provided raw input.
	RunRaw(ctx context.Context, input any) (any, error)
	// Respond constructs a [Part] with a [ToolResponse] for a given interrupted tool request.
	Respond(toolReq *Part, outputData any, opts *RespondOptions) *Part
	// Restart constructs a [Part] with a new [ToolRequest] to re-trigger a tool,
	// potentially with new input and metadata.
	Restart(toolReq *Part, opts *RestartOptions) *Part
	// Register registers the tool with the given registry.
	Register(r api.Registry)
}

// toolInterruptError represents an intentional interruption of tool execution.
type toolInterruptError struct {
	Metadata map[string]any
}

func (e *toolInterruptError) Error() string {
	return "tool execution interrupted"
}

// Determine whether the error is an interrupt error returned by the tool。
func IsToolInterruptError(err error) (map[string]any, bool) {
	var tie *toolInterruptError
	if errors.As(err, &tie) {
		return tie.Metadata, true
	}
	return nil, false
}

// InterruptOptions provides configuration for tool interruption.
type InterruptOptions struct {
	Metadata map[string]any
}

// RestartOptions provides configuration options for restarting a tool.
type RestartOptions struct {
	// ReplaceInput allows replacing the existing input arguments to the tool with different ones,
	// for example if the user revised an action before confirming. When input is replaced,
	// the existing tool request will be amended in the message history.
	ReplaceInput any
	// ResumedMetadata is the metadata you want to provide to the tool to aide in reprocessing.
	// Defaults to true if none is supplied.
	ResumedMetadata any
}

// RespondOptions provides configuration options for responding to a tool request.
type RespondOptions struct {
	// Metadata is additional metadata to include in the response.
	Metadata map[string]any
}

// ToolContext provides context and utility functions for tool execution.
type ToolContext struct {
	context.Context
	// Interrupt is a function that can be used to interrupt the tool execution.
	// Interrupting tool execution returns the control to the caller with the
	// total model response so far.
	Interrupt func(opts *InterruptOptions) error
	// Resumed is optional metadata that can be used to resume the tool execution.
	// Map is not nil only if the tool was interrupted.
	Resumed map[string]any
	// OriginalInput is the original input to the tool if the tool was interrupted, otherwise nil.
	OriginalInput any
}

// DefineTool creates a new [Tool] and registers it.
func DefineTool[In, Out any](
	r api.Registry,
	name, description string,
	fn ToolFunc[In, Out],
) Tool {
	metadata, wrappedFn := implementTool(name, description, fn)
	toolAction := core.DefineAction(r, name, api.ActionTypeTool, metadata, nil, wrappedFn)
	return &tool{Action: toolAction}
}

// DefineToolWithInputSchema creates a new [Tool] with a custom input schema and registers it.
func DefineToolWithInputSchema[Out any](
	r api.Registry,
	name, description string,
	inputSchema map[string]any,
	fn ToolFunc[any, Out],
) Tool {
	metadata, wrappedFn := implementTool(name, description, fn)
	toolAction := core.DefineAction(r, name, api.ActionTypeTool, metadata, inputSchema, wrappedFn)
	return &tool{Action: toolAction}
}

// NewTool creates a new [Tool]. It can be passed directly to [Generate].
func NewTool[In, Out any](name, description string, fn ToolFunc[In, Out]) Tool {
	metadata, wrappedFn := implementTool(name, description, fn)
	metadata["dynamic"] = true
	toolAction := core.NewAction(name, api.ActionTypeTool, metadata, nil, wrappedFn)
	return &tool{Action: toolAction}
}

// NewToolWithInputSchema creates a new [Tool] with a custom input schema. It can be passed directly to [Generate].
func NewToolWithInputSchema[Out any](name, description string, inputSchema map[string]any, fn ToolFunc[any, Out]) Tool {
	metadata, wrappedFn := implementTool(name, description, fn)
	metadata["dynamic"] = true
	toolAction := core.NewAction(name, api.ActionTypeTool, metadata, inputSchema, wrappedFn)
	return &tool{Action: toolAction}
}

// implementTool creates the metadata and wrapped function common to both DefineTool and NewTool.
func implementTool[In, Out any](name, description string, fn ToolFunc[In, Out]) (map[string]any, func(context.Context, In) (Out, error)) {
	metadata := map[string]any{
		"type":        api.ActionTypeTool,
		"name":        name,
		"description": description,
	}
	wrappedFn := func(ctx context.Context, input In) (Out, error) {
		toolCtx := &ToolContext{
			Context: ctx,
			Interrupt: func(opts *InterruptOptions) error {
				return &toolInterruptError{
					Metadata: opts.Metadata,
				}
			},
			Resumed:       resumedCtxKey.FromContext(ctx),
			OriginalInput: origInputCtxKey.FromContext(ctx),
		}
		return fn(toolCtx, input)
	}

	return metadata, wrappedFn
}

// Name returns the name of the tool.
func (t *tool) Name() string {
	return t.Action.Name()
}

// Definition returns [ToolDefinition] for for this tool.
func (t *tool) Definition() *ToolDefinition {
	desc := t.Action.Desc()
	return &ToolDefinition{
		Name:         desc.Name,
		Description:  desc.Description,
		InputSchema:  desc.InputSchema,
		OutputSchema: desc.OutputSchema,
	}
}

// RunRaw runs this tool using the provided raw map format data (JSON parsed
// as map[string]any).
func (t *tool) RunRaw(ctx context.Context, input any) (any, error) {
	if t == nil {
		return nil, core.NewError(core.INVALID_ARGUMENT, "Tool.RunRaw: tool called on a nil tool; check that all tools are defined")
	}

	mi, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("error marshalling tool input for %v: %v", t.Name(), err)
	}
	output, err := t.RunJSON(ctx, mi, nil)
	if err != nil {
		return nil, fmt.Errorf("error calling tool %v: %w", t.Name(), err)
	}

	var uo any
	err = json.Unmarshal(output, &uo)
	if err != nil {
		return nil, fmt.Errorf("error parsing tool output for %v: %v", t.Name(), err)
	}
	return uo, nil
}

// LookupTool looks up the tool in the registry by provided name and returns it.
func LookupTool(r api.Registry, name string) Tool {
	if name == "" {
		return nil
	}
	provider, id := api.ParseName(name)
	key := api.NewKey(api.ActionTypeTool, provider, id)
	action := r.LookupAction(key)
	if action == nil {
		return nil
	}
	return &tool{Action: action}
}

// Respond creates a tool response for an interrupted tool call to pass to the [WithToolResponses] option to [Generate].
// If the part provided is not a tool request, it returns nil.
func (t *tool) Respond(toolReq *Part, output any, opts *RespondOptions) *Part {
	if toolReq == nil || !toolReq.IsToolRequest() {
		return nil
	}

	if opts == nil {
		opts = &RespondOptions{}
	}

	newToolResp := NewResponseForToolRequest(toolReq, output)
	newToolResp.Metadata = map[string]any{
		"interruptResponse": true,
	}
	if opts.Metadata != nil {
		newToolResp.Metadata["interruptResponse"] = opts.Metadata
	}

	return newToolResp
}

// Restart creates a tool request for an interrupted tool call to pass to the [WithToolRestarts] option to [Generate].
// If the part provided is not a tool request, it returns nil.
func (t *tool) Restart(p *Part, opts *RestartOptions) *Part {
	if p == nil || !p.IsToolRequest() {
		return nil
	}

	if opts == nil {
		opts = &RestartOptions{}
	}

	newInput := p.ToolRequest.Input
	var originalInput any

	if opts.ReplaceInput != nil {
		originalInput = newInput
		newInput = opts.ReplaceInput
	}

	newMeta := maps.Clone(p.Metadata)
	if newMeta == nil {
		newMeta = make(map[string]any)
	}

	newMeta["resumed"] = true
	if opts.ResumedMetadata != nil {
		newMeta["resumed"] = opts.ResumedMetadata
	}

	if originalInput != nil {
		newMeta["replacedInput"] = originalInput
	}

	delete(newMeta, "interrupt")

	newToolReq := NewToolRequestPart(&ToolRequest{
		Name:  p.ToolRequest.Name,
		Ref:   p.ToolRequest.Ref,
		Input: newInput,
	})
	newToolReq.Metadata = newMeta

	return newToolReq
}

// resolveUniqueTools resolves the list of tool refs to a list of all tool names and new tools that must be registered.
// Returns an error if there are tool refs with duplicate names.
func resolveUniqueTools(r api.Registry, toolRefs []ToolRef) (toolNames []string, newTools []Tool, err error) {
	toolMap := make(map[string]bool)

	for _, toolRef := range toolRefs {
		name := toolRef.Name()

		if toolMap[name] {
			return nil, nil, core.NewError(core.INVALID_ARGUMENT, "duplicate tool %q", name)
		}
		toolMap[name] = true
		toolNames = append(toolNames, name)

		if LookupTool(r, name) == nil {
			if tool, ok := toolRef.(Tool); ok {
				newTools = append(newTools, tool)
			}
		}
	}

	return toolNames, newTools, nil
}
