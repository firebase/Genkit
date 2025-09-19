// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
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
	"errors"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/registry"
)

// BackgroundModel represents a model that can run operations in the background.
type BackgroundModel interface {
	// Name returns the registry name of the background model.
	Name() string

	// StartOperation starts a background operation.
	StartOperation(ctx context.Context, req *ModelRequest) (*core.Operation[*ModelResponse], error)

	// CheckOperation checks the status of a background operation.
	CheckOperation(ctx context.Context, operation *core.Operation[*ModelResponse]) (*core.Operation[*ModelResponse], error)

	// CancelOperation cancels a background operation.
	CancelOperation(ctx context.Context, operation *core.Operation[*ModelResponse]) (*core.Operation[*ModelResponse], error)

	// Register registers the model with the given registry.
	Register(r api.Registry)
}

// backgroundModel is the concrete implementation of BackgroundModel interface.
type backgroundModel struct {
	*core.BackgroundActionDef[*ModelRequest, *ModelResponse]
}

// Name returns the registry name of the background model.
func (bm *backgroundModel) Name() string {
	if bm == nil || bm.BackgroundActionDef == nil {
		return ""
	}
	return bm.BackgroundActionDef.Name()
}

// Register registers the model with the given registry.
func (bm *backgroundModel) Register(r api.Registry) {

	if bm == nil || bm.BackgroundActionDef == nil {
		return
	}
	core.DefineBackgroundAction(r, bm.BackgroundActionDef.Name(), nil, bm.BackgroundActionDef.Start, bm.BackgroundActionDef.Check, bm.BackgroundActionDef.Cancel)
}

func (bm *backgroundModel) StartOperation(ctx context.Context, req *ModelRequest) (*core.Operation[*ModelResponse], error) {
	if bm == nil || bm.BackgroundActionDef == nil {
		return nil, core.NewError(core.INVALID_ARGUMENT, "BackgroundModel.StartOperation: background model is nil")
	}
	return bm.BackgroundActionDef.Start(ctx, req)
}

func (bm *backgroundModel) CheckOperation(ctx context.Context, operation *core.Operation[*ModelResponse]) (*core.Operation[*ModelResponse], error) {
	if bm == nil || bm.BackgroundActionDef == nil {
		return nil, core.NewError(core.INVALID_ARGUMENT, "BackgroundModel.CheckOperation: background model is nil")
	}
	return bm.BackgroundActionDef.Check(ctx, operation)
}

func (bm *backgroundModel) CancelOperation(ctx context.Context, operation *core.Operation[*ModelResponse]) (*core.Operation[*ModelResponse], error) {
	if bm == nil || bm.BackgroundActionDef == nil {
		return nil, core.NewError(core.INVALID_ARGUMENT, "BackgroundModel.CancelOperation: background model is nil")
	}
	return bm.BackgroundActionDef.Cancel(ctx, operation)
}

// StartOperationFunc starts a background operation
type StartOperationFunc[In, Out any] = func(ctx context.Context, input In) (*core.Operation[Out], error)

// CheckOperationFunc checks the status of a background operation
type CheckOperationFunc[Out any] = func(ctx context.Context, operation *core.Operation[Out]) (*core.Operation[Out], error)

// CancelOperationFunc cancels a background operation
type CancelOperationFunc[Out any] = func(ctx context.Context, operation *core.Operation[Out]) (*core.Operation[Out], error)

// BackgroundModelOptions holds configuration for defining a background model
type BackgroundModelOptions struct {
	ModelOptions
	Metadata map[string]any `json:"metadata,omitempty"`
	Start    StartOperationFunc[*ModelRequest, *ModelResponse]
	Check    CheckOperationFunc[*ModelResponse]
	Cancel   CancelOperationFunc[*ModelResponse]
}

// LookupBackgroundModel looks up a BackgroundAction registered by [DefineBackgroundModel].
// It returns nil if the background model was not found.
func LookupBackgroundModel(r api.Registry, name string) BackgroundModel {
	action := core.LookupBackgroundAction[*ModelRequest, *ModelResponse](r, name)
	if action == nil {
		return nil
	}
	return &backgroundModel{action}
}

// NewBackgroundModel defines a new model that runs in the background
func NewBackgroundModel(
	name string,
	opts *BackgroundModelOptions,
) BackgroundModel {
	if name == "" {
		panic("ai.NewBackgroundModel: name is required")
	}

	if opts == nil {
		opts = &BackgroundModelOptions{}
	}

	metadata := make(map[string]any)
	if opts.Metadata != nil {
		for k, v := range opts.Metadata {
			metadata[k] = v
		}
	}

	// Add model-specific metadata
	label := opts.Label
	if label == "" {
		label = name
	}
	metadata["model"] = map[string]any{
		"label":    label,
		"versions": opts.Versions,
		"supports": opts.Supports,
	}
	if opts.ConfigSchema != nil {
		metadata["customOptions"] = opts.ConfigSchema
		if modelMeta, ok := metadata["model"].(map[string]any); ok {
			modelMeta["customOptions"] = opts.ConfigSchema
		}
	}

	return &backgroundModel{core.NewBackgroundAction[*ModelRequest, *ModelResponse](name, metadata,
		opts.Start, opts.Check, opts.Cancel)}
}

// DefineBackgroundModel defines and registers a new model that runs in the background.
func DefineBackgroundModel(
	r *registry.Registry,
	name string,
	opts *BackgroundModelOptions,
) BackgroundModel {
	if opts == nil {
		opts = &BackgroundModelOptions{}
	}

	m := NewBackgroundModel(name, opts)
	m.Register(r)
	return m
}

// GenerateOperation generates a model response as a long-running operation based on the provided options.
func GenerateOperation(ctx context.Context, r *registry.Registry, opts ...GenerateOption) (*core.Operation[*ModelResponse], error) {

	resp, err := Generate(ctx, r, opts...)
	if err != nil {
		return nil, err
	}

	if resp.Operation == nil {
		return nil, core.NewError(core.FAILED_PRECONDITION, "model did not return an operation")
	}

	var action string
	if v, ok := resp.Operation["action"].(string); ok {
		action = v
	} else {
		return nil, core.NewError(core.INTERNAL, "operation missing or invalid 'action' field")
	}
	var id string
	if v, ok := resp.Operation["id"].(string); ok {
		id = v
	} else {
		return nil, core.NewError(core.INTERNAL, "operation missing or invalid 'id' field")
	}
	var done bool
	if v, ok := resp.Operation["done"].(bool); ok {
		done = v
	} else {
		return nil, core.NewError(core.INTERNAL, "operation missing or invalid 'done' field")
	}
	var metadata map[string]any
	if v, ok := resp.Operation["metadata"].(map[string]any); ok {
		metadata = v
	}

	op := &core.Operation[*ModelResponse]{
		Action:   action,
		ID:       id,
		Done:     done,
		Metadata: metadata,
	}

	if op.Done {
		if output, ok := resp.Operation["output"]; ok {
			if modelResp, ok := output.(*ModelResponse); ok {
				op.Output = modelResp
			} else {
				op.Output = resp
			}
		} else {
			op.Output = resp
		}
	}

	if errorData, ok := resp.Operation["error"]; ok {
		if errorMap, ok := errorData.(map[string]any); ok {
			if message, ok := errorMap["message"].(string); ok {
				op.Error = errors.New(message)
			}
		}
	}

	return op, nil
}
