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

package core

import (
	"context"
	"fmt"
	"time"

	"github.com/firebase/genkit/go/core/api"
)

// Operation represents a background task operation
type Operation[Out any] struct {
	Action   string         `json:"action,omitempty"`   // The action that created this operation
	ID       string         `json:"id"`                 // Unique identifier for tracking
	Done     bool           `json:"done,omitempty"`     // Whether the operation is complete
	Output   Out            `json:"output,omitempty"`   // Result when done
	Error    error          `json:"error,omitempty"`    // Error if failed
	Metadata map[string]any `json:"metadata,omitempty"` // Additional info
}

// BackgroundActionDef implements BackgroundAction
type BackgroundActionDef[In, Out any] struct {
	startAction  *ActionDef[In, *Operation[Out], struct{}]
	checkAction  *ActionDef[*Operation[Out], *Operation[Out], struct{}]
	cancelAction *ActionDef[*Operation[Out], *Operation[Out], struct{}]
	name         string
}

// Start initiates a background operation
func (b *BackgroundActionDef[In, Out]) Start(ctx context.Context, input In) (*Operation[Out], error) {
	return b.startAction.Run(ctx, input, nil)
}

// Check polls the status of a background operation
func (b *BackgroundActionDef[In, Out]) Check(ctx context.Context, op *Operation[Out]) (*Operation[Out], error) {
	return b.checkAction.Run(ctx, op, nil)
}

// Cancel attempts to cancel a background operation
func (b *BackgroundActionDef[In, Out]) Cancel(ctx context.Context, op *Operation[Out]) (*Operation[Out], error) {
	return b.cancelAction.Run(ctx, op, nil)
}

// Name returns the action name
func (b *BackgroundActionDef[In, Out]) Name() string {
	return b.name
}

// DefineBackgroundAction creates and registers a background action with three component actions
func DefineBackgroundAction[In, Out any](
	r api.Registry,
	name string,
	metadata map[string]any,
	startFunc func(context.Context, In) (*Operation[Out], error),
	checkFunc func(context.Context, *Operation[Out]) (*Operation[Out], error),
	cancelFunc func(context.Context, *Operation[Out]) (*Operation[Out], error),
) *BackgroundActionDef[In, Out] {
	if startFunc == nil {
		panic("DefineBackgroundAction requires a start function")
	}
	if checkFunc == nil {
		panic("DefineBackgroundAction requires a check function")
	}
	startAction := defineAction(r, name, api.ActionTypeBackgroundModel, metadata, nil,
		func(ctx context.Context, input In, _ func(context.Context, struct{}) error) (*Operation[Out], error) {
			startTime := time.Now()
			operation, err := startFunc(ctx, input)
			if err != nil {
				return nil, err
			}
			if operation.Metadata == nil {
				operation.Metadata = make(map[string]any)
			}
			operation.Metadata["latencyMs"] = float64(time.Since(startTime).Nanoseconds()) / 1e6
			operation.Action = fmt.Sprintf("/%s/%s", api.ActionTypeBackgroundModel, name)
			return operation, nil
		})

	checkAction := defineAction(r, name, api.ActionTypeCheckOperation,
		map[string]any{"description": fmt.Sprintf("Check status of %s operation", name)},
		nil,
		func(ctx context.Context, op *Operation[Out], _ func(context.Context, struct{}) error) (*Operation[Out], error) {
			updatedOp, err := checkFunc(ctx, op)
			if err != nil {
				return nil, err
			}
			// Ensure action reference is maintained
			updatedOp.Action = fmt.Sprintf("/%s/%s", api.ActionTypeCheckOperation, name)
			return updatedOp, nil
		})

	var cancelAction *ActionDef[*Operation[Out], *Operation[Out], struct{}]
	if cancelFunc != nil {
		cancelAction = defineAction(r, name, api.ActionTypeCancelOperation,
			map[string]any{"description": fmt.Sprintf("Cancel %s operation", name)},
			nil,
			func(ctx context.Context, op *Operation[Out], _ func(context.Context, struct{}) error) (*Operation[Out], error) {
				cancelledOp, err := cancelFunc(ctx, op)
				if err != nil {
					return nil, err
				}
				cancelledOp.Action = fmt.Sprintf("/%s/%s", api.ActionTypeCancelOperation, name)
				return cancelledOp, nil
			})
	}

	return &BackgroundActionDef[In, Out]{
		startAction:  startAction,
		checkAction:  checkAction,
		cancelAction: cancelAction,
		name:         name,
	}
}

// NewBackgroundAction creates a new background action without registering it.
func NewBackgroundAction[In, Out any](
	name string,
	metadata map[string]any,
	startFunc func(context.Context, In) (*Operation[Out], error),
	checkFunc func(context.Context, *Operation[Out]) (*Operation[Out], error),
	cancelFunc func(context.Context, *Operation[Out]) (*Operation[Out], error),
) *BackgroundActionDef[In, Out] {
	if startFunc == nil {
		panic("NewBackgroundAction requires a start function")
	}
	if checkFunc == nil {
		panic("NewBackgroundAction requires a check function")
	}

	startAction := NewAction(name, api.ActionTypeBackgroundModel, metadata, nil,
		func(ctx context.Context, input In) (*Operation[Out], error) {
			startTime := time.Now()
			operation, err := startFunc(ctx, input)
			if err != nil {
				return nil, err
			}
			if operation.Metadata == nil {
				operation.Metadata = make(map[string]any)
			}
			operation.Metadata["latencyMs"] = float64(time.Since(startTime).Nanoseconds()) / 1e6
			operation.Action = fmt.Sprintf("/%s/%s", api.ActionTypeBackgroundModel, name)
			return operation, nil
		})

	checkAction := NewAction(name, api.ActionTypeCheckOperation,
		map[string]any{"description": fmt.Sprintf("Check status of %s operation", name)},
		nil,
		func(ctx context.Context, op *Operation[Out]) (*Operation[Out], error) {
			updatedOp, err := checkFunc(ctx, op)
			if err != nil {
				return nil, err
			}
			// Ensure action reference is maintained
			updatedOp.Action = fmt.Sprintf("/%s/%s", api.ActionTypeCheckOperation, name)
			return updatedOp, nil
		})

	var cancelAction *ActionDef[*Operation[Out], *Operation[Out], struct{}]
	if cancelFunc != nil {
		cancelAction = NewAction(name, api.ActionTypeCancelOperation,
			map[string]any{"description": fmt.Sprintf("Cancel %s operation", name)},
			nil,
			func(ctx context.Context, op *Operation[Out]) (*Operation[Out], error) {
				cancelledOp, err := cancelFunc(ctx, op)
				if err != nil {
					return nil, err
				}
				cancelledOp.Action = fmt.Sprintf("/%s/%s", api.ActionTypeCancelOperation, name)
				return cancelledOp, nil
			})
	}

	return &BackgroundActionDef[In, Out]{
		startAction:  startAction,
		checkAction:  checkAction,
		cancelAction: cancelAction,
		name:         name,
	}
}

// LookupBackgroundAction finds and assembles a background action from the registry
func LookupBackgroundAction[In, Out any](r api.Registry, name string) *BackgroundActionDef[In, Out] {

	startAction := ResolveActionFor[In, *Operation[Out], struct{}](r, api.ActionTypeBackgroundModel, name)
	if startAction == nil {
		return nil
	}

	checkAction := ResolveActionFor[*Operation[Out], *Operation[Out], struct{}](r, api.ActionTypeCheckOperation, name)
	if checkAction == nil {
		return nil
	}
	cancelAction := ResolveActionFor[*Operation[Out], *Operation[Out], struct{}](r, api.ActionTypeCancelOperation, name)

	bgAction := BackgroundActionDef[In, Out]{
		startAction:  startAction,
		checkAction:  checkAction,
		cancelAction: cancelAction,
		name:         name,
	}
	return &bgAction
}
