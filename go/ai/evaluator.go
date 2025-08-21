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
	"fmt"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal/registry"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/trace"
)

// EvaluatorFunc is the function type for evaluator implementations.
type EvaluatorFunc = func(context.Context, *EvaluatorCallbackRequest) (*EvaluatorCallbackResponse, error)

// BatchEvaluatorFunc is the function type for batch evaluator implementations.
type BatchEvaluatorFunc = func(context.Context, *EvaluatorRequest) (*EvaluatorResponse, error)

// Evaluator represents a evaluator action.
type Evaluator interface {
	// Name returns the name of the evaluator.
	Name() string
	// Evaluates a dataset.
	Evaluate(ctx context.Context, req *EvaluatorRequest) (*EvaluatorResponse, error)
}

// evaluator is an action with functions specific to evaluating a dataset.
type evaluator core.ActionDef[*EvaluatorRequest, *EvaluatorResponse, struct{}]

// Example is a single example that requires evaluation
type Example struct {
	TestCaseId string   `json:"testCaseId,omitempty"`
	Input      any      `json:"input"`
	Output     any      `json:"output,omitempty"`
	Context    []any    `json:"context,omitempty"`
	Reference  any      `json:"reference,omitempty"`
	TraceIds   []string `json:"traceIds,omitempty"`
}

// EvaluatorRequest is the data we pass to evaluate a dataset.
// The Options field is specific to the actual evaluator implementation.
type EvaluatorRequest struct {
	Dataset      []*Example `json:"dataset"`
	EvaluationId string     `json:"evalRunId"`
	Options      any        `json:"options,omitempty"`
}

// ScoreStatus is an enum used to indicate if a Score has passed or failed. This
// drives additional features in tooling / the Dev UI.
type ScoreStatus int

const (
	ScoreStatusUnknown ScoreStatus = iota
	ScoreStatusFail
	ScoreStatusPass
)

var statusName = map[ScoreStatus]string{
	ScoreStatusUnknown: "UNKNOWN",
	ScoreStatusFail:    "FAIL",
	ScoreStatusPass:    "PASS",
}

func (ss ScoreStatus) String() string {
	return statusName[ss]
}

// Score is the evaluation score that represents the result of an evaluator.
// This struct includes information such as the score (numeric, string or other
// types), the reasoning provided for this score (if any), the score status (if
// any) and other details.
type Score struct {
	Id      string         `json:"id,omitempty"`
	Score   any            `json:"score,omitempty"`
	Status  string         `json:"status,omitempty" jsonschema:"enum=UNKNOWN,enum=FAIL,enum=PASS"`
	Error   string         `json:"error,omitempty"`
	Details map[string]any `json:"details,omitempty"`
}

// EvaluationResult is the result of running the evaluator on a single Example.
// An evaluator may provide multiple scores simultaneously (e.g. if they are using
// an API to score on multiple criteria)
type EvaluationResult struct {
	TestCaseId string  `json:"testCaseId"`
	TraceID    string  `json:"traceId,omitempty"`
	SpanID     string  `json:"spanId,omitempty"`
	Evaluation []Score `json:"evaluation"`
}

// EvaluatorResponse is a collection of [EvaluationResult] structs, it
// represents the result on the entire input dataset.
type EvaluatorResponse = []EvaluationResult

type EvaluatorOptions struct {
	// ConfigSchema is the JSON schema for the evaluator's config.
	ConfigSchema map[string]any `json:"configSchema,omitempty"`
	// DisplayName is the name of the evaluator as it appears in the UI.
	DisplayName string `json:"displayName"`
	// Definition is the definition of the evaluator.
	Definition string `json:"definition"`
	// IsBilled is a flag indicating if the evaluator is billed.
	IsBilled bool `json:"isBilled,omitempty"`
}

// EvaluatorCallbackRequest is the data we pass to the callback function
// provided in defineEvaluator. The Options field is specific to the actual
// evaluator implementation.
type EvaluatorCallbackRequest struct {
	Input   Example `json:"input"`
	Options any     `json:"options,omitempty"`
}

// EvaluatorCallbackResponse is the result on evaluating a single [Example]
type EvaluatorCallbackResponse = EvaluationResult

// DefineEvaluator registers the given evaluator function as an action, and
// returns a [Evaluator] that runs it. This method process the input dataset
// one-by-one.
func DefineEvaluator(r *registry.Registry, name string, opts *EvaluatorOptions, fn EvaluatorFunc) Evaluator {
	if name == "" {
		panic("ai.DefineEvaluator: evaluator name is required")
	}

	if opts == nil {
		opts = &EvaluatorOptions{}
	}

	// TODO(ssbushi): Set this on `evaluator` key on action metadata
	metadata := map[string]any{
		"type": core.ActionTypeEvaluator,
		"evaluator": map[string]any{
			"evaluatorIsBilled":    opts.IsBilled,
			"evaluatorDisplayName": opts.DisplayName,
			"evaluatorDefinition":  opts.Definition,
		},
	}

	inputSchema := core.InferSchemaMap(EvaluatorRequest{})
	if inputSchema != nil && opts.ConfigSchema != nil {
		if _, ok := inputSchema["options"]; ok {
			inputSchema["options"] = opts.ConfigSchema
		}
	}

	return (*evaluator)(core.DefineActionWithInputSchema(r, name, core.ActionTypeEvaluator, metadata, inputSchema, func(ctx context.Context, req *EvaluatorRequest) (output *EvaluatorResponse, err error) {
		var results []EvaluationResult
		for _, datapoint := range req.Dataset {
			if datapoint.TestCaseId == "" {
				datapoint.TestCaseId = uuid.New().String()
			}
			_, err := tracing.RunInNewSpan(ctx, r.TracingState(), fmt.Sprintf("TestCase %s", datapoint.TestCaseId), "evaluator", false, datapoint,
				func(ctx context.Context, input *Example) (*EvaluatorCallbackResponse, error) {
					traceId := trace.SpanContextFromContext(ctx).TraceID().String()
					spanId := trace.SpanContextFromContext(ctx).SpanID().String()

					callbackRequest := EvaluatorCallbackRequest{
						Input:   *input,
						Options: req.Options,
					}

					result, err := fn(ctx, &callbackRequest)
					if err != nil {
						failedScore := Score{
							Status: ScoreStatusFail.String(),
							Error:  fmt.Sprintf("Evaluation of test case %s failed: \n %s", input.TestCaseId, err.Error()),
						}
						failedResult := EvaluationResult{
							TestCaseId: input.TestCaseId,
							Evaluation: []Score{failedScore},
							TraceID:    traceId,
							SpanID:     spanId,
						}
						results = append(results, failedResult)
						// return error to mark span as failed
						return nil, err
					}

					result.TraceID = traceId
					result.SpanID = spanId

					results = append(results, *result)

					return result, nil
				})
			if err != nil {
				logger.FromContext(ctx).Debug("EvaluatorAction", "err", err)
				continue
			}
		}
		return &results, nil
	}))
}

// DefineBatchEvaluator registers the given evaluator function as an action, and
// returns a [Evaluator] that runs it. This method provide the full
// [EvaluatorRequest] to the callback function, giving more flexibilty to the
// user for processing the data, such as batching or parallelization.
func DefineBatchEvaluator(r *registry.Registry, name string, opts *EvaluatorOptions, fn BatchEvaluatorFunc) Evaluator {
	if name == "" {
		panic("ai.DefineBatchEvaluator: batch evaluator name is required")
	}

	if opts == nil {
		opts = &EvaluatorOptions{}
	}

	metadata := map[string]any{
		"type": core.ActionTypeEvaluator,
		"evaluator": map[string]any{
			"evaluatorIsBilled":    opts.IsBilled,
			"evaluatorDisplayName": opts.DisplayName,
			"evaluatorDefinition":  opts.Definition,
		},
	}

	return (*evaluator)(core.DefineAction(r, name, core.ActionTypeEvaluator, metadata, fn))
}

// LookupEvaluator looks up an [Evaluator] registered by [DefineEvaluator].
// It returns nil if the evaluator was not defined.
func LookupEvaluator(r *registry.Registry, name string) Evaluator {
	return (*evaluator)(core.LookupActionFor[*EvaluatorRequest, *EvaluatorResponse, struct{}](r, core.ActionTypeEvaluator, name))
}

// Evaluate calls the retrivers with provided options.
func Evaluate(ctx context.Context, r Evaluator, opts ...EvaluatorOption) (*EvaluatorResponse, error) {
	evalOpts := &evaluatorOptions{}
	for _, opt := range opts {
		err := opt.applyEvaluator(evalOpts)
		if err != nil {
			return nil, err
		}
	}

	req := &EvaluatorRequest{
		Dataset:      evalOpts.Dataset,
		EvaluationId: evalOpts.ID,
		Options:      evalOpts.Config,
	}

	return r.Evaluate(ctx, req)
}

// Name returns the name of the evaluator.
func (e evaluator) Name() string {
	return (*core.ActionDef[*EvaluatorRequest, *EvaluatorResponse, struct{}])(&e).Name()
}

// Evaluate runs the given [Evaluator].
func (e evaluator) Evaluate(ctx context.Context, req *EvaluatorRequest) (*EvaluatorResponse, error) {
	return (*core.ActionDef[*EvaluatorRequest, *EvaluatorResponse, struct{}])(&e).Run(ctx, req, nil)
}
