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

package googlegenai

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/plugins/internal/uri"
	"google.golang.org/genai"
)

// defineVeoModels defines a new Veo background model for video generation.
// using Google's Veo API through the genai client.
func newVeoModel(
	client *genai.Client,
	name string,
	info ai.ModelOptions,
) ai.BackgroundModel {

	startFunc := func(ctx context.Context, request *ai.ModelRequest) (*core.Operation[*ai.ModelResponse], error) {
		// Extract text prompt from the request
		prompt := extractTextFromRequest(request)
		if prompt == "" {
			return nil, fmt.Errorf("no text prompt found in request")
		}

		image := extractVeoImageFromRequest(request)

		videoConfig := toVeoParameters(request)

		operation, err := client.Models.GenerateVideos(
			ctx,
			name,
			prompt,
			image,
			&videoConfig,
		)
		if err != nil {
			return nil, fmt.Errorf("veo video generation failed: %w", err)
		}

		return fromVeoOperation(operation), nil
	}

	checkFunc := func(ctx context.Context, operation *core.Operation[*ai.ModelResponse]) (*core.Operation[*ai.ModelResponse], error) {
		veoOp, err := checkVeoOperation(ctx, client, operation)
		if err != nil {
			return nil, fmt.Errorf("veo operation status check failed: %w", err)
		}

		return fromVeoOperation(veoOp), nil
	}

	cancelFunc := func(ctx context.Context, operation *core.Operation[*ai.ModelResponse]) (*core.Operation[*ai.ModelResponse], error) {
		// Veo API doesn't currently support operation cancellation
		return nil, core.NewError(core.UNKNOWN, "veo model operation cancellation is not supported")
	}
	opts := ai.BackgroundModelOptions{
		ModelOptions: info,
		Start:        startFunc,
		Check:        checkFunc,
		Cancel:       cancelFunc,
	}
	return ai.NewBackgroundModel(name, &opts)
}

// extractTextFromRequest extracts the text prompt from a model request.
func extractTextFromRequest(request *ai.ModelRequest) string {
	if len(request.Messages) == 0 {
		return ""
	}
	for _, message := range request.Messages {
		for _, part := range message.Content {
			if part.Text != "" {
				return part.Text
			}
		}
	}
	return ""
}

// extractVeoImageFromRequest extracts image content from a model request for Veo.
func extractVeoImageFromRequest(request *ai.ModelRequest) *genai.Image {
	if len(request.Messages) == 0 {
		return nil
	}

	for _, message := range request.Messages {
		for _, part := range message.Content {
			if part.IsMedia() {
				_, data, err := uri.Data(part)
				if err != nil {
					return nil
				}
				return &genai.Image{
					ImageBytes: data,
					MIMEType:   part.ContentType}
			}
		}
	}

	return nil
}

// toVeoParameters converts model request configuration to Veo video generation parameters.
func toVeoParameters(request *ai.ModelRequest) genai.GenerateVideosConfig {
	params := genai.GenerateVideosConfig{}
	if request.Config != nil {
		if config, ok := request.Config.(*genai.GenerateVideosConfig); ok {
			return *config
		}
	}
	return params
}

// fromVeoOperation converts a Veo API operation to a Genkit core operation.
func fromVeoOperation(veoOp *genai.GenerateVideosOperation) *core.Operation[*ai.ModelResponse] {
	operation := &core.Operation[*ai.ModelResponse]{
		ID:   veoOp.Name,
		Done: veoOp.Done,
	}

	// Handle operation errors if present
	if veoOp.Error != nil {
		// veoOp.Error is a map[string]any, convert to string for the operation
		if errorMsg, ok := veoOp.Error["message"].(string); ok {
			operation.Error = fmt.Errorf("%s", errorMsg)
		} else {
			operation.Error = fmt.Errorf("operation error: %v", veoOp.Error)
		}
	}

	// Convert successful response with generated videos
	if veoOp.Response != nil && veoOp.Response.GeneratedVideos != nil {
		// Convert generated video samples to model response format
		content := make([]*ai.Part, 0, len(veoOp.Response.GeneratedVideos))
		for _, sample := range veoOp.Response.GeneratedVideos {
			if sample.Video != nil && sample.Video.URI != "" {
				content = append(content, ai.NewMediaPart("video/mp4", sample.Video.URI))
			}
		}

		if len(content) > 0 {
			response := &ai.ModelResponse{
				Message: &ai.Message{
					Role:    ai.RoleModel,
					Content: content,
				},
				FinishReason: ai.FinishReasonStop,
			}
			operation.Output = response
		}
	}

	return operation
}

// checkVeoOperation checks the status of a long-running Veo video generation operation..
func checkVeoOperation(ctx context.Context, client *genai.Client, ops *core.Operation[*ai.ModelResponse]) (*genai.GenerateVideosOperation, error) {
	genaiOps := &genai.GenerateVideosOperation{
		Name: ops.ID,
	}
	operation, err := client.Operations.GetVideosOperation(ctx, genaiOps, nil)
	return operation, err
}
