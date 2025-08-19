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

package anthropic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/internal/uri"
	"github.com/invopop/jsonschema"

	"github.com/anthropics/anthropic-sdk-go"
)

const (
	ToolNameRegex = `^[a-zA-Z0-9_-]{1,64}$`
)

func DefineModel(g *genkit.Genkit, client anthropic.Client, provider, name string, info ai.ModelInfo) ai.Model {
	label := "Anthropic"
	if provider == "vertexai" {
		label = "Vertex AI"
	}
	meta := &ai.ModelInfo{
		Label:    label + "-" + name,
		Supports: info.Supports,
		Versions: info.Versions,
	}
	return genkit.DefineModel(g, provider, name, meta, func(
		ctx context.Context,
		input *ai.ModelRequest,
		cb func(context.Context, *ai.ModelResponseChunk) error,
	) (*ai.ModelResponse, error) {
		return generate(ctx, client, name, input, cb)
	})
}

// Generate function defines how a generate request is done in Anthropic models
func generate(
	ctx context.Context,
	client anthropic.Client,
	model string,
	input *ai.ModelRequest,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	req, err := toAnthropicRequest(model, input)
	if err != nil {
		return nil, fmt.Errorf("unable to generate anthropic request: %w", err)
	}

	// no streaming
	if cb == nil {
		msg, err := client.Messages.New(ctx, *req)
		if err != nil {
			return nil, err
		}

		r, err := toGenkitResponse(msg)
		if err != nil {
			return nil, err
		}

		r.Request = input
		return r, nil
	} else {
		stream := client.Messages.NewStreaming(ctx, *req)
		message := anthropic.Message{}
		for stream.Next() {
			event := stream.Current()
			err := message.Accumulate(event)
			if err != nil {
				return nil, err
			}

			content := []*ai.Part{}
			switch event := event.AsAny().(type) {
			case anthropic.ContentBlockDeltaEvent:
				content = append(content, ai.NewTextPart(event.Delta.Text))
				cb(ctx, &ai.ModelResponseChunk{
					Content: content,
				})
			case anthropic.MessageStopEvent:
				r, err := toGenkitResponse(&message)
				if err != nil {
					return nil, err
				}
				r.Request = input
				return r, nil
			}
		}
		if stream.Err() != nil {
			return nil, stream.Err()
		}
	}

	return nil, nil
}

func toAnthropicRole(role ai.Role) (anthropic.MessageParamRole, error) {
	switch role {
	case ai.RoleUser:
		return anthropic.MessageParamRoleUser, nil
	case ai.RoleModel:
		return anthropic.MessageParamRoleAssistant, nil
	case ai.RoleTool:
		return anthropic.MessageParamRoleAssistant, nil
	default:
		return "", fmt.Errorf("unknown role given: %q", role)
	}
}

// toAnthropicRequest translates [ai.ModelRequest] to an Anthropic request
func toAnthropicRequest(model string, i *ai.ModelRequest) (*anthropic.MessageNewParams, error) {
	messages := make([]anthropic.MessageParam, 0)

	req, err := configFromRequest(i)
	if err != nil {
		return nil, err
	}

	if req.Model == "" {
		return nil, errors.New("anthropic model not provided in request")
	}
	// configure system prompt (if given)
	sysBlocks := []anthropic.TextBlockParam{}
	for _, message := range i.Messages {
		if message.Role == ai.RoleSystem {
			// only text is supported for system messages
			sysBlocks = append(sysBlocks, anthropic.TextBlockParam{Text: message.Text()})
		} else if message.Content[len(message.Content)-1].IsToolResponse() {
			// if the last message is a ToolResponse, the conversation must continue
			// and the ToolResponse message must be sent as a user
			// see: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#handling-tool-use-and-tool-result-content-blocks
			parts, err := toAnthropicParts(message.Content)
			if err != nil {
				return nil, err
			}
			messages = append(messages, anthropic.NewUserMessage(parts...))
		} else {
			parts, err := toAnthropicParts(message.Content)
			if err != nil {
				return nil, err
			}
			role, err := toAnthropicRole(message.Role)
			if err != nil {
				return nil, err
			}
			messages = append(messages, anthropic.MessageParam{
				Role:    role,
				Content: parts,
			})
		}
	}

	req.System = sysBlocks
	req.Messages = messages

	tools, err := toAnthropicTools(i.Tools)
	if err != nil {
		return nil, err
	}
	req.Tools = tools

	return req, nil
}

// mapToStruct unmarshals a map[String]any to the expected type
func mapToStruct(m map[string]any, v any) error {
	jsonData, err := json.Marshal(m)
	if err != nil {
		return err
	}
	return json.Unmarshal(jsonData, v)
}

// configFromRequest converts any supported config type to [anthropic.MessageNewParams]
func configFromRequest(input *ai.ModelRequest) (*anthropic.MessageNewParams, error) {
	var result anthropic.MessageNewParams

	switch config := input.Config.(type) {
	case anthropic.MessageNewParams:
		result = config
	case *anthropic.MessageNewParams:
		result = *config
	case map[string]any:
		if err := mapToStruct(config, &result); err != nil {
			return nil, err
		}
	case nil:
		// Empty configuration is considered valid
	default:
		return nil, fmt.Errorf("unexpected config type: %T", input.Config)
	}
	return &result, nil
}

// toAnthropicTools translates [ai.ToolDefinition] to an anthropic.ToolParam type
func toAnthropicTools(tools []*ai.ToolDefinition) ([]anthropic.ToolUnionParam, error) {
	resp := make([]anthropic.ToolUnionParam, 0)
	regex := regexp.MustCompile(ToolNameRegex)

	for _, t := range tools {
		if t.Name == "" {
			return nil, fmt.Errorf("tool name is required")
		}
		if !regex.MatchString(t.Name) {
			return nil, fmt.Errorf("tool name must match regex: %s", ToolNameRegex)
		}

		resp = append(resp, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        t.Name,
				Description: anthropic.String(t.Description),
				InputSchema: toAnthropicSchema[map[string]any](),
			},
		})
	}

	return resp, nil
}

// toAnthropicSchema generates a JSON schema for the requested input type
func toAnthropicSchema[T any]() anthropic.ToolInputSchemaParam {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: true,
		DoNotReference:            true,
	}
	var v T
	schema := reflector.Reflect(v)
	return anthropic.ToolInputSchemaParam{
		Properties: schema.Properties,
	}
}

// toAnthropicParts translates [ai.Part] to an anthropic.ContentBlockParamUnion type
func toAnthropicParts(parts []*ai.Part) ([]anthropic.ContentBlockParamUnion, error) {
	blocks := []anthropic.ContentBlockParamUnion{}

	for _, p := range parts {
		switch {
		case p.IsText():
			blocks = append(blocks, anthropic.NewTextBlock(p.Text))
		case p.IsMedia():
			contentType, data, _ := uri.Data(p)
			blocks = append(blocks, anthropic.NewImageBlockBase64(contentType, base64.StdEncoding.EncodeToString(data)))
		case p.IsData():
			contentType, data, _ := uri.Data(p)
			blocks = append(blocks, anthropic.NewImageBlockBase64(contentType, base64.RawStdEncoding.EncodeToString(data)))
		case p.IsToolRequest():
			toolReq := p.ToolRequest
			blocks = append(blocks, anthropic.NewToolUseBlock(toolReq.Ref, toolReq.Input, toolReq.Name))
		case p.IsToolResponse():
			toolResp := p.ToolResponse
			output, err := json.Marshal(toolResp.Output)
			if err != nil {
				return nil, fmt.Errorf("unable to parse tool response, err: %w", err)
			}
			blocks = append(blocks, anthropic.NewToolResultBlock(toolResp.Ref, string(output), false))
		case p.IsReasoning():
			signature := []byte{}
			if p.Metadata != nil {
				if sig, ok := p.Metadata["signature"].([]byte); ok {
					signature = sig
				}
			}
			blocks = append(blocks, anthropic.NewThinkingBlock(string(signature), p.Text))
		default:
			return nil, errors.New("unknown part type in the request")
		}
	}

	return blocks, nil
}

// toGenkitResponse translates an Anthropic Message to [ai.ModelResponse]
func toGenkitResponse(m *anthropic.Message) (*ai.ModelResponse, error) {
	r := ai.ModelResponse{}

	switch m.StopReason {
	case anthropic.StopReasonMaxTokens:
		r.FinishReason = ai.FinishReasonLength
	case anthropic.StopReasonStopSequence:
		r.FinishReason = ai.FinishReasonStop
	case anthropic.StopReasonEndTurn:
		r.FinishReason = ai.FinishReasonStop
	case anthropic.StopReasonToolUse:
		r.FinishReason = ai.FinishReasonStop
	default:
		r.FinishReason = ai.FinishReasonUnknown
	}

	msg := &ai.Message{}
	msg.Role = ai.RoleModel
	for _, part := range m.Content {
		var p *ai.Part
		switch part.AsAny().(type) {
		case anthropic.ThinkingBlock:
			p = ai.NewReasoningPart(part.Text, []byte(part.Signature))
		case anthropic.TextBlock:
			p = ai.NewTextPart(string(part.Text))
		case anthropic.ToolUseBlock:
			p = ai.NewToolRequestPart(&ai.ToolRequest{
				Ref:   part.ID,
				Input: part.Input,
				Name:  part.Name,
			})
		default:
			return nil, fmt.Errorf("unknown part: %#v", part)
		}
		msg.Content = append(msg.Content, p)
	}

	r.Message = msg
	r.Usage = &ai.GenerationUsage{
		InputTokens:  int(m.Usage.InputTokens),
		OutputTokens: int(m.Usage.OutputTokens),
	}
	return &r, nil
}
