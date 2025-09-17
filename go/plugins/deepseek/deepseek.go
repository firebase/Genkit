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

package deepseek

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/internal"

	deepseek "github.com/cohesion-org/deepseek-go"
)

const (
	provider            = "deepseek"
	deepseekLabelPrefix = "DeepSeek"
)

type DeepSeek struct {
	APIKey string // DeepSeek API key, if not provided env var DEEPSEEK_API_KEY is looked up

	dsclient *deepseek.Client
	mu       sync.Mutex
	initted  bool
}

// Name returns the name of the plugin
func (d *DeepSeek) Name() string {
	return provider
}

// Init initializes the DeepSeek plugin and all its known models
func (d *DeepSeek) Init(ctx context.Context, g *genkit.Genkit) (err error) {
	if d == nil {
		d = &DeepSeek{}
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.initted {
		return errors.New("plugin already initialized")
	}
	defer func() {
		if err != nil {
			err = fmt.Errorf("DeepSeek.Init: %w", err)
		}
	}()

	apiKey := d.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("DEEPSEEK_API_KEY")
		if apiKey == "" {
			return fmt.Errorf("DeepSeek requires setting DEEPSEEK_API_KEY environment variable")
		}
	}

	dsc := deepseek.NewClient(apiKey)
	if dsc == nil {
		return errors.New("unable to create deepseek client")
	}
	d.dsclient = dsc
	d.initted = true

	return nil
}

// Model returns the [ai.Model] with the given name
func (d *DeepSeek) Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, provider, name)
}

// DefineModel defines an unknown model with the given name
func (d *DeepSeek) DefineModel(g *genkit.Genkit, name string, info ai.ModelInfo) (ai.Model, error) {
	return defineModel(g, *d.dsclient, provider, name, info), nil
}

// ListActions lists all resolvable actions by the plugin
func (d *DeepSeek) ListActions(ctx context.Context) []core.ActionDesc {
	actions := []core.ActionDesc{}
	models, err := listModels(ctx, *d.dsclient)
	if err != nil {
		return nil
	}

	for _, name := range models {
		metadata := map[string]any{
			"model": map[string]any{
				"supports": map[string]any{
					"media":       false, // official deepseek models do not support media content
					"multiturn":   true,
					"systemRole":  true,
					"tools":       true,
					"toolChoice":  true,
					"constrained": "no-tools",
				},
				"versions": []string{},
				"stage":    string(ai.ModelStageStable),
			},
		}
		metadata["label"] = fmt.Sprintf("%s - %s", deepseekLabelPrefix, name)
		actions = append(actions, core.ActionDesc{
			Type:     core.ActionTypeModel,
			Name:     fmt.Sprintf("%s/%s", provider, name),
			Key:      fmt.Sprintf("/%s/%s/%s", core.ActionTypeModel, provider, name),
			Metadata: metadata,
		})

	}

	return actions
}

// ResolveAction resolves all available actions from the plugin
func (d *DeepSeek) ResolveAction(g *genkit.Genkit, atype core.ActionType, name string) error {
	switch atype {
	case core.ActionTypeModel:
		defineModel(g, *d.dsclient, provider, name, ai.ModelInfo{
			Label:    fmt.Sprintf("%s - %s", deepseekLabelPrefix, name),
			Stage:    ai.ModelStageStable,
			Versions: []string{},
			Supports: &internal.Multimodal,
		})
	}
	return nil
}

// defineModel defines a model in the Genkit core
func defineModel(g *genkit.Genkit, client deepseek.Client, provider, name string, info ai.ModelInfo) ai.Model {
	return genkit.DefineModel(g, provider, name, &info, func(
		ctx context.Context,
		input *ai.ModelRequest,
		cb func(context.Context, *ai.ModelResponseChunk) error,
	) (*ai.ModelResponse, error) {
		return generate(ctx, client, name, input, cb)
	})
}

// listModels fetches the official deepseek models from the API
func listModels(ctx context.Context, client deepseek.Client) ([]string, error) {
	apiModels, err := deepseek.ListAllModels(&client, ctx)
	if err != nil {
		return nil, err
	}
	models := []string{}
	for _, m := range apiModels.Data {
		if m.Object != "model" && m.OwnedBy != "deepseek" {
			continue
		}
	}

	return models, nil
}

func generate(ctx context.Context, client deepseek.Client, model string, input *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	req, err := toDeepSeekRequest(input)
	if err != nil {
		return nil, err
	}

	req.Model = model

	// constrained generation
	hasOutput := input.Output != nil
	isJsonFormat := hasOutput && input.Output.Format == "json"
	isJsonContentType := hasOutput && input.Output.ContentType == "application/json"
	req.JSONMode = isJsonFormat || isJsonContentType
	if req.JSONMode {
		req.ResponseFormat = &deepseek.ResponseFormat{
			Type: "json_object",
		}
	}

	// no stream request
	if cb == nil {
		resp, err := client.CreateChatCompletion(ctx, req)
		if err != nil {
			return nil, err
		}
		r := translateResponse(resp)
		r.Request = input

		return r, nil
	}

	return nil, nil
}

func toDeepSeekRequest(input *ai.ModelRequest) (*deepseek.ChatCompletionRequest, error) {
	req, err := configFromRequest(input)
	if err != nil {
		return nil, err
	}

	// Genkit primitive fields must be used instead of deepseek config fields
	// i.e.: system prompt, tools, cached content, response schema
	if req.Model != "" {
		return nil, errors.New("model must be set using Genkit feature: ai.WithModelName() or ai.WithModel()")
	}
	if req.Messages != nil {
		return nil, errors.New("messages must be set using Genkit feature: ai.WithMessages()")
	}
	if req.ResponseFormat != nil {
		return nil, errors.New("response format must be set using Genkit feature: ai.WithOutputType()")
	}
	if req.ToolChoice != nil {
		return nil, errors.New("tool choice must be set using Genkit feature: ai.WithToolChoice()")
	}
	if req.Tools != nil {
		return nil, errors.New("tools must be set using Genkit feature: ai.WithTools()")
	}

	messages := []deepseek.ChatCompletionMessage{}
	for _, m := range input.Messages {
		if m.Role == ai.RoleSystem {
			messages = append(messages, deepseek.ChatCompletionMessage{
				Role:    toDeepSeekRole(m.Role),
				Content: m.Text(),
			})
		} else if m.Content[len(m.Content)-1].IsToolResponse() {
			parts, err := toDeepSeekParts(m.Content, m.Role)
			if err != nil {
				return nil, err
			}
			messages = append(messages, parts...)
		} else {
			parts, err := toDeepSeekParts(m.Content, m.Role)
			if err != nil {
				return nil, err
			}
			messages = append(messages, parts...)
		}
	}
	req.Messages = messages

	if len(input.Tools) > 0 {
		tools, err := toDeepSeekTool(input.Tools)
		if err != nil {
			return nil, err
		}
		req.Tools = tools

		tc := toDeepSeekToolChoice(input.ToolChoice)
		req.ToolChoice = tc
	}

	return req, nil
}

// toDeepSeekRole translates an [ai.Role] to a DeepSeek role
func toDeepSeekRole(role ai.Role) string {
	var r string
	switch role {
	case ai.RoleSystem:
		r = deepseek.ChatMessageRoleSystem
	case ai.RoleModel:
		r = deepseek.ChatMessageRoleAssistant
	case ai.RoleTool:
		r = deepseek.ChatMessageRoleTool
	default:
		r = deepseek.ChatMessageRoleUser
	}
	return r
}

// toDeepSeekParts translates an array of [ai.Part] to [deepseek.ChatCompletionMessage]
func toDeepSeekParts(parts []*ai.Part, role ai.Role) ([]deepseek.ChatCompletionMessage, error) {
	res := make([]deepseek.ChatCompletionMessage, 0, len(parts))
	for _, p := range parts {
		part, err := toDeepSeekPart(p, role)
		if err != nil {
			return nil, err
		}
		res = append(res, part)
	}

	return res, nil
}

// toDeepSeekParts translates an [ai.Part] to [deepseek.ChatCompletionMessage]
func toDeepSeekPart(p *ai.Part, r ai.Role) (deepseek.ChatCompletionMessage, error) {
	role := toDeepSeekRole(r)
	switch {
	case p.IsReasoning():
		content := ""
		if p.Metadata != nil {
			if c, ok := p.Metadata["content"].(string); ok {
				content = c
			}
		}
		return deepseek.ChatCompletionMessage{
			Role:             toDeepSeekRole(r),
			Content:          content,
			ReasoningContent: p.Text,
		}, nil
	case p.IsText():
		return deepseek.ChatCompletionMessage{
			Content: p.Text,
			Role:    role,
		}, nil
	default:
		panic("unknown part type in the request")
	}
}

// toDeepSeekTool translates a slice of [ai.ToolDefinition] to a slice of [deepseek.Tool]
func toDeepSeekTool(inTools []*ai.ToolDefinition) ([]deepseek.Tool, error) {
	tools := []deepseek.Tool{}
	for _, t := range inTools {
		tools = append(tools, deepseek.Tool{
			Type: "function",
			Function: deepseek.Function{
				Name:        t.Name,
				Description: t.Description,
				Parameters: &deepseek.FunctionParameters{
					Type:       "object",
					Properties: t.InputSchema,
				},
			},
		})
	}
	return tools, nil
}

// toDeepSeekToolChoice translates an [ai.ToolChoice] to a DeepSeek Tool choice
func toDeepSeekToolChoice(choice ai.ToolChoice) string {
	switch choice {
	case ai.ToolChoiceAuto:
		return "auto"
	case ai.ToolChoiceNone:
		return "none"
	case ai.ToolChoiceRequired:
		return "required"
	}
	return "none"
}

// translateResponse translates a [deepseek.ChatCompletionResponse] to an [ai.ModelResponse]
func translateResponse(resp *deepseek.ChatCompletionResponse) *ai.ModelResponse {
	var r *ai.ModelResponse
	if len(resp.Choices) > 0 {
		r = translateCandidate(resp.Choices[0])
	} else {
		r = &ai.ModelResponse{}
	}

	if r.Usage == nil {
		r.Usage = &ai.GenerationUsage{}
	}

	r.Usage.InputTokens = resp.Usage.PromptTokens
	r.Usage.CachedContentTokens = resp.Usage.PromptCacheHitTokens
	r.Usage.OutputTokens = resp.Usage.CompletionTokens
	r.Usage.TotalTokens = resp.Usage.TotalTokens

	return r
}

// translateCandidate translates a [deepseek.Choice] to an [ai.ModelResponse]
func translateCandidate(cand deepseek.Choice) *ai.ModelResponse {
	m := &ai.ModelResponse{}
	switch cand.FinishReason {
	case "stop":
		m.FinishReason = ai.FinishReasonStop
	case "length":
		m.FinishReason = ai.FinishReasonLength
	case "content_filter":
		m.FinishReason = ai.FinishReasonBlocked
	case "tool_calls":
		m.FinishReason = ai.FinishReasonOther
	case "insufficient_system_resource":
		m.FinishReason = ai.FinishReasonOther
	default:
		m.FinishReason = ai.FinishReasonOther
	}

	msg := &ai.Message{}
	msg.Role = ai.Role(cand.Message.Role)

	var p *ai.Part
	// there's only one part in [deepseek.Choice], it could be either text or reasoning
	if cand.Message.ReasoningContent != "" {
		p = ai.NewReasoningPart(cand.Message.ReasoningContent, map[string]any{
			"content": cand.Message.Content,
		})
	} else {
		p = ai.NewTextPart(cand.Message.Content)
	}

	msg.Content = append(msg.Content, p)
	m.Message = msg
	return m
}

// configFromRequest ensures a valid DeepSeek configuration is used
func configFromRequest(input *ai.ModelRequest) (*deepseek.ChatCompletionRequest, error) {
	var config deepseek.ChatCompletionRequest

	switch cfg := input.Config.(type) {
	case deepseek.ChatCompletionRequest:
		config = cfg
	case *deepseek.ChatCompletionRequest:
		config = *cfg
	case map[string]any:
		if err := internal.MapToStruct(cfg, config); err != nil {
			return nil, err
		}
	case nil:
	default:
		return nil, fmt.Errorf("unexpected config type: %T", input.Config)
	}

	return &config, nil
}
