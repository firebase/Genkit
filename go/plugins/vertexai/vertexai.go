// Copyright 2024 Google LLC
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

package vertexai

import (
	"context"
	"errors"
	"fmt"
	"runtime"

	aiplatform "cloud.google.com/go/aiplatform/apiv1"
	"cloud.google.com/go/vertexai/genai"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/internal/uri"
	"google.golang.org/api/option"
)

const provider = "vertexai"

// Config provides configuration options for the Init function.
type Config struct {
	// The project holding the resources.
	ProjectID string
	// The location of the resources.
	// Defaults to "us-central1".
	Location string
	// Generative models to provide.
	Models []string
	// Embedding models to provide.
	Embedders []string
}

func Init(ctx context.Context, cfg Config) (err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("vertexai.Init: %w", err)
		}
	}()

	if cfg.ProjectID == "" {
		return errors.New("missing ProjectID")
	}
	if cfg.Location == "" {
		cfg.Location = "us-central1"
	}
	// TODO(#345): call ListModels. See googleai.go.
	if len(cfg.Models) == 0 && len(cfg.Embedders) == 0 {
		return errors.New("need at least one model or embedder")
	}

	if err := initModels(ctx, cfg); err != nil {
		return err
	}
	return initEmbedders(ctx, cfg)
}

func initModels(ctx context.Context, cfg Config) error {
	// Client for Gemini SDK.
	gclient, err := genai.NewClient(ctx, cfg.ProjectID, cfg.Location)
	if err != nil {
		return err
	}
	for _, name := range cfg.Models {
		meta := &ai.ModelMetadata{
			Label: "Vertex AI - " + name,
			Supports: ai.ModelCapabilities{
				Multiturn: true,
			},
		}
		g := &generator{model: name, client: gclient}
		ai.DefineModel(provider, name, meta, g.generate)
	}
	return nil
}

func initEmbedders(ctx context.Context, cfg Config) error {
	// Client for prediction SDK.
	endpoint := fmt.Sprintf("%s-aiplatform.googleapis.com:443", cfg.Location)
	numConns := max(runtime.GOMAXPROCS(0), 4)
	o := []option.ClientOption{
		option.WithEndpoint(endpoint),
		option.WithGRPCConnectionPool(numConns),
	}

	pclient, err := aiplatform.NewPredictionClient(ctx, o...)
	if err != nil {
		return err
	}
	for _, name := range cfg.Embedders {
		fullName := fmt.Sprintf("projects/%s/locations/%s/publishers/google/models/%s", cfg.ProjectID, cfg.Location, name)
		ai.DefineEmbedder(provider, name, func(ctx context.Context, req *ai.EmbedRequest) ([]float32, error) {
			return embed(ctx, fullName, pclient, req)
		})
	}
	return nil
}

// Model returns the [ai.ModelAction] with the given name.
// It returns nil if the model was not configured.
func Model(name string) *ai.ModelAction {
	return ai.LookupModel(provider, name)
}

// Embedder returns the [ai.EmbedderAction] with the given name.
// It returns nil if the embedder was not configured.
func Embedder(name string) *ai.EmbedderAction {
	return ai.LookupEmbedder(provider, name)
}

type generator struct {
	model  string
	client *genai.Client
}

func (g *generator) generate(ctx context.Context, input *ai.GenerateRequest, cb func(context.Context, *ai.GenerateResponseChunk) error) (*ai.GenerateResponse, error) {
	if cb != nil {
		panic("streaming not supported yet") // TODO: streaming
	}
	gm := g.client.GenerativeModel(g.model)

	// Translate from a ai.GenerateRequest to a genai request.
	gm.SetCandidateCount(int32(input.Candidates))
	if c, ok := input.Config.(*ai.GenerationCommonConfig); ok && c != nil {
		if c.MaxOutputTokens != 0 {
			gm.SetMaxOutputTokens(int32(c.MaxOutputTokens))
		}
		if len(c.StopSequences) > 0 {
			gm.StopSequences = c.StopSequences
		}
		if c.Temperature != 0 {
			gm.SetTemperature(float32(c.Temperature))
		}
		if c.TopK != 0 {
			gm.SetTopK(float32(c.TopK))
		}
		if c.TopP != 0 {
			gm.SetTopP(float32(c.TopP))
		}
	}

	// Start a "chat".
	cs := gm.StartChat()

	// All but the last message goes in the history field.
	messages := input.Messages
	for len(messages) > 1 {
		m := messages[0]
		messages = messages[1:]
		parts, err := convertParts(m.Content)
		if err != nil {
			return nil, err
		}
		cs.History = append(cs.History, &genai.Content{
			Parts: parts,
			Role:  string(m.Role),
		})
	}
	// The last message gets added to the parts slice.
	var parts []genai.Part
	if len(messages) > 0 {
		var err error
		parts, err = convertParts(messages[0].Content)
		if err != nil {
			return nil, err
		}
	}

	// Convert input.Tools and append to gm.Tools.
	for _, t := range input.Tools {
		schema := &genai.Schema{
			Type:       genai.TypeObject,
			Properties: make(map[string]*genai.Schema),
		}
		for k, v := range t.InputSchema {
			typ := genai.TypeUnspecified
			switch v {
			case "string":
				typ = genai.TypeString
			case "float64":
				typ = genai.TypeNumber
			case "int":
				typ = genai.TypeInteger
			case "bool":
				typ = genai.TypeBoolean
			default:
				return nil, fmt.Errorf("schema value %q not supported", v)
			}
			schema.Properties[k] = &genai.Schema{Type: typ}
		}

		fd := &genai.FunctionDeclaration{
			Name:        t.Name,
			Parameters:  schema,
			Description: t.Description,
		}

		gm.Tools = append(gm.Tools, &genai.Tool{
			FunctionDeclarations: []*genai.FunctionDeclaration{fd},
		})
	}
	// TODO: gm.ToolConfig?

	// Send out the actual request.
	resp, err := cs.SendMessage(ctx, parts...)
	if err != nil {
		return nil, err
	}

	r := translateResponse(resp)
	r.Request = input
	return r, nil
}

// translateCandidate translates from a genai.GenerateContentResponse to an ai.GenerateResponse.
func translateCandidate(cand *genai.Candidate) *ai.Candidate {
	c := &ai.Candidate{}
	c.Index = int(cand.Index)
	switch cand.FinishReason {
	case genai.FinishReasonStop:
		c.FinishReason = ai.FinishReasonStop
	case genai.FinishReasonMaxTokens:
		c.FinishReason = ai.FinishReasonLength
	case genai.FinishReasonSafety:
		c.FinishReason = ai.FinishReasonBlocked
	case genai.FinishReasonRecitation:
		c.FinishReason = ai.FinishReasonBlocked
	case genai.FinishReasonOther:
		c.FinishReason = ai.FinishReasonOther
	default: // Unspecified
		c.FinishReason = ai.FinishReasonUnknown
	}
	m := &ai.Message{}
	m.Role = ai.Role(cand.Content.Role)
	for _, part := range cand.Content.Parts {
		var p *ai.Part
		switch part := part.(type) {
		case genai.Text:
			p = ai.NewTextPart(string(part))
		case genai.Blob:
			p = ai.NewMediaPart(part.MIMEType, string(part.Data))
		case genai.FunctionCall:
			p = ai.NewToolRequestPart(&ai.ToolRequest{
				Name:  part.Name,
				Input: part.Args,
			})
		default:
			panic(fmt.Sprintf("unknown part %#v", part))
		}
		m.Content = append(m.Content, p)
	}
	c.Message = m
	return c
}

// Translate from a genai.GenerateContentResponse to a ai.GenerateResponse.
func translateResponse(resp *genai.GenerateContentResponse) *ai.GenerateResponse {
	r := &ai.GenerateResponse{}
	for _, c := range resp.Candidates {
		r.Candidates = append(r.Candidates, translateCandidate(c))
	}
	return r
}

// convertParts converts a slice of *ai.Part to a slice of genai.Part.
func convertParts(parts []*ai.Part) ([]genai.Part, error) {
	res := make([]genai.Part, 0, len(parts))
	for _, p := range parts {
		part, err := convertPart(p)
		if err != nil {
			return nil, err
		}
		res = append(res, part)
	}
	return res, nil
}

// convertPart converts a *ai.Part to a genai.Part.
func convertPart(p *ai.Part) (genai.Part, error) {
	switch {
	case p.IsText():
		return genai.Text(p.Text), nil
	case p.IsMedia():
		contentType, data, err := uri.Data(p)
		if err != nil {
			return nil, err
		}
		return genai.Blob{MIMEType: contentType, Data: data}, nil
	case p.IsData():
		panic("vertexai does not support Data parts")
	case p.IsToolResponse():
		toolResp := p.ToolResponse
		fr := genai.FunctionResponse{
			Name:     toolResp.Name,
			Response: toolResp.Output,
		}
		return fr, nil
	case p.IsToolRequest():
		toolReq := p.ToolRequest
		fc := genai.FunctionCall{
			Name: toolReq.Name,
			Args: toolReq.Input,
		}
		return fc, nil
	default:
		panic("unknown part type in a request")
	}
}
