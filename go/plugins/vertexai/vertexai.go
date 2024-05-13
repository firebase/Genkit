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

	"cloud.google.com/go/vertexai/genai"
	"github.com/google/genkit/go/ai"
	"github.com/google/genkit/go/genkit"
)

func newClient(ctx context.Context, projectID, location string) (*genai.Client, error) {
	return genai.NewClient(ctx, projectID, location)
}

type generator struct {
	model  string
	client *genai.Client
}

func (g *generator) Generate(ctx context.Context, input *ai.GenerateRequest, cb genkit.StreamingCallback[*ai.Candidate]) (*ai.GenerateResponse, error) {
	if cb != nil {
		panic("streaming not supported yet") // TODO: streaming
	}
	gm := g.client.GenerativeModel(g.model)

	// Translate from a ai.GenerateRequest to a genai request.
	gm.SetCandidateCount(int32(input.Candidates))
	if c, ok := input.Config.(*ai.GenerationCommonConfig); ok {
		gm.SetMaxOutputTokens(int32(c.MaxOutputTokens))
		gm.StopSequences = c.StopSequences
		gm.SetTemperature(float32(c.Temperature))
		gm.SetTopK(float32(c.TopK))
		gm.SetTopP(float32(c.TopP))
	}

	// Start a "chat".
	cs := gm.StartChat()

	// All but the last message goes in the history field.
	messages := input.Messages
	for len(messages) > 1 {
		m := messages[0]
		messages = messages[1:]
		cs.History = append(cs.History, &genai.Content{
			Parts: convertParts(m.Content),
			Role:  string(m.Role),
		})
	}
	// The last message gets added to the parts slice.
	var parts []genai.Part
	if len(messages) > 0 {
		parts = convertParts(messages[0].Content)
	}
	//TODO: convert input.Tools and append to gm.Tools

	// Send out the actual request.
	resp, err := cs.SendMessage(ctx, parts...)
	if err != nil {
		return nil, err
	}

	// Translate from a genai.GenerateContentResponse to a ai.GenerateResponse.
	r := &ai.GenerateResponse{
		Request: input,
	}
	for _, cand := range resp.Candidates {
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
				p = ai.NewBlobPart(part.MIMEType, string(part.Data))
			case genai.FunctionResponse:
				p = ai.NewBlobPart("TODO", string(part.Name))
			default:
				panic("unknown part type")
			}
			m.Content = append(m.Content, p)
		}
		c.Message = m
		r.Candidates = append(r.Candidates, c)
	}
	return r, nil
}

// NewGenerator returns an action which sends a request to
// the vertex AI model and returns the response.
func NewGenerator(ctx context.Context, model, projectID, location string) (ai.Generator, error) {
	client, err := newClient(ctx, projectID, location)
	if err != nil {
		return nil, err
	}
	return &generator{
		model:  model,
		client: client,
	}, nil
}

// Init registers all the actions in this package with ai.
func Init(ctx context.Context, model, projectID, location string) error {
	g, err := NewGenerator(ctx, model, projectID, location)
	if err != nil {
		return err
	}
	ai.RegisterGenerator("google-vertexai", model, &ai.GeneratorMetadata{
		Label: "Vertex AI - " + model,
		Supports: ai.GeneratorCapabilities{
			Multiturn: true,
		},
	}, g)

	return nil
}

// convertParts converts a slice of *ai.Part to a slice of genai.Part.
func convertParts(parts []*ai.Part) []genai.Part {
	res := make([]genai.Part, 0, len(parts))
	for _, p := range parts {
		res = append(res, convertPart(p))
	}
	return res
}

// convertPart converts a *ai.Part to a genai.Part.
func convertPart(p *ai.Part) genai.Part {
	switch {
	case p.IsPlainText():
		return genai.Text(p.Text())
	default:
		return genai.Blob{MIMEType: p.ContentType(), Data: []byte(p.Text())}
	}
}
