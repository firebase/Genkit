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

package snippets

import (
	"context"
	"encoding/base64"
	"fmt"
	"os"
)

// !+import
import "github.com/firebase/genkit/go/ai"
import "github.com/firebase/genkit/go/plugins/vertexai"

// !-import

// Globals for simplification only.
// Bad style: don't do this.
var ctx = context.Background()
var gemini15pro *ai.Model

func m1() error {
	// !+init
	projectID := os.Getenv("GCLOUD_PROJECT")
	if err := vertexai.Init(ctx, projectID, "us-central1"); err != nil {
		return err
	}
	// !-init
	_ = projectID

	// !+model
	gemini15pro := vertexai.Model("gemini-1.5-pro")
	// !-model

	// !+call
	request := ai.GenerateRequest{Messages: []*ai.Message{
		{Content: []*ai.Part{ai.NewTextPart("Tell me a joke.")}},
	}}
	response, err := gemini15pro.Generate(ctx, &request, nil)
	if err != nil {
		return err
	}

	responseText, err := response.Text()
	if err != nil {
		return err
	}
	fmt.Println(responseText)
	// !-call
	return nil
}

func opts() error {
	// !+options
	request := ai.GenerateRequest{
		Messages: []*ai.Message{
			{Content: []*ai.Part{ai.NewTextPart("Tell me a joke about dogs.")}},
		},
		Config: ai.GenerationCommonConfig{
			Temperature:     1.67,
			StopSequences:   []string{"abc"},
			MaxOutputTokens: 3,
		},
	}
	// !-options
	_ = request
	return nil
}

func streaming() error {
	// !+streaming
	request := ai.GenerateRequest{Messages: []*ai.Message{
		{Content: []*ai.Part{ai.NewTextPart("Tell a long story about robots and ninjas.")}},
	}}
	response, err := gemini15pro.Generate(
		ctx,
		&request,
		func(ctx context.Context, grc *ai.GenerateResponseChunk) error {
			text, err := grc.Text()
			if err != nil {
				return err
			}
			fmt.Printf("Chunk: %s\n", text)
			return nil
		})
	if err != nil {
		return err
	}

	// You can also still get the full response.
	responseText, err := response.Text()
	if err != nil {
		return err
	}
	fmt.Println(responseText)

	// !-streaming
	return nil
}

func multi() error {
	// !+multimodal
	imageBytes, err := os.ReadFile("img.jpg")
	if err != nil {
		return err
	}
	encodedImage := base64.StdEncoding.EncodeToString(imageBytes)

	request := ai.GenerateRequest{Messages: []*ai.Message{
		{Content: []*ai.Part{
			ai.NewTextPart("Describe the following image."),
			ai.NewMediaPart("", "data:image/jpeg;base64,"+encodedImage),
		}},
	}}
	gemini15pro.Generate(ctx, &request, nil)
	// !-multimodal
	return nil
}

func tools() error {
	// !+tools
	myJoke := &ai.ToolDefinition{
		Name:        "myJoke",
		Description: "useful when you need a joke to tell",
		InputSchema: make(map[string]any),
		OutputSchema: map[string]any{
			"joke": "string",
		},
	}
	ai.DefineTool(
		myJoke,
		nil,
		func(ctx context.Context, input map[string]any) (map[string]any, error) {
			return map[string]any{"joke": "haha Just kidding no joke! got you"}, nil
		},
	)

	request := ai.GenerateRequest{
		Messages: []*ai.Message{
			{Content: []*ai.Part{ai.NewTextPart("Tell me a joke.")},
				Role: ai.RoleUser},
		},
		Tools: []*ai.ToolDefinition{myJoke},
	}
	response, err := gemini15pro.Generate(ctx, &request, nil)
	// !-tools
	_ = response
	return err
}

func history() error {
	var prompt string
	// !+hist1
	history := []*ai.Message{{
		Content: []*ai.Part{ai.NewTextPart(prompt)},
		Role:    ai.RoleUser,
	}}

	request := ai.GenerateRequest{Messages: history}
	response, err := gemini15pro.Generate(context.Background(), &request, nil)
	// !-hist1
	_ = err
	// !+hist2
	history = append(history, response.Candidates[0].Message)
	// !-hist2

	// !+hist3
	history = append(history, &ai.Message{
		Content: []*ai.Part{ai.NewTextPart(prompt)},
		Role:    ai.RoleUser,
	})

	request = ai.GenerateRequest{Messages: history}
	response, err = gemini15pro.Generate(ctx, &request, nil)
	// !-hist3
	// !+hist4
	history = []*ai.Message{{
		Content: []*ai.Part{ai.NewTextPart("Talk like a pirate.")},
		Role:    ai.RoleSystem,
	}}
	// !-hist4
	return nil
}
