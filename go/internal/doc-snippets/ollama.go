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
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/ollama"
)

func ollamaEx(ctx context.Context) error {
	g, err := genkit.New(nil)
	if err != nil {
		log.Fatal(err)
	}

	// [START init]
	// Init with Ollama's default local address.
	if err := ollama.Init(ctx, &ollama.Config{
		ServerAddress: "http://127.0.0.1:11434",
	}); err != nil {
		return err
	}
	// [END init]

	// [START definemodel]
	model := ollama.DefineModel(
		g,
		ollama.ModelDefinition{
			Name: "gemma2",
			Type: "chat", // "chat" or "generate"
		},
		&ai.ModelCapabilities{
			Multiturn:  true,
			SystemRole: true,
			Tools:      false,
			Media:      false,
		},
	)
	// [END definemodel]

	// [START gen]
	text, err := genkit.GenerateText(ctx, g,
		ai.WithModel(model),
		ai.WithTextPrompt("Tell me a joke."))
	if err != nil {
		return err
	}
	// [END gen]

	_ = text

	return nil
}
