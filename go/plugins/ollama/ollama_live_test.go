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

package ollama_test

import (
	"context"
	"flag"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	ollamaPlugin "github.com/firebase/genkit/go/plugins/ollama"
)

var (
	serverAddress = flag.String("server-address", "http://localhost:11434", "Ollama server address")
	modelName     = flag.String("model-name", "tinyllama", "model name")
	testLive      = flag.Bool("test-live", false, "run live tests")
)

/*
To run this test, you need to have the Ollama server running. You can set the server address using the OLLAMA_SERVER_ADDRESS environment variable.
If the environment variable is not set, the test will default to http://localhost:11434 (the default address for the Ollama server).
*/
func TestLive(t *testing.T) {
	if !*testLive {
		t.Skip("skipping go/plugins/ollama live test")
	}

	ctx := context.Background()

	o := &ollamaPlugin.Ollama{ServerAddress: *serverAddress}
	g := genkit.Init(ctx, genkit.WithPlugins(o))

	// Define the model
	o.DefineModel(g, ollamaPlugin.ModelDefinition{Name: *modelName}, nil)

	// Use the Ollama model
	m := ollamaPlugin.Model(g, *modelName)
	if m == nil {
		t.Fatalf(`failed to find model: %s`, *modelName)
	}

	// Generate a response from the model
	resp, err := genkit.Generate(ctx, g,
		ai.WithModel(m),
		ai.WithConfig(&ai.GenerationCommonConfig{Temperature: 1}),
		ai.WithPrompt("I'm hungry, what should I eat?"),
	)
	if err != nil {
		t.Fatalf("failed to generate response: %s", err)
	}

	if resp == nil {
		t.Fatalf("response is nil")
	}

	// Get the text from the response
	text := resp.Text()
	// log.Println("Response:", text)

	// Assert that the response text is as expected
	if text == "" {
		t.Fatalf("expected non-empty response, got: %s", text)
	}
}
