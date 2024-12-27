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

// [START main]
package main

import (
	"context"
	"log"

	// Import Genkit and the Google AI plugin
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/googleai"
)

func main() {
	ctx := context.Background()

	// Initialize the Google AI plugin. When you pass nil for the
	// Config parameter, the Google AI plugin will get the API key from the
	// GOOGLE_GENAI_API_KEY environment variable, which is the recommended
	// practice.
	if err := googleai.Init(ctx, nil); err != nil {
		log.Fatal(err)
	}

	// The Google AI API provides access to several generative models. Here,
	// we specify gemini-1.5-flash.
	m := googleai.Model("gemini-1.5-flash")
	if m == nil {
		log.Fatal("Failed to find model")
	}

	// Construct a request and send it to the model API (Google AI).
	resp, err := ai.Generate(ctx, m,
		ai.WithConfig(&ai.GenerationCommonConfig{Temperature: 1}),
		ai.WithTextPrompt(`Suggest an item for the menu of a pirate themed restaurant`))
	if err != nil {
		log.Fatal()
	}

	// Handle the response from the model API. In this sample, we just
	// convert it to a string. but more complicated flows might coerce the
	// response into structured output or chain the response into another
	// LLM call.
	text := resp.Text()
	println(text)
}
// [END main]
