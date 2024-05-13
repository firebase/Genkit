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

// This program can be manually tested like so:
// Start the server listening on port 3100:
//
//	go run . &
//
// Tell it to run an action:
//
//	curl -d '{"key":"/flow/testAllCoffeeFlows/testAllCoffeeFlows", "input":{"start": {"input":null}}}'  http://localhost:3100/api/runAction
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/google/genkit/go/ai"
	"github.com/google/genkit/go/genkit"
	"github.com/google/genkit/go/genkit/dotprompt"
	"github.com/google/genkit/go/plugins/googleai"
	"github.com/invopop/jsonschema"
)

const simpleGreetingPromptTemplate = `
You're a barista at a nice coffee shop.
A regular customer named {{customerName}} enters.
Greet the customer in one sentence, and recommend a coffee drink.
`

type simpleGreetingInput struct {
	CustomerName string `json:"customerName"`
}

const greetingWithHistoryPromptTemplate = `
{{role "user"}}
Hi, my name is {{customerName}}. The time is {{currentTime}}. Who are you?

{{role "model"}}
I am Barb, a barista at this nice underwater-themed coffee shop called Krabby Kooffee.
I know pretty much everything there is to know about coffee,
and I can cheerfully recommend delicious coffee drinks to you based on whatever you like.

{{role "user"}}
Great. Last time I had {{previousOrder}}.
I want you to greet me in one sentence, and recommend a drink.
`

type customerTimeAndHistoryInput struct {
	CustomerName  string `json:"customerName"`
	CurrentTime   string `json:"currentTime"`
	PreviousOrder string `json:"previousOrder"`
}

type testAllCoffeeFlowsOutput struct {
	Pass    bool     `json:"pass"`
	Replies []string `json:"replies,omitempty"`
	Error   string   `json:"error,omitempty"`
}

func main() {
	apiKey := os.Getenv("GOOGLE_GENAI_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "coffee-shop example requires setting GOOGLE_GENAI_API_KEY in the environment.")
		fmt.Fprintln(os.Stderr, "You can get an API key at https://ai.google.dev.")
		os.Exit(1)
	}

	if err := googleai.Init(context.Background(), "gemini-1.0-pro", apiKey); err != nil {
		log.Fatal(err)
	}

	simpleGreetingPrompt, err := dotprompt.Define("simpleGreeting",
		&dotprompt.Frontmatter{
			Name:  "simpleGreeting",
			Model: "google-genai/gemini-1.0-pro",
			Input: dotprompt.FrontmatterInput{
				Schema: jsonschema.Reflect(simpleGreetingInput{}),
			},
			Output: &ai.GenerateRequestOutput{
				Format: ai.OutputFormatText,
			},
		},
		simpleGreetingPromptTemplate,
		nil,
	)
	if err != nil {
		log.Fatal(err)
	}

	simpleGreetingFlow := genkit.DefineFlow("simpleGreeting", func(ctx context.Context, input *simpleGreetingInput, _ genkit.NoStream) (string, error) {
		vars, err := simpleGreetingPrompt.BuildVariables(input)
		if err != nil {
			return "", err
		}
		ai := &dotprompt.ActionInput{Variables: vars}
		resp, err := simpleGreetingPrompt.Execute(ctx, ai)
		if err != nil {
			return "", err
		}
		text, err := resp.Text()
		if err != nil {
			return "", fmt.Errorf("simpleGreeting: %v", err)
		}
		return text, nil
	})

	greetingWithHistoryPrompt, err := dotprompt.Define("greetingWithHistory",
		&dotprompt.Frontmatter{
			Name:  "greetingWithHistory",
			Model: "google-genai/gemini-1.0-pro",
			Input: dotprompt.FrontmatterInput{
				Schema: jsonschema.Reflect(customerTimeAndHistoryInput{}),
			},
			Output: &ai.GenerateRequestOutput{
				Format: ai.OutputFormatText,
			},
		},
		greetingWithHistoryPromptTemplate,
		nil,
	)
	if err != nil {
		log.Fatal(err)
	}

	greetingWithHistoryFlow := genkit.DefineFlow("greetingWithHistory", func(ctx context.Context, input *customerTimeAndHistoryInput, _ genkit.NoStream) (string, error) {
		vars, err := greetingWithHistoryPrompt.BuildVariables(input)
		if err != nil {
			return "", err
		}
		ai := &dotprompt.ActionInput{Variables: vars}
		resp, err := greetingWithHistoryPrompt.Execute(ctx, ai)
		if err != nil {
			return "", err
		}
		text, err := resp.Text()
		if err != nil {
			return "", fmt.Errorf("greetingWithHistory: %v", err)
		}
		return text, nil
	})

	genkit.DefineFlow("testAllCoffeeFlows", func(ctx context.Context, _ struct{}, _ genkit.NoStream) (*testAllCoffeeFlowsOutput, error) {
		test1, err := genkit.RunFlow(ctx, simpleGreetingFlow, &simpleGreetingInput{
			CustomerName: "Sam",
		})
		if err != nil {
			out := &testAllCoffeeFlowsOutput{
				Pass:  false,
				Error: err.Error(),
			}
			return out, nil
		}
		test2, err := genkit.RunFlow(ctx, greetingWithHistoryFlow, &customerTimeAndHistoryInput{
			CustomerName:  "Sam",
			CurrentTime:   "09:45am",
			PreviousOrder: "Caramel Macchiato",
		})
		if err != nil {
			out := &testAllCoffeeFlowsOutput{
				Pass:  false,
				Error: err.Error(),
			}
			return out, nil
		}
		out := &testAllCoffeeFlowsOutput{
			Pass: true,
			Replies: []string{
				test1,
				test2,
			},
		}
		return out, nil
	})

	if err := genkit.StartDevServer(""); err != nil {
		log.Fatal(err)
	}
}
