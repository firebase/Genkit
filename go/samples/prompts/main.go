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

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/vertexai"
)

func main() {
	ctx := context.Background()
	g, err := genkit.Init(ctx, genkit.WithPlugins(&vertexai.VertexAI{}))
	if err != nil {
		log.Fatal(err)
	}

	SimplePrompt(ctx, g)
	PromptWithInput(ctx, g)
	PromptWithOutputType(ctx, g)
	PromptWithTool(ctx, g)
	PromptWithMessageHistory(ctx, g)
	PromptWithExecuteOverrides(ctx, g)
	PromptWithFunctions(ctx, g)

	<-ctx.Done()
}

func SimplePrompt(ctx context.Context, g *genkit.Genkit) {
	m := vertexai.Model(g, "gemini-2.0-flash")

	// Define prompt with default model and system text.
	helloPrompt, err := genkit.DefinePrompt(
		g, "SimplePrompt",
		ai.WithModel(m),
		ai.WithSystemText("You are a helpful AI assistant named Walt. Greet the user."),
	)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := helloPrompt.Execute(ctx)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Print(resp.Text())
}

func PromptWithInput(ctx context.Context, g *genkit.Genkit) {
	m := vertexai.Model(g, "gemini-2.0-flash")

	type HelloPromptInput struct {
		UserName string
		Theme    string
	}

	// Define prompt with input type and default input.
	helloPrompt, err := genkit.DefinePrompt(
		g, "PromptWithInput",
		ai.WithModel(m),
		ai.WithInputType(HelloPromptInput{UserName: "Alex", Theme: "beach vacation"}),
		ai.WithSystemText("You are a helpful AI assistant named Walt. Today's theme is {{Theme}}, respond in this style. Say hello to {{UserName}}."),
	)
	if err != nil {
		log.Fatal(err)
	}

	// Call the model with input that will override the default input.
	resp, err := helloPrompt.Execute(ctx, ai.WithInput(HelloPromptInput{UserName: "Bob"}))
	if err != nil {
		log.Fatal(err)
	}

	fmt.Print(resp.Text())
}

func PromptWithOutputType(ctx context.Context, g *genkit.Genkit) {
	m := vertexai.Model(g, "gemini-2.0-flash")

	type CountryList struct {
		Countries []string
	}

	// Define prompt with output type.
	helloPrompt, err := genkit.DefinePrompt(
		g, "PromptWithOutputType",
		ai.WithModel(m),
		ai.WithOutputType(CountryList{}),
		ai.WithConfig(&ai.GenerationCommonConfig{Temperature: 0.5}),
		ai.WithSystemText("You are a geography teacher. When asked a question about geography, return a list of countries that match the question."),
		ai.WithPromptText("Give me the 10 biggest countries in the world by habitants."),
	)

	if err != nil {
		log.Fatal(err)
	}

	// Call the model.
	resp, err := helloPrompt.Execute(ctx)
	if err != nil {
		log.Fatal(err)
	}

	var countryList CountryList
	err = json.Unmarshal([]byte(resp.Text()), &countryList)
	if err != nil {
		log.Fatal(err)
	}

	for _, country := range countryList.Countries {
		fmt.Println(country)
	}
}

func PromptWithTool(ctx context.Context, g *genkit.Genkit) {
	m := vertexai.Model(g, "gemini-2.0-flash")

	gablorkenTool := genkit.DefineTool(g, "gablorken", "use when need to calculate a gablorken",
		func(ctx *ai.ToolContext, input struct {
			Value float64
			Over  float64
		}) (float64, error) {
			return math.Pow(input.Value, input.Over), nil
		},
	)

	// Define prompt with tool and tool settings.
	helloPrompt, err := genkit.DefinePrompt(
		g, "PromptWithTool",
		ai.WithModel(m),
		ai.WithToolChoice(ai.ToolChoiceAuto),
		ai.WithMaxTurns(1),
		ai.WithTools(gablorkenTool),
		ai.WithPromptText("what is a gablorken of 2 over 3.5?"),
	)

	if err != nil {
		log.Fatal(err)
	}

	// Call the model.
	resp, err := helloPrompt.Execute(ctx)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Print(resp.Text())
}

func PromptWithMessageHistory(ctx context.Context, g *genkit.Genkit) {
	m := vertexai.Model(g, "gemini-2.0-flash")

	// Define prompt with default messages prepended.
	helloPrompt, err := genkit.DefinePrompt(
		g, "PromptWithMessageHistory",
		ai.WithModel(m),
		ai.WithSystemText("You are a helpful AI assistant named Walt"),
		ai.WithMessages(
			ai.NewUserTextMessage("Hi, my name is Bob"),
			ai.NewModelTextMessage("Hi, my name is Walt, what can I help you with?"),
		),
		ai.WithPromptText("So Walt, What is my name?"),
	)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := helloPrompt.Execute(ctx)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Print(resp.Text())
}

func PromptWithExecuteOverrides(ctx context.Context, g *genkit.Genkit) {
	m := vertexai.Model(g, "gemini-2.0-flash")

	// Define prompt with default settings.
	helloPrompt, err := genkit.DefinePrompt(
		g, "PromptWithExecuteOverrides",
		ai.WithModel(m),
		ai.WithSystemText("You are a helpful AI assistant named Walt."),
		ai.WithMessages(ai.NewUserTextMessage("Hi, my name is Bob!")),
	)
	if err != nil {
		log.Fatal(err)
	}

	// Call the model and add additional messages from the user.
	resp, err := helloPrompt.Execute(ctx,
		ai.WithModel(vertexai.Model(g, "gemini-2.0-pro")),
		ai.WithMessages(ai.NewUserTextMessage("And I like turtles.")),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Print(resp.Text())
}

func PromptWithFunctions(ctx context.Context, g *genkit.Genkit) {
	m := vertexai.Model(g, "gemini-2.0-flash")

	type HelloPromptInput struct {
		UserName string
		Theme    string
	}

	// Define prompt with system and prompt functions.
	helloPrompt, err := genkit.DefinePrompt(
		g, "PromptWithFunctions",
		ai.WithModel(m),
		ai.WithInputType(HelloPromptInput{Theme: "pirate"}),
		ai.WithSystemFn(func(ctx context.Context, input any) (string, error) {
			return fmt.Sprintf("You are a helpful AI assistant named Walt. Talk in the style of: %s", input.(HelloPromptInput).Theme), nil
		}),
		ai.WithPromptFn(func(ctx context.Context, input any) (string, error) {
			return fmt.Sprintf("Hello, my name is %s", input.(HelloPromptInput).UserName), nil
		}),
	)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := helloPrompt.Execute(ctx, ai.WithInput(HelloPromptInput{UserName: "Bob"}))
	if err != nil {
		log.Fatal(err)
	}

	fmt.Print(resp.Text())
}
