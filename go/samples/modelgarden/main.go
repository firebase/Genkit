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
	"encoding/base64"
	"errors"
	"io"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/vertexai/modelgarden"
)

func main() {
	ctx := context.Background()

	g := genkit.Init(ctx, genkit.WithPlugins(&modelgarden.Anthropic{}))

	// Define a simple flow that generates jokes about a given topic
	genkit.DefineFlow(g, "joke-teller", func(ctx context.Context, input string) (string, error) {
		m := modelgarden.AnthropicModel(g, "claude-3-5-sonnet-v2")
		if m == nil {
			return "", errors.New("joke-teller: failed to find model")
		}

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&ai.GenerationCommonConfig{
				Temperature: 1.0,
			}),
			ai.WithPrompt(`Tell a short joke about %s`, input))
		if err != nil {
			return "", err
		}

		text := resp.Text()
		return text, nil
	})

	genkit.DefineFlow(g, "image-descriptor", func(ctx context.Context, foo string) (string, error) {
		m := modelgarden.AnthropicModel(g, "claude-3-5-sonnet-v2")
		if m == nil {
			return "", errors.New("image-descriptor: failed to find model")
		}

		img, err := fetchImgAsBase64()
		if err != nil {
			log.Fatal(err)
		}

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&ai.GenerationCommonConfig{
				Temperature: 1.0,
			}),
			ai.WithMessages(ai.NewUserMessage(
				ai.NewTextPart("Can you describe what's in this image?"),
				ai.NewMediaPart("image/jpeg", "data:image/jpeg;base64,"+img)),
			))
		if err != nil {
			return "", err
		}

		text := resp.Text()
		return text, nil
	})

	type Recipe struct {
		Steps       []string
		Ingredients []string
	}

	genkit.DefineFlow(g, "chef-bot", func(ctx context.Context, input string) (Recipe, error) {
		m := modelgarden.AnthropicModel(g, "claude-3-5-sonnet-v2")
		r := Recipe{}
		if m == nil {
			return r, errors.New("chef-bot: failed to find model")
		}

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&ai.GenerationCommonConfig{
				Temperature: 1.0,
			}),
			ai.WithPrompt(`Send me a recipe for %s`, input),
			ai.WithOutputType(Recipe{}))
		if err != nil {
			return r, err
		}

		err = resp.Output(&r)
		if err != nil {
			return r, err
		}
		return r, nil
	})

	<-ctx.Done()
}

func fetchImgAsBase64() (string, error) {
	imgUrl := "https://pd.w.org/2025/07/58268765f177911d4.13750400-2048x1365.jpg"
	resp, err := http.Get(imgUrl)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", err
	}

	imageBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	base64string := base64.StdEncoding.EncodeToString(imageBytes)
	return base64string, nil
}
