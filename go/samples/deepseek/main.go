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

package main

import (
	"context"
	"log"

	deepseek_ai "github.com/cohesion-org/deepseek-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/deepseek"
)

func main() {
	ctx := context.Background()

	g, err := genkit.Init(ctx, genkit.WithPlugins(&deepseek.DeepSeek{}))
	if err != nil {
		log.Fatal(err)
	}

	genkit.DefineFlow(g, "basic-prompt", func(ctx context.Context, input any) (string, error) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithConfig(deepseek_ai.ChatCompletionRequest{
				Temperature: float32(1.0),
			}),
		)
		if err != nil {
			return "", err
		}
		text := resp.Text()
		return text, nil
	})

	<-ctx.Done()
}
