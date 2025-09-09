// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0

// This program can be manually tested like so:
// Start the server listening on port 3100:
//
//	genkit start -o -- go run .

package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	oai_deepseek "github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
	"github.com/firebase/genkit/go/plugins/server"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func main() {
	ctx := context.Background()

	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		log.Fatalf("no DEEPSEEK_API_KEY environment variable set")
	}
	ds := oai_deepseek.DeepSeek{
		Opts: []option.RequestOption{
			option.WithAPIKey(apiKey),
		},
	}
	g, err := genkit.Init(ctx, genkit.WithPlugins(&ds))
	if err != nil {
		log.Fatalf("failed to create Genkit: %v", err)
	}

	genkit.DefineFlow(g, "basic", func(ctx context.Context, subject string) (string, error) {
		dsChat := ds.Model(g, "deepseek-chat")

		prompt := fmt.Sprintf("tell me a joke about %s", subject)
		config := &openai.ChatCompletionNewParams{Temperature: openai.Float(0.5)}
		foo, err := genkit.Generate(ctx, g, ai.WithModel(dsChat), ai.WithPrompt(prompt), ai.WithConfig(config))
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("foo: %s", foo.Text()), nil
	})

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
