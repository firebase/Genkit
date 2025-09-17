package main

import (
	"context"
	"fmt"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/ollama"
	"log"
)

const (
	OllamaAddr   = "http://localhost:11434"
	DefaultModel = "llama3.1:latest"
)

func main() {
	ctx := context.Background()
	// Initialize Ollama Plugin
	ollamaPlugin := ollama.Ollama{
		ServerAddress: OllamaAddr,
	}
	// Initialize Genkit with the Ollama AI plugin
	g, err := genkit.Init(
		ctx,
		genkit.WithPlugins(&ollamaPlugin),
	)
	if err != nil {
		log.Fatalf("genkit init failed: %v", err)
	}

	//Define model
	ollamaPlugin.DefineModel(g, ollama.ModelDefinition{
		Name: DefaultModel,
		Type: "chat",
	}, nil)

	resp, err := genkit.GenerateText(ctx, g,
		ai.WithModelName("ollama/llama3.1:latest"),
		ai.WithPrompt("tell me a short joke."),
	)
	if err != nil {
		log.Fatalf("GenerateText failed: %v", err)
	}
	fmt.Println("result:\n", resp)
}
