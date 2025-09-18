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
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	// Initialize Genkit with the Google AI plugin. When you pass nil for the
	// Config parameter, the Google AI plugin will get the API key from the
	// GEMINI_API_KEY or GOOGLE_API_KEY environment variable, which is the recommended
	// practice.
	g := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
		genkit.WithDefaultModel("googleai/gemini-2.5-flash"))

	// Define a simple flow that generates jokes about a given topic
	genkit.DefineFlow(g, "joke-teller", func(ctx context.Context, input string) (string, error) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithConfig(&genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](1.0),
				ThinkingConfig: &genai.ThinkingConfig{
					ThinkingBudget: genai.Ptr[int32](0),
				},
			}),
			ai.WithPrompt("Tell short jokes about %s", input))
		if err != nil {
			return "", err
		}

		return resp.Text(), nil
	})

	// Define a simple flow that generates jokes about a given topic with a context of bananas
	genkit.DefineFlow(g, "context", func(ctx context.Context, input string) (string, error) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName("googleai/gemini-2.0-flash"),
			ai.WithConfig(&genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](1.0),
			}),
			ai.WithPrompt("Tell short jokes about %s", input),
			ai.WithDocs(ai.DocumentFromText("Bananas are plentiful in the tropics.", nil)))
		if err != nil {
			return "", err
		}

		text := resp.Text()
		return text, nil
	})

	// Simple flow that generates a brief comic strip
	genkit.DefineFlow(g, "comic-strip-generator", func(ctx context.Context, input string) ([]string, error) {
		if input == "" {
			input = `A little blue gopher with big eyes trying to learn Python,
				use a cartoon style, the story should be tragic because he
				chose the wrong programming language, the proper programing
				language for a gopher should be Go`
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName("googleai/gemini-2.5-flash-image-preview"), // nano banana
			ai.WithConfig(&genai.GenerateContentConfig{
				Temperature:        genai.Ptr[float32](0.5),
				ResponseModalities: []string{"IMAGE", "TEXT"},
			}),
			ai.WithPrompt("generate a short story about %s and for each scene, generate an image for it", input))
		if err != nil {
			return nil, err
		}

		story := []string{}
		for _, p := range resp.Message.Content {
			if p.IsMedia() || p.IsText() {
				story = append(story, p.Text)
			}
		}

		return story, nil
	})

	// A flow that uses Romeo and Juliet as cache contents to answer questions about the book
	genkit.DefineFlow(g, "cached-contents", func(ctx context.Context, input string) (string, error) {
		// Romeo and Juliet
		url := "https://www.gutenberg.org/cache/epub/1513/pg1513.txt"
		prompt := "I'll provide you with some text contents that I want you to use to answer further questions"
		content, err := readTextFromURL(url)
		if err != nil {
			return "", err
		}

		resp, err := genkit.Generate(ctx, g,
			ai.WithConfig(&genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](1.0),
				ThinkingConfig: &genai.ThinkingConfig{
					ThinkingBudget: genai.Ptr[int32](0),
				},
			}),
			ai.WithMessages(
				ai.NewUserTextMessage(content).WithCacheTTL(360), // create cache contents
			),
			ai.WithPrompt(prompt))
		if err != nil {
			return "", err
		}

		// use previous messages to keep the conversation going and
		// ask questions related to the cached content
		prompt = "Write a brief summary of the character development of Juliet"
		if input != "" {
			prompt = input
		}
		resp, err = genkit.Generate(ctx, g,
			ai.WithConfig(&genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](1.0),
				ThinkingConfig: &genai.ThinkingConfig{
					ThinkingBudget: genai.Ptr[int32](0),
				},
			}),
			ai.WithMessages(resp.History()...),
			ai.WithPrompt(prompt))
		if err != nil {
			return "", nil
		}
		return resp.Text(), nil
	})

	// Define a flow to demonstrate code execution
	genkit.DefineFlow(g, "code-execution", func(ctx context.Context, _ any) (string, error) {
		m := googlegenai.GoogleAIModel(g, "gemini-2.5-flash")
		if m == nil {
			return "", fmt.Errorf("failed to find model")
		}

		problem := "find the sum of first 5 prime numbers"
		fmt.Printf("Problem: %s\n", problem)

		// Generate response with code execution enabled
		fmt.Println("Sending request to Gemini...")
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](0.2),
				Tools: []*genai.Tool{
					{
						CodeExecution: &genai.ToolCodeExecution{},
					},
				},
			}),
			ai.WithPrompt(problem))
		if err != nil {
			return "", err
		}

		// You can also use the helper function for simpler code
		fmt.Println("\n=== INTERNAL CODE EXECUTION ===")
		displayCodeExecution(resp.Message)

		fmt.Println("\n=== COMPLETE INTERNAL CODE EXECUTION ===")
		text := resp.Text()
		fmt.Println(text)

		return text, nil
	})

	genkit.DefineFlow(g, "image-descriptor", func(ctx context.Context, foo string) (string, error) {
		img, err := fetchImgAsBase64()
		if err != nil {
			return "", err
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithConfig(&genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](1.0),
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

	genkit.DefineFlow(g, "image-generation", func(ctx context.Context, input string) ([]string, error) {
		r, err := genkit.Generate(ctx, g,
			ai.WithModelName("googleai/imagen-4.0-generate-001"),
			ai.WithPrompt("Generate an image of %s", input),
			ai.WithConfig(&genai.GenerateImagesConfig{
				NumberOfImages:    2,
				AspectRatio:       "9:16",
				SafetyFilterLevel: genai.SafetyFilterLevelBlockLowAndAbove,
				PersonGeneration:  genai.PersonGenerationAllowAll,
				OutputMIMEType:    "image/jpeg",
			}),
		)
		if err != nil {
			return nil, err
		}

		var images []string
		for _, m := range r.Message.Content {
			images = append(images, m.Text)
		}
		return images, nil
	})

	// Define a simple flow that generates audio transcripts from a given audio
	genkit.DefineFlow(g, "speech-to-text-flow", func(ctx context.Context, input any) (string, error) {
		audio, err := os.Open("./genkit.wav")
		if err != nil {
			return "", err
		}
		defer audio.Close()

		audioBytes, err := io.ReadAll(audio)
		if err != nil {
			return "", err
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName("googleai/gemini-2.5-flash"),
			ai.WithMessages(ai.NewUserMessage(
				ai.NewTextPart("Can you transcribe the next audio?"),
				ai.NewMediaPart("audio/wav", "data:audio/wav;base64,"+base64.StdEncoding.EncodeToString(audioBytes)))),
		)
		if err != nil {
			return "", err
		}

		return resp.Text(), nil
	})

	// Simple flow that generates an audio from a given text
	genkit.DefineFlow(g, "text-to-speech-flow", func(ctx context.Context, input string) (string, error) {
		prompt := "Genkit is the best Gen AI library!"
		if input != "" {
			prompt = input
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithConfig(&genai.GenerateContentConfig{
				Temperature:        genai.Ptr[float32](1.0),
				ResponseModalities: []string{"AUDIO"},
				SpeechConfig: &genai.SpeechConfig{
					VoiceConfig: &genai.VoiceConfig{
						PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
							VoiceName: "Algenib",
						},
					},
				},
			}),
			ai.WithModelName("googleai/gemini-2.5-flash-preview-tts"),
			ai.WithPrompt("Say: %s", prompt))
		if err != nil {
			return "", err
		}

		// base64 encoded audio
		return resp.Text(), nil
	})

	type greetingStyle struct {
		Style    string `json:"style"`
		Location string `json:"location"`
		Name     string `json:"name"`
	}

	type greeting struct {
		Greeting string `json:"greeting"`
	}

	// Define a simple flow that prompts an LLM to generate greetings using a
	// given style.
	genkit.DefineFlow(g, "assistant-greeting", func(ctx context.Context, input greetingStyle) (string, error) {
		// Look up the prompt by name
		prompt := genkit.LookupPrompt(g, "example")
		if prompt == nil {
			return "", fmt.Errorf("assistantreetingFlow: failed to find prompt")
		}

		// Execute the prompt with the provided input
		resp, err := prompt.Execute(ctx, ai.WithInput(input))
		if err != nil {
			return "", err
		}

		var output greeting
		if err = resp.Output(&output); err != nil {
			return "", err
		}

		return output.Greeting, nil
	})

	<-ctx.Done()
}

// Helper functions

// readTextFromURL reads the text contents from a given URL
func readTextFromURL(url string) (string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return "", fmt.Errorf("failed to sent HTTP GET request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("non 2XX status code received: %d", resp.StatusCode)
	}

	contents, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	return string(contents), nil
}

// displayCodeExecution prints the code execution results from a message in a formatted way.
// This is a helper for applications that want to display code execution results to users.
func displayCodeExecution(msg *ai.Message) {
	// Extract and display executable code
	code := googlegenai.GetExecutableCode(msg)
	fmt.Printf("Language: %s\n", code.Language)
	fmt.Printf("```%s\n%s\n```\n", code.Language, code.Code)

	// Extract and display execution results
	result := googlegenai.GetCodeExecutionResult(msg)
	fmt.Printf("\nExecution result:\n")
	fmt.Printf("Status: %s\n", result.Outcome)
	fmt.Printf("Output:\n")
	if strings.TrimSpace(result.Output) == "" {
		fmt.Printf("  <no output>\n")
	} else {
		lines := strings.SplitSeq(result.Output, "\n")
		for line := range lines {
			fmt.Printf("  %s\n", line)
		}
	}

	// Display any explanatory text
	for _, part := range msg.Content {
		if part.IsText() {
			fmt.Printf("\nExplanation:\n%s\n", part.Text)
		}
	}
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

	imageData, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	base64string := base64.StdEncoding.EncodeToString(imageData)
	return base64string, nil
}
