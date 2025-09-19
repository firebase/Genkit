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
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	// Initialize Genkit with the Google AI plugin and Gemini 2.0 Flash.
	g := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
	)

	resp, err := genkit.GenerateOperation(ctx, g,
		ai.WithMessages(ai.NewUserTextMessage("Mouse eating cheese")),
		ai.WithModelName("googleai/veo-2.0-generate-001"),
		ai.WithConfig(&genai.GenerateVideosConfig{
			NumberOfVideos:  1,
			AspectRatio:     "16:9",
			DurationSeconds: genai.Ptr(int32(5)),
		}),
	)
	if err != nil {
		log.Fatalf("could not start operation: %v", err)
	}

	// Get the background model for status checking
	bgAction := genkit.LookupBackgroundModel(g, "googleai/veo-2.0-generate-001")
	if bgAction == nil {
		log.Fatalf("background model not found")
	}

	// Wait for operation to complete
	currentOp := resp
	for {
		// Check if operation completed with error
		if currentOp.Error != nil {
			log.Fatalf("operation failed: %s", currentOp.Error)
		}

		// Check if operation is complete
		if currentOp.Done {
			break
		}

		log.Printf("Operation %s is still running...", currentOp.ID)

		// Wait before polling again (avoid busy waiting)
		select {
		case <-ctx.Done():
			log.Fatalf("context cancelled: %v", ctx.Err())
		case <-time.After(2 * time.Second): // Poll every 2 seconds
		}

		// Check operation status
		updatedOp, err := bgAction.CheckOperation(ctx, currentOp)
		if err != nil {
			log.Fatalf("failed to check operation status: %v", err)
		}

		currentOp = updatedOp
	}

	if currentOp != nil {
		opJson, _ := json.Marshal(currentOp.Output)
		fmt.Printf("%s", opJson)

		// Download the generated video
		if err := downloadGeneratedVideo(ctx, currentOp); err != nil {
			log.Printf("Failed to download video: %v", err)
		} else {
			fmt.Println("Video successfully downloaded to veo3_video.mp4")
		}
	}
}

// For testing purpose need to be removed if needed
// downloadGeneratedVideo downloads the generated video from the operation result
func downloadGeneratedVideo(ctx context.Context, operation *core.Operation[*ai.ModelResponse]) error {
	// Get the API key from environment
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GOOGLE_API_KEY")
	}
	if apiKey == "" {
		return fmt.Errorf("no API key found. Please set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
	}

	// Parse the operation output to extract video URL
	if operation.Output == nil {
		return fmt.Errorf("operation output is nil")
	}

	modelResponse := operation.Output
	if modelResponse.Message == nil {
		return fmt.Errorf("model response message is nil")
	}

	// Find the media part in the message content
	var videoURL string
	for _, part := range modelResponse.Message.Content {
		if part.IsMedia() && part.Text != "" {
			videoURL = part.Text
			break
		}
	}

	if videoURL == "" {
		return fmt.Errorf("no video URL found in the operation output")
	}

	// Append API key to the URL if it's not already there
	downloadURL := videoURL
	if !strings.Contains(downloadURL, "key=") {
		separator := "&"
		if !strings.Contains(downloadURL, "?") {
			separator = "?"
		}
		downloadURL = fmt.Sprintf("%s%skey=%s", videoURL, separator, apiKey)
	}

	req, err := http.NewRequestWithContext(ctx, "GET", downloadURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %v", err)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download video: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download video: HTTP %d", resp.StatusCode)
	}

	// Create the output file
	filename := "veo3_video.mp4"
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create output file: %v", err)
	}
	defer file.Close()

	// Copy the video content to the file
	_, err = io.Copy(file, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to write video data to file: %v", err)
	}

	return nil
}
