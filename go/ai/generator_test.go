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

package ai

import (
	"context"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/internal/registry"
	test_utils "github.com/firebase/genkit/go/tests/utils"
	"github.com/google/go-cmp/cmp"
)

type StructuredResponse struct {
	Subject  string
	Location string
}

var r, _ = registry.New()

func init() {
	// Set up default formats
	ConfigureFormats(r)
}

// echoModel attributes
var (
	modelName = "echo"
	metadata  = ModelInfo{
		Label: modelName,
		Supports: &ModelSupports{
			Multiturn:   true,
			Tools:       true,
			SystemRole:  true,
			Media:       false,
			Constrained: ConstrainedSupportNone,
		},
		Versions: []string{"echo-001", "echo-002"},
		Stage:    ModelStageDeprecated,
	}

	echoModel = DefineModel(r, "test", modelName, &metadata, func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
		if msc != nil {
			msc(ctx, &ModelResponseChunk{
				Content: []*Part{NewTextPart("stream!")},
			})
		}
		textResponse := ""
		for _, m := range gr.Messages {
			if m.Role == RoleUser {
				textResponse = m.Text()
			}
		}
		return &ModelResponse{
			Request: gr,
			Message: NewModelTextMessage(textResponse),
		}, nil
	})
)

// with tools
var gablorkenTool = DefineTool(r, "gablorken", "use when need to calculate a gablorken",
	func(ctx *ToolContext, input struct {
		Value float64
		Over  float64
	},
	) (float64, error) {
		return math.Pow(input.Value, input.Over), nil
	},
)

func TestValidMessage(t *testing.T) {
	t.Parallel()

	t.Run("Valid message with text format", func(t *testing.T) {
		message := &Message{
			Content: []*Part{
				NewTextPart("Hello, World!"),
			},
		}
		outputSchema := &ModelOutputConfig{
			Format: OutputFormatText,
		}
		_, err := validTestMessage(message, outputSchema)
		if err != nil {
			t.Fatal(err)
		}
	})

	t.Run("Valid message with JSON format and matching schema", func(t *testing.T) {
		json := `{
			"name": "John",
			"age": 30,
			"address": {
				"street": "123 Main St",
				"city": "New York",
				"country": "USA"
			}
		}`
		message := &Message{
			Content: []*Part{
				NewTextPart(JSONMarkdown(json)),
			},
		}
		outputSchema := &ModelOutputConfig{
			Format: OutputFormatJSON,
			Schema: map[string]any{
				"type":     "object",
				"required": []string{"name", "age", "address"},
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"age":  map[string]any{"type": "integer"},
					"address": map[string]any{
						"type":     "object",
						"required": []string{"street", "city", "country"},
						"properties": map[string]any{
							"street":  map[string]any{"type": "string"},
							"city":    map[string]any{"type": "string"},
							"country": map[string]any{"type": "string"},
						},
					},
					"phone": map[string]any{"type": "string"},
				},
			},
		}
		message, err := validTestMessage(message, outputSchema)
		if err != nil {
			t.Fatal(err)
		}
		text := message.Text()
		if strings.TrimSpace(text) != strings.TrimSpace(json) {
			t.Fatalf("got %q, want %q", json, text)
		}
	})

	t.Run("Invalid message with JSON format and non-matching schema", func(t *testing.T) {
		message := &Message{
			Content: []*Part{
				NewTextPart(JSONMarkdown(`{"name": "John", "age": "30"}`)),
			},
		}
		outputSchema := &ModelOutputConfig{
			Format: OutputFormatJSON,
			Schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"age":  map[string]any{"type": "integer"},
				},
			},
		}
		_, err := validTestMessage(message, outputSchema)
		errorContains(t, err, "data did not match expected schema")
	})

	t.Run("Message with invalid JSON", func(t *testing.T) {
		message := &Message{
			Content: []*Part{
				NewTextPart(JSONMarkdown(`{"name": "John", "age": 30`)), // Missing trailing }.
			},
		}
		outputSchema := &ModelOutputConfig{
			Format: OutputFormatJSON,
		}
		_, err := validTestMessage(message, outputSchema)
		t.Log(err)
		errorContains(t, err, "not a valid JSON")
	})

	t.Run("No message", func(t *testing.T) {
		outputSchema := &ModelOutputConfig{
			Format: OutputFormatJSON,
		}
		_, err := validTestMessage(nil, outputSchema)
		errorContains(t, err, "message is empty")
	})

	t.Run("Empty message", func(t *testing.T) {
		message := &Message{}
		outputSchema := &ModelOutputConfig{
			Format: OutputFormatJSON,
		}
		_, err := validTestMessage(message, outputSchema)
		errorContains(t, err, "message has no content")
	})

	t.Run("Candidate contains unexpected field", func(t *testing.T) {
		message := &Message{
			Content: []*Part{
				NewTextPart(JSONMarkdown(`{"name": "John", "height": 190}`)),
			},
		}
		outputSchema := &ModelOutputConfig{
			Format: OutputFormatJSON,
			Schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"age":  map[string]any{"type": "integer"},
				},
				"additionalProperties": false,
			},
		}
		_, err := validTestMessage(message, outputSchema)
		errorContains(t, err, "data did not match expected schema")
	})

	t.Run("Invalid expected schema", func(t *testing.T) {
		message := &Message{
			Content: []*Part{
				NewTextPart(JSONMarkdown(`{"name": "John", "age": 30}`)),
			},
		}
		outputSchema := &ModelOutputConfig{
			Format: OutputFormatJSON,
			Schema: map[string]any{
				"type": "invalid",
			},
		}
		_, err := validTestMessage(message, outputSchema)
		errorContains(t, err, "failed to validate data against expected schema")
	})
}

func TestGenerate(t *testing.T) {
	JSON := "\n{\"subject\": \"bananas\", \"location\": \"tropics\"}\n"
	JSONmd := "```json" + JSON + "```"

	bananaModel := DefineModel(r, "test", "banana", &metadata, func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
		if msc != nil {
			msc(ctx, &ModelResponseChunk{
				Content: []*Part{NewJSONPart(JSON)},
			})
		}

		return &ModelResponse{
			Request: gr,
			Message: NewModelTextMessage(JSONmd),
		}, nil
	})

	t.Run("constructs request", func(t *testing.T) {
		wantText := JSON
		wantStreamText := "{\"subject\": \"bananas\", \"location\": \"tropics\"}"
		wantRequest := &ModelRequest{
			Messages: []*Message{
				{
					Role: RoleSystem,
					Content: []*Part{
						NewTextPart("You are a helpful assistant."),
						{
							ContentType: "plain/text",
							Text:        "ignored (conformance message)",
							Metadata:    map[string]any{"purpose": string("output")},
						},
					},
				},
				NewUserTextMessage("How many bananas are there?"),
				NewModelTextMessage("There are at least 10 bananas."),
				{
					Role: RoleUser,
					Content: []*Part{
						NewTextPart("Where can they be found?"),
						{
							Text: "\n\nUse the following information " +
								"to complete your task:\n\n- [0]: Bananas are plentiful in the tropics.\n\n",
							Metadata: map[string]any{"purpose": "context"},
						},
					},
				},
			},
			Config: &GenerationCommonConfig{Temperature: 1},
			Docs:   []*Document{DocumentFromText("Bananas are plentiful in the tropics.", nil)},
			Output: &ModelOutputConfig{
				Format:      OutputFormatJSON,
				ContentType: "application/json",
			},
			Tools: []*ToolDefinition{
				{
					Description: "use when need to calculate a gablorken",
					InputSchema: map[string]any{
						"additionalProperties": bool(false),
						"properties": map[string]any{
							"Over":  map[string]any{"type": string("number")},
							"Value": map[string]any{"type": string("number")},
						},
						"required": []any{string("Value"), string("Over")},
						"type":     string("object"),
					},
					Name:         "gablorken",
					OutputSchema: map[string]any{"type": string("number")},
				},
			},
			ToolChoice: ToolChoiceAuto,
		}

		streamText := ""
		res, err := Generate(context.Background(), r,
			WithModel(bananaModel),
			WithSystem("You are a helpful assistant."),
			WithMessages(
				NewUserTextMessage("How many bananas are there?"),
				NewModelTextMessage("There are at least 10 bananas."),
			),
			WithPrompt("Where can they be found?"),
			WithConfig(&GenerationCommonConfig{
				Temperature: 1,
			}),
			WithDocs(DocumentFromText("Bananas are plentiful in the tropics.", nil)),
			WithOutputType(struct {
				Subject  string `json:"subject"`
				Location string `json:"location"`
			}{}),
			WithTools(gablorkenTool),
			WithToolChoice(ToolChoiceAuto),
			WithStreaming(func(ctx context.Context, grc *ModelResponseChunk) error {
				streamText += grc.Text()
				return nil
			}),
		)
		if err != nil {
			t.Fatal(err)
		}

		gotText := res.Text()
		if diff := cmp.Diff(gotText, wantText); diff != "" {
			t.Errorf("Text() diff (+got -want):\n%s", diff)
		}
		if diff := cmp.Diff(streamText, wantStreamText); diff != "" {
			t.Errorf("Text() diff (+got -want):\n%s", diff)
		}
		if diff := cmp.Diff(wantRequest, res.Request, test_utils.IgnoreNoisyParts([]string{
			"{*ai.ModelRequest}.Messages[0].Content[1].Text", "{*ai.ModelRequest}.Messages[0].Content[1].Metadata",
		})); diff != "" {
			t.Errorf("Request diff (+got -want):\n%s", diff)
		}
	})

	t.Run("handles tool interrupts", func(t *testing.T) {
		interruptTool := DefineTool(r, "interruptor", "always interrupts",
			func(ctx *ToolContext, input any) (any, error) {
				return nil, ctx.Interrupt(&InterruptOptions{
					Metadata: map[string]any{
						"reason": "test interrupt",
					},
				})
			},
		)

		info := &ModelInfo{
			Supports: &ModelSupports{
				Multiturn: true,
				Tools:     true,
			},
		}
		interruptModel := DefineModel(r, "test", "interrupt", info,
			func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
				return &ModelResponse{
					Request: gr,
					Message: &Message{
						Role: RoleModel,
						Content: []*Part{
							NewToolRequestPart(&ToolRequest{
								Name:  "interruptor",
								Input: nil,
							}),
						},
					},
				}, nil
			})

		res, err := Generate(context.Background(), r,
			WithModel(interruptModel),
			WithPrompt("trigger interrupt"),
			WithTools(interruptTool),
		)
		if err != nil {
			t.Fatal(err)
		}
		if res.FinishReason != "interrupted" {
			t.Errorf("expected finish reason 'interrupted', got %q", res.FinishReason)
		}
		if res.FinishMessage != "One or more tool calls resulted in interrupts." {
			t.Errorf("unexpected finish message: %q", res.FinishMessage)
		}

		if len(res.Message.Content) != 1 {
			t.Fatalf("expected 1 content part, got %d", len(res.Message.Content))
		}

		metadata := res.Message.Content[0].Metadata
		if metadata == nil {
			t.Fatal("expected metadata in content part")
		}

		interrupt, ok := metadata["interrupt"].(map[string]any)
		if !ok {
			t.Fatal("expected interrupt metadata")
		}

		reason, ok := interrupt["reason"].(string)
		if !ok || reason != "test interrupt" {
			t.Errorf("expected interrupt reason 'test interrupt', got %v", reason)
		}
	})

	t.Run("handles multiple parallel tool calls", func(t *testing.T) {
		roundCount := 0
		info := &ModelInfo{
			Supports: &ModelSupports{
				Multiturn: true,
				Tools:     true,
			},
		}
		parallelModel := DefineModel(r, "test", "parallel", info,
			func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
				roundCount++
				if roundCount == 1 {
					return &ModelResponse{
						Request: gr,
						Message: &Message{
							Role: RoleModel,
							Content: []*Part{
								NewToolRequestPart(&ToolRequest{
									Name:  "gablorken",
									Input: map[string]any{"Value": 2, "Over": 3},
								}),
								NewToolRequestPart(&ToolRequest{
									Name:  "gablorken",
									Input: map[string]any{"Value": 3, "Over": 2},
								}),
							},
						},
					}, nil
				}
				var sum float64
				for _, msg := range gr.Messages {
					if msg.Role == RoleTool {
						for _, part := range msg.Content {
							if part.ToolResponse != nil {
								sum += part.ToolResponse.Output.(float64)
							}
						}
					}
				}
				return &ModelResponse{
					Request: gr,
					Message: &Message{
						Role: RoleModel,
						Content: []*Part{
							NewTextPart(fmt.Sprintf("Final result: %d", int(sum))),
						},
					},
				}, nil
			})

		res, err := Generate(context.Background(), r,
			WithModel(parallelModel),
			WithPrompt("trigger parallel tools"),
			WithTools(gablorkenTool),
		)
		if err != nil {
			t.Fatal(err)
		}

		finalPart := res.Message.Content[0]
		if finalPart.Text != "Final result: 17" {
			t.Errorf("expected final result text to be 'Final result: 17', got %q", finalPart.Text)
		}
	})

	t.Run("handles multiple rounds of tool calls", func(t *testing.T) {
		roundCount := 0
		info := &ModelInfo{
			Supports: &ModelSupports{
				Multiturn: true,
				Tools:     true,
			},
		}
		multiRoundModel := DefineModel(r, "test", "multiround", info,
			func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
				roundCount++
				if roundCount == 1 {
					return &ModelResponse{
						Request: gr,
						Message: &Message{
							Role: RoleModel,
							Content: []*Part{
								NewToolRequestPart(&ToolRequest{
									Name:  "gablorken",
									Input: map[string]any{"Value": 2, "Over": 3},
								}),
							},
						},
					}, nil
				}
				if roundCount == 2 {
					return &ModelResponse{
						Request: gr,
						Message: &Message{
							Role: RoleModel,
							Content: []*Part{
								NewToolRequestPart(&ToolRequest{
									Name:  "gablorken",
									Input: map[string]any{"Value": 3, "Over": 2},
								}),
							},
						},
					}, nil
				}
				return &ModelResponse{
					Request: gr,
					Message: &Message{
						Role: RoleModel,
						Content: []*Part{
							NewTextPart("Final result"),
						},
					},
				}, nil
			})

		res, err := Generate(context.Background(), r,
			WithModel(multiRoundModel),
			WithPrompt("trigger multiple rounds"),
			WithTools(gablorkenTool),
			WithMaxTurns(2),
		)
		if err != nil {
			t.Fatal(err)
		}

		if roundCount != 3 {
			t.Errorf("expected 3 rounds, got %d", roundCount)
		}

		if res.Text() != "Final result" {
			t.Errorf("expected final message 'Final result', got %q", res.Text())
		}
	})

	t.Run("exceeds maximum turns", func(t *testing.T) {
		info := &ModelInfo{
			Supports: &ModelSupports{
				Multiturn: true,
				Tools:     true,
			},
		}
		infiniteModel := DefineModel(r, "test", "infinite", info,
			func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
				return &ModelResponse{
					Request: gr,
					Message: &Message{
						Role: RoleModel,
						Content: []*Part{
							NewToolRequestPart(&ToolRequest{
								Name:  "gablorken",
								Input: map[string]any{"Value": 2, "Over": 2},
							}),
						},
					},
				}, nil
			})

		_, err := Generate(context.Background(), r,
			WithModel(infiniteModel),
			WithPrompt("trigger infinite loop"),
			WithTools(gablorkenTool),
			WithMaxTurns(2),
		)

		if err == nil {
			t.Fatal("expected error for exceeding maximum turns")
		}
		if !strings.Contains(err.Error(), "exceeded maximum tool call iterations (2)") {
			t.Errorf("unexpected error message: %v", err)
		}
	})

	t.Run("applies middleware", func(t *testing.T) {
		middlewareCalled := false
		testMiddleware := func(next ModelFunc) ModelFunc {
			return func(ctx context.Context, req *ModelRequest, cb ModelStreamCallback) (*ModelResponse, error) {
				middlewareCalled = true
				req.Messages = append(req.Messages, NewUserTextMessage("middleware was here"))
				return next(ctx, req, cb)
			}
		}

		res, err := Generate(context.Background(), r,
			WithModel(echoModel),
			WithPrompt("test middleware"),
			WithMiddleware(testMiddleware),
		)
		if err != nil {
			t.Fatal(err)
		}

		if !middlewareCalled {
			t.Error("middleware was not called")
		}

		expectedText := "middleware was here"
		if res.Text() != expectedText {
			t.Errorf("got text %q, want %q", res.Text(), expectedText)
		}
	})
}

func TestModelVersion(t *testing.T) {
	t.Run("valid version", func(t *testing.T) {
		_, err := Generate(context.Background(), r,
			WithModel(echoModel),
			WithConfig(&GenerationCommonConfig{
				Temperature: 1,
				Version:     "echo-001",
			}),
			WithPrompt("tell a joke about batman"))
		if err != nil {
			t.Errorf("model version should be valid")
		}
	})
	t.Run("invalid version", func(t *testing.T) {
		_, err := Generate(context.Background(), r,
			WithModel(echoModel),
			WithConfig(&GenerationCommonConfig{
				Temperature: 1,
				Version:     "echo-im-not-a-version",
			}),
			WithPrompt("tell a joke about batman"))
		if err == nil {
			t.Errorf("model version should be invalid: %v", err)
		}
	})
}

func TestLookupModel(t *testing.T) {
	t.Run("should return model", func(t *testing.T) {
		if LookupModel(r, "test", modelName) == nil {
			t.Errorf("LookupModel did not return model")
		}
	})
	t.Run("should return nil", func(t *testing.T) {
		if LookupModel(r, "foo", "bar") != nil {
			t.Errorf("LookupModel did not return nil")
		}
	})
}

func TestLookupModelByName(t *testing.T) {
	t.Run("should return model", func(t *testing.T) {
		model, _ := LookupModelByName(r, "test/"+modelName)
		if model == nil {
			t.Errorf("LookupModelByName did not return model")
		}
	})
	t.Run("should return nil", func(t *testing.T) {
		_, err := LookupModelByName(r, "foo/bar")
		if err == nil {
			t.Errorf("LookupModelByName did not return error")
		}
	})
}

func JSONMarkdown(text string) string {
	return "```json\n" + text + "\n```"
}

func errorContains(t *testing.T, err error, want string) {
	t.Helper()
	if err == nil {
		t.Error("got nil, want error")
	} else if !strings.Contains(err.Error(), want) {
		t.Errorf("got error message %q, want it to contain %q", err, want)
	}
}

func validTestMessage(m *Message, output *ModelOutputConfig) (*Message, error) {
	resolvedFormat, err := resolveFormat(r, output.Schema, output.Format)
	if err != nil {
		return nil, err
	}

	handler, err := resolvedFormat.Handler(output.Schema)
	if err != nil {
		return nil, err
	}

	return handler.ParseMessage(m)
}
