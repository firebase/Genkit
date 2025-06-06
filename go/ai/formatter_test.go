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

package ai

import (
	"context"
	"testing"

	"github.com/firebase/genkit/go/internal/base"
	"github.com/google/go-cmp/cmp"
)

func TestConstrainedGenerate(t *testing.T) {
	JSON := "\n{\"foo\": \"bar\"}\n"
	JSONmd := "```json" + JSON + "```"

	modelInfo := ModelInfo{
		Label: modelName,
		Supports: &ModelSupports{
			Multiturn:   true,
			Tools:       true,
			SystemRole:  true,
			Media:       false,
			Constrained: ConstrainedSupportAll,
		},
		Versions: []string{"echo-001", "echo-002"},
	}

	formatModel := DefineModel(r, "test", "format", &modelInfo, func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
		if msc != nil {
			msc(ctx, &ModelResponseChunk{
				Content: []*Part{NewJSONPart("{\"Foo\": \"bar\"}")},
			})
		}

		return &ModelResponse{
			Request: gr,
			Message: NewModelTextMessage(JSONmd),
		}, nil
	})

	t.Run("doesn't inject instructions when model supports native contrained generation", func(t *testing.T) {
		wantText := JSON
		wantStreamText := "{\"Foo\": \"bar\"}"
		wantRequest := &ModelRequest{
			Messages: []*Message{
				{
					Role: RoleUser,
					Content: []*Part{
						NewTextPart("generate json"),
					},
				},
			},
			Output: &ModelOutputConfig{
				Format: OutputFormatJSON,
				Schema: map[string]any{
					"additionalProperties": bool(false),
					"properties": map[string]any{
						"foo": map[string]any{"type": string("string")},
					},
					"required": []any{string("foo")},
					"type":     string("object"),
				},
				Constrained: true,
				ContentType: "application/json",
			},
			Config: &GenerationCommonConfig{Temperature: 1},
			Tools:  []*ToolDefinition{},
		}

		streamText := ""
		res, err := Generate(context.Background(), r,
			WithModel(formatModel),
			WithPrompt("generate json"),
			WithConfig(&GenerationCommonConfig{
				Temperature: 1,
			}),
			WithStreaming(func(ctx context.Context, grc *ModelResponseChunk) error {
				streamText += grc.Text()
				return nil
			}),
			WithOutputType(struct {
				Foo string `json:"foo"`
			}{}),
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
		if diff := cmp.Diff(res.Request, wantRequest); diff != "" {
			t.Errorf("Request diff (+got -want):\n%s", diff)
		}
	})

	t.Run("doesn't use format instructions when explicitly instructed not to", func(t *testing.T) {
		wantText := JSON
		wantStreamText := "{\"Foo\": \"bar\"}"
		wantRequest := &ModelRequest{
			Messages: []*Message{
				{
					Role: RoleUser,
					Content: []*Part{
						NewTextPart("generate json"),
					},
				},
			},
			Output: &ModelOutputConfig{
				Format:      OutputFormatJSON,
				ContentType: "application/json",
			},
			Config: &GenerationCommonConfig{Temperature: 1},
			Tools:  []*ToolDefinition{},
		}

		streamText := ""
		res, err := Generate(context.Background(), r,
			WithModel(formatModel),
			WithPrompt("generate json"),
			WithConfig(&GenerationCommonConfig{
				Temperature: 1,
			}),
			WithStreaming(func(ctx context.Context, grc *ModelResponseChunk) error {
				streamText += grc.Text()
				return nil
			}),
			WithOutputType(struct {
				Foo string `json:"foo"`
			}{}),
			WithOutputInstructions(""),
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
		if diff := cmp.Diff(res.Request, wantRequest); diff != "" {
			t.Errorf("Request diff (+got -want):\n%s", diff)
		}
	})

	t.Run("uses format instructions given by user", func(t *testing.T) {
		customInstructions := "The generated output should be in JSON format and conform to the following schema:\n\n```{\"additionalProperties\":false,\"properties\":{\"foo\":{\"type\":\"string\"}},\"required\":[\"foo\"],\"type\":\"object\"}```"
		wantText := JSON
		wantStreamText := "{\"Foo\": \"bar\"}"
		wantRequest := &ModelRequest{
			Messages: []*Message{
				{
					Role: RoleUser,
					Content: []*Part{
						NewTextPart("generate json"),
						{
							Kind:        PartText,
							ContentType: "plain/text",
							Text:        customInstructions,
							Metadata:    map[string]any{"purpose": "output"},
						},
					},
				},
			},
			Output: &ModelOutputConfig{
				Format:      OutputFormatJSON,
				ContentType: "application/json",
			},
			Config: &GenerationCommonConfig{Temperature: 1},
			Tools:  []*ToolDefinition{},
		}

		streamText := ""
		res, err := Generate(context.Background(), r,
			WithModel(formatModel),
			WithPrompt("generate json"),
			WithConfig(&GenerationCommonConfig{
				Temperature: 1,
			}),
			WithStreaming(func(ctx context.Context, grc *ModelResponseChunk) error {
				streamText += grc.Text()
				return nil
			}),
			WithOutputType(struct {
				Foo string `json:"foo"`
			}{}),
			WithOutputInstructions(customInstructions),
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
		if diff := cmp.Diff(res.Request, wantRequest); diff != "" {
			t.Errorf("Request diff (+got -want):\n%s", diff)
		}
	})

	t.Run("uses simulated constrained generation when explicitly told to do so", func(t *testing.T) {
		wantText := JSON
		wantStreamText := "{\"Foo\": \"bar\"}"
		wantRequest := &ModelRequest{
			Messages: []*Message{
				{
					Role: RoleUser,
					Content: []*Part{
						NewTextPart("generate json"),
						{
							Kind:        PartText,
							ContentType: "plain/text",
							Text:        "Output should be in JSON format and conform to the following schema:\n\n```{\"additionalProperties\":false,\"properties\":{\"foo\":{\"type\":\"string\"}},\"required\":[\"foo\"],\"type\":\"object\"}```",
							Metadata:    map[string]any{"purpose": "output"},
						},
					},
				},
			},
			Output: &ModelOutputConfig{
				Format:      OutputFormatJSON,
				ContentType: "application/json",
			},
			Config: &GenerationCommonConfig{Temperature: 1},
			Tools:  []*ToolDefinition{},
		}

		streamText := ""
		res, err := Generate(context.Background(), r,
			WithModel(formatModel),
			WithPrompt("generate json"),
			WithConfig(&GenerationCommonConfig{
				Temperature: 1,
			}),
			WithStreaming(func(ctx context.Context, grc *ModelResponseChunk) error {
				streamText += grc.Text()
				return nil
			}),
			WithOutputType(struct {
				Foo string `json:"foo"`
			}{}),
			WithCustomConstrainedOutput(),
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
		if diff := cmp.Diff(res.Request, wantRequest); diff != "" {
			t.Errorf("Request diff (+got -want):\n%s", diff)
		}
	})

	t.Run("works with prompts", func(t *testing.T) {
		wantText := JSON
		wantRequest := &ModelRequest{
			Messages: []*Message{
				{
					Role: RoleUser,
					Content: []*Part{
						NewTextPart("generate json"),
						{
							Kind:        PartText,
							ContentType: "plain/text",
							Text:        "Output should be in JSON format and conform to the following schema:\n\n```{\"additionalProperties\":false,\"properties\":{\"foo\":{\"type\":\"string\"}},\"required\":[\"foo\"],\"type\":\"object\"}```",
							Metadata:    map[string]any{"purpose": "output"},
						},
					},
				},
			},
			Output: &ModelOutputConfig{
				Format:      OutputFormatJSON,
				ContentType: "application/json",
			},
			Config: &GenerationCommonConfig{Temperature: 1},
			Tools:  []*ToolDefinition{},
		}

		p, err := DefinePrompt(r, "formatPrompt",
			WithPrompt("generate json"),
			WithModel(formatModel),
			WithOutputType(struct {
				Foo string `json:"foo"`
			}{}),
			WithCustomConstrainedOutput(),
		)
		if err != nil {
			t.Fatal(err)
		}

		res, err := p.Execute(
			context.Background(),
			WithConfig(&GenerationCommonConfig{Temperature: 1}),
		)
		if err != nil {
			t.Fatal(err)
		}

		gotText := res.Text()
		if diff := cmp.Diff(gotText, wantText); diff != "" {
			t.Errorf("Text() diff (+got -want):\n%s", diff)
		}
		if diff := cmp.Diff(res.Request, wantRequest); diff != "" {
			t.Errorf("Request diff (+got -want):\n%s", diff)
		}
	})
}

func TestHandlers(t *testing.T) {
	type TestEnum struct {
		FavColor string `json:"fav_color,omitempty" jsonschema:"enum=red,enum=green,enum=blue"`
	}

	enumSchema := base.SchemaAsMap(base.InferJSONSchema(TestEnum{}))

	tests := []struct {
		name         string
		format       string
		schema       map[string]any
		output       ModelOutputConfig
		instructions string
		wantErr      bool
	}{
		{
			name:   "not existing format",
			format: "foobar",
			schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"age":  map[string]any{"type": "integer"},
				},
				"additionalProperties": false,
			},
			wantErr: true,
		},
		{
			name:   "text handler",
			format: "text",
			schema: nil,
			output: ModelOutputConfig{
				ContentType: "text/plain",
			},
			instructions: "",
			wantErr:      false,
		},
		{
			name:   "json handler",
			format: "json",
			schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"age":  map[string]any{"type": "integer"},
				},
				"additionalProperties": false,
			},
			output: ModelOutputConfig{
				Format: "json",
				Schema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name": map[string]any{"type": "string"},
						"age":  map[string]any{"type": "integer"},
					},
					"additionalProperties": false,
				},
				ContentType: "application/json",
			},
			instructions: "Output should be in JSON format and conform to the following schema:\n\n```{\"additionalProperties\":false,\"properties\":{\"age\":{\"type\":\"integer\"},\"name\":{\"type\":\"string\"}},\"type\":\"object\"}```",
			wantErr:      false,
		},
		{
			name:   "jsonl handler",
			format: OutputFormatJSONL,
			schema: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name": map[string]any{"type": "string"},
						"age":  map[string]any{"type": "integer"},
					},
				},
				"additionalProperties": false,
			},
			output: ModelOutputConfig{
				Format: OutputFormatJSONL,
				Schema: map[string]any{
					"type": "array",
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"name": map[string]any{"type": "string"},
							"age":  map[string]any{"type": "integer"},
						},
					},
					"additionalProperties": false,
				},
				ContentType: "application/jsonl",
			},
			instructions: "Output should be JSONL format, a sequence of JSON objects (one per line) separated by a newline '\\n' character. Each line should be a JSON object conforming to the following schema:\n\n```{\"properties\":{\"age\":{\"type\":\"integer\"},\"name\":{\"type\":\"string\"}},\"type\":\"object\"}```",
			wantErr:      false,
		},
		{
			name:   "array handler",
			format: "array",
			schema: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id":   map[string]any{"type": "integer"},
						"name": map[string]any{"type": "string"},
					},
				},
				"additionalProperties": false,
			},
			output: ModelOutputConfig{
				Format: OutputFormatArray,
				Schema: map[string]any{
					"type": "array",
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"id":   map[string]any{"type": "integer"},
							"name": map[string]any{"type": "string"},
						},
					},
					"additionalProperties": false,
				},
				Constrained: false,
				ContentType: "application/json",
			},
			instructions: "Output should be a JSON array conforming to the following schema:\n\n```{\"properties\":{\"id\":{\"type\":\"integer\"},\"name\":{\"type\":\"string\"}},\"type\":\"object\"}```",
			wantErr:      false,
		},
		{
			name:   "enum handler",
			format: "enum",
			schema: enumSchema,
			output: ModelOutputConfig{
				Format:      OutputFormatEnum,
				Schema:      enumSchema,
				Constrained: false,
				ContentType: "text/enum",
			},
			instructions: "Output should be ONLY one of the following enum values. Do not output any additional information or add quotes.\n\n```red\ngreen\nblue```",
			wantErr:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			format, err := resolveFormat(r, tt.schema, tt.format)
			if (err != nil) != tt.wantErr {
				t.Errorf("resolveFormat() error = %v, wantErr %v", err, tt.wantErr)
				if err != nil {
					t.Logf("Error message: %v", err)
				}
			}

			if !tt.wantErr {
				handler, err := format.Handler(tt.schema)
				if err != nil {
					t.Fatalf("Handler() error = %v", err)
				}
				instructions := handler.Instructions()
				config := handler.Config()

				if diff := cmp.Diff(tt.instructions, instructions); diff != "" {
					t.Errorf("Instructions diff (+got -want):\n%s", diff)
				}
				if diff := cmp.Diff(tt.output, config); diff != "" {
					t.Errorf("config diff (+got -want):\n%s", diff)
				}
			}
		})
	}
}

func TestJsonParser(t *testing.T) {
	tests := []struct {
		name     string
		schema   map[string]any
		response *Message
		want     *Message
		wantErr  bool
	}{
		{
			name: "parses json schema",
			schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"age":  map[string]any{"type": "integer"},
				},
				"additionalProperties": false,
			},
			response: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart(JSONMarkdown(`{"name": "John", "age": 19}`)),
				},
			},
			want: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewJSONPart("\n{\"name\": \"John\", \"age\": 19}\n"),
				},
			},
			wantErr: false,
		},
		{
			name: "contains unexpected field fails",
			schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"age":  map[string]any{"type": "integer"},
				},
				"additionalProperties": false,
			},
			response: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart(JSONMarkdown(`{"name": "John", "height": 190}`)),
				},
			},
			wantErr: true,
		},
		{
			name: "parses JSON with preamble and code fence",
			schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"id": map[string]any{"type": "integer"},
				},
				"additionalProperties": false,
			},
			response: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("Here is the JSON:\n\n```json\n{\"id\": 1}\n```"),
				},
			},
			want: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewJSONPart("\n{\"id\": 1}\n"),
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			formatter := jsonFormatter{}
			handler, err := formatter.Handler(tt.schema)
			if err != nil {
				t.Fatalf("Handler() error = %v", err)
			}
			message, err := handler.ParseMessage(tt.response)

			if (err != nil) != tt.wantErr {
				t.Errorf("ParseMessage() error = %v, wantErr %v", err, tt.wantErr)
				if err != nil {
					t.Logf("Error message: %v", err)
				}
			}

			if !tt.wantErr {
				if diff := cmp.Diff(tt.want, message); diff != "" {
					t.Errorf("Request msgs diff (+got -want):\n%s", diff)
				}
			}
		})
	}
}

func TestJsonStreamingParser(t *testing.T) {
	tests := []struct {
		name     string
		schema   map[string]any
		response []*ModelResponseChunk
		want     []*ModelResponseChunk
		wantErr  bool
	}{
		{
			name: "parses complete JSON object",
			schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
					"age":  map[string]any{"type": "integer"},
				},
				"additionalProperties": false,
			},
			response: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart(JSONMarkdown(`{"name": "John", "age": 19}`)),
					},
				},
			},
			want: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart("{\"name\": \"John\", \"age\": 19}"),
					},
				},
			},
			wantErr: false,
		},
		{
			name: "handles partial JSON",
			schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"id":   map[string]any{"type": "integer"},
					"name": map[string]any{"type": "string"},
				},
				"additionalProperties": false,
			},
			response: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart("{\"id\": 1"),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart(", \"name\": \"test\"}"),
					},
				},
			},
			want: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart("{\"id\": 1}"),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart("{\"id\": 1, \"name\": \"test\"}"),
					},
				},
			},
			wantErr: false,
		},
		{
			name: "handles object with array JSON",
			schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"Countries": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type": "string",
						},
					},
				},
				"additionalProperties": false,
			},
			response: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart("{\""),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart("Countries\": [\"China\", \"India\", \"United States\", \"Indonesia\", \""),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart("Pakistan\", \"Brazil\", \"Nigeria\", \"Bangladesh\", \"Russia\", \"Mexico"),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart("\"]}"),
					},
				},
			},
			want: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart("{}"),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart("{\"Countries\": [\"China\", \"India\", \"United States\", \"Indonesia\", \"\"]}"),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart("{\"Countries\": [\"China\", \"India\", \"United States\", \"Indonesia\", \"Pakistan\", \"Brazil\", \"Nigeria\", \"Bangladesh\", \"Russia\", \"Mexico\"]}"),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart("{\"Countries\": [\"China\", \"India\", \"United States\", \"Indonesia\", \"Pakistan\", \"Brazil\", \"Nigeria\", \"Bangladesh\", \"Russia\", \"Mexico\"]}"),
					},
				},
			},
			wantErr: false,
		},
		{
			name: "handles preamble with code fence",
			schema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"id": map[string]any{"type": "integer"},
				},
				"additionalProperties": false,
			},
			response: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart("Here is the JSON:\n\n```json\n"),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart("{\"id\": 1}\n```"),
					},
				},
			},
			want: []*ModelResponseChunk{
				nil,
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart("{\"id\": 1}"),
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			formatter := jsonFormatter{}
			handler, err := formatter.Handler(tt.schema)
			if err != nil {
				t.Fatalf("Handler() error = %v", err)
			}

			var previousParts []*Part
			for i, chunk := range tt.response {
				previousParts = append(previousParts, chunk.Content...)
				chunk.Content = previousParts

				messageChunk, err := handler.ParseChunk(chunk)
				if err != nil {
					t.Errorf("ParseChunk() error = %v, wantErr %v", err, tt.wantErr)
				}

				if diff := cmp.Diff(tt.want[i], messageChunk); diff != "" {
					t.Errorf("Request msgs diff (+got -want):\n%s", diff)
				}
			}
		})
	}
}

func TestTextParser(t *testing.T) {
	tests := []struct {
		name     string
		response *Message
		want     *Message
		wantErr  bool
	}{
		{
			name: "parses complete text response",
			response: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("Hello World"),
				},
			},
			want: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("Hello World"),
				},
			},
			wantErr: false,
		},
		{
			name: "handles empty response",
			response: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart(""),
				},
			},
			want: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart(""),
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			formatter := textFormatter{}
			handler, err := formatter.Handler(nil)
			if err != nil {
				t.Fatalf("Handler() error = %v", err)
			}
			message, err := handler.ParseMessage(tt.response)

			if (err != nil) != tt.wantErr {
				t.Errorf("ParseMessage() error = %v, wantErr %v", err, tt.wantErr)
				if err != nil {
					t.Logf("Error message: %v", err)
				}
			}

			if !tt.wantErr {
				if diff := cmp.Diff(tt.want, message); diff != "" {
					t.Errorf("Request msgs diff (+got -want):\n%s", diff)
				}
			}
		})
	}
}

func TestTextStreamingParser(t *testing.T) {
	tests := []struct {
		name     string
		response []*ModelResponseChunk
		want     []*ModelResponseChunk
		wantErr  bool
	}{
		{
			name: "emits text chunks as they arrive",
			response: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart("Hello"),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart(" world"),
					},
				},
			},
			want: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart("Hello"),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart(" world"),
					},
				},
			},
			wantErr: false,
		},
		{
			name: "handles empty chunks",
			response: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart(""),
					},
				},
			},
			want: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart(""),
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			formatter := textFormatter{}
			handler, err := formatter.Handler(nil)
			if err != nil {
				t.Fatalf("Handler() error = %v", err)
			}

			for i, chunk := range tt.response {
				messageChunk, err := handler.ParseChunk(chunk)
				if err != nil {
					t.Errorf("ParseChunk() error = %v, wantErr %v", err, tt.wantErr)
				}

				if diff := cmp.Diff(tt.want[i], messageChunk); diff != "" {
					t.Errorf("Request msgs diff (+got -want):\n%s", diff)
				}
			}
		})
	}
}

func TestJsonlParser(t *testing.T) {
	tests := []struct {
		name     string
		schema   map[string]any
		response *Message
		want     *Message
		wantErr  bool
	}{
		{
			name: "parses jsonl schema",
			schema: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id":   map[string]any{"type": "integer"},
						"name": map[string]any{"type": "string"},
					},
				},
				"additionalProperties": false,
			},
			response: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart(JSONMarkdown(`{"id": 1, "name": "test"}\n{"id": 2}\n`)),
				},
			},
			want: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewJSONPart("{\"id\": 1, \"name\": \"test\"}"),
					NewJSONPart("{\"id\": 2}"),
				},
			},
			wantErr: false,
		},
		{
			name: "contains unexpected field fails",
			schema: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name": map[string]any{"type": "string"},
						"age":  map[string]any{"type": "integer"},
					},
				},
				"additionalProperties": false,
			},
			response: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart(JSONMarkdown(`{"id": 1, "foo": "bar"}\n{"id": 2}\n`)),
				},
			},
			wantErr: true,
		},
		{
			name: "parses JSONl with preamble and code fence",
			schema: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id": map[string]any{"type": "integer"},
					},
				},
				"additionalProperties": false,
			},
			response: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("Here are the objects:\n\n```\n{\"id\": 1}\n{\"id\": 2}\n```"),
				},
			},
			want: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewJSONPart("{\"id\": 1}"),
					NewJSONPart("{\"id\": 2}"),
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			formatter := jsonlFormatter{}
			defer func() {
				if r := recover(); r != nil && tt.wantErr {
					t.Log("Test passed, panic was caught!")
				}
			}()
			handler, err := formatter.Handler(tt.schema)
			if err != nil {
				t.Fatalf("Handler() error = %v", err)
			}

			message, err := handler.ParseMessage(tt.response)
			if err != nil {
				t.Errorf("Parse failed")
			}

			if !tt.wantErr {
				if diff := cmp.Diff(tt.want, message); diff != "" {
					t.Errorf("Request msgs diff (+got -want):\n%s", diff)
				}
			}
		})
	}
}

func TestJsonlStreamingParser(t *testing.T) {
	tests := []struct {
		name     string
		schema   map[string]any
		response []*ModelResponseChunk
		want     []*ModelResponseChunk
		wantErr  bool
	}{
		{
			name: "emits complete JSON objects as they arrive",
			schema: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id":   map[string]any{"type": "integer"},
						"name": map[string]any{"type": "string"},
					},
				},
				"additionalProperties": false,
			},
			response: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart(`{"id": 1, "name": "first"}\n`),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart(`{"id": 2, "name": "second"}\n{"id": 3`),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart(`, "name": "third"}\n`),
					},
				},
			},
			want: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart(`{"id": 1, "name": "first"}`),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart(`{"id": 2, "name": "second"}`),
						NewJSONPart(`{"id": 3}`),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart(`{"id": 3, "name": "third"}`),
					},
				},
			},
			wantErr: false,
		},
		{
			name: "handles single object",
			schema: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id":   map[string]any{"type": "integer"},
						"name": map[string]any{"type": "string"},
					},
				},
				"additionalProperties": false,
			},
			response: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart(JSONMarkdown(`{"id": 1, "name": "single"}\n`)),
					},
				},
			},
			want: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart(`{"id": 1, "name": "single"}`),
					},
				},
			},
			wantErr: false,
		},
		{
			name: "handles preamble with code fence",
			schema: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id":   map[string]any{"type": "integer"},
						"name": map[string]any{"type": "string"},
					},
				},
				"additionalProperties": false,
			},
			response: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart("Here are the objects:\n\n```\n"),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart("{\"id\": 1, \"name\": \"item\"}\n```"),
					},
				},
			},
			want: []*ModelResponseChunk{
				{
					Role:    RoleModel,
					Content: nil,
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart(`{"id": 1, "name": "item"}`),
					},
				},
			},
			wantErr: false,
		},
		{
			name: "ignores non-object lines",
			schema: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id": map[string]any{"type": "integer"},
					},
				},
				"additionalProperties": false,
			},
			response: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart(JSONMarkdown(`First object:\n{"id": 1}\nSecond object:\n{"id": 2}\n`)),
					},
				},
			},
			want: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart(`{"id": 1}`),
						NewJSONPart(`{"id": 2}`),
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			formatter := jsonlFormatter{}
			handler, err := formatter.Handler(tt.schema)
			if err != nil {
				t.Fatalf("Handler() error = %v", err)
			}

			var previousParts []*Part
			var receivedChunks []*ModelResponseChunk
			for _, chunk := range tt.response {
				previousParts = append(previousParts, chunk.Content...)
				chunk.Content = previousParts

				messageChunk, err := handler.ParseChunk(chunk)
				if err != nil {
					t.Errorf("ParseChunk() error = %v, wantErr %v", err, tt.wantErr)
				}

				if messageChunk != nil {
					receivedChunks = append(receivedChunks, messageChunk)
				}
			}

			if diff := cmp.Diff(tt.want, receivedChunks); diff != "" {
				t.Errorf("Request msgs diff (+got -want):\n%s", diff)
			}
		})
	}
}

func TestArrayParser(t *testing.T) {
	tests := []struct {
		name     string
		schema   map[string]any
		response *Message
		want     *Message
		wantErr  bool
	}{
		{
			name: "parses array schema",
			schema: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id":   map[string]any{"type": "integer"},
						"name": map[string]any{"type": "string"},
					},
					"required": []string{"id"},
				},
				"additionalProperties": false,
			},
			response: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart(JSONMarkdown(`[{"id": 1, "name": "test"}, {"id": 2}]\n`)),
				},
			},
			want: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewJSONPart("[{\"id\": 1, \"name\": \"test\"}, {\"id\": 2}]"),
				},
			},
			wantErr: false,
		},
		{
			name: "contains unexpected field fails",
			schema: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"name": map[string]any{"type": "string"},
						"age":  map[string]any{"type": "integer"},
					},
					"required": []string{"name", "age"},
				},
				"additionalProperties": false,
			},
			response: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart(JSONMarkdown(`[{"id": 1, "foo": "bar"}]`)),
				},
			},
			wantErr: true,
		},
		{
			name: "parses empty array",
			schema: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id": map[string]any{"type": "integer"},
					},
				},
				"additionalProperties": false,
			},
			response: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("[]"),
				},
			},
			want: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewJSONPart("[]"),
				},
			},
			wantErr: false,
		},
		{
			name: "parses array with preamble and code fence",
			schema: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id": map[string]any{"type": "integer"},
					},
				},
				"additionalProperties": false,
			},
			response: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("Here are the objects:\n\n```\n[{\"id\": 1},{\"id\": 2}]\n```"),
				},
			},
			want: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewJSONPart("[{\"id\": 1},{\"id\": 2}]"),
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			formatter := arrayFormatter{}
			handler, err := formatter.Handler(tt.schema)
			if err != nil {
				t.Fatalf("Handler() error = %v", err)
			}
			message, err := handler.ParseMessage(tt.response)

			if (err != nil) != tt.wantErr {
				t.Errorf("ParseMessage() error = %v, wantErr %v", err, tt.wantErr)
				if err != nil {
					t.Logf("Error message: %v", err)
				}
			}

			if !tt.wantErr {
				if diff := cmp.Diff(tt.want, message); diff != "" {
					t.Errorf("Request msgs diff (+got -want):\n%s", diff)
				}
			}
		})
	}
}

func TestArrayStreamingParser(t *testing.T) {
	tests := []struct {
		name     string
		schema   map[string]any
		response []*ModelResponseChunk
		want     []*ModelResponseChunk
		wantErr  bool
	}{
		{
			name: "emits complete array items as they arrive",
			schema: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id":   map[string]any{"type": "integer"},
						"name": map[string]any{"type": "string"},
					},
					"required": []string{"id"},
				},
				"additionalProperties": false,
			},
			response: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart(`[{"id": 1,`),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart(`"name": "first"}`),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart(`,{"id": 2,"name": "second"}]`),
					},
				},
			},
			want: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart(`[{"id": 1}]`),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart(`[{"id": 1,"name": "first"}]`),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart(`[{"id": 1,"name": "first"},{"id": 2,"name": "second"}]`),
					},
				},
			},
			wantErr: false,
		},
		{
			name: "handles single item arrays",
			schema: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id":   map[string]any{"type": "integer"},
						"name": map[string]any{"type": "string"},
					},
				},
				"additionalProperties": false,
			},
			response: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart(JSONMarkdown(`[{"id": 1, "name": "single"}]`)),
					},
				},
			},
			want: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart(`[{"id": 1, "name": "single"}]`),
					},
				},
			},
			wantErr: false,
		},
		{
			name: "handles preamble with code fence",
			schema: map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"id":   map[string]any{"type": "integer"},
						"name": map[string]any{"type": "string"},
					},
				},
				"additionalProperties": false,
			},
			response: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart("Here is the array you requested:\n\n```json\n["),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewTextPart("{\"id\": 1, \"name\": \"item\"}\n```"),
					},
				},
			},
			want: []*ModelResponseChunk{
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart(`[]`),
					},
				},
				{
					Role: RoleModel,
					Content: []*Part{
						NewJSONPart(`[{"id": 1, "name": "item"}]`),
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			formatter := arrayFormatter{}
			handler, err := formatter.Handler(tt.schema)
			if err != nil {
				t.Fatalf("Handler() error = %v", err)
			}

			var previousParts []*Part
			var receivedChunks []*ModelResponseChunk
			for _, chunk := range tt.response {
				previousParts = append(previousParts, chunk.Content...)
				chunk.Content = previousParts

				messageChunk, err := handler.ParseChunk(chunk)
				if err != nil {
					t.Errorf("ParseChunk() error = %v, wantErr %v", err, tt.wantErr)
				}

				if messageChunk != nil {
					receivedChunks = append(receivedChunks, messageChunk)
				}
			}

			if diff := cmp.Diff(tt.want, receivedChunks); diff != "" {
				t.Errorf("Request msgs diff (+got -want):\n%s", diff)
			}
		})
	}
}

func TestEnumParser(t *testing.T) {
	type TestEnum struct {
		FavColor string `json:"fav_color,omitempty" jsonschema:"enum=red,enum=green,enum=blue"`
	}

	enumSchema := base.SchemaAsMap(base.InferJSONSchema(TestEnum{}))

	tests := []struct {
		name     string
		schema   map[string]any
		response *Message
		want     *Message
		wantErr  bool
	}{
		{
			name:   "parses simple enum value",
			schema: enumSchema,
			response: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("red"),
				},
			},
			want: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("red"),
				},
			},
			wantErr: false,
		},
		{
			name:   "trims whitespace",
			schema: enumSchema,
			response: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("  green\n"),
				},
			},
			want: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("green"),
				},
			},
			wantErr: false,
		},
		{
			name:   "fails on invalid enum",
			schema: enumSchema,
			response: &Message{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("VALUE1"),
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			formatter := enumFormatter{}
			handler, err := formatter.Handler(tt.schema)
			if err != nil {
				t.Fatalf("Handler() error = %v", err)
			}
			message, err := handler.ParseMessage(tt.response)

			if (err != nil) != tt.wantErr {
				t.Errorf("ParseMessage() error = %v, wantErr %v", err, tt.wantErr)
				if err != nil {
					t.Logf("Error message: %v", err)
				}
			}

			if !tt.wantErr {
				if diff := cmp.Diff(tt.want, message); diff != "" {
					t.Errorf("Request msgs diff (+got -want):\n%s", diff)
				}
			}
		})
	}
}

func TestEnumStreamingParser(t *testing.T) {
	type TestEnum struct {
		FavColor string `json:"fav_color,omitempty" jsonschema:"enum=red,enum=green,enum=blue"`
	}

	enumSchema := base.SchemaAsMap(base.InferJSONSchema(TestEnum{}))

	// Test enum streaming parser works in same way as regular parser
	tests := []struct {
		name     string
		schema   map[string]any
		response *ModelResponseChunk
		want     *ModelResponseChunk
		wantErr  bool
	}{
		{
			name:   "parses simple enum value",
			schema: enumSchema,
			response: &ModelResponseChunk{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("red"),
				},
			},
			want: &ModelResponseChunk{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("red"),
				},
			},
			wantErr: false,
		},
		{
			name:   "trims whitespace",
			schema: enumSchema,
			response: &ModelResponseChunk{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("  green\n"),
				},
			},
			want: &ModelResponseChunk{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("green"),
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			formatter := enumFormatter{}
			handler, err := formatter.Handler(tt.schema)
			if err != nil {
				t.Fatalf("Handler() error = %v", err)
			}

			messageChunk, err := handler.ParseChunk(tt.response)
			if err != nil {
				t.Errorf("ParseChunk() error = %v, wantErr %v", err, tt.wantErr)
			}

			if diff := cmp.Diff(tt.want, messageChunk); diff != "" {
				t.Errorf("Request msgs diff (+got -want):\n%s", diff)
			}
		})
	}
}

func TestFormatterStreaming(t *testing.T) {
	modelInfo := ModelInfo{
		Label: modelName,
		Supports: &ModelSupports{
			Multiturn:   true,
			Tools:       true,
			SystemRole:  true,
			Media:       false,
			Constrained: ConstrainedSupportAll,
		},
		Versions: []string{"echo-001", "echo-002"},
	}

	type TestJson struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}

	type TestEnum struct {
		FavColor string `json:"fav_color,omitempty" jsonschema:"enum=red,enum=green,enum=blue"`
	}

	tests := []struct {
		name       string
		format     string
		outputtype any
		response   *ModelResponseChunk
		want       *ModelResponseChunk
	}{
		{
			name:       "JSON streaming",
			format:     "json",
			outputtype: TestJson{},
			response: &ModelResponseChunk{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart(JSONMarkdown(`{"name": "John", "age": 19}`)),
				},
			},
			want: &ModelResponseChunk{

				Role: RoleModel,
				Content: []*Part{
					NewJSONPart("{\"name\": \"John\", \"age\": 19}"),
				},
			},
		},
		{
			name:   "Text streaming",
			format: "text",
			response: &ModelResponseChunk{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("Hello"),
				},
			},
			want: &ModelResponseChunk{

				Role: RoleModel,
				Content: []*Part{
					NewTextPart("Hello"),
				},
			},
		},
		{
			name:       "JSONL streaming",
			format:     "jsonl",
			outputtype: []TestJson{},
			response: &ModelResponseChunk{

				Role: RoleModel,
				Content: []*Part{
					NewTextPart(`{"name": "John", "age": 19}\n`),
				},
			},
			want: &ModelResponseChunk{
				Role: RoleModel,
				Content: []*Part{
					NewJSONPart(`{"name": "John", "age": 19}`),
				},
			},
		},
		{
			name:       "array streaming",
			format:     "array",
			outputtype: []TestJson{},
			response: &ModelResponseChunk{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart(`[{"name": "John", "age": 19,`),
				},
			},
			want: &ModelResponseChunk{
				Role: RoleModel,
				Content: []*Part{
					NewJSONPart(`[{"name": "John", "age": 19}]`),
				},
			},
		},
		{
			name:       "parses simple enum value",
			format:     "enum",
			outputtype: TestEnum{},
			response: &ModelResponseChunk{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("red"),
				},
			},
			want: &ModelResponseChunk{
				Role: RoleModel,
				Content: []*Part{
					NewTextPart("red"),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			formatModel := DefineModel(r, "test", tt.name, &modelInfo, func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
				if msc != nil {
					msc(ctx, tt.response)
				}

				return &ModelResponse{
					Request: gr,
					Message: NewModelTextMessage(tt.response.Text()),
				}, nil
			})

			streamText := ""

			var options []GenerateOption
			options = append(options, WithOutputFormat(tt.format))
			options = append(options, WithModel(formatModel))
			options = append(options, WithSystem("You are a helpful assistant."))
			options = append(options, WithStreaming(func(ctx context.Context, grc *ModelResponseChunk) error {
				streamText += grc.Text()
				return nil
			}))

			if tt.outputtype != nil {
				options = append(options, WithOutputType(tt.outputtype))
			}

			_, err := Generate(context.Background(), r, options...)
			if err != nil {
				t.Fatal(err)
			}

			if diff := cmp.Diff(streamText, tt.want.Text()); diff != "" {
				t.Errorf("Text() diff (+got -want):\n%s", diff)
			}
		})
	}
}
