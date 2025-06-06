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
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	partialparser "github.com/blaze2305/partial-json-parser"
	"github.com/blaze2305/partial-json-parser/options"
	"github.com/firebase/genkit/go/internal/base"
)

type jsonlFormatter struct{}

// Name returns the name of the formatter.
func (j jsonlFormatter) Name() string {
	return OutputFormatJSONL
}

// Handler returns a new formatter handler for the given schema.
func (j jsonlFormatter) Handler(schema map[string]any) (FormatHandler, error) {
	if schema == nil || !base.ValidateIsJSONArray(schema) {
		return nil, fmt.Errorf("schema is not valid JSONL")
	}

	jsonBytes, err := json.Marshal(schema["items"])
	if err != nil {
		return nil, fmt.Errorf("error marshalling schema to JSONL: %w", err)
	}

	instructions := fmt.Sprintf("Output should be JSONL format, a sequence of JSON objects (one per line) separated by a newline '\\n' character. Each line should be a JSON object conforming to the following schema:\n\n```%s```", string(jsonBytes))

	handler := &jsonlHandler{
		instructions: instructions,
		config: ModelOutputConfig{
			Format:      OutputFormatJSONL,
			Schema:      schema,
			ContentType: "application/jsonl",
		},
	}

	return handler, nil
}

type jsonlHandler struct {
	instructions  string
	config        ModelOutputConfig
	previousParts []*Part
}

// Instructions returns the instructions for the formatter.
func (j jsonlHandler) Instructions() string {
	return j.instructions
}

// Config returns the output config for the formatter.
func (j jsonlHandler) Config() ModelOutputConfig {
	return j.config
}

// StreamCallback handler for streaming formatted responses
func (j jsonlHandler) StreamCallback(cb ModelStreamCallback) ModelStreamCallback {
	return func(ctx context.Context, mrc *ModelResponseChunk) error {
		j.previousParts = append(j.previousParts, mrc.Content...)
		mrc.Content = j.previousParts

		parsed, err := j.ParseChunk(mrc)
		if err != nil {
			return err
		}

		return cb(ctx, parsed)
	}
}

// ParseMessage parses the message and returns the formatted message.
func (j jsonlHandler) ParseMessage(m *Message) (*Message, error) {
	if j.config.Format == OutputFormatJSONL {
		if m == nil {
			return nil, errors.New("message is empty")
		}
		if len(m.Content) == 0 {
			return nil, errors.New("message has no content")
		}

		var newParts []*Part
		for _, part := range m.Content {
			if !part.IsText() {
				newParts = append(newParts, part)
			} else {
				lines := base.GetJsonLines(part.Text, "{")
				for _, line := range lines {
					if j.config.Schema != nil {
						var schemaBytes []byte
						schemaBytes, err := json.Marshal(j.config.Schema["items"])
						if err != nil {
							return nil, fmt.Errorf("expected schema is not valid: %w", err)
						}
						if err = base.ValidateRaw([]byte(line), schemaBytes); err != nil {
							return nil, err
						}
					}

					newParts = append(newParts, NewJSONPart(line))
				}
			}
		}
		m.Content = newParts
	}

	return m, nil
}

// ParseChunk parse the chunk and returns a new formatted chunk.
func (j jsonlHandler) ParseChunk(c *ModelResponseChunk) (*ModelResponseChunk, error) {
	if j.config.Format == OutputFormatJSONL {
		if c == nil {
			return nil, errors.New("chunk is empty")
		}
		if len(c.Content) == 0 {
			return nil, errors.New("chunk has no content")
		}

		// Get all chunks streamed so far
		text := c.Text()

		startIndex := 0
		// If there are previous chunks, adjust startIndex based on the last newline
		// in the previous text to ensure complete lines are processed from the accumulatedText.
		noParts := len(c.Content)
		if c.Content != nil && noParts > 1 {
			var sb strings.Builder
			i := 0
			for i < noParts-1 {
				sb.WriteString(c.Content[i].Text)
				i++
			}

			previousText := sb.String()
			lastNewline := strings.LastIndex(previousText, `\n`)

			if lastNewline != -1 {
				// Exclude the newline
				startIndex = lastNewline + 2
			}
		}

		text = text[startIndex:]

		var newParts []*Part
		lines := base.GetJsonLines(text, "{")
		for _, line := range lines {
			if line != "" {
				var err error
				line, err = partialparser.ParseMalformedString(line, options.ALL, false)
				if err != nil {
					return nil, errors.New("message is not a valid JSON")
				}

				newParts = append(newParts, NewJSONPart(line))
			}
		}

		c.Content = newParts
	}
	return c, nil
}
