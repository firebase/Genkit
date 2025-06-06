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

	partialparser "github.com/blaze2305/partial-json-parser"
	"github.com/blaze2305/partial-json-parser/options"
	"github.com/firebase/genkit/go/internal/base"
)

type jsonFormatter struct{}

// Name returns the name of the formatter.
func (j jsonFormatter) Name() string {
	return OutputFormatJSON
}

// Handler returns a new formatter handler for the given schema.
func (j jsonFormatter) Handler(schema map[string]any) (FormatHandler, error) {
	var instructions string
	if schema != nil {
		jsonBytes, err := json.Marshal(schema)
		if err != nil {
			return nil, fmt.Errorf("error marshalling schema to JSON: %w", err)
		}

		instructions = fmt.Sprintf("Output should be in JSON format and conform to the following schema:\n\n```%s```", string(jsonBytes))
	}

	handler := &jsonHandler{
		instructions: instructions,
		config: ModelOutputConfig{
			Format:      OutputFormatJSON,
			Schema:      schema,
			ContentType: "application/json",
		},
	}

	return handler, nil
}

// jsonHandler is a handler for the JSON formatter.
type jsonHandler struct {
	instructions  string
	config        ModelOutputConfig
	previousParts []*Part
}

// Instructions returns the instructions for the formatter.
func (j jsonHandler) Instructions() string {
	return j.instructions
}

// Config returns the output config for the formatter.
func (j jsonHandler) Config() ModelOutputConfig {
	return j.config
}

// StreamCallback handler for streaming formatted responses
func (j jsonHandler) StreamCallback(cb ModelStreamCallback) ModelStreamCallback {
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
func (j jsonHandler) ParseMessage(m *Message) (*Message, error) {
	if j.config.Format == OutputFormatJSON {
		if m == nil {
			return nil, errors.New("message is empty")
		}
		if len(m.Content) == 0 {
			return nil, errors.New("message has no content")
		}

		for i, part := range m.Content {
			if !part.IsText() {
				continue
			}

			text := base.ExtractJSONFromMarkdown(part.Text)
			if j.config.Schema != nil {
				var schemaBytes []byte
				schemaBytes, err := json.Marshal(j.config.Schema)
				if err != nil {
					return nil, fmt.Errorf("expected schema is not valid: %w", err)
				}
				if err = base.ValidateRaw([]byte(text), schemaBytes); err != nil {
					return nil, err
				}
			} else {
				if !base.ValidJSON(text) {
					return nil, errors.New("message is not a valid JSON")
				}
			}

			m.Content[i] = NewJSONPart(text)
		}
	}

	return m, nil
}

// ParseChunk parse the chunk and returns a new formatted chunk.
func (j jsonHandler) ParseChunk(c *ModelResponseChunk) (*ModelResponseChunk, error) {
	if j.config.Format == OutputFormatJSON {
		if c == nil {
			return nil, errors.New("chunk is empty")
		}

		if len(c.Content) == 0 {
			return nil, errors.New("chunk has no content")
		}

		// Get all chunks streamed so far
		text := c.Text()
		text = base.ExtractJSONFromMarkdown(text)
		// Try and extract a json object
		text = base.GetJsonObject(text)
		if text != "" {
			var err error
			text, err = partialparser.ParseMalformedString(text, options.ALL, false)
			if err != nil {
				return nil, errors.New("message is not a valid JSON")
			}
		} else {
			return nil, nil
		}

		c.Content = []*Part{NewJSONPart(text)}
	}
	return c, nil
}
