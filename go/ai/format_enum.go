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
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strings"
)

type enumFormatter struct{}

// Name returns the name of the formatter.
func (e enumFormatter) Name() string {
	return OutputFormatEnum
}

// Handler returns a new formatter handler for the given schema.
func (e enumFormatter) Handler(schema map[string]any) (FormatHandler, error) {
	enums := objectEnums(schema)
	if schema == nil || len(enums) == 0 {
		return nil, fmt.Errorf("schema is not valid JSON enum")
	}

	instructions := fmt.Sprintf("Output should be ONLY one of the following enum values. Do not output any additional information or add quotes.\n\n```%s```", strings.Join(enums, "\n"))

	handler := &enumHandler{
		instructions: instructions,
		config: ModelOutputConfig{
			Format:      OutputFormatEnum,
			Schema:      schema,
			ContentType: "text/enum",
		},
		enums: enums,
	}

	return handler, nil
}

type enumHandler struct {
	instructions  string
	config        ModelOutputConfig
	enums         []string
	previousParts []*Part
}

// Instructions returns the instructions for the formatter.
func (e enumHandler) Instructions() string {
	return e.instructions
}

// Config returns the output config for the formatter.
func (e enumHandler) Config() ModelOutputConfig {
	return e.config
}

// StreamCallback handler for streaming formatted responses
func (e enumHandler) StreamCallback(cb ModelStreamCallback) ModelStreamCallback {
	return func(ctx context.Context, mrc *ModelResponseChunk) error {
		e.previousParts = append(e.previousParts, mrc.Content...)
		mrc.Content = e.previousParts

		parsed, err := e.ParseChunk(mrc)
		if err != nil {
			return err
		}

		return cb(ctx, parsed)
	}
}

// ParseMessage parses the message and returns the formatted message.
func (e enumHandler) ParseMessage(m *Message) (*Message, error) {
	if e.config.Format == OutputFormatEnum {
		if m == nil {
			return nil, errors.New("message is empty")
		}
		if len(m.Content) == 0 {
			return nil, errors.New("message has no content")
		}

		var err error
		m.Content, err = getEnums(m.Content, e.enums)
		if err != nil {
			return nil, err
		}
	}

	return m, nil
}

// ParseChunk parse the chunk and returns a new formatted chunk.
func (e enumHandler) ParseChunk(c *ModelResponseChunk) (*ModelResponseChunk, error) {
	if e.config.Format == OutputFormatEnum {
		if c == nil {
			return nil, errors.New("chunk is empty")
		}
		if len(c.Content) == 0 {
			return nil, errors.New("chunk has no content")
		}

		var err error
		c.Content, err = getEnums(c.Content, e.enums)
		if err != nil {
			return nil, err
		}
	}

	return c, nil
}

// Get enum strings from json schema
func objectEnums(schema map[string]any) []string {
	var enums []string

	if properties, ok := schema["properties"].(map[string]any); ok {
		for _, propValue := range properties {
			if propMap, ok := propValue.(map[string]any); ok {
				if enumSlice, ok := propMap["enum"].([]any); ok {
					for _, enumVal := range enumSlice {
						if enumStr, ok := enumVal.(string); ok {
							enums = append(enums, enumStr)
						}
					}
				}
			}
		}
	}

	return enums
}

// Helper function to get matching enums from parts.
func getEnums(parts []*Part, enums []string) ([]*Part, error) {
	for i, part := range parts {
		if !part.IsText() {
			continue
		}

		// replace single and double quotes
		re := regexp.MustCompile(`['"]`)
		clean := re.ReplaceAllString(part.Text, "")

		// trim whitespace
		trimmed := strings.TrimSpace(clean)

		if !slices.Contains(enums, trimmed) {
			return nil, fmt.Errorf("message %s not in list of valid enums: %s", trimmed, strings.Join(enums, ", "))
		}

		parts[i] = NewTextPart(trimmed)
	}

	return parts, nil
}
