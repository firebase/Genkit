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
)

type textFormatter struct{}

// Name returns the name of the formatter.
func (t textFormatter) Name() string {
	return OutputFormatText
}

// Handler returns a new formatter handler for the given schema.
func (t textFormatter) Handler(schema map[string]any) (FormatHandler, error) {
	handler := &textHandler{
		config: ModelOutputConfig{
			ContentType: "text/plain",
		},
	}

	return handler, nil
}

type textHandler struct {
	instructions string
	config       ModelOutputConfig
}

// Config returns the output config for the formatter.
func (t textHandler) Config() ModelOutputConfig {
	return t.config
}

// Instructions returns the instructions for the formatter.
func (t textHandler) Instructions() string {
	return t.instructions
}

// StreamCallback handler for streaming formatted responses
func (t textHandler) StreamCallback(cb ModelStreamCallback) ModelStreamCallback {
	return func(ctx context.Context, mrc *ModelResponseChunk) error {
		return cb(ctx, mrc)
	}
}

// ParseMessage parses the message and returns the formatted message.
func (t textHandler) ParseMessage(m *Message) (*Message, error) {
	return m, nil
}

// ParseChunk parse the chunk and returns a new formatted chunk.
func (t textHandler) ParseChunk(c *ModelResponseChunk) (*ModelResponseChunk, error) {
	return c, nil
}
