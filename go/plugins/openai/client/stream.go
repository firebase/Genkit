package client

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// Recv reads and returns the next message from the stream
// Returns io.EOF when the stream is finished
func (s *ChatCompletionStream) Recv() (string, error) {
	reader := bufio.NewReader(s.reader)

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			return "", fmt.Errorf("failed to read stream: %w", err)
		}

		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			return "", io.EOF
		}

		var chunk StreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return "", fmt.Errorf("failed to parse chunk: %w", err)
		}

		// Return content if present in the chunk
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			return chunk.Choices[0].Delta.Content, nil
		}
	}
}

// Close closes the stream
func (s *ChatCompletionStream) Close() error {
	return s.resp.Body.Close()
}
