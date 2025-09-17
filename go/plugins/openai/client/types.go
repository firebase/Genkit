package client

import (
	"io"
	"net/http"
)

// ChatRequest represents a request to the chat completions API
// See: https://platform.openai.com/docs/api-reference/chat/create
type ChatRequest struct {
	// Required
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`

	// Optional
	Temperature         float64  `json:"temperature,omitempty"`
	TopP                float64  `json:"top_p,omitempty"`
	N                   int      `json:"n,omitempty"`
	Stream              bool     `json:"stream,omitempty"`
	Stop                []string `json:"stop,omitempty"`
	MaxCompletionTokens int      `json:"max_completion_tokens,omitempty"`
}

// Role defines the possible roles in a chat conversation
// These roles are explictly defined in OpenAI's API
// https://platform.openai.com/docs/guides/text-generation#messages-and-roles
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// ChatMessage represents a message in the conversation
type ChatMessage struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
}

// ChatResponse represents a response from the chat completions API
// https://platform.openai.com/docs/api-reference/chat/object
type ChatResponse struct {
	ID                string       `json:"id"`
	Object            string       `json:"object"`  // "chat.completion"
	Created           int64        `json:"created"` // unix timestamp in seconds
	Model             string       `json:"model"`   // model used
	SystemFingerprint string       `json:"system_fingerprint"`
	Choices           []ChatChoice `json:"choices"`
	Usage             Usage        `json:"usage"`
}

type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	LogProbs     *LogProbs   `json:"logprobs,omitempty"`
	FinishReason string      `json:"finish_reason"`
}

type LogProbs struct {
	Content []ContentLogProb `json:"content"`
}

type ContentLogProb struct {
	Token       string    `json:"token"`
	LogProb     float64   `json:"logprob"`
	Bytes       []int     `json:"bytes,omitempty"`
	TopLogProbs []TopProb `json:"top_logprobs,omitempty"`
}

type TopProb struct {
	Token   string  `json:"token"`
	LogProb float64 `json:"logprob"`
	Bytes   []int   `json:"bytes,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Streaming types
type ChatCompletionStream struct {
	reader io.ReadCloser
	resp   *http.Response
}

type StreamChunk struct {
	Choices []StreamChoice `json:"choices"`
}

type StreamChoice struct {
	Delta struct {
		Content string `json:"content"`
	} `json:"delta"`
}
