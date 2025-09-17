package client

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestNewClient(t *testing.T) {
	tests := []struct {
		name    string
		apiKey  string
		wantErr bool
	}{
		{
			name:    "valid api key",
			apiKey:  "test-key",
			wantErr: false,
		},
		{
			name:    "empty api key",
			apiKey:  "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewClient(tt.apiKey).Build()
			if (err != nil) != tt.wantErr {
				t.Errorf("NewClient() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && client == nil {
				t.Error("NewClient() returned nil client without error")
			}
		})
	}
}

func TestClientBuilder(t *testing.T) {
	tests := []struct {
		name      string
		builder   *ClientBuilder
		wantErr   bool
		wantURL   string
		wantOrgID string
		timeout   time.Duration
	}{
		{
			name: "valid options",
			builder: NewClient("test-key").
				WithBaseURL("https://custom.openai.com").
				WithTimeout(60 * time.Second).
				WithOrganization("org-123"),
			wantURL:   "https://custom.openai.com",
			wantOrgID: "org-123",
			timeout:   60 * time.Second,
		},
		{
			name:    "empty base URL",
			builder: NewClient("test-key").WithBaseURL(""),
			wantErr: true,
		},
		{
			name:    "negative timeout",
			builder: NewClient("test-key").WithTimeout(-1 * time.Second),
			wantErr: true,
		},
		{
			name:    "nil http client",
			builder: NewClient("test-key").WithHTTPClient(nil),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := tt.builder.Build()

			if (err != nil) != tt.wantErr {
				t.Errorf("Build() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if client.baseURL != tt.wantURL {
					t.Errorf("expected baseURL %q, got %q", tt.wantURL, client.baseURL)
				}
				if client.orgID != tt.wantOrgID {
					t.Errorf("expected orgID %q, got %q", tt.wantOrgID, client.orgID)
				}
				if tt.timeout != 0 && client.httpClient.Timeout != tt.timeout {
					t.Errorf("expected timeout %v, got %v", tt.timeout, client.httpClient.Timeout)
				}
			}
		})
	}
}

// mockSuccessResponse returns a mock successful chat response
func mockSuccessResponse(w http.ResponseWriter, _ *http.Request) {
	response := ChatResponse{
		ID: "test-id",
		Choices: []ChatChoice{
			{
				Message: ChatMessage{
					Role:    "assistant",
					Content: "Hello!",
				},
			},
		},
	}
	json.NewEncoder(w).Encode(response)
}

// mockErrorResponse returns a mock server error
func mockErrorResponse(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusInternalServerError)
	w.Write([]byte("internal server error"))
}

// verifyHeaders checks if the request contains expected headers
func verifyHeaders(t *testing.T, r *http.Request) {
	if r.Header.Get("Authorization") != "Bearer test-key" {
		t.Errorf("expected Authorization header 'Bearer test-key', got %q", r.Header.Get("Authorization"))
	}
	if r.Header.Get("Content-Type") != "application/json" {
		t.Errorf("expected Content-Type header 'application/json', got %q", r.Header.Get("Content-Type"))
	}
}

func TestChatBuilder(t *testing.T) {
	tests := []struct {
		name          string
		setupChat     func(*Client) *ChatBuilder
		wantErr       bool
		wantResponse  string
		serverHandler http.HandlerFunc
	}{
		{
			name: "successful chat completion",
			setupChat: func(c *Client) *ChatBuilder {
				return c.NewChat("gpt-3.5-turbo").
					AddMessage("user", "Hi").
					WithTemperature(0.7).
					WithMaxCompletionTokens(100)
			},
			wantErr:       false,
			wantResponse:  "Hello!",
			serverHandler: mockSuccessResponse,
		},
		{
			name: "empty model",
			setupChat: func(c *Client) *ChatBuilder {
				return c.NewChat("")
			},
			wantErr: true,
		},
		{
			name: "invalid temperature",
			setupChat: func(c *Client) *ChatBuilder {
				return c.NewChat("gpt-3.5-turbo").
					WithTemperature(3.0)
			},
			wantErr: true,
		},
		{
			name: "invalid role",
			setupChat: func(c *Client) *ChatBuilder {
				return c.NewChat("gpt-3.5-turbo").
					AddMessage("invalid-role", "content")
			},
			wantErr: true,
		},
		{
			name: "no messages",
			setupChat: func(c *Client) *ChatBuilder {
				return c.NewChat("gpt-3.5-turbo")
			},
			wantErr: true,
		},
		{
			name: "server error",
			setupChat: func(c *Client) *ChatBuilder {
				return c.NewChat("gpt-3.5-turbo").
					AddMessage("user", "Hi")
			},
			wantErr:       true,
			serverHandler: mockErrorResponse,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a new test server for each test case
			handler := tt.serverHandler
			if handler == nil {
				handler = func(w http.ResponseWriter, r *http.Request) {
					verifyHeaders(t, r)
				}
			}
			server := httptest.NewServer(http.HandlerFunc(handler))
			defer server.Close()

			// Create a new client with the test server URL
			client, err := NewClient("test-key").
				WithBaseURL(server.URL).
				Build()
			if err != nil {
				t.Fatalf("failed to create client: %v", err)
			}

			// Setup and execute the chat
			chat := tt.setupChat(client)
			resp, err := chat.Execute(context.Background())

			if (err != nil) != tt.wantErr {
				t.Errorf("Execute() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && tt.wantResponse != "" {
				if resp.Choices[0].Message.Content != tt.wantResponse {
					t.Errorf("expected response content %q, got %q", tt.wantResponse, resp.Choices[0].Message.Content)
				}
			}
		})
	}
}

func TestChatBuilderStream(t *testing.T) {
	tests := []struct {
		name          string
		setupChat     func(*Client) *ChatBuilder
		wantErr       bool
		serverHandler http.HandlerFunc
	}{
		{
			name: "successful stream",
			setupChat: func(c *Client) *ChatBuilder {
				return c.NewChat("gpt-3.5-turbo").
					AddMessage("user", "Hi")
			},
			serverHandler: func(w http.ResponseWriter, r *http.Request) {
				// Verify request headers and stream flag
				verifyHeaders(t, r)

				var req struct {
					Stream bool `json:"stream"`
				}
				json.NewDecoder(r.Body).Decode(&req)
				if !req.Stream {
					t.Error("expected stream flag to be true")
				}

				// Write format response
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprintf(w, "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n")
				fmt.Fprintf(w, "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"!\"}}]}\n\n")
				fmt.Fprintf(w, "data: [DONE]\n\n")
			},
		},
		{
			name: "empty model",
			setupChat: func(c *Client) *ChatBuilder {
				return c.NewChat("")
			},
			wantErr: true,
		},
		{
			name: "no messages",
			setupChat: func(c *Client) *ChatBuilder {
				return c.NewChat("gpt-3.5-turbo")
			},
			wantErr: true,
		},
		{
			name: "server error",
			setupChat: func(c *Client) *ChatBuilder {
				return c.NewChat("gpt-3.5-turbo").
					AddMessage("user", "Hi")
			},
			wantErr:       true,
			serverHandler: mockErrorResponse,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := tt.serverHandler
			if handler == nil {
				handler = func(w http.ResponseWriter, r *http.Request) {
					verifyHeaders(t, r)
				}
			}
			server := httptest.NewServer(http.HandlerFunc(handler))
			defer server.Close()

			client, err := NewClient("test-key").
				WithBaseURL(server.URL).
				Build()
			if err != nil {
				t.Fatalf("failed to create client: %v", err)
			}

			stream, err := tt.setupChat(client).ExecuteStream(context.Background())
			if (err != nil) != tt.wantErr {
				t.Errorf("ExecuteStream() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && stream == nil {
				t.Error("expected non-nil stream")
			}

			if stream != nil {
				stream.Close()
			}
		})
	}
}
