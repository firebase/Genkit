// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
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
	"log"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

func main() {
	ctx := context.Background()
	g, err := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.VertexAI{}))
	if err != nil {
		log.Fatal(err)
	}

	// Modify the state using tools
	chatTool := genkit.DefineTool(
		g,
		"updateName",
		"Call the updateName tool, when a user tell you their name",
		func(ctx *ai.ToolContext, input struct {
			Name string
		}) (string, error) {
			// Set name in state
			s, err := ai.SessionFromContext(ctx)
			if err != nil {
				return "", err
			}

			err = s.UpdateState(map[string]any{
				"username": input.Name,
			})
			if err != nil {
				return "", err
			}

			return "changed username to " + input.Name, nil
		},
	)

	StatefulGeneration(ctx, g, chatTool)
	StatefulPrompt(ctx, g, chatTool)
	ChatWithSession(ctx, g)
}

func StatefulGeneration(ctx context.Context, g *genkit.Genkit, chatTool ai.Tool) {
	m := googlegenai.VertexAIModel(g, "gemini-1.5-pro")

	// To include state in a session, you need to instantiate a session explicitly
	s, err := ai.NewSession(ctx,
		ai.WithSessionData(ai.SessionData{
			State: map[string]any{
				"username": "Michael",
			},
		},
		),
	)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := genkit.Generate(
		ctx,
		g,
		ai.WithModel(m),
		ai.WithSystem("You're Kitt from Knight Rider. Address the user as Kitt would and always introduce yourself."),
		ai.WithSession(*s),
		ai.WithTools(chatTool),
		ai.WithPrompt("Hello, my name is Earl"),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Print(resp.Text())
	data, err := s.GetData()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Print(data.State["username"])
}

func StatefulPrompt(ctx context.Context, g *genkit.Genkit, chatTool ai.Tool) {
	m := googlegenai.VertexAIModel(g, "gemini-1.5-pro")

	// To override default in-mem session storage
	store := &MyOwnSessionStore{
		SessionData: make(map[string]ai.SessionData),
	}

	s, err := ai.NewSession(ctx,
		ai.WithSessionData(ai.SessionData{
			State: map[string]any{
				"username": "Michael",
			},
		},
		),
		ai.WithSessionStore(store),
	)
	if err != nil {
		log.Fatal(err)
	}

	prompt, err := genkit.DefinePrompt(
		g,
		"StatefulPrompt",
		ai.WithModel(m),
		ai.WithSystem("You're Kitt from Knight Rider. Address the user as Kitt would and always introduce yourself."),
		ai.WithTools(chatTool),
		ai.WithPrompt("Hello, my name is Earl"),
	)
	if err != nil {
		log.Fatal(err)
	}

	resp, err := prompt.Execute(ctx, ai.WithSession(*s))

	fmt.Print(resp.Text())
	data, err := s.GetData()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Print(data.State["username"])
}

func ChatWithSession(ctx context.Context, g *genkit.Genkit) {
	m := googlegenai.VertexAIModel(g, "gemini-1.5-pro")

	// Simulate an existing persisted session
	store := &ai.InMemorySessionStore{}
	sessionID := DummyPersistedHistory(ctx, store)

	// Load session from store
	s, err := ai.LoadSession(ctx, sessionID, store)
	if err != nil {
		log.Fatal(err)
	}

	data, err := s.GetData()
	if err != nil {
		log.Fatal(err)
	}
	messages := data.Threads["default"]

	resp, err := genkit.Generate(
		ctx,
		g,
		ai.WithModel(m),
		ai.WithSystem("You are a helpful assistant."),
		ai.WithSession(*s),
		ai.WithMessages(messages...),
		ai.WithPrompt("What is my name?"),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Print(resp.Text())

	messages = append(messages, resp.Message)
	s.UpdateMessages("default", messages)
	if err != nil {
		log.Fatal(err)
	}
}

func DummyPersistedHistory(ctx context.Context, store *ai.InMemorySessionStore) string {
	history := ai.SessionData{
		Threads: make(map[string][]*ai.Message),
	}
	history.Threads["default"] = []*ai.Message{
		{
			Role:    ai.RoleUser,
			Content: []*ai.Part{ai.NewTextPart("my name is Earl")},
		},
	}

	s, err := ai.NewSession(
		ctx,
		ai.WithSessionData(history),
		ai.WithSessionStore(store),
	)
	if err != nil {
		log.Fatal(err)
	}

	return s.GetID()
}

type MyOwnSessionStore struct {
	SessionData map[string]ai.SessionData
}

func (s *MyOwnSessionStore) Get(sessionId string) (data ai.SessionData, err error) {
	d, err := os.ReadFile("/tmp/" + sessionId)
	if err != nil {
		return data, err
	}
	err = json.Unmarshal(d, &data)
	if err != nil {
		return data, err
	}

	if data.Threads == nil {
		data.Threads = make(map[string][]*ai.Message)
	}

	s.SessionData[sessionId] = data
	return s.SessionData[sessionId], nil
}

func (s *MyOwnSessionStore) Save(sessionId string, data ai.SessionData) error {
	s.SessionData[sessionId] = data
	d, err := json.Marshal(data)
	if err != nil {
		return err
	}
	err = os.WriteFile("/tmp/"+sessionId, []byte(d), 0644)
	if err != nil {
		return err
	}
	return nil
}
