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
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/internal/registry"
)

type hello struct {
	Name string `json:"name"`
}

var sessionTool = getNameTool(r)
var sessionModel = getSessionModel(r)

func getNameTool(r *registry.Registry) Tool {
	return DefineTool(r, "updateName",
		"use this tool to update the name of the user",
		func(ctx *ToolContext, input struct {
			Name string
		}) (string, error) {
			// Set name in state
			session, err := SessionFromContext(ctx)
			if err != nil {
				return "", err
			}

			err = session.UpdateState(input)
			if err != nil {
				return "", err
			}

			return "Hello, my name is " + input.Name, nil
		},
	)
}

func getSessionModel(r *registry.Registry) Model {
	return DefineModel(r, "test", "chat", &ModelInfo{
		Supports: &ModelSupports{
			Tools:      true,
			Multiturn:  true,
			SystemRole: true,
		},
	}, func(ctx context.Context, gr *ModelRequest, msc ModelStreamCallback) (*ModelResponse, error) {
		toolCalled := false
		for _, msg := range gr.Messages {
			if msg.Content[0].IsToolResponse() {
				toolCalled = true
			}
		}

		if !toolCalled && len(gr.Tools) == 1 {
			part := NewToolRequestPart(&ToolRequest{
				Name:  "updateName",
				Input: map[string]any{"Name": "Earl"},
			})

			return &ModelResponse{
				Request: gr,
				Message: &Message{
					Role:    RoleModel,
					Content: []*Part{part},
				},
			}, nil
		}

		if msc != nil {
			msc(ctx, &ModelResponseChunk{
				Content: []*Part{NewTextPart("3!")},
			})
			msc(ctx, &ModelResponseChunk{
				Content: []*Part{NewTextPart("2!")},
			})
			msc(ctx, &ModelResponseChunk{
				Content: []*Part{NewTextPart("1!")},
			})
		}

		textResponse := ""
		var contentTexts []string
		for _, m := range gr.Messages {
			if m.Role != RoleUser && m.Role != RoleModel {
				textResponse += fmt.Sprintf("%s: ", m.Role)
			}

			if m.Role == RoleTool {
				contentTexts = append(contentTexts, m.Content[0].ToolResponse.Output.(string))
			}

			for _, c := range m.Content {
				contentTexts = append(contentTexts, c.Text)
			}
		}

		textResponse += strings.Join(contentTexts, "; ")
		textResponse += "; config: " + base.PrettyJSONString(gr.Config)

		return &ModelResponse{
			Request: gr,
			Message: NewModelTextMessage(fmt.Sprintf("Echo: %s", textResponse)),
		}, nil
	})
}

func TestSession(t *testing.T) {
	ctx := context.Background()
	session, err := NewSession(ctx)
	if err != nil {
		t.Fatal(err.Error())
	}

	if session.id == "" {
		t.Error("session id is empty")
	}

	if session.store == nil {
		t.Error("session store is nil")
	}

	if session.data.Threads == nil {
		t.Error("session threads are nil")
	}

	_helper_check_stored_messages(t, session, "hello")
}

func TestDefaultInMemSessionStore(t *testing.T) {
	ss := InMemorySessionStore{
		data: make(map[string]SessionData),
	}
	ss.data["testID"] = SessionData{
		Threads: map[string][]*Message{
			"testThread": {NewUserTextMessage("testMessage")},
		},
		State: map[string]any{
			"state": "testState",
		},
	}

	// test found
	data, err := ss.Get("testID")
	if err != nil {
		t.Fatal(err.Error())
	}

	if data.Threads["testThread"][0].Content[0].Text != "testMessage" {
		t.Error("message not found")
	}

	if data.State["state"] != "testState" {
		t.Error("state not found")
	}

	// test not found
	_, err = ss.Get("notfound")
	if err == nil {
		t.Error("no error on not found")
	}
}

func TestMultiSessionsAndThreads(t *testing.T) {
	s1, err := _helper_session_with_stored_messages(map[string][]*Message{
		"thread1": {NewUserTextMessage("Hello 1")},
	})
	if err != nil {
		t.Fatal(err.Error())
	}

	s2, err := _helper_session_with_stored_messages(map[string][]*Message{
		"thread2": {NewUserTextMessage("Hello 2")},
	})
	if err != nil {
		t.Fatal(err.Error())
	}

	s3, err := _helper_session_with_stored_messages(map[string][]*Message{
		"thread3": {NewUserTextMessage("Hello 3")},
		"thread4": {NewUserTextMessage("Hello 4")},
	})
	if err != nil {
		t.Fatal(err.Error())
	}

	var tests = []struct {
		name    string
		session *Session
		data    map[string][]*Message
		found   bool
	}{
		{
			name:    "s1 thread, message 1",
			session: s1,
			data: map[string][]*Message{
				"thread1": {NewUserTextMessage("Hello 1")},
			},
			found: true,
		},
		{
			name:    "s2 thread, message 2",
			session: s2,
			data: map[string][]*Message{
				"thread2": {NewUserTextMessage("Hello 2")},
			},
			found: true,
		},
		{
			name:    "s2 message in s1",
			session: s1,
			data: map[string][]*Message{
				"thread3": {NewUserTextMessage("Hello 2")},
			},
			found: false,
		},
		{
			name:    "s3 multithread",
			session: s3,
			data: map[string][]*Message{
				"thread3": {NewUserTextMessage("Hello 3")},
				"thread4": {NewUserTextMessage("Hello 4")},
			},
			found: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			data, err := test.session.store.Get(test.session.id)
			if err != nil {
				t.Fatal(err.Error())
			}

			for thread, msgs := range test.data {
				_, ok := data.Threads[thread]
				if !test.found {
					if ok {
						t.Error("thread found")
					} else {
						continue
					}
				} else if test.found && !ok {
					t.Fatal("thread not found")
				}

				for id, msg := range msgs {
					if msg.Content[0].Text == data.Threads[thread][id].Content[0].Text {
						if !test.found {
							t.Error("message found")
						}
					} else if test.found {
						t.Error("message not found")
					}
				}
			}
		})
	}
}

func TestLoadSessionFromStore(t *testing.T) {
	ctx := context.Background()
	session, err := NewSession(ctx)
	if err != nil {
		t.Fatal(err.Error())
	}

	_helper_check_stored_messages(t, session, "hello")

	loadS, err := LoadSession(ctx, session.id, session.store)
	if err != nil {
		t.Fatal(err.Error())
	}

	_helper_check_stored_messages(t, loadS, "hello")
}

func TestSessionStateFormat(t *testing.T) {
	var tests = []struct {
		name         string
		state        any
		stateType    string
		defaultState map[string]any
		newState     any
		fail         bool
	}{
		{
			name:         "structInput",
			state:        hello{},
			stateType:    "struct",
			defaultState: map[string]any{"name": ""},
			newState:     hello{Name: "new world"},
			fail:         false,
		},
		{
			name:         "structInputWithDefaults",
			state:        hello{Name: "world"},
			stateType:    "struct",
			defaultState: map[string]any{"name": "world"},
			newState:     hello{Name: "new world"},
			fail:         false,
		},
		{
			name:         "stringInput",
			state:        "world",
			stateType:    "primitive",
			defaultState: map[string]any{"state": "world"},
			newState:     "new world",
			fail:         true,
		},
		{
			name:         "intInput",
			state:        1,
			stateType:    "primitive",
			defaultState: map[string]any{"state": 1},
			newState:     2,
			fail:         true,
		},
		{
			name:         "floatInput",
			state:        3.14159,
			stateType:    "primitive",
			defaultState: map[string]any{"state": 3.14159},
			newState:     2.71828,
			fail:         true,
		},
		{
			name:         "mapInput",
			stateType:    "map",
			state:        map[string]any{"name": "world"},
			defaultState: map[string]any{"name": "world"},
			newState:     map[string]any{"name": "new world"},
			fail:         true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			session, err := NewSession(
				context.Background(),
				WithSessionStateType(test.state),
			)
			if err != nil {
				want := "only structs are allowed as types"
				if test.fail && strings.Contains(err.Error(), want) {
					return
				}
				t.Fatal(err)
			}

			for k, v1 := range test.defaultState {
				if v2, ok := session.data.State[k]; !ok || v1 != v2 {
					fmt.Print(session.data.State)
					t.Errorf("default state not set")
				}
			}

			err = session.UpdateState(test.newState)
			if err != nil {
				t.Fatal(err)
			}

			switch test.stateType {
			case "struct":
				if reflect.DeepEqual(test.newState, session.data.State) {
					t.Errorf("state not updated")
				}
			case "primitive":
				if v, ok := session.data.State["state"]; !ok || v != test.newState {
					t.Errorf("state not updated")
				}
			case "map":
				for k, v1 := range test.newState.(map[string]any) {
					if v2, ok := session.data.State[k]; !ok || v1 != v2 {
						t.Errorf("state not updated")
					}
				}
			}
		})
	}
}

func TestSessionWithOptions(t *testing.T) {
	session, err := NewSession(
		context.Background(),
		WithSessionStore(&TestInMemSessionStore{
			SessionData: make(map[string]SessionData),
		}),
		WithSessionID("test"),
		WithSessionData(SessionData{
			State: map[string]any{
				"state": "testState",
			},
			Threads: map[string][]*Message{
				"test": {NewUserTextMessage("hello")},
			},
		}),
	)

	if err != nil {
		t.Fatal(err.Error())
	}

	if session.store == nil {
		t.Error("session store is nil")
	}

	store := reflect.TypeOf(session.store)
	if store.String() != "*ai.TestInMemSessionStore" {
		t.Errorf("default store not overwritten, got %s", store.String())
	}

	if session.id != "test" {
		t.Error("session id not overwritten")
	}

	data, err := session.store.Get(session.id)
	if err != nil {
		t.Fatal(err.Error())
	}

	if data.State["state"] != "testState" {
		t.Error("session state not overwritten")
	}

	_helper_check_stored_messages(t, session, "hello")
}

func TestSessionWithOptionsErrorHandling(t *testing.T) {
	var tests = []struct {
		name string
		with SessionOption
	}{
		{
			name: "WithSessionStore",
			with: WithSessionStore(&TestInMemSessionStore{
				SessionData: make(map[string]SessionData),
			}),
		},
		{
			name: "WithSessionID",
			with: WithSessionID("test"),
		},
		{
			name: "WithSessionData",
			with: WithSessionData(SessionData{
				State: map[string]any{
					"state": "testState",
				},
				Threads: map[string][]*Message{
					"test": {NewUserTextMessage("hello")},
				},
			}),
		},
		{
			name: "WithStateType",
			with: WithSessionStateType(hello{Name: "world"}),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := NewSession(
				context.Background(),
				test.with,
				test.with,
			)

			if err == nil {
				t.Errorf("%s could be set twice", test.name)
			}
		})
	}
}

func TestSessionFromContext(t *testing.T) {
	ctx := context.Background()
	session, err := NewSession(ctx)

	if err != nil {
		t.Fatal(err.Error())
	}

	ctx = session.SetContext(ctx)

	sCtx, err := SessionFromContext(ctx)
	if err != nil {
		t.Fatal(err.Error())
	}

	if sCtx.id != session.id {
		t.Error("session id not found in context")
	}
}

func TestSessionStateUpdateGenerate(t *testing.T) {
	ctx := context.Background()

	session, err := NewSession(ctx,
		WithSessionStateType(HelloPromptInput{}),
	)
	if err != nil {
		t.Fatal(err.Error())
	}

	res, err := Generate(
		ctx,
		r,
		WithModel(sessionModel),
		WithSystem("update state"),
		WithPrompt("What's your name?"),
		WithTools(sessionTool),
		WithSession(*session),
	)
	if err != nil {
		t.Fatal(err.Error())
	}

	want := "Hello, my name is Earl"
	if !strings.Contains(res.Text(), want) {
		t.Errorf("got %q want %q", res.Text(), want)
	}

	data, err := session.GetData()
	if err != nil {
		t.Fatal(err.Error())
	}
	if data.State["Name"] != "Earl" {
		t.Error("session state not set")
	}
}

func TestSessionStateUpdatePrompt(t *testing.T) {
	session, err := NewSession(context.Background(),
		WithSessionStateType(HelloPromptInput{}),
	)
	if err != nil {
		t.Fatal(err.Error())
	}
	p, err := DefinePrompt(
		r,
		"promptSession",
		WithModel(sessionModel),
		WithSystem("update state"),
		WithPrompt("What's your name?"),
		WithTools(sessionTool),
	)
	if err != nil {
		t.Fatal(err)
	}

	res, err := p.Execute(context.Background(), WithSession(*session))
	if err != nil {
		t.Fatal(err)
	}

	want := "Hello, my name is Earl"
	if !strings.Contains(res.Text(), want) {
		t.Errorf("got %q want %q", res.Text(), want)
	}

	data, err := session.GetData()
	if err != nil {
		t.Fatal(err.Error())
	}
	if data.State["Name"] != "Earl" {
		t.Error("session state not set")
	}
}

func _helper_session_with_stored_messages(threads map[string][]*Message) (*Session, error) {
	ctx := context.Background()
	session, err := NewSession(ctx)
	if err != nil {
		return nil, err
	}

	for thread, msgs := range threads {
		err = session.UpdateMessages(thread, msgs)
		if err != nil {
			return nil, err
		}
	}

	return session, nil
}

func _helper_check_stored_messages(t *testing.T, s *Session, msg string) {
	msgs := []*Message{NewUserTextMessage(msg)}
	err := s.UpdateMessages("default", msgs)
	if err != nil {
		t.Fatal(err.Error())
	}

	data, err := s.store.Get(s.id)
	if err != nil {
		t.Fatal(err.Error())
	}

	msgs, ok := data.Threads["default"]
	if !ok {
		t.Fatal("thread not found")
	}
	if msgs[0].Content[0].Text != msg {
		t.Error("message not found")
	}
}

type TestInMemSessionStore struct {
	SessionData map[string]SessionData
}

func (s *TestInMemSessionStore) Get(sessionId string) (data SessionData, err error) {
	return s.SessionData[sessionId], nil
}

func (s *TestInMemSessionStore) Save(sessionId string, data SessionData) error {
	s.SessionData[sessionId] = data
	return nil
}
