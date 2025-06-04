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
	"sync"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/google/uuid"
)

// Sessions are by default stored in memory only.
// Implement the Store interface to persist state.
type SessionStore interface {
	Get(sessionId string) (data SessionData, err error)
	Save(sessionId string, data SessionData) error
}

type SessionData struct {
	State       map[string]any        `json:"state,omitempty"`       // Any state that should be stored
	StateSchema map[string]any        `json:"stateschema,omitempty"` // Schema for state variables
	Threads     map[string][]*Message `json:"threads,omitempty"`     // Messages by thread name
}

type Session struct {
	id    string       // The session id
	data  *SessionData // The data for the session
	store SessionStore // The store for the session, defaults to in-memory storage
}

var sessionKey = base.NewContextKey[*Session]() // A session key

// NewSession creates a new session with the provided options.
// If no store is provided, it defaults to in-memory storage.
func NewSession(ctx context.Context, opts ...SessionOption) (session *Session, err error) {
	s := &Session{}

	sessOpts := &sessionOptions{}
	for _, opt := range opts {
		if err := opt.applySession(sessOpts); err != nil {
			return nil, core.NewError(core.INVALID_ARGUMENT, "ai.NewSession: error applying options: %v", err)
		}
	}

	s.store = &InMemorySessionStore{}
	if sessOpts.Store != nil {
		s.store = sessOpts.Store
	}

	s.id = uuid.New().String()
	if sessOpts.ID != "" {
		s.id = sessOpts.ID
	}

	if sessOpts.Data != nil {
		s.data = sessOpts.Data
	}

	if s.data == nil {
		s.data = &SessionData{
			Threads: make(map[string][]*Message),
		}
	}

	// Set default state
	if sessOpts.Schema != nil {
		s.data.StateSchema = sessOpts.Schema
		state, err := getState(s.data.StateSchema, sessOpts.DefaultState)
		if err != nil {
			return nil, err
		}

		s.data.State = state
	}

	// Initialize session in store
	s.store.Save(s.id, *s.data)

	return s, nil
}

// Load loads sessiondata from store, and returns a session.
func LoadSession(ctx context.Context, sessionId string, store SessionStore) (session *Session, err error) {
	sessionData, err := store.Get(sessionId)
	if err != nil {
		return nil, err
	}

	session = &Session{
		id:    sessionId,
		data:  &sessionData,
		store: store,
	}

	return session, nil
}

// GetID returns the session id.
func (s *Session) GetID() string {
	return s.id
}

// GetData returns the sessions data
func (s *Session) GetData() (SessionData, error) {
	return s.store.Get(s.id)
}

// UpdateState takes any data and sets it as state in the store.
func (s *Session) UpdateState(state any) error {
	var err error

	// Ensure sessionData is up to date
	sessionData, err := s.store.Get(s.id)
	if err != nil {
		return err
	}
	s.data = &sessionData

	// Use schema to get state
	if s.data.StateSchema != nil {
		s.data.State, err = getState(s.data.StateSchema, state)
		if err != nil {
			return err
		}
	} else {
		data, err := json.Marshal(state)
		if err != nil {
			return err
		}

		err = json.Unmarshal(data, &s.data.State)
		if err != nil {
			return fmt.Errorf("genkit.UpdateState: %w, state doesn't match existing schema", err)
		}
	}

	return s.store.Save(s.id, *s.data)
}

// UpdateMessages takes a threadName and a slice of messages.
func (s *Session) UpdateMessages(threadName string, messages []*Message) error {
	// Ensure sessionData is up to date
	sessionData, err := s.store.Get(s.id)
	if err != nil {
		return err
	}
	s.data = &sessionData

	s.data.Threads[threadName] = messages
	return s.store.Save(s.id, *s.data)
}

// Set current session in context.
func (s *Session) SetContext(ctx context.Context) context.Context {
	return sessionKey.NewContext(ctx, s)
}

// Find current session in context, returns session if found.
func SessionFromContext(ctx context.Context) (session *Session, err error) {
	if s := sessionKey.FromContext(ctx); s != nil {
		return s, nil
	}

	return nil, errors.New("genkit.SessionFromContext: session not found")
}

// Helper function to derive state from schema.
func getState(schema map[string]any, state any) (map[string]any, error) {
	data, err := json.Marshal(state)
	if err != nil {
		return nil, err
	}

	err = json.Unmarshal(data, &schema)
	if err != nil {
		return nil, fmt.Errorf("genkit.getState: %w, only structs are allowed as types", err)
	}

	return schema, nil
}

// Default in-memory session store.
type InMemorySessionStore struct {
	data map[string]SessionData // The session data keyed by session id
	mu   sync.RWMutex
}

func (s *InMemorySessionStore) Get(sessionId string) (data SessionData, err error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if _, ok := s.data[sessionId]; !ok {
		return data, errors.New("genkit.InMemorySessionStore.Get: session not found")
	}
	return s.data[sessionId], nil
}

func (s *InMemorySessionStore) Save(sessionId string, data SessionData) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.data[sessionId]; !ok {
		s.data = make(map[string]SessionData)
	}
	s.data[sessionId] = data
	return nil
}
