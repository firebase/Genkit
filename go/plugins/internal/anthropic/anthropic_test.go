// Copyright 2025 Google LLC
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
//
// SPDX-License-Identifier: Apache-2.0

package anthropic

import (
	"reflect"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
)

func TestAnthropic(t *testing.T) {
	t.Run("to anthropic role", func(t *testing.T) {
		r, err := toAnthropicRole(ai.RoleModel)
		if err != nil {
			t.Error(err)
		}
		if r != anthropic.MessageParamRoleAssistant {
			t.Errorf("want: %q, got: %q", anthropic.MessageParamRoleAssistant, r)
		}
		r, err = toAnthropicRole(ai.RoleUser)
		if err != nil {
			t.Error(err)
		}
		if r != anthropic.MessageParamRoleUser {
			t.Errorf("want: %q, got: %q", anthropic.MessageParamRoleUser, r)
		}
		r, err = toAnthropicRole(ai.RoleSystem)
		if err == nil {
			t.Errorf("should have failed, got: %q", r)
		}
		r, err = toAnthropicRole(ai.RoleTool)
		if err != nil {
			t.Error(err)
		}
		if r != anthropic.MessageParamRoleAssistant {
			t.Errorf("want: %q, got: %q", anthropic.MessageParamRoleAssistant, r)
		}
		r, err = toAnthropicRole("unknown")
		if err == nil {
			t.Errorf("should have failed, got: %q", r)
		}
	})
}

func TestAnthropicConfig(t *testing.T) {
	emptyConfig := anthropic.MessageNewParams{}
	expectedConfig := anthropic.MessageNewParams{
		Temperature: anthropic.Float(1.0),
		TopK:        anthropic.Int(1),
	}

	tests := []struct {
		name           string
		inputReq       *ai.ModelRequest
		expectedConfig *anthropic.MessageNewParams
		expectedErr    string
	}{
		{
			name: "Input is anthropic.MessageNewParams struct",
			inputReq: &ai.ModelRequest{
				Config: anthropic.MessageNewParams{
					Temperature: anthropic.Float(1.0),
					TopK:        anthropic.Int(1),
				},
			},
			expectedConfig: &expectedConfig,
			expectedErr:    "",
		},
		{
			name: "Input is *anthropic.MessageNewParams struct",
			inputReq: &ai.ModelRequest{
				Config: &anthropic.MessageNewParams{
					Temperature: anthropic.Float(1.0),
					TopK:        anthropic.Int(1),
				},
			},
			expectedConfig: &expectedConfig,
			expectedErr:    "",
		},
		{
			name: "Input is map[string]any",
			inputReq: &ai.ModelRequest{
				Config: map[string]any{
					"temperature": 1.0,
					"top_k":       1,
				},
			},
			expectedConfig: &expectedConfig,
			expectedErr:    "",
		},
		{
			name: "Input is map[string]any (empty)",
			inputReq: &ai.ModelRequest{
				Config: map[string]any{},
			},
			expectedConfig: &emptyConfig,
			expectedErr:    "",
		},
		{
			name: "Input is nil",
			inputReq: &ai.ModelRequest{
				Config: nil,
			},
			expectedConfig: &emptyConfig,
			expectedErr:    "",
		},
		{
			name: "Input is an unexpected type",
			inputReq: &ai.ModelRequest{
				Config: 123,
			},
			expectedConfig: &emptyConfig,
			expectedErr:    "unexpected config type: int",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotConfig, err := configFromRequest(tt.inputReq)
			if tt.expectedErr != "" {
				if err == nil {
					t.Errorf("expecting error containing %q, got nil", tt.expectedErr)
				} else if !strings.Contains(err.Error(), tt.expectedErr) {
					t.Errorf("expecting error to contain %q, but got: %q", tt.expectedErr, err.Error())
				}
				return
			}
			if err != nil {
				t.Errorf("expected no error, got: %v", err)
			}
			if !reflect.DeepEqual(gotConfig, tt.expectedConfig) {
				t.Errorf("configFromRequest() got config = %+v, want %+v", gotConfig, tt.expectedConfig)
			}
		})
	}
}
