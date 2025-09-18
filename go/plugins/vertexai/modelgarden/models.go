// Copyright 2025 Google LLC
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
//
// SPDX-License-Identifier: Apache-2.0

package modelgarden

import (
	"github.com/firebase/genkit/go/ai"
)

const provider = "vertexai"

// supported anthropic models
var anthropicModels = map[string]ai.ModelOptions{
	"claude-3-5-sonnet-v2": {
		Label: "Vertex AI Model Garden - Claude 3.5 Sonnet",
		Supports: &ai.ModelSupports{
			Multiturn:   true,
			Tools:       true,
			ToolChoice:  true,
			SystemRole:  true,
			Media:       true,
			Constrained: ai.ConstrainedSupportNone,
		},
		Versions: []string{"claude-3-5-sonnet-v2@20241022"},
	},
	"claude-3-5-sonnet": {
		Label: "Vertex AI Model Garden - Claude 3.5 Sonnet",
		Supports: &ai.ModelSupports{
			Multiturn:   true,
			Tools:       true,
			ToolChoice:  true,
			SystemRole:  true,
			Media:       true,
			Constrained: ai.ConstrainedSupportNone,
		},

		Versions: []string{"claude-3-5-sonnet@20240620"},
	},
	"claude-3-sonnet": {
		Label: "Vertex AI Model Garden - Claude 3 Sonnet",
		Supports: &ai.ModelSupports{
			Multiturn:   true,
			Tools:       true,
			ToolChoice:  true,
			SystemRole:  true,
			Media:       true,
			Constrained: ai.ConstrainedSupportNone,
		},

		Versions: []string{"claude-3-sonnet@20240229"},
	},
	"claude-3-haiku": {
		Label: "Vertex AI Model Garden - Claude 3 Haiku",
		Supports: &ai.ModelSupports{
			Multiturn:   true,
			Tools:       true,
			ToolChoice:  true,
			SystemRole:  true,
			Media:       true,
			Constrained: ai.ConstrainedSupportNone,
		},

		Versions: []string{"claude-3-haiku@20240307"},
	},
	"claude-3-opus": {
		Label: "Vertex AI Model Garden - Claude 3 Opus",
		Supports: &ai.ModelSupports{
			Multiturn:   true,
			Tools:       true,
			ToolChoice:  true,
			SystemRole:  true,
			Media:       true,
			Constrained: ai.ConstrainedSupportNone,
		},

		Versions: []string{"claude-3-opus@20240229"},
	},
	"claude-3-7-sonnet": {
		Label: "Vertex AI Model Garden - Claude 3.7 Sonnet",
		Supports: &ai.ModelSupports{
			Multiturn:   true,
			Tools:       true,
			ToolChoice:  true,
			SystemRole:  true,
			Media:       true,
			Constrained: ai.ConstrainedSupportNone,
		},

		Versions: []string{"claude-3-7-sonnet@20250219"},
	},
	"claude-opus-4": {
		Label: "Vertex AI Model Garden - Claude Opus 4",
		Supports: &ai.ModelSupports{
			Multiturn:   true,
			Tools:       true,
			ToolChoice:  true,
			SystemRole:  true,
			Media:       true,
			Constrained: ai.ConstrainedSupportNone,
		},

		Versions: []string{"claude-opus-4@20250514"},
	},
	"claude-sonnet-4": {
		Label: "Vertex AI Model Garden - Claude Sonnet 4",
		Supports: &ai.ModelSupports{
			Multiturn:   true,
			Tools:       true,
			ToolChoice:  true,
			SystemRole:  true,
			Media:       true,
			Constrained: ai.ConstrainedSupportNone,
		},

		Versions: []string{"claude-sonnet-4@20250514"},
	},
}
