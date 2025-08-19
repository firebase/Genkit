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

package modelgarden

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	ant "github.com/firebase/genkit/go/plugins/internal/anthropic"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/vertex"
)

const (
	provider = "vertexai"
)

type Anthropic struct {
	ProjectID string
	Location  string

	client  anthropic.Client
	mu      sync.Mutex
	initted bool
}

func (a *Anthropic) Name() string {
	return provider
}

func (a *Anthropic) Init(ctx context.Context, g *genkit.Genkit) (err error) {
	if a == nil {
		a = &Anthropic{}
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	if a.initted {
		return errors.New("plugin already initialized")
	}
	defer func() {
		if err != nil {
			err = fmt.Errorf("Anthropic.Init: %w", err)
		}
	}()

	projectID := a.ProjectID
	if projectID == "" {
		projectID = os.Getenv("GOOGLE_CLOUD_PROJECT")
		if projectID == "" {
			return fmt.Errorf("Vertex AI Modelgarden requires setting GOOGLE_CLOUD_PROJECT in the environment. You can get a project ID at https://console.cloud.google.com/home/dashboard")
		}
	}

	location := a.Location
	if location == "" {
		location = os.Getenv("GOOGLE_CLOUD_LOCATION")
		if location == "" {
			location = os.Getenv("GOOGLE_CLOUD_REGION")
		}
		if location == "" {
			return fmt.Errorf("Vertex AI Modelgarden requires setting GOOGLE_CLOUD_LOCATION or GOOGLE_CLOUD_REGION in the environment. You can get a location at https://cloud.google.com/vertex-ai/docs/general/locations")
		}
	}

	c := anthropic.NewClient(
		vertex.WithGoogleAuth(context.Background(), location, projectID),
	)

	a.initted = true
	a.client = c

	for name, mi := range ant.AnthropicModels {
		ant.DefineModel(g, a.client, provider, name, mi)
	}

	return nil
}

// AnthropicModel returns the [ai.Model] with the given name.
// It returns nil if the model was not defined
func AnthropicModel(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, provider, name)
}

// DefineModel adds the model to the registry
func (a *Anthropic) DefineModel(g *genkit.Genkit, name string, info *ai.ModelInfo) (ai.Model, error) {
	var mi ai.ModelInfo
	if info == nil {
		var ok bool
		mi, ok = ant.AnthropicModels[name]
		if !ok {
			return nil, fmt.Errorf("%s.DefineModel: called with unknown model %q and nil ModelInfo", provider, name)
		}
	} else {
		mi = *info
	}
	return ant.DefineModel(g, a.client, provider, name, mi), nil
}
