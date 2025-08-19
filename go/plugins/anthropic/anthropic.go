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
	"context"
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	ant "github.com/firebase/genkit/go/plugins/internal/anthropic"
)

const (
	provider             = "anthropic"
	anthropicLabelPrefix = "Anthropic"
)

type Anthropic struct {
	APIKey  string           // If not provided, defaults to ANTHROPIC_API_KEY
	aclient anthropic.Client // Anthropic client
	mu      sync.Mutex       // Mutex to control access
	initted bool             // Whether the plugin has been initialized
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

	apiKey := a.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	if apiKey == "" {
		return fmt.Errorf("Anthropic requires setting ANTHROPIC_API_KEY in the environment")
	}

	ac := anthropic.NewClient(
		option.WithAPIKey(apiKey),
	)
	a.aclient = ac
	a.initted = true

	return nil
}

func (a *Anthropic) DefineModel(g *genkit.Genkit, name string, info *ai.ModelInfo) ai.Model {
	return ant.DefineModel(g, a.aclient, provider, name, *info)
}

func (a *Anthropic) IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupModel(g, provider, name) != nil
}

func (a *Anthropic) ListActions(ctx context.Context) []core.ActionDesc {
	actions := []core.ActionDesc{}

	models, err := listModels(ctx, &a.aclient)
	if err != nil {
		return nil
	}

	for _, name := range models {
		metadata := map[string]any{
			"model": map[string]any{
				"supports": map[string]any{
					"media":       true,
					"multiturn":   true,
					"systemRole":  true,
					"tools":       true,
					"toolChoice":  true,
					"constrained": "no-tools",
				},
			},
			"versions": []string{},
			"stage":    string(ai.ModelStageStable),
		}
		metadata["label"] = fmt.Sprintf("%s - %s", anthropicLabelPrefix, name)

		actions = append(actions, core.ActionDesc{
			Type:     core.ActionTypeModel,
			Name:     fmt.Sprintf("%s/%s", provider, name),
			Key:      fmt.Sprintf("/%s/%s/%s", core.ActionTypeModel, provider, name),
			Metadata: metadata,
		})

	}
	return actions
}

func (a *Anthropic) ResolveAction(g *genkit.Genkit, atype core.ActionType, name string) error {
	switch atype {
	case core.ActionTypeModel:
		ant.DefineModel(g, a.aclient, provider, name, ai.ModelInfo{
			Label:    fmt.Sprintf("%s - %s", anthropicLabelPrefix, name),
			Stage:    ai.ModelStageStable,
			Versions: []string{},
			Supports: &ai.ModelSupports{
				Media:       true,
				Multiturn:   true,
				SystemRole:  true,
				Tools:       true,
				ToolChoice:  true,
				Constrained: ai.ConstrainedSupportNoTools,
			},
		})
	}
	return nil
}
