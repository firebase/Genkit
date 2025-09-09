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

package deepseek

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go/option"
)

const (
	provider = "deepseek"
	baseURL  = "https://api.deepseek.com/v1"
)

type DeepSeek struct {
	Opts             []option.RequestOption
	openAICompatible compat_oai.OpenAICompatible
}

func (d *DeepSeek) Name() string {
	return provider
}

func (d *DeepSeek) Init(ctx context.Context, g *genkit.Genkit) error {
	d.Opts = append(d.Opts, option.WithBaseURL(baseURL))

	d.openAICompatible.Opts = d.Opts
	d.openAICompatible.Provider = provider
	if err := d.openAICompatible.Init(ctx, g); err != nil {
		return err
	}

	return nil
}

func (d *DeepSeek) Model(g *genkit.Genkit, name string) ai.Model {
	return d.openAICompatible.Model(g, name, provider)
}

func (d *DeepSeek) DefineModel(g *genkit.Genkit, name string, info ai.ModelInfo) (ai.Model, error) {
	return d.openAICompatible.DefineModel(g, provider, name, info)
}

func (d *DeepSeek) ListActions(ctx context.Context) []core.ActionDesc {
	fmt.Printf("listing actions!!\n\n")
	return d.openAICompatible.ListActions(ctx)
}

func (d *DeepSeek) ResolveAction(g *genkit.Genkit, atype core.ActionType, name string) error {
	fmt.Printf("resolving actions!!\n\n")
	return d.openAICompatible.ResolveAction(g, atype, name)
}
