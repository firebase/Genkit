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

package milvus

import (
	"context"
	"errors"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

type Milvus struct {
	mu      sync.Mutex    // Mutex to control access.
	initted bool          // Whether the plugin has been initialized.
	Engine  *MilvusEngine // Client for the Milvus database.
}

// Name returns the name of the plugin.
func (m *Milvus) Name() string {
	return provider
}

// Init initializes the Milvus plugin.
func (m *Milvus) Init(ctx context.Context) []api.Action {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.initted {
		panic("milvus.Init: plugin already initialized")
	}

	if m.Engine == nil {
		panic("milvus.Init: Engine not set")
	}

	if m.Engine.client == nil {
		panic("milvus.Init: client not set")
	}

	m.initted = true
	return []api.Action{}
}

type CollectionConfig struct {
	// Name is the Milvus collection name.
	Name string
	// IdKey is the column name that stores a document identifier.
	IdKey string
	// ScoreKey is the metadata key used to store similarity scores in results.
	ScoreKey string
	// VectorKey is the column name that stores the embedding vector.
	VectorKey string
	// TextKey is the column name that stores the original document text.
	TextKey string
	// VectorDim is the dimensionality of the embedding vectors.
	VectorDim int
	// Embedder is the embedding model used to embed queries and documents.
	Embedder ai.Embedder
	// EmbedderOptions are passed to the embedder when generating embeddings.
	EmbedderOptions any
}

// DefineRetriever defines a Retriever with the given configuration.
func DefineRetriever(_ context.Context, g *genkit.Genkit, cfg *CollectionConfig, opts *ai.RetrieverOptions) (*DocStore, ai.Retriever, error) {
	m, ok := genkit.LookupPlugin(g, provider).(*Milvus)
	if !ok {
		return nil, nil, errors.New("milvus plugin not found; did you call genkit.Init with the milvus plugin")
	}

	ds, err := m.newDocStore(cfg)
	if err != nil {
		return nil, nil, err
	}
	return ds, genkit.DefineRetriever(g, api.NewName(provider, cfg.Name), opts, ds.Retrieve), nil
}
