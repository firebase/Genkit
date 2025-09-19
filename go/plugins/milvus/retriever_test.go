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
	"flag"
	"fmt"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/fakeembedder"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

var (
	testAddr       = flag.String("test-milvus-addr", "", "Milvus address to use for tests")
	testCollection = flag.String("test-milvus-collection", "", "Milvus collection to use for tests")
	testAPIKey     = flag.String("test-milvus-api-key", "", "Milvus API key to use for tests")
)

func TestRetriever(t *testing.T) {
	t.Run("Vector search", func(t *testing.T) {
		if *testAddr == "" {
			t.Skip("skipping test because -test-milvus-addr flag not used")
		}
		if *testCollection == "" {
			t.Skip("skipping test because -test-milvus-collection flag not used")
		}
		if *testAPIKey == "" {
			t.Skip("skipping test because -test-milvus-api-key flag not used")
		}

		ctx := context.Background()

		dim := 1536

		v1 := make([]float32, dim)
		v2 := make([]float32, dim)
		v3 := make([]float32, dim)
		for i := range v1 {
			v1[i] = float32(i)
			v2[i] = float32(i)
			v3[i] = float32(dim - i)
		}
		v2[0] = 1

		id1 := int64(100)
		id2 := int64(200)
		id3 := int64(300)

		d1 := ai.DocumentFromText("hello1", map[string]any{"id": id1})
		d2 := ai.DocumentFromText("hello2", map[string]any{"id": id2})
		d3 := ai.DocumentFromText("goodbye", map[string]any{"id": id3})

		embedder := fakeembedder.New()
		embedder.Register(d1, v1)
		embedder.Register(d2, v2)
		embedder.Register(d3, v3)

		engine, err := NewEngine(ctx, WithAddress(*testAddr), WithAPIKey(*testAPIKey))
		if err != nil {
			t.Fatalf("failed to create engine: %v", err)
		}
		defer func(engine *MilvusEngine, ctx context.Context) {
			err := engine.Close(ctx)
			if err != nil {
				t.Fatalf("failed to close engine: %v", err)
			}
		}(engine, ctx)

		_, err = engine.GetClient().
			Delete(ctx, milvusclient.NewDeleteOption("products").
				WithInt64IDs("id", []int64{id1, id2, id3}))
		if err != nil {
			t.Fatalf("failed to delete documents: %v", err)
		}

		m := &Milvus{
			Engine: engine,
		}

		g := genkit.Init(ctx, genkit.WithPlugins(m))

		emdOpts := &ai.EmbedderOptions{
			Dimensions: dim,
			Supports: &ai.EmbedderSupports{
				Input: []string{"text"},
			},
			ConfigSchema: nil,
		}

		collCfg := &CollectionConfig{
			Name:      *testCollection,
			IdKey:     "id",
			ScoreKey:  "score",
			VectorKey: "vector",
			TextKey:   "text",
			VectorDim: dim,
			Embedder:  genkit.DefineEmbedder(g, "fake/embedder3", emdOpts, embedder.Embed),
		}

		retOpts := &ai.RetrieverOptions{
			ConfigSchema: core.InferSchemaMap(RetrieverOptions{}),
			Label:        "milvus",
			Supports: &ai.RetrieverSupports{
				Media: false,
			},
		}

		ds, retriever, err := DefineRetriever(ctx, g, collCfg, retOpts)
		if err != nil {
			t.Fatal(err)
		}

		if err := Index(ctx, []*ai.Document{d1, d2, d3}, ds); err != nil {
			t.Fatalf("index failed: %v", err)
		}
		retrieverOptions := &RetrieverOptions{
			Limit:   2,
			Columns: []string{"id"},
		}

		retrieverResp, err := genkit.Retrieve(ctx, g,
			ai.WithRetriever(retriever),
			ai.WithDocs(d1),
			ai.WithConfig(retrieverOptions),
		)
		if err != nil {
			t.Fatalf("retrieve failed: %v", err)
		}
		docs := retrieverResp.Documents
		if len(docs) != 2 {
			t.Errorf("expected 2 docs, got %d", len(docs))
		}

		for _, doc := range docs {
			text := doc.Content[0].Text
			if !strings.HasPrefix(text, "hello") {
				t.Errorf("returned doc text %q does not start with %q", text, "hello")
			}
			id, ok := doc.Metadata["id"].(int64)
			if !ok {
				t.Errorf("missing metadata entry for name: %v", doc.Metadata)
			}
			if id != id1 && id != id2 {
				t.Errorf("metadata id entry %d does not match expected id %d", id, id1)
			}
		}
	})

	t.Run("Filter search", func(t *testing.T) {
		if *testAddr == "" {
			t.Skip("skipping test because -test-milvus-addr flag not used")
		}
		if *testCollection == "" {
			t.Skip("skipping test because -test-milvus-collection flag not used")
		}
		if *testAPIKey == "" {
			t.Skip("skipping test because -test-milvus-api-key flag not used")
		}

		ctx := context.Background()

		dim := 1536

		v1 := make([]float32, dim)
		v2 := make([]float32, dim)
		v3 := make([]float32, dim)
		for i := range v1 {
			v1[i] = float32(i)
			v2[i] = float32(i)
			v3[i] = float32(dim - i)
		}
		v2[0] = 1

		id1 := int64(1000)
		id2 := int64(2000)
		id3 := int64(3000)

		d1 := ai.DocumentFromText("sunday1", map[string]any{"id": id1})
		d2 := ai.DocumentFromText("sunday2", map[string]any{"id": id2})
		d3 := ai.DocumentFromText("tuesday", map[string]any{"id": id3})

		embedder := fakeembedder.New()
		embedder.Register(d1, v1)
		embedder.Register(d2, v2)
		embedder.Register(d3, v3)

		engine, err := NewEngine(ctx, WithAddress(*testAddr), WithAPIKey(*testAPIKey))
		if err != nil {
			t.Fatalf("failed to create engine: %v", err)
		}
		defer func(engine *MilvusEngine, ctx context.Context) {
			err := engine.Close(ctx)
			if err != nil {
				t.Fatalf("failed to close engine: %v", err)
			}
		}(engine, ctx)

		_, err = engine.GetClient().
			Delete(ctx, milvusclient.NewDeleteOption("products").
				WithInt64IDs("id", []int64{id1, id2, id3}))
		if err != nil {
			t.Fatalf("failed to delete documents: %v", err)
		}

		m := &Milvus{
			Engine: engine,
		}

		g := genkit.Init(ctx, genkit.WithPlugins(m))

		emdOpts := &ai.EmbedderOptions{
			Dimensions: dim,
			Supports: &ai.EmbedderSupports{
				Input: []string{"text"},
			},
			ConfigSchema: nil,
		}

		collCfg := &CollectionConfig{
			Name:      *testCollection,
			IdKey:     "id",
			ScoreKey:  "score",
			VectorKey: "vector",
			TextKey:   "text",
			VectorDim: dim,
			Embedder:  genkit.DefineEmbedder(g, "fake/embedder3", emdOpts, embedder.Embed),
		}

		retOpts := &ai.RetrieverOptions{
			ConfigSchema: core.InferSchemaMap(RetrieverOptions{}),
			Label:        "milvus",
			Supports: &ai.RetrieverSupports{
				Media: false,
			},
		}

		ds, retriever, err := DefineRetriever(ctx, g, collCfg, retOpts)
		if err != nil {
			t.Fatal(err)
		}

		if err := Index(ctx, []*ai.Document{d1, d2, d3}, ds); err != nil {
			t.Fatalf("index failed: %v", err)
		}
		retrieverOptions := &RetrieverOptions{
			Limit:   2,
			Filter:  fmt.Sprintf("id == %d", id2),
			Columns: []string{"id"},
		}

		retrieverResp, err := genkit.Retrieve(ctx, g,
			ai.WithRetriever(retriever),
			ai.WithDocs(d1),
			ai.WithConfig(retrieverOptions),
		)
		if err != nil {
			t.Fatalf("retrieve failed: %v", err)
		}
		docs := retrieverResp.Documents
		if len(docs) != 1 {
			t.Errorf("expected 1 docs, got %d", len(docs))
		}

		doc := docs[0]
		text := doc.Content[0].Text
		if d2.Content[0].Text != "sunday2" {
			t.Errorf("returned doc text %q does not match expected text %q", text, d2.Content[0].Text)
		}
		id, ok := doc.Metadata["id"].(int64)
		if !ok {
			t.Errorf("missing metadata entry for name: %v", doc.Metadata)
		}
		if id != id2 {
			t.Errorf("metadata id entry %d does not match expected id %d", id, id1)
		}
	})
}
