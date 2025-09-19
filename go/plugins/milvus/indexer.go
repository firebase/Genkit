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
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// Index helper function to get started with indexing
func Index(ctx context.Context, docs []*ai.Document, ds *DocStore) error {
	if len(docs) == 0 {
		return nil
	}

	ereq := &ai.EmbedRequest{
		Input:   docs,
		Options: ds.config.EmbedderOptions,
	}
	eres, err := ds.config.Embedder.Embed(ctx, ereq)
	if err != nil {
		return fmt.Errorf("milvus index embedding failed: %v", err)
	}
	if len(eres.Embeddings) != len(docs) {
		return fmt.Errorf(
			"milvus index embedding failed: expected %d embeddings, got %d",
			len(docs), len(eres.Embeddings),
		)
	}

	ids := make([]int64, 0)
	vectors := make([][]float32, 0)
	texts := make([]string, 0)

	for i, de := range eres.Embeddings {
		doc := docs[i]

		metadata := doc.Metadata
		if metadata == nil {
			return fmt.Errorf("milvus index metadata is nil")
		}

		id, ok := metadata[ds.config.IdKey].(int64)
		if !ok {
			return fmt.Errorf("milvus index id key %s is not int64", ds.config.IdKey)
		}
		ids = append(ids, id)

		var text strings.Builder
		for _, p := range doc.Content {
			text.WriteString(p.Text)
		}
		texts = append(texts, text.String())

		vectors = append(vectors, de.Embedding)
	}

	opts := milvusclient.NewColumnBasedInsertOption(ds.config.Name).
		WithInt64Column(ds.config.IdKey, ids).
		WithVarcharColumn(ds.config.TextKey, texts).
		WithFloatVectorColumn(ds.config.VectorKey, ds.config.VectorDim, vectors)

	_, err = ds.engine.client.Insert(ctx, opts)
	if err != nil {
		return fmt.Errorf("milvus index insert failed: %v", err)
	}

	return nil
}
