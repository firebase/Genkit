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
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// RetrieverOptions may be passed in the Options field
// [ai.RetrieverRequest] to pass options to Milvus.
// The option field should be either nil or
// a value of type *RetrieverOptions.
type RetrieverOptions struct {
	// Limit is the maximum number of results to retrieve.
	Limit int `json:"limit,omitempty"`
	// Columns list additional scalar fields to return with each result.
	// The text column is always included automatically.
	Columns []string `json:"columns,omitempty"`
	// Partitions restrict the search to the specified partition names.
	Partitions []string `json:"partitions,omitempty"`
	// Offset skips the first N results (useful for pagination).
	Offset int `json:"offset,omitempty"`
	// Filter is a boolean expression to post-filter rows in Milvus.
	Filter string `json:"filter,omitempty"`
	// FilterOptions passes engine-specific search parameters (e.g., nprobe, ef).
	FilterOptions map[string]string `json:"filterOptions,omitempty"`
}

// Retriever returns the retriever for the given class.
func Retriever(g *genkit.Genkit, collectionName string) ai.Retriever {
	return genkit.LookupRetriever(g, api.NewName(provider, collectionName))
}

// Retrieve implements ai.Retriever. It embeds the incoming query using the
// configured embedder, then performs an ANN search in Milvus against the
// configured collection. The result set is converted into ai.Document values
// with similarity score and optional extra columns attached as metadata.
func (ds *DocStore) Retrieve(ctx context.Context, req *ai.RetrieverRequest) (*ai.RetrieverResponse, error) {
	limit := 3
	outColumns := []string{ds.config.TextKey}
	var partitions []string
	var additionalColumns []string
	var offset int
	var filter string
	var filterOptions map[string]string
	if req.Options != nil {
		ropt, ok := req.Options.(*RetrieverOptions)
		if !ok {
			return nil, errors.New("milvus.Retrieve invalid options")
		}
		if ropt.Limit > 0 {
			limit = ropt.Limit
		}
		if len(ropt.Columns) > 0 {
			outColumns = append(outColumns, ropt.Columns...)
			additionalColumns = ropt.Columns
		}
		if len(ropt.Partitions) > 0 {
			partitions = ropt.Partitions
		}
		if ropt.Offset > 0 {
			offset = ropt.Offset
		}
		if ropt.Filter != "" {
			filter = ropt.Filter
		}
		if len(ropt.FilterOptions) > 0 {
			filterOptions = ropt.FilterOptions
		}
	}

	ereq := &ai.EmbedRequest{
		Input:   []*ai.Document{req.Query},
		Options: ds.config.EmbedderOptions,
	}
	eres, err := ds.config.Embedder.Embed(ctx, ereq)
	if err != nil {
		return nil, fmt.Errorf("milvus.Retrieve embedding failed: %v", err)
	}

	if len(eres.Embeddings) != 1 {
		return nil, fmt.Errorf("milvus.Retrieve embedding failed: expected 1 embedding, got %d", len(eres.Embeddings))
	}
	embedding := eres.Embeddings[0].Embedding
	sopts := milvusclient.NewSearchOption(
		ds.config.Name,
		limit,
		[]entity.Vector{entity.FloatVector(embedding)},
	).
		WithANNSField(ds.config.VectorKey).
		WithOutputFields(outColumns...)

	if len(partitions) > 0 {
		sopts.WithConsistencyLevel(entity.ClStrong).
			WithPartitions(partitions...)
	}
	if offset > 0 {
		sopts.WithOffset(offset)
	}
	if filter != "" {
		sopts.WithFilter(filter)
	}
	if len(filterOptions) > 0 {
		for k, v := range filterOptions {
			sopts.WithSearchParam(k, v)
		}
	}

	res, err := ds.engine.client.Search(ctx, sopts)
	if err != nil {
		return nil, fmt.Errorf("milvus.Retrieve failed: %v", err)
	}
	var docs []*ai.Document
	for _, item := range res {
		for j, score := range item.Scores {
			text, err := item.GetColumn(ds.config.TextKey).GetAsString(j)
			if err != nil {
				return nil, fmt.Errorf("milvus.Retrieve failed: %v", err)
			}
			metadata := map[string]any{
				ds.config.ScoreKey: score,
			}

			for _, column := range additionalColumns {
				value, err := item.GetColumn(column).Get(j)
				if err != nil {
					return nil, fmt.Errorf("milvus.Retrieve failed: %v", err)
				}
				metadata[column] = value
			}

			docs = append(docs, ai.DocumentFromText(text, metadata))
		}
	}

	return &ai.RetrieverResponse{
		Documents: docs,
	}, nil
}
