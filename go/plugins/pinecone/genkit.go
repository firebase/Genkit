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

package pinecone

import (
	"context"
	"crypto/md5"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
)

const provider = "pinecone"

// defaultTextKey is the default metadata key we use to store the
// document text as metadata of the documented stored in pinecone.
// This lets us map back from the document returned by a query to
// the text. In other words, rather than remembering the mapping
// between documents and pinecone data ourselves, we just store the
// documents in pinecone.
const defaultTextKey = "_content"

// Config provides configuration options for the Init function.
type Config struct {
	// API key for Pinecone.
	// If it is the empty string, it is read from the PINECONE_API_KEY
	// environment variable.
	APIKey string
	// The controller host name. This implies which index to use.
	// If it is the empty string, it is read from the PINECONE_CONTROLLER_HOST
	// environment variable.
	Host string
	// Embedder to use. Required.
	Embedder        *ai.EmbedderAction
	EmbedderOptions any
	// The metadata key to use to store document text
	// in Pinecone; the default is "_content".
	TextKey string
}

// Init initializes the Pinecone plugin.
func Init(ctx context.Context, cfg Config) (err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("pinecone.Init: %w", err)
		}
	}()

	client, err := NewClient(ctx, cfg.APIKey)
	if err != nil {
		return err
	}
	index, err := client.Index(ctx, cfg.Host)
	if err != nil {
		return err
	}
	if cfg.TextKey == "" {
		cfg.TextKey = defaultTextKey
	}
	r := &docStore{
		index:           index,
		embedder:        cfg.Embedder,
		embedderOptions: cfg.EmbedderOptions,
		textKey:         cfg.TextKey,
	}
	name := index.host // the resolved host
	ai.DefineIndexer(provider, name, r.Index)
	ai.DefineRetriever(provider, name, r.Retrieve)
	return nil
}

// Indexer returns the indexer with the given index name.
func Indexer(name string) *ai.IndexerAction {
	return ai.LookupIndexer(provider, name)
}

// Retriever returns the retriever with the given index name.
func Retriever(name string) *ai.RetrieverAction {
	return ai.LookupRetriever(provider, name)
}

// IndexerOptions may be passed in the Options field
// of [ai.IndexerRequest] to pass options to Pinecone.
// The Options field should be either nil or a value of type *IndexerOptions.
type IndexerOptions struct {
	Namespace string `json:"namespace,omitempty"` // Pinecone namespace to use
}

// RetrieverOptions may be passed in the Options field
// of [ai.RetrieverRequest] to pass options to Pinecone.
// The Options field should be either nil or a value of type *RetrieverOptions.
type RetrieverOptions struct {
	Namespace string `json:"namespace,omitempty"` // Pinecone namespace to use
	Count     int    `json:"count,omitempty"`     // maximum number of values to retrieve
}

// docStore implements the genkit [ai.DocumentStore] interface.
type docStore struct {
	index           *Index
	embedder        *ai.EmbedderAction
	embedderOptions any
	textKey         string
}

// Index implements the genkit Retriever.Index method.
func (ds *docStore) Index(ctx context.Context, req *ai.IndexerRequest) error {
	if len(req.Documents) == 0 {
		return nil
	}

	namespace := ""
	if req.Options != nil {
		// TODO(iant): This is plausible when called directly
		// from Go code, but what will it look like when
		// run from a resumed flow?
		options, ok := req.Options.(*IndexerOptions)
		if !ok {
			return fmt.Errorf("pinecone.Index options have type %T, want %T", req.Options, &IndexerOptions{})
		}
		namespace = options.Namespace
	}

	// Use the embedder to convert each Document into a vector.
	vecs := make([]Vector, 0, len(req.Documents))
	for _, doc := range req.Documents {
		ereq := &ai.EmbedRequest{
			Document: doc,
			Options:  ds.embedderOptions,
		}
		vals, err := ai.Embed(ctx, ds.embedder, ereq)
		if err != nil {
			return fmt.Errorf("pinecone index embedding failed: %v", err)
		}

		id, err := docID(doc)
		if err != nil {
			return err
		}

		var metadata map[string]any
		if doc.Metadata == nil {
			metadata = make(map[string]any)
		} else {
			metadata = maps.Clone(doc.Metadata)
		}
		// TODO(iant): This seems to be what the TypeScript code does,
		// but it loses the structure of the document.
		var sb strings.Builder
		for _, p := range doc.Content {
			sb.WriteString(p.Text)
		}
		metadata[ds.textKey] = sb.String()

		v := Vector{
			ID:       id,
			Values:   vals,
			Metadata: metadata,
		}
		vecs = append(vecs, v)
	}

	if err := ds.index.Upsert(ctx, vecs, namespace); err != nil {
		return err
	}

	// Pinecone is only eventually consistent.
	// Wait until the vectors are visible.
	wait := func() (bool, error) {
		delay := 10 * time.Millisecond
		for i := 0; i < 20; i++ {
			vec, err := ds.index.QueryByID(ctx, vecs[0].ID, WantValues, namespace)
			if err != nil {
				return false, err
			}
			if vec != nil {
				// For some reason Pinecone doesn't
				// reliably return a vector with the
				// same ID.
				for _, v := range vecs {
					if vec.ID == v.ID && slices.Equal(vec.Values, v.Values) {
						return true, nil
					}
				}
			}
			time.Sleep(delay)
			delay *= 2
		}
		return false, nil
	}
	seen, err := wait()
	if err != nil {
		return err
	}
	if !seen {
		return errors.New("inserted Pinecone records never became visible")
	}

	return nil
}

// Retrieve implements the genkit Retriever.Retrieve method.
func (ds *docStore) Retrieve(ctx context.Context, req *ai.RetrieverRequest) (*ai.RetrieverResponse, error) {
	var (
		namespace string
		count     int
	)
	if req.Options != nil {
		// TODO(iant): This is plausible when called directly
		// from Go code, but what will it look like when
		// run from a resumed flow?
		ropt, ok := req.Options.(*RetrieverOptions)
		if !ok {
			return nil, fmt.Errorf("pinecone.Retrieve options have type %T, want %T", req.Options, &RetrieverOptions{})
		}
		namespace = ropt.Namespace
		count = ropt.Count
	}

	// Use the embedder to convert the document we want to
	// retrieve into a vector.
	ereq := &ai.EmbedRequest{
		Document: req.Document,
		Options:  ds.embedderOptions,
	}
	vals, err := ai.Embed(ctx, ds.embedder, ereq)
	if err != nil {
		return nil, fmt.Errorf("pinecone retrieve embedding failed: %v", err)
	}

	results, err := ds.index.Query(ctx, vals, count, WantMetadata, namespace)
	if err != nil {
		return nil, err
	}

	var docs []*ai.Document
	for _, result := range results {
		text, _ := result.Metadata[ds.textKey].(string)
		if text == "" {
			return nil, errors.New("Pinecone retrieve failed to fetch original document text")
		}
		delete(result.Metadata, ds.textKey)
		// TODO(iant): This is what the TypeScript code does,
		// but it loses information for multimedia documents.
		d := ai.DocumentFromText(text, result.Metadata)
		docs = append(docs, d)
	}

	ret := &ai.RetrieverResponse{
		Documents: docs,
	}
	return ret, nil
}

// docID returns the ID to use for a Document.
// This is intended to be the same as the genkit Typescript computation.
func docID(doc *ai.Document) (string, error) {
	b, err := json.Marshal(doc)
	if err != nil {
		return "", fmt.Errorf("pinecone: error marshaling document: %v", err)
	}
	return fmt.Sprintf("%02x", md5.Sum(b)), nil
}
