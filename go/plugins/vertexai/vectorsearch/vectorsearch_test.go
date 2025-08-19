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

package vectorsearch

import (
	"context"
	"errors"
	"net/http"
	"os"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/stretchr/testify/assert"
)

type fakeEmbedder struct {
	resp *ai.EmbedResponse
	err  error
}

func (f *fakeEmbedder) Name() string { return "fake-embedder" }

func (f *fakeEmbedder) Embed(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	if f.err != nil {
		return nil, f.err
	}
	return f.resp, nil
}

// ---------- tests: Name ----------

func TestVectorsearch_Name(t *testing.T) {
	v := &VertexAIVectorSearch{}
	assert.Equal(t, vectorsearchProvider, v.Name())
}

// ---------- tests: Init ----------

func TestInit_AlreadyInitialized(t *testing.T) {
	v := &VertexAIVectorSearch{initted: true}
	err := v.Init(context.Background(), nil)
	assert.Error(t, err)
}

func TestInit_MissingProjectID_ReturnsError(t *testing.T) {
	unset := func(k string) func() {
		old, had := os.LookupEnv(k)
		_ = os.Unsetenv(k)
		return func() {
			if had {
				_ = os.Setenv(k, old)
			} else {
				_ = os.Unsetenv(k)
			}
		}
	}
	restore := []func(){}
	for _, e := range []string{"GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_REGION"} {
		restore = append(restore, unset(e))
	}
	defer func() {
		for _, r := range restore {
			r()
		}
	}()

	v := &VertexAIVectorSearch{}
	err := v.Init(context.Background(), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "requires setting GOOGLE_CLOUD_PROJECT")
}

func TestInit_MissingLocation_ReturnsError(t *testing.T) {
	restoreProj := setEnv(t, "GOOGLE_CLOUD_PROJECT", "proj")
	defer restoreProj()
	restoreLoc := clearEnv(t, "GOOGLE_CLOUD_LOCATION")
	defer restoreLoc()
	restoreReg := clearEnv(t, "GOOGLE_CLOUD_REGION")
	defer restoreReg()

	v := &VertexAIVectorSearch{}
	err := v.Init(context.Background(), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "requires setting GOOGLE_CLOUD_LOCATION")
}

func setEnv(t *testing.T, k, v string) func() {
	t.Helper()
	old, had := os.LookupEnv(k)
	assert.NoError(t, os.Setenv(k, v))
	return func() {
		if had {
			_ = os.Setenv(k, old)
		} else {
			_ = os.Unsetenv(k)
		}
	}
}

func clearEnv(t *testing.T, k string) func() {
	t.Helper()
	old, had := os.LookupEnv(k)
	_ = os.Unsetenv(k)
	return func() {
		if had {
			_ = os.Setenv(k, old)
		} else {
			_ = os.Unsetenv(k)
		}
	}
}

// ---------- tests: Retrieve ----------

func TestRetrieve_Success(t *testing.T) {
	embedder := &fakeEmbedder{
		resp: &ai.EmbedResponse{
			Embeddings: []*ai.Embedding{{Embedding: []float32{0.1}}},
		},
	}
	v := newVSWithToken("tok123", nil)
	findNeighborsJSON := `{"nearestNeighbors":[{"neighbors":[{"datapoint":{"datapointId":"doc-1"},"distance":0.4}]}]}`
	withTransport(t, rtFunc(func(r *http.Request) (*http.Response, error) {
		assert.Equal(t, "Bearer tok123", r.Header.Get("Authorization"))
		return newHTTPResponse(http.StatusOK, findNeighborsJSON), nil
	}), func() {
		docRetriever := func(ctx context.Context, neighbors []Neighbor, options any) ([]*ai.Document, error) {
			assert.Len(t, neighbors, 1)
			assert.Equal(t, "doc-1", neighbors[0].Datapoint.DatapointId)
			return []*ai.Document{{Content: []*ai.Part{{Text: "Hello"}}}}, nil
		}

		req := &ai.RetrieverRequest{
			Query: &ai.Document{Content: []*ai.Part{{Text: "q"}}},
			Options: &RetrieveParams{
				Embedder:          embedder,
				NeighborCount:     1,
				ProjectNumber:     "123",
				IndexEndpointID:   "ep",
				PublicDomainName:  "public.example",
				DeployedIndexID:   "dep",
				DocumentRetriever: docRetriever,
			},
		}

		resp, err := v.Retrieve(context.Background(), req)
		assert.NoError(t, err)
		assert.NotNil(t, resp)
		assert.Equal(t, "Hello", resp.Documents[0].Content[0].Text)
	})
}

func TestRetrieve_EmbedderError(t *testing.T) {
	v := newVSWithToken("tok", nil)
	req := &ai.RetrieverRequest{
		Query: &ai.Document{Content: []*ai.Part{{Text: "q"}}},
		Options: &RetrieveParams{
			Embedder: &fakeEmbedder{err: errors.New("fail")},
			DocumentRetriever: func(context.Context, []Neighbor, any) ([]*ai.Document, error) {
				return nil, nil
			},
		},
	}
	_, err := v.Retrieve(context.Background(), req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "error generating embedding for query")
}

func TestRetrieve_NoEmbeddings(t *testing.T) {
	v := newVSWithToken("tok", nil)
	req := &ai.RetrieverRequest{
		Query: &ai.Document{Content: []*ai.Part{{Text: "q"}}},
		Options: &RetrieveParams{
			Embedder: &fakeEmbedder{resp: &ai.EmbedResponse{Embeddings: []*ai.Embedding{}}},
			DocumentRetriever: func(context.Context, []Neighbor, any) ([]*ai.Document, error) {
				return nil, nil
			},
		},
	}
	_, err := v.Retrieve(context.Background(), req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no embeddings generated for query")
}

func TestRetrieve_NoNeighborsReturnsNil(t *testing.T) {
	embedder := &fakeEmbedder{
		resp: &ai.EmbedResponse{Embeddings: []*ai.Embedding{{Embedding: []float32{1}}}},
	}
	v := newVSWithToken("tok", nil)
	findNeighborsJSON := `{"nearestNeighbors":[{"neighbors":[]}]}`
	withTransport(t, rtFunc(func(r *http.Request) (*http.Response, error) {
		return newHTTPResponse(http.StatusOK, findNeighborsJSON), nil
	}), func() {
		req := &ai.RetrieverRequest{
			Query: &ai.Document{Content: []*ai.Part{{Text: "q"}}},
			Options: &RetrieveParams{
				Embedder:          embedder,
				DocumentRetriever: func(context.Context, []Neighbor, any) ([]*ai.Document, error) { return nil, nil },
			},
		}
		resp, err := v.Retrieve(context.Background(), req)
		assert.NoError(t, err)
		assert.Nil(t, resp)
	})
}
