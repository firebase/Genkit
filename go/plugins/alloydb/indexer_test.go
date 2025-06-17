// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package alloydb

import (
	"context"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/stretchr/testify/require"
)

func TestIndex_Success_NoDocuments(t *testing.T) {
	ds := DocStore{}
	err := ds.Index(context.Background(), nil)
	require.NoError(t, err)

}

func TestIndex_Fail_EmbedReturnError(t *testing.T) {
	ds := DocStore{
		config: &Config{Embedder: mockEmbedderFail{}},
	}

	docs := []*ai.Document{{
		Content: []*ai.Part{{
			Kind:        ai.PartText,
			ContentType: "text/plain",
			Text:        "This is a test",
		}},
		Metadata: nil,
	}}

	err := ds.Index(context.Background(), docs)
	require.Error(t, err)
}
