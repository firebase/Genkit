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

import "errors"

// DocStore is a lightweight handle bound to a specific Milvus collection
// and its embedding configuration. It implements ai.Retriever by providing
// the Retrieve method.
type DocStore struct {
	// engine is the Milvus engine used to communicate with the database.
	engine *MilvusEngine
	// config holds collection-specific configuration and embedding settings.
	config *CollectionConfig
}

// newDocStore validates the collection configuration and creates a DocStore
// instance bound to the Milvus engine. The validation ensures that required
// keys and embedding parameters are present before any operations are performed.
func (m *Milvus) newDocStore(config *CollectionConfig) (*DocStore, error) {
	if config.Name == "" {
		return nil, errors.New("milvus.Init collection name must be set")
	}
	if config.IdKey == "" {
		return nil, errors.New("milvus.Init id key must be set")
	}
	if config.VectorKey == "" {
		return nil, errors.New("milvus.Init vector key must be set")
	}
	if config.TextKey == "" {
		return nil, errors.New("milvus.Init text key must be set")
	}
	if config.VectorDim <= 0 {
		return nil, errors.New("milvus.Init vector dimension must be set")
	}
	if config.Embedder == nil {
		return nil, errors.New("milvus.Init embedder must be set")
	}

	return &DocStore{
		engine: m.Engine,
		config: config,
	}, nil
}
