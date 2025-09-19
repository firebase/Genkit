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

	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// MilvusEngine owns the lifecycle of a Milvus client.
// It is a thin wrapper that constructs and exposes the client used by
// the rest of the plugin.
type MilvusEngine struct {
	// client is the underlying Milvus SDK client used for all operations.
	client *milvusclient.Client
}

// NewMilvusEngine creates a new MilvusEngine and underlying client using the
// provided options. It validates required configuration and returns an error
// if the client cannot be created.
func NewEngine(ctx context.Context, opts ...EngineOption) (*MilvusEngine, error) {
	engine := new(MilvusEngine)
	cfg, err := applyOptions(opts)
	if err != nil {
		return nil, fmt.Errorf("failed applying options: %v", err)
	}
	client, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address:       cfg.address,
		Username:      cfg.username,
		Password:      cfg.password,
		DBName:        cfg.dbName,
		EnableTLSAuth: cfg.enableTlsAuth,
		APIKey:        cfg.apiKey,
		DialOptions:   cfg.dialOptions,
		DisableConn:   cfg.disableConn,
		ServerVersion: cfg.serverVersion,
	})
	if err != nil {
		return nil, fmt.Errorf("failed creating milvus client: %v", err)
	}
	engine.client = client
	return engine, nil
}

// GetClient returns the underlying Milvus client instance.
func (e *MilvusEngine) GetClient() *milvusclient.Client {
	return e.client
}

// Close closes the underlying Milvus client and releases resources.
func (e *MilvusEngine) Close(ctx context.Context) error {
	return e.client.Close(ctx)
}

// applyOptions applies the provided EngineOption functions into a concrete
// configuration object and performs basic validation.
func applyOptions(opts []EngineOption) (engineConfig, error) {
	cfg := &engineConfig{}
	for _, opt := range opts {
		opt(cfg)
	}
	if cfg.address == "" {
		return engineConfig{}, errors.New("address must be set")
	}
	return *cfg, nil
}
