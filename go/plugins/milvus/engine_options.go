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

import "google.golang.org/grpc"

// EngineOption configures the MilvusEngine during construction.
// Multiple options can be provided to NewMilvusEngine to customize
// the underlying Milvus client connection.
type EngineOption func(p *engineConfig)

// engineConfig holds configuration accumulated from provided Options.
// It is internal to the package and not exported.
type engineConfig struct {
	// address is the Milvus server address, e.g. "localhost:19530".
	address string
	// username used for Milvus authentication (optional).
	username string
	// password used for Milvus authentication (optional).
	password string
	// dbName selects a logical database within Milvus (optional).
	dbName string
	// enableTlsAuth enables TLS authentication when true.
	enableTlsAuth bool
	// apiKey carries API key for managed Milvus services (optional).
	apiKey string
	// dialOptions are gRPC dial options passed to the client.
	dialOptions []grpc.DialOption
	// disableConn prevents establishing the connection on creation (testing).
	disableConn bool
	// serverVersion hints the expected server version (optional).
	serverVersion string
}

// WithAddress sets the Milvus server address to connect to.
func WithAddress(address string) EngineOption {
	return func(p *engineConfig) {
		p.address = address
	}
}

// WithUsername sets the username for Milvus authentication.
func WithUsername(username string) EngineOption {
	return func(p *engineConfig) {
		p.username = username
	}
}

// WithPassword sets the password for Milvus authentication.
func WithPassword(password string) EngineOption {
	return func(p *engineConfig) {
		p.password = password
	}
}

// WithDbName sets the Milvus database name to use.
func WithDbName(dbName string) EngineOption {
	return func(p *engineConfig) {
		p.dbName = dbName
	}
}

// WithEnableTlsAuth toggles TLS authentication against the Milvus server.
func WithEnableTlsAuth(enableTlsAuth bool) EngineOption {
	return func(p *engineConfig) {
		p.enableTlsAuth = enableTlsAuth
	}
}

// WithAPIKey sets an API key for Milvus deployments that require it.
func WithAPIKey(apiKey string) EngineOption {
	return func(p *engineConfig) {
		p.apiKey = apiKey
	}
}

// WithDialOptions appends custom gRPC dial options for the client connection.
func WithDialOptions(dialOptions ...grpc.DialOption) EngineOption {
	return func(p *engineConfig) {
		p.dialOptions = dialOptions
	}
}

// WithDisableConn prevents establishing the network connection on engine creation.
// This can be useful in tests where a real Milvus server is not available.
func WithDisableConn(disableConn bool) EngineOption {
	return func(p *engineConfig) {
		p.disableConn = disableConn
	}
}

// WithServerVersion specifies the expected Milvus server version.
func WithServerVersion(serverVersion string) EngineOption {
	return func(p *engineConfig) {
		p.serverVersion = serverVersion
	}
}
