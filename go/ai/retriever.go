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

package ai

import (
	"context"
	"errors"
	"fmt"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/internal/registry"
)

// RetrieverFunc is the function type for retriever implementations.
type RetrieverFunc = func(context.Context, *RetrieverRequest) (*RetrieverResponse, error)

// Retriever represents a document retriever.
type Retriever interface {
	// Name returns the name of the retriever.
	Name() string
	// Retrieve retrieves the documents.
	Retrieve(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error)
}

// retriever is an action with functions specific to document retrieval such as Retrieve().
type retriever core.ActionDef[*RetrieverRequest, *RetrieverResponse, struct{}]

// RetrieverArg is the interface for retriever arguments. It can either be the retriever action itself or a reference to be looked up.
type RetrieverArg interface {
	Name() string
}

// RetrieverRef is a struct to hold retriever name and configuration.
type RetrieverRef struct {
	name   string
	config any
}

// NewRetrieverRef creates a new RetrieverRef with the given name and configuration.
func NewRetrieverRef(name string, config any) RetrieverRef {
	return RetrieverRef{name: name, config: config}
}

// Name returns the name of the retriever.
func (r RetrieverRef) Name() string {
	return r.name
}

// Config returns the configuration to use by default for this retriever.
func (r RetrieverRef) Config() any {
	return r.config
}

// RetrieverSupports defines the supported capabilities of the retriever.
type RetrieverSupports struct {
	// Media indicates whether the retriever supports media content.
	Media bool `json:"media,omitempty"`
}

// RetrieverOptions represents the configuration options for a retriever.
type RetrieverOptions struct {
	// ConfigSchema is the JSON schema for the retriever's config.
	ConfigSchema map[string]any `json:"configSchema,omitempty"`
	// Label is a user-friendly name for the retriever.
	Label string `json:"label,omitempty"`
	// Supports defines the capabilities of the retriever, such as media support.
	Supports *RetrieverSupports `json:"supports,omitempty"`
}

// DefineRetriever registers the given retrieve function as an action, and returns a
// [Retriever] that runs it.
func DefineRetriever(r *registry.Registry, name string, opts *RetrieverOptions, fn RetrieverFunc) Retriever {
	if name == "" {
		panic("ai.Retrieve: retriever name is required")
	}

	if opts == nil {
		opts = &RetrieverOptions{
			Label: name,
		}
	}
	if opts.Supports == nil {
		opts.Supports = &RetrieverSupports{}
	}

	metadata := map[string]any{
		"type": core.ActionTypeRetriever,
		"info": map[string]any{
			"label": opts.Label,
			"supports": map[string]any{
				"media": opts.Supports.Media,
			},
		},
		"retriever": map[string]any{
			"customOptions": opts.ConfigSchema,
		},
	}

	inputSchema := core.InferSchemaMap(RetrieverRequest{})
	if inputSchema != nil && opts.ConfigSchema != nil {
		if _, ok := inputSchema["options"]; ok {
			inputSchema["options"] = opts.ConfigSchema
		}
	}

	return (*retriever)(core.DefineAction(r, name, core.ActionTypeRetriever, metadata, fn))
}

// LookupRetriever looks up a [Retriever] registered by [DefineRetriever].
// It returns nil if the retriever was not defined.
func LookupRetriever(r *registry.Registry, name string) Retriever {
	return (*retriever)(core.LookupActionFor[*RetrieverRequest, *RetrieverResponse, struct{}](r, core.ActionTypeRetriever, name))
}

// Name returns the name of the retriever.
func (r retriever) Name() string {
	return (*core.ActionDef[*RetrieverRequest, *RetrieverResponse, struct{}])(&r).Name()
}

// Retrieve runs the given [Retriever].
func (r retriever) Retrieve(ctx context.Context, req *RetrieverRequest) (*RetrieverResponse, error) {
	return (*core.ActionDef[*RetrieverRequest, *RetrieverResponse, struct{}])(&r).Run(ctx, req, nil)
}

// Retrieve calls the retriever with the provided options.
func Retrieve(ctx context.Context, r *registry.Registry, opts ...RetrieverOption) (*RetrieverResponse, error) {
	retOpts := &retrieverOptions{}
	for _, opt := range opts {
		if err := opt.applyRetriever(retOpts); err != nil {
			return nil, fmt.Errorf("ai.Retrieve: error applying options: %w", err)
		}
	}

	if len(retOpts.Documents) > 1 {
		return nil, errors.New("ai.Retrieve: only supports a single document as input")
	}

	ret, ok := retOpts.Retriever.(Retriever)
	if !ok {
		ret = LookupRetriever(r, retOpts.Retriever.Name())
	}

	if ret == nil {
		return nil, fmt.Errorf("ai.Retrieve: retriever not found: %s", retOpts.Retriever.Name())
	}

	if retRef, ok := retOpts.Retriever.(RetrieverRef); ok && retOpts.Config == nil {
		retOpts.Config = retRef.Config()
	}

	req := &RetrieverRequest{
		Query:   retOpts.Documents[0],
		Options: retOpts.Config,
	}

	return ret.Retrieve(ctx, req)
}
