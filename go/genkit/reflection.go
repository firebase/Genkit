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

package genkit

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal"
	"go.opentelemetry.io/otel/trace"
)

type streamingCallback[Stream any] = func(context.Context, Stream) error

// runtimeFileData is the data written to the file describing this runtime.
type runtimeFileData struct {
	ID                       string `json:"id"`
	PID                      int    `json:"pid"`
	ReflectionServerURL      string `json:"reflectionServerUrl"`
	Timestamp                string `json:"timestamp"`
	GenkitVersion            string `json:"genkitVersion"`
	ReflectionApiSpecVersion int    `json:"reflectionApiSpecVersion"`
}

// reflectionServer encapsulates everything needed to serve the Reflection API.
type reflectionServer struct {
	*http.Server
	RuntimeFilePath string // Path to the runtime file that was written at startup.
}

// findAvailablePort finds an available TCP port starting from a given port.
// It checks up to 100 ports.
func findAvailablePort(startPort int) (int, error) {
	for port := startPort; port < startPort+100; port++ {
		addr := fmt.Sprintf("127.0.0.1:%d", port)
		if l, err := net.Listen("tcp", addr); err == nil {
			l.Close()
			return port, nil
		}
	}
	return -1, fmt.Errorf("no available port in range %d-%d", startPort, startPort+99)
}

// startReflectionServer starts the Reflection API server. If the
// GENKIT_REFLECTION_PORT environment variable is set, it will listen on that
// port. Otherwise, it searches for an available port starting at 3100.
func startReflectionServer(ctx context.Context, g *Genkit, errCh chan<- error, serverStartCh chan<- struct{}) *reflectionServer {
	if g == nil {
		errCh <- fmt.Errorf("nil Genkit provided")
		return nil
	}

	var port int
	var err error
	if portStr := os.Getenv("GENKIT_REFLECTION_PORT"); portStr != "" {
		p, err := strconv.Atoi(portStr)
		if err != nil {
			errCh <- fmt.Errorf("invalid GENKIT_REFLECTION_PORT: %w", err)
			return nil
		}
		port = p
	} else {
		startPort := 3100
		port, err = findAvailablePort(startPort)
		if err != nil {
			errCh <- fmt.Errorf("failed to find available port: %w", err)
			return nil
		}
		if port != startPort {
			slog.Warn(fmt.Sprintf("Port %d is already in use, using next available port %d instead.", startPort, port))
		}
	}

	addr := fmt.Sprintf("127.0.0.1:%d", port)

	s := &reflectionServer{
		Server: &http.Server{
			Addr:    addr,
			Handler: serveMux(g),
		},
	}

	slog.Debug("starting reflection server", "addr", s.Addr)

	if err := s.writeRuntimeFile(s.Addr, port); err != nil {
		errCh <- fmt.Errorf("failed to write runtime file: %w", err)
		return nil
	}

	serverCtx, cancel := context.WithCancel(context.Background())

	go func() {
		// First check that the port is available before signaling a server start success.
		listener, err := net.Listen("tcp", s.Addr)
		if err != nil {
			errCh <- fmt.Errorf("failed to create listener: %w", err)
			return
		}

		close(serverStartCh)

		if err := s.Serve(listener); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
		// If the server shuts down unexpectedly, this will trigger the cleanup.
		cancel()
	}()

	go func() {
		// Blocks here until the context is done or the server crashes.
		select {
		case <-ctx.Done():
		case <-serverCtx.Done():
			return
		}

		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		if err := s.Shutdown(shutdownCtx); err != nil {
			slog.Error("reflection server shutdown error", "error", err)
		}

		if err := s.cleanupRuntimeFile(); err != nil {
			slog.Error("failed to cleanup runtime file", "error", err)
		}
	}()

	return s
}

// writeRuntimeFile writes a file describing the runtime to the project root.
func (s *reflectionServer) writeRuntimeFile(url string, port int) error {
	projectRoot, err := findProjectRoot()
	if err != nil {
		return fmt.Errorf("failed to find project root: %w", err)
	}

	runtimesDir := filepath.Join(projectRoot, ".genkit", "runtimes")
	if err := os.MkdirAll(runtimesDir, 0755); err != nil {
		return fmt.Errorf("failed to create runtimes directory: %w", err)
	}

	runtimeID := os.Getenv("GENKIT_RUNTIME_ID")
	if runtimeID == "" {
		runtimeID = strconv.Itoa(os.Getpid())
	}

	timestamp := time.Now().UTC().Format(time.RFC3339)
	// remove colons to avoid problems with different OS file name restrictions
	timestamp = strings.ReplaceAll(timestamp, ":", "_")

	s.RuntimeFilePath = filepath.Join(runtimesDir, fmt.Sprintf("%d-%d-%s.json", os.Getpid(), port, timestamp))

	data := runtimeFileData{
		ID:                       runtimeID,
		PID:                      os.Getpid(),
		ReflectionServerURL:      fmt.Sprintf("http://%s", url),
		Timestamp:                timestamp,
		GenkitVersion:            "go/" + internal.Version,
		ReflectionApiSpecVersion: internal.GENKIT_REFLECTION_API_SPEC_VERSION,
	}

	fileContent, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal runtime data: %w", err)
	}

	if err := os.WriteFile(s.RuntimeFilePath, fileContent, 0644); err != nil {
		return fmt.Errorf("failed to write runtime file: %w", err)
	}

	slog.Debug("runtime file written", "path", s.RuntimeFilePath)
	return nil
}

// cleanupRuntimeFile removes the runtime file associated with the dev server.
func (s *reflectionServer) cleanupRuntimeFile() error {
	if s.RuntimeFilePath == "" {
		return nil
	}

	content, err := os.ReadFile(s.RuntimeFilePath)
	if err != nil {
		return fmt.Errorf("failed to read runtime file: %w", err)
	}

	var data runtimeFileData
	if err := json.Unmarshal(content, &data); err != nil {
		return fmt.Errorf("failed to unmarshal runtime data: %w", err)
	}

	if data.PID == os.Getpid() {
		if err := os.Remove(s.RuntimeFilePath); err != nil {
			return fmt.Errorf("failed to remove runtime file: %w", err)
		}
		slog.Debug("runtime file cleaned up", "path", s.RuntimeFilePath)
	}

	return nil
}

// findProjectRoot finds the project root by looking for a go.mod file.
func findProjectRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}

	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			slog.Warn("could not find project root (go.mod not found)")
			return os.Getwd()
		}
		dir = parent
	}
}

// serveMux returns a new ServeMux configured for the required Reflection API endpoints.
func serveMux(g *Genkit) *http.ServeMux {
	mux := http.NewServeMux()
	// Skip wrapHandler here to avoid logging constant polling requests.
	mux.HandleFunc("GET /api/__health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	mux.HandleFunc("GET /api/actions", wrapReflectionHandler(handleListActions(g)))
	mux.HandleFunc("POST /api/runAction", wrapReflectionHandler(handleRunAction(g)))
	mux.HandleFunc("POST /api/notify", wrapReflectionHandler(handleNotify()))
	return mux
}

// wrapReflectionHandler wraps an HTTP handler function with common logging and error handling.
func wrapReflectionHandler(h func(w http.ResponseWriter, r *http.Request) error) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		logger.FromContext(ctx).Debug("request start", "method", r.Method, "path", r.URL.Path)

		var err error
		defer func() {
			if err != nil {
				logger.FromContext(ctx).Error("request end", "err", err)
			} else {
				logger.FromContext(ctx).Debug("request end")
			}
		}()

		w.Header().Set("x-genkit-version", "go/"+internal.Version)

		if err = h(w, r); err != nil {
			errorResponse := core.ToReflectionError(err)
			w.WriteHeader(errorResponse.Code)
			writeJSON(ctx, w, errorResponse)
		}
	}
}

// handleRunAction looks up an action by name in the registry, runs it with the
// provided JSON input, and writes back the JSON-marshaled request.
func handleRunAction(g *Genkit) func(w http.ResponseWriter, r *http.Request) error {
	return func(w http.ResponseWriter, r *http.Request) error {
		ctx := r.Context()

		var body struct {
			Key             string          `json:"key"`
			Input           json.RawMessage `json:"input"`
			Context         json.RawMessage `json:"context"`
			TelemetryLabels json.RawMessage `json:"telemetryLabels"`
		}
		defer r.Body.Close()
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			return core.NewError(core.INVALID_ARGUMENT, err.Error())
		}

		stream, err := parseBoolQueryParam(r, "stream")
		if err != nil {
			return err
		}

		logger.FromContext(ctx).Debug("running action", "key", body.Key, "stream", stream)

		var cb streamingCallback[json.RawMessage]
		if stream {
			w.Header().Set("Content-Type", "text/plain")
			w.Header().Set("Transfer-Encoding", "chunked")
			// Stream results are newline-separated JSON.
			cb = func(ctx context.Context, msg json.RawMessage) error {
				_, err := fmt.Fprintf(w, "%s\n", msg)
				if err != nil {
					return err
				}
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
				return nil
			}
		}

		var contextMap core.ActionContext = nil
		if body.Context != nil {
			json.Unmarshal(body.Context, &contextMap)
		}

		resp, err := runAction(ctx, g, body.Key, body.Input, body.TelemetryLabels, cb, contextMap)
		if err != nil {
			if stream {
				reflectErr, err := json.Marshal(core.ToReflectionError(err))
				if err != nil {
					return err
				}

				_, err = fmt.Fprintf(w, "%s\n\n", reflectErr)
				if err != nil {
					return err
				}

				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
				return nil
			}
			return err
		}

		return writeJSON(ctx, w, resp)
	}
}

// handleNotify configures the telemetry server URL from the request.
func handleNotify() func(w http.ResponseWriter, r *http.Request) error {
	return func(w http.ResponseWriter, r *http.Request) error {
		var body struct {
			TelemetryServerURL       string `json:"telemetryServerUrl"`
			ReflectionApiSpecVersion int    `json:"reflectionApiSpecVersion"`
		}

		defer r.Body.Close()
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			return core.NewError(core.INVALID_ARGUMENT, err.Error())
		}

		if os.Getenv("GENKIT_TELEMETRY_SERVER") == "" && body.TelemetryServerURL != "" {
			tracing.WriteTelemetryImmediate(tracing.NewHTTPTelemetryClient(body.TelemetryServerURL))
			slog.Debug("connected to telemetry server", "url", body.TelemetryServerURL)
		}

		if body.ReflectionApiSpecVersion != internal.GENKIT_REFLECTION_API_SPEC_VERSION {
			slog.Error("Genkit CLI version is not compatible with runtime library. Please use `genkit-cli` version compatible with runtime library version.")
		}

		w.WriteHeader(http.StatusOK)
		_, err := w.Write([]byte("OK"))
		return err
	}
}

// handleListActions lists all the registered actions.
// The list is sorted by action name and contains unique action names.
func handleListActions(g *Genkit) func(w http.ResponseWriter, r *http.Request) error {
	return func(w http.ResponseWriter, r *http.Request) error {
		ads := listResolvableActions(r.Context(), g)
		descMap := map[string]api.ActionDesc{}
		for _, d := range ads {
			descMap[d.Key] = d
		}
		return writeJSON(r.Context(), w, descMap)
	}
}

// listActions lists all the registered actions.
func listActions(g *Genkit) []api.ActionDesc {
	ads := []api.ActionDesc{}

	actions := g.reg.ListActions()
	for _, a := range actions {
		ads = append(ads, a.Desc())
	}

	sort.Slice(ads, func(i, j int) bool {
		return ads[i].Name < ads[j].Name
	})

	return ads
}

// listResolvableActions lists all the registered and resolvable actions.
func listResolvableActions(ctx context.Context, g *Genkit) []api.ActionDesc {
	ads := listActions(g)
	keys := make(map[string]struct{})

	plugins := g.reg.ListPlugins()
	for _, p := range plugins {
		dp, ok := p.(api.DynamicPlugin)
		if !ok {
			// Not all plugins are DynamicPlugins; skip if not.
			continue
		}

		for _, desc := range dp.ListActions(ctx) {
			if _, exists := keys[desc.Name]; !exists {
				ads = append(ads, desc)
				keys[desc.Name] = struct{}{}
			}
		}
	}

	sort.Slice(ads, func(i, j int) bool {
		return ads[i].Name < ads[j].Name
	})

	return ads
}

// TODO: Pull these from common types in genkit-tools.

type runActionResponse struct {
	Result    json.RawMessage `json:"result"`
	Telemetry telemetry       `json:"telemetry"`
}

type telemetry struct {
	TraceID string `json:"traceId"`
}

func runAction(ctx context.Context, g *Genkit, key string, input json.RawMessage, telemetryLabels json.RawMessage, cb streamingCallback[json.RawMessage], runtimeContext map[string]any) (*runActionResponse, error) {
	action := g.reg.ResolveAction(key)
	if action == nil {
		return nil, core.NewError(core.NOT_FOUND, "action %q not found", key)
	}
	if runtimeContext != nil {
		ctx = core.WithActionContext(ctx, runtimeContext)
	}

	var traceID string
	output, err := tracing.RunInNewSpan(ctx, "dev-run-action-wrapper", "", true, input, func(ctx context.Context, input json.RawMessage) (json.RawMessage, error) {
		tracing.SetCustomMetadataAttr(ctx, "genkit-dev-internal", "true")
		// Set telemetry labels from payload to span
		if telemetryLabels != nil {
			var telemetryAttributes map[string]string
			err := json.Unmarshal(telemetryLabels, &telemetryAttributes)
			if err != nil {
				return nil, core.NewError(core.INTERNAL, "Error unmarshalling telemetryLabels: %v", err)
			}
			for k, v := range telemetryAttributes {
				tracing.SetCustomMetadataAttr(ctx, k, v)
			}
		}
		traceID = trace.SpanContextFromContext(ctx).TraceID().String()
		return action.RunJSON(ctx, input, cb)
	})
	if err != nil {
		return nil, err
	}

	return &runActionResponse{
		Result:    output,
		Telemetry: telemetry{TraceID: traceID},
	}, nil
}

// writeJSON writes a JSON-marshaled value to the response writer.
func writeJSON(ctx context.Context, w http.ResponseWriter, value any) error {
	w.Header().Set("Content-Type", "application/json")

	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	_, err = w.Write(data)
	if err != nil {
		logger.FromContext(ctx).Error("writing output", "err", err)
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	return nil
}
