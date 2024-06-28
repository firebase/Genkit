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

// This file implements production and development servers.
//
// The genkit CLI sends requests to the development server.
// See js/common/src/reflectionApi.ts.
//
// The production server has a route for each flow. It
// is intended for production deployments.

package core

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/tracing"
	"go.opentelemetry.io/otel/trace"
)

// InternalInit is for use by the genkit package only.
// It is not subject to compatibility guarantees.
func InternalInit(ctx context.Context, opts *Options) error {
	if opts == nil {
		opts = &Options{}
	}
	globalRegistry.freeze()

	var mu sync.Mutex
	var servers []*http.Server
	var wg sync.WaitGroup
	errCh := make(chan error, 2)

	if currentEnvironment() == EnvironmentDev {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s := startReflectionServer(errCh)
			mu.Lock()
			servers = append(servers, s)
			mu.Unlock()
		}()
	}

	if opts.FlowAddr != "-" {
		wg.Add(1)
		go func() {
			defer wg.Done()
			s := startFlowServer(opts.FlowAddr, opts.Flows, errCh)
			mu.Lock()
			servers = append(servers, s)
			mu.Unlock()
		}()
	}

	serverStartCh := make(chan struct{})
	go func() {
		wg.Wait()
		close(serverStartCh)
	}()

	// It will block here until either all servers start up or there is an error in starting one.
	select {
	case <-serverStartCh:
		slog.Info("all servers started successfully")
	case err := <-errCh:
		return fmt.Errorf("failed to start servers: %w", err)
	case <-ctx.Done():
		return ctx.Err()
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGINT, syscall.SIGTERM)

	// It will block here (i.e. servers will run) until we get an interrupt signal.
	select {
	case sig := <-sigCh:
		slog.Info("received signal, initiating shutdown", "signal", sig)
	case err := <-errCh:
		slog.Error("server error", "err", err)
		return err
	case <-ctx.Done():
		slog.Info("context cancelled, initiating shutdown")
	}

	return shutdownServers(servers)
}

// startReflectionServer starts the Reflection API server listening at the
// value of the environment variable GENKIT_REFLECTION_PORT for the port,
// or ":3100" if it is empty.
func startReflectionServer(errCh chan<- error) *http.Server {
	slog.Info("starting reflection server")
	addr := serverAddress("", "GENKIT_REFLECTION_PORT", "127.0.0.1:3100")
	mux := newDevServeMux(globalRegistry)
	return startServer(addr, mux, errCh)
}

// startFlowServer starts a production server listening at the given address.
// The Server has a route for each defined flow.
// If addr is "", it uses the value of the environment variable PORT
// for the port, and if that is empty it uses ":3400".
//
// To construct a server with additional routes, use [NewFlowServeMux].
func startFlowServer(addr string, flows []string, errCh chan<- error) *http.Server {
	slog.Info("starting flow server")
	addr = serverAddress(addr, "PORT", "127.0.0.1:3400")
	mux := NewFlowServeMux(flows)
	return startServer(addr, mux, errCh)
}

// startServer starts an HTTP server listening on the address.
// It returns the server an
func startServer(addr string, handler http.Handler, errCh chan<- error) *http.Server {
	server := &http.Server{
		Addr:    addr,
		Handler: handler,
	}

	go func() {
		slog.Info("server listening", "addr", addr)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- fmt.Errorf("server error on %s: %w", addr, err)
		}
	}()

	return server
}

// shutdownServers initiates shutdown of the servers and waits for the shutdown to complete.
// After 5 seconds, it will timeout.
func shutdownServers(servers []*http.Server) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var wg sync.WaitGroup
	for _, server := range servers {
		wg.Add(1)
		go func(srv *http.Server) {
			defer wg.Done()
			if err := srv.Shutdown(ctx); err != nil {
				slog.Error("server shutdown failed", "addr", srv.Addr, "err", err)
			} else {
				slog.Info("server shutdown successfully", "addr", srv.Addr)
			}
		}(server)
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		slog.Info("all servers shut down successfully")
	case <-ctx.Done():
		return errors.New("server shutdown timed out")
	}

	return nil
}

// Options are options to [InternalInit].
type Options struct {
	// If "-", do not start a FlowServer.
	// Otherwise, start a FlowServer on the given address, or the
	// default if empty.
	FlowAddr string
	// The names of flows to serve.
	// If empty, all registered flows are served.
	Flows []string
}

type devServer struct {
	reg *registry
}

func newDevServeMux(r *registry) *http.ServeMux {
	mux := http.NewServeMux()
	s := &devServer{r}
	handle(mux, "GET /api/__health", func(w http.ResponseWriter, _ *http.Request) error {
		return nil
	})
	handle(mux, "POST /api/runAction", s.handleRunAction)
	handle(mux, "GET /api/actions", s.handleListActions)
	handle(mux, "GET /api/envs/{env}/traces/{traceID}", s.handleGetTrace)
	handle(mux, "GET /api/envs/{env}/traces", s.handleListTraces)
	handle(mux, "GET /api/envs/{env}/flowStates", s.handleListFlowStates)

	return mux
}

// handleRunAction looks up an action by name in the registry, runs it with the
// provided JSON input, and writes back the JSON-marshaled request.
func (s *devServer) handleRunAction(w http.ResponseWriter, r *http.Request) error {
	ctx := r.Context()
	var body struct {
		Key   string          `json:"key"`
		Input json.RawMessage `json:"input"`
	}
	defer r.Body.Close()
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		return &httpError{http.StatusBadRequest, err}
	}
	stream := false
	if s := r.FormValue("stream"); s != "" {
		var err error
		stream, err = strconv.ParseBool(s)
		if err != nil {
			return err
		}
	}
	logger.FromContext(ctx).Debug("running action",
		"key", body.Key,
		"stream", stream)
	var callback streamingCallback[json.RawMessage]
	if stream {
		// Stream results are newline-separated JSON.
		callback = func(ctx context.Context, msg json.RawMessage) error {
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
	resp, err := runAction(ctx, s.reg, body.Key, body.Input, callback)
	if err != nil {
		return err
	}
	return writeJSON(ctx, w, resp)
}

type runActionResponse struct {
	Result    json.RawMessage `json:"result"`
	Telemetry telemetry       `json:"telemetry"`
}

type telemetry struct {
	TraceID string `json:"traceId"`
}

func runAction(ctx context.Context, reg *registry, key string, input json.RawMessage, cb streamingCallback[json.RawMessage]) (*runActionResponse, error) {
	action := reg.lookupAction(key)
	if action == nil {
		return nil, &httpError{http.StatusNotFound, fmt.Errorf("no action with key %q", key)}
	}
	var traceID string
	output, err := tracing.RunInNewSpan(ctx, reg.tstate, "dev-run-action-wrapper", "", true, input, func(ctx context.Context, input json.RawMessage) (json.RawMessage, error) {
		tracing.SetCustomMetadataAttr(ctx, "genkit-dev-internal", "true")
		traceID = trace.SpanContextFromContext(ctx).TraceID().String()
		return action.runJSON(ctx, input, cb)
	})
	if err != nil {
		return nil, err
	}
	return &runActionResponse{
		Result:    output,
		Telemetry: telemetry{TraceID: traceID},
	}, nil
}

// handleListActions lists all the registered actions.
func (s *devServer) handleListActions(w http.ResponseWriter, r *http.Request) error {
	descs := s.reg.listActions()
	descMap := map[string]actionDesc{}
	for _, d := range descs {
		descMap[d.Key] = d
	}
	return writeJSON(r.Context(), w, descMap)
}

// handleGetTrace returns a single trace from a TraceStore.
func (s *devServer) handleGetTrace(w http.ResponseWriter, r *http.Request) error {
	env := r.PathValue("env")
	ts := s.reg.lookupTraceStore(Environment(env))
	if ts == nil {
		return &httpError{http.StatusNotFound, fmt.Errorf("no TraceStore for environment %q", env)}
	}
	tid := r.PathValue("traceID")
	td, err := ts.Load(r.Context(), tid)
	if errors.Is(err, fs.ErrNotExist) {
		return &httpError{http.StatusNotFound, fmt.Errorf("no %s trace with ID %q", env, tid)}
	}
	if err != nil {
		return err
	}
	return writeJSON(r.Context(), w, td)
}

// handleListTraces returns a list of traces from a TraceStore.
func (s *devServer) handleListTraces(w http.ResponseWriter, r *http.Request) error {
	env := r.PathValue("env")
	ts := s.reg.lookupTraceStore(Environment(env))
	if ts == nil {
		return &httpError{http.StatusNotFound, fmt.Errorf("no TraceStore for environment %q", env)}
	}
	limit := 0
	if lim := r.FormValue("limit"); lim != "" {
		var err error
		limit, err = strconv.Atoi(lim)
		if err != nil {
			return &httpError{http.StatusBadRequest, err}
		}
	}
	ctoken := r.FormValue("continuationToken")
	tds, ctoken, err := ts.List(r.Context(), &tracing.Query{Limit: limit, ContinuationToken: ctoken})
	if errors.Is(err, tracing.ErrBadQuery) {
		return &httpError{http.StatusBadRequest, err}
	}
	if err != nil {
		return err
	}
	if tds == nil {
		tds = []*tracing.Data{}
	}
	return writeJSON(r.Context(), w, listTracesResult{tds, ctoken})
}

type listTracesResult struct {
	Traces            []*tracing.Data `json:"traces"`
	ContinuationToken string          `json:"continuationToken"`
}

func (s *devServer) handleListFlowStates(w http.ResponseWriter, r *http.Request) error {
	// TODO(jba): implement.
	return writeJSON(r.Context(), w, listFlowStatesResult{[]flowStater{}, ""})
}

type listFlowStatesResult struct {
	FlowStates        []flowStater `json:"flowStates"`
	ContinuationToken string       `json:"continuationToken"`
}

// NewFlowServeMux constructs a [net/http.ServeMux].
// If flows is non-empty, the each of the named flows is registered as a route.
// Otherwise, all defined flows are registered.
//
// All routes take a single query parameter, "stream", which if true will stream the
// flow's results back to the client. (Not all flows support streaming, however.)
//
// To use the returned ServeMux as part of a server with other routes, either add routes
// to it, or install it as part of another ServeMux, like so:
//
//	mainMux := http.NewServeMux()
//	mainMux.Handle("POST /flow/", http.StripPrefix("/flow/", NewFlowServeMux()))
func NewFlowServeMux(flows []string) *http.ServeMux {
	return newFlowServeMux(globalRegistry, flows)
}

func newFlowServeMux(r *registry, flows []string) *http.ServeMux {
	mux := http.NewServeMux()
	m := map[string]bool{}
	for _, f := range flows {
		m[f] = true
	}
	for _, f := range r.listFlows() {
		if len(flows) == 0 || m[f.Name()] {
			handle(mux, "POST /"+f.Name(), nonDurableFlowHandler(f))
		}
	}
	return mux
}

func nonDurableFlowHandler(f flow) func(http.ResponseWriter, *http.Request) error {
	return func(w http.ResponseWriter, r *http.Request) error {
		defer r.Body.Close()
		input, err := io.ReadAll(r.Body)
		if err != nil {
			return err
		}
		stream, err := parseBoolQueryParam(r, "stream")
		if err != nil {
			return err
		}
		if stream {
			// TODO(jba): implement streaming.
			return &httpError{http.StatusNotImplemented, errors.New("streaming")}
		} else {
			// TODO(jba): telemetry
			out, err := f.runJSON(r.Context(), json.RawMessage(input), nil)
			if err != nil {
				return err
			}
			// Responses for non-streaming, non-durable flows are passed back
			// with the flow result stored in a field called "result."
			_, err = fmt.Fprintf(w, `{"result": %s}\n`, out)
			return err
		}
	}
}

// serverAddress determines a server address.
func serverAddress(arg, envVar, defaultValue string) string {
	if arg != "" {
		return arg
	}
	if port := os.Getenv(envVar); port != "" {
		return "127.0.0.1:" + port
	}
	return defaultValue
}

func listenAndServe(addr string, mux *http.ServeMux) error {
	server := &http.Server{
		Addr:    addr,
		Handler: mux,
	}
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM)
	go func() {
		<-sigCh
		slog.Info("received SIGTERM, shutting down server")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Shutdown(ctx); err != nil {
			slog.Error("server shutdown failed", "err", err)
		} else {
			slog.Info("server shutdown successfully")
		}
	}()
	slog.Info("listening", "addr", addr)
	return server.ListenAndServe()
}

// requestID is a unique ID for each request.
var requestID atomic.Int64

// handle registers pattern on mux with an http.Handler that calls f.
// If f returns a non-nil error, the handler calls http.Error.
// If the error is an httpError, the code it contains is used as the status code;
// otherwise a 500 status is used.
func handle(mux *http.ServeMux, pattern string, f func(w http.ResponseWriter, r *http.Request) error) {
	mux.HandleFunc(pattern, func(w http.ResponseWriter, r *http.Request) {
		id := requestID.Add(1)
		// Create a logger that always outputs the requestID, and store it in the request context.
		log := slog.Default().With("reqID", id)
		log.Info("request start",
			"method", r.Method,
			"path", r.URL.Path)
		var err error
		defer func() {
			if err != nil {
				log.Error("request end", "err", err)
			} else {
				log.Info("request end")
			}
		}()
		err = f(w, r)
		if err != nil {
			// If the error is an httpError, serve the status code it contains.
			// Otherwise, assume this is an unexpected error and serve a 500.
			var herr *httpError
			if errors.As(err, &herr) {
				http.Error(w, herr.Error(), herr.code)
			} else {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
		}
	})
}

type httpError struct {
	code int
	err  error
}

func (e *httpError) Error() string {
	return fmt.Sprintf("%s: %s", http.StatusText(e.code), e.err)
}

func parseBoolQueryParam(r *http.Request, name string) (bool, error) {
	b := false
	if s := r.FormValue(name); s != "" {
		var err error
		b, err = strconv.ParseBool(s)
		if err != nil {
			return false, &httpError{http.StatusBadRequest, err}
		}
	}
	return b, nil
}

func currentEnvironment() Environment {
	if v := os.Getenv("GENKIT_ENV"); v != "" {
		return Environment(v)
	}
	return EnvironmentProd
}

func writeJSON(ctx context.Context, w http.ResponseWriter, value any) error {
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
