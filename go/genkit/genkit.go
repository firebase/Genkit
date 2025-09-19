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

// Package genkit provides Genkit functionality for application developers.
package genkit

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/registry"
)

// Genkit encapsulates a Genkit instance, providing access to its registry,
// configuration, and core functionalities. It serves as the central hub for
// defining and managing Genkit resources like flows, models, tools, and prompts.
//
// A Genkit instance is created using [Init].
type Genkit struct {
	reg *registry.Registry // Registry for actions, values, and other resources.
}

// genkitOptions are options for configuring the Genkit instance.
type genkitOptions struct {
	DefaultModel string       // Default model to use if no other model is specified.
	PromptDir    string       // Directory where dotprompts are stored. Will be loaded automatically on initialization.
	Plugins      []api.Plugin // Plugin to initialize automatically.
}

type GenkitOption interface {
	apply(g *genkitOptions) error
}

// apply applies the options to the Genkit options.
func (o *genkitOptions) apply(gOpts *genkitOptions) error {
	if o.DefaultModel != "" {
		if gOpts.DefaultModel != "" {
			return errors.New("cannot set default model more than once (WithDefaultModel)")
		}
		gOpts.DefaultModel = o.DefaultModel
	}

	if o.PromptDir != "" {
		if gOpts.PromptDir != "" {
			return errors.New("cannot set prompt directory more than once (WithPromptDir)")
		}
		gOpts.PromptDir = o.PromptDir
	}

	if len(o.Plugins) > 0 {
		if gOpts.Plugins != nil {
			return errors.New("cannot set plugins more than once (WithPlugins)")
		}
		gOpts.Plugins = o.Plugins
	}

	return nil
}

// WithPlugins provides a list of plugins to initialize when creating the Genkit instance.
// Each plugin's [Plugin.Init] method will be called sequentially during [Init].
// This option can only be applied once.
func WithPlugins(plugins ...api.Plugin) GenkitOption {
	return &genkitOptions{Plugins: plugins}
}

// WithDefaultModel sets the default model name to use for generation tasks
// when no specific model is provided in the request options. The name should
// correspond to a model registered either by a plugin or via [DefineModel].
// This option can only be applied once.
func WithDefaultModel(model string) GenkitOption {
	return &genkitOptions{DefaultModel: model}
}

// WithPromptDir specifies the directory where `.prompt` files are located.
// Prompts are automatically loaded from this directory during [Init].
// The default directory is "prompts" relative to the project root where
// [Init] is called.
//
// Invalid prompt files will result in logged errors during initialization,
// while valid files that define invalid prompts will cause [Init] to panic.
// This option can only be applied once.
func WithPromptDir(dir string) GenkitOption {
	return &genkitOptions{PromptDir: dir}
}

// Init creates and initializes a new [Genkit] instance with the provided options.
// It sets up the registry, initializes plugins ([WithPlugins]), loads prompts
// ([WithPromptDir]), and configures other settings like the default model
// ([WithDefaultModel]).
//
// During local development (when the `GENKIT_ENV` environment variable is set to `dev`),
// Init also starts the Reflection API server as a background goroutine. This server
// provides metadata about registered actions and is used by developer tools.
// By default, it listens on port 3100.
//
// The provided context should handle application shutdown signals (like SIGINT, SIGTERM)
// to ensure graceful termination of background processes, including the reflection server.
//
// Example:
//
//	package main
//
//	import (
//		"context"
//		"log"
//
//		"github.com/firebase/genkit/go/ai"
//		"github.com/firebase/genkit/go/genkit"
//		"github.com/firebase/genkit/go/plugins/googlegenai" // Example plugin
//	)
//
//	func main() {
//		ctx := context.Background()
//
//		// Assumes a prompt file at ./prompts/jokePrompt.prompt
//		g := genkit.Init(ctx,
//			genkit.WithPlugins(&googlegenai.GoogleAI{}),
//			genkit.WithDefaultModel("googleai/gemini-2.0-flash"),
//			genkit.WithPromptDir("./prompts"),
//		)
//
//		// Generate text using the default model
//		funFact, err := genkit.GenerateText(ctx, g, ai.WithPrompt("Tell me a fake fun fact!"))
//		if err != nil {
//			log.Fatalf("GenerateText failed: %v", err)
//		}
//		log.Println("Generated Fact:", funFact)
//
//		// Look up and execute a loaded prompt
//		jokePrompt := genkit.LookupPrompt(g, "jokePrompt")
//		if jokePrompt == nil {
//			log.Fatalf("Prompt 'jokePrompt' not found.")
//		}
//
//		resp, err := jokePrompt.Execute(ctx, nil) // Execute with default input (if any)
//		if err != nil {
//			log.Fatalf("jokePrompt.Execute failed: %v", err)
//		}
//		log.Println("Generated joke:", resp.Text())
//	}
func Init(ctx context.Context, opts ...GenkitOption) *Genkit {
	ctx, _ = signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)

	gOpts := &genkitOptions{}
	for _, opt := range opts {
		if err := opt.apply(gOpts); err != nil {
			panic(fmt.Errorf("genkit.Init: error applying options: %w", err))
		}
	}

	r := registry.New()
	g := &Genkit{reg: r}

	for _, plugin := range gOpts.Plugins {
		actions := plugin.Init(ctx)
		for _, action := range actions {
			action.Register(r)
		}
		r.RegisterPlugin(plugin.Name(), plugin)
	}

	ai.ConfigureFormats(r)
	ai.DefineGenerateAction(ctx, r)
	ai.LoadPromptDir(r, gOpts.PromptDir, "")

	r.RegisterValue(api.DefaultModelKey, gOpts.DefaultModel)
	r.RegisterValue(api.PromptDirKey, gOpts.PromptDir)

	if api.CurrentEnvironment() == api.EnvironmentDev {
		errCh := make(chan error, 1)
		serverStartCh := make(chan struct{})

		go func() {
			if s := startReflectionServer(ctx, g, errCh, serverStartCh); s == nil {
				return
			}
			if err := <-errCh; err != nil {
				slog.Error("reflection server error", "err", err)
			}
		}()

		select {
		case err := <-errCh:
			panic(fmt.Errorf("genkit.Init: reflection server startup failed: %w", err))
		case <-serverStartCh:
			slog.Debug("reflection server started successfully")
		case <-ctx.Done():
			panic(ctx.Err())
		}
	}

	return g
}

// RegisterAction registers a [api.Action] that was previously created by calling
// NewX instead of DefineX.
//
// Example:
//
//	model := ai.NewModel(...)
//	genkit.RegisterAction(g, model)
func RegisterAction(g *Genkit, action api.Registerable) {
	action.Register(g.reg)
}

// DefineFlow defines a non-streaming flow, registers it as a [core.Action] of type Flow,
// and returns a [core.Flow] runner.
// The provided function `fn` takes an input of type `In` and returns an output of type `Out`.
// Flows are the primary mechanism for orchestrating multi-step AI tasks in Genkit.
// Each run of a flow is traced, and steps within the flow can be traced using [Run].
//
// Example:
//
//	myFlow := genkit.DefineFlow(g, "mySimpleFlow",
//		func(ctx context.Context, name string) (string, error) {
//			greeting := fmt.Sprintf("Hello, %s!", name)
//			// You could add more steps here, potentially using genkit.Run()
//			return greeting, nil
//		},
//	)
//
//	// Later, run the flow:
//	result, err := myFlow.Run(ctx, "World")
//	if err != nil {
//		// handle error
//	}
//	fmt.Println(result) // Output: Hello, World!
func DefineFlow[In, Out any](g *Genkit, name string, fn core.Func[In, Out]) *core.Flow[In, Out, struct{}] {
	return core.DefineFlow(g.reg, name, fn)
}

// DefineStreamingFlow defines a streaming flow, registers it as a [core.Action] of type Flow,
// and returns a [core.Flow] runner capable of streaming.
//
// The provided function `fn` takes an input of type `In`. It can optionally stream
// intermediate results of type `Stream` by invoking the provided callback function.
// Finally, it returns a final output of type `Out`.
//
// If the function supports streaming and the callback is non-nil when the flow is run,
// it should invoke the callback periodically with `Stream` values. The final `Out` value,
// typically an aggregation of the streamed data, is returned at the end.
// If the callback is nil or the function doesn't support streaming for a given input,
// it should simply compute and return the `Out` value directly.
//
// Example:
//
//	counterFlow := genkit.DefineStreamingFlow(g, "counter",
//		func(ctx context.Context, limit int, stream func(context.Context, int) error) (string, error) {
//			if stream == nil { // Non-streaming case
//				return fmt.Sprintf("Counted up to %d", limit), nil
//			}
//			// Streaming case
//			for i := 1; i <= limit; i++ {
//				if err := stream(ctx, i); err != nil {
//					return "", fmt.Errorf("streaming error: %w", err)
//				}
//				// time.Sleep(100 * time.Millisecond) // Optional delay
//			}
//			return fmt.Sprintf("Finished counting to %d", limit), nil
//		},
//	)
//
//	// Later, run the flow with streaming:
//	streamCh, err := counterFlow.Stream(ctx, 5)
//	if err != nil {
//		// handle error
//	}
//	for result := range streamCh {
//		if result.Err != nil {
//			log.Printf("Stream error: %v", result.Err)
//			break
//		}
//		if result.Done {
//			fmt.Println("Final Output:", result.Output) // Output: Finished counting to 5
//		} else {
//			fmt.Println("Stream Chunk:", result.Stream) // Outputs: 1, 2, 3, 4, 5
//		}
//	}
func DefineStreamingFlow[In, Out, Stream any](g *Genkit, name string, fn core.StreamingFunc[In, Out, Stream]) *core.Flow[In, Out, Stream] {
	return core.DefineStreamingFlow(g.reg, name, fn)
}

// Run executes the given function `fn` within the context of the current flow run,
// creating a distinct trace span for this step. It's used to add observability
// to specific sub-operations within a flow defined by [DefineFlow] or [DefineStreamingFlow].
// The `name` parameter provides a label for the trace span.
// It returns the output of `fn` and any error it produces.
//
// Example (within a DefineFlow function):
//
//	complexFlow := genkit.DefineFlow(g, "complexTask",
//		func(ctx context.Context, input string) (string, error) {
//			// Step 1: Process input (traced as "process-input")
//			processedInput, err := genkit.Run(ctx, "process-input", func() (string, error) {
//				// ... some processing ...
//				return strings.ToUpper(input), nil
//			})
//			if err != nil {
//				return "", err
//			}
//
//			// Step 2: Generate response (traced as "generate-response")
//			response, err := genkit.Run(ctx, "generate-response", func() (string, error) {
//				// ... call an AI model or another service ...
//				return "Response for " + processedInput, nil
//			})
//			if err != nil {
//				return "", err
//			}
//
//			return response, nil
//		},
//	)
func Run[Out any](ctx context.Context, name string, fn func() (Out, error)) (Out, error) {
	return core.Run(ctx, name, fn)
}

// ListFlows returns a slice of all [api.Action] instances that represent
// flows registered with the Genkit instance `g`.
// This is useful for introspection or for dynamically exposing flow endpoints,
// for example, in an HTTP server.
func ListFlows(g *Genkit) []api.Action {
	acts := listActions(g)
	flows := []api.Action{}
	for _, act := range acts {
		if act.Type == api.ActionTypeFlow {
			flows = append(flows, g.reg.LookupAction(act.Key))
		}
	}
	return flows
}

// ListTools returns a slice of all [ai.Tool] instances that are registered
// with the Genkit instance `g`. This is useful for introspection and for
// exposing tools to external systems like MCP servers.
func ListTools(g *Genkit) []ai.Tool {
	acts := g.reg.ListActions()
	tools := []ai.Tool{}
	for _, action := range acts {
		tool := LookupTool(g, action.Desc().Name)
		if tool != nil {
			tools = append(tools, tool)
		}
	}
	return tools
}

// DefineModel defines a custom model implementation, registers it as a [core.Action]
// of type Model, and returns an [ai.Model] interface.
//
// The `provider` and `name` arguments form the unique identifier for the model
// (e.g., "myProvider/myModel"). The `info` argument provides metadata about the
// model's capabilities ([ai.ModelInfo]). The `fn` argument ([ai.ModelFunc])
// implements the actual generation logic, handling input requests ([ai.ModelRequest])
// and producing responses ([ai.ModelResponse]), potentially streaming chunks
// ([ai.ModelResponseChunk]) via the callback.
//
// Example:
//
//	echoModel := genkit.DefineModel(g, "custom/echo",
//		&ai.ModelOptions{
//			Label:    "Echo Model",
//			Supports: &ai.ModelSupports{Multiturn: true},
//		},
//		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
//			// Simple echo implementation
//			resp := &ai.ModelResponse{
//				Message: &ai.Message{
//					Role:    ai.RoleModel,
//					Content: []*ai.Part{},
//				},
//			}
//			// Combine content from the last user message
//			var responseText strings.Builder
//			if len(req.Messages) > 0 {
//				lastMsg := req.Messages[len(req.Messages)-1]
//				if lastMsg.Role == ai.RoleUser {
//					for _, part := range lastMsg.Content {
//						if part.IsText() {
//							responseText.WriteString(part.Text)
//						}
//					}
//				}
//			}
//			if responseText.Len() == 0 {
//				responseText.WriteString("...")
//			}
//
//			resp.Message.Content = append(resp.Message.Content, ai.NewTextPart(responseText.String()))
//
//			// Example of streaming (optional)
//			if cb != nil {
//				chunk := &ai.ModelResponseChunk{ Index: 0, Content: resp.Message.Content }
//				if err := cb(ctx, chunk); err != nil {
//					return nil, err // Handle streaming error
//				}
//			}
//
//			resp.FinishReason = ai.FinishReasonStop
//			return resp, nil
//		},
//	)
func DefineModel(g *Genkit, name string, opts *ai.ModelOptions, fn ai.ModelFunc) ai.Model {
	return ai.DefineModel(g.reg, name, opts, fn)
}

func DefineBackgroundModel(g *Genkit, name string, opts *ai.BackgroundModelOptions) ai.BackgroundModel {
	return ai.DefineBackgroundModel(g.reg, name, opts)
}

// LookupModel retrieves a registered [ai.Model] by its provider and name.
// It returns the model instance if found, or `nil` if no model with the
// given identifier is registered (e.g., via [DefineModel] or a plugin).
// It will try to resolve the model dynamically by matching the provider name;
// this does not necessarily mean the model is valid.
func LookupModel(g *Genkit, name string) ai.Model {
	return ai.LookupModel(g.reg, name)
}

// LookupBackgroundModel retrieves a registered background model by its provider and name.
// It returns the background action instance if found, or `nil` if no background model with the
// given identifier is registered.
func LookupBackgroundModel(g *Genkit, name string) ai.BackgroundModel {
	return ai.LookupBackgroundModel(g.reg, name)
}

// DefineTool defines a tool that can be used by models during generation,
// registers it as a [core.Action] of type Tool, and returns an [ai.ToolDef].
// Tools allow models to interact with external systems or perform specific computations.
//
// The `name` is the identifier the model uses to request the tool. The `description`
// helps the model understand when to use the tool. The function `fn` implements
// the tool's logic, taking an [ai.ToolContext] and an input of type `In`, and
// returning an output of type `Out`. The input and output types determine the
// `inputSchema` and `outputSchema` in the tool's definition, which guide the model
// on how to provide input and interpret output.
//
// Example:
//
//	weatherTool := genkit.DefineTool(g, "getWeather", "Fetches the weather for a given city",
//		func(ctx *ai.ToolContext, city string) (string, error) {
//			// In a real scenario, call a weather API
//			log.Printf("Tool: Fetching weather for %s", city)
//			if city == "Paris" {
//				return "Sunny, 25°C", nil
//			}
//			return "Cloudy, 18°C", nil
//		},
//	)
//
//	// Use the tool in a generation request:
//	resp, err := genkit.Generate(ctx, g,
//		ai.WithPrompt("What's the weather like in Paris?"),
//		ai.WithTools(weatherTool), // Make the tool available
//		// Optionally use ai.WithToolChoice(...)
//	)
//	if err != nil {
//		log.Fatalf("Generate failed: %v", err)
//	}
//
//	fmt.Println(resp.Text()) // Might output something like "The weather in Paris is Sunny, 25°C."
func DefineTool[In, Out any](g *Genkit, name, description string, fn ai.ToolFunc[In, Out]) ai.Tool {
	return ai.DefineTool(g.reg, name, description, fn)
}

// DefineToolWithInputSchema defines a tool with a custom input schema that can be used by models during generation,
// registers it as a [core.Action] of type Tool, and returns an [ai.Tool].
//
// This variant of [DefineTool] allows specifying a JSON Schema for the tool's input, providing more
// control over input validation and model guidance. The input parameter to the tool function will be
// of type `any` and should be validated/processed according to the schema.
//
// The `name` is the identifier the model uses to request the tool. The `description` helps the model
// understand when to use the tool. The `inputSchema` defines the expected structure and constraints
// of the input. The function `fn` implements the tool's logic, taking an [ai.ToolContext] and an
// input of type `any`, and returning an output of type `Out`.
//
// Example:
//
//	// Define a custom input schema
//	inputSchema := map[string]any{
//		"type": "object",
//		"properties": map[string]any{
//			"city": map[string]any{"type": "string"},
//			"unit": map[string]any{
//				"type": "string",
//				"enum": []any{"C", "F"},
//			},
//		},
//		"required": []string{"city"},
//	}
//
//	// Define the tool with the schema
//	weatherTool := genkit.DefineToolWithInputSchema(g, "getWeather",
//		"Fetches the weather for a given city with unit preference",
//		inputSchema,
//		func(ctx *ai.ToolContext, input any) (string, error) {
//			// Parse and validate input
//			data := input.(map[string]any)
//			city := data["city"].(string)
//			unit := "C" // default
//			if u, ok := data["unit"].(string); ok {
//				unit = u
//			}
//			// Implementation...
//			return fmt.Sprintf("Weather in %s: 25°%s", city, unit), nil
//		},
//	)
func DefineToolWithInputSchema[Out any](g *Genkit, name, description string, inputSchema map[string]any, fn ai.ToolFunc[any, Out]) ai.Tool {
	return ai.DefineToolWithInputSchema(g.reg, name, description, inputSchema, fn)
}

// LookupTool retrieves a registered [ai.Tool] by its name.
// It returns the tool instance if found, or `nil` if no tool with the
// given name is registered (e.g., via [DefineTool]).
func LookupTool(g *Genkit, name string) ai.Tool {
	return ai.LookupTool(g.reg, name)
}

// DefinePrompt defines a prompt programmatically, registers it as a [core.Action]
// of type Prompt, and returns an executable [ai.prompt].
//
// This provides an alternative to defining prompts in `.prompt` files, offering
// more flexibility through Go code. Prompts encapsulate configuration (model, parameters),
// message templates (system, user, history), input/output schemas, and associated tools.
//
// Prompts can be executed in two main ways:
//  1. Render + Generate: Call [Prompt.Render] to get [ai.GenerateActionOptions],
//     modify them if needed, and pass them to [GenerateWithRequest].
//  2. Execute: Call [Prompt.Execute] directly, passing input and execution options.
//
// Options ([ai.PromptOption]) are used to configure the prompt during definition.
//
// Example:
//
//	type GeoInput struct {
//		Country string `json:"country"`
//	}
//
//	type GeoOutput struct {
//		Capital string `json:"capital"`
//	}
//
//	// Define the prompt
//	capitalPrompt := genkit.DefinePrompt(g, "findCapital",
//		ai.WithDescription("Finds the capital of a country."),
//		ai.WithModelName("googleai/gemini-1.5-flash"), // Specify the model
//		ai.WithSystem("You are a helpful geography assistant."),
//		ai.WithPrompt("What is the capital of {{country}}?"),
//		ai.WithInputType(GeoInput{Country: "USA"}),
//		ai.WithOutputType(GeoOutput{}),
//		ai.WithConfig(&ai.GenerationCommonConfig{Temperature: 0.5}),
//	)
//
//	// Option 1: Render + Generate (using default input "USA")
//	actionOpts, err := capitalPrompt.Render(ctx, nil) // nil input uses default
//	if err != nil {
//		log.Fatalf("Render failed: %v", err)
//	}
//	resp1, err := genkit.GenerateWithRequest(ctx, g, actionOpts, nil, nil)
//	if err != nil {
//		log.Fatalf("GenerateWithRequest failed: %v", err)
//	}
//	var out1 GeoOutput
//	if err = resp1.Output(&out1); err != nil {
//		log.Fatalf("Output failed: %v", err)
//	}
//	fmt.Printf("Capital of USA: %s\n", out1.Capital) // Output: Capital of USA: Washington D.C.
//
//	// Option 2: Execute directly (with new input)
//	resp2, err := capitalPrompt.Execute(ctx, ai.WithInput(GeoInput{Country: "France"}))
//	if err != nil {
//		log.Fatalf("Execute failed: %v", err)
//	}
//	var out2 GeoOutput
//	if err = resp2.Output(&out2); err != nil {
//		log.Fatalf("Output failed: %v", err)
//	}
//	fmt.Printf("Capital of France: %s\n", out2.Capital) // Output: Capital of France: Paris
func DefinePrompt(g *Genkit, name string, opts ...ai.PromptOption) ai.Prompt {
	return ai.DefinePrompt(g.reg, name, opts...)
}

// LookupPrompt retrieves a registered [ai.prompt] by its name.
// Prompts can be registered via [DefinePrompt] or loaded automatically from
// `.prompt` files in the directory specified by [WithPromptDir] or [LoadPromptDir].
// It returns the prompt instance if found, or `nil` otherwise.
func LookupPrompt(g *Genkit, name string) ai.Prompt {
	return ai.LookupPrompt(g.reg, name)
}

// GenerateWithRequest performs a model generation request using explicitly provided
// [ai.GenerateActionOptions]. This function is typically used in conjunction with
// prompts defined via [DefinePrompt], where [ai.prompt.Render] produces the
// `actionOpts`. It allows fine-grained control over the request sent to the model.
//
// It accepts optional model middleware (`mw`) for intercepting/modifying the request/response,
// and an optional streaming callback (`cb`) of type [ai.ModelStreamCallback] to receive
// response chunks as they arrive.
//
// Example (using options rendered from a prompt):
//
//	myPrompt := genkit.LookupPrompt(g, "myDefinedPrompt")
//	actionOpts, err := myPrompt.Render(ctx, map[string]any{"topic": "go programming"})
//	if err != nil {
//		// handle error
//	}
//
//	// Optional: Modify actionOpts here if needed
//	// actionOpts.Config = &ai.GenerationCommonConfig{ Temperature: 0.8 }
//
//	resp, err := genkit.GenerateWithRequest(ctx, g, actionOpts, nil, nil) // No middleware or streaming
//	if err != nil {
//		// handle error
//	}
//	fmt.Println(resp.Text())
func GenerateWithRequest(ctx context.Context, g *Genkit, actionOpts *ai.GenerateActionOptions, mw []ai.ModelMiddleware, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
	return ai.GenerateWithRequest(ctx, g.reg, actionOpts, mw, cb)
}

// Generate performs a model generation request using a flexible set of options
// provided via [ai.GenerateOption] arguments. It's a convenient way to make
// generation calls without pre-defining a prompt object.
//
// Example:
//
//	resp, err := genkit.Generate(ctx, g,
//		ai.WithModelName("googleai/gemini-2.0-flash"),
//		ai.WithPrompt("Write a short poem about clouds."),
//		ai.WithConfig(&genai.GenerateContentConfig{MaxOutputTokens: 50}),
//	)
//	if err != nil {
//		log.Fatalf("Generate failed: %v", err)
//	}
//
//	fmt.Println(resp.Text())
func Generate(ctx context.Context, g *Genkit, opts ...ai.GenerateOption) (*ai.ModelResponse, error) {
	return ai.Generate(ctx, g.reg, opts...)
}

// GenerateOperation performs a background model generation request)
func GenerateOperation(ctx context.Context, g *Genkit, opts ...ai.GenerateOption) (*core.Operation[*ai.ModelResponse], error) {
	return ai.GenerateOperation(ctx, g.reg, opts...)
}

// GenerateText performs a model generation request similar to [Generate], but
// directly returns the generated text content as a string. It's a convenience
// wrapper for cases where only the textual output is needed.
// It accepts the same [ai.GenerateOption] arguments as [Generate].
//
// Example:
//
//	joke, err := genkit.GenerateText(ctx, g,
//		ai.WithPrompt("Tell me a funny programming joke."),
//	)
//	if err != nil {
//		log.Fatalf("GenerateText failed: %v", err)
//	}
//	fmt.Println(joke)
func GenerateText(ctx context.Context, g *Genkit, opts ...ai.GenerateOption) (string, error) {
	return ai.GenerateText(ctx, g.reg, opts...)
}

// GenerateData performs a model generation request, expecting structured output
// (typically JSON) that conforms to the schema of the provided `value` argument.
// It attempts to unmarshal the model's response directly into the `value`.
// The `value` argument must be a pointer to a struct or map.
//
// Use [ai.WithOutputType] or [ai.WithOutputFormat](ai.OutputFormatJSON) in the
// options to instruct the model to generate JSON. [ai.WithOutputType] is preferred
// as it infers the JSON schema from the `value` type and passes it to the model.
//
// It returns the full [ai.ModelResponse] along with any error. The generated data
// populates the `value` pointed to.
//
// Example:
//
//	type BookInfo struct {
//		Title  string `json:"title"`
//		Author string `json:"author"`
//		Year   int    `json:"year"`
//	}
//
//	book, _, err := genkit.GenerateData[BookInfo](ctx, g,
//		ai.WithPrompt("Tell me about 'The Hitchhiker's Guide to the Galaxy'."),
//	)
//	if err != nil {
//		log.Fatalf("GenerateData failed: %v", err)
//	}
//
//	log.Printf("Book: %+v\n", book) // Output: Book: {Title:The Hitchhiker's Guide to the Galaxy Author:Douglas Adams Year:1979}
func GenerateData[Out any](ctx context.Context, g *Genkit, opts ...ai.GenerateOption) (*Out, *ai.ModelResponse, error) {
	return ai.GenerateData[Out](ctx, g.reg, opts...)
}

// Retrieve performs a document retrieval request using a flexible set of options
// provided via [ai.RetrieverOption] arguments. It's a convenient way to retrieve
// relevant documents from registered retrievers without directly calling the
// retriever instance.
//
// Example:
//
//	resp, err := genkit.Retrieve(ctx, g,
//		ai.WithRetriever(ai.NewRetrieverRef("myRetriever", nil)),
//		ai.WithTextDocs("What is the capital of France?"),
//	)
//	if err != nil {
//		log.Fatalf("Retrieve failed: %v", err)
//	}
//
//	for _, doc := range resp.Documents {
//		fmt.Printf("Document: %+v\n", doc)
//	}
func Retrieve(ctx context.Context, g *Genkit, opts ...ai.RetrieverOption) (*ai.RetrieverResponse, error) {
	return ai.Retrieve(ctx, g.reg, opts...)
}

// Embed performs an embedding request using a flexible set of options
// provided via [ai.EmbedderOption] arguments. It's a convenient way to generate
// embeddings from registered embedders without directly calling the embedder instance.
//
// Example:
//
//	resp, err := genkit.Embed(ctx, g,
//		ai.WithEmbedder(ai.NewEmbedderRef("myEmbedder", nil)),
//		ai.WithTextDocs("Hello, world!"),
//	)
//	if err != nil {
//		log.Fatalf("Embed failed: %v", err)
//	}
//
//	for i, embedding := range resp.Embeddings {
//		fmt.Printf("Embedding %d: %v\n", i, embedding.Embedding)
//	}
func Embed(ctx context.Context, g *Genkit, opts ...ai.EmbedderOption) (*ai.EmbedResponse, error) {
	return ai.Embed(ctx, g.reg, opts...)
}

// DefineRetriever defines a custom retriever implementation, registers it as a
// [core.Action] of type Retriever, and returns an [ai.Retriever].
// Retrievers are used to find documents relevant to a given query, often by
// performing similarity searches in a vector database.
//
// The `provider` and `name` form the unique identifier. The `ret` function
// contains the logic to process an [ai.RetrieverRequest] (containing the query)
// and return an [ai.RetrieverResponse] (containing the relevant documents).
func DefineRetriever(g *Genkit, name string, opts *ai.RetrieverOptions, fn ai.RetrieverFunc) ai.Retriever {
	return ai.DefineRetriever(g.reg, name, opts, fn)
}

// LookupRetriever retrieves a registered [ai.Retriever] by its provider and name.
// It returns the retriever instance if found, or `nil` if no retriever with the
// given identifier is registered (e.g., via [DefineRetriever] or a plugin).
func LookupRetriever(g *Genkit, name string) ai.Retriever {
	return ai.LookupRetriever(g.reg, name)
}

// DefineEmbedder defines a custom text embedding implementation, registers it as a
// [core.Action] of type Embedder, and returns an [ai.Embedder].
// Embedders convert text documents or queries into numerical vector representations (embeddings).
//
// The `provider` and `name` are specified in the `opts` parameter which forms the unique identifier.
// The `embed` function contains the logic to process an [ai.EmbedRequest] (containing documents or a query)
// and return an [ai.EmbedResponse] (containing the corresponding embeddings).
func DefineEmbedder(g *Genkit, name string, opts *ai.EmbedderOptions, fn ai.EmbedderFunc) ai.Embedder {
	return ai.DefineEmbedder(g.reg, name, opts, fn)
}

// LookupEmbedder retrieves a registered [ai.Embedder] by its provider and name.
// It returns the embedder instance if found, or `nil` if no embedder with the
// given identifier is registered (e.g., via [DefineEmbedder] or a plugin).
// It will try to resolve the embedder dynamically if the embedder is not found.
func LookupEmbedder(g *Genkit, name string) ai.Embedder {
	return ai.LookupEmbedder(g.reg, name)
}

// LookupPlugin retrieves a registered plugin instance by its name.
// Plugins are registered during initialization via [WithPlugins].
// It returns the plugin instance as `Plugin` if found, or `nil` otherwise.
// The caller is responsible for type-asserting the returned value to the
// specific plugin api.
func LookupPlugin(g *Genkit, name string) api.Plugin {
	return g.reg.LookupPlugin(name)
}

// DefineEvaluator defines an evaluator that processes test cases one by one,
// registers it as a [core.Action] of type Evaluator, and returns an [ai.Evaluator].
// Evaluators are used to assess the quality or performance of AI models or flows
// based on a dataset of test cases.
//
// This variant calls the provided `eval` function for each individual test case
// ([ai.EvaluatorCallbackRequest]) in the evaluation dataset.
//
// The `provider` and `name` form the unique identifier. `options` provide
// metadata about the evaluator ([ai.EvaluatorOptions]). The `eval` function
// implements the logic to score a single test case and returns the results
// in an [ai.EvaluatorCallbackResponse].
func DefineEvaluator(g *Genkit, name string, opts *ai.EvaluatorOptions, fn ai.EvaluatorFunc) ai.Evaluator {
	return ai.DefineEvaluator(g.reg, name, opts, fn)
}

// DefineBatchEvaluator defines an evaluator that processes the entire dataset at once,
// registers it as a [core.Action] of type Evaluator, and returns an [ai.Evaluator].
//
// This variant provides the full evaluation request ([ai.EvaluatorRequest]), including
// the entire dataset, to the `eval` function. This allows for more flexible processing,
// such as batching calls to external services or parallelizing computations.
//
// The `provider` and `name` form the unique identifier. `options` provide
// metadata about the evaluator ([ai.EvaluatorOptions]). The `eval` function
// implements the logic to score the dataset and returns the aggregated results
// in an [ai.EvaluatorResponse].
func DefineBatchEvaluator(g *Genkit, name string, opts *ai.EvaluatorOptions, fn ai.BatchEvaluatorFunc) ai.Evaluator {
	return ai.DefineBatchEvaluator(g.reg, name, opts, fn)
}

// LookupEvaluator retrieves a registered [ai.Evaluator] by its provider and name.
// It returns the evaluator instance if found, or `nil` if no evaluator with the
// given identifier is registered (e.g., via [DefineEvaluator], [DefineBatchEvaluator],
// or a plugin).
func LookupEvaluator(g *Genkit, name string) ai.Evaluator {
	return ai.LookupEvaluator(g.reg, name)
}

// Evaluate performs an evaluation request using a flexible set of options
// provided via [ai.EvaluatorOption] arguments. It's a convenient way to run
// evaluations using registered evaluators without directly calling the
// evaluator instance.
//
// Example:
//
//	dataset := []*ai.Example{
//		{
//			Input: "What is the capital of France?",
//			Reference: "Paris",
//		},
//	}
//
//	resp, err := genkit.Evaluate(ctx, g,
//		ai.WithEvaluator(ai.NewEvaluatorRef("myEvaluator", nil)),
//		ai.WithDataset(dataset),
//	)
//	if err != nil {
//		log.Fatalf("Evaluate failed: %v", err)
//	}
//
//	for _, result := range *resp {
//		fmt.Printf("Evaluation result: %+v\n", result)
//	}
func Evaluate(ctx context.Context, g *Genkit, opts ...ai.EvaluatorOption) (*ai.EvaluatorResponse, error) {
	return ai.Evaluate(ctx, g.reg, opts...)
}

// LoadPromptDir loads all `.prompt` files from the specified directory `dir`
// into the registry, associating them with the given `namespace`.
// Files starting with `_` are treated as partials and are not registered as
// executable prompts but can be included in other prompts.
//
// If `dir` is empty, it defaults to "./prompts". If the directory doesn't exist,
// it logs a debug message (if using the default) or panics (if specified).
// The `namespace` acts as a prefix to the prompt name (e.g., namespace "myApp" and
// file "greeting.prompt" results in prompt name "myApp/greeting"). Use an empty
// string for no namespace.
//
// This function is often called implicitly by [Init] using the directory specified
// by [WithPromptDir], but can be called explicitly to load prompts from other
// locations or with different namespaces.
func LoadPromptDir(g *Genkit, dir string, namespace string) {
	ai.LoadPromptDir(g.reg, dir, namespace)
}

// LoadPrompt loads a single `.prompt` file specified by `path` into the registry,
// associating it with the given `namespace`, and returns the resulting [ai.prompt].
//
// The `path` should be the full path to the `.prompt` file.
// The `namespace` acts as a prefix to the prompt name (e.g., namespace "myApp" and
// path "/path/to/greeting.prompt" results in prompt name "myApp/greeting"). Use an
// empty string for no namespace.
//
// This provides a way to load specific prompt files programmatically, outside of the
// automatic loading done by [Init] or [LoadPromptDir].
//
// Example:
//
//	// Load a specific prompt file with a namespace
//	customPrompt := genkit.LoadPrompt(g, "./prompts/analyzer.prompt", "analysis")
//	if customPrompt == nil {
//		log.Fatal("Custom prompt not found or failed to parse.")
//	}
//
//	// Execute the loaded prompt
//	resp, err := customPrompt.Execute(ctx, ai.WithInput(map[string]any{"text": "some data"}))
//	// ... handle response and error ...
func LoadPrompt(g *Genkit, path string, namespace string) ai.Prompt {
	dir, filename := filepath.Split(path)
	if dir != "" {
		dir = filepath.Clean(dir)
	}

	return ai.LoadPrompt(g.reg, dir, filename, namespace)
}

// DefinePartial wraps DefinePartial to register a partial template with the given name and source.
// Partials can be referenced in templates with the syntax {{>partialName}}.
func DefinePartial(g *Genkit, name string, source string) {
	g.reg.RegisterPartial(name, source)
}

// DefineHelper wraps DefineHelper to register a helper function with the given name.
// This allows for extending the templating capabilities with custom logic.
//
// Example usage:
//
//	genkit.DefineHelper(g, "uppercase", func(s string) string {
//		return strings.ToUpper(s)
//	})
//
// In a template, you would use it as:
//
//	{{uppercase "hello"}} => "HELLO"
func DefineHelper(g *Genkit, name string, fn any) {
	g.reg.RegisterHelper(name, fn)
}

// DefineFormat defines a new [ai.Formatter] and registers it in the registry.
// Formatters control how model responses are structured and parsed.
//
// Formatters can be used with [ai.WithOutputFormat] to inject specific formatting
// instructions into prompts and automatically format the model response according
// to the desired output structure.
//
// Built-in formatters include:
//   - "text": Plain text output (default if no format specified)
//   - "json": Structured JSON output (default when an output schema is provided)
//   - "jsonl": JSON Lines format for streaming structured data
//
// Example:
//
//	// Define a custom formatter
//	type csvFormatter struct{}
//	func (f csvFormatter) Name() string { return "csv" }
//	func (f csvFormatter) Handler(schema map[string]any) (ai.FormatHandler, error) {
//		// Implementation details...
//	}
//
//	// Register the formatter
//	genkit.DefineFormat(g, "csv", csvFormatter{})
//
//	// Use the formatter in a generation request
//	resp, err := genkit.Generate(ctx, g,
//		ai.WithPrompt("List 3 countries and their capitals"),
//		ai.WithOutputFormat("csv"), // Use the custom formatter
//	)
func DefineFormat(g *Genkit, name string, formatter ai.Formatter) {
	ai.DefineFormat(g.reg, name, formatter)
}

// IsDefinedFormat checks if a formatter with the given name is registered in the registry.
func IsDefinedFormat(g *Genkit, name string) bool {
	return g.reg.LookupValue("/format/"+name) != nil
}

// DefineResource defines a resource and registers it with the Genkit instance.
// Resources provide content that can be referenced in prompts via URI.
//
// Example:
//
//	DefineResource(g, "company-docs", &ai.ResourceOptions{
//	  URI: "file:///docs/handbook.pdf",
//	  Description: "Company handbook",
//	}, func(ctx context.Context, input *ai.ResourceInput) (*ai.ResourceOutput, error) {
//	  content, err := os.ReadFile("/docs/handbook.pdf")
//	  if err != nil {
//	    return nil, err
//	  }
//	  return &ai.ResourceOutput{
//	    Content: []*ai.Part{ai.NewTextPart(string(content))},
//	  }, nil
//	})
func DefineResource(g *Genkit, name string, opts *ai.ResourceOptions, fn ai.ResourceFunc) ai.Resource {
	return ai.DefineResource(g.reg, name, opts, fn)
}

// FindMatchingResource finds a resource that matches the given URI.
func FindMatchingResource(g *Genkit, uri string) (ai.Resource, *ai.ResourceInput, error) {
	return ai.FindMatchingResource(g.reg, uri)
}

// NewResource creates an unregistered resource action that can be temporarily
// attached during generation via WithResources option.
//
// Example:
//
//	resource := NewResource("user-data", &ai.ResourceOptions{
//	  Template: "user://profile/{id}",
//	}, func(ctx context.Context, input *ai.ResourceInput) (*ai.ResourceOutput, error) {
//	  userID := input.Variables["id"]
//	  // Load user data dynamically...
//	  return &ai.ResourceOutput{Content: []*ai.Part{ai.NewTextPart(userData)}}, nil
//	})
//
//	// Use in generation:
//	ai.Generate(ctx, g,
//	  ai.WithPrompt([]*ai.Part{
//	    ai.NewTextPart("Analyze this user:"),
//	    ai.NewResourcePart("user://profile/123"),
//	  }),
//	  ai.WithResources(resource),
//	)
func NewResource(name string, opts *ai.ResourceOptions, fn ai.ResourceFunc) ai.Resource {
	return ai.NewResource(name, opts, fn)
}

// ListResources returns a slice of all resource actions
func ListResources(g *Genkit) []ai.Resource {
	acts := g.reg.ListActions()
	resources := []ai.Resource{}
	for _, action := range acts {
		actionDesc := action.Desc()
		if actionDesc.Type == api.ActionTypeResource {
			resource := ai.LookupResource(g.reg, actionDesc.Name)
			if resource != nil {
				resources = append(resources, resource)
			}
		}
	}
	return resources
}
