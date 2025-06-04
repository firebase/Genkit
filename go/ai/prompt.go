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

package ai

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/internal/registry"
	"github.com/google/dotprompt/go/dotprompt"
	"github.com/invopop/jsonschema"
)

// Prompt is a prompt template that can be executed to generate a model response.
type Prompt struct {
	promptOptions
	registry *registry.Registry
	action   core.ActionDef[any, *GenerateActionOptions, struct{}]
}

// DefinePrompt creates and registers a new Prompt.
func DefinePrompt(r *registry.Registry, name string, opts ...PromptOption) (*Prompt, error) {
	pOpts := &promptOptions{}
	for _, opt := range opts {
		if err := opt.applyPrompt(pOpts); err != nil {
			return nil, fmt.Errorf("ai.DefinePrompt: error applying options: %w", err)
		}
	}

	p := &Prompt{
		registry:      r,
		promptOptions: *pOpts,
	}

	modelName := p.ModelName
	if modelName == "" && p.Model != nil {
		modelName = p.Model.Name()
	}

	if modelRef, ok := pOpts.Model.(ModelRef); ok && pOpts.Config == nil {
		pOpts.Config = modelRef.Config()
	}

	meta := p.Metadata
	if meta == nil {
		meta = map[string]any{}
	}

	var tools []string
	for _, value := range pOpts.commonGenOptions.Tools {
		tools = append(tools, value.Name())
	}

	var inputSchema map[string]any
	if p.InputSchema != nil {
		inputSchema = base.SchemaAsMap(p.InputSchema)
	}

	promptMeta := map[string]any{
		"prompt": map[string]any{
			"name":         name,
			"description":  p.Description,
			"model":        modelName,
			"config":       p.Config,
			"input":        map[string]any{"schema": inputSchema},
			"output":       map[string]any{"schema": p.OutputSchema},
			"defaultInput": p.DefaultInput,
			"tools":        tools,
		},
	}
	maps.Copy(meta, promptMeta)

	p.action = *core.DefineActionWithInputSchema(r, "", name, core.ActionTypeExecutablePrompt, meta, p.InputSchema, p.buildRequest)

	return p, nil
}

// LookupPrompt looks up a [Prompt] registered by [DefinePrompt].
// It returns nil if the prompt was not defined.
func LookupPrompt(r *registry.Registry, name string) *Prompt {
	action := core.LookupActionFor[any, *GenerateActionOptions, struct{}](r, core.ActionTypeExecutablePrompt, "", name)
	if action == nil {
		return nil
	}
	return &Prompt{
		registry: r,
		action:   *action,
	}
}

// Name returns the name of the prompt.
func (p *Prompt) Name() string { return p.action.Name() }

// Execute renders a prompt, does variable substitution and
// passes the rendered template to the AI model specified by the prompt.
func (p *Prompt) Execute(ctx context.Context, opts ...PromptExecuteOption) (*ModelResponse, error) {
	if p == nil {
		return nil, errors.New("Prompt.Execute: execute called on a nil Prompt; check that all prompts are defined")
	}

	genOpts := &promptExecutionOptions{}
	for _, opt := range opts {
		if err := opt.applyPromptExecute(genOpts); err != nil {
			return nil, fmt.Errorf("Prompt.Execute: error applying options: %w", err)
		}
	}

	p.MessagesFn = mergeMessagesFn(p.MessagesFn, genOpts.MessagesFn)

	// Render() should populate all data from the prompt. Prompt fields should
	// *not* be referenced in this function as it may have been loaded from
	// the registry and is missing the options passed in at definition.
	actionOpts, err := p.Render(ctx, genOpts.Input)
	if err != nil {
		return nil, err
	}

	if modelRef, ok := genOpts.Model.(ModelRef); ok && genOpts.Config == nil {
		genOpts.Config = modelRef.Config()
	}
	if genOpts.Config != nil {
		actionOpts.Config = genOpts.Config
	}
	if len(genOpts.Documents) > 0 {
		actionOpts.Docs = genOpts.Documents
	}
	if genOpts.ToolChoice != "" {
		actionOpts.ToolChoice = genOpts.ToolChoice
	}

	modelName := genOpts.ModelName
	if modelName == "" && genOpts.Model != nil {
		modelName = genOpts.Model.Name()
	}
	if modelName != "" {
		actionOpts.Model = modelName
	}

	if genOpts.MaxTurns != 0 {
		actionOpts.MaxTurns = genOpts.MaxTurns
	}

	if genOpts.ReturnToolRequests != nil {
		actionOpts.ReturnToolRequests = *genOpts.ReturnToolRequests
	}

	if genOpts.Session != nil {
		// Set session details in context
		ctx = genOpts.Session.SetContext(ctx)
	}

	return GenerateWithRequest(ctx, p.registry, actionOpts, genOpts.Middleware, genOpts.Stream)
}

// Render renders the prompt template based on user input.
func (p *Prompt) Render(ctx context.Context, input any) (*GenerateActionOptions, error) {
	if p == nil {
		return nil, errors.New("Prompt.Render: called on a nil prompt; check that all prompts are defined")
	}

	if len(p.Middleware) > 0 {
		logger.FromContext(ctx).Warn(fmt.Sprintf("middleware set on prompt %q will be ignored during Prompt.Render", p.Name()))
	}

	// TODO: This is hacky; we should have a helper that fetches the metadata.
	if input == nil {
		input = p.action.Desc().Metadata["prompt"].(map[string]any)["defaultInput"]
	}

	return p.action.Run(ctx, input, nil)
}

// mergeMessagesFn merges two messages functions.
func mergeMessagesFn(promptFn, reqFn MessagesFn) MessagesFn {
	if reqFn == nil {
		return promptFn
	}

	if promptFn == nil {
		return reqFn
	}

	return func(ctx context.Context, input any) ([]*Message, error) {
		promptMsgs, err := promptFn(ctx, input)
		if err != nil {
			return nil, err
		}

		reqMsgs, err := reqFn(ctx, input)
		if err != nil {
			return nil, err
		}

		return append(promptMsgs, reqMsgs...), nil
	}
}

// buildVariables returns a map holding prompt field values based
// on a struct or a pointer to a struct. The struct value should have
// JSON tags that correspond to the Prompt's input schema.
// Only exported fields of the struct will be used.
func buildVariables(variables any) (map[string]any, error) {
	if variables == nil {
		return nil, nil
	}

	v := reflect.Indirect(reflect.ValueOf(variables))
	if v.Kind() == reflect.Map {
		return variables.(map[string]any), nil
	}
	if v.Kind() != reflect.Struct {
		return nil, errors.New("prompt.buildVariables: fields not a struct or pointer to a struct or a map")
	}
	vt := v.Type()

	// TODO: Verify the struct with p.Config.InputSchema.

	m := make(map[string]any)

fieldLoop:
	for i := range vt.NumField() {
		ft := vt.Field(i)
		if ft.PkgPath != "" {
			continue
		}

		jsonTag := ft.Tag.Get("json")
		jsonName, rest, _ := strings.Cut(jsonTag, ",")
		if jsonName == "" {
			jsonName = ft.Name
		}

		vf := v.Field(i)

		// If the field is the zero value, and omitempty is set,
		// don't pass it as a prompt input variable.
		if vf.IsZero() {
			for rest != "" {
				var key string
				key, rest, _ = strings.Cut(rest, ",")
				if key == "omitempty" {
					continue fieldLoop
				}
			}
		}

		m[jsonName] = vf.Interface()
	}

	return m, nil
}

// buildRequest prepares a [GenerateActionOptions] based on the prompt,
// using the input variables and other information in the [Prompt].
func (p *Prompt) buildRequest(ctx context.Context, input any) (*GenerateActionOptions, error) {
	m, err := buildVariables(input)
	if err != nil {
		return nil, err
	}

	messages := []*Message{}
	messages, err = renderSystemPrompt(ctx, p.promptOptions, messages, m, input, p.registry.Dotprompt)
	if err != nil {
		return nil, err
	}
	messages, err = renderMessages(ctx, p.promptOptions, messages, m, input, p.registry.Dotprompt)
	if err != nil {
		return nil, err
	}
	messages, err = renderUserPrompt(ctx, p.promptOptions, messages, m, input, p.registry.Dotprompt)
	if err != nil {
		return nil, err
	}

	var tools []string
	for _, t := range p.Tools {
		tools = append(tools, t.Name())
	}

	modelName := p.ModelName
	if modelName == "" && p.Model != nil {
		modelName = p.Model.Name()
	}

	config := p.Config
	if modelRef, ok := p.Model.(ModelRef); ok && config == nil {
		config = modelRef.Config()
	}

	return &GenerateActionOptions{
		Model:              modelName,
		Config:             config,
		ToolChoice:         p.ToolChoice,
		MaxTurns:           p.MaxTurns,
		ReturnToolRequests: p.ReturnToolRequests != nil && *p.ReturnToolRequests,
		Messages:           messages,
		Tools:              tools,
		Output: &GenerateActionOutputConfig{
			Format:       p.OutputFormat,
			JsonSchema:   p.OutputSchema,
			Instructions: p.OutputInstructions,
			Constrained:  !p.CustomConstrained,
		},
	}, nil
}

// renderSystemPrompt renders a system prompt message.
func renderSystemPrompt(ctx context.Context, opts promptOptions, messages []*Message, input map[string]any, raw any, dp *dotprompt.Dotprompt) ([]*Message, error) {
	if opts.SystemFn == nil {
		return messages, nil
	}

	templateText, err := opts.SystemFn(ctx, raw)
	if err != nil {
		return nil, err
	}

	parts, err := renderPrompt(ctx, opts, templateText, input, dp)
	if err != nil {
		return nil, err
	}

	if len(parts) != 0 {
		messages = append(messages, NewSystemMessage(parts...))
	}

	return messages, nil
}

// renderUserPrompt renders a user prompt message.
func renderUserPrompt(ctx context.Context, opts promptOptions, messages []*Message, input map[string]any, raw any, dp *dotprompt.Dotprompt) ([]*Message, error) {
	if opts.PromptFn == nil {
		return messages, nil
	}

	templateText, err := opts.PromptFn(ctx, raw)
	if err != nil {
		return nil, err
	}

	parts, err := renderPrompt(ctx, opts, templateText, input, dp)
	if err != nil {
		return nil, err
	}

	if len(parts) != 0 {
		messages = append(messages, NewUserMessage(parts...))
	}

	return messages, nil
}

// renderMessages renders a slice of messages.
func renderMessages(ctx context.Context, opts promptOptions, messages []*Message, input map[string]any, raw any, dp *dotprompt.Dotprompt) ([]*Message, error) {
	if opts.MessagesFn == nil {
		return messages, nil
	}

	msgs, err := opts.MessagesFn(ctx, raw)
	if err != nil {
		return nil, err
	}

	for _, msg := range msgs {
		msgParts := []*Part{}
		for _, part := range msg.Content {
			if part.IsText() {
				parts, err := renderPrompt(ctx, opts, part.Text, input, dp)
				if err != nil {
					return nil, err
				}
				msgParts = append(msgParts, parts...)
			}
		}
		msg.Content = msgParts
	}

	return append(messages, msgs...), nil
}

// renderPrompt renders a prompt template using dotprompt functionalities
func renderPrompt(ctx context.Context, opts promptOptions, templateText string, input map[string]any, dp *dotprompt.Dotprompt) ([]*Part, error) {
	renderedFunc, err := dp.Compile(templateText, &dotprompt.PromptMetadata{})
	if err != nil {
		return nil, err
	}

	return renderDotpromptToParts(ctx, renderedFunc, input, &dotprompt.PromptMetadata{
		Input: dotprompt.PromptMetadataInput{
			Default: opts.DefaultInput,
		},
	})
}

// renderDotpromptToParts executes a dotprompt prompt function and converts the result to a slice of parts
func renderDotpromptToParts(ctx context.Context, promptFn dotprompt.PromptFunction, input map[string]any, additionalMetadata *dotprompt.PromptMetadata) ([]*Part, error) {
	// Prepare the context for rendering
	context := map[string]any{}
	actionCtx := core.FromContext(ctx)
	maps.Copy(context, actionCtx)

	// Call the prompt function with the input and context
	rendered, err := promptFn(&dotprompt.DataArgument{
		Input:   input,
		Context: context,
	}, additionalMetadata)
	if err != nil {
		return nil, fmt.Errorf("failed to render prompt: %w", err)
	}

	// Ensure the rendered prompt contains exactly one message
	if len(rendered.Messages) != 1 {
		return nil, fmt.Errorf("parts template must produce only one message")
	}

	// Convert dotprompt.Part to Part
	convertedParts, err := convertToPartPointers(rendered.Messages[0].Content)
	if err != nil {
		return nil, fmt.Errorf("failed to convert parts: %w", err)
	}

	return convertedParts, nil
}

// convertToPartPointers converts []dotprompt.Part to []*Part
func convertToPartPointers(parts []dotprompt.Part) ([]*Part, error) {
	result := make([]*Part, len(parts))
	for i, part := range parts {
		if p, ok := part.(*dotprompt.TextPart); ok {
			if p.Text != "" {
				result[i] = NewTextPart(p.Text)
			}
		} else {
			return nil, fmt.Errorf("unsupported prompt format: %T", part)
		}
	}
	return result, nil
}

// LoadPromptDir loads prompts and partials from the input directory for the given namespace.
func LoadPromptDir(r *registry.Registry, dir string, namespace string) error {
	useDefaultDir := false
	if dir == "" {
		dir = "./prompts"
		useDefaultDir = true
	}

	path, err := filepath.Abs(dir)
	if err != nil {
		if !useDefaultDir {
			return fmt.Errorf("failed to resolve prompt directory %q: %w", dir, err)
		}
		slog.Debug("default prompt directory not found, skipping loading .prompt files", "dir", dir)
		return nil
	}

	if _, err := os.Stat(path); os.IsNotExist(err) {
		if !useDefaultDir {
			return fmt.Errorf("failed to resolve prompt directory %q: %w", dir, err)
		}
		slog.Debug("Default prompt directory not found, skipping loading .prompt files", "dir", dir)
		return nil
	}

	return loadPromptDir(r, path, namespace)
}

// loadPromptDir recursively loads prompts and partials from the directory.
func loadPromptDir(r *registry.Registry, dir string, namespace string) error {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return fmt.Errorf("failed to read prompt directory structure: %w", err)
	}

	for _, entry := range entries {
		filename := entry.Name()
		path := filepath.Join(dir, filename)
		if entry.IsDir() {
			if err := loadPromptDir(r, path, namespace); err != nil {
				return err
			}
		} else if strings.HasSuffix(filename, ".prompt") {
			if strings.HasPrefix(filename, "_") {
				partialName := strings.TrimSuffix(filename[1:], ".prompt")
				source, err := os.ReadFile(path)
				if err != nil {
					slog.Error("Failed to read partial file", "error", err)
					continue
				}
				if err = r.DefinePartial(partialName, string(source)); err != nil {
					return err
				}
				slog.Debug("Registered Dotprompt partial", "name", partialName, "file", path)
			} else {
				if _, err := LoadPrompt(r, dir, filename, namespace); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// LoadPrompt loads a single prompt into the registry.
func LoadPrompt(r *registry.Registry, dir, filename, namespace string) (*Prompt, error) {
	name := strings.TrimSuffix(filename, ".prompt")
	name, variant, _ := strings.Cut(name, ".")

	sourceFile := filepath.Join(dir, filename)
	source, err := os.ReadFile(sourceFile)
	if err != nil {
		slog.Error("Failed to read prompt file", "file", sourceFile, "error", err)
		return nil, nil
	}

	parsedPrompt, err := r.Dotprompt.Parse(string(source))
	if err != nil {
		slog.Error("Failed to parse file as dotprompt", "file", sourceFile, "error", err)
		return nil, nil
	}

	metadata, err := r.Dotprompt.RenderMetadata(string(source), &parsedPrompt.PromptMetadata)
	if err != nil {
		slog.Error("Failed to render dotprompt metadata", "file", sourceFile, "error", err)
		return nil, nil
	}

	toolRefs := make([]ToolRef, len(metadata.Tools))
	for i, tool := range metadata.Tools {
		toolRefs[i] = ToolName(tool)
	}

	promptMetadata := map[string]any{
		"template": parsedPrompt.Template,
	}
	maps.Copy(promptMetadata, metadata.Metadata)

	promptOptMetadata := map[string]any{
		"type":   "prompt",
		"prompt": promptMetadata,
	}
	maps.Copy(promptOptMetadata, metadata.Metadata)

	opts := &promptOptions{
		commonGenOptions: commonGenOptions{
			configOptions: configOptions{
				Config: (map[string]any)(metadata.Config),
			},
			ModelName: metadata.Model,
			Tools:     toolRefs,
		},
		DefaultInput: metadata.Input.Default,
		Metadata:     promptOptMetadata,
		Description:  metadata.Description,
	}

	if toolChoice, ok := metadata.Raw["toolChoice"].(ToolChoice); ok {
		opts.ToolChoice = toolChoice
	}

	if maxTurns, ok := metadata.Raw["maxTurns"].(int); !ok {
		opts.MaxTurns = maxTurns
	}

	if returnToolRequests, ok := metadata.Raw["returnToolRequests"].(bool); !ok {
		opts.ReturnToolRequests = &returnToolRequests
	}

	if inputSchema, ok := metadata.Input.Schema.(*jsonschema.Schema); ok {
		opts.InputSchema = inputSchema
	}

	if metadata.Output.Format != "" {
		opts.OutputFormat = metadata.Output.Format
	}

	if outputSchema, ok := metadata.Output.Schema.(*jsonschema.Schema); ok {
		opts.OutputSchema = base.SchemaAsMap(outputSchema)
		if opts.OutputFormat == "" {
			opts.OutputFormat = OutputFormatJSON
		}
	}

	key := promptKey(name, variant, namespace)
	prompt, err := DefinePrompt(r, key, opts, WithPrompt(parsedPrompt.Template))
	if err != nil {
		slog.Error("Failed to register dotprompt", "file", sourceFile, "error", err)
		return nil, err
	}

	slog.Debug("Registered Dotprompt", "name", key, "file", sourceFile)

	return prompt, nil
}

// promptKey generates a unique key for the prompt in the registry.
func promptKey(name string, variant string, namespace string) string {
	if namespace != "" {
		return fmt.Sprintf("%s/%s%s", namespace, name, variantKey(variant))
	}
	return fmt.Sprintf("%s%s", name, variantKey(variant))
}

// variantKey formats the variant part of the key.
func variantKey(variant string) string {
	if variant != "" {
		return fmt.Sprintf(".%s", variant)
	}
	return ""
}
