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
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/google/dotprompt/go/dotprompt"
	"github.com/invopop/jsonschema"
)

// Prompt is the interface for a prompt that can be executed and rendered.
type Prompt interface {
	// Name returns the name of the prompt.
	Name() string
	// Execute executes the prompt with the given options and returns a [ModelResponse].
	Execute(ctx context.Context, opts ...PromptExecuteOption) (*ModelResponse, error)
	// Render renders the prompt with the given input and returns a [GenerateActionOptions] to be used with [GenerateWithRequest].
	Render(ctx context.Context, input any) (*GenerateActionOptions, error)
}

// prompt is a prompt template that can be executed to generate a model response.
type prompt struct {
	core.ActionDef[any, *GenerateActionOptions, struct{}]
	promptOptions
	registry api.Registry
}

// DefinePrompt creates a new [Prompt] and registers it.
func DefinePrompt(r api.Registry, name string, opts ...PromptOption) Prompt {
	if name == "" {
		panic("ai.DefinePrompt: name is required")
	}

	pOpts := &promptOptions{}
	for _, opt := range opts {
		if err := opt.applyPrompt(pOpts); err != nil {
			panic(fmt.Errorf("ai.DefinePrompt: error applying options: %w", err))
		}
	}

	p := &prompt{
		registry:      r,
		promptOptions: *pOpts,
	}

	var modelName string
	if pOpts.Model != nil {
		modelName = pOpts.Model.Name()
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

	promptMeta := map[string]any{
		"type": api.ActionTypeExecutablePrompt,
		"prompt": map[string]any{
			"name":         name,
			"description":  p.Description,
			"model":        modelName,
			"config":       p.Config,
			"input":        map[string]any{"schema": p.InputSchema},
			"output":       map[string]any{"schema": p.OutputSchema},
			"defaultInput": p.DefaultInput,
			"tools":        tools,
			"maxTurns":     p.MaxTurns,
		},
	}
	maps.Copy(meta, promptMeta)

	p.ActionDef = *core.DefineAction(r, name, api.ActionTypeExecutablePrompt, meta, p.InputSchema, p.buildRequest)

	return p
}

// LookupPrompt looks up a [Prompt] registered by [DefinePrompt].
// It returns nil if the prompt was not defined.
func LookupPrompt(r api.Registry, name string) Prompt {
	action := core.LookupActionFor[any, *GenerateActionOptions, struct{}](r, api.ActionTypeExecutablePrompt, name)
	if action == nil {
		return nil
	}
	return &prompt{
		ActionDef: *action,
		registry:  r,
	}
}

// Execute renders a prompt, does variable substitution and
// passes the rendered template to the AI model specified by the prompt.
func (p *prompt) Execute(ctx context.Context, opts ...PromptExecuteOption) (*ModelResponse, error) {
	if p == nil {
		return nil, errors.New("Prompt.Execute: execute called on a nil Prompt; check that all prompts are defined")
	}

	execOpts := &promptExecutionOptions{}
	for _, opt := range opts {
		if err := opt.applyPromptExecute(execOpts); err != nil {
			return nil, fmt.Errorf("Prompt.Execute: error applying options: %w", err)
		}
	}
	// Render() should populate all data from the prompt. Prompt fields should
	// *not* be referenced in this function as it may have been loaded from
	// the registry and is missing the options passed in at definition.
	actionOpts, err := p.Render(ctx, execOpts.Input)
	if err != nil {
		return nil, err
	}

	if modelRef, ok := execOpts.Model.(ModelRef); ok && execOpts.Config == nil {
		execOpts.Config = modelRef.Config()
	}

	if execOpts.Config != nil {
		actionOpts.Config = execOpts.Config
	}

	if len(execOpts.Documents) > 0 {
		actionOpts.Docs = execOpts.Documents
	}

	if execOpts.ToolChoice != "" {
		actionOpts.ToolChoice = execOpts.ToolChoice
	}

	if execOpts.Model != nil {
		actionOpts.Model = execOpts.Model.Name()
	}

	if execOpts.MaxTurns != 0 {
		actionOpts.MaxTurns = execOpts.MaxTurns
	}

	if execOpts.ReturnToolRequests != nil {
		actionOpts.ReturnToolRequests = *execOpts.ReturnToolRequests
	}

	if execOpts.MessagesFn != nil {
		m, err := buildVariables(execOpts.Input)
		if err != nil {
			return nil, err
		}

		tempOpts := promptOptions{
			commonGenOptions: commonGenOptions{
				MessagesFn: execOpts.MessagesFn,
			},
		}

		actionOpts.Messages, err = renderMessages(ctx, tempOpts, actionOpts.Messages, m, execOpts.Input, p.registry.Dotprompt())
		if err != nil {
			return nil, err
		}
	}

	toolRefs := execOpts.Tools
	if len(toolRefs) == 0 {
		toolRefs = make([]ToolRef, 0, len(actionOpts.Tools))
		for _, toolName := range actionOpts.Tools {
			toolRefs = append(toolRefs, ToolName(toolName))
		}
	}

	toolNames, newTools, err := resolveUniqueTools(p.registry, toolRefs)
	if err != nil {
		return nil, err
	}
	actionOpts.Tools = toolNames

	r := p.registry
	if len(newTools) > 0 {
		if !r.IsChild() {
			r = r.NewChild()
		}
		for _, t := range newTools {
			t.Register(p.registry)
		}
	}

	return GenerateWithRequest(ctx, r, actionOpts, execOpts.Middleware, execOpts.Stream)
}

// Render renders the prompt template based on user input.
func (p *prompt) Render(ctx context.Context, input any) (*GenerateActionOptions, error) {
	if p == nil {
		return nil, errors.New("Prompt.Render: called on a nil prompt; check that all prompts are defined")
	}

	if len(p.Middleware) > 0 {
		logger.FromContext(ctx).Warn(fmt.Sprintf("middleware set on prompt %q will be ignored during Prompt.Render", p.Name()))
	}

	// TODO: This is hacky; we should have a helper that fetches the metadata.
	if input == nil {
		input = p.Desc().Metadata["prompt"].(map[string]any)["defaultInput"]
	}

	return p.Run(ctx, input, nil)
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
// using the input variables and other information in the [prompt].
func (p *prompt) buildRequest(ctx context.Context, input any) (*GenerateActionOptions, error) {
	m, err := buildVariables(input)
	if err != nil {
		return nil, err
	}

	messages := []*Message{}
	messages, err = renderSystemPrompt(ctx, p.promptOptions, messages, m, input, p.registry.Dotprompt())
	if err != nil {
		return nil, err
	}
	messages, err = renderMessages(ctx, p.promptOptions, messages, m, input, p.registry.Dotprompt())
	if err != nil {
		return nil, err
	}
	messages, err = renderUserPrompt(ctx, p.promptOptions, messages, m, input, p.registry.Dotprompt())
	if err != nil {
		return nil, err
	}

	var tools []string
	for _, t := range p.Tools {
		tools = append(tools, t.Name())
	}

	config := p.Config
	if modelRef, ok := p.Model.(ModelRef); ok && config == nil {
		config = modelRef.Config()
	}

	var modelName string
	if p.Model != nil {
		modelName = p.Model.Name()
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

	convertedParts := []*Part{}
	for _, message := range rendered.Messages {
		parts, err := convertToPartPointers(message.Content)
		if err != nil {
			return nil, fmt.Errorf("failed to convert parts: %w", err)
		}
		convertedParts = append(convertedParts, parts...)
	}

	return convertedParts, nil
}

// convertToPartPointers converts []dotprompt.Part to []*Part
func convertToPartPointers(parts []dotprompt.Part) ([]*Part, error) {
	result := make([]*Part, len(parts))
	for i, part := range parts {
		switch p := part.(type) {
		case *dotprompt.TextPart:
			if p.Text != "" {
				result[i] = NewTextPart(p.Text)
			}
		case *dotprompt.MediaPart:
			ct, err := contentType(p.Media.URL)
			if err != nil {
				return nil, err
			}
			result[i] = NewMediaPart(ct, p.Media.URL)
		}
	}
	return result, nil
}

// LoadPromptDir loads prompts and partials from the input directory for the given namespace.
func LoadPromptDir(r api.Registry, dir string, namespace string) {
	useDefaultDir := false
	if dir == "" {
		dir = "./prompts"
		useDefaultDir = true
	}

	path, err := filepath.Abs(dir)
	if err != nil {
		if !useDefaultDir {
			panic(fmt.Errorf("failed to resolve prompt directory %q: %w", dir, err))
		}
		slog.Debug("default prompt directory not found, skipping loading .prompt files", "dir", dir)
		return
	}

	if _, err := os.Stat(path); os.IsNotExist(err) {
		if !useDefaultDir {
			panic(fmt.Errorf("failed to resolve prompt directory %q: %w", dir, err))
		}
		slog.Debug("Default prompt directory not found, skipping loading .prompt files", "dir", dir)
		return
	}

	loadPromptDir(r, path, namespace)
}

// loadPromptDir recursively loads prompts and partials from the directory.
func loadPromptDir(r api.Registry, dir string, namespace string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		panic(fmt.Errorf("failed to read prompt directory structure: %w", err))
	}

	for _, entry := range entries {
		filename := entry.Name()
		path := filepath.Join(dir, filename)
		if entry.IsDir() {
			loadPromptDir(r, path, namespace)
		} else if strings.HasSuffix(filename, ".prompt") {
			if strings.HasPrefix(filename, "_") {
				partialName := strings.TrimSuffix(filename[1:], ".prompt")
				source, err := os.ReadFile(path)
				if err != nil {
					slog.Error("Failed to read partial file", "error", err)
					continue
				}
				r.RegisterPartial(partialName, string(source))
				slog.Debug("Registered Dotprompt partial", "name", partialName, "file", path)
			} else {
				LoadPrompt(r, dir, filename, namespace)
			}
		}
	}
}

// LoadPrompt loads a single prompt into the registry.
func LoadPrompt(r api.Registry, dir, filename, namespace string) Prompt {
	name := strings.TrimSuffix(filename, ".prompt")
	name, variant, _ := strings.Cut(name, ".")

	sourceFile := filepath.Join(dir, filename)
	source, err := os.ReadFile(sourceFile)
	if err != nil {
		slog.Error("Failed to read prompt file", "file", sourceFile, "error", err)
		return nil
	}

	parsedPrompt, err := r.Dotprompt().Parse(string(source))
	if err != nil {
		slog.Error("Failed to parse file as dotprompt", "file", sourceFile, "error", err)
		return nil
	}

	metadata, err := r.Dotprompt().RenderMetadata(string(source), &parsedPrompt.PromptMetadata)
	if err != nil {
		slog.Error("Failed to render dotprompt metadata", "file", sourceFile, "error", err)
		return nil
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
			Model: NewModelRef(metadata.Model, nil),
			Tools: toolRefs,
		},
		DefaultInput: metadata.Input.Default,
		Metadata:     promptOptMetadata,
		Description:  metadata.Description,
	}

	if toolChoice, ok := metadata.Raw["toolChoice"].(ToolChoice); ok {
		opts.ToolChoice = toolChoice
	}

	if maxTurns, ok := metadata.Raw["maxTurns"].(int); ok {
		opts.MaxTurns = maxTurns
	}

	if returnToolRequests, ok := metadata.Raw["returnToolRequests"].(bool); ok {
		opts.ReturnToolRequests = &returnToolRequests
	}

	if inputSchema, ok := metadata.Input.Schema.(*jsonschema.Schema); ok {
		opts.InputSchema = base.SchemaAsMap(inputSchema)
	}

	if inputSchema, ok := metadata.Input.Schema.(map[string]any); ok {
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
	prompt := DefinePrompt(r, key, opts, WithPrompt(parsedPrompt.Template))

	slog.Debug("Registered Dotprompt", "name", key, "file", sourceFile)

	return prompt
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

// contentType determines the MIME content type of the given data URI
func contentType(uri string) (string, error) {
	if uri == "" {
		return "", errors.New("found empty URI in part")
	}

	if strings.HasPrefix(uri, "gs://") || strings.HasPrefix(uri, "http") {
		return "", errors.New("data URI is the only media type supported")
	}
	if contents, isData := strings.CutPrefix(uri, "data:"); isData {
		prefix, _, found := strings.Cut(contents, ",")
		if !found {
			return "", errors.New("failed to parse data URI: missing comma")
		}

		if p, isBase64 := strings.CutSuffix(prefix, ";base64"); isBase64 {
			return p, nil
		}
	}

	return "", errors.New("uri content type not found")
}
