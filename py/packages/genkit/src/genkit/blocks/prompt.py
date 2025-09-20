# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


"""Prompt management and templating for the Genkit framework.

This module provides types and utilities for managing prompts and templates
used with AI models in the Genkit framework. It enables consistent prompt
generation and management across different parts of the application.
"""
import asyncio
from asyncio import Future
from collections.abc import AsyncIterator
from typing import Any, Awaitable, Protocol, runtime_checkable

from dotpromptz.typing import PromptFunction,  DataArgument, PromptMetadata, PromptInputConfig
from pydantic import BaseModel, ConfigDict

from genkit.aio import Channel
from genkit.blocks.generate import (
    StreamingCallback as ModelStreamingCallback,
    generate_action,
)
from genkit.blocks.model import (
    GenerateResponseChunkWrapper,
    GenerateResponseWrapper,
    ModelMiddleware,
)
from genkit.core.action import ActionRunContext
from genkit.core.registry import Registry
from genkit.core.schema import to_json_schema
from genkit.core.typing import (
    GenerateActionOptions,
    GenerateActionOutputConfig,
    GenerationCommonConfig,
    Message,
    Part,
    Resume,
    Role,
    ToolChoice, DocumentData
)


class PromptCache:
    """Model for a prompt cache."""

    user_prompt: PromptFunction | None = None
    system: PromptFunction | None = None
    messages: PromptFunction | None = None




class PromptConfig(BaseModel):
    """Model for a prompt action."""
    variant: str | None = None
    model: str | None = None
    config: GenerationCommonConfig | dict[str, Any] | None = None
    description: str | None = None
    input_schema: type | dict[str, Any] | None = None
    system: str | Part | list[Part] | None = None
    prompt: str | Part | list[Part] | None = None
    messages: str | list[Message] | None = None
    output_format: str | None = None
    output_content_type: str | None = None
    output_instructions: bool | str | None = None
    output_schema: type | dict[str, Any] | None = None
    output_constrained: bool | None = None
    max_turns: int | None = None
    return_tool_requests: bool | None = None
    metadata: dict[str, Any] | None = None
    tools: list[str] | None = None
    tool_choice: ToolChoice | None = None
    use: list[ModelMiddleware] | None = None
    docs: list[DocumentData] | None = None
    tool_responses: list[Part] | None = None

class ExecutablePrompt:
    """A prompt that can be executed with a given input and configuration."""

    def __init__(
        self,
        registry: Registry,
        variant: str | None = None,
        model: str | None = None,
        config: GenerationCommonConfig | dict[str, Any] | None = None,
        description: str | None = None,
        input_schema: type | dict[str, Any] | None = None,
        system: str | Part | list[Part] | None = None,
        prompt: str | Part | list[Part] | None = None,
        messages: str | list[Message] | None = None,
        output_format: str | None = None,
        output_content_type: str | None = None,
        output_instructions: bool | str | None = None,
        output_schema: type | dict[str, Any] | None = None,
        output_constrained: bool | None = None,
        max_turns: int | None = None,
        return_tool_requests: bool | None = None,
        metadata: dict[str, Any] | None = None,
        tools: list[str] | None = None,
        tool_choice: ToolChoice | None = None,
        use: list[ModelMiddleware] | None = None,
        # TODO:
        #  docs: list[Document]):
    ):
        """Initializes an ExecutablePrompt instance.

        Args:
            registry: The registry to use for resolving models and tools.
            variant: The variant of the prompt.
            model: The model to use for generation.
            config: The generation configuration.
            description: A description of the prompt.
            input_schema: The input schema for the prompt.
            system: The system message for the prompt.
            prompt: The user prompt.
            messages: A list of messages to include in the prompt.
            output_format: The output format.
            output_content_type: The output content type.
            output_instructions: Instructions for formatting the output.
            output_schema: The output schema.
            output_constrained: Whether the output should be constrained to the output schema.
            max_turns: The maximum number of turns in a conversation.
            return_tool_requests: Whether to return tool requests.
            metadata: Metadata to associate with the prompt.
            tools: A list of tool names to use with the prompt.
            tool_choice: The tool choice strategy.
            use: A list of model middlewares to apply.
        """
        self._registry = registry
        self._variant = variant
        self._model = model
        self._config = config
        self._description = description
        self._input_schema = input_schema
        self._system = system
        self._prompt = prompt
        self._messages = messages
        self._output_format = output_format
        self._output_content_type = output_content_type
        self._output_instructions = output_instructions
        self._output_schema = output_schema
        self._output_constrained = output_constrained
        self._max_turns = max_turns
        self._return_tool_requests = return_tool_requests
        self._metadata = metadata
        self._tools = tools
        self._tool_choice = tool_choice
        self._use = use
        self._cache_prompt = PromptCache()

    async def __call__(
        self,
        input: Any | None = None,
        config: GenerationCommonConfig | dict[str, Any] | None = None,
        on_chunk: ModelStreamingCallback | None = None,
        context: dict[str, Any] | None = None,
    ) -> GenerateResponseWrapper:
        """Executes the prompt with the given input and configuration.

        Args:
            input: The input to the prompt.
            config: The generation configuration.
            on_chunk: A callback function to be called for each chunk of the
                response.
            context: The action run context.

        Returns:
            The generated response.
        """
        return await generate_action(
            self._registry,
            await self.render(input=input, config=config),
            on_chunk=on_chunk,
            middleware=self._use,
            context=context if context else ActionRunContext._current_context(),
        )

    def stream(
        self,
        input: Any | None = None,
        config: GenerationCommonConfig | dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> tuple[
        AsyncIterator[GenerateResponseChunkWrapper],
        Future[GenerateResponseWrapper],
    ]:
        """Streams the prompt with the given input and configuration.

        Args:
            input: The input to the prompt.
            config: The generation configuration.
            context: The action run context.
            timeout: The timeout for the streaming action.

        Returns:
            A tuple containing an async iterator of response chunks and a future
            that resolves to the complete response.
        """
        stream = Channel(timeout=timeout)

        resp = self.__call__(
            input=input,
            config=config,
            context=context,
            on_chunk=lambda c: stream.send(c),
        )
        stream.set_close_future(resp)

        return (stream, stream.closed)

    async def render(
        self,
        input: dict[str, Any] | None = None,
        config: GenerationCommonConfig | dict[str, Any] | None = None,
    ) -> GenerateActionOptions:
        """Renders the prompt with the given input and configuration.

        Args:
            input: The input to the prompt.
            config: The generation configuration.

        Returns:
            The rendered prompt as a GenerateActionOptions object.
        """
        # TODO: run str prompt/system/message through dotprompt using input
        return await to_generate_action_options(
            self._registry,
            PromptConfig(
                model=self._model,
                prompt=self._prompt,
                system=self._system,
                messages=self._messages,
                tools=self._tools,
                return_tool_requests=self._return_tool_requests,
                tool_choice=self._tool_choice,
                config=config if config else self._config,
                max_turns=self._max_turns,
                output_format=self._output_format,
                output_content_type=self._output_content_type,
                output_instructions=self._output_instructions,
                output_schema=self._output_schema,
                output_constrained=self._output_constrained,
                input_schema=input if input else self._input_schema,
            ),
            self._cache_prompt
        )


def define_prompt(
    registry: Registry,
    variant: str | None = None,
    model: str | None = None,
    config: GenerationCommonConfig | dict[str, Any] | None = None,
    description: str | None = None,
    input_schema: type | dict[str, Any] | None = None,
    system: str | Part | list[Part] | None = None,
    prompt: str | Part | list[Part] | None = None,
    messages: str | list[Message] | None = None,
    output_format: str | None = None,
    output_content_type: str | None = None,
    output_instructions: bool | str | None = None,
    output_schema: type | dict[str, Any] | None = None,
    output_constrained: bool | None = None,
    max_turns: int | None = None,
    return_tool_requests: bool | None = None,
    metadata: dict[str, Any] | None = None,
    tools: list[str] | None = None,
    tool_choice: ToolChoice | None = None,
    use: list[ModelMiddleware] | None = None,
    # TODO:
    #  docs: list[Document]
) -> ExecutablePrompt:
    """Defines an executable prompt.

    Args:
        registry: The registry to use for resolving models and tools.
        variant: The variant of the prompt.
        model: The model to use for generation.
        config: The generation configuration.
        description: A description of the prompt.
        input_schema: The input schema for the prompt.
        system: The system message for the prompt.
        prompt: The user prompt.
        messages: A list of messages to include in the prompt.
        output_format: The output format.
        output_content_type: The output content type.
        output_instructions: Instructions for formatting the output.
        output_schema: The output schema.
        output_constrained: Whether the output should be constrained to the output schema.
        max_turns: The maximum number of turns in a conversation.
        return_tool_requests: Whether to return tool requests.
        metadata: Metadata to associate with the prompt.
        tools: A list of tool names to use with the prompt.
        tool_choice: The tool choice strategy.
        use: A list of model middlewares to apply.

    Returns:
        An ExecutablePrompt instance.
    """
    return ExecutablePrompt(
        registry,
        variant=variant,
        model=model,
        config=config,
        description=description,
        input_schema=input_schema,
        system=system,
        prompt=prompt,
        messages=messages,
        output_format=output_format,
        output_content_type=output_content_type,
        output_instructions=output_instructions,
        output_schema=output_schema,
        output_constrained=output_constrained,
        max_turns=max_turns,
        return_tool_requests=return_tool_requests,
        metadata=metadata,
        tools=tools,
        tool_choice=tool_choice,
        use=use,
    )


async def to_generate_action_options(
    registry: Registry,
    options: PromptConfig,
    prompt_cache: PromptCache | None = None,
) -> GenerateActionOptions:
    """Converts the given parameters to a GenerateActionOptions object.

    Args:
        registry: The registry to use for resolving models and tools.
        model: The model to use for generation.
        prompt: The user prompt.
        system: The system message for the prompt.
        messages: A list of messages to include in the prompt.
        tools: A list of tool names to use with the prompt.
        return_tool_requests: Whether to return tool requests.
        tool_choice: The tool choice strategy.
        tool_responses: tool response parts corresponding to interrupts.
        config: The generation configuration.
        max_turns: The maximum number of turns in a conversation.
        output_format: The output format.
        output_content_type: The output content type.
        output_instructions: Instructions for formatting the output.
        output_schema: The output schema.
        output_constrained: Whether the output should be constrained to the output schema.
        docs: A list of documents to be used for grounding.

    Returns:
        A GenerateActionOptions object.
    """
    model = options.model or registry.default_model
    if model is None:
        raise Exception('No model configured.')
    resolved_msgs: list[Message] = []
    if options.system:
        result = await render_system_prompt(registry, options.input_schema, options, prompt_cache)
        resolved_msgs.append(result)
    if options.messages:
        resolved_msgs += options.messages
    if options.prompt:
        resolved_msgs.append(Message(role=Role.USER, content=_normalize_prompt_arg(options.prompt)))

    # If is schema is set but format is not explicitly set, default to
    # `json` format.
    if options.output_schema and not options.output_format:
        output_format = 'json'
    else:
        output_format = options.output_format

    output = GenerateActionOutputConfig()
    if output_format:
        output.format = output_format
    if options.output_content_type:
        output.content_type = options.output_content_type
    if options.output_instructions is not None:
        output.instructions = options.output_instructions
    if options.output_schema:
        output.json_schema = to_json_schema(options.output_schema)
    if options.output_constrained is not None:
        output.constrained = options.output_constrained

    resume = None
    if options.tool_responses:
        resume = Resume(respond=[r.root for r in options.tool_responses])

    return GenerateActionOptions(
        model=model,
        messages=resolved_msgs,
        config=options.config,
        tools=options.tools,
        return_tool_requests=options.return_tool_requests,
        tool_choice=options.tool_choice,
        output=output,
        max_turns=options.max_turns,
        docs=options.docs,
        resume=resume,
    )


def _normalize_prompt_arg(
    prompt: str | Part | list[Part] | None,
) -> list[Part] | None:
    """Normalizes different prompt input types into a list of Parts.

    Ensures that the prompt content, whether provided as a string, a single Part,
    or a list of Parts, is consistently represented as a list of Part objects.

    Args:
        prompt: The prompt input, which can be a string, a Part, a list of Parts,
            or None.

    Returns:
        A list containing the normalized Part(s), or None if the input `prompt`
        was None.
    """
    if not prompt:
        return None
    if isinstance(prompt, str):
        return [Part(text=prompt)]
    elif hasattr(prompt, '__len__'):
        return prompt
    else:
        return [prompt]



async def render_system_prompt(
    registry: Registry,
    input: dict[str, Any],
    options: PromptConfig,
    prompt_cache: PromptCache | None = None,
) -> Message:
    """Renders a system prompt."""

    if isinstance(options.system, str):

        if prompt_cache and prompt_cache.system is None:
            prompt_cache.system = await registry.dotprompt.compile(options.system)


            return Message(
                role=Role.USER,
                content=await render_dotprompt_to_parts(
                    ActionRunContext._current_context(),
                    prompt_cache.system,
                    input,
                    PromptMetadata(
                        input=PromptInputConfig(

                        )
                    ),
                )
            )
        rendered = await registry.dotprompt.compile(options.system)
        result = await render_dotprompt_to_parts(
                ActionRunContext._current_context(),
                rendered,
                input,
                PromptMetadata(
                    input=PromptInputConfig(
                    )
                ),
            )
        return Message(
            role=Role.USER,
            content=result
        )


    return Message(
        role=Role.SYSTEM,
        content=_normalize_prompt_arg(options.system)
    )


async def render_dotprompt_to_parts(
    context: dict[str, Any],
    prompt_function: PromptFunction,
    input_: dict[str, Any],
    options: PromptMetadata | None = None,
) -> list[Part]:
    """Renders a prompt using dotprompt."""

    rendered = await prompt_function(
        data=DataArgument[dict[str, Any]](
            input=input_,
            context=context,
        ),
        options=options,
    )

    if len(rendered.messages) > 1:
        raise Exception("parts template must produce only one message")

    return [Part(root=e.content) for e in rendered.messages]
