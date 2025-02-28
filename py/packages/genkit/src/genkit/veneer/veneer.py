# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Veneer user-facing API for application developers who use the SDK."""

import logging
import os
import threading
from http.server import HTTPServer
from typing import Any

from genkit.ai.embedding import EmbedRequest, EmbedResponse
from genkit.ai.generate import StreamingCallback as ModelStreamingCallback
from genkit.ai.generate import generate_action
from genkit.ai.model import GenerateResponseWrapper
from genkit.core.action import ActionKind
from genkit.core.environment import is_dev_environment
from genkit.core.reflection import make_reflection_server
from genkit.core.schema import to_json_schema
from genkit.core.typing import (
    GenerateActionOptions,
    GenerationCommonConfig,
    Message,
    Output,
    Part,
    Role,
    TextPart,
    ToolChoice,
)
from genkit.veneer import server
from genkit.veneer.plugin import Plugin
from genkit.veneer.registry import GenkitRegistry

DEFAULT_REFLECTION_SERVER_SPEC = server.ServerSpec(
    scheme='http', host='127.0.0.1', port=3100
)

logger = logging.getLogger(__name__)


class Genkit(GenkitRegistry):
    """Veneer user-facing API for application developers who use the SDK."""

    def __init__(
        self,
        plugins: list[Plugin] | None = None,
        model: str | None = None,
        reflection_server_spec=DEFAULT_REFLECTION_SERVER_SPEC,
    ) -> None:
        """Initialize a new Genkit instance.

        Args:
            plugins: Optional list of plugins to initialize.
            model: Optional model name to use.
            reflection_server_spec: Optional server spec for the reflection
                server.
        """
        super().__init__()
        self.registry.default_model = model

        if is_dev_environment():
            runtimes_dir = os.path.join(os.getcwd(), '.genkit/runtimes')
            server.create_runtime(
                runtime_dir=runtimes_dir,
                reflection_server_spec=reflection_server_spec,
                at_exit_fn=os.remove,
            )
            self.thread = threading.Thread(
                target=self.start_server,
                args=(
                    reflection_server_spec.host,
                    reflection_server_spec.port,
                ),
            )
            self.thread.start()

        if not plugins:
            logger.warning('No plugins provided to Genkit')
        else:
            for plugin in plugins:
                if isinstance(plugin, Plugin):
                    plugin.initialize(ai=self)

                    def resolver(kind, name):
                        return plugin.resolve_action(self, kind, name)

                    self.registry.register_action_resolver(
                        plugin.plugin_name(), resolver
                    )
                else:
                    raise ValueError(
                        f'Invalid {plugin=} provided to Genkit: '
                        f'must be of type `genkit.veneer.plugin.Plugin`'
                    )

    def start_server(self, host: str, port: int) -> None:
        """Start the HTTP server for handling requests.

        Args:
            host: The hostname to bind to.
            port: The port number to listen on.
        """
        httpd = HTTPServer(
            (host, port),
            make_reflection_server(registry=self.registry),
        )
        httpd.serve_forever()

    async def generate(
        self,
        model: str | None = None,
        prompt: str | Part | list[Part] | None = None,
        system: str | Part | list[Part] | None = None,
        messages: list[Message] | None = None,
        tools: list[str] | None = None,
        return_tool_requests: bool | None = None,
        tool_choice: ToolChoice = None,
        config: GenerationCommonConfig | dict[str, Any] | None = None,
        max_turns: int | None = None,
        on_chunk: ModelStreamingCallback | None = None,
        context: dict[str, Any] | None = None,
        output_format: str | None = None,
        content_type: str | None = None,
        output_instructions: bool | str | None = None,
        output_schema: type | dict[str, Any] | None = None,
        constrained: bool | None = None,
        # TODO:
        #  docs: list[Document]
        #  use: list[ModelMiddleware]
        #  resume: ResumeOptions
    ) -> GenerateResponseWrapper:
        """Generates text or structured data using a language model.

        This function provides a flexible interface for interacting with various language models,
        supporting both simple text generation and more complex interactions involving tools and
        structured conversations.

        Args:
            model: Optional. The name of the model to use for generation. If not provided, a default
                   model may be used.
            prompt: Optional. A single prompt string, a `Part` object, or a list of `Part` objects
                    to provide as input to the model. This is used for simple text generation.
            system: Optional. A system message string, a `Part` object, or a list of `Part` objects
                    to provide context or instructions to the model, especially for chat-based models.
            messages: Optional. A list of `Message` objects representing a conversation history.
                      This is used for chat-based models to maintain context.
            tools: Optional. A list of tool names (strings) that the model can use.
            return_tool_requests: Optional. If `True`, the model will return tool requests instead of
                                  executing them directly.
            tool_choice: Optional. A `ToolChoice` object specifying how the model should choose
                         which tool to use.
            config: Optional. A `GenerationCommonConfig` object or a dictionary containing configuration
                    parameters for the generation process. This allows fine-tuning the model's
                    behavior.
            max_turns: Optional. The maximum number of turns in a conversation.
            on_chunk: Optional. A callback function of type `ModelStreamingCallback` that is called
                      for each chunk of generated text during streaming.
            context: Optional. A dictionary containing additional context information that can be
                     used during generation.

        Returns:
            A `GenerateResponseWrapper` object containing the model's response, which may include
            generated text, tool requests, or other relevant information.

        Note:
            - The `tools`, `return_tool_requests`, and `tool_choice` arguments are used for models
              that support tool usage.
            - The `on_chunk` argument enables streaming responses, allowing you to process the
              generated content as it becomes available.
        """
        model = model or self.registry.default_model
        if model is None:
            raise Exception('No model configured.')
        if (
            config
            and not isinstance(config, GenerationCommonConfig)
            and not isinstance(config, dict)
        ):
            raise AttributeError('Invalid generate config provided')

        resolved_msgs: list[Message] = []
        if system:
            resolved_msgs.append(
                Message(role=Role.SYSTEM, content=_normalize_prompt_arg(system))
            )
        if messages:
            resolved_msgs += messages
        if prompt:
            resolved_msgs.append(
                Message(role=Role.USER, content=_normalize_prompt_arg(prompt))
            )

        output = Output()
        if output_format:
            output.format = output_format
        if content_type:
            output.content_type = content_type
        if output_instructions != None:
            output.instructions = output_instructions
        if output_schema:
            output.json_schema = to_json_schema(output_schema)
        if constrained != None:
            output.constrained = constrained

        return await generate_action(
            self.registry,
            GenerateActionOptions(
                model=model,
                messages=resolved_msgs,
                config=config,
                tools=tools,
                return_tool_requests=return_tool_requests,
                tool_choice=tool_choice,
                output=output,
                max_turns=max_turns,
            ),
            on_chunk=on_chunk,
        )

    async def embed(
        self, model: str | None = None, documents: list[str] | None = None
    ) -> EmbedResponse:
        """Calculates embeddings for the given texts.

        Args:
            model: Optional embedder model name to use.
            documents: Texts to embed.

        Returns:
            The generated response with embeddings.
        """
        embed_action = self.registry.lookup_action(ActionKind.EMBEDDER, model)

        return (
            await embed_action.arun(EmbedRequest(documents=documents))
        ).response


def _normalize_prompt_arg(prompt: str | Part | list[Part] | None) -> list[Part]:
    if not prompt:
        return None
    if isinstance(prompt, str):
        return [TextPart(text=prompt)]
    elif hasattr(prompt, '__len__'):
        return prompt
    else:
        return [prompt]
