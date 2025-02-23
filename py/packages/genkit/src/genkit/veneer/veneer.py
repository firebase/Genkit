# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Veneer user-facing API for application developers who use the SDK."""

import logging
import os
import threading
from collections.abc import Callable
from http.server import HTTPServer
from typing import Any

from genkit.ai.model import ModelFn
from genkit.ai.prompt import PromptFn
from genkit.core.action import ActionKind
from genkit.core.environment import is_dev_environment
from genkit.core.plugin_abc import Plugin
from genkit.core.reflection import make_reflection_server
from genkit.core.registry import Registry
from genkit.core.typing import (
    GenerateRequest,
    GenerateResponse,
    GenerationCommonConfig,
    Message,
)
from genkit.veneer import server

DEFAULT_REFLECTION_SERVER_SPEC = server.ServerSpec(
    scheme='http', host='127.0.0.1', port=3100
)

logger = logging.getLogger(__name__)


class Genkit:
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
        self.model = model
        self.registry = Registry()

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
                    plugin.initialize(registry=self.registry)
                else:
                    raise ValueError(
                        f'Invalid {plugin=} provided to Genkit: '
                        f'must be of type `genkit.core.plugin_abc.Plugin`'
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

    def generate(
        self,
        model: str | None = None,
        prompt: str | None = None,
        messages: list[Message] | None = None,
        system: str | None = None,
        tools: list[str] | None = None,
        config: GenerationCommonConfig | None = None,
    ) -> GenerateResponse:
        """Generate text using a language model.

        Args:
            model: Optional model name to use.
            prompt: Optional raw prompt string.
            messages: Optional list of messages for chat models.
            system: Optional system message for chat models.
            tools: Optional list of tools to use.
            config: Optional generation configuration.

        Returns:
            The generated text response.
        """
        model = model if model is not None else self.model
        if model is None:
            raise Exception('No model configured.')
        if config and not isinstance(config, GenerationCommonConfig):
            raise AttributeError('Invalid generate config provided')

        model_action = self.registry.lookup_action(ActionKind.MODEL, model)

        return model_action.fn(
            GenerateRequest(
                messages=messages,
                config=config,
            )
        ).response

    def flow(self, name: str | None = None) -> Callable[[Callable], Callable]:
        """Decorator to register a function as a flow.

        Args:
            name: Optional name for the flow. If not provided, uses the
                function name.

        Returns:
            A decorator function that registers the flow.
        """

        def wrapper(func: Callable) -> Callable:
            flow_name = name if name is not None else func.__name__
            action = self.registry.register_action(
                name=flow_name,
                kind=ActionKind.FLOW,
                fn=func,
                span_metadata={'genkit:metadata:flow:name': flow_name},
            )

            def decorator(*args: Any, **kwargs: Any) -> GenerateResponse:
                return action.fn(*args, **kwargs).response

            return decorator

        return wrapper

    def define_model(
        self,
        name: str,
        fn: ModelFn,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Define a custom model action.

        Args:
            name: Name of the model.
            fn: Function implementing the model behavior.
            metadata: Optional metadata for the model.
        """
        self.registry.register_action(
            name=name,
            kind=ActionKind.MODEL,
            fn=fn,
            metadata=metadata,
        )

    def define_prompt(
        self,
        name: str,
        fn: PromptFn,
        model: str | None = None,
    ) -> Callable[[Any | None], GenerateResponse]:
        """Define a custom prompt action.

        Args:
            name: Name of the prompt.
            fn: Function implementing the prompt behavior.
            model: Optional model name to use.

        Returns:
            A function that generates text using the prompt.
        """

        def prompt(input_prompt: Any | None = None) -> GenerateResponse:
            req = fn(input_prompt)
            return self.generate(messages=req.messages, model=model)

        action = self.registry.register_action(
            kind=ActionKind.PROMPT,
            name=name,
            fn=fn,
        )

        def wrapper(input_prompt: Any | None = None) -> GenerateResponse:
            return action.fn(input_prompt)

        return wrapper
