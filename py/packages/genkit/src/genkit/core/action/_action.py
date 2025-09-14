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

"""Action module for defining and managing remotely callable functions.

This module provides the core `Action` class, the fundamental building block for
defining operations within the Genkit framework.

## What is an Action?

An Action represents a named, observable, and strongly-typed unit of work that
wraps a standard Python function (either synchronous or asynchronous). It serves
as a consistent interface for executing logic like calling models, running tools,
or orchestrating flows.

## How it Works:

1.  Initialization

    *   An `Action` is created with a unique `name`, a `kind` (e.g., MODEL,
        TOOL, FLOW), and the Python function (`fn`) containing the core logic.

    *   It automatically inspects the function's type hints (specifically the
        first argument for input and the return annotation for output) using
        Pydantic's `TypeAdapter` to generate JSON schemas (`input_schema`,
        `output_schema`). These are stored for validation and metadata. It
        raises a `TypeError` if the function signature has more than two
        arguments (input, context).

    *   It internally creates tracing wrappers (`_fn`, `_afn`) around the
        original function using the `_make_tracing_wrappers` helper. These
        wrappers handle OpenTelemetry span creation, recording input/output
        metadata, and standardizing error handling by raising `GenkitError`
        with a trace ID. The `_afn` wrapper ensures even synchronous functions
        can be awaited.

2.  Execution Methods

    *   `run()`: Executes the action synchronously. It calls the internal
        synchronous tracing wrapper (`_fn`).

    *   `arun()`: Executes the action asynchronously. It calls the internal
        asynchronous tracing wrapper (`_afn`). This wrapper handles
        awaiting the original function if it was async or running it via
        `ensure_async` if it was sync.

    *   `arun_raw()`: Similar to `arun`, but performs Pydantic validation on
        the `raw_input` before calling `arun`.

    *   `stream()`: Initiates an asynchronous execution via `arun` and returns
        an `AsyncIterator` (via `Channel`) for receiving chunks and an
        `asyncio.Future` that resolves with the final `ActionResponse`.

3.  Streaming and Context

    *   During execution (`run`/`arun`/`stream`), an `ActionRunContext` instance
        is created.

    *   This context holds an `on_chunk` callback (provided by the caller, e.g.,
        by `stream()`) and any user-provided `context` dictionary.

    *   If the wrapped function (`fn`) accepts a context argument (`ctx`), this
        `ActionRunContext` instance is passed, allowing the function to send
        intermediate chunks using `ctx.send_chunk()`.

    *   A `ContextVar` (`_action_context`) is also used to propagate the user
        context dictionary implicitly.

The `Action` class provides a robust way to define executable units, abstracting
away the complexities of sync/async handling (for async callers), schema
generation, tracing, and streaming mechanics.
"""

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from functools import cached_property
from typing import Any

from pydantic import BaseModel, TypeAdapter

from genkit.aio import Channel, ensure_async
from genkit.core.error import GenkitError
from genkit.core.tracing import tracer

from ._tracing import record_input_metadata, record_output_metadata
from ._util import extract_action_args_and_types, noop_streaming_callback
from .types import ActionKind, ActionMetadataKey, ActionResponse
from ...blocks.prompt import to_generate_action_options

# TODO: add typing, generics
StreamingCallback = Callable[[Any], None]

_action_context: ContextVar[dict[str, Any] | None] = ContextVar('context')
_action_context.set(None)


class ActionRunContext:
    """Provides context for an action's execution, including streaming support.

    This class holds context information relevant to a single execution of an
    Action. It manages the streaming callback (`on_chunk`) and any additional
    context dictionary provided during the action call.

    Attributes:
        context: A dictionary containing arbitrary context data.
        is_streaming: Whether the action is being executed in streaming mode.
    """

    def __init__(
        self,
        on_chunk: StreamingCallback | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initializes an ActionRunContext instance.

        Sets up the context with an optional streaming callback and an optional
        context dictionary. If `on_chunk` is None, a no-op callback is used.

        Args:
            on_chunk: A callable to be invoked when a chunk of data is ready
                      during streaming execution. Defaults to a no-op function.
            context: An optional dictionary containing context data to be made
                     available within the action execution. Defaults to an empty
                     dictionary.
        """
        self._on_chunk = on_chunk if on_chunk is not None else noop_streaming_callback
        self._context = context if context is not None else {}

    @property
    def context(self) -> dict[str, Any]:
        return self._context

    @cached_property
    def is_streaming(self) -> bool:
        """Indicates whether the action is being executed in streaming mode.

        This property checks if a valid streaming callback (`on_chunk`)
        was provided during initialization.

        Returns:
            True if a streaming callback (other than the no-op default) is set,
            False otherwise.
        """
        return self._on_chunk != noop_streaming_callback

    def send_chunk(self, chunk: Any) -> None:
        """Send a chunk to from the action to the client.

        Args:
            chunk: The chunk to send to the client.
        """
        self._on_chunk(chunk)

    @staticmethod
    def _current_context() -> dict[str, Any] | None:
        """Obtains current context if running within an action.

        Returns:
            The current context if running within an action, None otherwise.
        """
        return _action_context.get(None)


class Action:
    """Represents a strongly-typed, remotely callable function within Genkit.

    Actions are the fundamental building blocks for defining operations in Genkit.
    They are named, observable (via tracing), and support both streaming and
    non-streaming execution modes. An Action wraps a Python function, handling
    input validation, execution, tracing, and output serialization.

    Attributes:
        name: A unique identifier for the action.
        kind: The type category of the action (e.g., MODEL, TOOL, FLOW).
        description: An optional human-readable description.
        input_schema: The JSON schema definition for the expected input type.
        output_schema: The JSON schema definition for the expected output type.
        metadata: A dictionary for storing arbitrary metadata associated with the action.
        is_async: Whether the action is asynchronous.
    """

    def __init__(
        self,
        kind: ActionKind,
        name: str,
        fn: Callable[..., Any],
        metadata_fn: Callable[..., Any] | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        span_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize an Action.

        Args:
            kind: The kind of action (e.g., TOOL, MODEL, etc.).
            name: Unique name identifier for this action.
            fn: The function to call when the action is executed.
            metadata_fn: The function to be used to infer metadata (e.g.
                schemas).
            description: Optional human-readable description of the action.
            metadata: Optional dictionary of metadata about the action.
            span_metadata: Optional dictionary of tracing span metadata.
        """
        self._kind = kind
        self._name = name
        self._metadata = metadata if metadata else {}
        self._description = description
        self._is_async = inspect.iscoroutinefunction(fn)

        input_spec = inspect.getfullargspec(metadata_fn if metadata_fn else fn)
        action_args, arg_types = extract_action_args_and_types(input_spec)
        n_action_args = len(action_args)
        self._fn, self._afn = _make_tracing_wrappers(name, kind, span_metadata, n_action_args, fn)
        self._initialize_io_schemas(action_args, arg_types, input_spec)

    @property
    def kind(self) -> ActionKind:
        return self._kind

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str | None:
        return self._description

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @cached_property
    def input_type(self) -> type | None:
        return self._input_type

    @cached_property
    def input_schema(self) -> dict[str, Any]:
        return self._input_schema

    @cached_property
    def output_schema(self) -> dict[str, Any]:
        return self._output_schema

    @property
    def is_async(self) -> bool:
        return self._is_async

    def run(
        self,
        input: Any = None,
        on_chunk: StreamingCallback | None = None,
        context: dict[str, Any] | None = None,
        telemetry_labels: dict[str, Any] | None = None,
    ) -> ActionResponse:
        """Executes the action synchronously with the given input.

        This method runs the action's underlying function synchronously.
        It handles input validation, tracing, and output serialization.
        If the action's function is async, it will be run in the current event loop.

        Args:
            input: The input data for the action. It should conform to the action's
                   input schema.
            on_chunk: An optional callback function to receive streaming output chunks.
                      Note: For synchronous execution of streaming actions, chunks
                      will be delivered synchronously via this callback.
            context: An optional dictionary containing context data for the execution.
            telemetry_labels: Optional labels for telemetry.

        Returns:
            An ActionResponse object containing the final result and trace ID.

        Raises:
            GenkitError: If an error occurs during action execution.
        """
        # TODO: handle telemetry_labels

        if context:
            _action_context.set(context)

        return self._fn(
            input,
            ActionRunContext(on_chunk=on_chunk, context=_action_context.get(None)),
        )

    async def arun(
        self,
        input: Any = None,
        on_chunk: StreamingCallback | None = None,
        context: dict[str, Any] | None = None,
        telemetry_labels: dict[str, Any] | None = None,
    ) -> ActionResponse:
        """Executes the action asynchronously with the given input.

        This method runs the action's underlying function asynchronously.
        It handles input validation, tracing, and output serialization.
        If the action's function is synchronous, it will be wrapped to run
        asynchronously.

        Args:
            input: The input data for the action. It should conform to the action's
                   input schema.
            on_chunk: An optional callback function to receive streaming output chunks.
            context: An optional dictionary containing context data for the execution.
            telemetry_labels: Optional labels for telemetry.

        Returns:
            An awaitable ActionResponse object containing the final result and trace ID.

        Raises:
            GenkitError: If an error occurs during action execution.
        """
        # TODO: handle telemetry_labels

        if context:
            _action_context.set(context)

        return await self._afn(
            input,
            ActionRunContext(on_chunk=on_chunk, context=_action_context.get(None)),
        )

    async def arun_raw(
        self,
        raw_input: Any,
        on_chunk: StreamingCallback | None = None,
        context: dict[str, Any] | None = None,
        telemetry_labels: dict[str, Any] | None = None,
    ):
        """Executes the action asynchronously with raw, unvalidated input.

        This method bypasses the Pydantic input validation and calls the underlying
        action function directly with the provided `raw_input`. It still handles
        tracing and context management.

        Use this method when you need to work with input that may not conform
        to the defined schema or when you have already validated the input.

        Args:
            raw_input: The raw input data to pass directly to the action function.
            on_chunk: An optional callback function to receive streaming output chunks.
            context: An optional dictionary containing context data for the execution.
            telemetry_labels: Optional labels for telemetry.

        Returns:
            An awaitable ActionResponse object containing the final result and trace ID.

        Raises:
            GenkitError: If an error occurs during action execution.
        """
        input_action = self._input_type.validate_python(raw_input) if self._input_type is not None else None
        return await self.arun(
            input=input_action,
            on_chunk=on_chunk,
            context=context,
            telemetry_labels=telemetry_labels,
        )

    def stream(
        self,
        input: Any = None,
        context: dict[str, Any] | None = None,
        telemetry_labels: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> tuple[
        AsyncIterator,
        asyncio.Future[ActionResponse],
    ]:
        """Executes the action asynchronously and provides a streaming response.

        This method initiates an asynchronous action execution and returns immediately
        with a tuple containing an async iterator for the chunks and a future for the
        final response.

        Args:
            input: The input data for the action. It should conform to the action's
                   input schema.
            context: An optional dictionary containing context data for the execution.
            telemetry_labels: Optional labels for telemetry.
            timeout: Optional timeout for the stream.

        Returns:
            A tuple: (chunk_iterator, final_response_future)
            - chunk_iterator: An AsyncIterator yielding output chunks as they become available.
            - final_response_future: An asyncio.Future that will resolve to the
                                     complete ActionResponse when the action finishes.
        """
        stream = Channel(timeout=timeout)

        resp = self.arun(
            input=input,
            context=context,
            telemetry_labels=telemetry_labels,
            on_chunk=lambda c: stream.send(c),
        )
        stream.set_close_future(resp)

        result_future: asyncio.Future[ActionResponse] = asyncio.Future()
        stream.closed.add_done_callback(lambda _: result_future.set_result(stream.closed.result().response))

        return (stream, result_future)

    def _initialize_io_schemas(
        self,
        action_args: list[str],
        arg_types: list[type],
        input_spec: inspect.FullArgSpec,
    ):
        """Initializes input/output schemas based on function signature and hints.

        Uses Pydantic's TypeAdapter to generate JSON schemas for the first
        argument (if present) and the return type annotation (if present).
        Stores schemas on the instance (input_schema, output_schema) and in
        the metadata dictionary.

        Args:
            action_args: List of detected argument names.
            arg_types: List of detected argument types.
            input_spec: The FullArgSpec object from inspecting the function.

        Raises:
            TypeError: If the function has more than two arguments.
        """
        if len(action_args) > 2:
            raise TypeError(f'can only have up to 2 arg: {action_args}')

        if len(action_args) > 0:
            type_adapter = TypeAdapter(arg_types[0])
            self._input_schema = type_adapter.json_schema()
            self._input_type = type_adapter
            self._metadata[ActionMetadataKey.INPUT_KEY] = self._input_schema
        else:
            self._input_schema = TypeAdapter(Any).json_schema()
            self._input_type = None
            self._metadata[ActionMetadataKey.INPUT_KEY] = self._input_schema

        if ActionMetadataKey.RETURN in input_spec.annotations:
            type_adapter = TypeAdapter(input_spec.annotations[ActionMetadataKey.RETURN])
            self._output_schema = type_adapter.json_schema()
            self._metadata[ActionMetadataKey.OUTPUT_KEY] = self._output_schema
        else:
            self._output_schema = TypeAdapter(Any).json_schema()
            self._metadata[ActionMetadataKey.OUTPUT_KEY] = self._output_schema

# equivalent of ActionDesc
class ActionMetadata(BaseModel):
    """Metadata for actions."""

    kind: ActionKind
    name: str
    description: str | None = None
    input_schema: Any | None = None
    input_json_schema: dict[str, Any] | None = None
    output_schema: Any | None = None
    output_json_schema: dict[str, Any] | None = None
    stream_schema: Any | None = None
    metadata: dict[str, Any] | None = None


_SyncTracingWrapper = Callable[[Any | None, ActionRunContext], ActionResponse]
_AsyncTracingWrapper = Callable[[Any | None, ActionRunContext], ActionResponse]


def _make_tracing_wrappers(
    name: str, kind: ActionKind, span_metadata: dict[str, Any], n_action_args: int, fn: Callable[..., Any]
) -> tuple[_SyncTracingWrapper, _AsyncTracingWrapper]:
    """Make the sync and async tracing wrappers for an action function.

    Args:
        name: The name of the action.
        kind: The kind of action.
        span_metadata: The span metadata for the action.
        n_action_args: The arguments of the action.
        fn: The function to wrap.
    """

    async def async_tracing_wrapper(input: Any | None, ctx: ActionRunContext) -> ActionResponse:
        """Wrap the function in an async tracing wrapper.

        Args:
            input: The input to the action.
            ctx: The context to pass to the action.

        Returns:
            The action response.
        """
        afn = ensure_async(fn)
        with tracer.start_as_current_span(name) as span:
            trace_id = str(span.get_span_context().trace_id)
            record_input_metadata(
                span=span,
                kind=kind,
                name=name,
                span_metadata=span_metadata,
                input=input,
            )

            try:
                match n_action_args:
                    case 0:
                        output = await afn()
                    case 1:
                        output = await afn(input)
                    case 2:
                        output = await afn(input, ctx)
                    case _:
                        raise ValueError('action fn must have 0-2 args...')
            except Exception as e:
                raise GenkitError(
                    cause=e.cause if isinstance(e, GenkitError) and e.cause else e,
                    message=f'Error while running action {name}',
                    trace_id=trace_id,
                ) from e

            record_output_metadata(span, output=output)
            return ActionResponse(response=output, trace_id=trace_id)

    def sync_tracing_wrapper(input: Any | None, ctx: ActionRunContext) -> ActionResponse:
        """Wrap the function in a sync tracing wrapper.

        Args:
            input: The input to the action.
            ctx: The context to pass to the action.

        Returns:
            The action response.
        """
        with tracer.start_as_current_span(name) as span:
            trace_id = str(span.get_span_context().trace_id)
            record_input_metadata(
                span=span,
                kind=kind,
                name=name,
                span_metadata=span_metadata,
                input=input,
            )

            try:
                match n_action_args:
                    case 0:
                        output = fn()
                    case 1:
                        output = fn(input)
                    case 2:
                        output = fn(input, ctx)
                    case _:
                        raise ValueError('action fn must have 0-2 args...')
            except Exception as e:
                raise GenkitError(
                    cause=e,
                    message=f'Error while running action {name}',
                    trace_id=trace_id,
                ) from e

            record_output_metadata(span, output=output)
            return ActionResponse(response=output, trace_id=trace_id)

    return sync_tracing_wrapper, async_tracing_wrapper





def define_action_with_input_schema(
    registry: Any,
    name: str,
    action_kind: ActionKind,
    provider: str | None = None,
    metadata: dict | None = None,
    input_schema: dict | None = None,
    fn: Callable[..., Any] = None
) -> Action:
    """
    Crea y registra una nueva Action con un esquema de entrada definido dinámicamente.

    Args:
        registry: El registro donde se guardará la acción
        provider: El proveedor de la acción
        name: Nombre de la acción
        action_type: Tipo de acción
        metadata: Metadatos adicionales para la acción
        input_schema: Esquema JSON que define la estructura de entrada
        fn: Función que implementa la lógica de la acción

    Returns:
        ActionDef: Definición de la acción creada
    """
    full_name = name

    return define_action(
        registry,
        full_name,
        action_kind,
        provider,
        metadata,
        input_schema,
        fn
    )



def define_action(
    registry: Any,
    name: str,
    action_kind: ActionKind,
    provider: str | None = None,
    metadata: dict | None = None,
    input_schema: dict | None = None,
    fn: Callable[..., Any] = None
):
    full_name = name

    if provider:
        full_name = f"{provider}/{name}"

    # Construir la clave para el registro
    key = f"/{action_kind}/{full_name}"

    return registry.register_action(
        action_kind,
        key,
        fn=fn,
        metadata=metadata,
        metadata_fn=input_schema
    )


