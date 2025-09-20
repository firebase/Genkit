#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the action module."""

import json
from typing import Any

import pytest
from pydantic import BaseModel, Field

from genkit.ai import Genkit, ToolRunContext, tool_response
from genkit.blocks.document import Document
from genkit.blocks.formats.types import FormatDef, Formatter, FormatterConfig
from genkit.blocks.model import MessageWrapper, text_from_message
from genkit.core.action import ActionRunContext
from genkit.core.typing import (
    BaseEvalDataPoint,
    Details,
    DocumentData,
    DocumentPart,
    EvalFnResponse,
    EvalRequest,
    EvalResponse,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChunk,
    Message,
    Metadata,
    ModelInfo,
    OutputConfig,
    Part,
    RetrieverRequest,
    RetrieverResponse,
    Role,
    Score,
    Supports,
    TextPart,
    ToolChoice,
    ToolDefinition,
    ToolRequest,
    ToolResponse,
)
from genkit.testing import (
    EchoModel,
    ProgrammableModel,
    define_echo_model,
    define_programmable_model,
)

# type SetupFixture = tuple[Genkit, EchoModel, ProgrammableModel]
SetupFixture = tuple[Genkit, EchoModel, ProgrammableModel]


@pytest.fixture
def setup_test():
    """Setup a test fixture for the veneer tests."""
    ai = Genkit(model='echoModel')

    pm, _ = define_programmable_model(ai)
    echo, _ = define_echo_model(ai)

    return (ai, echo, pm)


@pytest.mark.asyncio
async def test_generate_uses_default_model(setup_test: SetupFixture) -> None:
    """Test that the generate function uses the default model."""
    ai, *_ = setup_test

    want_txt = '[ECHO] user: "hi" {"temperature":11.0}'

    response = await ai.generate(prompt='hi', config={'temperature': 11})

    assert response.text == want_txt

    _, response = ai.generate_stream(prompt='hi', config={'temperature': 11})

    assert (await response).text == want_txt


@pytest.mark.asyncio
async def test_generate_with_explicit_model(setup_test: SetupFixture) -> None:
    """Test that the generate function uses the explicit model."""
    ai, *_ = setup_test

    response = await ai.generate(model='echoModel', prompt='hi', config={'temperature': 11})

    assert response.text == '[ECHO] user: "hi" {"temperature":11.0}'

    _, response = ai.generate_stream(model='echoModel', prompt='hi', config={'temperature': 11})

    assert (await response).text == '[ECHO] user: "hi" {"temperature":11.0}'


@pytest.mark.asyncio
async def test_generate_with_str_prompt(setup_test: SetupFixture) -> None:
    """Test that the generate function with a string prompt works."""
    ai, *_ = setup_test

    response = await ai.generate(prompt='hi', config={'temperature': 11})

    assert response.text == '[ECHO] user: "hi" {"temperature":11.0}'


@pytest.mark.asyncio
async def test_generate_with_part_prompt(setup_test: SetupFixture) -> None:
    """Test that the generate function with a part prompt works."""
    ai, *_ = setup_test

    want_txt = '[ECHO] user: "hi" {"temperature":11.0}'

    response = await ai.generate(prompt=Part(text='hi'), config={'temperature': 11})

    assert response.text == want_txt

    _, response = ai.generate_stream(prompt=Part(text='hi'), config={'temperature': 11})

    assert (await response).text == want_txt


@pytest.mark.asyncio
async def test_generate_with_part_list_prompt(setup_test: SetupFixture) -> None:
    """Test that the generate function with a list of parts prompt works."""
    ai, *_ = setup_test

    want_txt = '[ECHO] user: "hello","world" {"temperature":11.0}'

    response = await ai.generate(
        prompt=[Part(text='hello'), Part(text='world')],
        config={'temperature': 11},
    )

    assert response.text == want_txt

    _, response = ai.generate_stream(
        prompt=[Part(text='hello'), Part(text='world')],
        config={'temperature': 11},
    )

    assert (await response).text == want_txt


@pytest.mark.asyncio
async def test_generate_with_str_system(setup_test: SetupFixture) -> None:
    """Test that the generate function with a string system works."""
    ai, *_ = setup_test

    want_txt = '[ECHO] system: "talk like pirate" user: "hi" {"temperature":11.0}'

    response = await ai.generate(system='talk like pirate', prompt='hi', config={'temperature': 11})

    assert response.text == want_txt

    _, response = ai.generate_stream(system='talk like pirate', prompt='hi', config={'temperature': 11})

    assert (await response).text == want_txt


@pytest.mark.asyncio
async def test_generate_with_part_system(setup_test: SetupFixture) -> None:
    """Test that the generate function with a part system works."""
    ai, *_ = setup_test

    want_txt = '[ECHO] system: "talk like pirate" user: "hi" {"temperature":11.0}'

    response = await ai.generate(
        system=Part(text='talk like pirate'),
        prompt='hi',
        config={'temperature': 11},
    )

    assert response.text == want_txt

    _, response = ai.generate_stream(
        system=Part(text='talk like pirate'),
        prompt='hi',
        config={'temperature': 11},
    )

    assert (await response).text == want_txt


@pytest.mark.asyncio
async def test_generate_with_part_list_system(setup_test: SetupFixture) -> None:
    """Test that the generate function with a list of parts system works."""
    ai, *_ = setup_test

    want_txt = '[ECHO] system: "talk","like pirate" user: "hi" {"temperature":11.0}'

    response = await ai.generate(
        system=[Part(text='talk'), Part(text='like pirate')],
        prompt='hi',
        config={'temperature': 11},
    )

    assert response.text == want_txt

    _, response = ai.generate_stream(
        system=[Part(text='talk'), Part(text='like pirate')],
        prompt='hi',
        config={'temperature': 11},
    )

    assert (await response).text == want_txt


@pytest.mark.asyncio
async def test_generate_with_messages(setup_test: SetupFixture) -> None:
    """Test that the generate function with a list of messages works."""
    ai, *_ = setup_test

    response = await ai.generate(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(text='hi')],
            ),
        ],
        config={'temperature': 11},
    )

    assert response.text == '[ECHO] user: "hi" {"temperature":11.0}'

    _, response = ai.generate_stream(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(text='hi')],
            ),
        ],
        config={'temperature': 11},
    )

    assert (await response).text == '[ECHO] user: "hi" {"temperature":11.0}'


@pytest.mark.asyncio
async def test_generate_with_system_prompt_messages(
    setup_test: SetupFixture,
) -> None:
    """Generate function with a system prompt and messages works."""
    ai, *_ = setup_test

    want_txt = '[ECHO] system: "talk like pirate" user: "hi" model: "bye" user: "hi again"'

    response = await ai.generate(
        system='talk like pirate',
        prompt='hi again',
        messages=[
            Message(
                role=Role.USER,
                content=[Part(text='hi')],
            ),
            Message(
                role=Role.MODEL,
                content=[Part(text='bye')],
            ),
        ],
    )

    assert response.text == want_txt

    _, response = ai.generate_stream(
        system='talk like pirate',
        prompt='hi again',
        messages=[
            Message(
                role=Role.USER,
                content=[Part(text='hi')],
            ),
            Message(
                role=Role.MODEL,
                content=[Part(text='bye')],
            ),
        ],
    )

    assert (await response).text == want_txt


@pytest.mark.asyncio
async def test_generate_with_tools(setup_test: SetupFixture) -> None:
    """Test that the generate function with tools works."""
    ai, echo, *_ = setup_test

    class ToolInput(BaseModel):
        value: int = Field(None, description='value field')

    @ai.tool(name='testTool')
    def test_tool(input: ToolInput):
        """The tool."""
        return input.value

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        tool_choice=ToolChoice.REQUIRED,
        tools=['testTool'],
    )

    want_txt = f'[ECHO] user: "hi" tools=testTool tool_choice={ToolChoice.REQUIRED}'

    want_request = [
        ToolDefinition(
            name='testTool',
            description='The tool.',
            input_schema={
                'properties': {
                    'value': {
                        'default': None,
                        'description': 'value field',
                        'title': 'Value',
                        'type': 'integer',
                    }
                },
                'title': 'ToolInput',
                'type': 'object',
            },
            outputSchema={},
        )
    ]

    assert response.text == want_txt
    assert echo.last_request.tools == want_request

    _, response = ai.generate_stream(
        model='echoModel',
        prompt='hi',
        tool_choice=ToolChoice.REQUIRED,
        tools=['testTool'],
    )

    assert (await response).text == want_txt
    assert echo.last_request.tools == want_request


@pytest.mark.asyncio
async def test_generate_with_iterrupting_tools(
    setup_test: SetupFixture,
) -> None:
    """Test that the generate function with tools works."""
    ai, _, pm, *_ = setup_test

    class ToolInput(BaseModel):
        value: int = Field(None, description='value field')

    @ai.tool(name='test_tool')
    def test_tool(input: ToolInput):
        """The tool."""
        return input.value + 7

    @ai.tool(name='test_interrupt')
    def test_interrupt(input: ToolInput, ctx: ToolRunContext):
        """The interrupt."""
        ctx.interrupt({'banana': 'yes please'})

    tool_request_msg = MessageWrapper(
        Message(
            role=Role.MODEL,
            content=[
                Part(text='call these tools'),
                Part(tool_request=ToolRequest(input={'value': 5}, name='test_interrupt', ref='123')),
                Part(tool_request=ToolRequest(input={'value': 5}, name='test_tool', ref='234')),
            ],
        )
    )
    pm.responses.append(
        GenerateResponse(
            finishReason=FinishReason.STOP,
            message=tool_request_msg,
        )
    )
    pm.responses.append(
        GenerateResponse(
            finishReason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(text='tool called')]),
        )
    )

    response = await ai.generate(
        model='programmableModel',
        prompt='hi',
        tools=['test_tool', 'test_interrupt'],
    )

    want_request = [
        ToolDefinition(
            name='test_tool',
            description='The tool.',
            input_schema={
                'properties': {
                    'value': {
                        'default': None,
                        'description': 'value field',
                        'title': 'Value',
                        'type': 'integer',
                    }
                },
                'title': 'ToolInput',
                'type': 'object',
            },
            outputSchema={},
        ),
        ToolDefinition(
            name='test_interrupt',
            description='The interrupt.',
            input_schema={
                'properties': {
                    'value': {
                        'default': None,
                        'description': 'value field',
                        'title': 'Value',
                        'type': 'integer',
                    }
                },
                'title': 'ToolInput',
                'type': 'object',
            },
            outputSchema={},
        ),
    ]

    assert response.text == 'call these tools'
    assert response.message == MessageWrapper(
        Message(
            role=Role.MODEL,
            content=[
                Part(
                    text='call these tools',
                ),
                Part(
                    tool_request=ToolRequest(ref='123', name='test_interrupt', input={'value': 5}),
                    metadata=Metadata(root={'interrupt': {'banana': 'yes please'}}),
                ),
                Part(
                    tool_request=ToolRequest(ref='234', name='test_tool', input={'value': 5}),
                    metadata=Metadata(root={'pendingOutput': 12}),
                ),
            ],
        )
    )
    assert pm.last_request.tools == want_request


@pytest.mark.asyncio
async def test_generate_with_interrupt_respond(
    setup_test: SetupFixture,
) -> None:
    """Test that the generate function with tools works."""
    ai, _, pm, *_ = setup_test

    class ToolInput(BaseModel):
        value: int = Field(None, description='value field')

    @ai.tool(name='test_tool')
    def test_tool(input: ToolInput):
        """The tool."""
        return input.value + 7

    @ai.tool(name='test_interrupt')
    def test_interrupt(input: ToolInput, ctx: ToolRunContext):
        """The interrupt."""
        ctx.interrupt({'banana': 'yes please'})

    tool_request_msg = MessageWrapper(
        Message(
            role=Role.MODEL,
            content=[
                Part(text='call these tools'),
                Part(tool_request=ToolRequest(input={'value': 5}, name='test_interrupt', ref='123')),
                Part(tool_request=ToolRequest(input={'value': 5}, name='test_tool', ref='234')),
            ],
        )
    )
    pm.responses.append(
        GenerateResponse(
            finishReason=FinishReason.STOP,
            message=tool_request_msg,
        )
    )
    pm.responses.append(
        GenerateResponse(
            finishReason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(text='tool called')]),
        )
    )

    interrupted_response = await ai.generate(
        model='programmableModel',
        prompt='hi',
        tools=['test_tool', 'test_interrupt'],
    )

    assert interrupted_response.finish_reason == 'interrupted'
    assert interrupted_response.tool_requests == [
        Part(
            tool_request=ToolRequest(ref='123', name='test_interrupt', input={'value': 5}),
            metadata={'interrupt': {'banana': 'yes please'}},
        ).root,
        Part(
            tool_request=ToolRequest(ref='234', name='test_tool', input={'value': 5}),
            metadata={'pendingOutput': 12},
        ).root,
    ]

    assert interrupted_response.messages == [
        Message(
            role='user',
            content=[Part(text='hi')],
        ),
        Message(
            role='model',
            content=[
                Part(
                    text='call these tools',
                ),
                Part(
                    tool_request=ToolRequest(
                        ref='123',
                        name='test_interrupt',
                        input={'value': 5},
                    ),
                    metadata={'interrupt': {'banana': 'yes please'}},
                ),
                Part(
                    tool_request=ToolRequest(ref='234', name='test_tool', input={'value': 5}),
                    metadata={'pendingOutput': 12},
                ),
            ],
        ),
    ]

    response = await ai.generate(
        model='programmableModel',
        messages=interrupted_response.messages,
        tool_responses=[tool_response(interrupted_response.interrupts[0], {'bar': 2})],
        tools=['test_tool', 'test_interrupt'],
    )

    assert response.text == 'tool called'

    assert response.messages == [
        Message(
            role='user',
            content=[Part(text='hi')],
        ),
        Message(
            role='model',
            content=[
                Part(text='call these tools'),
                Part(
                    tool_request=ToolRequest(ref='123', name='test_interrupt', input={'value': 5}),
                    metadata={'resolvedInterrupt': {'banana': 'yes please'}},
                ),
                Part(
                    tool_request=ToolRequest(ref='234', name='test_tool', input={'value': 5}),
                    metadata={'pendingOutput': 12},
                ),
            ],
            metadata=None,
        ),
        Message(
            role='tool',
            content=[
                Part(
                    tool_response=ToolResponse(ref='123', name='test_interrupt', output={'bar': 2}),
                    metadata=Metadata(root={'interruptResponse': True}),
                ),
                Part(
                    tool_response=ToolResponse(ref='234', name='test_tool', output=12),
                    metadata={'source': 'pending'},
                ),
            ],
            metadata={'resumed': True},
        ),
        Message(
            role='model',
            content=[Part(text='tool called')],
            metadata=None,
        ),
    ]


@pytest.mark.asyncio
async def test_generate_with_tools_and_output(setup_test: SetupFixture) -> None:
    """Test that the generate function with tools and output works."""
    ai, _, pm, *_ = setup_test

    class ToolInput(BaseModel):
        value: int = Field(None, description='value field')

    @ai.tool(name='testTool')
    def test_tool(input: ToolInput):
        """The tool."""
        return 'abc'

    tool_request_msg = MessageWrapper(
        Message(
            role=Role.MODEL,
            content=[Part(tool_request=ToolRequest(input={'value': 5}, name='testTool', ref='123'))],
        )
    )
    pm.responses.append(
        GenerateResponse(
            finishReason=FinishReason.STOP,
            message=tool_request_msg,
        )
    )
    pm.responses.append(
        GenerateResponse(
            finishReason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(text='tool called')]),
        )
    )

    response = await ai.generate(
        model='programmableModel',
        prompt='hi',
        tool_choice=ToolChoice.REQUIRED,
        tools=['testTool'],
    )

    assert response.text == 'tool called'
    assert response.request.messages[0] == Message(role=Role.USER, content=[Part(text='hi')])
    assert response.request.messages[1] == tool_request_msg
    assert response.request.messages[2] == Message(
        role=Role.TOOL,
        content=[Part(tool_response=ToolResponse(ref='123', name='testTool', output='abc'))],
    )
    assert pm.last_request.tools == [
        ToolDefinition(
            name='testTool',
            description='The tool.',
            input_schema={
                'properties': {
                    'value': {
                        'default': None,
                        'description': 'value field',
                        'title': 'Value',
                        'type': 'integer',
                    }
                },
                'title': 'ToolInput',
                'type': 'object',
            },
            outputSchema={},
        )
    ]


@pytest.mark.asyncio
async def test_generate_stream_with_tools(setup_test: SetupFixture) -> None:
    """Test that the generate stream function with tools works."""
    ai, _, pm, *_ = setup_test

    class ToolInput(BaseModel):
        value: int = Field(None, description='value field')

    @ai.tool(name='testTool')
    def test_tool(input: ToolInput):
        """The tool."""
        return 'abc'

    tool_request_msg = MessageWrapper(
        Message(
            role=Role.MODEL,
            content=[Part(tool_request=ToolRequest(input={'value': 5}, name='testTool', ref='123'))],
        )
    )
    pm.responses.append(
        GenerateResponse(
            finishReason=FinishReason.STOP,
            message=tool_request_msg,
        )
    )
    pm.responses.append(
        GenerateResponse(
            finishReason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(text='tool called')]),
        )
    )
    pm.chunks = [
        [
            GenerateResponseChunk(
                role=tool_request_msg.role,
                content=tool_request_msg.content,
            )
        ],
        [GenerateResponseChunk(role=Role.MODEL, content=[Part(text='tool called')])],
    ]

    stream, aresponse = ai.generate_stream(
        model='programmableModel',
        prompt='hi',
        tool_choice=ToolChoice.REQUIRED,
        tools=['testTool'],
    )

    chunks = []
    async for chunk in stream:
        summary = ''
        if chunk.role:
            summary += f'{chunk.role} '
        for p in chunk.content:
            summary += str(type(p.root).__name__)
            if isinstance(p.root, TextPart):
                summary += f' {p.root.text}'
        chunks.append(summary)

    response = await aresponse

    assert response.text == 'tool called'
    assert response.request.messages[0] == Message(role=Role.USER, content=[Part(text='hi')])
    assert response.request.messages[1] == tool_request_msg
    assert response.request.messages[2] == Message(
        role=Role.TOOL,
        content=[Part(tool_response=ToolResponse(ref='123', name='testTool', output='abc'))],
    )
    assert chunks == [
        'model ToolRequestPart',
        'tool ToolResponsePart',
        'model TextPart tool called',
    ]


@pytest.mark.asyncio
async def test_generate_stream_no_need_to_await_response(
    setup_test: SetupFixture,
) -> None:
    """Test that the generate stream function no need to await response."""
    ai, _, pm, *_ = setup_test

    pm.responses.append(
        GenerateResponse(
            finishReason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(text='something else')]),
        )
    )
    pm.chunks = [
        [
            GenerateResponseChunk(role=Role.MODEL, content=[Part(text='h')]),
            GenerateResponseChunk(role=Role.MODEL, content=[Part(text='i')]),
        ],
    ]

    stream, _ = ai.generate_stream(model='programmableModel', prompt='do it')
    chunks = ''
    async for chunk in stream:
        chunks += chunk.text
    assert chunks == 'hi'


@pytest.mark.asyncio
async def test_generate_with_output(setup_test: SetupFixture) -> None:
    """Test that the generate function with output works."""
    ai, *_ = setup_test

    class TestSchema(BaseModel):
        foo: int = Field(None, description='foo field')
        bar: str = Field(None, description='bar field')

    want = GenerateRequest(
        messages=[
            Message(role=Role.USER, content=[Part(text='hi')]),
        ],
        config={},
        tools=[],
        output=OutputConfig(
            format='json',
            schema_={
                'properties': {
                    'foo': {
                        'default': None,
                        'description': 'foo field',
                        'title': 'Foo',
                        'type': 'integer',
                    },
                    'bar': {
                        'default': None,
                        'description': 'bar field',
                        'title': 'Bar',
                        'type': 'string',
                    },
                },
                'title': 'TestSchema',
                'type': 'object',
            },
            constrained=True,
            content_type='application/json',
        ),
    )

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        output_constrained=True,
        output_format='json',
        output_content_type='application/json',
        output_schema=TestSchema,
        output_instructions=False,
    )

    assert response.request == want

    _, response = ai.generate_stream(
        model='echoModel',
        prompt='hi',
        output_constrained=True,
        output_format='json',
        output_content_type='application/json',
        output_schema=TestSchema,
        output_instructions=False,
    )

    assert (await response).request == want


@pytest.mark.asyncio
async def test_generate_defaults_to_json_format(
    setup_test: SetupFixture,
) -> None:
    """When output_schema is provided, format will default to json."""
    ai, *_ = setup_test

    class TestSchema(BaseModel):
        foo: int = Field(None, description='foo field')
        bar: str = Field(None, description='bar field')

    want = GenerateRequest(
        messages=[
            Message(role=Role.USER, content=[Part(text='hi')]),
        ],
        config={},
        tools=[],
        output=OutputConfig(
            format='json',
            schema_={
                'properties': {
                    'foo': {
                        'default': None,
                        'description': 'foo field',
                        'title': 'Foo',
                        'type': 'integer',
                    },
                    'bar': {
                        'default': None,
                        'description': 'bar field',
                        'title': 'Bar',
                        'type': 'string',
                    },
                },
                'title': 'TestSchema',
                'type': 'object',
            },
            # these get populated by the format
            constrained=True,
            content_type='application/json',
        ),
    )

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        output_schema=TestSchema,
    )

    assert response.request == want

    _, response = ai.generate_stream(
        model='echoModel',
        prompt='hi',
        output_schema=TestSchema,
    )

    assert (await response).request == want


@pytest.mark.asyncio
async def test_generate_json_format_unconstrained(
    setup_test: SetupFixture,
) -> None:
    """When output_schema is provided, format will default to json."""
    ai, *_ = setup_test

    class TestSchema(BaseModel):
        foo: int = Field(None, description='foo field')
        bar: str = Field(None, description='bar field')

    want = GenerateRequest(
        messages=[
            Message(role=Role.USER, content=[Part(text='hi')]),
        ],
        config={},
        tools=[],
        output=OutputConfig(
            format='json',
            schema_={
                'properties': {
                    'foo': {
                        'default': None,
                        'description': 'foo field',
                        'title': 'Foo',
                        'type': 'integer',
                    },
                    'bar': {
                        'default': None,
                        'description': 'bar field',
                        'title': 'Bar',
                        'type': 'string',
                    },
                },
                'title': 'TestSchema',
                'type': 'object',
            },
            constrained=False,
            content_type='application/json',
        ),
    )

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        output_schema=TestSchema,
        output_constrained=False,
    )

    assert response.request == want

    _, response = ai.generate_stream(
        model='echoModel',
        prompt='hi',
        output_schema=TestSchema,
        output_constrained=False,
    )

    assert (await response).request == want


@pytest.mark.asyncio
async def test_generate_with_middleware(
    setup_test: SetupFixture,
) -> None:
    """When middleware is provided, applies it."""
    ai, *_ = setup_test

    async def pre_middle(req, ctx, next):
        txt = ''.join(text_from_message(m) for m in req.messages)
        return await next(
            GenerateRequest(
                messages=[
                    Message(role=Role.USER, content=[Part(text=f'PRE {txt}')]),
                ],
            ),
            ctx,
        )

    async def post_middle(req, ctx, next):
        resp: GenerateResponse = await next(req, ctx)
        txt = text_from_message(resp.message)
        return GenerateResponse(
            finishReason=resp.finish_reason,
            message=Message(role=Role.USER, content=[Part(text=f'{txt} POST')]),
        )

    want = '[ECHO] user: "PRE hi" POST'

    response = await ai.generate(model='echoModel', prompt='hi', use=[pre_middle, post_middle])

    assert response.text == want

    _, response = ai.generate_stream(model='echoModel', prompt='hi', use=[pre_middle, post_middle])

    assert (await response).text == want


@pytest.mark.asyncio
async def test_generate_passes_through_current_action_context(
    setup_test,
) -> None:
    """Test that generate uses current action context by default."""
    ai, *_ = setup_test

    async def inject_context(req, ctx, next):
        txt = ''.join(text_from_message(m) for m in req.messages)
        return await next(
            GenerateRequest(
                messages=[
                    Message(
                        role=Role.USER,
                        content=[Part(text=f'{txt} {ctx.context}')],
                    ),
                ],
            ),
            ctx,
        )

    async def action_fn():
        return await ai.generate(model='echoModel', prompt='hi', use=[inject_context])

    action = ai.registry.register_action(name='test_action', kind='custom', fn=action_fn)
    action_response = await action.arun(context={'foo': 'bar'})

    assert action_response.response.text == '''[ECHO] user: "hi {'foo': 'bar'}"'''


@pytest.mark.asyncio
async def test_generate_uses_explicitly_passed_in_context(
    setup_test,
) -> None:
    """Generate uses specific context instead of current action context."""
    ai, *_ = setup_test

    async def inject_context(req, ctx, next):
        txt = ''.join(text_from_message(m) for m in req.messages)
        return await next(
            GenerateRequest(
                messages=[
                    Message(
                        role=Role.USER,
                        content=[Part(text=f'{txt} {ctx.context}')],
                    ),
                ],
            ),
            ctx,
        )

    async def action_fn():
        return await ai.generate(
            model='echoModel',
            prompt='hi',
            use=[inject_context],
            context={'bar': 'baz'},
        )

    action = ai.registry.register_action(name='test_action', kind='custom', fn=action_fn)
    action_response = await action.arun(context={'foo': 'bar'})

    assert action_response.response.text == '''[ECHO] user: "hi {'bar': 'baz'}"'''


@pytest.mark.asyncio
async def test_generate_json_format_unconstrained_with_instructions(
    setup_test: SetupFixture,
) -> None:
    """When output_schema is provided, format will default to json."""
    ai, *_ = setup_test

    class TestSchema(BaseModel):
        foo: int = Field(None, description='foo field')
        bar: str = Field(None, description='bar field')

    want = GenerateRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(text='hi'),
                    Part(
                        text='Output should be in JSON format and conform to the following schema:\n\n```\n{\n  "properties": {\n    "foo": {\n      "default": null,\n      "description": "foo field",\n      "title": "Foo",\n      "type": "integer"\n    },\n    "bar": {\n      "default": null,\n      "description": "bar field",\n      "title": "Bar",\n      "type": "string"\n    }\n  },\n  "title": "TestSchema",\n  "type": "object"\n}\n```\n',
                        metadata=Metadata(root={'purpose': 'output'}),
                    ),
                ],
            )
        ],
        config={},
        tools=[],
        output=OutputConfig(
            format='json',
            schema_={
                'properties': {
                    'foo': {
                        'default': None,
                        'description': 'foo field',
                        'title': 'Foo',
                        'type': 'integer',
                    },
                    'bar': {
                        'default': None,
                        'description': 'bar field',
                        'title': 'Bar',
                        'type': 'string',
                    },
                },
                'title': 'TestSchema',
                'type': 'object',
            },
            constrained=False,
            content_type='application/json',
        ),
    )

    response = await ai.generate(
        model='echoModel',
        prompt='hi',
        output_schema=TestSchema,
        output_instructions=True,
        output_constrained=False,
    )

    assert response.request == want

    _, response = ai.generate_stream(
        model='echoModel',
        prompt='hi',
        output_schema=TestSchema,
        output_instructions=True,
        output_constrained=False,
    )

    assert (await response).request == want


@pytest.mark.asyncio
async def test_generate_simulates_doc_grounding(
    setup_test: SetupFixture,
) -> None:
    """Test that generate simulates doc grounding."""
    ai, *_ = setup_test

    want_msg = Message(
        role=Role.USER,
        content=[
            Part(text='hi'),
            Part(
                text='\n\nUse the following information to complete your task:' + '\n\n- [0]: doc content 1\n\n',
                metadata=Metadata(root={'purpose': 'context'}),
            ),
        ],
    )

    response = await ai.generate(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(text='hi')],
            ),
        ],
        docs=[DocumentData(content=[DocumentPart(text='doc content 1')])],
    )

    assert response.request.messages[0] == want_msg

    _, response = ai.generate_stream(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(text='hi')],
            ),
        ],
        docs=[DocumentData(content=[DocumentPart(text='doc content 1')])],
    )

    assert (await response).request.messages[0] == want_msg


class TestFormat(FormatDef):
    """Test format for testing the format."""

    def __init__(self):
        """Initialize the format."""
        super().__init__(
            'banana',
            FormatterConfig(
                format='json',
                content_type='application/banana',
                constrained=True,
            ),
        )

    def handle(self, schema) -> Formatter:
        """Handle the format."""

        def message_parser(msg: Message):
            """Parse the message."""
            return f'banana {"".join(p.root.text for p in msg.content)}'

        def chunk_parser(chunk: GenerateResponseChunk) -> str:
            """Parse the chunk."""
            return f'banana chunk {"".join(p.root.text for p in chunk.content)}'

        instructions: str | None

        if schema:
            instructions = f'schema: {json.dumps(schema)}'

        return Formatter(
            chunk_parser=chunk_parser,
            message_parser=message_parser,
            instructions=instructions,
        )


@pytest.mark.asyncio
async def test_define_format(setup_test: SetupFixture) -> None:
    """Test that the define format function works."""
    ai, _, pm, *_ = setup_test

    ai.define_format(TestFormat())

    class TestSchema(BaseModel):
        foo: int = Field(None, description='foo field')
        bar: str = Field(None, description='bar field')

    pm.responses = [
        (
            GenerateResponse(
                finishReason=FinishReason.STOP,
                message=Message(role=Role.MODEL, content=[Part(text='model says')]),
            )
        )
    ]
    pm.chunks = [
        [
            GenerateResponseChunk(role='model', content=[Part(text='1')]),
            GenerateResponseChunk(role='model', content=[Part(text='2')]),
            GenerateResponseChunk(role='model', content=[Part(text='3')]),
        ]
    ]

    chunks = []

    stream, aresponse = ai.generate_stream(
        model='programmableModel',
        prompt='hi',
        output_format='banana',
        output_schema=TestSchema,
    )

    async for chunk in stream:
        chunks.append(chunk.output)

    response = await aresponse

    assert response.output == 'banana model says'
    assert chunks == ['banana chunk 1', 'banana chunk 2', 'banana chunk 3']

    assert response.request == GenerateRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(text='hi'),
                    Part(
                        text='schema: {"properties": {"foo": {"default": null, "description": "foo field", "title": "Foo", "type": "integer"}, "bar": {"default": null, "description": "bar field", "title": "Bar", "type": "string"}}, "title": "TestSchema", "type": "object"}',
                        metadata=Metadata(root={'purpose': 'output'}),
                    ),
                ],
            ),
        ],
        config={},
        tools=[],
        output=OutputConfig(
            format='json',
            schema_={
                'properties': {
                    'foo': {
                        'default': None,
                        'description': 'foo field',
                        'title': 'Foo',
                        'type': 'integer',
                    },
                    'bar': {
                        'default': None,
                        'description': 'bar field',
                        'title': 'Bar',
                        'type': 'string',
                    },
                },
                'title': 'TestSchema',
                'type': 'object',
            },
            # these get populated by the format
            constrained=True,
            content_type='application/banana',
        ),
    )


def test_define_model_default_metadata(setup_test: SetupFixture) -> None:
    """Test that the define model function works."""
    ai, _, _, *_ = setup_test

    def foo_model_fn():
        return GenerateResponse(message=Message(role=Role.MODEL, content=[Part(text='banana!')]))

    action = ai.define_model(
        name='foo',
        fn=foo_model_fn,
    )

    assert action.metadata['model'] == {
        'label': 'foo',
    }


def test_define_model_with_schema(setup_test: SetupFixture) -> None:
    """Test that the define model function with schema works."""
    ai, _, _, *_ = setup_test

    class Config(BaseModel):
        field_a: str = Field(description='a field')
        field_b: str = Field(description='b field')

    def foo_model_fn():
        return GenerateResponse(message=Message(role=Role.MODEL, content=[Part(text='banana!')]))

    action = ai.define_model(
        name='foo',
        fn=foo_model_fn,
        config_schema=Config,
    )
    assert action.metadata['model'] == {
        'customOptions': {
            'properties': {
                'field_a': {
                    'description': 'a field',
                    'title': 'Field A',
                    'type': 'string',
                },
                'field_b': {
                    'description': 'b field',
                    'title': 'Field B',
                    'type': 'string',
                },
            },
            'required': [
                'field_a',
                'field_b',
            ],
            'title': 'Config',
            'type': 'object',
        },
        'label': 'foo',
    }


def test_define_model_with_info(setup_test: SetupFixture) -> None:
    """Test that the define model function with info works."""
    ai, _, _, *_ = setup_test

    def foo_model_fn():
        return GenerateResponse(message=Message(role=Role.MODEL, content=[Part(text='banana!')]))

    action = ai.define_model(
        name='foo',
        fn=foo_model_fn,
        info=ModelInfo(label='Foo Bar', supports=Supports(multiturn=True, tools=True)),
    )
    assert action.metadata['model'] == {
        'label': 'Foo Bar',
        'supports': {
            'multiturn': True,
            'tools': True,
        },
    }


def test_define_retriever_default_metadata(setup_test: SetupFixture) -> None:
    """Test that the define retriever function works."""
    ai, _, _, *_ = setup_test

    def my_retriever(request: RetrieverRequest, ctx: ActionRunContext):
        return RetrieverResponse(documents=[Document.from_text('Hello'), Document.from_text('World')])

    action = ai.define_retriever(
        name='fooRetriever',
        fn=my_retriever,
    )

    assert action.metadata['retriever'] == {
        'label': 'fooRetriever',
    }


def test_define_retriever_with_schema(setup_test: SetupFixture) -> None:
    """Test that the define retriever function with schema works."""
    ai, _, _, *_ = setup_test

    class Config(BaseModel):
        field_a: str = Field(description='a field')
        field_b: str = Field(description='b field')

    def my_retriever(request: RetrieverRequest, ctx: ActionRunContext):
        return RetrieverResponse(documents=[Document.from_text('Hello'), Document.from_text('World')])

    action = ai.define_retriever(
        name='fooRetriever',
        fn=my_retriever,
        config_schema=Config,
    )

    assert action.metadata['retriever'] == {
        'customOptions': {
            'properties': {
                'field_a': {
                    'description': 'a field',
                    'title': 'Field A',
                    'type': 'string',
                },
                'field_b': {
                    'description': 'b field',
                    'title': 'Field B',
                    'type': 'string',
                },
            },
            'required': [
                'field_a',
                'field_b',
            ],
            'title': 'Config',
            'type': 'object',
        },
        'label': 'fooRetriever',
    }


def test_define_evaluator_simple(setup_test: SetupFixture) -> None:
    """Test that the define evaluator function works."""
    ai, _, _, *_ = setup_test

    def my_eval_fn(datapoint: BaseEvalDataPoint, options: Any | None):
        return EvalFnResponse(
            test_case_id=datapoint.test_case_id,
            evaluation=Score(score=True, details=Details(reasoning='I think it is true')),
        )

    action = ai.define_evaluator(
        name='my_eval',
        display_name='Test evaluator',
        definition='Test evaluator that always returns True',
        fn=my_eval_fn,
    )

    assert action.metadata['evaluator'] == {
        'label': 'my_eval',
        'evaluatorDefinition': 'Test evaluator that always returns True',
        'evaluatorDisplayName': 'Test evaluator',
        'evaluatorIsBilled': False,
    }


def test_define_evaluator_custom_config(setup_test: SetupFixture) -> None:
    """Test that the define evaluator function works."""
    ai, _, _, *_ = setup_test

    class CustomOption(BaseModel):
        foo_bar: str = Field('baz', description='foo_bar field')

    def my_eval_fn(datapoint: BaseEvalDataPoint, options: CustomOption | None):
        return EvalFnResponse(
            test_case_id=datapoint.test_case_id,
            evaluation=Score(score=True, details=Details(reasoning=options.foo_bar)),
        )

    action = ai.define_evaluator(
        name='my_eval',
        display_name='Test evaluator',
        definition='Test evaluator that always returns True',
        fn=my_eval_fn,
        config_schema=CustomOption,
    )

    assert action.metadata['evaluator'] == {
        'label': 'my_eval',
        'evaluatorDefinition': 'Test evaluator that always returns True',
        'evaluatorDisplayName': 'Test evaluator',
        'evaluatorIsBilled': False,
        'customOptions': {
            'properties': {
                'foo_bar': {
                    'default': 'baz',
                    'description': 'foo_bar field',
                    'title': 'Foo Bar',
                    'type': 'string',
                }
            },
            'title': 'CustomOption',
            'type': 'object',
        },
    }


def test_define_batch_evaluator(setup_test: SetupFixture) -> None:
    """Test that the define batch evaluator function works."""
    ai, _, _, *_ = setup_test

    def my_eval_fn(req: EvalRequest, options: Any | None):
        eval_responses: list[EvalFnResponse] = []
        for index in range(len(req.dataset)):
            datapoint = req.dataset[index]
            eval_responses.append(
                EvalFnResponse(
                    test_case_id=f'testCase{index}',
                    evaluation=Score(
                        score=True,
                        details=Details(reasoning=f'I think {datapoint.input} is true'),
                    ),
                )
            )

        return EvalResponse(eval_responses)

    action = ai.define_batch_evaluator(
        name='my_eval',
        display_name='Test evaluator',
        definition='Test evaluator that always returns True',
        fn=my_eval_fn,
    )

    assert action.metadata['evaluator'] == {
        'label': 'my_eval',
        'evaluatorDefinition': 'Test evaluator that always returns True',
        'evaluatorDisplayName': 'Test evaluator',
        'evaluatorIsBilled': False,
    }


@pytest.mark.asyncio
async def test_define_sync_flow(setup_test: SetupFixture) -> None:
    ai, _, _, *_ = setup_test

    @ai.flow()
    def my_flow(input: str, ctx):
        ctx.send_chunk(1)
        ctx.send_chunk(2)
        ctx.send_chunk(3)
        return input

    assert my_flow('banana') == 'banana'

    stream, response = my_flow.stream('banana2')

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert chunks == [1, 2, 3]
    assert (await response) == 'banana2'


@pytest.mark.asyncio
async def test_define_async_flow(setup_test: SetupFixture) -> None:
    ai, _, _, *_ = setup_test

    @ai.flow()
    async def my_flow(input: str, ctx):
        ctx.send_chunk(1)
        ctx.send_chunk(2)
        ctx.send_chunk(3)
        return input

    assert (await my_flow('banana')) == 'banana'

    stream, response = my_flow.stream('banana2')

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    assert chunks == [1, 2, 3]
    assert (await response) == 'banana2'
