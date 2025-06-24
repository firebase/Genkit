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

"""Unit tests for Ollama models package."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import ollama as ollama_api
import pytest

from genkit.plugins.ollama.constants import OllamaAPITypes
from genkit.plugins.ollama.models import ModelDefinition, OllamaModel, _convert_parameters
from genkit.types import (
    ActionRunContext,
    GenerateRequest,
    GenerateResponseChunk,
    GenerationUsage,
    Message,
    OutputConfig,
    Role,
    TextPart,
)


class TestOllamaModelGenerate(unittest.IsolatedAsyncioTestCase):
    """Tests for Generate method of OllamaModel."""

    async def asyncSetUp(self):
        """Common setup for all async tests."""
        self.mock_client = MagicMock()
        self.request = GenerateRequest(messages=[Message(role=Role.USER, content=[TextPart(text='Hello')])])
        self.ctx = ActionRunContext()

    @patch(
        'genkit.blocks.model.get_basic_usage_stats',
        return_value=GenerationUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        ),
    )
    async def test_generate_chat_non_streaming(self, mock_get_basic_usage_stats):
        """Test generate method with CHAT API type in non-streaming mode."""
        model_def = ModelDefinition(
            name='chat-model',
            api_type=OllamaAPITypes.CHAT,
        )
        ollama_model = OllamaModel(
            client=self.mock_client,
            model_definition=model_def,
        )

        # Mock internal methods
        mock_chat_response = ollama_api.ChatResponse(
            message=ollama_api.Message(
                role='',
                content='Generated chat text',
            ),
        )
        ollama_model._chat_with_ollama = AsyncMock(
            return_value=mock_chat_response,
        )
        ollama_model._generate_ollama_response = AsyncMock()
        ollama_model._build_multimodal_chat_response = MagicMock(
            return_value=[TextPart(text='Parsed chat content')],
        )
        ollama_model.get_usage_info = MagicMock(
            return_value=GenerationUsage(
                input_tokens=5,
                output_tokens=10,
                total_tokens=15,
            ),
        )
        ollama_model.is_streaming_request = MagicMock(return_value=False)

        response = await ollama_model.generate(self.request, self.ctx)

        # Assertions
        ollama_model._chat_with_ollama.assert_awaited_once_with(request=self.request, ctx=self.ctx)
        ollama_model._generate_ollama_response.assert_not_awaited()
        ollama_model._build_multimodal_chat_response.assert_called_once_with(chat_response=mock_chat_response)
        ollama_model.is_streaming_request.assert_called_once_with(ctx=self.ctx)
        ollama_model.get_usage_info.assert_called_once()

        self.assertEqual(response.message.role, Role.MODEL)
        self.assertEqual(len(response.message.content), 1)
        self.assertEqual(response.message.content[0].root.text, 'Parsed chat content')
        self.assertEqual(response.usage.input_tokens, 5)
        self.assertEqual(response.usage.output_tokens, 10)

    @patch(
        'genkit.blocks.model.get_basic_usage_stats',
        return_value=GenerationUsage(
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        ),
    )
    async def test_generate_generate_non_streaming(self, mock_get_basic_usage_stats):
        """Test generate method with GENERATE API type in non-streaming mode."""
        model_def = ModelDefinition(
            name='generate-model',
            api_type=OllamaAPITypes.GENERATE,
        )
        ollama_model = OllamaModel(
            client=self.mock_client,
            model_definition=model_def,
        )

        # Mock internal methods
        mock_generate_response = ollama_api.GenerateResponse(
            response='Generated text',
        )
        ollama_model._generate_ollama_response = AsyncMock(
            return_value=mock_generate_response,
        )
        ollama_model._chat_with_ollama = AsyncMock()
        ollama_model.is_streaming_request = MagicMock(return_value=False)
        ollama_model.get_usage_info = MagicMock(
            return_value=GenerationUsage(
                input_tokens=7,
                output_tokens=14,
                total_tokens=21,
            ),
        )

        response = await ollama_model.generate(self.request, self.ctx)

        # Assertions
        ollama_model._generate_ollama_response.assert_awaited_once_with(request=self.request, ctx=self.ctx)
        ollama_model._chat_with_ollama.assert_not_called()
        ollama_model.is_streaming_request.assert_called_once_with(ctx=self.ctx)
        ollama_model.get_usage_info.assert_called_once()

        self.assertEqual(response.message.role, Role.MODEL)
        self.assertEqual(len(response.message.content), 1)
        self.assertEqual(response.message.content[0].root.text, 'Generated text')
        self.assertEqual(response.usage.input_tokens, 7)
        self.assertEqual(response.usage.output_tokens, 14)

    @patch(
        'genkit.blocks.model.get_basic_usage_stats',
        return_value=GenerationUsage(),
    )
    async def test_generate_chat_streaming(self, mock_get_basic_usage_stats):
        """Test generate method with CHAT API type in streaming mode."""
        model_def = ModelDefinition(name='chat-model', api_type=OllamaAPITypes.CHAT)
        ollama_model = OllamaModel(client=self.mock_client, model_definition=model_def)
        streaming_ctx = ActionRunContext(on_chunk=True)

        # Mock internal methods
        mock_chat_response = ollama_api.ChatResponse(
            message=ollama_api.Message(
                role='',
                content='Generated chat text',
            ),
        )
        ollama_model._chat_with_ollama = AsyncMock(
            return_value=mock_chat_response,
        )
        ollama_model._build_multimodal_chat_response = MagicMock(
            return_value=[TextPart(text='Parsed chat content')],
        )
        ollama_model.is_streaming_request = MagicMock(return_value=True)
        ollama_model.get_usage_info = MagicMock(
            return_value=GenerationUsage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            ),
        )

        response = await ollama_model.generate(self.request, streaming_ctx)

        # Assertions for streaming behavior
        ollama_model._chat_with_ollama.assert_awaited_once_with(
            request=self.request,
            ctx=streaming_ctx,
        )
        ollama_model.is_streaming_request.assert_called_once_with(
            ctx=streaming_ctx,
        )
        self.assertEqual(response.message.content, [])

    @patch(
        'genkit.blocks.model.get_basic_usage_stats',
        return_value=GenerationUsage(),
    )
    async def test_generate_generate_streaming(self, mock_get_basic_usage_stats):
        """Test generate method with GENERATE API type in streaming mode."""
        model_def = ModelDefinition(
            name='generate-model',
            api_type=OllamaAPITypes.GENERATE,
        )
        ollama_model = OllamaModel(client=self.mock_client, model_definition=model_def)
        streaming_ctx = ActionRunContext(on_chunk=True)

        # Mock internal methods
        mock_generate_response = ollama_api.GenerateResponse(
            response='Generated text',
        )
        ollama_model._generate_ollama_response = AsyncMock(
            return_value=mock_generate_response,
        )
        ollama_model.is_streaming_request = MagicMock(return_value=True)
        ollama_model.get_usage_info = MagicMock(
            return_value=GenerationUsage(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
            ),
        )

        response = await ollama_model.generate(self.request, streaming_ctx)

        # Assertions for streaming behavior
        ollama_model._generate_ollama_response.assert_awaited_once_with(
            request=self.request,
            ctx=streaming_ctx,
        )
        ollama_model.is_streaming_request.assert_called_once_with(
            ctx=streaming_ctx,
        )
        self.assertEqual(response.message.content, [])

    @patch(
        'genkit.blocks.model.get_basic_usage_stats',
        return_value=GenerationUsage(),
    )
    async def test_generate_chat_api_response_none(self, mock_get_basic_usage_stats):
        """Test generate method when _chat_with_ollama returns None."""
        model_def = ModelDefinition(name='chat-model', api_type=OllamaAPITypes.CHAT)
        ollama_model = OllamaModel(client=self.mock_client, model_definition=model_def)

        ollama_model._chat_with_ollama = AsyncMock(return_value=None)
        ollama_model._build_multimodal_chat_response = MagicMock()
        ollama_model.is_streaming_request = MagicMock(return_value=False)
        ollama_model.get_usage_info = MagicMock(return_value=GenerationUsage())

        response = await ollama_model.generate(self.request, self.ctx)

        ollama_model._chat_with_ollama.assert_awaited_once()
        ollama_model._build_multimodal_chat_response.assert_not_called()
        self.assertEqual(response.message.content[0].root.text, 'Failed to get response from Ollama API')
        self.assertEqual(response.usage.input_tokens, None)
        self.assertEqual(response.usage.output_tokens, None)

    @patch(
        'genkit.blocks.model.get_basic_usage_stats',
        return_value=GenerationUsage(),
    )
    async def test_generate_generate_api_response_none(self, mock_get_basic_usage_stats):
        """Test generate method when _generate_ollama_response returns None."""
        model_def = ModelDefinition(name='generate-model', api_type=OllamaAPITypes.GENERATE)
        ollama_model = OllamaModel(client=self.mock_client, model_definition=model_def)

        ollama_model._generate_ollama_response = AsyncMock(return_value=None)
        ollama_model.is_streaming_request = MagicMock(return_value=False)
        ollama_model.get_usage_info = MagicMock(return_value=GenerationUsage())

        response = await ollama_model.generate(self.request, self.ctx)

        ollama_model._generate_ollama_response.assert_awaited_once()
        self.assertEqual(response.message.content[0].root.text, 'Failed to get response from Ollama API')
        self.assertEqual(response.usage.input_tokens, None)
        self.assertEqual(response.usage.output_tokens, None)


class TestOllamaModelChatWithOllama(unittest.IsolatedAsyncioTestCase):
    """Unit tests for OllamaModel._chat_with_ollama method."""

    async def asyncSetUp(self):
        """Common setup."""
        self.mock_ollama_client_instance = AsyncMock()
        self.mock_ollama_client_factory = MagicMock(return_value=self.mock_ollama_client_instance)
        self.model_definition = ModelDefinition(name='test-chat-model', api_type=OllamaAPITypes.CHAT)
        self.ollama_model = OllamaModel(client=self.mock_ollama_client_factory, model_definition=self.model_definition)
        self.request = GenerateRequest(messages=[Message(role=Role.USER, content=[TextPart(text='Hello')])])
        self.ctx = ActionRunContext(on_chunk=False)
        self.ctx.send_chunk = MagicMock()

        # Mocking methods of ollama_model that are called
        self.ollama_model.build_chat_messages = MagicMock(return_value=[{}])
        self.ollama_model.is_streaming_request = MagicMock(
            return_value=False,
        )
        self.ollama_model.build_request_options = MagicMock(
            return_value={'temperature': 0.7},
        )
        self.ollama_model._build_multimodal_chat_response = MagicMock(
            return_value=[TextPart(text='mocked content')],
        )

        self.mock_convert_parameters = MagicMock(return_value={'type': 'string'})

    async def test_non_streaming_chat_success(self):
        """Test _chat_with_ollama in non-streaming mode with successful response."""
        expected_response = ollama_api.ChatResponse(
            message=ollama_api.Message(
                role='',
                content='Ollama non-stream response',
            ),
        )
        self.mock_ollama_client_instance.chat.return_value = expected_response

        response = await self.ollama_model._chat_with_ollama(self.request, self.ctx)

        self.assertIsNotNone(response)
        self.assertEqual(response.message.content, 'Ollama non-stream response')
        self.ollama_model.build_chat_messages.assert_called_once_with(self.request)
        self.ollama_model.is_streaming_request.assert_called_once_with(ctx=self.ctx)
        self.ctx.send_chunk.assert_not_called()
        self.mock_ollama_client_instance.chat.assert_awaited_once_with(
            model=self.model_definition.name,
            messages=self.ollama_model.build_chat_messages.return_value,
            tools=[],
            options=self.ollama_model.build_request_options.return_value,
            format='',
            stream=False,
        )

        self.ollama_model._build_multimodal_chat_response.assert_not_called()

    async def test_streaming_chat_success(self):
        """Test _chat_with_ollama in streaming mode with multiple chunks."""
        self.ollama_model.is_streaming_request.return_value = True
        self.ctx.is_streaming = True

        # Simulate an async iterator of chunks
        async def mock_streaming_chunks():
            yield ollama_api.ChatResponse(
                message=ollama_api.Message(
                    role='',
                    content='chunk1',
                ),
            )
            yield ollama_api.ChatResponse(
                message=ollama_api.Message(
                    role='',
                    content='chunk2',
                ),
            )

        self.mock_ollama_client_instance.chat.return_value = mock_streaming_chunks()

        response = await self.ollama_model._chat_with_ollama(self.request, self.ctx)

        self.assertIsNotNone(response)
        self.ollama_model.build_chat_messages.assert_called_once_with(self.request)
        self.ollama_model.is_streaming_request.assert_called_once_with(ctx=self.ctx)
        self.mock_ollama_client_instance.chat.assert_awaited_once_with(
            model=self.model_definition.name,
            messages=self.ollama_model.build_chat_messages.return_value,
            tools=[],
            options=self.ollama_model.build_request_options.return_value,
            format='',
            stream=True,
        )
        self.assertEqual(self.ctx.send_chunk.call_count, 2)
        self.assertEqual(self.ollama_model._build_multimodal_chat_response.call_count, 2)
        self.ctx.send_chunk.assert_any_call(chunk=unittest.mock.ANY)
        self.ollama_model._build_multimodal_chat_response.assert_any_call(chat_response=unittest.mock.ANY)

    async def test_chat_with_output_format_string(self):
        """Test _chat_with_ollama with request.output.format string."""
        self.request.output = OutputConfig(format='json')

        expected_response = ollama_api.ChatResponse(
            message=ollama_api.Message(
                role='',
                content='json output',
            ),
        )
        self.mock_ollama_client_instance.chat.return_value = expected_response

        await self.ollama_model._chat_with_ollama(self.request, self.ctx)

        call_args, call_kwargs = self.mock_ollama_client_instance.chat.call_args
        self.assertIn('format', call_kwargs)
        self.assertEqual(call_kwargs['format'], 'json')

    async def test_chat_with_output_format_schema(self):
        """Test _chat_with_ollama with request.output.schema_ dictionary."""
        schema_dict = {'type': 'object', 'properties': {'name': {'type': 'string'}}}
        self.request.output = OutputConfig(schema_=schema_dict)

        expected_response = ollama_api.ChatResponse(
            message=ollama_api.Message(
                role='',
                content='schema output',
            ),
        )
        self.mock_ollama_client_instance.chat.return_value = expected_response

        await self.ollama_model._chat_with_ollama(self.request, self.ctx)

        call_args, call_kwargs = self.mock_ollama_client_instance.chat.call_args
        self.assertIn('format', call_kwargs)
        self.assertEqual(call_kwargs['format'], schema_dict)

    async def test_chat_with_no_output_format(self):
        """Test _chat_with_ollama with no output format specified."""
        self.request.output = OutputConfig(format=None, schema_=None)

        expected_response = ollama_api.ChatResponse(
            message=ollama_api.Message(
                role='',
                content='normal output',
            ),
        )
        self.mock_ollama_client_instance.chat.return_value = expected_response

        await self.ollama_model._chat_with_ollama(self.request, self.ctx)

        call_args, call_kwargs = self.mock_ollama_client_instance.chat.call_args
        self.assertIn('format', call_kwargs)
        self.assertEqual(call_kwargs['format'], '')

    async def test_chat_api_raises_exception(self):
        """Test _chat_with_ollama handles exception from client.chat."""
        self.mock_ollama_client_instance.chat.side_effect = Exception('Ollama API Error')

        with self.assertRaisesRegex(Exception, 'Ollama API Error'):
            await self.ollama_model._chat_with_ollama(self.request, self.ctx)

        self.mock_ollama_client_instance.chat.assert_awaited_once()
        self.ctx.send_chunk.assert_not_called()


class TestOllamaModelGenerateOllamaResponse(unittest.IsolatedAsyncioTestCase):
    """Unit tests for OllamaModel._generate_ollama_response."""

    async def asyncSetUp(self):
        """Common setup."""
        self.mock_ollama_client_instance = AsyncMock()
        self.mock_ollama_client_factory = MagicMock(return_value=self.mock_ollama_client_instance)

        self.model_definition = ModelDefinition(name='test-generate-model', api_type=OllamaAPITypes.GENERATE)
        self.ollama_model = OllamaModel(client=self.mock_ollama_client_factory, model_definition=self.model_definition)
        self.request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[TextPart(text='Test generate message')],
                )
            ],
            config={'temperature': 0.8},
        )
        self.ctx = ActionRunContext(on_chunk=False)
        self.ctx.send_chunk = MagicMock()

        # Patch internal methods of OllamaModel that _generate_ollama_response calls
        self.ollama_model.build_prompt = MagicMock(return_value='Mocked prompt from build_prompt')
        self.ollama_model.is_streaming_request = MagicMock(return_value=False)
        self.ollama_model.build_request_options = MagicMock(return_value={'temperature': 0.8})

    async def test_non_streaming_generate_success(self):
        """Test _generate_ollama_response in non-streaming mode with successful response."""
        expected_response = ollama_api.GenerateResponse(response='Full generated text')
        self.mock_ollama_client_instance.generate.return_value = expected_response

        response = await self.ollama_model._generate_ollama_response(self.request, self.ctx)

        self.assertIsNotNone(response)
        self.assertEqual(response.response, 'Full generated text')

        self.ollama_model.build_prompt.assert_called_once_with(self.request)
        self.ollama_model.is_streaming_request.assert_called_once_with(ctx=self.ctx)
        self.ollama_model.build_request_options.assert_called_once_with(config=self.request.config)
        self.mock_ollama_client_instance.generate.assert_awaited_once_with(
            model=self.model_definition.name,
            prompt=self.ollama_model.build_prompt.return_value,
            options=self.ollama_model.build_request_options.return_value,
            stream=False,
        )
        self.ctx.send_chunk.assert_not_called()

    async def test_streaming_generate_success(self):
        """Test _generate_ollama_response in streaming mode with multiple chunks."""
        self.ollama_model.is_streaming_request.return_value = True

        # Simulate an async iterator of chunks
        async def mock_streaming_chunks():
            yield ollama_api.GenerateResponse(response='chunk1 ')
            yield ollama_api.GenerateResponse(response='chunk2')

        self.mock_ollama_client_instance.generate.return_value = mock_streaming_chunks()

        response = await self.ollama_model._generate_ollama_response(self.request, self.ctx)

        self.assertIsNotNone(response)

        self.ollama_model.build_prompt.assert_called_once_with(self.request)
        self.ollama_model.is_streaming_request.assert_called_once_with(ctx=self.ctx)
        self.mock_ollama_client_instance.generate.assert_awaited_once_with(
            model=self.model_definition.name,
            prompt=self.ollama_model.build_prompt.return_value,
            options=self.ollama_model.build_request_options.return_value,
            stream=True,
        )
        self.assertEqual(self.ctx.send_chunk.call_count, 2)
        self.ctx.send_chunk.assert_any_call(
            chunk=GenerateResponseChunk(role=Role.MODEL, index=1, content=[TextPart(text='chunk1 ')])
        )
        self.ctx.send_chunk.assert_any_call(
            chunk=GenerateResponseChunk(role=Role.MODEL, index=2, content=[TextPart(text='chunk2')])
        )

    async def test_generate_api_raises_exception(self):
        """Test _generate_ollama_response handles exception from client.generate."""
        self.mock_ollama_client_instance.generate.side_effect = Exception('Ollama generate API Error')

        with self.assertRaisesRegex(Exception, 'Ollama generate API Error'):
            await self.ollama_model._generate_ollama_response(self.request, self.ctx)

        self.mock_ollama_client_instance.generate.assert_awaited_once()
        self.ctx.send_chunk.assert_not_called()


@pytest.mark.parametrize(
    'input_schema, expected_output',
    [
        ({}, None),
        ({'properties': {'name': {'type': 'string'}}}, None),
        (
            {'type': 'object'},
            ollama_api.Tool.Function.Parameters(type='object', properties={}),
        ),
        (
            {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'description': 'User name'},
                    'age': {'type': 'integer', 'description': 'User age'},
                },
                'required': ['name'],
            },
            ollama_api.Tool.Function.Parameters(
                type='object',
                required=['name'],
                properties={
                    'name': ollama_api.Tool.Function.Parameters.Property(type='string', description='User name'),
                    'age': ollama_api.Tool.Function.Parameters.Property(type='integer', description='User age'),
                },
            ),
        ),
        (
            {
                'type': 'object',
                'properties': {
                    'city': {'type': 'string', 'description': 'City name'},
                },
            },
            ollama_api.Tool.Function.Parameters(
                type='object',
                required=None,
                properties={
                    'city': ollama_api.Tool.Function.Parameters.Property(
                        type='string',
                        description='City name',
                    ),
                },
            ),
        ),
        (
            {
                'type': 'object',
                'properties': {},
            },
            ollama_api.Tool.Function.Parameters(
                type='object',
                required=None,
                properties={},
            ),
        ),
        # Test 8: Object schema with nested properties
        (
            {
                'type': 'object',
                'properties': {
                    'address': {'type': 'object', 'properties': {'street': {'type': 'string'}}},
                    'zip': {'type': 'string'},
                },
            },
            ollama_api.Tool.Function.Parameters(
                type='object',
                required=None,
                properties={
                    'address': ollama_api.Tool.Function.Parameters.Property(type='object', description=''),
                    'zip': ollama_api.Tool.Function.Parameters.Property(type='string', description=''),
                },
            ),
        ),
        (
            {'type': 'object', 'description': 'A general description'},
            ollama_api.Tool.Function.Parameters(
                type='object',
                required=None,
                properties={},
            ),
        ),
    ],
)
def test_convert_parameters(input_schema, expected_output):
    """Unit Tests for _convert_parameters function with various input schemas."""
    result = _convert_parameters(input_schema)
    assert result == expected_output
