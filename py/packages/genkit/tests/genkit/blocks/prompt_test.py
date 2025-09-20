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


"""Tests for the action module."""

import pytest
from pydantic import BaseModel, Field

from genkit.ai import Genkit
from genkit.core.typing import (
    Message,
    Role,
    TextPart,
    ToolChoice,
)
from genkit.testing import (
    define_echo_model,
    define_programmable_model,
)


def setup_test():
    """Setup a test fixture for the veneer tests."""
    ai = Genkit(model='echoModel')

    pm, _ = define_programmable_model(ai)
    echo, _ = define_echo_model(ai)

    return (ai, echo, pm)


@pytest.mark.asyncio
async def test_simple_prompt() -> None:
    """Test simple prompt rendering."""
    ai, *_ = setup_test()

    want_txt = '[ECHO] user: "hi" {"temperature":11.0}'

    my_prompt = ai.define_prompt(prompt='hi', config={'temperature': 11})

    response = await my_prompt()

    assert response.text == want_txt

    _, response = my_prompt.stream()

    assert (await response).text == want_txt


@pytest.mark.asyncio
async def test_simple_prompt_with_override_config() -> None:
    """Test the config provided at render time is used."""
    ai, *_ = setup_test()

    want_txt = '[ECHO] user: "hi" {"temperature":12.0}'

    my_prompt = ai.define_prompt(prompt='hi', config={'banana': True})

    response = await my_prompt(config={'temperature': 12})

    assert response.text == want_txt

    _, response = my_prompt.stream(config={'temperature': 12})

    assert (await response).text == want_txt


@pytest.mark.asyncio
async def test_prompt_with_system() -> None:
    """Test that the propmt utilises both prompt and system prompt."""
    ai, *_ = setup_test()

    want_txt = '[ECHO] system: "talk like a pirate" user: "hi"'

    my_prompt = ai.define_prompt(prompt='hi', system='talk like a pirate')

    response = await my_prompt()

    assert response.text == want_txt

    _, response = my_prompt.stream()

    assert (await response).text == want_txt


@pytest.mark.asyncio
async def test_prompt_with_kitchensink() -> None:
    """Test that the rendering works with all the options."""
    ai, *_ = setup_test()

    class PromptInput(BaseModel):
        name: str = Field(None, description='the name')

    class ToolInput(BaseModel):
        value: int = Field(None, description='value field')

    @ai.tool(name='testTool')
    def test_tool(input: ToolInput):
        """The tool."""
        return 'abc'

    my_prompt = ai.define_prompt(
        system='pirate',
        prompt='hi',
        messages=[Message(role=Role.USER, content=[TextPart(text='history')])],
        tools=['testTool'],
        tool_choice=ToolChoice.REQUIRED,
        max_turns=5,
        input_schema=PromptInput,
        output_constrained=True,
        output_format='json',
        description='a prompt descr',
    )

    want_txt = '[ECHO] system: "pirate" user: "history" user: "hi" tools=testTool tool_choice=required output={"format":"json","constrained":true,"contentType":"application/json"}'

    response = await my_prompt()

    assert response.text == want_txt

    _, response = my_prompt.stream()

    assert (await response).text == want_txt
