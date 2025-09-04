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

"""Unit tests for Vectorstore Plugin."""

from unittest.mock import MagicMock

import pytest

from genkit.ai import Genkit
from genkit.core.action.types import ActionKind


@pytest.mark.parametrize(
    'kind, name',
    [
        (ActionKind.INDEXER, 'test_indexer'),
        (ActionKind.RETRIEVER, 'test_retriever'),
    ],
)
def test_action_resolve(kind, name, vectorstore_plugin_instance):
    """Test initialize method of Vectorstore plugin."""
    ai_mock = MagicMock(spec=Genkit)
    assert hasattr(vectorstore_plugin_instance, "resolve_action")

    if kind == ActionKind.RETRIEVER:
        vectorstore_plugin_instance._configure_dev_local_retriever = MagicMock()

        assert vectorstore_plugin_instance.resolve_action(ai_mock, kind, name) is None

        vectorstore_plugin_instance._configure_dev_local_retriever.assert_called_once_with(
            ai_mock,
            name
        )
    else:
        vectorstore_plugin_instance._configure_dev_local_indexer = MagicMock()

        assert vectorstore_plugin_instance.resolve_action(ai_mock, kind, name) is None

        vectorstore_plugin_instance._configure_dev_local_indexer.assert_called_once_with(
            ai_mock,
            name
        )
