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

"""Conftest for Vectorstore plugin."""

from unittest.mock import patch

import pytest

from genkit.plugins.dev_local_vectorstore import DevLocalVectorStore
from genkit.plugins.google_genai import VertexEmbeddingModels


@pytest.fixture
@patch('ollama.AsyncClient')
def vectorstore_plugin_instance(ollama_async_client):
    """Common instance of ollama plugin."""
    return DevLocalVectorStore(
            name='menu-items',
            embedder=VertexEmbeddingModels.TEXT_EMBEDDING_004_ENG,
            embedder_options={'taskType': 'RETRIEVAL_DOCUMENT'},
        )
