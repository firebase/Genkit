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

"""Local file-based vectorstore plugin that provides retriever and indexer for Genkit."""

from typing import Any

from genkit.ai import GenkitRegistry, Plugin
from genkit.core.action import Action
from genkit.core.registry import ActionKind
from genkit.types import Docs

from .indexer import (
    DevLocalVectorStoreIndexer,
)
from .retriever import (
    DevLocalVectorStoreRetriever,
    RetrieverOptionsSchema,
)


class DevLocalVectorStore(Plugin):
    """Local file-based vectorstore plugin that provides retriever and indexer.

    NOT INTENDED FOR USE IN PRODUCTION
    """

    name = 'devLocalVectorstore'
    _indexers: dict[str, DevLocalVectorStoreIndexer] = {}

    def __init__(self, name: str, embedder: str, embedder_options: dict[str, Any] | None = None):
        self.index_name = name
        self.embedder = embedder
        self.embedder_options = embedder_options

    def initialize(self, ai: GenkitRegistry) -> None:
        """Initialize the plugin by registering actions with the registry.

        This method registers the Local Vector Store actions with the provided
        registry, making them available for use in the Genkit framework.

        Args:
            ai: The registry to register actions with.

        Returns:
            None
        """
        self._configure_dev_local_retriever(ai=ai)
        self._configure_dev_local_indexer(ai=ai)

    def resolve_action(
        self,
        ai: GenkitRegistry,
        kind: ActionKind,
        name: str,
    ) -> None:

        if kind is ActionKind.RETRIEVER:
            self._configure_dev_local_retriever(ai, name)
            return None

        if kind is ActionKind.INDEXER:
            self._configure_dev_local_indexer(ai, name)


        return None

    def _configure_dev_local_retriever(self, ai: GenkitRegistry, name: str | None = None) -> Action:
        """Registers Local Vector Store retriever for provided parameters.

        Args:
            ai: The registry to register retriever with.
            params: Parameters to register retriever with.

        Returns:
            registered Action instance
        """
        retriever = DevLocalVectorStoreRetriever(
            ai=ai,
            index_name=name or self.index_name,
            embedder=self.embedder,
            embedder_options=self.embedder_options,
        )

        return ai.define_retriever(
            name=self.index_name,
            config_schema=RetrieverOptionsSchema,
            fn=retriever.retrieve,
        )

    def _configure_dev_local_indexer(self, ai: GenkitRegistry, name: str | None = None) -> Action:
        """Registers Local Vector Store indexer for provided parameters.

        Args:
            ai: The registry to register indexer with.
            params: Parameters to register indexer with.

        Returns:
            registered Action instance
        """
        indexer = DevLocalVectorStoreIndexer(
            ai=ai,
            index_name= name or self.index_name,
            embedder=self.embedder,
            embedder_options=self.embedder_options,
        )

        DevLocalVectorStore._indexers[self.index_name] = indexer




    @classmethod
    async def index(cls, index_name: str, documents: Docs) -> None:
        """Lookups the Local Vector Store indexer for provided index name.

        If matching indexer found - invokes indexing for provided documents

        Args:
            index_name: name of the indexer to look up
            documents: list of documents to index

        Returns:
            None

        Raises:
            KeyError: if index name is not found among registered indexers.
        """
        matching_indexer = cls._indexers.get(index_name)
        if not matching_indexer:
            raise KeyError(
                f'Failed to find indexer matching name: {index_name}!r\nRegistered indexers: {cls._indexers.keys()}'
            )
        return await matching_indexer.index(documents)
