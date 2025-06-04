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

from collections.abc import Callable

from genkit.ai import GenkitRegistry
from genkit.plugins.compat_oai.models import (
    SUPPORTED_OPENAI_COMPAT_MODELS,
    get_default_model_info,
)
from genkit.plugins.compat_oai.models.model import OpenAIModel
from genkit.plugins.compat_oai.typing import OpenAIConfig

from .client import OpenAIClient

OPENAI_COMPAT = 'openai-compat'
MODELGARDEN_PLUGIN_NAME = 'modelgarden'


def model_garden_name(name: str) -> str:
    """Create a Model Garden action name.

    Args:
        name: Base name for the action.

    Returns:
        The fully qualified Model Garden action name.
    """
    return f'{MODELGARDEN_PLUGIN_NAME}/{name}'


class ModelGarden:
    """Manages integration with Google's Model Garden service for Genkit.

    This class provides a convenient way to interact with models hosted on
    Google's Model Garden, allowing them to be exposed as Genkit models
    with OpenAI compatibility. It handles client initialization, model
    information retrieval, and dynamic model definition within the Genkit
    registry.
    """

    def __init__(
        self,
        model: str,
        location: str,
        project_id: str,
        registry: GenkitRegistry,
    ) -> None:
        """Initializes the ModelGarden instance.

        Args:
            model: The name of the specific model to be used from Model Garden
                in the way <publisher>/<model> (e.g., 'meta/llama3.2-pro-max').
            location: The Google Cloud region where the Model Garden service
                is hosted (e.g., 'us-central1').
            project_id: The Google Cloud project ID where the Model Garden
                model is deployed.
            registry: An instance of `GenkitRegistry` to register the
                OpenAI-compatible model with Genkit.
        """
        self.name = model
        self.ai = registry
        openai_params = {'location': location, 'project_id': project_id}
        self.client = OpenAIClient(**openai_params)

    def get_model_info(self) -> dict[str, str] | None:
        """Retrieves metadata and supported features for the specified model.

        This method looks up the model's information from a predefined list
        of supported OpenAI-compatible models or provides default information.

        Returns:
            A dictionary containing the model's 'name' and 'supports' features,
            or None if no information can be found (though typically, a default
            is provided). The 'supports' key contains a dictionary representing
            the model's capabilities (e.g., tools, streaming).
        """
        model_info = SUPPORTED_OPENAI_COMPAT_MODELS.get(self.name, get_default_model_info(self.name))
        return {
            'name': model_info.label,
            'supports': model_info.supports.model_dump(),
        }

    def to_openai_compatible_model(self) -> Callable:
        """Converts the Model Garden model into an OpenAI-compatible Genkit model function.

        This method wraps the underlying Model Garden client and its generation
        logic into a callable that adheres to the OpenAI model interface expected
        by Genkit, specifically providing a `generate` method.

        Returns:
            A callable function (specifically, the `generate` method of an
            `OpenAIModel` instance) that can be used by Genkit.
        """
        openai_model = OpenAIModel(self.name, self.client, self.ai)
        return openai_model.generate

    def define_model(self) -> None:
        """Defines and registers the Model Garden model with the Genkit registry.

        This method orchestrates the retrieval of model metadata and the creation
        of the OpenAI-compatible generation function, then registers this model
        within the Genkit framework using `self.ai.define_model`.
        """
        model_info = self.get_model_info()
        generate_fn = self.to_openai_compatible_model()
        self.ai.define_model(
            name=model_garden_name(self.name),
            fn=generate_fn,
            config_schema=OpenAIConfig,
            metadata={
                'model': model_info,
            },
        )
