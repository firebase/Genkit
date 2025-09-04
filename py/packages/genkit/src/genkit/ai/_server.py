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

"""Functionality used by the Genkit veneer to start multiple servers.

The following servers may be started depending upon the host environment:

- Reflection API server.
- Flows server.

The reflection API server is started only in dev mode, which is enabled by the
setting the environment variable `GENKIT_ENV` to `dev`. By default, the
reflection API server binds and listens on (localhost, 3100 or the next available
port).  The flows server is the production servers that exposes flows and actions
over HTTP.
"""

import atexit
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ServerSpec:
    """ServerSpec encapsulates the scheme, host and port information.

    This class defines the server binding and listening configuration.
    """

    port: int
    scheme: str = 'http'
    host: str = 'localhost'

    @property
    def url(self) -> str:
        """URL evaluates to the host base URL given the server specs."""
        return f'{self.scheme}://{self.host}:{self.port}'


def create_runtime(
    runtime_dir: str,
    reflection_server_spec: ServerSpec,
    at_exit_fn: Callable[[Path], None] | None = None,
) -> Path:
    """Create a runtime configuration for use with the genkit CLI.

    The runtime information is stored in the form of a timestamped JSON file.
    Note that the file will be cleaned up as soon as the program terminates.

    Args:
        runtime_dir: The directory to store the runtime file in.
        reflection_server_spec: The server specification for the reflection
            server.
        at_exit_fn: A function to call when the runtime file is deleted.

    Returns:
        A path object representing the created runtime metadata file.
    """
    if not os.path.exists(runtime_dir):
        os.makedirs(runtime_dir)

    current_datetime = datetime.now()
    runtime_file_name = f'{os.getpid()}-{reflection_server_spec.port}-{current_datetime.isoformat()}.json'
    runtime_file_path = Path(os.path.join(runtime_dir, runtime_file_name))
    metadata = json.dumps({
        'reflectionApiSpecVersion': 1,
        'id': f'{os.getpid()}',
        'pid': os.getpid(),
        'reflectionServerUrl': reflection_server_spec.url,
        'timestamp': f'{current_datetime.isoformat()}',
    })
    runtime_file_path.write_text(metadata, encoding='utf-8')

    def cleanup_runtime() -> None:
        logger.info('Cleaning up runtime file', path=runtime_file_path)
        at_exit_fn(runtime_file_path)

    if at_exit_fn:
        atexit.register(cleanup_runtime)
    return runtime_file_path


def init_default_runtime(spec: ServerSpec) -> None:
    """Initialize the runtime for the Genkit instance."""
    runtimes_dir = os.path.join(os.getcwd(), '.genkit/runtimes')
    create_runtime(
        runtime_dir=runtimes_dir,
        reflection_server_spec=spec,
        at_exit_fn=os.remove,
    )
