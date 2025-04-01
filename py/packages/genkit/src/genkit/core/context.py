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

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Union


class RequestMethod(StrEnum):
    """Request method."""

    GET = 'GET'
    PUT = 'PUT'
    POST = 'POST'
    DELETE = 'DELETE'
    OPTIONS = 'OPTIONS'
    QUERY = 'QUERY'


@dataclass
class RequestData[T]:
    """A universal type that request handling extensions (e.g. flask) can map their request to.
    This allows ContextProviders to build consistent interfacese on any web framework.
    Headers must be lowercase to ensure portability."""

    method: RequestMethod
    headers: dict[str, str]
    input: T


type ContextProvider[T] = Callable[[RequestData[T]], Union[dict[str, Any], Awaitable[dict[str, Any]]]]
"""Middleware can read request data and add information to the context that will be passed to the
Action. If middleware throws an error, that error will fail the request and the Action will not
be invoked. Expected cases should return a UserFacingError, which allows the request handler to
know what data is safe to return to end users.

Middleware can provide validation in addition to parsing. For example, an auth middleware can have
policies for validating auth in addition to passing auth context to the Action.
"""
