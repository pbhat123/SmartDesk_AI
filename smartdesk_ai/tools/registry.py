"""Registry mapping each tool name to its Python function (FR-G3).

Only the executor looks tools up here; the model sees `definitions.py`, never the registry.
`recover` is an optional hook the executor uses before retrying a call that may have already
taken effect (FR-G6).
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from smartdesk_ai.tools.definitions import get_definition


class UnknownToolError(KeyError):
    """No tool with this name is defined or registered."""


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    fn: Callable[[Any], Any]
    recover: Callable[[str], Any | None] | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self, name: str, fn: Callable[[Any], Any], *, recover: Callable[[str], Any | None] | None = None
    ) -> None:
        try:
            get_definition(name)
        except KeyError:
            raise UnknownToolError(name) from None
        if name in self._tools:
            raise ValueError(f"tool {name!r} is already registered")
        self._tools[name] = RegisteredTool(name, fn, recover)

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolError(name) from None
