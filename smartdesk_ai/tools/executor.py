"""Code-level tool executor (FR-G3 to FR-G6, FR-G9). Not a tool the model sees.

Every tool call goes through `ToolExecutor.execute`: lookup in the registry, argument validation,
then at most `tools.max_attempts` attempts with backoff, retrying transient failures only.
"""
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from smartdesk_ai.config.settings import Settings
from smartdesk_ai.tools.definitions import get_definition
from smartdesk_ai.tools.errors import TRANSIENT
from smartdesk_ai.tools.registry import ToolRegistry, UnknownToolError


class Outcome(StrEnum):
    SUCCESS = "success"
    VALIDATION_ERROR = "validation_error"
    UNAVAILABLE = "unavailable"  # transient failures used up every attempt
    ERROR = "error"  # a permanent failure: unknown tool or a non-transient error in the function


class ToolResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool: str
    outcome: Outcome
    attempts: int
    data: Any = None
    error: str | None = None  # safe for the model to read; never holds user input


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        settings: Settings,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._registry = registry
        self._cfg = settings.tools
        self._sleep = sleep

    def execute(self, name: str, raw_args: dict[str, Any], *, agent: str) -> ToolResult:
        try:
            tool = self._registry.get(name)
            definition = get_definition(name)
        except (UnknownToolError, KeyError):
            return ToolResult(tool=name, outcome=Outcome.ERROR, attempts=0, error="unknown tool")
        try:
            args = definition.args_model.model_validate(raw_args)
        except ValidationError as exc:
            fields = sorted({".".join(map(str, e["loc"])) for e in exc.errors()})
            return ToolResult(
                tool=name, outcome=Outcome.VALIDATION_ERROR, attempts=0,
                error=f"invalid arguments: {', '.join(fields)}",
            )
        return self._run_with_retry(name, tool.fn, args)

    def _run_with_retry(self, name: str, fn: Callable[[Any], Any], args: BaseModel) -> ToolResult:
        for attempt in range(1, self._cfg.max_attempts + 1):
            try:
                data = fn(args)
            except TRANSIENT:
                if attempt < self._cfg.max_attempts:
                    self._sleep(self._cfg.backoff_seconds * 2 ** (attempt - 1))
                continue
            except Exception as exc:  # permanent: not retried; the message is not kept (FR-G11)
                return ToolResult(tool=name, outcome=Outcome.ERROR, attempts=attempt,
                                  error=type(exc).__name__)
            return ToolResult(tool=name, outcome=Outcome.SUCCESS, attempts=attempt, data=data)
        return ToolResult(
            tool=name, outcome=Outcome.UNAVAILABLE, attempts=self._cfg.max_attempts,
            error="the service is temporarily unavailable",
        )
