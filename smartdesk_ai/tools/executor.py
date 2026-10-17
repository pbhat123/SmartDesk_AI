"""Code-level tool executor (FR-G3 to FR-G6, FR-G9). Not a tool the model sees.

Every tool call goes through `ToolExecutor.execute`: lookup in the registry, argument validation,
then at most `tools.max_attempts` attempts with backoff, retrying transient failures only.
"""
import time
import uuid
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from smartdesk_ai.config.settings import Settings
from smartdesk_ai.ops.metrics import MetricsRecorder, ToolCallRecord
from smartdesk_ai.tickets.schemas import EMAIL_PATTERN
from smartdesk_ai.tickets.sqlite_map import TicketStore
from smartdesk_ai.tools.definitions import get_definition
from smartdesk_ai.tools.errors import TRANSIENT
from smartdesk_ai.tools.registry import RegisteredTool, ToolRegistry, UnknownToolError


class Outcome(StrEnum):
    SUCCESS = "success"
    VALIDATION_ERROR = "validation_error"
    UNAVAILABLE = "unavailable"  # transient failures used up every attempt
    ERROR = "error"  # a permanent failure: unknown tool or a non-transient error in the function


class ErrorCode(StrEnum):
    SCHEMA = "schema"
    CONFIRMATION_REQUIRED = "confirmation_required"
    EMAIL_REQUIRED = "email_required"
    EMAIL_INVALID = "email_invalid"


# Tools that need the employee's email, and tools that need the employee's confirmation (FR-G5).
EMAIL_TOOLS = frozenset({"create_ticket", "get_tickets"})
CONFIRMED_TOOLS = frozenset({"create_ticket"})
ASK_FOR_EMAIL = "Ask the employee for their work email address before trying again."


class ToolResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool: str
    outcome: Outcome
    attempts: int
    data: Any = None
    code: ErrorCode | None = None
    error: str | None = None  # safe for the model to read; never holds user input


def _issue_number(data: Any) -> int | None:
    number = data.get("number") if isinstance(data, dict) else getattr(data, "number", None)
    return number if isinstance(number, int) else None


def _rejected(name: str, code: ErrorCode, message: str) -> ToolResult:
    return ToolResult(tool=name, outcome=Outcome.VALIDATION_ERROR, attempts=0, code=code, error=message)


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        settings: Settings,
        *,
        sleep: Callable[[float], None] = time.sleep,
        request_store: TicketStore | None = None,
        metrics: MetricsRecorder | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._registry = registry
        self._cfg = settings.tools
        self._sleep = sleep
        self._store = request_store
        self._metrics = metrics
        self._clock = clock

    def execute(
        self, name: str, raw_args: dict[str, Any], *, agent: str, confirmed: bool = False
    ) -> ToolResult:
        started = self._clock()
        attempt_ms: list[tuple[float, str]] = []
        result = self._execute(name, raw_args, confirmed, attempt_ms)
        if self._metrics is not None:
            self._metrics.record(ToolCallRecord(
                tool=name, agent=agent, outcome=result.outcome.value,
                error_code=result.code.value if result.code else None,
                attempts=result.attempts,
                attempt_latencies_ms=[ms for ms, _ in attempt_ms],
                attempt_outcomes=[o for _, o in attempt_ms],
                total_latency_ms=(self._clock() - started) * 1000,
            ))
        return result

    def _execute(
        self, name: str, raw_args: dict[str, Any], confirmed: bool, attempt_ms: list[tuple[float, str]]
    ) -> ToolResult:
        try:
            tool = self._registry.get(name)
            definition = get_definition(name)
        except (UnknownToolError, KeyError):
            return ToolResult(tool=name, outcome=Outcome.ERROR, attempts=0, error="unknown tool")
        try:
            args = definition.args_model.model_validate(raw_args)
        except ValidationError as exc:
            fields = sorted({".".join(map(str, e["loc"])) for e in exc.errors()})
            return _rejected(name, ErrorCode.SCHEMA, f"invalid arguments: {', '.join(fields)}")
        if name in EMAIL_TOOLS:
            email = getattr(args, "email", None)
            if not email:
                return _rejected(name, ErrorCode.EMAIL_REQUIRED, ASK_FOR_EMAIL)
            if not EMAIL_PATTERN.fullmatch(email):
                return _rejected(name, ErrorCode.EMAIL_INVALID, f"The email address is not valid. {ASK_FOR_EMAIL}")
        if name in CONFIRMED_TOOLS and not confirmed:
            return _rejected(
                name, ErrorCode.CONFIRMATION_REQUIRED,
                "The employee has not confirmed the summary yet; show it and ask for a yes.",
            )
        request_id = None
        if name in CONFIRMED_TOOLS:  # create_ticket: a fresh ID per call, never from the model (FR-G6)
            request_id = f"req-{uuid.uuid4().hex}"
            args = args.model_copy(update={"request_id": request_id})
            if self._store is not None:
                self._store.begin(request_id)
        result = self._run_with_retry(name, tool, args, request_id, attempt_ms)
        if result.outcome is Outcome.SUCCESS and request_id and self._store is not None:
            number = _issue_number(result.data)
            if number is not None:
                self._store.complete(request_id, number)
                self._store.link_email(args.email, number)  # FR-B3, FR-C10
        return result

    def _run_with_retry(
        self, name: str, tool: RegisteredTool, args: BaseModel, request_id: str | None,
        attempt_ms: list[tuple[float, str]],
    ) -> ToolResult:
        for attempt in range(1, self._cfg.max_attempts + 1):
            attempt_start = self._clock()
            try:
                if attempt > 1 and request_id and tool.recover is not None:
                    # An earlier attempt may have created the issue even though its response was lost.
                    existing = tool.recover(request_id)
                    if existing is not None:
                        attempt_ms.append(((self._clock() - attempt_start) * 1000, "success"))
                        return ToolResult(tool=name, outcome=Outcome.SUCCESS, attempts=attempt, data=existing)
                data = tool.fn(args)
            except TRANSIENT:
                attempt_ms.append(((self._clock() - attempt_start) * 1000, "transient_failure"))
                if attempt < self._cfg.max_attempts:
                    self._sleep(self._cfg.backoff_seconds * 2 ** (attempt - 1))
                continue
            except Exception as exc:  # permanent: not retried; the message is not kept (FR-G11)
                attempt_ms.append(((self._clock() - attempt_start) * 1000, "error"))
                return ToolResult(tool=name, outcome=Outcome.ERROR, attempts=attempt,
                                  error=type(exc).__name__)
            attempt_ms.append(((self._clock() - attempt_start) * 1000, "success"))
            return ToolResult(tool=name, outcome=Outcome.SUCCESS, attempts=attempt, data=data)
        return ToolResult(
            tool=name, outcome=Outcome.UNAVAILABLE, attempts=self._cfg.max_attempts,
            error="the service is temporarily unavailable",
        )
