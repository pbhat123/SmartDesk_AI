"""Outage fallback between Main-Model and Light-Model (FR-F2).

Each model is retried first. If it still fails, the call goes to the other tier, except:
- at budget level 1 or higher a call never goes to Main-Model (FR-F2, FR-H8);
- before a Main to Light fallback the prompt must fit Light-Model's context window minus its max output.
If nothing can answer, the result carries a graceful message and `ok` is False.
"""
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import tiktoken
from pydantic import BaseModel

from smartdesk_ai.config.settings import Settings, Tier
from smartdesk_ai.llm.factory import ModelBuilder, get_llm_for_tier, openai_builder
from smartdesk_ai.llm.retry import call_with_retry
from smartdesk_ai.llm.tiers import resolve_tier

logger = logging.getLogger(__name__)

GRACEFUL_MESSAGE = (
    "I'm having trouble reaching my language model right now. "
    "Please try again in a few minutes."
)

SERVICE_UNAVAILABLE_MESSAGE = "Sorry, the service is temporarily unavailable. Please try again later."


@dataclass(frozen=True)
class StepTiming:
    """One model attempt (with its retries) inside an LLM call."""

    tier: Tier
    latency_ms: float
    ok: bool


@dataclass(frozen=True)
class LLMCall:
    """What one `invoke_with_fallback` call did, for cost tracking and LLM records (FR-H7, FR-H9).
    `prompt` is the raw prompt: whoever stores it must redact it first."""

    task: str
    tier: Tier  # the tier that answered, or the primary one when nothing did
    model_id: str
    model_version: str | None
    prompt: Any
    completion: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    latency_ms: float
    steps: tuple[StepTiming, ...]
    fell_back: bool
    ok: bool


LLMObserver = Callable[[LLMCall], None]


@dataclass(frozen=True)
class LLMResult:
    ok: bool
    text: str
    tier: Tier | None  # tier that answered; None when nothing did
    fell_back: bool
    tool_calls: tuple[dict[str, Any], ...] = ()  # set when the call was made with `tools`


def count_tokens(prompt: Any, model_id: str) -> int:
    """Prompt tokens by the model's own tiktoken encoding (FR-H5)."""
    text = prompt if isinstance(prompt, str) else " ".join(
        str(getattr(m, "content", m)) for m in prompt
    )
    return len(tiktoken.encoding_for_model(model_id).encode(text))


def _fallback_tier(primary: Tier, budget_level: int) -> Tier | None:
    other: Tier = "light" if primary == "main" else "main"
    if other == "main" and budget_level >= 1:
        return None
    return other


def _reply_text(reply: Any, schema: type[BaseModel] | None) -> str:
    if schema is None:
        return str(reply.content)
    parsed = reply.get("parsed")
    if isinstance(parsed, BaseModel):
        return parsed.model_dump_json()
    return str(getattr(reply.get("raw"), "content", ""))


def invoke_with_fallback(
    task: str,
    prompt: Any,
    settings: Settings,
    *,
    budget_level: int = 0,
    builder: ModelBuilder = openai_builder,
    sleep: Callable[[float], None] = time.sleep,
    schema: type[BaseModel] | None = None,
    tier: Tier | None = None,
    observer: LLMObserver | None = None,
    tools: list[Any] | None = None,
) -> LLMResult:
    """Run `prompt` with retry and the Main/Light fallback. With `schema`, the model's
    structured-output mode is used and the parsed reply comes back as JSON text. A reply that
    fails the schema is not an outage: its raw text is returned for the caller to reject. With `tools`
    the model gets them bound and the requested tool calls come back in `LLMResult.tool_calls`."""
    primary = resolve_tier(task, settings, budget_level)
    if tier is not None and budget_level < 1:  # an explicit tier, never above what the budget allows
        primary = tier
    candidates: list[Tier] = [primary]
    fallback = _fallback_tier(primary, budget_level)
    if fallback is not None:
        candidates.append(fallback)

    started = time.perf_counter()
    steps: list[StepTiming] = []
    for candidate in candidates:
        is_fallback = candidate != primary
        if is_fallback and candidate == "light":
            light = settings.models.light
            if count_tokens(prompt, light.id) > light.context_window - light.max_output:
                continue  # oversize prompt is never sent to Light-Model
        attempt_start = time.perf_counter()
        try:
            handle = get_llm_for_tier(task, candidate, settings, builder=builder)
            if tools:
                model = handle.model.bind_tools(tools)
            else:
                model = handle.model.with_structured_output(schema, include_raw=True) if schema else handle.model
            reply = call_with_retry(
                lambda: model.invoke(prompt),
                attempts=settings.llm.retry_attempts,
                backoff=settings.llm.backoff_seconds,
                sleep=sleep,
            )
        except Exception:
            steps.append(StepTiming(candidate, (time.perf_counter() - attempt_start) * 1000, False))
            continue  # message deliberately not kept: it could echo prompt text (FR-G11)
        steps.append(StepTiming(candidate, (time.perf_counter() - attempt_start) * 1000, True))
        text = _reply_text(reply, schema)
        if observer is not None:
            _observe(observer, task, candidate, settings, prompt, text, reply, schema, steps, started, is_fallback)
        calls = tuple(getattr(reply, "tool_calls", None) or ()) if tools else ()
        return LLMResult(True, text, candidate, is_fallback, calls)
    if observer is not None:
        model_id = getattr(settings.models, primary).id
        _safely(observer, LLMCall(task, primary, model_id, None, prompt, "", 0, 0, 0,
                                  (time.perf_counter() - started) * 1000, tuple(steps), False, False))
    return LLMResult(False, GRACEFUL_MESSAGE, None, False)


def _safely(observer: LLMObserver, call: LLMCall) -> None:
    try:
        observer(call)
    except Exception as error:  # bookkeeping never breaks a model call
        logger.warning("LLM observer failed (%s)", type(error).__name__)


def _observe(observer, task, tier, settings, prompt, text, reply, schema, steps, started, fell_back) -> None:
    """Token counts from the provider's usage metadata when present, else counted locally."""
    message = reply.get("raw") if schema is not None and isinstance(reply, dict) else reply
    usage = getattr(message, "usage_metadata", None) or {}
    model_id = getattr(settings.models, tier).id
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if input_tokens is None:
        input_tokens = _estimate(prompt, model_id)
    if output_tokens is None:
        output_tokens = _estimate(text, model_id)
    cached = (usage.get("input_token_details") or {}).get("cache_read", 0)
    version = (getattr(message, "response_metadata", None) or {}).get("model_name")
    _safely(observer, LLMCall(task, tier, model_id, version, prompt, text, int(input_tokens), int(cached),
                              int(output_tokens), (time.perf_counter() - started) * 1000, tuple(steps),
                              fell_back, True))


def _estimate(value: Any, model_id: str) -> int:
    try:
        return count_tokens(value, model_id)
    except Exception:
        return 0


class LLMUnavailableError(Exception):
    """Both tiers failed (FR-F2). `message` is the graceful text for the employee."""

    def __init__(self, message: str = GRACEFUL_MESSAGE) -> None:
        super().__init__(message)
        self.message = message


class FallbackChat:
    """A chat model for one task that goes through the outage fallback on every call.

    `invoke` returns an object with `.content`, or raises `LLMUnavailableError` when nothing
    can answer. Pass `schema` to use structured-output mode (the live client). Inside a request scope
    (`ops.runtime`) the budget level is read per call (FR-H8) and every call is reported for cost
    tracking and LLM records (FR-H7, FR-H9); outside one it behaves exactly as before."""

    def __init__(
        self,
        task: str,
        settings: Settings,
        *,
        builder: ModelBuilder = openai_builder,
        budget_level: int = 0,
        schema: type[BaseModel] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        tier: Tier | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        self._args = (task, settings)
        self._budget_level = budget_level
        self._kwargs = {"builder": builder, "sleep": sleep, "schema": schema, "tier": tier, "tools": tools}
        self.agent = task.lower()

    def invoke(self, messages: Any) -> SimpleNamespace:
        from smartdesk_ai.ops.runtime import current_scope  # ops imports this module, so import late

        scope = current_scope()
        level, observer = self._budget_level, None
        if scope is not None:
            ctx, services = scope
            level = max(level, services.budget_level(ctx))
            if level >= 2:  # cutoff 2: no LLM calls at all (FR-H8)
                raise LLMUnavailableError(SERVICE_UNAVAILABLE_MESSAGE)
            observer = lambda call: services.on_llm_call(ctx, self.agent, call)  # noqa: E731
        result = invoke_with_fallback(
            self._args[0], messages, self._args[1], budget_level=level, observer=observer, **self._kwargs)
        if not result.ok:
            raise LLMUnavailableError(result.text)
        return SimpleNamespace(content=result.text, tool_calls=list(result.tool_calls))
