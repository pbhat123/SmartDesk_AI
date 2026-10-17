"""Outage fallback between Main-Model and Light-Model (FR-F2).

Each model is retried first. If it still fails, the call goes to the other tier, except:
- at budget level 1 or higher a call never goes to Main-Model (FR-F2, FR-H8);
- before a Main to Light fallback the prompt must fit Light-Model's context window minus its max output.
If nothing can answer, the result carries a graceful message and `ok` is False.
"""
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

GRACEFUL_MESSAGE = (
    "I'm having trouble reaching my language model right now. "
    "Please try again in a few minutes."
)


@dataclass(frozen=True)
class LLMResult:
    ok: bool
    text: str
    tier: Tier | None  # tier that answered; None when nothing did
    fell_back: bool


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
) -> LLMResult:
    """Run `prompt` with retry and the Main/Light fallback. With `schema`, the model's
    structured-output mode is used and the parsed reply comes back as JSON text. A reply that
    fails the schema is not an outage: its raw text is returned for the caller to reject."""
    primary = resolve_tier(task, settings, budget_level)
    if tier is not None and budget_level < 1:  # an explicit tier, never above what the budget allows
        primary = tier
    candidates: list[Tier] = [primary]
    fallback = _fallback_tier(primary, budget_level)
    if fallback is not None:
        candidates.append(fallback)

    for tier in candidates:
        is_fallback = tier != primary
        if is_fallback and tier == "light":
            light = settings.models.light
            if count_tokens(prompt, light.id) > light.context_window - light.max_output:
                continue  # oversize prompt is never sent to Light-Model
        try:
            handle = get_llm_for_tier(task, tier, settings, builder=builder)
            model = handle.model.with_structured_output(schema, include_raw=True) if schema else handle.model
            reply = call_with_retry(
                lambda: model.invoke(prompt),
                attempts=settings.llm.retry_attempts,
                backoff=settings.llm.backoff_seconds,
                sleep=sleep,
            )
        except Exception:
            continue  # message deliberately not kept: it could echo prompt text (FR-G11)
        return LLMResult(True, _reply_text(reply, schema), tier, is_fallback)
    return LLMResult(False, GRACEFUL_MESSAGE, None, False)


class LLMUnavailableError(Exception):
    """Both tiers failed (FR-F2). `message` is the graceful text for the employee."""

    def __init__(self, message: str = GRACEFUL_MESSAGE) -> None:
        super().__init__(message)
        self.message = message


class FallbackChat:
    """A chat model for one task that goes through the outage fallback on every call.

    `invoke` returns an object with `.content`, or raises `LLMUnavailableError` when nothing
    can answer. Pass `schema` to use structured-output mode (the live client)."""

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
    ) -> None:
        self._args = (task, settings)
        self._kwargs = {"budget_level": budget_level, "builder": builder, "sleep": sleep, "schema": schema,
                        "tier": tier}

    def invoke(self, messages: Any) -> SimpleNamespace:
        result = invoke_with_fallback(self._args[0], messages, self._args[1], **self._kwargs)
        if not result.ok:
            raise LLMUnavailableError(result.text)
        return SimpleNamespace(content=result.text)
