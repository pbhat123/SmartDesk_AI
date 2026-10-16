"""Outage fallback between Main-Model and Light-Model (FR-F2).

Each model is retried first. If it still fails, the call goes to the other tier, except:
- at budget level 1 or higher a call never goes to Main-Model (FR-F2, FR-H8);
- before a Main to Light fallback the prompt must fit Light-Model's context window.
If nothing can answer, the result carries a graceful message and `ok` is False.
"""
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from smartdesk_ai.config.settings import Settings, Tier
from smartdesk_ai.llm.factory import ModelBuilder, get_llm_for_tier, openai_builder
from smartdesk_ai.llm.retry import call_with_retry
from smartdesk_ai.llm.tiers import resolve_tier

GRACEFUL_MESSAGE = (
    "I'm having trouble reaching my language model right now. "
    "Please try again in a few minutes."
)
CHARS_PER_TOKEN = 4  # rough estimate; avoids a tokenizer download and is safe for a window check


@dataclass(frozen=True)
class LLMResult:
    ok: bool
    text: str
    tier: Tier | None  # tier that answered; None when nothing did
    fell_back: bool


def estimate_tokens(prompt: Any) -> int:
    text = prompt if isinstance(prompt, str) else " ".join(
        str(getattr(m, "content", m)) for m in prompt
    )
    return len(text) // CHARS_PER_TOKEN + 1


def _fallback_tier(primary: Tier, budget_level: int) -> Tier | None:
    other: Tier = "light" if primary == "main" else "main"
    if other == "main" and budget_level >= 1:
        return None
    return other


def invoke_with_fallback(
    task: str,
    prompt: Any,
    settings: Settings,
    *,
    budget_level: int = 0,
    builder: ModelBuilder = openai_builder,
    sleep: Callable[[float], None] = time.sleep,
) -> LLMResult:
    primary = resolve_tier(task, settings, budget_level)
    candidates: list[Tier] = [primary]
    fallback = _fallback_tier(primary, budget_level)
    if fallback is not None:
        candidates.append(fallback)

    for tier in candidates:
        is_fallback = tier != primary
        if is_fallback and tier == "light":
            if estimate_tokens(prompt) > settings.models.light.context_window:
                continue  # oversize prompt is never sent to Light-Model
        try:
            handle = get_llm_for_tier(task, tier, settings, builder=builder)
            reply = call_with_retry(
                lambda: handle.model.invoke(prompt),
                attempts=settings.llm.retry_attempts,
                backoff=settings.llm.backoff_seconds,
                sleep=sleep,
            )
        except Exception:
            continue  # message deliberately not kept: it could echo prompt text (FR-G11)
        return LLMResult(True, str(reply.content), tier, is_fallback)
    return LLMResult(False, GRACEFUL_MESSAGE, None, False)
