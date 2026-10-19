"""Context compaction (FR-H5): when the orchestrator's messages fill `compaction.trigger_fraction` of the
active model's context window, older turns are replaced by a running summary.

The window is Main-Model's, or Light-Model's once the budget has switched (FR-H8). The email and the pending
ticket draft live in their own state fields and are untouched; the last `keep_last_turns` employee turns
and any messages from `protect_from` on (an open ticket conversation) are kept as they are. The summary is
made by Light-Model in slices that fit its window; at budget level 0, history that exceeds Light-Model's
window goes to Main-Model in one call instead (as long as it fits). Tokens are counted with tiktoken.
"""
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

import tiktoken
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.agent.state import STATUS_PLACEHOLDER, STATUS_REPLY
from smartdesk_ai.config.settings import Settings, Tier
from smartdesk_ai.guardrails.pii import PiiRedactor, mask_emails_in
from smartdesk_ai.llm.factory import ModelBuilder, openai_builder
from smartdesk_ai.llm.fallback import FallbackChat
from smartdesk_ai.llm.tiers import resolve_tier

logger = logging.getLogger(__name__)

SUMMARY_PREFIX = "Summary of the earlier conversation: "
SAFETY_MARGIN = 16  # tokens left unused in a slice for the per-message labels
MIN_SLICE_TOKENS = 64
FALLBACK_ENCODING = "o200k_base"

Summarizer = Callable[[str, str, Tier], str]  # (summary so far, transcript of the next part, tier) -> new summary


def _encoding(model_id: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model_id)
    except KeyError:
        return tiktoken.get_encoding(FALLBACK_ENCODING)


def _label(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "Employee"
    return "Assistant" if isinstance(message, AIMessage) else "System"


def _line(message: BaseMessage) -> str:
    """One message as text for the summarizer and the window count. A ticket status reply quotes the support team's
    comment, which no model may see (FR-C9), so it stands as a placeholder."""
    if getattr(message, "name", None) == STATUS_REPLY:
        return f"{_label(message)}: {STATUS_PLACEHOLDER}"
    return f"{_label(message)}: {message.content}"


def count_tokens_in(messages: Sequence[BaseMessage], model_id: str) -> int:
    encoding = _encoding(model_id)
    return sum(len(encoding.encode(_line(m))) for m in messages)


def is_summary(message: BaseMessage) -> bool:
    return isinstance(message, SystemMessage) and str(message.content).startswith(SUMMARY_PREFIX)


def mask_summary(text: str) -> str:
    """No raw email, phone, card or ID in a summary, whatever the model wrote."""
    return mask_emails_in(PiiRedactor(None, 0.5).redact(text).text)


class CompactionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    summary: str
    cut: int  # messages[:cut] were replaced by the summary
    tier: Tier
    slices: int  # summarizer calls
    messages: list[Any]  # the summary message (it takes the first message's id) plus messages[cut:]


class Compactor:
    def __init__(self, settings: Settings, summarizer: Summarizer, *, overhead_tokens: int = 0) -> None:
        self._settings = settings
        self._summarize = summarizer
        self._overhead = overhead_tokens  # the compaction prompt's own size

    def active_model(self, budget_level: int):
        """The model whose window the orchestrator's context is measured against."""
        tier = resolve_tier("ORCHESTRATOR", self._settings, budget_level)
        return getattr(self._settings.models, tier)

    def needs(self, messages: Sequence[BaseMessage], budget_level: int = 0) -> bool:
        model = self.active_model(budget_level)
        used = count_tokens_in(messages, model.id)
        return used >= self._settings.compaction.trigger_fraction * model.context_window

    def _cut(self, messages: Sequence[BaseMessage], protect_from: int | None) -> int | None:
        keep = self._settings.compaction.keep_last_turns
        humans = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        if len(humans) <= keep:
            return None
        cut = humans[-keep]
        if protect_from is not None:
            cut = min(cut, protect_from)
        return cut if cut >= 2 else None

    def compact(
        self, messages: Sequence[BaseMessage], *, budget_level: int = 0, protect_from: int | None = None
    ) -> CompactionResult | None:
        """The messages with their older part summarized, or None when nothing needs doing (or the
        summarizer is down, in which case compaction is skipped and retried on a later turn)."""
        if not self.needs(messages, budget_level):
            return None
        cut = self._cut(messages, protect_from)
        if cut is None:
            return None
        older = list(messages[:cut])
        prior = ""
        if older and is_summary(older[0]):
            prior, older = str(older[0].content)[len(SUMMARY_PREFIX):], older[1:]
        if not older:
            return None
        try:
            summary, tier, slices = self._summarize_older(prior, older, budget_level)
        except Exception as error:
            logger.warning("compaction skipped, summarizer failed (%s)", type(error).__name__)
            return None
        first = messages[0]
        kept = [SystemMessage(SUMMARY_PREFIX + summary, id=first.id), *messages[cut:]]
        logger.info("context compacted: %d messages summarized in %d call(s) on %s", cut, slices, tier)
        return CompactionResult(summary=summary, cut=cut, tier=tier, slices=slices, messages=kept)

    def _summarize_older(self, prior: str, older: list[BaseMessage], budget_level: int) -> tuple[str, Tier, int]:
        light, main = self._settings.models.light, self._settings.models.main
        light_limit = light.context_window - light.max_output - self._overhead - SAFETY_MARGIN
        main_limit = main.context_window - self._overhead - SAFETY_MARGIN
        transcript = "\n".join(_line(m) for m in older)
        light_tokens = len(_encoding(light.id).encode(prior + "\n" + transcript))
        if budget_level < 1 and light_tokens > light_limit:
            if len(_encoding(main.id).encode(prior + "\n" + transcript)) <= main_limit:
                return mask_summary(self._summarize(prior, transcript, "main")), "main", 1
        elif light_tokens <= light_limit:
            return mask_summary(self._summarize(prior, transcript, "light")), "light", 1
        return self._summarize_in_slices(prior, older, light, light_limit)

    def _summarize_in_slices(self, prior: str, older: list[BaseMessage], light, light_limit: int):
        encoding = _encoding(light.id)
        lines = [_line(m) for m in older]
        calls, summary, index = 0, prior, 0
        while index < len(lines):
            budget = max(light_limit - len(encoding.encode(summary)), MIN_SLICE_TOKENS)
            chunk: list[str] = []
            used = 0
            while index < len(lines):
                tokens = encoding.encode(lines[index])
                if len(tokens) > budget:  # a single oversized message is cut down to fit
                    tokens = tokens[:budget]
                    lines[index] = encoding.decode(tokens)
                if used + len(tokens) + 1 > budget and chunk:
                    break
                chunk.append(lines[index])
                used += len(tokens) + 1
                index += 1
            summary = mask_summary(self._summarize(summary, "\n".join(chunk), "light"))
            calls += 1
        return summary, "light", calls


def make_summarizer(
    settings: Settings,
    *,
    builder: ModelBuilder = openai_builder,
    budget_level: int = 0,
    sleep: Callable[[float], None] = time.sleep,
) -> Summarizer:
    """The COMPACTION task (Light-Model by the tier map) with the active compaction prompt. A call for
    "main" is allowed only at budget level 0; every call goes through the outage fallback."""
    prompt = load_prompt("compaction", settings).text
    chats: dict[Tier, FallbackChat] = {}

    def summarize(prior: str, transcript: str, tier: Tier) -> str:
        if tier not in chats:
            chats[tier] = FallbackChat("COMPACTION", settings, builder=builder, budget_level=budget_level,
                                       sleep=sleep, tier=tier)
        human = f"Summary so far:\n{prior or '(none)'}\n\nNext part of the conversation:\n{transcript}"
        reply = chats[tier].invoke([SystemMessage(prompt), HumanMessage(human)])
        return mask_summary(str(reply.content).strip())

    return summarize
