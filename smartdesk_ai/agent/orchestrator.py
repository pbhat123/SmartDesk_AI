"""The orchestrator's planning step (FR-I3, FR-I4): from the classifier's sub-questions to the parts the workers answer.

The orchestrator sees the whole conversation, masked (`transcript_for`), and rewrites each part of the message into a
standalone sub-question. The code around the model keeps a plain question exactly as the employee asked it, so
retrieval, the answer cache key and the answer prompt input are the same as for a single question (FR-I5):
a single part is searched with the employee's own words unless the model says it needed the earlier turns.
"""
from collections.abc import Sequence

import tiktoken
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict

from smartdesk_ai.agent.followup import contextual_question
from smartdesk_ai.agent.state import STATUS_PLACEHOLDER, STATUS_REPLY, IntentResult, SubQuestion
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.guardrails.pii import EMAIL
from smartdesk_ai.ops.compaction import encoding_for, is_summary

EMAIL_PLACEHOLDER = "[email]"


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True)

    parts: list[SubQuestion]
    dropped: int = 0  # parts past `max_parts`, named to the employee but not answered


def _label(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "Employee"
    return "Assistant" if isinstance(message, AIMessage) else "System"


class OrchestratorLimits(BaseModel):
    """What bounds the orchestrator's work: parts per message (FR-I4) and the history it reads (FR-I3), both from config."""

    model_config = ConfigDict(frozen=True)

    max_parts: int
    history_max_tokens: int
    model_id: str  # whose tokenizer counts the history

    @classmethod
    def from_settings(cls, settings: Settings) -> "OrchestratorLimits":
        return cls(max_parts=settings.orchestrator.max_parts,
                   history_max_tokens=settings.orchestrator.history_max_tokens, model_id=settings.models.main.id)


def _render(message: BaseMessage) -> str:
    content = STATUS_PLACEHOLDER if getattr(message, "name", None) == STATUS_REPLY else EMAIL.sub(
        EMAIL_PLACEHOLDER, str(message.content))
    return content if _label(message) == "System" else f"{_label(message)}: {content}"


def _cut(text: str, limit: int, encoding: tiktoken.Encoding) -> str:
    """`text` cut to at most `limit` tokens, keeping its start."""
    tokens = encoding.encode(text)
    return text if len(tokens) <= limit else encoding.decode(tokens[:max(limit, 0)])


def transcript_for(messages: Sequence[BaseMessage], *, max_tokens: int | None = None, model_id: str | None = None) -> str:
    """The earlier conversation as text for the orchestrator: every message before the current one, with email
    addresses replaced (the email is in the session state, the model does not need it). Empty on the first turn.
    The messages are already redacted and compacted when they reach the graph (FR-H1, FR-H5). A ticket status reply is
    left out: it quotes the support team's comment, which no model may see (FR-C9).

    With `max_tokens` the text is the compaction summary (if there is one; at most half the cap) plus the most recent
    whole turns that fit, newest first, written oldest first; a turn is an employee message and what followed it. If even
    the newest turn does not fit it is cut to fit, so the model always sees what the employee just talked about."""
    earlier = list(messages[:-1] if messages and isinstance(messages[-1], HumanMessage) else messages)
    if max_tokens is None:
        return "\n".join(_render(m) for m in earlier)
    encoding = encoding_for(model_id or "")
    head = ""
    if earlier and is_summary(earlier[0]):
        head, earlier = _cut(_render(earlier[0]), max_tokens // 2, encoding), earlier[1:]
    turns: list[list[str]] = []
    for message in earlier:
        if isinstance(message, HumanMessage) or not turns:
            turns.append([])
        turns[-1].append(_render(message))
    budget = max_tokens - (len(encoding.encode(head)) + 1 if head else 0)
    chosen: list[str] = []
    for turn in reversed(turns):
        text = "\n".join(turn)
        cost = len(encoding.encode(text)) + 1  # the newline that joins it
        if cost <= budget:
            chosen.append(text)
            budget -= cost
            continue
        if not chosen:
            chosen.append(_cut(text, budget - 1, encoding))
        break
    return "\n".join([*([head] if head else []), *reversed(chosen)])


def _without_emails(text: str) -> str:
    """A sub-question never carries the employee's email: a search does not need it and it must not reach a worker."""
    return " ".join(EMAIL.sub("", text).split())


def plan_parts(raw_text: str, last_topic: str | None, result: IntentResult, *, max_parts: int) -> Plan:
    """The parts to answer for a kb_query. The model's sub-questions are used as they are for a message split into
    several; for one part, only when the model says it is a follow-up it had to rewrite. Otherwise the part is the
    employee's own words, or today's follow-up rule for a phrase such as "what about sick leave?"."""
    one = len(result.parts) == 1
    cleaned = [p.model_copy(update={"text": _without_emails(p.text)}) for p in result.parts]
    cleaned = [p for p in cleaned if p.text]
    dropped = max(len(cleaned) - max_parts, 0)
    cleaned = cleaned[:max_parts]
    if not cleaned:
        domain = result.domain or "MISC"
        return Plan(parts=[SubQuestion(part_id="p1", text=contextual_question(raw_text, last_topic), domain=domain)])
    if one:
        only = cleaned[0]
        text, rewritten = (only.text, True) if only.rewritten else (contextual_question(raw_text, last_topic), False)
        return Plan(parts=[SubQuestion(part_id="p1", text=text, domain=only.domain, rewritten=rewritten)])
    parts = [SubQuestion(part_id=f"p{i}", text=p.text, domain=p.domain, rewritten=True) for i, p in enumerate(cleaned, 1)]
    return Plan(parts=parts, dropped=dropped)
