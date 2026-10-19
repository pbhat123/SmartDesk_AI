"""The orchestrator's planning step (FR-I3, FR-I4): from the classifier's sub-questions to the parts the workers answer.

The orchestrator sees the whole conversation, masked (`transcript_for`), and rewrites each part of the message into a
standalone sub-question. The code around the model keeps a plain question exactly as the employee asked it, so
retrieval, the answer cache key and the answer prompt input are the same as for a single question (FR-I5):
a single part is searched with the employee's own words unless the model says it needed the earlier turns.
"""
import re
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict

from smartdesk_ai.agent.followup import contextual_question
from smartdesk_ai.agent.state import STATUS_PLACEHOLDER, STATUS_REPLY, IntentResult, SubQuestion
from smartdesk_ai.guardrails.pii import EMAIL

EMAIL_PLACEHOLDER = "[email]"


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True)

    parts: list[SubQuestion]
    dropped: int = 0  # parts past `max_parts`, named to the employee but not answered


def _label(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "Employee"
    return "Assistant" if isinstance(message, AIMessage) else "System"


def transcript_for(messages: Sequence[BaseMessage]) -> str:
    """The earlier conversation as text for the orchestrator: every message before the current one, with email
    addresses replaced (the email is in the session state, the model does not need it). Empty on the first turn.
    The messages are already redacted and compacted when they reach the graph (FR-H1, FR-H5). A ticket status reply is
    left out: it quotes the support team's comment, which no model may see (FR-C9)."""
    earlier = list(messages[:-1] if messages and isinstance(messages[-1], HumanMessage) else messages)
    lines = []
    for message in earlier:
        content = STATUS_PLACEHOLDER if getattr(message, "name", None) == STATUS_REPLY else EMAIL.sub(
            EMAIL_PLACEHOLDER, str(message.content))
        lines.append(content if _label(message) == "System" else f"{_label(message)}: {content}")
    return "\n".join(lines)


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
