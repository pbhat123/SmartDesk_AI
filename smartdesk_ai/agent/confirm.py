"""Reading the employee's reply at the approval gate (FR-G12, FR-B2).

An exact "yes" or "no" is accepted without a model call. Any other reply is read by the
CONFIRM_PARSE task (Light-Model, FR-H17). The parser never guesses: a malformed reply, a model
outage, or anything it cannot read is `unclear`, and the gate asks again or cancels. `rule_parser`
is the model-free version, for which every other reply is unclear.
"""
import re
import time
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, field_validator

from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.agent.state import Category, ConfirmDecision, Priority, TicketDraft
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.factory import ModelBuilder, openai_builder
from smartdesk_ai.llm.fallback import FallbackChat
from smartdesk_ai.llm.structured import parse_model_reply

ConfirmParser = Callable[[str, TicketDraft], ConfirmDecision]

TITLE_MAX = 80
MAX_UNCLEAR = 2  # the second unclear reply at a gate cancels (FR-G12)

REASK = "Sorry, I didn't catch that. Please reply yes to create the ticket, no to cancel, or tell me what to change."
CANCELLED = "Okay, I won't create the ticket. Is there anything else I can help with?"
GAVE_UP = "I couldn't tell what you'd like, so I haven't created the ticket. Just ask if you want to try again."


# Wording that tries to get past the gate. A reply with any of it is never a yes, whatever else it says (FR-B2, FR-G12).
BYPASS = re.compile(
    r"\b(?:skip\w*|bypass\w*|overrid\w+|(?:pre-?|already\s+|previously\s+)(?:confirm\w*|approv\w*|agreed|said)|approv\w*"
    r"|no\s+need\s+to\s+(?:confirm|ask|check)|without\s+(?:asking|confirm\w*|checking)|don'?t\s+(?:ask|confirm))\b", re.I)
# A yes needs an explicit agreement word in the reply itself, even when the model reads the reply as yes.
AFFIRM = re.compile(
    r"\b(?:yes|yeah|yep|yup|sure|ok|okay|correct|confirm\w*|go\s+ahead|looks?\s+good|that'?s\s+right|please\s+(?:do|create)"
    r"|create\s+it|proceed)\b", re.I)


def tries_to_bypass(reply: str) -> bool:
    return BYPASS.search(reply) is not None


def exact_decision(reply: str) -> ConfirmDecision | None:
    """"yes" or "no" alone (any case, trailing punctuation allowed), or None. No model call."""
    word = reply.strip().rstrip(".!").strip().lower()
    if word in ("yes", "no"):
        return ConfirmDecision(decision=word)  # type: ignore[arg-type]
    return None


def rule_parser(reply: str, draft: TicketDraft) -> ConfirmDecision:
    if tries_to_bypass(reply):
        return ConfirmDecision(decision="unclear")
    return exact_decision(reply) or ConfirmDecision(decision="unclear")


def apply_edit(draft: TicketDraft, decision: ConfirmDecision) -> tuple[TicketDraft, bool]:
    """The draft with the employee's changes applied, and whether anything changed. Added details
    go on the end of the description, in the employee's words (FR-B6)."""
    from smartdesk_ai.agent.ticket_agent import add_details  # ticket_agent imports this module

    updated = draft
    title = " ".join((decision.title or "").split())
    if title and len(title) <= TITLE_MAX:
        updated = updated.model_copy(update={"title": title})
    if decision.description and decision.description.strip():
        updated = updated.model_copy(update={"description": decision.description.strip()})
    if decision.details:
        updated = add_details(updated, decision.details)
    if decision.category is not None:
        updated = updated.model_copy(update={"category": decision.category})
    if decision.priority is not None:
        updated = updated.model_copy(update={"priority": decision.priority})
    return updated, updated != draft


class ConfirmReply(BaseModel):
    """The exact shape the CONFIRM_PARSE model must return (FR-H2)."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["yes", "no", "edit", "unclear"]
    title: str | None = None
    description: str | None = None
    details: str | None = None
    category: Category | None = None
    priority: Priority | None = None

    @field_validator("category", mode="before")
    @classmethod
    def _category_case(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("priority", mode="before")
    @classmethod
    def _priority_case(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


def ground_edit(decision: ConfirmDecision, raw: str, draft: TicketDraft) -> ConfirmDecision:
    """Keep an edit in the employee's words (FR-B13). A new title or description that adds anything
    not in their reply or the current draft is dropped; added details that do are replaced by the
    reply itself. No email address survives."""
    from smartdesk_ai.agent.ticket_agent import is_grounded, _strip_emails  # ticket_agent imports this module

    said = [raw, draft.title, draft.description]

    def keep(value: str | None) -> str | None:
        clean = _strip_emails(value or "")
        return clean if clean and is_grounded(clean, said) else None

    details = None
    if decision.details:
        details = keep(decision.details) or _strip_emails(raw) or None
    return decision.model_copy(
        update={"title": keep(decision.title), "description": keep(decision.description), "details": details}
    )


def parse_confirmation(llm: Any, prompt: str, reply: str, draft: TicketDraft) -> ConfirmDecision:
    """An exact yes/no makes no model call. Otherwise one CONFIRM_PARSE call; any failure is unclear."""
    unclear = ConfirmDecision(decision="unclear")
    if tries_to_bypass(reply):  # decided here, never by the model (FR-B2, FR-G12)
        return unclear
    exact = exact_decision(reply)
    if exact is not None:
        return exact
    human = (
        f"Current ticket:\nTitle: {draft.title}\nDescription: {draft.description}\n"
        f"Category: {draft.category}\nPriority: {draft.priority or 'not set'}\n\nEmployee reply: {reply}"
    )
    try:
        response = llm.invoke([SystemMessage(prompt), HumanMessage(human)])
        parsed = parse_model_reply(ConfirmReply, getattr(response, "content", response))
    except Exception:  # a bad reply (ReplyError) or an outage: ask again rather than guess
        return unclear
    decision = ConfirmDecision(**parsed.model_dump())
    if decision.decision == "yes" and not AFFIRM.search(reply):
        return unclear
    has_changes = any(v is not None for v in (decision.title, decision.description, decision.details,
                                              decision.category, decision.priority))
    if decision.decision == "yes" and has_changes:  # "yes, but ..." is never a plain yes
        decision = decision.model_copy(update={"decision": "edit"})
    if decision.decision == "edit":
        return ground_edit(decision, reply, draft)
    if decision.decision in ("yes", "no", "unclear"):
        return ConfirmDecision(decision=decision.decision)
    return decision


def make_confirm_parser(
    settings: Settings,
    *,
    builder: ModelBuilder = openai_builder,
    budget_level: int = 0,
    structured: bool | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ConfirmParser:
    """A parser for the CONFIRM_PARSE task and the active confirm_parse prompt. The model is built
    only when a reply needs it. The live client uses structured-output mode."""
    use_schema = structured if structured is not None else builder is openai_builder
    llm = FallbackChat(
        "CONFIRM_PARSE", settings, builder=builder, budget_level=budget_level,
        schema=ConfirmReply if use_schema else None, sleep=sleep,
    )
    prompt = load_prompt("confirm_parse", settings).text

    def parser(reply: str, draft: TicketDraft) -> ConfirmDecision:
        return parse_confirmation(llm, prompt, reply, draft)

    return parser
