"""Ticket agent (flow B). Escalation handoff first (FR-A16); collection, drafting and the
approval gate arrive in T4.12 onward.

The handoff merges the "I don't know" notice, the ticket offer and the email request into one
message. Giving an email counts as agreeing. If the email is already known the offer is a plain
yes or no. A refusal ends the flow. Nothing here creates a ticket (FR-B2).
"""
import re
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ConfigDict

from smartdesk_ai.agent.answer import INSUFFICIENT_REPLY
from smartdesk_ai.agent.state import AgentState, PendingOffer
from smartdesk_ai.tickets.schemas import EMAIL_PATTERN

_EDGE_PUNCTUATION = "<>()[]{}\"',.;:!?"
_YES = {"yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "please", "yes please", "sure please",
        "please do", "go ahead", "do it"}
_NO = {"no", "n", "nope", "no thanks", "no thank you", "not now", "never mind", "nevermind", "no need"}

OFFER_WITH_EMAIL_REQUEST = (
    "I can open a support ticket for you. If you'd like that, please share your work email address."
)
OFFER_YES_NO = "Would you like me to open a support ticket for this? Please answer yes or no."
ASK_EMAIL = "Thanks. What's your work email address?"
DECLINED = "No problem, I won't open a ticket. Is there anything else I can help with?"
AGREED_KNOWN_EMAIL = "Thanks, let's get the ticket details together."
AGREED_NEW_EMAIL = "Thanks, I have your email. Let's get the ticket details together."
INVALID_EMAIL = "That doesn't look like a valid email address. Could you check it and send it again?"
ASK_EMAIL_FOR_TICKET = "To open a ticket I need your work email address. What is it?"
ASK_ISSUE = "What would you like the ticket to be about?"
READY_REPLY = "Thanks, let's get the ticket details together."

# Words that carry no issue by themselves: "I want to raise a ticket" says nothing about the problem.
_FILLER = frozenset(
    "a an the to for of my me i we you please can could would will want like need ticket tickets raise "
    "create open log file make submit new support request issue help with this that it".split()
)


class OfferOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["agreed", "ask_email", "invalid_email", "declined", "unclear"]
    reply: str = ""
    update: dict[str, Any] = {}  # state changes to apply


def escalation_offer(email_known: bool) -> str:
    """The one message after a KB gap: "I don't know" plus the offer (FR-A16)."""
    return f"{INSUFFICIENT_REPLY} {OFFER_YES_NO if email_known else OFFER_WITH_EMAIL_REQUEST}"


def find_email(text: str) -> str | None:
    """The first token that is a valid email address, if any. Never guessed (FR-B13)."""
    for token in text.split():
        candidate = token.strip(_EDGE_PUNCTUATION)
        if EMAIL_PATTERN.fullmatch(candidate):
            return candidate
    return None


def _words(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip(_EDGE_PUNCTUATION).lower())


def handle_offer_reply(state: AgentState, text: str) -> OfferOutcome:
    """Read the reply to a ticket offer or email request. Anything else is `unclear`, and the
    router decides (a new topic clears the offer, FR-D2)."""
    if state.awaiting not in ("offer_reply", "email") or state.pending_offer is None:
        return OfferOutcome(kind="unclear")
    email = find_email(text)
    if email is not None:
        return OfferOutcome(kind="agreed", reply=AGREED_NEW_EMAIL, update={"email": email, "awaiting": "none"})
    if state.awaiting == "email" and "@" in text:  # tried to give an email and got it wrong
        return OfferOutcome(kind="invalid_email", reply=INVALID_EMAIL)
    words = _words(text)
    if words in _NO or words.startswith("no "):
        return OfferOutcome(
            kind="declined", reply=DECLINED,
            update={"awaiting": "none", "pending_offer": None, "ticket": None},
        )
    if words in _YES:
        if state.email:
            return OfferOutcome(kind="agreed", reply=AGREED_KNOWN_EMAIL, update={"awaiting": "none"})
        return OfferOutcome(kind="ask_email", reply=ASK_EMAIL, update={"awaiting": "email"})
    return OfferOutcome(kind="unclear")


def _has_substance(text: str) -> bool:
    return any(word not in _FILLER for word in re.findall(r"[a-z0-9']+", text.lower()))


def employee_statements(state: AgentState) -> list[str]:
    """What the employee said about the current ticket, in order: their own messages since the
    ticket's context began, without yes/no and email-only replies and without email addresses.
    A bare "raise a ticket" says nothing about the issue and is left out (FR-B6, FR-B13)."""
    statements = []
    for message in state.messages[state.ticket_context_start:]:
        if not isinstance(message, HumanMessage):
            continue
        tokens = [t for t in str(message.content).split() if find_email(t) is None]
        text = " ".join(tokens).strip()
        if not text or _words(text) in _YES or _words(text) in _NO or not _has_substance(text):
            continue
        statements.append(text)
    return statements


class CollectOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["need_issue", "need_email", "ready"]
    reply: str = ""
    update: dict[str, Any] = {}


def collect_step(state: AgentState) -> CollectOutcome:
    """What the ticket still needs from the employee (FR-B1, FR-B5). The email is asked for only
    when the session does not have it, and is never guessed (FR-B13)."""
    statements = employee_statements(state)
    if not statements:
        return CollectOutcome(kind="need_issue", reply=ASK_ISSUE)
    if not state.email:
        marker = state.pending_offer or PendingOffer(unanswered_text=statements[-1])
        return CollectOutcome(
            kind="need_email", reply=ASK_EMAIL_FOR_TICKET,
            update={"awaiting": "email", "pending_offer": marker},
        )
    return CollectOutcome(kind="ready")


def make_t_collect_node() -> Callable[[AgentState], dict[str, Any]]:
    def t_collect(state: AgentState) -> dict[str, Any]:
        out = collect_step(state)
        steps = state.steps + ["t_collect"]
        reply = READY_REPLY if out.kind == "ready" else out.reply
        return {**out.update, "steps": steps, "reply": reply, "messages": [AIMessage(reply)]}

    return t_collect
