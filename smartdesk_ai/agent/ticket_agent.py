"""Ticket agent (flow B). Escalation handoff first (FR-A16); collection, drafting and the
approval gate arrive in T4.12 onward.

The handoff merges the "I don't know" notice, the ticket offer and the email request into one
message. Giving an email counts as agreeing. If the email is already known the offer is a plain
yes or no. A refusal ends the flow. Nothing here creates a ticket (FR-B2).
"""
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from smartdesk_ai.agent.answer import INSUFFICIENT_REPLY
from smartdesk_ai.agent.state import AgentState
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


class OfferOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["agreed", "ask_email", "declined", "unclear"]
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
