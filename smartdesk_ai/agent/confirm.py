"""Reading the employee's reply at the approval gate (FR-G12, FR-B2).

An exact "yes" or "no" is accepted without a model call. The reply parser for anything else
(CONFIRM_PARSE, FR-H17) is added in T4.17; `rule_parser` here is the model-free fallback, for which
every other reply is unclear.
"""
from collections.abc import Callable

from smartdesk_ai.agent.state import ConfirmDecision, TicketDraft

ConfirmParser = Callable[[str, TicketDraft], ConfirmDecision]

TITLE_MAX = 80
MAX_UNCLEAR = 2  # the second unclear reply at a gate cancels (FR-G12)

REASK = "Sorry, I didn't catch that. Please reply yes to create the ticket, no to cancel, or tell me what to change."
CANCELLED = "Okay, I won't create the ticket. Is there anything else I can help with?"
GAVE_UP = "I couldn't tell what you'd like, so I haven't created the ticket. Just ask if you want to try again."


def exact_decision(reply: str) -> ConfirmDecision | None:
    """"yes" or "no" alone (any case, trailing punctuation allowed), or None. No model call."""
    word = reply.strip().rstrip(".!").strip().lower()
    if word in ("yes", "no"):
        return ConfirmDecision(decision=word)  # type: ignore[arg-type]
    return None


def rule_parser(reply: str, draft: TicketDraft) -> ConfirmDecision:
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
