"""Ticket status flow (flow C, FR-C1 to FR-C7).

The lookup is a `get_tickets` call through the executor: the SQLite map first, the search fallback
second, then the status and the latest team comment read live from GitHub (FR-C2, FR-C10, FR-C11).
Team comment text is untrusted (FR-C9); it is never given to a model here.
"""
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ConfigDict

from smartdesk_ai.agent.state import AgentState, TicketRef
from smartdesk_ai.agent.ticket_agent import INVALID_EMAIL, find_email, is_refusal
from smartdesk_ai.tickets import TeamUpdate, Ticket
from smartdesk_ai.tools.executor import Outcome

AGENT = "status_flow"
ASK_EMAIL_FOR_STATUS = "To check on your tickets I need your work email address. What is it?"
DECLINED_EMAIL = "No problem. Just ask whenever you'd like me to check on a ticket."
LOOKUP_FAILED = "Sorry, I couldn't check your tickets just now. Please try again later."


class StatusLookup(BaseModel):
    model_config = ConfigDict(frozen=True)

    tickets: list[Ticket]
    latest_update: TeamUpdate | None = None  # set only when exactly one ticket came back


class StatusUnavailable(BaseModel):
    """The lookup could not be done; `message` is the polite reply (FR-F1)."""

    message: str


def lookup_tickets(executor: Any, email: str, number: int | None = None) -> StatusLookup | StatusUnavailable:
    """The employee's tickets, or just ticket `number` of theirs, with live status and team update."""
    args: dict[str, Any] = {"email": email}
    if number is not None:
        args["ticket_number"] = number
    result = executor.execute("get_tickets", args, agent=AGENT)
    if result.outcome is Outcome.SUCCESS:
        return StatusLookup(tickets=result.data.tickets, latest_update=result.data.latest_update)
    return StatusUnavailable(message=result.reply or LOOKUP_FAILED)


def ticket_ref(ticket: Ticket) -> TicketRef:
    return TicketRef(number=ticket.number, title=ticket.title, status=ticket.status.value, url=ticket.url)


def _say(state: AgentState, step: str, reply: str, **update: Any) -> dict[str, Any]:
    return {"steps": state.steps + [step], "reply": reply, "messages": [AIMessage(reply)], **update}


def make_s_lookup_node(executor: Any) -> Callable[[AgentState], dict[str, Any]]:
    def s_lookup(state: AgentState) -> dict[str, Any]:
        if not state.email:  # reuse the session email, else ask (FR-C1)
            return _say(state, "s_lookup", ASK_EMAIL_FOR_STATUS, awaiting="email", status_pending_email=True,
                        status_candidates=[])
        out = lookup_tickets(executor, state.email)
        if isinstance(out, StatusUnavailable):
            return _say(state, "s_lookup", out.message, status_candidates=[], awaiting="none")
        refs = [ticket_ref(t) for t in out.tickets]
        # The display of the result follows in T4.24 to T4.26.
        return _say(state, "s_lookup", f"I found {len(refs)} ticket(s) for you.", status_candidates=refs,
                    awaiting="none")

    return s_lookup


def s_email_reply(state: AgentState) -> dict[str, Any]:
    """The reply to the status flow's email request (FR-C1). Anything else is a new turn."""
    text = ""
    for message in reversed(state.messages):
        if isinstance(message, HumanMessage):
            text = str(message.content)
            break
    email = find_email(text)
    if email is not None:
        return {"steps": state.steps + ["s_email:ok"], "email": email, "awaiting": "none",
                "status_pending_email": False}
    if "@" in text:
        return _say(state, "s_email:invalid", INVALID_EMAIL)
    if is_refusal(text):
        return _say(state, "s_email:declined", DECLINED_EMAIL, awaiting="none", status_pending_email=False)
    return {"steps": state.steps + ["s_email:unclear"]}


def after_s_email_reply(state: AgentState) -> str:
    return {"s_email:ok": "s_lookup", "s_email:unclear": "route"}.get(state.steps[-1], "__end__")
