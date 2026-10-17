"""Ticket status flow (flow C, FR-C1 to FR-C7).

The lookup is a `get_tickets` call through the executor: the SQLite map first, the search fallback
second, then the status and the latest team comment read live from GitHub (FR-C2, FR-C10, FR-C11).
Team comment text is untrusted (FR-C9): no model is called in this flow after the lookup, and the text
reaches the employee, and so the saved transcript, only as `TeamUpdate.quoted()`.
"""
import re
from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, ConfigDict

from smartdesk_ai.agent.state import AgentState, TicketRef
from smartdesk_ai.agent.ticket_agent import INVALID_EMAIL, find_email, is_refusal
from smartdesk_ai.tickets import TeamUpdate, Ticket
from smartdesk_ai.tools.executor import Outcome

AGENT = "status_flow"
ASK_EMAIL_FOR_STATUS = "To check on your tickets I need your work email address. What is it?"
DECLINED_EMAIL = "No problem. Just ask whenever you'd like me to check on a ticket."
NO_UPDATE_YET = "No updates from the support team yet."
NO_TICKETS = "You have no tickets on file. If you expected to see one, please check the email address you gave me."
LOOKUP_FAILED = "Sorry, I couldn't check your tickets just now. Please try again later."
PICK_QUESTION = "Which one would you like to know more about? You can say \"the second one\" or give its number."

_ORDINALS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5,
    "5th": 5, "sixth": 6, "6th": 6, "seventh": 7, "7th": 7, "eighth": 8, "8th": 8, "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10,
}
_STOP = frozenset("that this with your have what which ticket tickets issue request about one the".split())


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


class Pick(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["chosen", "out_of_range", "none"]
    ref: TicketRef | None = None


def pick_ticket(text: str, candidates: Sequence[TicketRef]) -> Pick:
    """Which listed ticket the employee means: "the second one", "2", "#14", "the last one", or a word
    from its title. Only the tickets that were listed can be chosen."""
    lowered = text.lower()
    words = re.findall(r"[a-z0-9]+", lowered)
    ordinal = next((_ORDINALS[w] for w in words if w in _ORDINALS), None)
    if ordinal is None and "last" in words:
        ordinal = len(candidates)
    if ordinal is not None:
        if 1 <= ordinal <= len(candidates):
            return Pick(kind="chosen", ref=candidates[ordinal - 1])
        return Pick(kind="out_of_range")
    numbers = [int(n) for n in re.findall(r"#\s*(\d+)", lowered)]
    if numbers:  # "#14" always means the ticket number
        match = next((c for c in candidates if c.number == numbers[0]), None)
        return Pick(kind="chosen", ref=match) if match else Pick(kind="none")
    bare = [int(w) for w in words if w.isdigit()]
    if len(bare) == 1:  # the list's own numbering first; a bigger number may be a listed ticket number
        if 1 <= bare[0] <= len(candidates):
            return Pick(kind="chosen", ref=candidates[bare[0] - 1])
        match = next((c for c in candidates if c.number == bare[0]), None)
        return Pick(kind="chosen", ref=match) if match else Pick(kind="out_of_range")
    typed = {w for w in words if len(w) >= 4 and w not in _STOP}
    hits = [c for c in candidates if typed & {w for w in re.findall(r"[a-z0-9]+", c.title.lower()) if len(w) >= 4}]
    return Pick(kind="chosen", ref=hits[0]) if len(hits) == 1 else Pick(kind="none")


def older_line(count: int) -> str:
    return f"and {count} older {'one' if count == 1 else 'ones'}."


def render_list(refs: Sequence[TicketRef], older: int = 0) -> str:
    """The numbered list of the newest tickets, then how many older ones are not shown (FR-C3)."""
    lines = [f"I found {len(refs) + older} tickets for you:"]
    lines += [f"{i}. #{r.number} {r.title} ({r.status})" for i, r in enumerate(refs, 1)]
    if older:
        lines.append(older_line(older))
    return "\n".join(lines + ["", PICK_QUESTION])


def render_ticket(ticket: Ticket, update: TeamUpdate | None) -> str:
    """One ticket: ID, title, status, link, and the latest team update (FR-C7). Everything but the update
    is copied from GitHub's response; the update is untrusted text and appears only quoted (FR-C9)."""
    lines = [f"Ticket #{ticket.number}: {ticket.title}", f"Status: {ticket.status.value}", ticket.url, ""]
    if update is None:
        lines.append(NO_UPDATE_YET)
    else:
        lines += ["Latest update from the support team:", update.quoted()]
    return "\n".join(lines)


def _say(state: AgentState, step: str, reply: str, **update: Any) -> dict[str, Any]:
    return {"steps": state.steps + [step], "reply": reply, "messages": [AIMessage(reply)], **update}


def make_s_lookup_node(executor: Any, max_listed: int) -> Callable[[AgentState], dict[str, Any]]:
    def s_lookup(state: AgentState) -> dict[str, Any]:
        if not state.email:  # reuse the session email, else ask (FR-C1)
            return _say(state, "s_lookup", ASK_EMAIL_FOR_STATUS, awaiting="email", status_pending_email=True,
                        status_candidates=[])
        out = lookup_tickets(executor, state.email)
        if isinstance(out, StatusUnavailable):
            return _say(state, "s_lookup", out.message, status_candidates=[], awaiting="none")
        newest_first = sorted(out.tickets, key=lambda t: t.number, reverse=True)
        refs = [ticket_ref(t) for t in newest_first[:max_listed]]
        older = len(newest_first) - len(refs)
        if len(refs) >= 2:  # list briefly and let them choose (FR-C3)
            return _say(state, "s_lookup", render_list(refs, older), status_candidates=refs,
                        awaiting="ticket_pick")
        if not refs:  # say so plainly, and invent nothing (FR-C4)
            return _say(state, "s_lookup", NO_TICKETS, status_candidates=[], awaiting="none")
        return _say(state, "s_lookup", render_ticket(out.tickets[0], out.latest_update), status_candidates=[],
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


def make_s_pick_node(executor: Any) -> Callable[[AgentState], dict[str, Any]]:
    """The reply to the pick list (awaiting ticket_pick). The chosen ticket is looked up live."""

    def s_pick(state: AgentState) -> dict[str, Any]:
        text = ""
        for message in reversed(state.messages):
            if isinstance(message, HumanMessage):
                text = str(message.content)
                break
        pick = pick_ticket(text, state.status_candidates)
        if pick.kind == "none":
            return {"steps": state.steps + ["s_pick:unclear"]}
        if pick.kind == "out_of_range":
            reply = f"I only found {len(state.status_candidates)} tickets. {PICK_QUESTION}"
            return _say(state, "s_pick:range", reply)
        out = lookup_tickets(executor, state.email, pick.ref.number)
        if isinstance(out, StatusUnavailable):  # keep the list so they can try again
            return _say(state, "s_pick:unavailable", out.message)
        if not out.tickets:
            return _say(state, "s_pick:gone", "I couldn't find that ticket any more.", awaiting="none",
                        status_candidates=[])
        return _say(state, "s_pick:ok", render_ticket(out.tickets[0], out.latest_update), awaiting="none",
                    status_candidates=[])

    return s_pick


def after_s_pick(state: AgentState) -> str:
    return "route" if state.steps[-1] == "s_pick:unclear" else "__end__"
