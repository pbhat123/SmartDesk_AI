"""The agent graph (FR-A14, FR-A19, FR-D8, FR-G12; PLAN 4.2).

Every turn starts at `entry`. A reply to a pending offer or email request goes to `offer_reply`
(FR-A16); anything else goes to `route`, which classifies and picks a path. The KB path is
`kb_search` -> (`kb_retry`) -> `reply` or `offer`; `offer` is reachable only from the KB nodes,
so retrieval is always attempted before a ticket is offered (FR-A14). The ticket path starts at
`t_collect`. `steps` is the per-turn trace.
"""
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from smartdesk_ai.agent.answer import (
    INSUFFICIENT_REPLY,
    AnswerError,
    AnswerResult,
    AnswerUnavailable,
    should_escalate,
)
from smartdesk_ai.agent.chat_service import Turn
from smartdesk_ai.agent.intents import dialog_update_for
from smartdesk_ai.agent.router import (
    SMALLTALK_REPLY,
    RouteDecision,
    Router,
    clarifying_question,
    deflection_reply,
)
from smartdesk_ai.agent.state import (
    AgentState,
    ChunkRef,
    Domain,
    IntentResult,
    PartResult,
    PendingOffer,
    RouteKind,
    thread_config,
)
from smartdesk_ai.agent.ticket_agent import (
    TicketDeps,
    escalation_offer,
    handle_offer_reply,
    after_t_confirm,
    make_t_collect_node,
    make_t_confirm_node,
    make_t_create_node,
    make_t_draft_node,
    make_t_present_node,
    t_apply_edit,
    t_cancel,
    t_reask,
)
from smartdesk_ai.tools.executor import Outcome, ToolExecutor
from smartdesk_ai.tools.search_kb import KBChunk, SearchKBResult

OFFER_AGENT = "kb"
# Placeholder until the status flow (T4.22 to T4.26).
STATUS_PLACEHOLDER = "Ticket status checks are not available yet."


class KBFirstViolation(RuntimeError):
    """A ticket offer was attempted without a prior search_kb call (FR-A14)."""


@dataclass(frozen=True)
class Unavailable:
    """A search or the answer model could not be reached; `step` names it in the trace."""

    message: str
    step: str


@dataclass(frozen=True)
class KBDeps:
    executor: ToolExecutor
    answer: Callable[[str, Sequence[KBChunk]], AnswerResult]


def _question(state: AgentState) -> str:
    for message in reversed(state.messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _last_human_index(state: AgentState) -> int:
    for index in range(len(state.messages) - 1, -1, -1):
        if isinstance(state.messages[index], HumanMessage):
            return index
    return 0


def _chunk_refs(found: SearchKBResult) -> list[ChunkRef]:
    return [ChunkRef(entry_id=c.entry_id, title=c.title, score=c.score) for c in found.chunks]


# --- entry and routing -------------------------------------------------------------------------


def entry_node(state: AgentState) -> dict[str, Any]:
    """Start of every turn: a fresh trace and no leftovers from the previous turn."""
    return {"steps": ["start"], "part_result": None, "reply": ""}


def make_after_entry(has_router: bool) -> Callable[[AgentState], str]:
    def after_entry(state: AgentState) -> str:
        if state.awaiting in ("offer_reply", "email") and state.pending_offer is not None:
            return "offer_reply"
        return "route" if has_router else "kb_search"

    return after_entry


def offer_reply_node(state: AgentState) -> dict[str, Any]:
    """Read the reply to a ticket offer or email request (FR-A16). Anything else is a new turn."""
    out = handle_offer_reply(state, _question(state))
    steps = state.steps + [f"offer_reply:{out.kind}"]
    if out.kind == "unclear":
        return {"steps": steps}
    update = dict(out.update)
    if out.kind == "declined":
        update["ticket_context_start"] = len(state.messages)
    if out.kind == "agreed":  # the next node speaks; this reply would only be a second message
        agreed = IntentResult(intent="create_ticket", confidence=1.0)
        return {**update, "intent": agreed, "steps": steps}
    return {**update, "steps": steps, "reply": out.reply, "messages": [AIMessage(out.reply)]}


def make_after_offer_reply(has_router: bool) -> Callable[[AgentState], str]:
    def after_offer_reply(state: AgentState) -> str:
        last = state.steps[-1]
        if last == "offer_reply:agreed":
            return "t_collect"
        if last == "offer_reply:unclear":
            return "route" if has_router else "kb_search"
        return END

    return after_offer_reply


def _route_kind(decision: RouteDecision) -> RouteKind:
    if decision.clarify:
        return "clarify"
    if decision.deflect:
        return "deflect"
    return {"kb_query": "kb", "create_ticket": "ticket", "ticket_status": "status"}.get(
        decision.intent or "", "smalltalk"
    )


def make_route_node(route: Router) -> Callable[[AgentState], dict[str, Any]]:
    def route_node(state: AgentState) -> dict[str, Any]:
        decision = route(_question(state), state.awaiting, state.last_topic)
        intent = None
        if decision.intent is not None:
            intent = IntentResult(
                intent=decision.intent, confidence=decision.confidence or 0.0, route_source=decision.source
            )
        return {
            **dialog_update_for(state, decision.intent),
            "route_kind": _route_kind(decision),
            "search_domain": {"it_worker": "IT", "hr_worker": "HR"}.get(decision.route),
            "intent": intent,
            "steps": state.steps + ["route"],
        }

    return route_node


def after_route(state: AgentState) -> str:
    return {"kb": "kb_search", "ticket": "t_collect", "status": "status_stub"}.get(state.route_kind or "", "chitchat")


def chitchat_node(state: AgentState) -> dict[str, Any]:
    """Placeholder replies until the CHITCHAT task (T4.27)."""
    reply = {"deflect": deflection_reply(), "clarify": clarifying_question()}.get(
        state.route_kind or "", SMALLTALK_REPLY
    )
    return {"steps": state.steps + ["chitchat"], "reply": reply, "messages": [AIMessage(reply)]}


def status_stub_node(state: AgentState) -> dict[str, Any]:
    return {"steps": state.steps + ["status"], "reply": STATUS_PLACEHOLDER,
            "messages": [AIMessage(STATUS_PLACEHOLDER)]}


# --- KB path ---------------------------------------------------------------------------------


def _search_and_answer(deps: KBDeps, question: str, domain: Domain | None) -> PartResult | Unavailable:
    """One search plus one answer. Returns the result, or `Unavailable` after an outage."""
    args: dict[str, Any] = {"query": question}
    if domain is not None:
        args["domain"] = domain
    result = deps.executor.execute("search_kb", args, agent=OFFER_AGENT)
    if result.outcome is not Outcome.SUCCESS:
        return Unavailable(result.reply or "Sorry, I can't look that up right now. Please try again later.",
                           "search_failed")
    found: SearchKBResult = result.data
    try:
        answer = deps.answer(question, found.chunks)
    except AnswerUnavailable as outage:  # every model tier is down: tell them, never offer a ticket
        return Unavailable(outage.message, "llm_unavailable")
    except AnswerError:
        answer = AnswerResult(text=INSUFFICIENT_REPLY, answer="", insufficient=True)  # signal 3
    if should_escalate(found, answer):  # never show an answer that a signal rejected
        answer = AnswerResult(text=INSUFFICIENT_REPLY, answer="", insufficient=answer.insufficient)
    return PartResult(
        answer=answer.text,
        citations=answer.citations,
        chunks=_chunk_refs(found),
        answerable=found.answerable,
        llm_insufficient=answer.insufficient,
        searched_domain=domain,
    )


def make_kb_search_node(deps: KBDeps) -> Callable[[AgentState], dict[str, Any]]:
    def kb_search(state: AgentState) -> dict[str, Any]:
        outcome = _search_and_answer(deps, _question(state), state.search_domain)
        if isinstance(outcome, Unavailable):
            return {"steps": state.steps + ["search_kb", outcome.step], "part_result": None,
                    "reply": outcome.message, "messages": [AIMessage(outcome.message)]}
        return {"steps": state.steps + ["search_kb", "answer"], "part_result": outcome, "reply": outcome.answer}

    return kb_search


def needs_offer(state: AgentState) -> bool:
    part = state.part_result
    return part is not None and (not part.answerable or part.llm_insufficient)


def make_kb_retry_node(deps: KBDeps) -> Callable[[AgentState], dict[str, Any]]:
    """The one whole-KB retry (FR-A19): the same question with no domain filter."""

    def kb_retry(state: AgentState) -> dict[str, Any]:
        outcome = _search_and_answer(deps, _question(state), None)
        if isinstance(outcome, Unavailable):
            return {"steps": state.steps + ["search_kb", outcome.step], "part_result": None,
                    "reply": outcome.message, "messages": [AIMessage(outcome.message)]}
        part = outcome.model_copy(update={"retried": True})
        return {"steps": state.steps + ["search_kb", "answer"], "part_result": part, "reply": part.answer}

    return kb_retry


def after_kb_search(state: AgentState) -> str:
    part = state.part_result
    if part is None:
        return END
    if not needs_offer(state):
        return "reply"
    # Retry once, and only when the first search was filtered; a whole-KB search would repeat itself.
    return "kb_retry" if part.searched_domain is not None and not part.retried else "offer"


def after_kb_retry(state: AgentState) -> str:
    if state.part_result is None:
        return END
    return "offer" if needs_offer(state) else "reply"


def offer_node(state: AgentState) -> dict[str, Any]:
    if "search_kb" not in state.steps:
        raise KBFirstViolation("a ticket may only be offered after search_kb ran (FR-A14)")
    question = _question(state)
    reply = escalation_offer(email_known=bool(state.email))
    return {
        "steps": state.steps + ["offer"],
        "pending_offer": PendingOffer(unanswered_text=question),
        # The merged offer asks for the email when it is unknown (FR-A16); both states read the same replies.
        "awaiting": "offer_reply" if state.email else "email",
        "ticket_context_start": _last_human_index(state),  # the ticket covers the unanswered question on
        "reply": reply,
        "messages": [AIMessage(reply)],
    }


def reply_node(state: AgentState) -> dict[str, Any]:
    return {"steps": state.steps + ["reply"], "messages": [AIMessage(state.reply)]}


def after_t_collect(state: AgentState) -> str:
    """A ready ticket continues to the draft; a question to the employee ends the turn."""
    return "t_draft" if state.steps[-1] == "t_collect" and state.reply == "" else END


def after_t_draft(state: AgentState) -> str:
    """A drafting outage has already replied; otherwise show the summary."""
    return END if state.reply else "t_present"


def build_kb_graph(
    deps: KBDeps, checkpointer: Any = None, *, route: Router | None = None, tickets: TicketDeps | None = None
):
    """Compile the agent graph. Without a `route`, every new message goes to the KB path. Without
    `tickets`, the ticket path stops after collecting the email."""
    builder = StateGraph(AgentState)
    builder.add_node("entry", entry_node)
    builder.add_node("offer_reply", offer_reply_node)
    builder.add_node("kb_search", make_kb_search_node(deps))
    builder.add_node("kb_retry", make_kb_retry_node(deps))
    builder.add_node("offer", offer_node)
    builder.add_node("reply", reply_node)
    builder.add_node("t_collect", make_t_collect_node(drafting=tickets is not None))
    builder.add_edge(START, "entry")
    builder.add_conditional_edges("entry", make_after_entry(route is not None),
                                  ["offer_reply", "route", "kb_search"] if route else ["offer_reply", "kb_search"])
    builder.add_conditional_edges(
        "offer_reply", make_after_offer_reply(route is not None),
        ["t_collect", "route", "kb_search", END] if route else ["t_collect", "kb_search", END],
    )
    if route is not None:
        builder.add_node("route", make_route_node(route))
        builder.add_node("chitchat", chitchat_node)
        builder.add_node("status_stub", status_stub_node)
        builder.add_conditional_edges("route", after_route, ["kb_search", "t_collect", "status_stub", "chitchat"])
        builder.add_edge("chitchat", END)
        builder.add_edge("status_stub", END)
    builder.add_conditional_edges("kb_search", after_kb_search, ["kb_retry", "offer", "reply", END])
    builder.add_conditional_edges("kb_retry", after_kb_retry, ["offer", "reply", END])
    builder.add_edge("offer", END)
    builder.add_edge("reply", END)
    if tickets is None:
        builder.add_edge("t_collect", END)
    else:
        builder.add_node("t_draft", make_t_draft_node(tickets))
        builder.add_node("t_present", make_t_present_node())
        builder.add_conditional_edges("t_collect", after_t_collect, ["t_draft", END])
        builder.add_conditional_edges("t_draft", after_t_draft, ["t_present", END])
        builder.add_node("t_confirm", make_t_confirm_node(tickets))
        builder.add_node("t_apply_edit", t_apply_edit)
        builder.add_node("t_reask", t_reask)
        builder.add_node("t_create", make_t_create_node(tickets))
        builder.add_node("t_cancel", t_cancel)
        builder.add_edge("t_present", "t_confirm")
        builder.add_conditional_edges(
            "t_confirm", after_t_confirm, ["t_create", "t_cancel", "t_apply_edit", "t_reask"]
        )
        builder.add_edge("t_apply_edit", "t_present")
        builder.add_edge("t_reask", "t_confirm")
        builder.add_edge("t_create", END)
        builder.add_edge("t_cancel", END)
    return builder.compile(checkpointer=checkpointer)


def run_turn(
    graph: Any, thread_id: str, text: str, *, search_domain: Domain | None = None, email: str | None = None
) -> AgentState:
    """Run one employee turn on a thread and return the resulting state. A given `email`
    becomes the session email; without one, the thread keeps what it already knows."""
    config = thread_config(thread_id)
    if graph.get_state(config).next:  # paused at the approval gate: this message is the answer
        fresh = {"messages": [HumanMessage(text)], "steps": ["start"], "reply": "", "part_result": None}
        graph.invoke(Command(resume=text, update=fresh), config)
        return AgentState(**graph.get_state(config).values)
    turn_input: dict[str, Any] = {"messages": [HumanMessage(text)], "search_domain": search_domain}
    if email is not None:
        turn_input["email"] = email
    graph.invoke(turn_input, config)
    return AgentState(**graph.get_state(config).values)


def make_kb_responder(graph: Any) -> Callable[[str, str], Turn]:
    """A ChatService responder over the graph. The session ID is the thread ID. The debug
    details are the turn's trace and, when the KB ran, its chunks with scores and the generated
    answer (FR-A12)."""

    def kb_responder(session_id: str, text: str) -> Turn:
        state = run_turn(graph, session_id, text)
        part = state.part_result
        debug: dict[str, Any] = {"steps": state.steps}
        if part is not None:
            debug["chunks"] = [c.model_dump() for c in part.chunks]
            debug["answer"] = part.answer
        return Turn(text=state.reply, debug=debug)

    return kb_responder
