"""The agent graph, KB path (FR-A14, FR-A19; PLAN 4.2).

`kb_search` runs `search_kb` through the executor and answers from the chunks. A question the
KB cannot answer is retried once against the whole KB (`kb_retry`, FR-A19) and then goes to `offer`, which is reachable only from here, so retrieval is always
attempted before a ticket is offered (FR-A14). `steps` is the per-turn trace.
"""
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from smartdesk_ai.agent.answer import INSUFFICIENT_REPLY, AnswerError, AnswerResult, should_escalate
from smartdesk_ai.agent.ticket_agent import escalation_offer
from smartdesk_ai.agent.state import AgentState, ChunkRef, Domain, PartResult, PendingOffer, thread_config
from smartdesk_ai.tools.executor import Outcome, ToolExecutor
from smartdesk_ai.tools.search_kb import KBChunk, SearchKBResult

OFFER_AGENT = "kb"


class KBFirstViolation(RuntimeError):
    """A ticket offer was attempted without a prior search_kb call (FR-A14)."""


@dataclass(frozen=True)
class KBDeps:
    executor: ToolExecutor
    answer: Callable[[str, Sequence[KBChunk]], AnswerResult]


def _question(state: AgentState) -> str:
    for message in reversed(state.messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _chunk_refs(found: SearchKBResult) -> list[ChunkRef]:
    return [ChunkRef(entry_id=c.entry_id, title=c.title, score=c.score) for c in found.chunks]


def _search_and_answer(deps: KBDeps, question: str, domain: Domain | None) -> PartResult | str:
    """One search plus one answer. Returns the result, or a polite message after an outage."""
    args: dict[str, Any] = {"query": question}
    if domain is not None:
        args["domain"] = domain
    result = deps.executor.execute("search_kb", args, agent=OFFER_AGENT)
    if result.outcome is not Outcome.SUCCESS:
        return result.reply or "Sorry, I can't look that up right now. Please try again later."
    found: SearchKBResult = result.data
    try:
        answer = deps.answer(question, found.chunks)
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
        if isinstance(outcome, str):
            return {"steps": ["start", "search_kb", "search_failed"], "part_result": None, "reply": outcome,
                    "messages": [AIMessage(outcome)]}
        return {"steps": ["start", "search_kb", "answer"], "part_result": outcome, "reply": outcome.answer}

    return kb_search


def needs_offer(state: AgentState) -> bool:
    part = state.part_result
    return part is not None and (not part.answerable or part.llm_insufficient)


def make_kb_retry_node(deps: KBDeps) -> Callable[[AgentState], dict[str, Any]]:
    """The one whole-KB retry (FR-A19): the same question with no domain filter."""

    def kb_retry(state: AgentState) -> dict[str, Any]:
        outcome = _search_and_answer(deps, _question(state), None)
        if isinstance(outcome, str):
            return {"steps": state.steps + ["search_kb", "search_failed"], "part_result": None, "reply": outcome,
                    "messages": [AIMessage(outcome)]}
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
        "awaiting": "offer_reply",
        "reply": reply,
        "messages": [AIMessage(reply)],
    }


def reply_node(state: AgentState) -> dict[str, Any]:
    return {"steps": state.steps + ["reply"], "messages": [AIMessage(state.reply)]}


def build_kb_graph(deps: KBDeps, checkpointer: Any = None):
    builder = StateGraph(AgentState)
    builder.add_node("kb_search", make_kb_search_node(deps))
    builder.add_node("kb_retry", make_kb_retry_node(deps))
    builder.add_node("offer", offer_node)
    builder.add_node("reply", reply_node)
    builder.add_edge(START, "kb_search")
    builder.add_conditional_edges("kb_search", after_kb_search, ["kb_retry", "offer", "reply", END])
    builder.add_conditional_edges("kb_retry", after_kb_retry, ["offer", "reply", END])
    builder.add_edge("offer", END)
    builder.add_edge("reply", END)
    return builder.compile(checkpointer=checkpointer)


def run_turn(
    graph: Any, thread_id: str, text: str, *, search_domain: Domain | None = None, email: str | None = None
) -> AgentState:
    """Run one employee turn on a thread and return the resulting state. A given `email`
    becomes the session email; without one, the thread keeps what it already knows."""
    config = thread_config(thread_id)
    turn_input: dict[str, Any] = {"messages": [HumanMessage(text)], "search_domain": search_domain}
    if email is not None:
        turn_input["email"] = email
    graph.invoke(turn_input, config)
    return AgentState(**graph.get_state(config).values)
