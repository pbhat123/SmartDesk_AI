"""The agent graph (FR-A14, FR-A19, FR-D8, FR-G12; PLAN 4.2).

Every turn starts at `entry`. A reply to a pending offer or email request goes to `offer_reply`
(FR-A16); anything else goes to `route`, which classifies and picks a path. The KB path is
a worker (`it_worker`, `hr_worker` or `misc_search`, by the part's domain) -> (`kb_retry`) -> `reply` or `offer`; `offer`
is reachable only from the KB nodes,
so retrieval is always attempted before a ticket is offered (FR-A14). The ticket path starts at
`t_collect`. `steps` is the per-turn trace.
"""
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from langgraph.types import Command, Send

from smartdesk_ai.agent.agents import ORCHESTRATOR
from smartdesk_ai.agent.answer import (
    INSUFFICIENT_REPLY,
    AnswerError,
    AnswerResult,
    AnswerUnavailable,
    should_escalate,
)
from smartdesk_ai.agent.chat_service import Turn
from smartdesk_ai.agent.chitchat import Chitchat, static_chitchat
from smartdesk_ai.agent.followup import contextual_question, topic_after
from smartdesk_ai.agent.intents import dialog_update_for
from smartdesk_ai.agent.orchestrator import plan_parts, transcript_for
from smartdesk_ai.agent.router import RouteDecision, Router
from smartdesk_ai.agent.state import (
    AgentState,
    ChunkRef,
    Domain,
    IntentResult,
    PartResult,
    PendingOffer,
    RouteKind,
    SubQuestion,
    thread_config,
)
from smartdesk_ai.agent.status_flow import (
    after_s_email_reply,
    after_s_pick,
    make_s_lookup_node,
    make_s_pick_node,
    s_email_reply,
)
from smartdesk_ai.agent.ticket_agent import (
    TicketDeps,
    escalation_offer,
    partial_reply,
    handle_offer_reply,
    after_t_confirm,
    after_t_dup_ask,
    after_t_dup_check,
    make_t_collect_node,
    make_t_confirm_node,
    make_t_create_node,
    make_t_dup_ask_node,
    make_t_dup_check_node,
    make_t_draft_node,
    make_t_present_node,
    t_apply_edit,
    t_cancel,
    t_dup_reask,
    t_reask,
)
from smartdesk_ai.agent.synthesizer import Synthesizer, join_answers
from smartdesk_ai.agent.workers import AGENT_FOR_NODE, FILTER_FOR_NODE, MISC_SEARCH, NODE_FOR_DOMAIN, WORKER_AGENTS, Answerer
from smartdesk_ai.ops.cache import AnswerCache
from smartdesk_ai.ops.compaction import Compactor
from smartdesk_ai.ops.runtime import current_scope
from smartdesk_ai.tools.executor import Outcome, ToolExecutor
from smartdesk_ai.tools.search_kb import KBChunk, SearchKBResult


logger = logging.getLogger(__name__)

WORKER_FAILED_REPLY = "Sorry, I can't look that up right now. Please try again later."
DROPPED_NOTE = "I've answered the first {done} questions. Please ask the other {dropped} in a separate message."


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
    answer: Callable[[str, Sequence[KBChunk]], AnswerResult]  # the orchestrator's answering step: MISC parts and the retry
    cache: AnswerCache | None = None  # KB answers only; the ticket and status paths never see it
    workers: Mapping[str, Answerer] | None = None  # the answering step of each worker node; absent: `answer` for all
    synthesizer: Synthesizer | None = None  # merges several answered parts; absent: the answers are joined (FR-I4)

    def answerer_for(self, node: str) -> Answerer:
        return (self.workers or {}).get(node, self.answer)


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
    return [ChunkRef(entry_id=c.entry_id, title=c.title, score=c.score, dense=c.dense, rrf=c.rrf, rerank=c.rerank)
            for c in found.chunks]


# --- entry and routing -------------------------------------------------------------------------


def entry_node(state: AgentState) -> dict[str, Any]:
    """Start of every turn: a fresh trace and no leftovers from the previous turn."""
    return {"steps": ["start"], "part_result": None, "parts": [], "parts_dropped": 0, "part_results": None, "part_steps": None, "reply": ""}


def make_compact_node(compactor: Compactor, budget_level: Callable[[], int]) -> Callable[[AgentState], dict[str, Any]]:
    """Before each turn: when the messages fill the trigger share of the active window, replace the older
    ones by a running summary (FR-H5). The summary takes the first message's id, so it is swapped in place;
    the rest are removed. An open ticket conversation, the email and the draft stay as they are."""

    def compact(state: AgentState) -> dict[str, Any]:
        ticket_open = state.ticket is not None or state.awaiting != "none"
        result = compactor.compact(
            state.messages, budget_level=budget_level(), protect_from=state.ticket_context_start if ticket_open else None)
        if result is None:
            return {}
        removed = [RemoveMessage(id=m.id) for m in state.messages[1:result.cut]]
        return {
            "messages": [result.messages[0], *removed],
            "ticket_context_start": max(state.ticket_context_start - (result.cut - 1), 0),
        }

    return compact


def make_after_entry(has_router: bool) -> Callable[[AgentState], str]:
    def after_entry(state: AgentState) -> str:
        if state.awaiting == "email" and state.status_pending_email:
            return "s_email_reply"
        if state.awaiting == "ticket_pick" and state.status_candidates:
            return "s_pick"
        if state.awaiting in ("offer_reply", "email") and state.pending_offer is not None:
            return "offer_reply"
        return "route" if has_router else "single_part"

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
            return "route" if has_router else "single_part"
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


def make_route_node(route: Router, max_parts: int) -> Callable[[AgentState], dict[str, Any]]:
    def route_node(state: AgentState) -> dict[str, Any]:
        decision = route(_question(state), state.awaiting, state.last_topic, history=transcript_for(state.messages))
        intent = None
        if decision.intent is not None:
            intent = IntentResult(
                intent=decision.intent, confidence=decision.confidence or 0.0, route_source=decision.source
            )
        cleared = dialog_update_for(state, decision.intent)
        if cleared:  # a new topic leaves the status flow too
            cleared = {**cleared, "status_pending_email": False, "status_candidates": []}
        kind = _route_kind(decision)
        domain = {"it_worker": "IT", "hr_worker": "HR"}.get(decision.route)
        plan = None
        topic: dict[str, Any] = {}
        if kind == "kb":
            plan = plan_parts(_question(state), state.last_topic,
                              IntentResult(intent="kb_query", confidence=1.0, domain=domain, parts=decision.parts),
                              max_parts=max_parts)
            if plan.parts[-1].rewritten:  # the standalone form is the topic the next follow-up builds on
                topic = {"last_topic": plan.parts[-1].text}
            if len(plan.parts) == 1:
                domain = {"IT": "IT", "HR": "HR"}.get(plan.parts[0].domain)
            else:
                domain = None
        return {
            **cleared,
            **topic,
            "route_kind": kind,
            "search_domain": domain,
            "parts": plan.parts if plan else [],
            "parts_dropped": plan.dropped if plan else 0,
            "intent": intent,
            "steps": state.steps + ["route"],
        }

    return route_node


def after_route(state: AgentState) -> list[Send] | str:
    if state.route_kind == "kb":
        return fan_out(state)
    return {"ticket": "t_collect", "status": "s_lookup"}.get(state.route_kind or "", "chitchat")


def make_chitchat_node(chat: Chitchat) -> Callable[[AgentState], dict[str, Any]]:
    """Small talk, deflection or the clarifying question (FR-D3, D5). No tool is called."""

    def chitchat_node(state: AgentState) -> dict[str, Any]:
        situation = state.route_kind if state.route_kind in ("deflect", "clarify") else "smalltalk"
        reply = chat(situation, _question(state))  # type: ignore[arg-type]
        return {"steps": state.steps + ["chitchat"], "reply": reply, "messages": [AIMessage(reply)]}

    return chitchat_node


# --- KB path ---------------------------------------------------------------------------------


def _search_and_answer(deps: KBDeps, part: SubQuestion, node: str, *, agent: str | None = None) -> PartResult | Unavailable:
    """One search plus one answer for a sub-question, or the cached answer for the same sub-question (a hit skips
    retrieval, the worker and the judge). `node` is the worker that runs it: it fixes the domain filter and the
    answering step; `agent` is who the calls are recorded under (default: the node's own agent, FR-G9). Returns the
    result, or `Unavailable` after an outage."""
    agent = agent or AGENT_FOR_NODE[node]
    domain = FILTER_FOR_NODE[node]
    ident = {"part_id": part.part_id, "sub_question": part.text, "agent": agent}
    probe = deps.cache.lookup(part.text, domain) if deps.cache is not None else None
    if probe is not None and probe.part is not None:
        return probe.part.model_copy(update={**ident, "cache": probe.kind, "cache_similarity": probe.similarity,
                                                "cache_saved_ms": probe.saved_ms})
    started = time.perf_counter()
    args: dict[str, Any] = {"query": part.text}
    if domain is not None:
        args["domain"] = domain
    result = deps.executor.execute("search_kb", args, agent=agent)
    if result.outcome is not Outcome.SUCCESS:
        return Unavailable(result.reply or "Sorry, I can't look that up right now. Please try again later.",
                           "search_failed")
    found: SearchKBResult = result.data
    try:
        answer = deps.answerer_for(node)(part.text, found.chunks)
    except AnswerUnavailable as outage:  # every model tier is down: tell them, never offer a ticket
        return Unavailable(outage.message, "llm_unavailable")
    except AnswerError:
        answer = AnswerResult(text=INSUFFICIENT_REPLY, answer="", insufficient=True)  # signal 3
    if should_escalate(found, answer):  # never show an answer that a signal rejected
        answer = AnswerResult(text=INSUFFICIENT_REPLY, answer="", insufficient=answer.insufficient)
    made = PartResult(
        **ident,
        answer=answer.text,
        body=answer.answer,
        citations=answer.citations,
        chunks=_chunk_refs(found),
        answerable=found.answerable,
        llm_insufficient=answer.insufficient,
        searched_domain=domain,
        top_cosine=found.top_cosine,
        threshold=found.threshold,
    )
    if probe is None or probe.kind == "bypass":
        return made
    deps.cache.store(probe, made, (time.perf_counter() - started) * 1000)
    return made.model_copy(update={"cache": "miss"})


def _kb_steps(steps: list[str], part: PartResult) -> list[str]:
    """The trace of a KB turn: the search and answer steps, or one cache-hit step when the cache served it."""
    return steps + ([f"cache_hit:{part.cache}"] if part.cache in ("exact", "semantic") else ["search_kb", "answer"])


def make_single_part_node() -> Callable[[AgentState], dict[str, Any]]:
    """Without a router there is no orchestrator call: the whole message is one part, searched with the domain the
    caller forced (`search_domain`) or with the whole KB."""

    def single_part(state: AgentState) -> dict[str, Any]:
        text = contextual_question(_question(state), state.last_topic)
        return {"parts": [SubQuestion(part_id="p1", text=text, domain=state.search_domain or "MISC")]}

    return single_part


class WorkerInput(BaseModel):
    """What a worker is sent: its sub-question and nothing else (FR-I3). Plain data, so the checkpoint can hold it."""

    part: SubQuestion


def fan_out(state: AgentState) -> list[Send]:
    """One worker per part, all started in the same step and run in parallel (FR-I4, FR-I2)."""
    return [Send(NODE_FOR_DOMAIN[p.domain], WorkerInput(part=p).model_dump()) for p in state.parts]


def _part_of(outcome: PartResult | Unavailable, part: SubQuestion, agent: str) -> PartResult:
    if isinstance(outcome, Unavailable):
        return PartResult(part_id=part.part_id, sub_question=part.text, agent=agent, answer="", answerable=False,
                          outage=outcome.message, outage_step=outcome.step)
    return outcome


def _steps_of(result: PartResult) -> list[str]:
    if result.outage is not None:
        return ["search_kb", result.outage_step or "unavailable"]
    return _kb_steps([], result)


def make_worker_node(deps: KBDeps, node: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """An IT, HR or MISC worker: answers its part from the KB, or the cached answer for it (FR-I2, FR-G2). It writes only
    its own part's keys, so parts can run in parallel. It never raises: a failure is an outage for its part, because an
    exception out of a parallel step would leave the turn half-run (the next message would be read as a gate reply)."""

    def worker(payload: dict[str, Any]) -> dict[str, Any]:
        part = WorkerInput.model_validate(payload).part  # a Send argument reaches the node as the plain dict it was sent as
        try:
            outcome = _search_and_answer(deps, part, node)
        except Exception as error:  # the type only: the message could echo the employee's text (FR-G11)
            logger.warning("worker %s failed (%s)", node, type(error).__name__)
            outcome = Unavailable(WORKER_FAILED_REPLY, "worker_failed")
        result = _part_of(outcome, part, AGENT_FOR_NODE[node])
        return {"part_results": {part.part_id: result}, "part_steps": {part.part_id: _steps_of(result)}}

    return worker


def make_kb_retry_node(deps: KBDeps) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The one whole-KB retry of a part (FR-A19, FR-I6): the same sub-question with no domain filter, run by the
    orchestrator. Parts retry in parallel, each on its own."""

    def kb_retry(payload: dict[str, Any]) -> dict[str, Any]:
        part = WorkerInput.model_validate(payload).part
        try:
            outcome = _search_and_answer(deps, part, MISC_SEARCH)
        except Exception as error:
            logger.warning("whole-KB retry failed (%s)", type(error).__name__)
            outcome = Unavailable(WORKER_FAILED_REPLY, "worker_failed")
        result = _part_of(outcome, part, AGENT_FOR_NODE[MISC_SEARCH])
        if result.outage is None:
            result = result.model_copy(update={"retried": True})
        return {"part_results": {part.part_id: result}, "part_steps": {part.part_id: _steps_of(result)}}

    return kb_retry


def part_unanswered(result: PartResult) -> bool:
    """Any escalation signal fired for this part (FR-A15)."""
    return not result.answerable or result.llm_insufficient


def _needs_retry(result: PartResult | None) -> bool:
    # Retry once, and only when the first search was filtered; a whole-KB search would repeat itself (FR-I6).
    return (result is not None and result.outage is None and part_unanswered(result)
            and result.searched_domain is not None and not result.retried)


def retry_gate(state: AgentState) -> dict[str, Any]:
    """The join of the fan-out: it runs once every part has settled (and again after the retries)."""
    return {}


def after_retry_gate(state: AgentState) -> list[Send] | str:
    again = [p for p in state.parts if _needs_retry(state.part_results.get(p.part_id))]
    return [Send("kb_retry", WorkerInput(part=p).model_dump()) for p in again] or "compose"


def _results(state: AgentState) -> list[PartResult]:
    return [state.part_results[p.part_id] for p in state.parts]


def make_compose_node() -> Callable[[AgentState], dict[str, Any]]:
    """After the join: fold the parts' traces into the turn's, and set what the turn ends with. One part gives its result
    as before; several leave `part_result` empty (each part is in `part_results`)."""

    def compose(state: AgentState) -> dict[str, Any]:
        results = _results(state)
        steps = list(state.steps)
        if len(results) > 1:
            steps.append(f"fan_out:{len(results)}")
        for part in state.parts:
            steps += state.part_steps.get(part.part_id, [])
        out: dict[str, Any] = {"steps": steps}
        down = next((r for r in results if r.outage is not None), None)
        if down is not None:  # a search or every model tier was down: tell them, never offer a ticket
            return {**out, "part_result": None, "reply": down.outage, "messages": [AIMessage(down.outage)]}
        if len(results) == 1:
            only, part = results[0], state.parts[0]
            out.update({"part_result": only, "reply": only.answer})
            if not part.rewritten:  # a rewritten part already set the topic when the orchestrator planned it
                out["last_topic"] = topic_after(_question(state), state.last_topic)
        return out

    return compose


def after_compose(state: AgentState) -> str:
    results = _results(state)
    if any(r.outage is not None for r in results):
        return END
    answered = [r for r in results if not part_unanswered(r)]
    if len(answered) > 1:
        return "synthesize"  # merged first; an unanswered part is added after it (FR-I7)
    return "offer" if len(answered) < len(results) else "reply"


def after_synthesize(state: AgentState) -> str:
    return "offer" if any(part_unanswered(r) for r in _results(state)) else "reply"


def make_synthesize_node(deps: KBDeps) -> Callable[[AgentState], dict[str, Any]]:
    def synthesize(state: AgentState) -> dict[str, Any]:
        results = [r for r in _results(state) if not part_unanswered(r)]  # the answered parts only (FR-I3, FR-I7)
        outcome = deps.synthesizer.merge(_question(state), results) if deps.synthesizer else join_answers(results)
        steps = state.steps + ["synthesize"] + ([] if outcome.merged else ["synthesize_fallback"])
        return {"steps": steps, "reply": outcome.text}

    return synthesize


def _with_dropped_note(reply: str, state: AgentState) -> str:
    if not state.parts_dropped:
        return reply
    return f"{reply}\n\n{DROPPED_NOTE.format(done=len(state.parts), dropped=state.parts_dropped)}"


def offer_node(state: AgentState) -> dict[str, Any]:
    if "search_kb" not in state.steps:
        raise KBFirstViolation("a ticket may only be offered after search_kb ran (FR-A14)")
    results = _results(state)
    unanswered = [r.sub_question for r in results if part_unanswered(r)]
    answered = [r for r in results if not part_unanswered(r)]
    if len(results) == 1:  # the whole question is a gap: the original wording, the ticket covers the message
        reply = escalation_offer(email_known=bool(state.email))
        pending = PendingOffer(unanswered_text=_question(state))
        context_start = _last_human_index(state)
    else:  # some or all parts of a compound question: the answered ones first, then the gap and the offer (FR-I7)
        head = state.reply if len(answered) > 1 else (answered[0].answer if answered else "")
        gap = partial_reply(unanswered, email_known=bool(state.email))
        reply = f"{head}\n\n{gap}" if head else gap
        pending = PendingOffer(unanswered_text="\n".join(unanswered), unanswered_parts=unanswered)
        context_start = len(state.messages) + 1  # the ticket starts after this reply: it covers the unanswered parts only
    reply = _with_dropped_note(reply, state)
    return {
        "steps": state.steps + ["offer"],
        "pending_offer": pending,
        # The merged offer asks for the email when it is unknown (FR-A16); both states read the same replies.
        "awaiting": "offer_reply" if state.email else "email",
        "ticket_context_start": context_start,
        "reply": reply,
        "messages": [AIMessage(reply)],
    }


def reply_node(state: AgentState) -> dict[str, Any]:
    reply = _with_dropped_note(state.reply, state)
    return {"steps": state.steps + ["reply"], "reply": reply, "messages": [AIMessage(reply)]}


def after_t_collect(state: AgentState) -> str:
    """A ready ticket continues to the draft; a question to the employee ends the turn."""
    return "t_draft" if state.steps[-1] == "t_collect" and state.reply == "" else END


def after_t_draft(state: AgentState) -> str:
    """A drafting outage has already replied; otherwise show the summary."""
    return END if state.reply else "t_dup_check"


def build_kb_graph(
    deps: KBDeps,
    checkpointer: Any = None,
    *,
    route: Router | None = None,
    tickets: TicketDeps | None = None,
    status_max_listed: int | None = None,
    max_parts: int | None = None,
    chitchat: Chitchat | None = None,
    compactor: Compactor | None = None,
    budget_level: Callable[[], int] = lambda: 0,
):
    """Compile the agent graph. Without a `route`, every new message goes to the KB path. Without
    `tickets`, the ticket path stops after collecting the email. With a `route`, `status_max_listed`
    (config: status.max_listed) caps the status list. Without `chitchat`, small talk gets the fixed
    replies (FR-F7). With a `compactor`, each turn starts by compacting long histories (FR-H5)."""
    if route is not None and status_max_listed is None:
        raise ValueError("status_max_listed (config: status.max_listed) is required when routing is on")
    if route is not None and max_parts is None:
        raise ValueError("max_parts (config: orchestrator.max_parts) is required when routing is on")
    builder = StateGraph(AgentState)
    builder.add_node("entry", entry_node)
    builder.add_node("offer_reply", offer_reply_node)
    for worker in WORKER_AGENTS:
        builder.add_node(worker, make_worker_node(deps, worker))
        builder.add_edge(worker, "retry_gate")
    builder.add_node("single_part", make_single_part_node())
    builder.add_conditional_edges("single_part", fan_out, list(WORKER_AGENTS))
    builder.add_node("retry_gate", retry_gate)
    builder.add_node("kb_retry", make_kb_retry_node(deps))
    builder.add_node("compose", make_compose_node())
    builder.add_node("synthesize", make_synthesize_node(deps))
    builder.add_conditional_edges("retry_gate", after_retry_gate, ["kb_retry", "compose"])
    builder.add_edge("kb_retry", "retry_gate")
    builder.add_conditional_edges("compose", after_compose, ["reply", "offer", "synthesize", END])
    builder.add_conditional_edges("synthesize", after_synthesize, ["reply", "offer"])
    builder.add_node("offer", offer_node)
    builder.add_node("reply", reply_node)
    builder.add_node("t_collect", make_t_collect_node(drafting=tickets is not None))
    if compactor is not None:
        builder.add_node("compact", make_compact_node(compactor, budget_level))
        builder.add_edge(START, "compact")
        builder.add_edge("compact", "entry")
    else:
        builder.add_edge(START, "entry")
    builder.add_conditional_edges(
        "entry", make_after_entry(route is not None),
        ["offer_reply", "route", "single_part", "s_email_reply", "s_pick"] if route else ["offer_reply", "single_part"],
    )
    builder.add_conditional_edges(
        "offer_reply", make_after_offer_reply(route is not None),
        ["t_collect", "route", "single_part", END] if route else ["t_collect", "single_part", END],
    )
    if route is not None:
        builder.add_node("route", make_route_node(route, max_parts))
        builder.add_node("chitchat", make_chitchat_node(chitchat or static_chitchat))
        builder.add_node("s_lookup", make_s_lookup_node(deps.executor, status_max_listed))
        builder.add_node("s_email_reply", s_email_reply)
        builder.add_node("s_pick", make_s_pick_node(deps.executor))
        builder.add_conditional_edges("s_pick", after_s_pick, ["route", END])
        builder.add_conditional_edges("route", after_route, [*WORKER_AGENTS, "t_collect", "s_lookup", "chitchat"])
        builder.add_conditional_edges("s_email_reply", after_s_email_reply, ["s_lookup", "route", END])
        builder.add_edge("chitchat", END)
        builder.add_edge("s_lookup", END)
    builder.add_edge("offer", END)
    builder.add_edge("reply", END)
    if tickets is None:
        builder.add_edge("t_collect", END)
    else:
        builder.add_node("t_draft", make_t_draft_node(tickets))
        builder.add_node("t_present", make_t_present_node())
        builder.add_conditional_edges("t_collect", after_t_collect, ["t_draft", END])
        builder.add_node("t_dup_check", make_t_dup_check_node(tickets))
        builder.add_node("t_dup_ask", make_t_dup_ask_node(tickets))
        builder.add_node("t_dup_reask", t_dup_reask)
        builder.add_conditional_edges("t_draft", after_t_draft, ["t_dup_check", END])
        builder.add_conditional_edges("t_dup_check", after_t_dup_check, ["t_dup_ask", "t_present"])
        builder.add_conditional_edges("t_dup_ask", after_t_dup_ask, ["t_present", "t_cancel", "t_dup_reask"])
        builder.add_edge("t_dup_reask", "t_dup_ask")
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


def _part_debug(part: SubQuestion, result: PartResult, *, escalated: bool) -> dict[str, Any]:
    """What the debug panel shows for one part of a compound question (FR-A12)."""
    shown: dict[str, Any] = {
        "part_id": part.part_id, "sub_question": part.text, "domain": part.domain, "agent": result.agent,
        "retried": result.retried, "chunks": [c.model_dump() for c in result.chunks],
        "retrieval": {"top_cosine": result.top_cosine, "threshold": result.threshold,
                      "escalated": escalated and part_unanswered(result)},
    }
    if result.cache is not None:
        shown["cache"] = {"status": "miss" if result.cache == "miss" else "hit",
                          "kind": None if result.cache == "miss" else result.cache,
                          "similarity": result.cache_similarity, "saved_ms": result.cache_saved_ms}
    return shown


def make_kb_responder(graph: Any) -> Callable[[str, str], Turn]:
    """A ChatService responder over the graph. The session ID is the thread ID. The debug
    details are the turn's trace and, when the KB ran, its chunks with scores and the generated
    answer (FR-A12)."""

    def kb_responder(session_id: str, text: str) -> Turn:
        scope = current_scope()
        if scope is not None:  # an email the session already knows identifies the user for this request
            scope[0].set_email(graph.get_state(thread_config(session_id)).values.get("email"))
        state = run_turn(graph, session_id, text)
        if scope is not None:
            scope[0].set_email(state.email)
        part = state.part_result
        debug: dict[str, Any] = {"steps": state.steps}
        if len(state.parts) > 1 and state.part_results:  # a compound question: each part has its own retrieval
            debug["parts"] = [_part_debug(p, state.part_results[p.part_id], escalated=state.pending_offer is not None)
                              for p in state.parts if p.part_id in state.part_results]
        if part is not None:
            debug["chunks"] = [c.model_dump() for c in part.chunks]
            debug["answer"] = part.answer
            debug["retrieval"] = {  # what the UI's debug panel shows beside the chunks
                "top_cosine": part.top_cosine, "threshold": part.threshold,
                "escalated": state.pending_offer is not None and not part.citations}
            if part.cache is not None:  # the cache was consulted for this turn's question
                debug["cache"] = {"status": "miss" if part.cache == "miss" else "hit",
                                  "kind": None if part.cache == "miss" else part.cache,
                                  "similarity": part.cache_similarity, "saved_ms": part.cache_saved_ms}
        return Turn(text=state.reply, debug=debug)

    return kb_responder
