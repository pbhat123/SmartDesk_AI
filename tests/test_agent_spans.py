"""T7.9: every agent is its own span in the trace, with its tier and cost (FR-I9, FR-H7, FR-H9). Mock and scripted
models and a priced cost tracker, no network."""
import json
import threading

import pytest

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.guardrails.pii import PiiRedactor
from smartdesk_ai.llm.mock import MockDraftModel
from smartdesk_ai.ops.cost import CostTracker
from smartdesk_ai.ops.runtime import RequestContext, Services, agent_span, request_scope
from smartdesk_ai.ops.trace_log import TraceLog, make_masker
from tests.agent_helpers import assemble
from tests.test_fanout import COMPOUND, ByQuestion, Synth
from tests.test_misroute_retry import NeedsEntry, scripted, mock_settings

EMAIL = "jane.roe@example.com"


def traced(settings, tmp_path):
    trace = TraceLog(tmp_path / "t.jsonl", make_masker(PiiRedactor(None, 0.5)), prompt_versions=settings.prompts)
    cost = CostTracker(settings, priced=True)
    return trace, cost, Services(cost=cost, trace=trace)


def spans(trace):
    return [e for e in trace.events if e["kind"] == "agent_span"]


def run(stack, services, text, tid=None, request_id="r1", **kw):
    ctx = RequestContext(session_id="s1", request_id=request_id)
    with request_scope(ctx, services):
        return run_turn(stack.graph, tid or new_thread_id(), text, **kw)


def routed(built_index_dir, settings, tmp_path, **kw):
    kw.setdefault("llm", ByQuestion())
    return assemble(built_index_dir, mock_settings(settings), tmp_path, routed=True, synth_model=Synth(), **kw)


def test_a_compound_run_shows_each_agent_as_its_own_span_with_tier_and_cost(built_index_dir, settings, tmp_path):
    trace, cost, services = traced(settings, tmp_path)
    run(routed(built_index_dir, settings, tmp_path), services, COMPOUND)
    found = {(s["agent"], s.get("part_id")): s for s in spans(trace)}
    assert set(found) == {("orchestrator", None), ("it_worker", "p1"), ("hr_worker", "p2"), ("synthesizer", None)}
    for span in found.values():
        assert span["tiers"] == ["main"] and span["calls"] >= 1 and span["input_tokens"] > 0
        assert span["usd"] > 0 and span["latency_ms"] >= 0 and span["request_id"] == "r1" and span["session_id"] == "s1"


def test_the_span_costs_add_up_to_the_request_cost(built_index_dir, settings, tmp_path):
    trace, cost, services = traced(settings, tmp_path)
    run(routed(built_index_dir, settings, tmp_path), services, COMPOUND)
    assert sum(s["usd"] for s in spans(trace)) == pytest.approx(cost.total)
    assert sum(s["calls"] for s in spans(trace)) == sum(e.step == "llm" for e in cost.entries)


def test_a_single_question_has_an_orchestrator_span_and_one_worker_span_and_no_synthesizer(built_index_dir, settings, tmp_path):
    trace, _, services = traced(settings, tmp_path)
    run(routed(built_index_dir, settings, tmp_path), services, "How do I reset my forgotten password?")
    agents = {s["agent"] for s in spans(trace)}
    assert {"orchestrator", "it_worker"} <= agents and "synthesizer" not in agents
    assert [s["calls"] for s in spans(trace) if s["agent"] == "it_worker"] == [1]


def test_each_parallel_worker_counts_only_its_own_calls(built_index_dir, settings, tmp_path):
    trace, _, services = traced(settings, tmp_path)
    run(routed(built_index_dir, settings, tmp_path, llm=ByQuestion(barrier=threading.Barrier(2))), services, COMPOUND)
    workers = {s["part_id"]: s for s in spans(trace) if s["agent"] in ("it_worker", "hr_worker")}
    assert set(workers) == {"p1", "p2"} and workers["p1"]["calls"] == workers["p2"]["calls"] == 1


def test_the_grounding_judge_call_belongs_to_the_workers_span(built_index_dir, settings, tmp_path):
    from smartdesk_ai.agent.grounding import make_grounding_judge
    from smartdesk_ai.llm.mock import MockGroundingModel

    trace, cost, services = traced(settings, tmp_path)
    judge = make_grounding_judge(settings, builder=lambda m, c: MockGroundingModel())
    run(routed(built_index_dir, settings, tmp_path, judge=judge), services, COMPOUND)
    workers = [s for s in spans(trace) if s["agent"] in ("it_worker", "hr_worker")]
    assert all(s["calls"] == 2 for s in workers), "the answer call and the judge call"
    assert {e.agent for e in cost.entries} >= {"grounding_judge"}


def test_a_retry_is_the_orchestrators_span_for_that_part(built_index_dir, settings, tmp_path):
    trace, _, services = traced(settings, tmp_path)
    q = "How do I reset my forgotten password?"
    stack = assemble(built_index_dir, mock_settings(settings), tmp_path, llm=NeedsEntry({"password": "IT-PWD-01"}), routed=True,
                     route_model=scripted([(q, "HR")]), synth_model=Synth())
    run(stack, services, q)
    mine = [(s["agent"], s.get("part_id")) for s in spans(trace)]
    assert ("hr_worker", "p1") in mine and mine.count(("orchestrator", "p1")) == 1


def test_the_ticket_agent_is_a_span_on_the_light_tier(built_index_dir, settings, tmp_path):
    trace, _, services = traced(settings, tmp_path)
    stack = assemble(built_index_dir, mock_settings(settings), tmp_path, routed=True, drafter_llm=MockDraftModel(), llm=ByQuestion())
    tid = new_thread_id()
    run(stack, services, "I want to raise a ticket about my broken monitor", tid=tid, email=EMAIL)
    ticket = [s for s in spans(trace) if s["agent"] == "ticket_agent"]
    assert ticket and any(s["tiers"] == ["light"] and s["calls"] >= 1 for s in ticket)
    assert any(s["tool_calls"] >= 1 for s in ticket), "the duplicate check is a get_tickets call of the ticket agent"


def test_a_span_holds_no_text_and_no_raw_email(built_index_dir, settings, tmp_path):
    trace, _, services = traced(settings, tmp_path)
    stack = routed(built_index_dir, settings, tmp_path)
    run(stack, services, f"{COMPOUND} my email is {EMAIL}")
    for span in spans(trace):
        assert set(span) <= {"kind", "ts", "agent", "part_id", "tiers", "calls", "tool_calls", "input_tokens",
                             "output_tokens", "usd", "latency_ms", "session_id", "request_id", "user_id"}
    assert EMAIL not in (tmp_path / "t.jsonl").read_text()


def test_outside_a_request_scope_a_span_does_nothing():
    with agent_span("it_worker", "p1") as span:
        assert span is None


def test_a_span_that_saw_no_call_is_not_written(built_index_dir, settings, tmp_path):
    trace, _, services = traced(settings, tmp_path)
    with request_scope(RequestContext(session_id="s", request_id="r"), services):
        with agent_span("ticket_agent"):
            pass
    assert spans(trace) == []


def test_spans_reach_the_local_log_file(built_index_dir, settings, tmp_path):
    trace, _, services = traced(settings, tmp_path)
    run(routed(built_index_dir, settings, tmp_path), services, COMPOUND)
    rows = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert {r["agent"] for r in rows if r["kind"] == "agent_span"} >= {"it_worker", "hr_worker", "synthesizer"}
