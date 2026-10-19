"""T7.8: the ticket agent owns create_ticket and get_tickets, runs on its own behind the approval gate, and never
alongside another ticket action; no other agent can reach a ticket tool (FR-I8, FR-G1, FR-G5). Mock models and a
recording ticket client, no network."""
from collections import deque

import pytest

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.llm.mock import MockDraftModel
from smartdesk_ai.tools.executor import ErrorCode, Outcome, ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM
from tests.test_fanout import COMPOUND, ByQuestion, Synth

EMAIL = "priya.k@smartdesk.example"
TICKET_ARGS = {
    "create_ticket": {"title": "t", "description": "d", "category": "IT", "email": EMAIL},
    "get_tickets": {"email": EMAIL},
}


def executor_with_spies(settings):
    ran = []
    registry = ToolRegistry()
    for name in ("search_kb", "create_ticket", "get_tickets"):
        registry.register(name, lambda args, _n=name: ran.append(_n) or {"ok": True})
    return ToolExecutor(registry, settings, sleep=lambda _: None), ran


@pytest.mark.parametrize("agent", ["it_worker", "hr_worker", "orchestrator", "synthesizer"])
@pytest.mark.parametrize("tool", ["create_ticket", "get_tickets"])
def test_no_agent_but_the_ticket_agent_reaches_a_ticket_tool(settings, agent, tool):
    executor, ran = executor_with_spies(settings)
    result = executor.execute(tool, TICKET_ARGS[tool], agent=agent, confirmed=True)
    assert result.outcome is Outcome.VALIDATION_ERROR and result.code is ErrorCode.NOT_ALLOWED
    assert result.attempts == 0 and ran == [], "the tool function never ran, and nothing was retried"


@pytest.mark.parametrize("tool", ["create_ticket", "get_tickets"])
def test_the_ticket_agent_reaches_its_own_tools(settings, tool):
    executor, ran = executor_with_spies(settings)
    result = executor.execute(tool, TICKET_ARGS[tool], agent="ticket_agent", confirmed=True)
    assert result.code is not ErrorCode.NOT_ALLOWED and ran in ([tool], [])


def test_the_ticket_agent_does_not_search_the_kb(settings):
    executor, ran = executor_with_spies(settings)
    result = executor.execute("search_kb", {"query": "vpn"}, agent="ticket_agent")
    assert result.code is ErrorCode.NOT_ALLOWED and ran == []


@pytest.mark.parametrize("agent", ["it_worker", "hr_worker", "orchestrator"])
def test_the_workers_and_the_orchestrator_search_the_kb(settings, agent):
    executor, ran = executor_with_spies(settings)
    assert executor.execute("search_kb", {"query": "vpn"}, agent=agent).outcome is Outcome.SUCCESS and ran == ["search_kb"]


def test_a_refused_call_is_recorded_with_its_agent(settings, tmp_path):
    from smartdesk_ai.ops.metrics import MetricsRecorder

    registry = ToolRegistry()
    registry.register("create_ticket", lambda args: {"ok": True})
    metrics = MetricsRecorder(tmp_path / "m.jsonl")
    ToolExecutor(registry, settings, metrics=metrics).execute("create_ticket", TICKET_ARGS["create_ticket"],
                                                              agent="it_worker", confirmed=True)
    [row] = [__import__("json").loads(l) for l in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert (row["tool"], row["agent"], row["outcome"], row["error_code"]) == (
        "create_ticket", "it_worker", "validation_error", "agent_not_allowed")


# --- in the graph -----------------------------------------------------------------------------------------------

def mock_settings(settings):
    return settings.model_copy(update={"thresholds": settings.thresholds.model_copy(
        update={"escalation": settings.thresholds.escalation_mock})})


def stack(built_index_dir, settings, tmp_path, **kw):
    return assemble(built_index_dir, mock_settings(settings), tmp_path, routed=True, drafter_llm=MockDraftModel(),
                    synth_model=Synth(), **kw)


def reachable(graph, starts):
    edges = graph.get_graph().edges
    seen, queue = set(), deque(starts)
    while queue:
        node = queue.popleft()
        for edge in edges:
            if edge.source == node and edge.target not in seen:
                seen.add(edge.target)
                queue.append(edge.target)
    return seen


def test_no_ticket_node_is_reachable_from_a_worker_the_retry_or_the_synthesizer(built_index_dir, settings, tmp_path):
    graph = stack(built_index_dir, settings, tmp_path).graph
    after_kb = reachable(graph, ["it_worker", "hr_worker", "misc_search", "kb_retry", "retry_gate", "compose", "synthesize"])
    assert not {n for n in after_kb if n.startswith(("t_", "s_"))}, after_kb


def test_a_fan_out_turn_makes_no_ticket_call(built_index_dir, settings, tmp_path):
    s = stack(built_index_dir, settings, tmp_path, llm=ByQuestion())
    calls = []
    execute = s.executor.execute
    s.executor.execute = lambda name, args, *, agent, **kw: (calls.append((name, agent)), execute(name, args, agent=agent, **kw))[1]
    state = run_turn(s.graph, new_thread_id(), COMPOUND)
    assert calls and all(name == "search_kb" for name, _ in calls) and not any(step.startswith(("t_", "s_")) for step in state.steps)
    assert s.client._issues == {} or all(i.number < 5 for i in s.client._issues.values())  # nothing created


def test_a_ticket_turn_runs_no_worker(built_index_dir, settings, tmp_path):
    answer = FakeAnswerLLM()
    s = stack(built_index_dir, settings, tmp_path, llm=answer)
    tid = new_thread_id()
    run_turn(s.graph, tid, "I want to raise a ticket about my broken monitor", email=EMAIL)
    assert answer.calls == [] and s.spy.calls == []


def test_the_ticket_calls_of_a_partial_answer_flow_come_one_after_another(built_index_dir, settings, tmp_path):
    """The offer ends the turn; the ticket agent starts on the next one, so a ticket action never overlaps a KB search."""
    from tests.test_partial_answers import GAP, HR, Answers, scripted
    from smartdesk_ai.agent.confirm import make_confirm_parser
    from smartdesk_ai.llm.mock import MockConfirmModel

    s = assemble(built_index_dir, mock_settings(settings), tmp_path, llm=Answers(), routed=True, route_model=scripted(HR, GAP),
                 synth_model=Synth(), drafter_llm=MockDraftModel(),
                 parse_confirm=make_confirm_parser(settings, builder=lambda m, c: MockConfirmModel()))
    log = []
    execute = s.executor.execute
    s.executor.execute = lambda name, args, *, agent, **kw: (log.append((name, agent)), execute(name, args, agent=agent, **kw))[1]
    tid = new_thread_id()
    run_turn(s.graph, tid, "two questions")
    searches = len(log)
    run_turn(s.graph, tid, EMAIL)
    run_turn(s.graph, tid, "yes")
    assert all(n == "search_kb" for n, _ in log[:searches]) and searches >= 2
    after = log[searches:]
    assert [n for n, _ in after] == ["get_tickets", "create_ticket"] and {a for _, a in after} == {"ticket_agent"}


def test_the_drafter_gets_the_unanswered_part_and_no_earlier_answers(built_index_dir, settings, tmp_path):
    from types import SimpleNamespace

    class Spy:
        def __init__(self):
            self.inputs = []

        def invoke(self, messages):
            self.inputs.append(str(messages[-1].content))
            return MockDraftModel().invoke(messages)

    spy = Spy()
    s = assemble(built_index_dir, mock_settings(settings), tmp_path, llm=FakeAnswerLLM(["cite_top", "insufficient"]),
                 routed=True, drafter_llm=spy, synth_model=Synth())
    tid = new_thread_id()
    run_turn(s.graph, tid, "How do I reset my forgotten password?")
    run_turn(s.graph, tid, "My monitor keeps flickering")
    run_turn(s.graph, tid, EMAIL)
    assert spy.inputs == ["Employee statements:\n- My monitor keeps flickering"]
