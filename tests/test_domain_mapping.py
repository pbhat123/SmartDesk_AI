"""T7.2: who answers what (FR-I2, FR-G1, FR-G2): IT Support and equipment go to the IT worker; HR policy, onboarding
and payroll to the HR worker; a MISC question to no worker, the orchestrator runs the same answering step with no
domain filter; a MISC gap reaches the ticket agent with category MISC. Mock models, no network."""
import pytest

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.llm.mock import MockDraftModel
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM


def ask(stack, text, **kwargs):
    return run_turn(stack.graph, new_thread_id(), text, **kwargs)


def routed(built_index_dir, settings, tmp_path, mode="cite_top", **kwargs):
    return assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM(mode), routed=True, **kwargs)


@pytest.mark.parametrize("question", [
    "How do I request an extra monitor or a docking station?",   # IT equipment allocation (FR-A8)
    "How do I reset my forgotten password?",                      # IT Support
])
def test_it_questions_go_to_the_it_worker(built_index_dir, settings, tmp_path, question):
    stack = routed(built_index_dir, settings, tmp_path)
    state = ask(stack, question)
    assert stack.spy.calls[0][1] == "IT"  # the first search is the IT worker's; a weak match may be retried (FR-I6)
    assert [p.domain for p in state.parts] == ["IT"]


@pytest.mark.parametrize("question", [
    "When is my salary paid each month?",                         # payroll and comp
    "How many casual leave days do I get?",                       # HR policy
])
def test_hr_questions_go_to_the_hr_worker(built_index_dir, settings, tmp_path, question):
    stack = routed(built_index_dir, settings, tmp_path)
    state = ask(stack, question)
    assert stack.spy.calls[0][1] == "HR"
    assert [p.domain for p in state.parts] == ["HR"]


def test_a_misc_question_runs_the_answering_step_with_no_domain_filter(built_index_dir, settings, tmp_path):
    stack = routed(built_index_dir, settings, tmp_path)
    state = ask(stack, "Is there free parking at the Pune office?")
    assert [p.domain for p in state.parts] == ["MISC"]
    assert state.part_result.agent == "orchestrator" and state.part_result.searched_domain is None
    assert [d for _, d in stack.spy.calls] == [None]


def test_the_search_is_tagged_with_the_agent_that_ran_it(built_index_dir, settings, tmp_path):
    """FR-G9: the tool record names the calling agent; a worker calls search_kb itself with its fixed domain."""
    stack = routed(built_index_dir, settings, tmp_path)
    seen = []
    execute = stack.executor.execute
    stack.executor.execute = lambda name, args, *, agent, **kw: (seen.append((name, agent)), execute(name, args, agent=agent, **kw))[1]
    ask(stack, "How do I reset my forgotten password?")
    ask(stack, "When is my salary paid each month?")
    ask(stack, "Is there free parking at the Pune office?")
    assert [a for _, a in seen][:1] == ["it_worker"] and "hr_worker" in [a for _, a in seen]
    assert set(a for _, a in seen) <= {"it_worker", "hr_worker", "orchestrator"} and seen[-1][1] == "orchestrator"


def test_the_whole_kb_retry_is_run_by_the_orchestrator(built_index_dir, settings, tmp_path):
    stack = routed(built_index_dir, settings, tmp_path, mode=["insufficient", "cite_top"])
    seen = []
    execute = stack.executor.execute
    stack.executor.execute = lambda name, args, *, agent, **kw: (seen.append(agent), execute(name, args, agent=agent, **kw))[1]
    state = ask(stack, "How do I reset my forgotten password?")
    assert seen == ["it_worker", "orchestrator"] and state.part_result.retried


def test_a_misc_gap_question_reaches_the_ticket_agent_with_category_misc(built_index_dir, settings, tmp_path):
    stack = routed(built_index_dir, settings, tmp_path, mode="insufficient", drafter_llm=MockDraftModel())
    tid = new_thread_id()
    offer = run_turn(stack.graph, tid, "Is there free parking at the Pune office?")
    assert offer.pending_offer is not None and offer.awaiting == "email"
    draft = run_turn(stack.graph, tid, "priya.k@smartdesk.example")
    assert draft.ticket is not None and draft.ticket.category == "MISC"


def test_without_a_router_the_whole_kb_is_searched_by_the_misc_step(built_index_dir, settings, tmp_path):
    stack = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM())
    state = ask(stack, "How do I reset my forgotten password?")
    assert state.part_result.agent == "orchestrator" and [d for _, d in stack.spy.calls] == [None]
