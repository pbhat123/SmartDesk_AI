"""T4.8: KB first. Retrieval is always attempted before a ticket is offered (FR-A14)."""
import pytest

from evals.common import EVALS_DIR, load_jsonl
from evals.schemas import GapQuestion
from smartdesk_ai.agent.graph import KBFirstViolation, offer_node, run_turn
from smartdesk_ai.agent.state import AgentState, new_thread_id
from tests.agent_helpers import build_test_graph
from tests.llm_fakes import FakeAnswerLLM

GAPS = [GapQuestion(**r) for r in load_jsonl(EVALS_DIR / "gap_questions.jsonl")]


def test_a_ticket_offer_is_always_preceded_by_a_search_kb_call(built_index_dir, settings, tmp_path):
    graph, spy = build_test_graph(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"))
    offers = 0
    for gap in GAPS:
        state = run_turn(graph, new_thread_id(), gap.question)
        assert state.pending_offer is not None and state.awaiting == "email", gap.id  # the offer asks for the email
        assert state.steps.index("search_kb") < state.steps.index("offer"), gap.id
        offers += 1
    assert offers == len(GAPS) and len(spy.calls) >= len(GAPS)


def test_the_search_ran_for_the_offered_question(built_index_dir, settings, tmp_path):
    graph, spy = build_test_graph(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"))
    state = run_turn(graph, new_thread_id(), "Is there parking near the office?")
    assert spy.calls[0][0] == "Is there parking near the office?"
    assert state.pending_offer.unanswered_text == "Is there parking near the office?"


def test_the_offer_node_refuses_to_run_without_a_prior_search():
    with pytest.raises(KBFirstViolation):
        offer_node(AgentState(steps=["start"]))


def test_an_answered_question_makes_no_offer(built_index_dir, settings, tmp_path):
    graph, _ = build_test_graph(built_index_dir, settings, tmp_path)
    state = run_turn(graph, new_thread_id(), "How do I reset my forgotten password?")
    assert state.pending_offer is None and state.awaiting == "none"
    assert "offer" not in state.steps and "Sources:" in state.reply


def test_the_only_way_into_the_offer_node_is_through_a_kb_search(built_index_dir, settings, tmp_path):
    """The offer follows `compose`, which follows the retry gate, which only the workers and the retry feed (T4.9, T7.4)."""
    graph, _ = build_test_graph(built_index_dir, settings, tmp_path)
    edges = graph.get_graph().edges
    into = lambda target: {e.source for e in edges if e.target == target}  # noqa: E731
    assert into("offer") == {"compose", "synthesize"} and into("synthesize") == {"compose"}  # merged answers, then the gap
    assert into("compose") == {"retry_gate"}
    assert into("retry_gate") == {"it_worker", "hr_worker", "misc_search", "kb_retry"}
    assert into("kb_retry") == {"retry_gate"}


def test_the_trace_is_per_turn(built_index_dir, settings, tmp_path):
    graph, _ = build_test_graph(built_index_dir, settings, tmp_path)
    tid = new_thread_id()
    first = run_turn(graph, tid, "How do I reset my forgotten password?")
    second = run_turn(graph, tid, "How do I reset my forgotten password?")
    assert first.steps == second.steps


def test_a_search_outage_replies_politely_and_offers_nothing(built_index_dir, settings, tmp_path):
    graph, spy = build_test_graph(built_index_dir, settings, tmp_path)
    spy.fn = lambda args: (_ for _ in ()).throw(TimeoutError("down"))
    state = run_turn(graph, new_thread_id(), "How do I reset my forgotten password?")
    assert "try again later" in state.reply.lower()
    assert state.pending_offer is None and "offer" not in state.steps


def test_a_rejected_retrieval_never_shows_the_model_answer(built_index_dir, settings, tmp_path):
    graph, _ = build_test_graph(built_index_dir, settings, tmp_path)  # the mock model answers anything
    state = run_turn(graph, new_thread_id(), "zzqx wvvk plorf")
    assert state.part_result.answerable is False
    assert "Here is what the policy says." not in state.reply
    assert "I don't know" in state.reply and state.pending_offer is not None
