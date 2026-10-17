"""T4.9: a question that failed in its domain is retried once against the whole KB before any
ticket offer (FR-A19, FR-A14)."""
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from tests.agent_helpers import build_test_graph
from tests.llm_fakes import FakeAnswerLLM

Q = "Is the cafeteria open on weekends?"


def run(built_index_dir, settings, tmp_path, llm, domain):
    graph, spy = build_test_graph(built_index_dir, settings, tmp_path, llm=llm)
    return run_turn(graph, new_thread_id(), Q, search_domain=domain), spy


def test_an_unanswerable_question_gets_exactly_one_extra_no_domain_search(built_index_dir, settings, tmp_path):
    state, spy = run(built_index_dir, settings, tmp_path, FakeAnswerLLM("insufficient"), "IT")
    assert spy.calls == [(Q, "IT"), (Q, None)]
    assert state.part_result.retried is True and state.pending_offer is not None


def test_the_retry_comes_before_the_offer(built_index_dir, settings, tmp_path):
    state, _ = run(built_index_dir, settings, tmp_path, FakeAnswerLLM("insufficient"), "HR")
    assert state.steps == ["start", "search_kb", "answer", "search_kb", "answer", "offer"]


def test_a_retry_that_finds_an_answer_avoids_the_offer(built_index_dir, settings, tmp_path):
    graph, spy = build_test_graph(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM(["insufficient", "cite_top"]))
    state = run_turn(graph, new_thread_id(), "How do I reset my forgotten password?", search_domain="IT")
    assert [d for _, d in spy.calls] == ["IT", None]
    assert state.pending_offer is None and state.awaiting == "none"
    assert state.part_result.retried and "Sources:" in state.reply


def test_the_retry_runs_once_only(built_index_dir, settings, tmp_path):
    _, spy = run(built_index_dir, settings, tmp_path, FakeAnswerLLM("insufficient"), "IT")
    assert len(spy.calls) == 2 and sum(1 for _, d in spy.calls if d is None) == 1


def test_a_whole_kb_search_is_not_repeated(built_index_dir, settings, tmp_path):
    state, spy = run(built_index_dir, settings, tmp_path, FakeAnswerLLM("insufficient"), None)
    assert spy.calls == [(Q, None)]
    assert state.part_result.retried is False and state.pending_offer is not None


def test_an_answered_question_is_not_retried(built_index_dir, settings, tmp_path):
    graph, spy = build_test_graph(built_index_dir, settings, tmp_path)
    run_turn(graph, new_thread_id(), "How do I reset my forgotten password?", search_domain="IT")
    assert len(spy.calls) == 1


def test_low_retrieval_score_also_triggers_the_retry(built_index_dir, settings, tmp_path):
    graph, spy = build_test_graph(built_index_dir, settings, tmp_path)
    state = run_turn(graph, new_thread_id(), "zzqx wvvk plorf", search_domain="HR")
    assert [d for _, d in spy.calls] == ["HR", None]
    assert state.pending_offer is not None


def test_a_search_outage_during_the_retry_replies_politely_without_an_offer(built_index_dir, settings, tmp_path):
    graph, spy = build_test_graph(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"))
    real = spy.fn

    def flaky(args):
        if args.domain is None:
            raise TimeoutError("down")
        return real(args)

    spy.fn = flaky
    state = run_turn(graph, new_thread_id(), Q, search_domain="IT")
    assert "try again later" in state.reply.lower() and state.pending_offer is None
