"""T6.9: exact-match cache on the rewritten sub-question, KB answers only (SPEC section 4 cache rules and AC, FR-H7).
Mock models; no network."""
import pytest

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import ChunkRef, PartResult, new_thread_id
from smartdesk_ai.ops.cache import AnswerCache
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM
from tests.ui_helpers import mock_context

QUESTION = "How do I reset my forgotten password?"
EMAIL = "priya.k@smartdesk.example"


def answered(**changes) -> PartResult:
    base = dict(answer="Use the self-service page.\n\nSources: IT-1 (Password)", citations=["IT-1"],
                chunks=[ChunkRef(entry_id="IT-1", title="Password", score=0.8, dense=0.8)], answerable=True,
                top_cosine=0.8, threshold=0.4)
    return PartResult(**(base | changes))


@pytest.fixture
def cache(new_backend):
    return AnswerCache(new_backend(50), kb_version=lambda: "v1")


# --- the cache on its own -------------------------------------------------------------------------


def test_a_stored_answer_is_an_exact_hit_for_the_same_sub_question(cache):
    first = cache.lookup(QUESTION, "IT")
    assert first.kind == "miss" and first.part is None
    cache.store(first, answered(), compute_ms=120.0)
    again = cache.lookup(QUESTION, "IT")
    assert again.kind == "exact" and again.part.answer == answered().answer
    assert 0 < again.saved_ms <= 120.0  # what the miss cost, less the lookup itself


def test_case_and_spacing_do_not_make_a_new_question(cache):
    cache.store(cache.lookup(QUESTION, None), answered(), compute_ms=50.0)
    assert cache.lookup("  how do i RESET my   forgotten password?\n", None).kind == "exact"


def test_a_different_question_or_domain_is_a_miss(cache):
    cache.store(cache.lookup(QUESTION, "IT"), answered(), compute_ms=50.0)
    assert cache.lookup("How many sick days do I get?", "IT").kind == "miss"
    assert cache.lookup(QUESTION, "HR").kind == "miss"
    assert cache.lookup(QUESTION, None).kind == "miss"


def test_a_stored_part_carries_no_cache_flags_of_its_own_turn(cache):
    cache.store(cache.lookup(QUESTION, None), answered(retried=True, cache="miss", cache_saved_ms=3.0), compute_ms=9.0)
    part = cache.lookup(QUESTION, None).part
    assert part.retried is False and part.cache is None and part.cache_saved_ms is None


@pytest.mark.parametrize("part", [
    answered(llm_insufficient=True),  # the answer model or the judge said no (a judge-rejected draft ends here)
    answered(answerable=False),  # retrieval signals 1 or 2
    answered(citations=[]),  # an answer that names no retrieved entry is never shown
])
def test_only_a_shown_grounded_answer_is_stored(cache, part):
    cache.store(cache.lookup(QUESTION, None), part, compute_ms=10.0)
    assert len(cache) == 0 and cache.lookup(QUESTION, None).kind == "miss"


@pytest.mark.parametrize("text", [
    "Reset the password for [PHONE] please",
    "What is the leave policy for a.person@smartdesk.example?",
])
def test_a_question_with_redacted_or_personal_text_bypasses_the_cache(cache, text):
    probe = cache.lookup(text, None)
    assert probe.kind == "bypass"
    cache.store(probe, answered(), compute_ms=10.0)
    assert len(cache) == 0 and cache.lookup(text, None).kind == "bypass"


def test_a_changed_kb_version_makes_old_entries_unreachable(new_backend):
    version = ["v1"]
    cache = AnswerCache(new_backend(50), kb_version=lambda: version[0])
    cache.store(cache.lookup(QUESTION, None), answered(), compute_ms=10.0)
    version[0] = "v2"  # another process re-ingested
    assert cache.lookup(QUESTION, None).kind == "miss"


def test_clear_empties_the_cache(cache):
    cache.store(cache.lookup(QUESTION, None), answered(), compute_ms=10.0)
    cache.clear()
    assert len(cache) == 0 and cache.lookup(QUESTION, None).kind == "miss"


def test_the_oldest_entries_go_first_when_full(new_backend):
    cache = AnswerCache(new_backend(2), kb_version=lambda: "v1")
    for text in ("first question?", "second question?", "third question?"):
        cache.store(cache.lookup(text, None), answered(), compute_ms=10.0)
    assert len(cache) == 2
    assert cache.lookup("first question?", None).kind == "miss"
    assert cache.lookup("third question?", None).kind == "exact"


def test_the_stored_key_and_value_hold_no_question_text(cache):
    cache.store(cache.lookup("a question about hardware", None), answered(), compute_ms=10.0)
    for key, value in cache.backend.entries():
        assert "hardware" not in key and "hardware" not in value


# --- in the graph ---------------------------------------------------------------------------------


class CountingJudge:
    def __init__(self, verdict=True):
        self.verdict, self.calls = verdict, 0

    def __call__(self, question, answer, chunks):
        self.calls += 1
        return self.verdict


def graph_with_cache(built_index_dir, settings, tmp_path, cache, **kwargs):
    judge = kwargs.pop("judge", None) or CountingJudge()
    llm = kwargs.pop("llm", None) or FakeAnswerLLM()
    return assemble(built_index_dir, settings, tmp_path, llm=llm, judge=judge, cache=cache, **kwargs), llm, judge


def test_a_repeated_question_skips_retrieval_the_worker_and_the_judge(built_index_dir, settings, tmp_path, cache):
    stack, llm, judge = graph_with_cache(built_index_dir, settings, tmp_path, cache)
    first = run_turn(stack.graph, new_thread_id(), QUESTION)
    second = run_turn(stack.graph, new_thread_id(), QUESTION)  # another session: the cache is shared
    assert first.part_result.cache == "miss" and second.part_result.cache == "exact"
    assert second.reply == first.reply and "Sources:" in second.reply
    assert (len(stack.spy.calls), len(llm.calls), judge.calls) == (1, 1, 1)
    assert "cache_hit:exact" in second.steps and "search_kb" not in second.steps
    assert second.part_result.chunks == first.part_result.chunks  # the debug panel still has its chunks


def test_a_judge_rejected_answer_is_never_cached(built_index_dir, settings, tmp_path, cache):
    stack, llm, judge = graph_with_cache(built_index_dir, settings, tmp_path, cache, judge=CountingJudge(False))
    states = [run_turn(stack.graph, new_thread_id(), QUESTION) for _ in range(2)]
    assert [s.part_result.cache for s in states] == ["miss", "miss"] and judge.calls == 2 and len(cache) == 0
    assert all(s.pending_offer is not None for s in states)  # still the ticket offer, twice


def test_an_insufficient_answer_and_an_unanswerable_question_are_never_cached(built_index_dir, settings, tmp_path, cache):
    stack, llm, _ = graph_with_cache(built_index_dir, settings, tmp_path, cache, llm=FakeAnswerLLM("insufficient"))
    for text in (QUESTION, "zzqx wvvk plorf"):
        for _ in range(2):
            assert run_turn(stack.graph, new_thread_id(), text).part_result.cache == "miss"
    assert len(cache) == 0


def test_a_search_outage_is_not_cached(built_index_dir, settings, tmp_path, cache):
    stack, _, _ = graph_with_cache(built_index_dir, settings, tmp_path, cache)
    working = stack.spy.fn
    stack.spy.fn = lambda args: (_ for _ in ()).throw(TimeoutError("down"))
    run_turn(stack.graph, new_thread_id(), QUESTION)
    stack.spy.fn = working
    assert run_turn(stack.graph, new_thread_id(), QUESTION).part_result.cache == "miss"


def test_follow_ups_are_keyed_on_the_rewritten_question_so_they_are_not_mixed_up(built_index_dir, settings, tmp_path, cache):
    stack, _, _ = graph_with_cache(built_index_dir, settings, tmp_path, cache)
    one, two = new_thread_id(), new_thread_id()
    run_turn(stack.graph, one, "How do I reset my forgotten password?")
    run_turn(stack.graph, two, "How do I connect to the office VPN?")
    a = run_turn(stack.graph, one, "what about the wifi?")
    b = run_turn(stack.graph, two, "what about the wifi?")  # same words, another topic: not a hit
    assert a.part_result.cache == "miss" and b.part_result.cache == "miss"
    assert run_turn(stack.graph, new_thread_id(), "How do I reset my forgotten password?").part_result.cache == "exact"


def test_ticket_and_status_turns_never_touch_the_cache(built_index_dir, settings, tmp_path, cache):
    from smartdesk_ai.llm.mock import MockDraftModel

    stack, _, _ = graph_with_cache(built_index_dir, settings, tmp_path, cache, routed=True, seeded=True,
                                   drafter_llm=MockDraftModel())
    tid = new_thread_id()
    for text in ("What is the status of my ticket?", EMAIL, "What is the status of my ticket?",
                 "Please raise a ticket for my broken laptop screen"):
        state = run_turn(stack.graph, tid, text)
        assert state.part_result is None or state.part_result.cache is None, text
    assert len(cache) == 0 and stack.spy.calls == []


# --- through the chat service (cost) --------------------------------------------------------------


def test_a_cache_hit_adds_no_cost_beyond_the_orchestrator_call(tmp_path):
    ctx = mock_context(tmp_path)
    first = ctx.service.handle("s1", QUESTION)
    second = ctx.service.handle("s2", QUESTION)
    assert first.debug["cache"]["status"] == "miss" and second.debug["cache"]["status"] == "hit"
    assert second.debug["cache"]["kind"] == "exact" and second.text == first.text

    def agents(session):
        return {e.agent for e in ctx.service.services.cost.entries if e.session_id == session}

    assert {"it_worker", "grounding_judge"} <= agents("s1") and "retrieval" in agents("s1")
    assert agents("s2") and not agents("s2") & {"it_worker", "grounding_judge", "retrieval"}
    assert "search_kb" not in second.debug["steps"]
