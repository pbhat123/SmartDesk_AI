"""T7.6: whole-KB retry per part and misroute recovery (FR-I6, FR-A19, FR-A14). Scripted models, no network.

A part searched with a domain filter that cannot answer it is retried once with no filter, by the orchestrator, before
anything is offered. Parts retry independently and in parallel. A part that is still unanswered is marked so."""
import json
import re
import threading
from types import SimpleNamespace

import pytest

from evals.common import EVALS_DIR, load_jsonl
from evals.schemas import CompoundRow
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.llm.fallback import LLMUnavailableError
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM
from tests.test_fanout import Synth

ROWS = [CompoundRow(**r) for r in load_jsonl(EVALS_DIR / "compound_questions.jsonl")]
MISROUTED = [r for r in ROWS if any(p.likely_wrong_domain for p in r.parts)]


class NeedsEntry:
    """A worker model that answers only when an entry it needs is in the context, like a model that cannot answer from
    the wrong domain's entries. `needs` maps a question's words to the entry id (or id prefix, such as "IT-") that
    answers it."""

    def __init__(self, needs, barrier=None):
        self.needs, self.calls, self.barrier, self._lock = needs, [], barrier, threading.Lock()

    def invoke(self, messages):
        human = messages[-1].content
        question = human.split("\n\nContext:")[0]
        ids = re.findall(r"^\[([A-Z]+-[A-Z0-9-]+)\]", human, re.M)
        with self._lock:
            self.calls.append((question, tuple(ids)))
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        wanted = next((entry for words, entry in self.needs.items() if words in question), None)
        found = next((i for i in ids if wanted and i.startswith(wanted)), None)
        if found:
            body = {"answer": f"The answer is in {found}.", "insufficient": False, "citations": [found]}
        else:
            body = {"answer": "", "insufficient": True, "citations": []}
        return SimpleNamespace(content=json.dumps(body))


def scripted(parts):
    """An orchestrator that returns the given [(text, domain)] as the parts of every message."""
    reply = {"intent": "kb_query", "confidence": 0.95, "domain": parts[0][1],
             "parts": [{"text": t, "domain": d, "follow_up": False} for t, d in parts]}
    return SimpleNamespace(invoke=lambda messages: SimpleNamespace(content=json.dumps(reply)))


def mock_settings(settings):
    """Hashing embeddings score lower than real ones, so retrieval uses its own mock threshold (as the app does)."""
    return settings.model_copy(update={"thresholds": settings.thresholds.model_copy(
        update={"escalation": settings.thresholds.escalation_mock})})


def stack_for(built_index_dir, settings, tmp_path, parts, answer, **kw):
    return assemble(built_index_dir, mock_settings(settings), tmp_path, llm=answer, routed=True, route_model=scripted(parts),
                    synth_model=Synth(), **kw)


PASSWORD = "How do I reset my forgotten password?"


def test_a_misrouted_question_is_answered_after_one_whole_kb_retry(built_index_dir, settings, tmp_path):
    answer = NeedsEntry({"password": "IT-PWD-01"})
    stack = stack_for(built_index_dir, settings, tmp_path, [(PASSWORD, "HR")], answer)  # an IT question sent to HR
    state = run_turn(stack.graph, new_thread_id(), PASSWORD)
    assert [d for _, d in stack.spy.calls] == ["HR", None], "the HR worker first, then one search with no filter"
    result = state.part_results["p1"]
    assert result.retried and result.citations == ["IT-PWD-01"] and result.agent == "orchestrator"
    assert state.pending_offer is None and "IT-PWD-01" in state.reply
    assert state.steps == ["start", "route", "search_kb", "answer", "search_kb", "answer", "reply"]


def test_a_part_that_is_still_unanswerable_is_marked_unanswered_after_one_retry(built_index_dir, settings, tmp_path):
    stack = stack_for(built_index_dir, settings, tmp_path, [(PASSWORD, "HR")], NeedsEntry({"password": "NO-SUCH-1"}))
    state = run_turn(stack.graph, new_thread_id(), PASSWORD)
    assert len(stack.spy.calls) == 2, "the retry runs once only"
    result = state.part_results["p1"]
    assert result.retried and not result.citations and result.llm_insufficient
    assert state.pending_offer is not None and state.pending_offer.unanswered_text == PASSWORD


def test_the_retry_comes_before_any_offer(built_index_dir, settings, tmp_path):
    stack = stack_for(built_index_dir, settings, tmp_path, [(PASSWORD, "HR")], NeedsEntry({}))
    state = run_turn(stack.graph, new_thread_id(), PASSWORD)
    assert state.steps.index("offer") > max(i for i, s in enumerate(state.steps) if s == "search_kb")


def test_a_misc_part_has_no_filter_so_it_is_not_retried(built_index_dir, settings, tmp_path):
    stack = stack_for(built_index_dir, settings, tmp_path, [("Is there free parking at the Pune office?", "MISC")],
                      NeedsEntry({}))
    state = run_turn(stack.graph, new_thread_id(), "parking?")
    assert [d for _, d in stack.spy.calls] == [None] and not state.part_results["p1"].retried


def test_only_the_failing_part_is_retried(built_index_dir, settings, tmp_path):
    parts = [(PASSWORD, "HR"), ("How many casual leave days do I get?", "HR")]
    answer = NeedsEntry({"password": "IT-PWD-01", "casual leave": "HR-LEAVE-01"})
    stack = stack_for(built_index_dir, settings, tmp_path, parts, answer)
    state = run_turn(stack.graph, new_thread_id(), "two questions")
    retried = {pid: r.retried for pid, r in state.part_results.items()}
    assert retried == {"p1": True, "p2": False}
    assert sorted((d for _, d in stack.spy.calls), key=str) == ["HR", "HR", None]
    assert state.pending_offer is None and "IT-PWD-01" in state.reply and "HR-LEAVE-01" in state.reply
    assert state.steps.count("search_kb") == 3


def test_failing_parts_retry_in_parallel(built_index_dir, settings, tmp_path):
    """A two-party barrier on every model call: the first pair (both workers) and the second pair (both retries) each
    pass only if the two calls are in flight together."""
    parts = [(PASSWORD, "HR"), ("How do I connect to the VPN from home?", "HR")]
    answer = NeedsEntry({"password": "IT-PWD-01", "VPN": "IT-VPN-02"}, barrier=threading.Barrier(2))
    stack = stack_for(built_index_dir, settings, tmp_path, parts, answer)
    state = run_turn(stack.graph, new_thread_id(), "two questions")
    assert all(r.retried and r.citations for r in state.part_results.values()) and len(answer.calls) == 4


def test_a_retry_outage_ends_the_turn_with_the_polite_message_and_no_offer(built_index_dir, settings, tmp_path):
    class Down(NeedsEntry):
        def invoke(self, messages):
            if len(self.calls) >= 1:
                raise LLMUnavailableError()
            return super().invoke(messages)

    stack = stack_for(built_index_dir, settings, tmp_path, [(PASSWORD, "HR")], Down({"password": "IT-PWD-01"}))
    state = run_turn(stack.graph, new_thread_id(), PASSWORD)
    assert "try again" in state.reply.lower() and state.pending_offer is None and state.awaiting == "none"


def test_a_retry_that_answers_is_cached_under_the_whole_kb(built_index_dir, settings, tmp_path):
    from smartdesk_ai.ops.cache import AnswerCache, MemoryBackend

    cache = AnswerCache(MemoryBackend(10), kb_version=lambda: "v")
    answer = NeedsEntry({"password": "IT-PWD-01"})
    stack = stack_for(built_index_dir, settings, tmp_path, [(PASSWORD, "HR")], answer, cache=cache)
    run_turn(stack.graph, new_thread_id(), PASSWORD)
    before = len(answer.calls)
    state = run_turn(stack.graph, new_thread_id(), PASSWORD)
    result = state.part_results["p1"]
    assert result.retried and "IT-PWD-01" in state.reply and result.cache == "exact"
    assert len(answer.calls) == before + 1, "the failed HR attempt is not cached and runs again; the retry is served from the cache"


# --- the compound set: the misrouted parts of cmp-005 to cmp-008 ----------------------------------------------------

@pytest.mark.parametrize("row", MISROUTED, ids=[r.id for r in MISROUTED])
def test_every_misrouted_part_of_the_compound_set_is_recovered(built_index_dir, settings, tmp_path, row):
    """The orchestrator tags each part with the domain the wording invites (the wrong one where the row says so); the
    model answers only from an entry of the right domain, so a part sent to the wrong worker needs the whole-KB retry."""
    parts = [(p.sub_question, p.likely_wrong_domain or p.expected_domain) for p in row.parts if p.expected_domain != "MISC"]
    needs = {p.sub_question: f"{p.expected_domain}-" for p in row.parts if p.answerable}  # an entry of the right domain
    answer = NeedsEntry(needs)
    stack = stack_for(built_index_dir, settings, tmp_path, parts, answer)
    state = run_turn(stack.graph, new_thread_id(), row.question)
    wrong = [p for p in row.parts if p.likely_wrong_domain]
    assert wrong
    for index, part in enumerate(row.parts, 1):
        result = state.part_results[f"p{index}"]
        assert result.citations and result.citations[0].startswith(f"{part.expected_domain}-"), (part.sub_question, result.citations)
        assert result.retried == bool(part.likely_wrong_domain)
