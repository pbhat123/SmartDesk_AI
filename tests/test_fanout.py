"""T7.4: a compound question fans out to its workers in parallel (LangGraph Send), the results are joined, and the
synthesizer merges them once (FR-I4, FR-G2, FR-H2, FR-H7). Mock and scripted models, no network."""
import json
import re
import threading
import time
from types import SimpleNamespace

import pytest

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id, thread_config
from smartdesk_ai.guardrails.pii import PiiRedactor
from smartdesk_ai.llm.fallback import LLMUnavailableError
from smartdesk_ai.ops.cost import CostTracker
from smartdesk_ai.ops.runtime import RequestContext, Services, request_scope
from smartdesk_ai.ops.trace_log import TraceLog, make_masker
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM

COMPOUND = "How do I reset my password, and how many casual leave days do I get?"
IT_BODY = "Reset it at https://id.smartdesk.example/reset and the link lasts 30 minutes."
HR_BODY = "You get 12 casual leave days a year."


class ByQuestion:
    """A worker model that answers each question with a fixed body and cites the first context entry."""

    def __init__(self, delay_it=0.0, barrier=None):
        self.calls, self.delay_it, self.barrier, self.threads, self._lock = [], delay_it, barrier, set(), threading.Lock()

    def invoke(self, messages):
        human = messages[-1].content
        with self._lock:
            self.calls.append(human)
            self.threads.add(threading.current_thread().name)
            first_calls = len(self.calls) <= 2
        if self.barrier is not None and first_calls:
            self.barrier.wait(timeout=5)  # only passes if the other worker is in flight at the same time
        it = "password" in human.split("Context:")[0].lower()
        if it and self.delay_it:
            time.sleep(self.delay_it)
        ids = re.findall(r"^\[([A-Z]+-[A-Z0-9-]+)\]", human, re.M)
        return SimpleNamespace(content=json.dumps({"answer": IT_BODY if it else HR_BODY, "insufficient": False,
                                                    "citations": ids[:1]}))


class Synth:
    """A synthesizer model: records its input, replies with `reply` or by joining the answers it was given."""

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        human = messages[-1].content
        answers = re.findall(r"^Answer \d+: (.*)$", human, re.M)
        text = self.reply if self.reply is not None else " ".join(answers)
        return SimpleNamespace(content=text if isinstance(text, str) and text.startswith("{") else json.dumps({"answer": text}))


def stack_for(built_index_dir, settings, tmp_path, answer=None, synth=None, **kw):
    return assemble(built_index_dir, settings, tmp_path, llm=answer or ByQuestion(), routed=True,
                    synth_model=synth if synth is not None else Synth(), **kw)


def ask(stack, text=COMPOUND, tid=None):
    return run_turn(stack.graph, tid or new_thread_id(), text)


def test_a_compound_question_runs_both_workers_in_parallel_and_the_synthesizer_once(built_index_dir, settings, tmp_path):
    answer, synth = ByQuestion(barrier=threading.Barrier(2)), Synth()
    stack = stack_for(built_index_dir, settings, tmp_path, answer, synth)
    state = ask(stack)
    assert len(answer.calls) == 2 and len(answer.threads) == 2, "two workers answered, each on its own thread"
    assert len(synth.calls) == 1
    assert set(state.part_results) == {"p1", "p2"} and state.pending_offer is None
    cited = {c for r in state.part_results.values() for c in r.citations}
    assert {c.split("-")[0] for c in cited} == {"IT", "HR"}, cited
    assert all(c in state.reply for c in cited) and "Sources:" in state.reply
    assert state.steps[:3] == ["start", "route", "fan_out:2"] and state.steps[-2:] == ["synthesize", "reply"]


def test_the_search_domains_are_the_workers_domains(built_index_dir, settings, tmp_path):
    stack = stack_for(built_index_dir, settings, tmp_path)
    ask(stack)
    assert sorted(d for _, d in stack.spy.calls[:2]) == ["HR", "IT"]


def test_the_synthesizer_input_is_the_message_and_each_question_with_its_answer_only(built_index_dir, settings, tmp_path):
    synth = Synth()
    stack = stack_for(built_index_dir, settings, tmp_path, synth=synth)
    ask(stack)
    system, human = synth.calls[0]
    human = human.content
    assert human.startswith(f"Original message: {COMPOUND}\n\nQuestion 1: How do I reset my password?\nAnswer 1: {IT_BODY}")
    assert f"Question 2: How many casual leave days do I get?\nAnswer 2: {HR_BODY}" in human
    assert "Sources" not in human and not re.search(r"\b[A-Z]+-[A-Z0-9]+-\d+\b", human), "citations are added by code"


def test_the_merged_reply_gets_its_sources_line_from_the_worker_citations(built_index_dir, settings, tmp_path):
    state = ask(stack_for(built_index_dir, settings, tmp_path))
    body, sources = state.reply.split("\n\nSources: ")
    assert body == f"{IT_BODY} {HR_BODY}"
    ids = [part.split(" (")[0] for part in sources.split("; ")]
    assert ids == [c for p in ("p1", "p2") for c in state.part_results[p].citations]


# --- the synthesizer's checks: anything it adds goes back to the worker answers joined in order ---------

JOINED = f"{IT_BODY}\n\n{HR_BODY}"


@pytest.mark.parametrize("reply", [
    f"{IT_BODY} {HR_BODY} Contractors get 15 days.",                          # a number nobody gave
    f"{IT_BODY} {HR_BODY} See HR-LEAVE-99 for more.",                          # an entry id nobody gave
    f"{IT_BODY} {HR_BODY} Visit https://example.org/more.",                    # a URL nobody gave
    f"{IT_BODY} {HR_BODY} Ask Priya about it.",                                # a name nobody gave
    f"{IT_BODY} {HR_BODY} Write to help@smartdesk.example.",                   # an email nobody gave
    "   ",                                                                     # nothing
    "{not json at all",                                                        # not the reply shape
])
def test_an_unsupported_fact_or_a_bad_reply_falls_back_to_the_joined_answers(built_index_dir, settings, tmp_path, reply):
    state = ask(stack_for(built_index_dir, settings, tmp_path, synth=Synth(reply=reply)))
    assert state.reply.split("\n\nSources: ")[0] == JOINED
    assert "synthesize_fallback" in state.steps and "Sources:" in state.reply


def test_a_synthesizer_outage_falls_back_to_the_joined_answers(built_index_dir, settings, tmp_path):
    synth = Synth(error=LLMUnavailableError())
    state = ask(stack_for(built_index_dir, settings, tmp_path, synth=synth))
    assert state.reply.split("\n\nSources: ")[0] == JOINED and "synthesize_fallback" in state.steps
    assert state.pending_offer is None, "the answers were found: no ticket offer"


def test_without_a_synthesizer_the_answers_are_joined(built_index_dir, settings, tmp_path):
    stack = assemble(built_index_dir, settings, tmp_path, llm=ByQuestion(), routed=True)
    state = ask(stack)
    assert state.reply.split("\n\nSources: ")[0] == JOINED


def test_a_faithful_merge_is_used_as_the_model_wrote_it(built_index_dir, settings, tmp_path):
    merged = "To reset your password use https://id.smartdesk.example/reset (the link lasts 30 minutes). You also get 12 casual leave days a year."
    state = ask(stack_for(built_index_dir, settings, tmp_path, synth=Synth(reply=merged)))
    assert state.reply.split("\n\nSources: ")[0] == merged and "synthesize_fallback" not in state.steps


def test_the_join_follows_the_order_of_the_questions_not_of_who_finished_first(built_index_dir, settings, tmp_path):
    stack = stack_for(built_index_dir, settings, tmp_path, answer=ByQuestion(delay_it=0.3), synth=Synth(reply="x 99"))
    state = ask(stack)
    assert state.reply.split("\n\nSources: ")[0] == JOINED  # IT was slower and is still first


# --- every part goes through the same answering step -----------------------------------------------------

def test_the_judge_runs_once_per_part(built_index_dir, settings, tmp_path):
    calls = []
    judge = lambda question, answer, chunks: calls.append(question) or True  # noqa: E731
    stack = stack_for(built_index_dir, settings, tmp_path, judge=judge)
    ask(stack)
    assert sorted(calls) == ["How do I reset my password?", "How many casual leave days do I get?"]


def test_each_part_uses_the_cache_on_its_own(built_index_dir, settings, tmp_path):
    from smartdesk_ai.ops.cache import AnswerCache, MemoryBackend

    cache = AnswerCache(MemoryBackend(100), kb_version=lambda: "v1")
    answer, synth = ByQuestion(), Synth()
    stack = stack_for(built_index_dir, settings, tmp_path, answer, synth, cache=cache)
    ask(stack)
    first = len(answer.calls)
    state = ask(stack)
    assert len(answer.calls) == first and state.steps.count("cache_hit:exact") == 2
    assert len(synth.calls) == 2, "the merged reply is not cached; the synthesizer runs again"


def test_parts_beyond_the_limit_are_named_not_answered(built_index_dir, settings, tmp_path):
    parts = [{"text": f"How many casual leave days do I get in case {i}?", "domain": "HR", "follow_up": False}
             for i in range(1, 7)]
    reply = {"intent": "kb_query", "confidence": 0.95, "domain": "HR", "parts": parts}
    model = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content=json.dumps(reply)))
    answer = ByQuestion()
    stack = assemble(built_index_dir, settings, tmp_path, llm=answer, routed=True, route_model=model, synth_model=Synth())
    state = ask(stack, "six questions")
    assert len(state.part_results) == settings.orchestrator.max_parts == 4 and state.parts_dropped == 2
    assert state.reply.endswith("I've answered the first 4 questions. Please ask the other 2 in a separate message.")


# --- what must hold when the parts run on other threads ----------------------------------------------------

class Budget:
    def __init__(self):
        self.threads = set()

    def level(self, task):
        self.threads.add(threading.current_thread().name)
        return 0


def test_cost_trace_and_budget_are_recorded_under_the_request_scope_from_inside_the_workers(
        built_index_dir, settings, tmp_path):
    """LangGraph runs the parts on its own threads; the request scope must follow them (verified, not assumed)."""
    cost = CostTracker(settings, priced=False)
    trace = TraceLog(tmp_path / "t.jsonl", make_masker(PiiRedactor(None, 0.5)), prompt_versions=settings.prompts)
    budget = Budget()
    services = Services(cost=cost, trace=trace, budget=budget)
    stack = stack_for(built_index_dir, settings, tmp_path, answer=ByQuestion(barrier=threading.Barrier(2)))
    ctx = RequestContext(session_id="s", request_id="r")
    with request_scope(ctx, services):
        run_turn(stack.graph, new_thread_id(), COMPOUND)
    agents = {e.agent for e in cost.entries if e.request_id == "r"}
    assert {"orchestrator", "it_worker", "hr_worker", "synthesizer"} <= agents, agents
    traced = {r["agent"] for r in trace.events if r["kind"] == "llm_call"}
    assert {"it_worker", "hr_worker", "synthesizer"} <= traced
    assert budget.threads - {threading.current_thread().name}, "the budget level was read inside a worker thread"


def test_a_worker_that_raises_gives_a_polite_reply_and_leaves_no_pending_step(built_index_dir, settings, tmp_path):
    class Boom:
        def lookup(self, question, domain):
            raise RuntimeError("boom: jane.roe@example.com")

        def store(self, *a):  # pragma: no cover
            pass

    stack = stack_for(built_index_dir, settings, tmp_path, cache=Boom())
    tid = new_thread_id()
    state = ask(stack, tid=tid)
    assert "try again later" in state.reply.lower() and state.pending_offer is None
    assert stack.graph.get_state(thread_config(tid)).next == (), "nothing is left to resume (it would read the next message as a gate reply)"
    assert "jane.roe" not in json.dumps(state.steps)
    again = ask(stack, "How do I reset my forgotten password?", tid)
    assert "try again later" in again.reply.lower()  # the cache still raises; the turn still ends cleanly


def test_a_turn_after_a_compound_turn_has_no_leftover_parts(built_index_dir, settings, tmp_path):
    stack = stack_for(built_index_dir, settings, tmp_path)
    tid = new_thread_id()
    ask(stack, tid=tid)
    state = ask(stack, "How do I reset my forgotten password?", tid)
    assert len(state.parts) == 1 and set(state.part_results) == {"p1"}
    assert "fan_out:2" not in state.steps and state.part_result is not None


def test_an_outage_in_one_part_is_an_outage_for_the_turn_with_no_ticket_offer(built_index_dir, settings, tmp_path):
    class HrDown(ByQuestion):
        def invoke(self, messages):
            if "casual leave" in messages[-1].content.split("Context:")[0]:
                raise LLMUnavailableError()
            return super().invoke(messages)

    state = ask(stack_for(built_index_dir, settings, tmp_path, answer=HrDown()))
    assert "try again" in state.reply.lower() and state.pending_offer is None and state.awaiting == "none"


# --- shared components under two threads ---------------------------------------------------------------------

def test_the_reranker_runs_one_forward_pass_at_a_time():
    from concurrent.futures import ThreadPoolExecutor

    from smartdesk_ai.rag.models import Chunk, ScoredChunk
    from smartdesk_ai.rag.rerank import CrossEncoderReranker

    inside, overlaps = [0], []

    def predict(pairs):
        inside[0] += 1
        overlaps.append(inside[0])
        time.sleep(0.05)
        inside[0] -= 1
        return [0.0 for _ in pairs]

    chunk = Chunk(chunk_id="c1", entry_id="E-1", domain="IT", title="t", header="h", text="x", embed_text="x",
                  source_file="f", piece=1, pieces=1)
    candidates = [ScoredChunk(chunk=chunk, score=0.5, stage="rrf")]
    reranker = CrossEncoderReranker(predict)
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda _: reranker.rerank("q", candidates), range(2)))
    assert max(overlaps) == 1


def test_two_concurrent_searches_give_what_they_give_one_after_the_other(built_index_dir, settings, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    stack = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM())
    asks = [("How do I reset my password?", "IT"), ("How many casual leave days do I get?", "HR")]

    def search(ask):
        found = stack.executor.execute("search_kb", {"query": ask[0], "domain": ask[1]}, agent="it_worker").data
        return [(c.entry_id, round(c.score, 6)) for c in found.chunks]

    sequential = [search(a) for a in asks]
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(search, asks)) == sequential
