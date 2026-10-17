"""T4.11 follow-up: an answer-LLM outage is not a ticket offer (FR-F2, FR-F1). Only a malformed or
schema-invalid reply counts as signal 3. Mock models, no sleeping."""
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from smartdesk_ai.agent.answer import AnswerReply, AnswerUnavailable, make_answerer
from smartdesk_ai.agent.graph import KBDeps, build_kb_graph, run_turn
from smartdesk_ai.agent.state import STATE_MODELS, new_thread_id
from smartdesk_ai.llm.fallback import GRACEFUL_MESSAGE, FallbackChat, LLMUnavailableError, invoke_with_fallback
from smartdesk_ai.memory.checkpointer import make_checkpointer
from smartdesk_ai.tools.search_kb import KBChunk
from tests.agent_helpers import build_test_graph
from tests.llm_fakes import FakeAnswerLLM

CHUNKS = [KBChunk(entry_id="IT-PWD-01", domain="IT", title="Password reset", header="h",
                  text="Use the portal.", score=0.8)]
QUESTION = "How do I reset my forgotten password?"


class Down:
    def invoke(self, messages):
        raise ConnectionError("down")


def no_sleep(_):
    pass


def answerer(settings, per_tier):
    """per_tier maps a model ID to a fake model."""
    return make_answerer(settings, builder=lambda model_id, cfg: per_tier[model_id], sleep=no_sleep)


def both_down(settings):
    return answerer(settings, {settings.models.main.id: Down(), settings.models.light.id: Down()})


def test_both_tiers_down_raises_answer_unavailable_with_the_graceful_message(settings):
    with pytest.raises(AnswerUnavailable) as caught:
        both_down(settings)(QUESTION, CHUNKS)
    assert caught.value.message == GRACEFUL_MESSAGE


def test_one_tier_down_falls_back_and_still_answers(settings):
    light = FakeAnswerLLM()
    answer = answerer(settings, {settings.models.main.id: Down(), settings.models.light.id: light})
    assert answer(QUESTION, CHUNKS).insufficient is False and len(light.calls) == 1


def test_an_outage_replies_politely_and_offers_no_ticket(built_index_dir, settings, tmp_path):
    graph, spy = build_test_graph(built_index_dir, settings, tmp_path)
    graph_deps = KBDeps(executor=_executor_of(graph, spy, settings), answer=both_down(settings))
    state = _run_with(graph_deps, tmp_path, QUESTION, "IT")
    assert state.reply == GRACEFUL_MESSAGE
    assert state.pending_offer is None and state.awaiting == "none"
    assert "offer" not in state.steps and state.steps[-1] == "llm_unavailable"
    assert len(spy.calls) == 1, "no whole-KB retry on an outage"


def test_a_malformed_reply_is_signal_3_and_does_offer_a_ticket(built_index_dir, settings, tmp_path):
    llm = FakeAnswerLLM("raw", raw="this is not json")
    graph, spy = build_test_graph(built_index_dir, settings, tmp_path, llm=llm)
    state = run_turn(graph, new_thread_id(), QUESTION)
    assert state.part_result.llm_insufficient is True
    assert state.pending_offer is not None and "offer" in state.steps


def test_a_schema_invalid_reply_is_signal_3(built_index_dir, settings, tmp_path):
    llm = FakeAnswerLLM("raw", raw='{"answer": "x", "insufficient": "maybe"}')
    graph, _ = build_test_graph(built_index_dir, settings, tmp_path, llm=llm)
    assert run_turn(graph, new_thread_id(), QUESTION).pending_offer is not None


def test_an_outage_during_the_retry_also_replies_politely(built_index_dir, settings, tmp_path):
    seq = _FailsSecond(settings)
    graph, spy = build_test_graph(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"))
    deps = KBDeps(executor=_executor_of(graph, spy, settings), answer=seq)
    state = _run_with(deps, tmp_path, QUESTION, "IT")
    assert state.reply == GRACEFUL_MESSAGE and state.pending_offer is None


# --- the fallback adapter -------------------------------------------------------------------


def test_fallback_chat_raises_when_nothing_answers(settings):
    chat = FallbackChat("WORKER_ANSWER", settings, builder=lambda m, c: Down(), sleep=no_sleep)
    with pytest.raises(LLMUnavailableError):
        chat.invoke(["hi"])


def test_fallback_chat_returns_the_reply_text(settings):
    chat = FallbackChat("WORKER_ANSWER", settings, builder=lambda m, c: FakeAnswerLLM("raw", raw="hello"),
                        sleep=no_sleep)
    assert chat.invoke(["hi"]).content == "hello"


class _Reply(BaseModel):
    answer: str


class Structured:
    """A chat model with structured output; `parsed` None models a schema violation."""

    def __init__(self, parsed, raw_text="not json"):
        self.parsed, self.raw_text, self.schema, self.include_raw = parsed, raw_text, None, None

    def with_structured_output(self, schema, include_raw=False):
        self.schema, self.include_raw = schema, include_raw
        return SimpleNamespace(
            invoke=lambda prompt: {"raw": SimpleNamespace(content=self.raw_text), "parsed": self.parsed,
                                   "parsing_error": None if self.parsed else ValueError("bad")}
        )


def test_structured_mode_returns_the_parsed_reply_as_json(settings):
    model = Structured(_Reply(answer="hi"))
    result = invoke_with_fallback("WORKER_ANSWER", "p", settings, builder=lambda m, c: model, schema=_Reply,
                                  sleep=no_sleep)
    assert result.ok and result.text == '{"answer":"hi"}' and model.schema is _Reply and model.include_raw


def test_a_schema_violation_is_not_an_outage(settings):
    model = Structured(None, raw_text="garbled")
    result = invoke_with_fallback("WORKER_ANSWER", "p", settings, builder=lambda m, c: model, schema=_Reply,
                                  sleep=no_sleep)
    assert result.ok and result.text == "garbled" and result.fell_back is False


# --- helpers ------------------------------------------------------------------------------


def _executor_of(graph, spy, settings):
    from smartdesk_ai.tools.executor import ToolExecutor
    from smartdesk_ai.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register("search_kb", spy)
    return ToolExecutor(registry, settings)


def _run_with(deps, tmp_path, text, domain):
    graph = build_kb_graph(deps, make_checkpointer(tmp_path / "outage.db", STATE_MODELS))
    return run_turn(graph, new_thread_id(), text, search_domain=domain)


class _FailsSecond:
    """First call: insufficient (so the whole-KB retry runs). Second call: outage."""

    def __init__(self, settings):
        self.settings, self.n = settings, 0

    def __call__(self, question, chunks):
        self.n += 1
        if self.n == 1:
            return answerer(self.settings, {self.settings.models.main.id: FakeAnswerLLM("insufficient"),
                                            self.settings.models.light.id: FakeAnswerLLM("insufficient")})(question, chunks)
        return both_down(self.settings)(question, chunks)
