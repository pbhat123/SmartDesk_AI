"""UI review fix 4: the reranker is reported truthfully. Mock mode skips it on purpose and says "skipped in mock mode"
(not "degraded"); live mode loads it once per process and records the load time and status; a failed load is degraded
with its reason; Monitoring shows the status and load time of every optional component (FR-F7, FR-A6)."""
import dataclasses
from pathlib import Path

import pytest

from smartdesk_ai.agent.factory import build_stack
from smartdesk_ai.ops.health import Component, HealthRegistry, Status
from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.ingest import DEFAULT_KB_DIR, ingest
from smartdesk_ai.rag.rerank import CrossEncoderReranker
from smartdesk_ai.ui.monitoring import load_monitoring
from tests.ui_helpers import element_text, mock_context, monitoring_app

SKIPPED_IN_MOCK = "skipped in mock mode"


# --- the registry ---------------------------------------------------------------------------------

def test_a_skipped_component_is_not_a_degraded_mode():
    registry = HealthRegistry()
    registry.mark_skipped(Component.RERANKER, SKIPPED_IN_MOCK)
    state = registry.get(Component.RERANKER)
    assert state.status is Status.SKIPPED and state.reason == SKIPPED_IN_MOCK and state.mode is None
    assert registry.degraded_modes() == []


def test_try_load_records_how_long_the_load_took_for_success_and_failure():
    registry = HealthRegistry()
    registry.try_load(Component.RERANKER, lambda: "model")
    assert registry.get(Component.RERANKER).load_seconds is not None
    assert registry.get(Component.RERANKER).load_seconds >= 0
    registry.try_load(Component.PRESIDIO, lambda: 1 / 0)
    assert registry.get(Component.PRESIDIO).load_seconds >= 0 and registry.get(Component.PRESIDIO).status is Status.DEGRADED


# --- mock mode -------------------------------------------------------------------------------------

def test_mock_mode_marks_the_reranker_skipped_and_a_search_does_not_turn_it_degraded(settings, tmp_path):
    registry = HealthRegistry()
    stack = build_stack(settings, environ={}, dotenv_path=None, data_dir=tmp_path, registry=registry)
    assert registry.get(Component.RERANKER).status is Status.SKIPPED
    stack.executor.execute("search_kb", {"query": "reset my password"}, agent="test")
    state = registry.get(Component.RERANKER)
    assert state.status is Status.SKIPPED and state.reason == SKIPPED_IN_MOCK
    assert Component.RERANKER not in [h.component for h in registry.degraded_modes()]


def test_the_monitoring_page_says_skipped_in_mock_mode_not_degraded(tmp_path):
    ctx = mock_context(tmp_path)
    ctx.service.handle("s", "How do I reset my forgotten password?")
    data = load_monitoring(ctx.settings, ctx.data_dir, ctx.kb_dir, ctx.registry, results_dir=ctx.results_dir)
    row = next(c for c in data.components if c.component == "reranker")
    assert (row.status, row.detail) == ("skipped", SKIPPED_IN_MOCK)
    assert "reranker" not in [d.component for d in data.degraded]
    shown = element_text(monitoring_app(ctx).main)
    assert SKIPPED_IN_MOCK in shown and "reranker not loaded" not in shown


def test_the_chat_sidebar_does_not_call_a_skipped_reranker_degraded(tmp_path):
    from tests.ui_helpers import app_test

    at = app_test(mock_context(tmp_path))
    assert not any("reranker" in c.value.lower() for c in at.sidebar.caption)


# --- live mode (no network: the model and the embeddings are stubs) -----------------------------------

@pytest.fixture
def live_ready(settings, tmp_path, monkeypatch):
    """A live-mode settings object, a data folder holding a built index, and counters on the model loads."""
    data = tmp_path / "data"
    ingest(DEFAULT_KB_DIR, data / "index", HashingEmbeddings(), settings.chunking)
    loads: list[str] = []
    monkeypatch.setattr(CrossEncoderReranker, "from_model",
                        classmethod(lambda cls, model_id: loads.append(model_id) or cls(lambda pairs: [0.0] * len(pairs))))
    monkeypatch.setattr("smartdesk_ai.agent.factory.get_embeddings", lambda s, e: HashingEmbeddings())
    live = settings.model_copy(update={"modes": settings.modes.model_copy(update={"llm": "live"})})
    return live, data, loads


def _live_stack(live, data, registry):
    return build_stack(live, environ={"OPENAI_API_KEY": "sk-test-not-a-real-key"}, dotenv_path=None, data_dir=data,
                       registry=registry)


def test_live_mode_loads_the_reranker_once_and_records_status_and_time(live_ready):
    live, data, loads = live_ready
    registry = HealthRegistry()
    _live_stack(live, data, registry)
    state = registry.get(Component.RERANKER)
    assert loads == [live.models.reranker.id] and state.status is Status.OK and state.load_seconds >= 0


def test_a_live_reranker_that_fails_to_load_is_degraded_with_its_reason(live_ready, monkeypatch):
    live, data, _ = live_ready

    def boom(cls, model_id):
        raise OSError("model files not found")

    monkeypatch.setattr(CrossEncoderReranker, "from_model", classmethod(boom))
    registry = HealthRegistry()
    _live_stack(live, data, registry)  # startup must not crash (FR-F7)
    state = registry.get(Component.RERANKER)
    assert state.status is Status.DEGRADED and "model files not found" in state.reason and state.load_seconds is not None


def test_a_live_strategy_without_rerank_does_not_load_the_model(live_ready, monkeypatch):
    live, data, loads = live_ready
    dense = live.model_copy(update={"retrieval": live.retrieval.model_copy(update={"strategy": "dense"})})
    registry = HealthRegistry()
    _live_stack(dense, data, registry)
    state = registry.get(Component.RERANKER)
    assert loads == [] and state.status is Status.SKIPPED and "dense" in state.reason


def test_the_monitoring_page_shows_the_live_reranker_status_and_load_time(live_ready):
    live, data, _ = live_ready
    registry = HealthRegistry()
    _live_stack(live, data, registry)
    ctx = dataclasses.replace(mock_context(data.parent / "ui"), settings=live, registry=registry, data_dir=data)
    row = next(c for c in load_monitoring(live, data, ctx.kb_dir, registry, results_dir=ctx.results_dir).components
               if c.component == "reranker")
    assert row.status == "ok" and row.load_seconds is not None
    shown = element_text(monitoring_app(ctx).main)
    assert "reranker" in shown and "load time" in shown.lower()


# --- once per process ----------------------------------------------------------------------------------

def test_the_app_context_is_built_once_per_process(monkeypatch):
    from smartdesk_ai.ui import session

    built = []
    monkeypatch.setattr(session, "build_ui_context", lambda: built.append(1) or object())
    session._default_context.clear()
    first, second = session._default_context(), session._default_context()
    session._default_context.clear()
    assert first is second and built == [1]
