"""T2.7: with the reranker unavailable the pipeline keeps the RRF order and reports dense cosine (FR-F7, FR-A15)."""
import pytest

from smartdesk_ai.ops.health import Component, HealthRegistry, Status, get_registry, reset_registry
from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.fusion import rrf
from smartdesk_ai.rag.index import open_index
from smartdesk_ai.rag.pipeline import retrieve
from smartdesk_ai.rag.rerank import CrossEncoderReranker, collapse_to_entries, load_reranker
from smartdesk_ai.rag.retrievers import bm25_search, dense_search
from tests.kb_helpers import load_all_entries

BODIES = {e.id: e.body for e in load_all_entries()}
QUERY = "how do I reset my forgotten password"


class Broken:
    """A reranker whose model fails when it is asked to score."""

    def __init__(self, exc=RuntimeError("model crashed")):
        self.exc = exc

    def rerank(self, query, candidates):
        raise self.exc


@pytest.fixture(scope="module")
def index(built_index_dir):
    return open_index(built_index_dir, HashingEmbeddings())


@pytest.fixture
def health():
    return HealthRegistry()


def test_a_failing_reranker_still_returns_results_and_records_the_degraded_mode(index, settings, health):
    result = retrieve(index, QUERY, cfg=settings, reranker=Broken(), health=health)
    assert result.entries
    assert result.score_kind == "dense_cosine"
    state = health.get(Component.RERANKER)
    assert state.status is Status.DEGRADED
    assert "RuntimeError" in state.reason and "model crashed" in state.reason
    assert [h.component for h in health.degraded_modes()] == [Component.RERANKER]


def test_an_unloaded_reranker_is_the_same_fallback(index, settings, health):
    result = retrieve(index, QUERY, cfg=settings, reranker=None, health=health)
    assert result.entries and result.score_kind == "dense_cosine"
    assert health.get(Component.RERANKER).status is Status.DEGRADED


def test_fallback_keeps_the_rrf_order_collapsed_to_entries(index, settings, health):
    r = settings.retrieval
    fused = rrf([bm25_search(index, QUERY, r.k_each), dense_search(index, QUERY, r.k_each)], k=r.rrf_k)
    expected = [s.chunk.entry_id for s in collapse_to_entries(fused, r.top_n)]
    result = retrieve(index, QUERY, cfg=settings, reranker=Broken(), health=health)
    assert [e.entry_id for e in result.entries] == expected
    assert len(set(expected)) == len(expected) <= r.top_n


def test_top_score_is_the_top_dense_cosine_similarity(index, settings, health):
    dense = dense_search(index, QUERY, settings.retrieval.k_each)
    result = retrieve(index, QUERY, cfg=settings, reranker=Broken(), health=health)
    assert result.top_score == pytest.approx(dense[0].score)
    assert -1.0 <= result.top_score <= 1.0
    by_chunk = {d.chunk.chunk_id: d.score for d in dense}
    for e in result.entries:
        assert e.score == pytest.approx(by_chunk.get(e.chunk_id, 0.0))


def test_fallback_results_still_carry_the_full_parent_text(index, settings, health):
    result = retrieve(index, QUERY, cfg=settings, reranker=None, health=health)
    assert all(e.text == BODIES[e.entry_id] for e in result.entries)


def test_fallback_respects_the_domain_filter(index, settings, health):
    result = retrieve(index, "laptop request approval", domain="IT", cfg=settings, reranker=None, health=health)
    assert result.entries and {e.domain for e in result.entries} == {"IT"}


def test_nothing_matching_in_fallback_gives_no_entries_and_zero_score(index, settings, health):
    result = retrieve(index, "", cfg=settings, reranker=None, health=health)
    assert result.entries == [] and result.top_score == 0.0 and result.score_kind == "dense_cosine"


def test_a_working_reranker_clears_the_degraded_mode(index, settings, health):
    retrieve(index, QUERY, cfg=settings, reranker=None, health=health)
    assert health.get(Component.RERANKER).status is Status.DEGRADED
    ok = CrossEncoderReranker(lambda pairs: [0.0] * len(pairs))
    result = retrieve(index, QUERY, cfg=settings, reranker=ok, health=health)
    assert result.score_kind == "reranker"
    assert health.get(Component.RERANKER).status is Status.OK


def test_interrupts_are_not_swallowed(index, settings, health):
    with pytest.raises(KeyboardInterrupt):
        retrieve(index, QUERY, cfg=settings, reranker=Broken(KeyboardInterrupt()), health=health)


def test_default_health_is_the_shared_registry(index, settings):
    reset_registry()
    retrieve(index, QUERY, cfg=settings, reranker=None)
    assert get_registry().get(Component.RERANKER).status is Status.DEGRADED
    reset_registry()


def test_load_reranker_returns_none_and_records_degraded_when_the_model_cannot_load(settings, health, monkeypatch):
    def boom(model_id):
        raise OSError(f"cannot download {model_id}")

    monkeypatch.setattr(CrossEncoderReranker, "from_model", staticmethod(boom))
    assert load_reranker(settings, health) is None
    state = health.get(Component.RERANKER)
    assert state.status is Status.DEGRADED and "BAAI/bge-reranker-base" in state.reason


def test_load_reranker_marks_ok_when_the_model_loads(settings, health, monkeypatch):
    fake = CrossEncoderReranker(lambda pairs: [0.0] * len(pairs))
    monkeypatch.setattr(CrossEncoderReranker, "from_model", staticmethod(lambda model_id: fake))
    assert load_reranker(settings, health) is fake
    assert health.get(Component.RERANKER).status is Status.OK
