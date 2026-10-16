"""retrieval.strategy: hybrid_rerank (default), dense, hybrid, dense_rerank wired through retrieve() (FR-A6, FR-H16)."""
import copy

import pytest
import yaml

from evals.retrieval_eval import rank_results
from smartdesk_ai.config.settings import ConfigError, load_settings
from smartdesk_ai.ops.health import Component, HealthRegistry, Status
from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.index import open_index
from smartdesk_ai.rag.pipeline import retrieve
from smartdesk_ai.rag.rerank import CrossEncoderReranker, collapse_to_entries
from smartdesk_ai.rag.retrievers import bm25_search, dense_search
from smartdesk_ai.rag.fusion import rrf
from tests.conftest import KB_DIR  # noqa: F401  (conftest builds the shared index)

STRATEGIES = ["hybrid_rerank", "dense", "hybrid", "dense_rerank"]
EVAL_NAME = {"hybrid_rerank": "hybrid+rerank", "dense": "dense", "hybrid": "hybrid", "dense_rerank": "dense+rerank"}
QUERY = "how do I reset my forgotten password"


class Recorder:
    """A reranker that records its candidates and scores them by position."""

    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates):
        self.calls.append([c.chunk.chunk_id for c in candidates])
        return CrossEncoderReranker(lambda pairs: [float(-i) for i in range(len(pairs))]).rerank(query, candidates)


class Boom:
    def rerank(self, query, candidates):
        raise RuntimeError("model crashed")


@pytest.fixture(scope="module")
def index(built_index_dir):
    return open_index(built_index_dir, HashingEmbeddings())


def cfg_for(settings, strategy):
    return settings.model_copy(update={"retrieval": settings.retrieval.model_copy(update={"strategy": strategy})})


# --- config ----------------------------------------------------------------------------

def test_the_default_strategy_is_the_full_hybrid_rerank_pipeline(settings):
    assert settings.retrieval.strategy == "hybrid_rerank"


def test_an_unknown_strategy_is_rejected_naming_the_key(tmp_path):
    raw = yaml.safe_load(open("config.yaml"))
    bad = copy.deepcopy(raw)
    bad["retrieval"]["strategy"] = "bm25_only"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ConfigError, match=r"retrieval\.strategy"):
        load_settings(path)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_listed_strategy_is_accepted(tmp_path, strategy):
    raw = yaml.safe_load(open("config.yaml"))
    raw["retrieval"]["strategy"] = strategy
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_settings(path).retrieval.strategy == strategy


# --- what each strategy does -------------------------------------------------------------

def test_dense_uses_neither_bm25_nor_the_reranker(index, settings):
    rec = Recorder()
    result = retrieve(index, QUERY, cfg=cfg_for(settings, "dense"), reranker=rec, health=HealthRegistry())
    dense = dense_search(index, QUERY, settings.retrieval.k_each)
    assert rec.calls == []
    assert result.ordered_by == "dense"
    assert [e.chunk_id for e in result.entries] == [s.chunk.chunk_id for s in collapse_to_entries(dense, settings.retrieval.top_n)]
    assert result.entries[0].score == pytest.approx(dense[0].score)


def test_hybrid_fuses_but_does_not_rerank(index, settings):
    rec = Recorder()
    result = retrieve(index, QUERY, cfg=cfg_for(settings, "hybrid"), reranker=rec, health=HealthRegistry())
    r = settings.retrieval
    fused = rrf([bm25_search(index, QUERY, r.k_each), dense_search(index, QUERY, r.k_each)], k=r.rrf_k)
    assert rec.calls == [] and result.ordered_by == "rrf"
    assert [e.chunk_id for e in result.entries] == [s.chunk.chunk_id for s in collapse_to_entries(fused, r.top_n)]


def test_dense_rerank_sends_only_the_dense_top_k_to_the_reranker(index, settings):
    rec = Recorder()
    result = retrieve(index, QUERY, cfg=cfg_for(settings, "dense_rerank"), reranker=rec, health=HealthRegistry())
    dense = dense_search(index, QUERY, settings.retrieval.k_each)
    assert rec.calls == [[s.chunk.chunk_id for s in dense]] and result.ordered_by == "rerank"


def test_hybrid_rerank_sends_the_fused_candidates_to_the_reranker(index, settings):
    rec = Recorder()
    result = retrieve(index, QUERY, cfg=cfg_for(settings, "hybrid_rerank"), reranker=rec, health=HealthRegistry())
    r = settings.retrieval
    fused = rrf([bm25_search(index, QUERY, r.k_each), dense_search(index, QUERY, r.k_each)], k=r.rrf_k)
    assert rec.calls == [[s.chunk.chunk_id for s in fused]] and result.ordered_by == "rerank"
    assert len(fused) > len(dense_search(index, QUERY, r.k_each))  # BM25 contributed extra candidates


# --- the escalation signal does not depend on the strategy --------------------------------

def test_top_cosine_answerable_and_threshold_are_identical_for_every_strategy(index, settings):
    seen = {s: retrieve(index, QUERY, cfg=cfg_for(settings, s), reranker=Recorder(), health=HealthRegistry()) for s in STRATEGIES}
    signals = {(r.top_cosine, r.answerable, r.threshold) for r in seen.values()}
    assert len(signals) == 1


def test_the_domain_filter_applies_to_every_strategy(index, settings):
    for s in STRATEGIES:
        result = retrieve(index, "laptop request approval", domain="IT", cfg=cfg_for(settings, s),
                          reranker=Recorder(), health=HealthRegistry())
        assert result.entries and {e.domain for e in result.entries} == {"IT"}, s


# --- degradation only concerns the rerank strategies --------------------------------------

@pytest.mark.parametrize("strategy,expected", [("hybrid_rerank", "rrf"), ("dense_rerank", "dense")])
def test_a_failing_reranker_keeps_the_candidate_order_of_that_strategy(index, settings, strategy, expected):
    health = HealthRegistry()
    result = retrieve(index, QUERY, cfg=cfg_for(settings, strategy), reranker=Boom(), health=health)
    assert result.entries and result.ordered_by == expected
    assert health.get(Component.RERANKER).status is Status.DEGRADED


@pytest.mark.parametrize("strategy", ["dense", "hybrid"])
def test_strategies_without_a_reranker_never_mark_it_degraded(index, settings, strategy):
    health = HealthRegistry()
    result = retrieve(index, QUERY, cfg=cfg_for(settings, strategy), reranker=None, health=health)
    assert result.entries
    assert health.get(Component.RERANKER).status is Status.UNKNOWN


# --- the evaluated strategies are the ones that run -----------------------------------------

@pytest.mark.parametrize("strategy", STRATEGIES)
def test_pipeline_order_matches_the_ablation_harness(index, settings, strategy):
    reranker = CrossEncoderReranker(lambda pairs: [float(-i) for i in range(len(pairs))])
    got = retrieve(index, QUERY, cfg=cfg_for(settings, strategy), reranker=reranker, health=HealthRegistry())
    expected = rank_results(EVAL_NAME[strategy], index, QUERY, settings, reranker)
    assert [e.entry_id for e in got.entries] == [c.entry_id for c in expected][: settings.retrieval.top_n]
