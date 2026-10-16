"""The retrieval pipeline (FR-A6): BM25 and dense, RRF, rerank, collapse to entries.

Each stage is a separate function; this module only wires them together. If the reranker is
missing or fails, the RRF order is kept and the scores become dense cosine similarity (FR-F7).
"""
from typing import Literal

from pydantic import BaseModel

from smartdesk_ai.config.settings import Settings
from smartdesk_ai.ops.health import Component, HealthRegistry, Status, get_registry
from smartdesk_ai.rag.answerable import Signals, assess
from smartdesk_ai.rag.fusion import rrf
from smartdesk_ai.rag.index import KBIndex
from smartdesk_ai.rag.models import RetrievedEntry, ScoredChunk
from smartdesk_ai.rag.rerank import Reranker, collapse_to_entries
from smartdesk_ai.rag.retrievers import bm25_search, dense_search

ScoreKind = Literal["reranker", "dense_cosine"]


class RetrievalResult(BaseModel):
    entries: list[RetrievedEntry]  # best first, at most `retrieval.top_n`, one per entry id
    top_score: float  # reranker: the best entry's score; dense_cosine: the top dense similarity; 0.0 if none
    score_kind: ScoreKind  # which scale the scores are on (FR-A15)
    answerable: bool  # signals 1 and 2 did not fire (FR-A15)
    signals: Signals
    threshold: float  # the threshold applied: escalation for reranker scores, fallback for dense cosine


def _result(entries: list[RetrievedEntry], top_score: float, kind: ScoreKind, cfg: Settings) -> RetrievalResult:
    a = assess(len(entries), top_score, kind, cfg.thresholds)
    return RetrievalResult(
        entries=entries, top_score=top_score, score_kind=kind,
        answerable=a.answerable, signals=a.signals, threshold=a.threshold,
    )


def _entries(index: KBIndex, best: list[ScoredChunk], score_of=lambda s: s.score) -> list[RetrievedEntry]:
    return [
        RetrievedEntry(**index.entries[s.chunk.entry_id].model_dump(), score=score_of(s), chunk_id=s.chunk.chunk_id)
        for s in best
    ]


def retrieve(
    index: KBIndex,
    query: str,
    *,
    domain: str | None = None,
    cfg: Settings,
    reranker: Reranker | None,
    health: HealthRegistry | None = None,
) -> RetrievalResult:
    health = health or get_registry()
    r = cfg.retrieval
    bm25 = bm25_search(index, query, r.k_each, domain)
    dense = dense_search(index, query, r.k_each, domain)
    fused = rrf([bm25, dense], k=r.rrf_k)

    if reranker is not None:
        try:
            ranked = reranker.rerank(query, fused)
        except Exception as exc:  # a crashed model degrades the service, it does not stop it
            health.mark_degraded(Component.RERANKER, f"{type(exc).__name__}: {exc}")
        else:
            if fused:  # an empty call proves nothing about the model
                health.mark_ok(Component.RERANKER)
            entries = _entries(index, collapse_to_entries(ranked, r.top_n))
            return _result(entries, entries[0].score if entries else 0.0, "reranker", cfg)
    elif health.get(Component.RERANKER).status is not Status.DEGRADED:
        health.mark_degraded(Component.RERANKER, "reranker not loaded")

    # Fallback: RRF order, scored by each piece's dense cosine (0.0 if it was not in the dense top-k).
    cosine = {d.chunk.chunk_id: d.score for d in dense}
    entries = _entries(index, collapse_to_entries(fused, r.top_n), lambda s: cosine.get(s.chunk.chunk_id, 0.0))
    return _result(entries, dense[0].score if dense else 0.0, "dense_cosine", cfg)
