"""The retrieval pipeline (FR-A6): BM25 and dense, RRF, rerank, collapse to entries.

Each stage is a separate function; this module only wires them together. The reranker only orders
the candidates. Whether the question is answerable is decided from the top dense cosine similarity
and one threshold (FR-A15), so a missing or failing reranker (FR-F7) changes the order of the
results but not the escalation rule.
"""
from typing import Literal

from pydantic import BaseModel

from smartdesk_ai.config.settings import Settings
from smartdesk_ai.ops.health import Component, HealthRegistry, Status, get_registry
from smartdesk_ai.rag.answerable import Signals, assess
from smartdesk_ai.rag.fusion import rrf
from smartdesk_ai.rag.index import KBIndex
from smartdesk_ai.rag.models import RetrievedEntry
from smartdesk_ai.rag.rerank import Reranker, collapse_to_entries
from smartdesk_ai.rag.retrievers import bm25_search, dense_search

OrderedBy = Literal["rerank", "rrf", "dense"]


class RetrievalResult(BaseModel):
    entries: list[RetrievedEntry]  # best first, at most `retrieval.top_n`, one per entry id
    ordered_by: OrderedBy  # which stage's score each entry's `score` is: reranker, RRF, or dense cosine
    top_cosine: float  # the best dense cosine similarity found, 0.0 if none: the escalation signal
    answerable: bool  # signals 1 and 2 did not fire (FR-A15)
    signals: Signals
    threshold: float  # the escalation threshold applied to `top_cosine`


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
    ranked, ordered_by = rrf([bm25, dense], k=r.rrf_k), "rrf"

    if reranker is not None:
        try:
            reranked = reranker.rerank(query, ranked)
        except Exception as exc:  # a crashed model degrades the order, it does not stop the service
            health.mark_degraded(Component.RERANKER, f"{type(exc).__name__}: {exc}")
        else:
            if ranked:  # an empty call proves nothing about the model
                health.mark_ok(Component.RERANKER)
            ranked, ordered_by = reranked, "rerank"
    elif health.get(Component.RERANKER).status is not Status.DEGRADED:
        health.mark_degraded(Component.RERANKER, "reranker not loaded")

    entries = [
        RetrievedEntry(**index.entries[s.chunk.entry_id].model_dump(), score=s.score, chunk_id=s.chunk.chunk_id)
        for s in collapse_to_entries(ranked, r.top_n)
    ]
    top_cosine = dense[0].score if dense else 0.0
    a = assess(len(entries), top_cosine, cfg.thresholds)
    return RetrievalResult(
        entries=entries, ordered_by=ordered_by, top_cosine=top_cosine,
        answerable=a.answerable, signals=a.signals, threshold=a.threshold,
    )
