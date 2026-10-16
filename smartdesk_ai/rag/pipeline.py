"""The retrieval pipeline (FR-A6): BM25 and dense, RRF, rerank, collapse to entries.

Each stage is a separate function; this module only wires them together.
"""
from typing import Literal

from pydantic import BaseModel

from smartdesk_ai.config.settings import Settings
from smartdesk_ai.rag.fusion import rrf
from smartdesk_ai.rag.index import KBIndex
from smartdesk_ai.rag.models import RetrievedEntry
from smartdesk_ai.rag.rerank import Reranker, collapse_to_entries
from smartdesk_ai.rag.retrievers import bm25_search, dense_search

ScoreKind = Literal["reranker", "dense_cosine"]


class RetrievalResult(BaseModel):
    entries: list[RetrievedEntry]  # best first, at most `retrieval.top_n`, one per entry id
    top_score: float  # the best entry's score, 0.0 when there are none
    score_kind: ScoreKind  # which scale the scores are on (FR-A15)


def retrieve(
    index: KBIndex, query: str, *, domain: str | None = None, cfg: Settings, reranker: Reranker
) -> RetrievalResult:
    r = cfg.retrieval
    fused = rrf(
        [bm25_search(index, query, r.k_each, domain), dense_search(index, query, r.k_each, domain)],
        k=r.rrf_k,
    )
    best = collapse_to_entries(reranker.rerank(query, fused), r.top_n)
    entries = [
        RetrievedEntry(**index.entries[s.chunk.entry_id].model_dump(), score=s.score, chunk_id=s.chunk.chunk_id)
        for s in best
    ]
    return RetrievalResult(
        entries=entries, top_score=entries[0].score if entries else 0.0, score_kind="reranker"
    )
