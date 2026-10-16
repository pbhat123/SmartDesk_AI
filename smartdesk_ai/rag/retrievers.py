"""Retrieval stages (FR-A6). Each stage is a plain function over an opened `KBIndex`, so
the pipeline and the ablations (FR-A7) can call any of them alone."""
from smartdesk_ai.rag.index import KBIndex, bm25_tokens
from smartdesk_ai.rag.models import ScoredChunk


def bm25_search(index: KBIndex, query: str, k: int, domain: str | None = None) -> list[ScoredChunk]:
    """Top-`k` chunks by BM25, optionally restricted to one domain. Chunks with no term match are dropped,
    so a query the KB cannot match returns nothing (escalation signal 1, FR-A15)."""
    tokens = bm25_tokens(query)
    if not tokens:
        return []
    scores = index.bm25.get_scores(tokens)
    hits = [
        ScoredChunk(chunk=chunk, score=float(score), stage="bm25")
        for chunk, score in zip(index.chunks, scores)
        if score > 0 and (domain is None or chunk.domain == domain)
    ]
    hits.sort(key=lambda h: (-h.score, h.chunk.chunk_id))
    return hits[:k]
