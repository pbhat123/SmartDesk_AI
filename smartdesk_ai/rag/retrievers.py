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


def dense_search(index: KBIndex, query: str, k: int, domain: str | None = None) -> list[ScoredChunk]:
    """Top-`k` chunks by cosine similarity of the query embedding, optionally restricted to one domain.
    The score is the cosine similarity itself (1 - Chroma's cosine distance). The best one is the
    escalation signal the single threshold applies to (FR-A15)."""
    if not query.strip():
        return []
    vector = index.embeddings.embed_query(query)
    if not any(vector):
        return []
    found = index.collection.query(
        query_embeddings=[vector],
        n_results=k,
        where={"domain": domain} if domain else None,
        include=["distances"],
    )
    by_id = index.by_id
    hits = [
        ScoredChunk(chunk=by_id[chunk_id], score=1.0 - float(distance), stage="dense")
        for chunk_id, distance in zip(found["ids"][0], found["distances"][0])
    ]
    hits.sort(key=lambda h: (-h.score, h.chunk.chunk_id))
    return hits
