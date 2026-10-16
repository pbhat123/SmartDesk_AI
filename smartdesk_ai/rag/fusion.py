"""Reciprocal Rank Fusion (FR-A6)."""
from smartdesk_ai.rag.models import ScoredChunk


def rrf(rankings: list[list[ScoredChunk]], *, k: int, weights: list[float] | None = None) -> list[ScoredChunk]:
    """Fuse ranked lists: a chunk's score is the sum of weight / (k + rank) over the lists it appears in
    (rank starts at 1; every weight is 1 unless `weights` says otherwise, e.g. to trust dense more).
    Input scores are ignored, only positions count. Each `chunk_id` appears once in the output, even if
    it is repeated within a list. Ties go to the better best-rank, then chunk_id."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"weights has {len(weights)} entries for {len(rankings)} lists")
    if any(w <= 0 for w in weights):
        raise ValueError("weights must be positive")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    chunks = {}
    for ranking, weight in zip(rankings, weights):
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item.chunk.chunk_id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunks.setdefault(chunk_id, item.chunk)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
            best_rank[chunk_id] = min(best_rank.get(chunk_id, rank), rank)
    order = sorted(scores, key=lambda cid: (-scores[cid], best_rank[cid], cid))
    return [ScoredChunk(chunk=chunks[cid], score=scores[cid], stage="rrf") for cid in order]
