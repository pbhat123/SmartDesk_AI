"""Cross-encoder reranking and collapse of pieces to entries (FR-A6, FR-A15)."""
import math
from collections.abc import Callable, Sequence
from typing import Protocol

from smartdesk_ai.config.settings import Settings
from smartdesk_ai.ops.health import Component, HealthRegistry, get_registry
from smartdesk_ai.rag.models import ScoredChunk

PairScorer = Callable[[list[tuple[str, str]]], Sequence[float]]


def sigmoid(x: float) -> float:
    """Logistic function that cannot overflow: maps a raw cross-encoder logit to 0..1 (FR-A15)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]: ...


class NoopReranker:
    """Keeps the incoming (RRF) order and scores. The substitute when the reranker is unavailable (FR-F7)."""

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        return list(candidates)


class CrossEncoderReranker:
    """Scores (query, passage) pairs with a cross-encoder and returns every candidate, best first,
    with sigmoid-normalised scores in 0..1. `predict` maps pairs to raw logits; tests pass a mock."""

    def __init__(self, predict: PairScorer) -> None:
        self._predict = predict

    @classmethod
    def from_cross_encoder(cls, model) -> "CrossEncoderReranker":
        """Wrap a sentence-transformers CrossEncoder. Its `predict` applies a sigmoid to single-label
        models by default, which would be squashed a second time here, so ask for the raw logits."""
        from torch.nn import Identity

        return cls(lambda pairs: [float(s) for s in model.predict(pairs, activation_fn=Identity())])

    @classmethod
    def from_model(cls, model_id: str) -> "CrossEncoderReranker":
        """Load a sentence-transformers cross-encoder. The first call downloads the model."""
        from sentence_transformers import CrossEncoder

        return cls.from_cross_encoder(CrossEncoder(model_id))

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        logits = self._predict([(query, c.chunk.embed_text) for c in candidates])
        scored = [ScoredChunk(chunk=c.chunk, score=sigmoid(float(x)), stage="rerank") for c, x in zip(candidates, logits)]
        scored.sort(key=lambda s: (-s.score, s.chunk.chunk_id))
        return scored


def collapse_to_entries(ranked: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
    """One result per entry, the best-scoring piece winning, then the best `top_n` entries."""
    best: dict[str, ScoredChunk] = {}
    for item in ranked:
        current = best.get(item.chunk.entry_id)
        if current is None or item.score > current.score:
            best[item.chunk.entry_id] = item
    return sorted(best.values(), key=lambda s: (-s.score, s.chunk.chunk_id))[:top_n]


def load_reranker(cfg: Settings, health: HealthRegistry | None = None) -> Reranker | None:
    """Load the configured cross-encoder. On any failure the reranker is recorded as degraded and None
    is returned, so startup never crashes and the pipeline falls back to RRF order (FR-F7)."""
    health = health or get_registry()
    return health.try_load(Component.RERANKER, lambda: CrossEncoderReranker.from_model(cfg.models.reranker.id))
