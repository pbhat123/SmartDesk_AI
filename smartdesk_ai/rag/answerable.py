"""Answerable signals 1 and 2 of the escalation rule (FR-A15, FR-A5).

Signal 3 (the LLM reports insufficient context) comes from the answering step, not from retrieval.
The threshold depends on the score scale: the escalation threshold applies to normalised reranker
scores, the fallback threshold to the top dense cosine similarity when the reranker is down (FR-F7).
"""
from typing import Literal

from pydantic import BaseModel

from smartdesk_ai.config.settings import Thresholds

ScoreKind = Literal["reranker", "dense_cosine"]


class Signals(BaseModel):
    no_chunks: bool  # signal 1: retrieval returned nothing
    below_threshold: bool  # signal 2: the top score is under the threshold for its score kind


class Answerability(BaseModel):
    answerable: bool
    signals: Signals
    threshold: float
    score_kind: ScoreKind


def assess(n_chunks: int, top_score: float, score_kind: ScoreKind, thresholds: Thresholds) -> Answerability:
    threshold = thresholds.escalation if score_kind == "reranker" else thresholds.fallback
    no_chunks = n_chunks == 0
    below = not no_chunks and top_score < threshold  # an empty result is signal 1 alone
    return Answerability(
        answerable=not (no_chunks or below),
        signals=Signals(no_chunks=no_chunks, below_threshold=below),
        threshold=threshold,
        score_kind=score_kind,
    )
