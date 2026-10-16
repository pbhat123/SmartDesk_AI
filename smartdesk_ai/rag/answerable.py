"""Answerable signals 1 and 2 of the escalation rule (FR-A5, FR-A15).

Signal 3 (the LLM reports insufficient context) comes from the answering step, not from retrieval.
Signal 2 compares the top dense cosine similarity with the single escalation threshold. It does not
depend on the reranker, which only orders candidates (SPEC v3.1), so the rule is the same whether or
not the reranker is available.
"""
from pydantic import BaseModel

from smartdesk_ai.config.settings import Thresholds


class Signals(BaseModel):
    no_chunks: bool  # signal 1: retrieval returned nothing
    below_threshold: bool  # signal 2: the top dense cosine is under the escalation threshold


class Answerability(BaseModel):
    answerable: bool
    signals: Signals
    threshold: float


def assess(n_chunks: int, top_cosine: float, thresholds: Thresholds) -> Answerability:
    no_chunks = n_chunks == 0
    below = not no_chunks and top_cosine < thresholds.escalation  # an empty result is signal 1 alone
    return Answerability(
        answerable=not (no_chunks or below),
        signals=Signals(no_chunks=no_chunks, below_threshold=below),
        threshold=thresholds.escalation,
    )
