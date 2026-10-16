"""Opt-in check of the real bge-reranker-base (local, offline, no API cost).
Run with: RUN_REAL_MODEL_TESTS=1 pytest tests/test_reranker_real.py"""
import os

import pytest

from smartdesk_ai.config.settings import load_settings
from smartdesk_ai.rag.models import Chunk, ScoredChunk
from smartdesk_ai.rag.rerank import CrossEncoderReranker

pytestmark = pytest.mark.skipif(not os.environ.get("RUN_REAL_MODEL_TESTS"), reason="set RUN_REAL_MODEL_TESTS=1")


def sc(entry, text):
    chunk = Chunk(chunk_id=f"{entry}#0", entry_id=entry, domain="IT", title=entry, header=f"Doc > {entry}",
                  text=text, source_file="it_support.md")
    return ScoredChunk(chunk=chunk, score=0.1, stage="rrf")


def test_relevant_passage_scores_high_and_irrelevant_near_zero():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    rr = CrossEncoderReranker.from_model(load_settings().models.reranker.id)
    out = rr.rerank("how do I reset my password", [
        sc("B", "Stock options vest over four years with a one year cliff."),
        sc("A", "Reset your password at id.smartdesk.example/reset after approving the MFA prompt."),
    ])
    assert [o.chunk.entry_id for o in out] == ["A", "B"]
    assert out[0].score > 0.8 and out[1].score < 0.05
