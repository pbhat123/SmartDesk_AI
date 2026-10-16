"""T2.6: cross-encoder rerank, sigmoid scores, entry collapse and the retrieval pipeline (FR-A6, FR-A15).
The cross-encoder is mocked: no model is downloaded."""
import math
import re

import pytest

from smartdesk_ai.config.settings import Chunking
from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.index import open_index
from smartdesk_ai.rag.ingest import ingest
from smartdesk_ai.rag.models import Chunk, ScoredChunk
from smartdesk_ai.rag.pipeline import retrieve
from smartdesk_ai.rag.rerank import CrossEncoderReranker, NoopReranker, collapse_to_entries, sigmoid
from tests.kb_helpers import KB_DIR, load_all_entries

BODIES = {e.id: e.body for e in load_all_entries()}
WORD = re.compile(r"[a-z0-9]+")


def chunk(chunk_id, domain="IT", text="text"):
    entry = chunk_id.split("#")[0]
    return Chunk(chunk_id=chunk_id, entry_id=entry, domain=domain, title=entry, header=f"Doc > {entry}",
                 text=text, source_file="it_support.md", piece=int(chunk_id.split("#")[1]))


def scored(chunk_id, score, stage="rrf"):
    return ScoredChunk(chunk=chunk(chunk_id), score=score, stage=stage)


class OverlapScorer:
    """Mock cross-encoder: raw logit grows with the number of query words found in the passage."""

    def __init__(self):
        self.calls = []

    def __call__(self, pairs):
        self.calls.append(pairs)
        out = []
        for query, passage in pairs:
            q, p = set(WORD.findall(query.lower())), set(WORD.findall(passage.lower()))
            out.append(2.0 * len(q & p) - 4.0)
        return out


# --- sigmoid and CrossEncoderReranker -------------------------------------------------

def test_sigmoid_maps_logits_into_zero_one_without_overflow():
    assert sigmoid(0.0) == 0.5
    assert sigmoid(1000.0) == 1.0 and sigmoid(-1000.0) == 0.0
    assert math.isclose(sigmoid(2.0), 1 / (1 + math.exp(-2.0)))


def test_reranker_scores_are_normalised_sorted_and_staged():
    scores = {"A": -3.0, "B": 6.0, "C": 0.5}
    rr = CrossEncoderReranker(lambda pairs: [scores[p[1].split("\n")[0].split(" > ")[1]] for p in pairs])
    out = rr.rerank("q", [scored("A#0", 0.1), scored("B#0", 0.2), scored("C#0", 0.3)])
    assert [o.chunk.chunk_id for o in out] == ["B#0", "C#0", "A#0"]
    assert all(0.0 <= o.score <= 1.0 and o.stage == "rerank" for o in out)
    assert out[0].score == pytest.approx(sigmoid(6.0))


def test_reranker_pairs_the_query_with_the_header_and_text():
    seen = []
    rr = CrossEncoderReranker(lambda pairs: seen.extend(pairs) or [0.0] * len(pairs))
    rr.rerank("my query", [scored("A#0", 1.0)])
    assert seen == [("my query", "Doc > A\ntext")]


def test_reranker_with_no_candidates_does_not_call_the_model():
    rr = CrossEncoderReranker(lambda pairs: pytest.fail("model must not be called"))
    assert rr.rerank("q", []) == []


def test_noop_reranker_keeps_the_given_order_and_scores():
    cands = [scored("A#0", 0.9), scored("B#0", 0.5)]
    assert NoopReranker().rerank("q", cands) == cands


# --- collapse -------------------------------------------------------------------------

def test_collapse_keeps_the_best_piece_per_entry_and_orders_by_score():
    ranked = [scored("A#1", 0.9, "rerank"), scored("B#0", 0.8, "rerank"), scored("A#0", 0.7, "rerank"),
              scored("C#0", 0.6, "rerank")]
    out = collapse_to_entries(ranked, top_n=5)
    assert [o.chunk.chunk_id for o in out] == ["A#1", "B#0", "C#0"]
    assert len({o.chunk.entry_id for o in out}) == len(out)


def test_collapse_returns_at_most_top_n_entries():
    ranked = [scored(f"E{i}#0", 1 - i / 10, "rerank") for i in range(8)]
    assert [o.chunk.entry_id for o in collapse_to_entries(ranked, top_n=5)] == [f"E{i}" for i in range(5)]


# --- pipeline -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def index(built_index_dir):
    return open_index(built_index_dir, HashingEmbeddings())


def test_pipeline_returns_at_most_five_unique_entries_with_normalised_scores(index, settings):
    result = retrieve(index, "how do I reset my password", cfg=settings, reranker=CrossEncoderReranker(OverlapScorer()))
    ids = [e.entry_id for e in result.entries]
    assert 1 <= len(ids) <= settings.retrieval.top_n == 5
    assert len(ids) == len(set(ids))
    assert all(0.0 <= e.score <= 1.0 for e in result.entries)
    assert [e.score for e in result.entries] == sorted((e.score for e in result.entries), reverse=True)
    assert result.score_kind == "reranker" and result.top_score == result.entries[0].score


def test_pipeline_finds_the_right_entry_with_a_sensible_mock_reranker(index, settings):
    result = retrieve(index, "reset my forgotten password", cfg=settings, reranker=CrossEncoderReranker(OverlapScorer()))
    assert result.entries[0].entry_id == "IT-PWD-01"


def test_no_chunk_id_appears_twice_in_the_reranker_input(index, settings):
    scorer = OverlapScorer()
    retrieve(index, "laptop request approval for my team", cfg=settings, reranker=CrossEncoderReranker(scorer))
    (pairs,) = scorer.calls
    assert len(pairs) == len(set(pairs)) <= 2 * settings.retrieval.k_each


def test_output_carries_the_full_parent_entry_text(index, settings):
    result = retrieve(index, "stock option vesting cliff", cfg=settings, reranker=CrossEncoderReranker(OverlapScorer()))
    for e in result.entries:
        assert e.text == BODIES[e.entry_id]
        assert e.header.count(" > ") == 1 and e.title and e.source_file.endswith(".md")


def test_domain_filter_reaches_both_retrievers(index, settings):
    result = retrieve(index, "laptop request approval", domain="IT", cfg=settings,
                      reranker=CrossEncoderReranker(OverlapScorer()))
    assert result.entries and {e.domain for e in result.entries} == {"IT"}


def test_nothing_matching_gives_no_entries_and_skips_the_model(index, settings):
    result = retrieve(index, "", cfg=settings, reranker=CrossEncoderReranker(lambda p: pytest.fail("no candidates")))
    assert result.entries == [] and result.top_score == 0.0


def test_split_entries_return_the_whole_entry_not_one_piece(tmp_path, settings):
    small = Chunking(target_min_tokens=50, target_max_tokens=100, split_above_tokens=150, overlap_pct=15)
    ingest(KB_DIR, tmp_path, HashingEmbeddings(), small)
    ix = open_index(tmp_path, HashingEmbeddings())
    assert any(c.pieces > 1 for c in ix.chunks)
    result = retrieve(ix, "reset my forgotten password", cfg=settings, reranker=CrossEncoderReranker(OverlapScorer()))
    assert len({e.entry_id for e in result.entries}) == len(result.entries)
    for e in result.entries:
        assert e.text == BODIES[e.entry_id]


# --- the wrapper around a real sentence-transformers model ------------------------------

class FakeCrossEncoder:
    """Records how predict is called, like sentence-transformers' CrossEncoder."""

    def __init__(self, logits):
        self.logits, self.kwargs = logits, None

    def predict(self, pairs, **kwargs):
        self.kwargs = kwargs
        return self.logits


def test_from_cross_encoder_asks_for_raw_logits_so_sigmoid_is_applied_once():
    from torch.nn import Identity

    model = FakeCrossEncoder([1.8651, -10.195])
    rr = CrossEncoderReranker.from_cross_encoder(model)
    out = rr.rerank("q", [scored("A#0", 0.1), scored("B#0", 0.2)])
    assert isinstance(model.kwargs["activation_fn"], Identity)
    assert out[0].score == pytest.approx(sigmoid(1.8651)) and out[0].score > 0.85
    assert out[1].score < 0.001  # an irrelevant passage must be able to score near 0
