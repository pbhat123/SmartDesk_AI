"""T2.5: RRF fusion (k=60) and dedupe by chunk_id (FR-A6)."""
import pytest

from smartdesk_ai.rag.fusion import rrf
from smartdesk_ai.rag.models import Chunk, ScoredChunk


def sc(chunk_id, stage="bm25", score=1.0):
    entry = chunk_id.split("#")[0]
    chunk = Chunk(chunk_id=chunk_id, entry_id=entry, domain="IT", title=entry, header=f"Doc > {entry}",
                  text="text", source_file="it_support.md")
    return ScoredChunk(chunk=chunk, score=score, stage=stage)


def ranking(*ids, stage="bm25"):
    return [sc(i, stage) for i in ids]


def order(fused):
    return [f.chunk.chunk_id for f in fused]


def test_config_rrf_constant_is_60(settings):
    assert settings.retrieval.rrf_k == 60


def test_known_rankings_fuse_to_the_hand_computed_order():
    bm25 = ranking("A#0", "B#0", "C#0")
    dense = ranking("C#0", "A#0", "D#0", stage="dense")
    fused = rrf([bm25, dense], k=60)
    # A: 1/61 + 1/62      = 0.032522
    # C: 1/63 + 1/61      = 0.032266
    # B: 1/62             = 0.016129
    # D: 1/63             = 0.015873
    assert order(fused) == ["A#0", "C#0", "B#0", "D#0"]
    assert [f.score for f in fused] == pytest.approx([1 / 61 + 1 / 62, 1 / 63 + 1 / 61, 1 / 62, 1 / 63])
    assert all(f.stage == "rrf" for f in fused)


def test_k_changes_the_scores_not_the_formula():
    fused = rrf([ranking("A#0", "B#0")], k=10)
    assert [f.score for f in fused] == pytest.approx([1 / 11, 1 / 12])


def test_a_chunk_in_both_lists_appears_once():
    fused = rrf([ranking("A#0", "B#0"), ranking("B#0", "A#0", stage="dense")], k=60)
    assert sorted(order(fused)) == ["A#0", "B#0"]


def test_no_chunk_id_appears_twice_even_inside_one_list():
    fused = rrf([ranking("A#0", "B#0", "A#0"), ranking("A#0", "C#0", stage="dense")], k=60)
    assert len(order(fused)) == len(set(order(fused))) == 3
    # A's duplicate in the first list must not add a second vote: 1/61 + 1/61
    assert fused[0].chunk.chunk_id == "A#0" and fused[0].score == pytest.approx(2 / 61)


def test_pieces_of_the_same_entry_are_distinct_chunks():
    fused = rrf([ranking("A#0", "A#1")], k=60)
    assert order(fused) == ["A#0", "A#1"]


def test_ties_break_by_best_rank_then_chunk_id():
    # X is 1st in one list only, Y is 1st in the other only: equal scores, equal best rank, so chunk_id decides.
    fused = rrf([ranking("Y#0"), ranking("X#0", stage="dense")], k=60)
    assert order(fused) == ["X#0", "Y#0"]
    # P is 2nd in both lists (2/62), Q is 1st then 3rd (1/61 + 1/63): P < Q, check the score order is respected.
    fused = rrf([ranking("Q#0", "P#0", "Z#0"), ranking("Z#0", "P#0", "Q#0", stage="dense")], k=60)
    assert order(fused)[0] in {"Q#0", "Z#0"} and order(fused)[-1] == "P#0"


def test_empty_inputs():
    assert rrf([], k=60) == []
    assert rrf([[], []], k=60) == []
    assert order(rrf([[], ranking("A#0")], k=60)) == ["A#0"]


def test_input_scores_are_ignored_only_ranks_matter():
    low = [sc("A#0", score=0.001), sc("B#0", score=0.0009)]
    high = [sc("A#0", score=999.0), sc("B#0", score=1.0)]
    assert [f.score for f in rrf([low], k=60)] == [f.score for f in rrf([high], k=60)]


def test_works_on_real_retriever_output(built_index_dir, settings):
    from smartdesk_ai.rag.embeddings import HashingEmbeddings
    from smartdesk_ai.rag.index import open_index
    from smartdesk_ai.rag.retrievers import bm25_search, dense_search

    index = open_index(built_index_dir, HashingEmbeddings())
    k = settings.retrieval.k_each
    q = "how do I reset my password"
    fused = rrf([bm25_search(index, q, k), dense_search(index, q, k)], k=settings.retrieval.rrf_k)
    assert len(order(fused)) == len(set(order(fused)))
    assert "IT-PWD-01" in [f.chunk.entry_id for f in fused[:3]]
    scores = [f.score for f in fused]
    assert scores == sorted(scores, reverse=True)
