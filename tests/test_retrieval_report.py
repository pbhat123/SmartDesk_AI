"""Per-question retrieval ranks, hit@k and MRR per strategy, with a lexical-overlap split to expose paraphrase bias
(FR-A7). Mock embeddings and reranker; the real report reads only cached embeddings and never calls the API."""
import json

import pytest

from evals import retrieval_report as rr
from evals.embedding_cache import CachedEmbeddings
from evals.harness import make_resources
from evals.schemas import KBQuestion
from smartdesk_ai.rag.index import open_index
from smartdesk_ai.rag.ingest import ingest
from tests.kb_helpers import KB_DIR


def test_hit_at_k_and_mrr_from_hand_made_ranks():
    m = rr.rank_metrics([1, 2, 4, None, 6])
    assert m["hit@1"] == pytest.approx(1 / 5) and m["hit@3"] == pytest.approx(2 / 5) and m["hit@5"] == pytest.approx(3 / 5)
    assert m["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 4 + 0 + 1 / 6) / 5)
    assert rr.rank_metrics([])["mrr"] == 0.0


def test_overlap_share_is_the_share_of_question_words_found_in_the_entry():
    assert rr.overlap_share("how do I reset my password quickly", "reset the password in the portal") == pytest.approx(2 / 3)
    assert rr.overlap_share("???", "anything") == 0.0
    assert rr.overlap_share("laptop battery swelling", "unrelated text about leave") == 0.0


def test_the_overlap_split_puts_the_lower_half_first():
    rows = [KBQuestion(id=f"q{i}", question="a question here?", expected_entry_id="E", expected_domain="IT",
                       type="simple", reference_answer="a reference answer of enough length", split="dev") for i in range(4)]
    low, high = rr.split_by_overlap(rows, {"q0": 0.9, "q1": 0.1, "q2": 0.5, "q3": 0.3})
    assert [r.id for r in low] == ["q1", "q3"] and [r.id for r in high] == ["q2", "q0"]


@pytest.fixture(scope="module")
def mock_setup(tmp_path_factory, settings):
    res = make_resources(mock=True)
    index_dir = tmp_path_factory.mktemp("rr_index")
    ingest(KB_DIR, index_dir, res.embeddings, settings.chunking)
    return res, open_index(index_dir, res.embeddings)


def test_the_report_covers_every_strategy_split_type_and_overlap_half(mock_setup, tmp_path):
    res, index = mock_setup
    report = rr.run(res, index, tmp_path)
    assert set(report["overall"]) == set(rr.STRATEGIES) and report["n"] == 53
    for strategy in rr.STRATEGIES:
        assert set(report["overall"][strategy]) == {"hit@1", "hit@3", "hit@5", "mrr"}
        assert 0 <= report["overall"][strategy]["hit@1"] <= report["overall"][strategy]["hit@5"] <= 1
    assert set(report["by_split"]) == {"dev", "test"} and {"simple", "how-to", "policy interpretation"} <= set(report["by_type"])
    assert set(report["by_overlap"]) == {"low", "high"} and 0 <= report["overlap_mean"] <= 1
    saved = json.loads((tmp_path / "retrieval_report.json").read_text())
    assert saved["ranks"]["dense"]["kb-001"] == report["ranks"]["dense"]["kb-001"]


def test_the_offline_embedder_fails_on_an_uncached_text_so_no_api_call_can_happen():
    emb = CachedEmbeddings(rr.OfflineEmbeddings("text-embedding-3-small"), None, price_per_million=0.0)
    with pytest.raises(RuntimeError, match="not in the embedding cache"):
        emb.embed_query("a question that was never embedded")
