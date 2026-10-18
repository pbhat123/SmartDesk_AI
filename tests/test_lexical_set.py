"""The held-out exact-term set (FR-A7, FR-A18): every query is built on a literal that only its expected entry contains,
and the report scores it with all five strategies. Mock embeddings, no network."""
import pytest

from evals import lexical_report as lx
from evals.embedding_cache import CachedEmbeddings
from evals.harness import make_resources
from evals.retrieval_report import STRATEGIES, OfflineEmbeddings
from smartdesk_ai.rag.index import open_index
from smartdesk_ai.rag.ingest import ingest
from tests.kb_helpers import KB_DIR, load_all_entries

ROWS = lx.lexical_rows()


def test_the_set_has_fifteen_unique_held_out_queries():
    assert len(ROWS) == 15 and len({r.id for r in ROWS}) == 15 and len({r.question for r in ROWS}) == 15
    assert all(r.heldout for r in ROWS)


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r.id)
def test_each_literal_is_in_the_query_and_only_in_the_expected_entry(row):
    holders = {e.id for e in load_all_entries() if row.literal.lower() in f"{e.title} {e.body}".lower()}
    assert holders == {row.expected_entry_id}, f"{row.literal!r} is in {holders}"
    assert row.literal.lower() in row.question.lower()


def test_the_report_scores_every_strategy_on_the_set(tmp_path, settings):
    res = make_resources(mock=True)
    index_dir = tmp_path / "idx"
    ingest(KB_DIR, index_dir, res.embeddings, settings.chunking)
    report = lx.build_report(open_index(index_dir, res.embeddings), ROWS, res)
    assert report["n"] == 15 and set(report["overall"]) == set(STRATEGIES)
    for m in report["overall"].values():
        assert 0 <= m["hit@1"] <= m["hit@3"] <= m["hit@5"] <= 1
    assert report["overall"]["bm25"]["hit@5"] > 0.5  # the exact terms are findable by keyword search


def test_the_default_run_cannot_call_the_api():
    emb = CachedEmbeddings(OfflineEmbeddings("text-embedding-3-small"), None, price_per_million=0.0)
    with pytest.raises(RuntimeError, match="not in the embedding cache"):
        emb.embed_query(ROWS[0].question)
