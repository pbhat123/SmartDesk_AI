"""T2.3: BM25 retriever stage (FR-A6)."""
import pytest

from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.index import IndexNotBuilt, bm25_tokens, open_index
from smartdesk_ai.rag.retrievers import bm25_search


class NoEmbeddings(HashingEmbeddings):
    """Proves BM25 is callable alone: any embedding call fails the test."""

    def embed_query(self, text):
        raise AssertionError("bm25_search must not call the embedder")

    def embed_documents(self, texts):
        raise AssertionError("bm25_search must not call the embedder")


@pytest.fixture(scope="module")
def index(built_index_dir):
    return open_index(built_index_dir, NoEmbeddings())


def ids(results):
    return [r.chunk.entry_id for r in results]


def test_returns_the_top_20_ranked_chunks(index, settings):
    results = bm25_search(index, "employee company work days", settings.retrieval.k_each)
    assert len(results) == 20
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(r.stage == "bm25" and r.score > 0 for r in results)
    assert len({r.chunk.chunk_id for r in results}) == 20


def test_k_limits_the_result_count(index):
    assert len(bm25_search(index, "employee company work days", 3)) == 3


def test_exact_wording_finds_the_right_entry(index):
    assert "IT-PWD-01" in ids(bm25_search(index, "I forgot my password, how do I reset it", 3))
    assert "HR-STOCK-01" in ids(bm25_search(index, "stock option vesting cliff", 3))


def test_the_header_is_indexed_too(index):
    assert ids(bm25_search(index, "Anti-harassment policy", 5))[0] == "HR-HARASS-01"


def test_domain_filter_restricts_results(index):
    it = bm25_search(index, "laptop request approval", 20, domain="IT")
    hr = bm25_search(index, "laptop request approval", 20, domain="HR")
    assert it and hr
    assert {r.chunk.domain for r in it} == {"IT"}
    assert {r.chunk.domain for r in hr} == {"HR"}


def test_no_filter_searches_the_whole_kb(index):
    domains = {r.chunk.domain for r in bm25_search(index, "laptop request approval", 20)}
    assert domains == {"IT", "HR"}


def test_unknown_words_and_stopword_only_queries_return_nothing(index):
    assert bm25_search(index, "zzqxv blorptastic", 20) == []
    assert bm25_search(index, "how do I", 20) == []
    assert bm25_search(index, "", 20) == []


def test_equal_scores_are_ordered_deterministically(index):
    a = [r.chunk.chunk_id for r in bm25_search(index, "extension", 20)]
    b = [r.chunk.chunk_id for r in bm25_search(index, "extension", 20)]
    assert a == b


def test_tokens_are_lowercase_words_without_stopwords():
    assert bm25_tokens("How do I reset the VPN password?") == ["reset", "vpn", "password"]


def test_opening_a_missing_index_gives_a_clear_error(tmp_path):
    with pytest.raises(IndexNotBuilt, match="smartdesk_ai.rag.ingest"):
        open_index(tmp_path / "nothing", HashingEmbeddings())
