"""T2.4: dense retriever stage (FR-A6). Embeddings are mocked."""
from dataclasses import replace

import pytest

from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.index import open_index
from smartdesk_ai.rag.retrievers import dense_search


class SpyEmbeddings(HashingEmbeddings):
    def __init__(self):
        super().__init__()
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return super().embed_query(text)


@pytest.fixture(scope="module")
def spy():
    return SpyEmbeddings()


@pytest.fixture(scope="module")
def index(built_index_dir, spy):
    return open_index(built_index_dir, spy)


def ids(results):
    return [r.chunk.entry_id for r in results]


def test_returns_the_top_20_ranked_chunks(index, settings):
    results = dense_search(index, "employee company work days policy", settings.retrieval.k_each)
    assert len(results) == 20
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(r.stage == "dense" and -1.0 <= r.score <= 1.0 for r in results)
    assert len({r.chunk.chunk_id for r in results}) == 20


def test_k_limits_the_result_count_and_k_above_the_corpus_is_safe(index):
    assert len(dense_search(index, "password reset", 3)) == 3
    assert len(dense_search(index, "password reset", 500)) == len(index.chunks)


def test_similar_wording_finds_the_right_entry_and_scores_are_cosine(index):
    top = dense_search(index, "forgot password reset yourself", 3)
    assert "IT-PWD-01" in ids(top)
    assert 0.0 < top[0].score <= 1.0


def test_results_carry_the_full_chunk(index):
    results = dense_search(index, "stock option vesting cliff", 3)
    chunk = next(r.chunk for r in results if r.chunk.entry_id == "HR-STOCK-01")
    assert chunk.domain == "HR" and chunk.chunk_id == "HR-STOCK-01#0"
    assert chunk.header.endswith("Stock-option vesting schedule") and chunk.text


def test_domain_filter_restricts_results(index):
    it = dense_search(index, "laptop request approval", 20, domain="IT")
    hr = dense_search(index, "laptop request approval", 20, domain="HR")
    assert it and hr
    assert {r.chunk.domain for r in it} == {"IT"}
    assert {r.chunk.domain for r in hr} == {"HR"}
    assert len(it) == sum(c.domain == "IT" for c in index.chunks)  # 20 IT chunks, all returned


def test_no_filter_searches_the_whole_kb(index):
    domains = {r.chunk.domain for r in dense_search(index, "laptop request approval", 20)}
    assert domains == {"IT", "HR"}


def test_the_query_is_embedded_once_as_given(index, spy):
    spy.queries.clear()
    dense_search(index, "How do I reset my password?", 5)
    assert spy.queries == ["How do I reset my password?"]


def test_callable_alone_without_the_bm25_index(index):
    alone = replace(index, bm25=None)
    assert dense_search(alone, "password reset", 3)


def test_empty_and_contentless_queries_return_nothing(index):
    assert dense_search(index, "", 5) == []
    assert dense_search(index, "how do I", 5) == []  # mock embeds stopwords to a zero vector


def test_ordering_is_deterministic(index):
    a = [r.chunk.chunk_id for r in dense_search(index, "extension", 20)]
    b = [r.chunk.chunk_id for r in dense_search(index, "extension", 20)]
    assert a == b
