"""Disk cache and cost accounting around an embedder, so a real eval run pays for each text once."""
import json

import pytest

from evals.embedding_cache import CachedEmbeddings
from smartdesk_ai.rag.embeddings import HashingEmbeddings


class CountingInner(HashingEmbeddings):
    def __init__(self):
        super().__init__(dim=16)
        self.doc_calls, self.query_calls = [], []

    def embed_documents(self, texts):
        self.doc_calls.append(list(texts))
        return super().embed_documents(texts)

    def embed_query(self, text):
        self.query_calls.append(text)
        return super().embed_query(text)


def test_documents_are_embedded_in_one_call_and_counted(tmp_path):
    inner = CountingInner()
    c = CachedEmbeddings(inner, tmp_path / "cache.json", price_per_million=0.02, count_tokens=lambda t: len(t.split()))
    c.embed_documents(["alpha beta", "gamma delta epsilon"])
    assert inner.doc_calls == [["alpha beta", "gamma delta epsilon"]]
    assert (c.api_calls, c.tokens) == (1, 5)
    assert c.cost_usd == pytest.approx(5 * 0.02 / 1e6)


def test_repeated_texts_are_served_from_the_cache_without_api_calls(tmp_path):
    inner = CountingInner()
    c = CachedEmbeddings(inner, tmp_path / "cache.json", price_per_million=0.02, count_tokens=lambda t: len(t.split()))
    first = c.embed_query("reset my password")
    again = c.embed_query("reset my password")
    c.embed_documents(["reset my password"])  # documents and queries share the cache
    assert first == again and inner.query_calls == ["reset my password"] and inner.doc_calls == []
    assert (c.api_calls, c.hits) == (1, 2)


def test_only_the_missing_documents_are_sent(tmp_path):
    inner = CountingInner()
    c = CachedEmbeddings(inner, tmp_path / "cache.json", price_per_million=0.0, count_tokens=len)
    c.embed_documents(["a", "b"])
    vectors = c.embed_documents(["b", "c", "a"])
    assert inner.doc_calls == [["a", "b"], ["c"]]
    assert vectors == inner.embed_documents(["b", "c", "a"])  # order of the request is preserved


def test_the_cache_survives_a_restart_and_a_second_run_costs_nothing(tmp_path):
    path = tmp_path / "cache.json"
    first = CachedEmbeddings(CountingInner(), path, price_per_million=0.02, count_tokens=len)
    first.embed_query("q1")
    first.embed_documents(["d1", "d2"])
    first.save()
    inner2 = CountingInner()
    second = CachedEmbeddings(inner2, path, price_per_million=0.02, count_tokens=len)
    second.embed_query("q1")
    second.embed_documents(["d1", "d2"])
    assert inner2.query_calls == [] and inner2.doc_calls == [] and (second.api_calls, second.cost_usd) == (0, 0.0)


def test_vectors_from_different_embedders_are_never_mixed(tmp_path):
    path = tmp_path / "cache.json"
    a = CachedEmbeddings(HashingEmbeddings(dim=16), path, price_per_million=0.0, count_tokens=len)
    a.embed_query("same text")
    a.save()
    b = CachedEmbeddings(HashingEmbeddings(dim=32), path, price_per_million=0.0, count_tokens=len)
    assert len(b.embed_query("same text")) == 32
    assert b.api_calls == 1


def test_it_reports_the_inner_embedder_name_so_ingest_state_matches(tmp_path):
    inner = HashingEmbeddings(dim=16)
    assert CachedEmbeddings(inner, None, price_per_million=0.0, count_tokens=len).name == inner.name


def test_without_a_path_nothing_is_written(tmp_path):
    c = CachedEmbeddings(CountingInner(), None, price_per_million=0.0, count_tokens=len)
    c.embed_query("x")
    c.save()
    assert list(tmp_path.iterdir()) == []


def test_the_cache_file_holds_hashes_not_the_text(tmp_path):
    path = tmp_path / "cache.json"
    c = CachedEmbeddings(CountingInner(), path, price_per_million=0.0, count_tokens=len)
    c.embed_query("a secret question about salaries")
    c.save()
    assert "secret" not in path.read_text()
    assert json.loads(path.read_text())
