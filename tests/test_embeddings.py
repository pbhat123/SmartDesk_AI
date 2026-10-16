"""Mock embeddings used by every retrieval test (no network)."""
import math

from smartdesk_ai.rag.embeddings import HashingEmbeddings


def test_vectors_are_deterministic_unit_length_and_fixed_size():
    e = HashingEmbeddings(dim=64)
    a, b = e.embed_query("reset my password"), e.embed_query("reset my password")
    assert a == b and len(a) == 64
    assert math.isclose(sum(x * x for x in a), 1.0, rel_tol=1e-9)


def test_shared_words_mean_higher_cosine_than_unrelated_text():
    e = HashingEmbeddings()
    cos = lambda x, y: sum(p * q for p, q in zip(x, y))
    q = e.embed_query("how do I reset my password")
    near = e.embed_query("reset your password yourself")
    far = e.embed_query("stock option vesting schedule cliff")
    assert cos(q, near) > cos(q, far)


def test_documents_and_queries_use_the_same_space_and_empty_text_is_safe():
    e = HashingEmbeddings()
    assert e.embed_documents(["abc def"])[0] == e.embed_query("abc def")
    assert e.embed_query("") == [0.0] * e.dim
