"""Embedding back-ends: a deterministic offline mock and the real OpenAI one.

Both expose the LangChain `Embeddings` shape (`embed_documents`, `embed_query`), so
ingest and the dense retriever do not care which one they get. Tests and `--mock`
runs use `HashingEmbeddings`; it never touches the network or the API budget.
"""
import hashlib
import math
import re
from typing import Protocol

from smartdesk_ai.config.settings import Settings

_WORD = re.compile(r"[a-z0-9₹]+")
_STOP = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it me my of on or so the to we what when "
    "where which who will with you your".split()
)


class Embeddings(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbeddings:
    """Signed bag-of-words hashing into `dim` buckets, L2-normalised.

    Texts that share content words get a high cosine, which is enough to exercise
    the dense retriever, fusion and ablation code deterministically.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in _WORD.findall(text.lower()):
            if word in _STOP:
                continue
            if len(word) > 4 and word.endswith("s"):
                word = word[:-1]
            digest = hashlib.md5(word.encode()).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            vec[bucket] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm else vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def get_embeddings(settings: Settings) -> Embeddings:
    """The real embedder (OpenAI). Calling it spends API budget: only for real ingest runs."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.models.embedding.id)
