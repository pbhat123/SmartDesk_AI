"""Embedding back-ends: a deterministic offline mock and the real OpenAI one.

Both expose the LangChain `Embeddings` shape (`embed_documents`, `embed_query`), so
ingest and the dense retriever do not care which one they get. Tests and `--mock`
runs use `HashingEmbeddings`; it never touches the network or the API budget.
"""
import hashlib
import math
import re
from typing import Protocol

from smartdesk_ai.config.env import EnvSettings, load_env
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.rag.index import STOPWORDS

_WORD = re.compile(r"[a-z0-9₹]+")


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

    @property
    def name(self) -> str:
        return f"hashing-{self.dim}"

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in _WORD.findall(text.lower()):
            if word in STOPWORDS:
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


def get_embeddings(settings: Settings, env: EnvSettings | None = None) -> Embeddings:
    """The real embedder (OpenAI). Calling it spends API budget: only for real ingest runs.
    The key comes from the project's env loader, because `.env` is not exported to `os.environ`."""
    env = env or load_env()
    if env.openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not set (see .env.example)")
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=settings.models.embedding.id, api_key=env.openai_api_key)


def embedder_id(embeddings: Embeddings) -> str:
    """A stable label for which embedding model produced the stored vectors, so switching models
    (or the mock and the real one) forces a full re-embed instead of mixing vector spaces."""
    for attr in ("name", "model", "model_name"):
        value = getattr(embeddings, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(embeddings).__name__
