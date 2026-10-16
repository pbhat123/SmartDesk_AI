"""A caching, cost-counting wrapper around an embedder, for eval runs.

Every unique text is sent to the embedding API once, ever: vectors are kept in a JSON file keyed
by embedder name and a hash of the text (never the text itself). `api_calls`, `tokens` and
`cost_usd` count only what was actually sent, so a run can report exactly what it spent.
"""
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from smartdesk_ai.rag.chunking import count_tokens as tiktoken_count
from smartdesk_ai.rag.embeddings import Embeddings, embedder_id


class CachedEmbeddings:
    def __init__(
        self,
        inner: Embeddings,
        path: Path | None,
        *,
        price_per_million: float,
        count_tokens: Callable[[str], int] = tiktoken_count,
    ) -> None:
        self._inner, self._path = inner, Path(path) if path else None
        self._price, self._count = price_per_million, count_tokens
        self.name = embedder_id(inner)
        self.api_calls = self.tokens = self.hits = 0
        self._store: dict[str, list[float]] = {}
        if self._path and self._path.is_file():
            self._store = json.loads(self._path.read_text()).get(self.name, {})

    @property
    def cost_usd(self) -> float:
        return self.tokens * self._price / 1_000_000

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if self._key(t) not in self._store))
        if missing:
            self.api_calls += 1  # one batched request
            self.tokens += sum(self._count(t) for t in missing)
            for text, vector in zip(missing, self._inner.embed_documents(missing)):
                self._store[self._key(text)] = vector
        self.hits += len(texts) - len(missing)
        return [self._store[self._key(t)] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        key = self._key(text)
        if key in self._store:
            self.hits += 1
        else:
            self.api_calls += 1
            self.tokens += self._count(text)
            self._store[key] = self._inner.embed_query(text)
        return self._store[key]

    def save(self) -> None:
        if not self._path:
            return
        data = json.loads(self._path.read_text()) if self._path.is_file() else {}
        data[self.name] = self._store
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self._path)
