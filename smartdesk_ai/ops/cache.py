"""The answer cache (SPEC section 4 cache rules, FR-H6, FR-H7, FR-F6).

KB answers only. The key is the orchestrator's rewritten standalone sub-question (plus the domain filter it was searched
with), so "what about sick leave?" is never mixed up with another topic. The text is the already masked one: the input
guard has run, and a question that still carries a redaction or an email is not looked up or stored at all. The stored key
is a salted digest and the value holds the part result only, so no question text sits in the cache or in Redis.

A hit skips retrieval, the worker and the judge. Only a shown, grounded answer is stored: never an escalation, a
judge-rejected draft, an unanswerable question or an outage, and the ticket and status paths never reach this module.
Entries carry the KB version; one made for another version (another process re-ingested) is never served. The cache must
never break a chat turn: any backend error is a miss (FR-F7).

Backends (FR-F6, FR-F7): `MemoryBackend` is the default and the substitute; `RedisBackend` is used when REDIS_URL is set
and reachable. `FallbackBackend` runs the cache in memory while Redis fails and asks Redis again after a wait, so a
Redis outage costs hits, never a turn. The KB change (FR-H6) clears the whole cache through `clear_answers`.

Semantic stage: when the exact key misses, the question is embedded (one embedding call, counted by the caller's costed
embedder, FR-H7) and compared with the stored questions' vectors of the same domain and KB version. The nearest one at or
above the config cut-off (`thresholds.semantic_cache_cutoff`, high on purpose) is a semantic hit. The vector of that
lookup is kept and stored with the answer, so storing costs no second embedding.
"""
import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from smartdesk_ai.agent.state import Domain, PartResult
from smartdesk_ai.config.env import EnvSettings
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.guardrails.pii import EMAIL, PLACEHOLDERS
from smartdesk_ai.ops.cache_log import CacheLog
from smartdesk_ai.ops.health import Component, HealthRegistry
from smartdesk_ai.ops.runtime import salted_digest

logger = logging.getLogger(__name__)

Kind = Literal["exact", "semantic", "miss", "bypass"]


class CacheBackend(Protocol):
    """A string store: the in-memory one, or Redis behind the same calls."""

    def get(self, key: str) -> str | None: ...
    def put(self, key: str, value: str) -> None: ...
    def entries(self) -> list[tuple[str, str]]: ...
    def clear(self) -> None: ...
    def __len__(self) -> int: ...


class MemoryBackend:
    """Thread-safe, bounded: once full, the oldest entry goes first."""

    def __init__(self, max_entries: int) -> None:
        self._max = max_entries
        self._lock = threading.Lock()
        self._data: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, value: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._data[key] = value
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def entries(self) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._data.items())

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class RedisBackend:
    """Entries in one Redis hash, with a sorted set of write times to find the oldest. The client must decode
    responses to str (`redis_from_url` does)."""

    def __init__(self, client: Any, *, namespace: str, max_entries: int) -> None:
        self._client, self._max = client, max_entries
        self._hash, self._order = f"{namespace}:entries", f"{namespace}:order"
        self._lock = threading.Lock()
        self._last = 0.0

    @staticmethod
    def _text(value: Any) -> str | None:
        return value.decode() if isinstance(value, bytes) else value

    def _next_score(self) -> float:
        with self._lock:  # strictly increasing, so "oldest" is never a tie
            self._last = max(time.time(), self._last + 1e-6)
            return self._last

    def get(self, key: str) -> str | None:
        return self._text(self._client.hget(self._hash, key))

    def put(self, key: str, value: str) -> None:
        self._client.hset(self._hash, key, value)
        self._client.zadd(self._order, {key: self._next_score()})
        excess = self._client.hlen(self._hash) - self._max
        if excess > 0:
            dropped = [self._text(member) for member, _ in self._client.zpopmin(self._order, excess)]
            if dropped:
                self._client.hdel(self._hash, *dropped)

    def entries(self) -> list[tuple[str, str]]:
        return [(self._text(k), self._text(v)) for k, v in self._client.hgetall(self._hash).items()]

    def clear(self) -> None:
        self._client.delete(self._hash, self._order)

    def __len__(self) -> int:
        return int(self._client.hlen(self._hash))


class FallbackBackend:
    """Use `primary` (Redis); while it fails, use `fallback` (memory) and ask the primary again only after
    `retry_after` seconds. `on_down(error)` runs for each failed attempt, `on_up()` when the primary answers again,
    after which the memory entries made during the outage are dropped (they could be stale by the next one)."""

    def __init__(
        self,
        primary: CacheBackend,
        fallback: CacheBackend,
        *,
        retry_after: float,
        on_down: Callable[[Exception], None],
        on_up: Callable[[], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._primary, self._fallback = primary, fallback
        self._retry_after, self._on_down, self._on_up, self._clock = retry_after, on_down, on_up, clock
        self._lock = threading.Lock()
        self._retry_at = 0.0
        self._down = False

    def _call(self, name: str, *args: Any) -> Any:
        now = self._clock()
        with self._lock:
            may_try = now >= self._retry_at
        if may_try:
            try:
                result = getattr(self._primary, name)(*args)
            except Exception as error:
                with self._lock:
                    self._retry_at, self._down = now + self._retry_after, True
                self._on_down(error)
            else:
                with self._lock:
                    recovered, self._down = self._down, False
                if recovered:
                    self._fallback.clear()
                    self._on_up()
                return result
        return getattr(self._fallback, name)(*args)

    def get(self, key: str) -> str | None:
        return self._call("get", key)

    def put(self, key: str, value: str) -> None:
        self._call("put", key, value)

    def entries(self) -> list[tuple[str, str]]:
        return self._call("entries")

    def clear(self) -> None:
        self._fallback.clear()
        self._call("clear")

    def __len__(self) -> int:
        return self._call("__len__")


def redis_from_url(url: str, timeout: float) -> Any:
    import redis

    return redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=timeout, socket_timeout=timeout)


def open_backend(
    settings: Settings,
    env: EnvSettings,
    registry: HealthRegistry,
    *,
    mode: str,
    client_factory: Callable[[str, float], Any] | None = None,
) -> CacheBackend:
    """Redis when REDIS_URL is set and answers a ping, otherwise memory (FR-F6, FR-F7). A failure is a degraded mode
    on the health registry (the exception type only: a message could echo the URL) and never an error. `mode` keeps
    the mock and live caches apart in a shared Redis."""
    cfg = settings.cache
    memory = MemoryBackend(cfg.max_entries)
    if not env.redis_url:
        return memory  # the registry already lists the missing URL as a degraded mode
    started = time.perf_counter()
    try:
        client = (client_factory or redis_from_url)(env.redis_url, cfg.redis_timeout_seconds)
        client.ping()
    except Exception as error:
        registry.mark_degraded(Component.REDIS, type(error).__name__, time.perf_counter() - started)
        return memory
    registry.mark_ok(Component.REDIS, time.perf_counter() - started)
    return FallbackBackend(
        RedisBackend(client, namespace=f"smartdesk:answers:{mode}", max_entries=cfg.max_entries), memory,
        retry_after=cfg.redis_retry_seconds,
        on_down=lambda error: registry.mark_degraded(Component.REDIS, type(error).__name__),
        on_up=lambda: registry.mark_ok(Component.REDIS))


def clear_answers(backend: CacheBackend, log: CacheLog | None = None) -> None:
    """The KB changed (FR-H6): drop every cached answer."""
    backend.clear()
    if log is not None:
        log.record("clear")


class CachedAnswer(BaseModel):
    kb_version: str
    domain: Domain | None = None
    vector: list[float] | None = None  # the question's embedding; None: found by exact match only
    compute_ms: float  # what the search and answer took when this entry was made
    part: PartResult


@dataclass(frozen=True)
class CacheLookup:
    kind: Kind
    key: str = ""
    kb_version: str = ""
    part: PartResult | None = None
    saved_ms: float | None = None
    similarity: float | None = None  # semantic hits: the cosine to the cached question
    vector: list[float] | None = None  # the question's embedding, kept so `store` makes no second call
    domain: Domain | None = None


def normalize(question: str) -> str:
    return " ".join(question.split()).casefold()


def cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm and len(a) == len(b) else 0.0


def is_cacheable_question(question: str) -> bool:
    """False for a question that still holds an email or a redaction: that is about a person, not the KB."""
    return EMAIL.search(question) is None and not any(mark in question for mark in PLACEHOLDERS.values())


def is_cacheable_part(part: PartResult) -> bool:
    return part.answerable and not part.llm_insufficient and bool(part.citations)


class AnswerCache:
    def __init__(
        self,
        backend: CacheBackend,
        *,
        kb_version: Callable[[], str],
        embed: Callable[[str], list[float]] | None = None,
        cutoff: float = 1.0,
        log: CacheLog | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.backend = backend
        self._kb_version = kb_version
        self._embed, self._cutoff = embed, cutoff
        self._log = log
        self._clock = clock

    def __len__(self) -> int:
        return len(self.backend)

    def _key(self, question: str, domain: Domain | None) -> str:
        return salted_digest(f"{domain or '-'}\n{normalize(question)}")

    def lookup(self, question: str, domain: Domain | None) -> CacheLookup:
        if not is_cacheable_question(question):
            return CacheLookup("bypass")
        started = self._clock()
        key = self._key(question, domain)
        try:
            version = self._kb_version()
            entry = self._read(key, version)
        except Exception:  # a broken cache is a miss, never a failed turn (FR-F7)
            logger.warning("answer cache lookup failed; treated as a miss", exc_info=True)
            return CacheLookup("miss", key=key, domain=domain)
        if entry is not None:
            saved = max(entry.compute_ms - (self._clock() - started) * 1000, 0.0)
            self._record("hit_exact", saved_ms=saved)
            return CacheLookup("exact", key=key, kb_version=version, part=entry.part, saved_ms=saved)
        vector, nearest, similarity = self._nearest(question, domain, version)
        if nearest is not None and similarity >= self._cutoff:
            saved = max(nearest.compute_ms - (self._clock() - started) * 1000, 0.0)
            self._record("hit_semantic", saved_ms=saved, similarity=similarity)
            return CacheLookup("semantic", key=key, kb_version=version, part=nearest.part, saved_ms=saved,
                               similarity=similarity)
        self._record("miss")
        return CacheLookup("miss", key=key, kb_version=version, vector=vector, domain=domain)

    def _nearest(
        self, question: str, domain: Domain | None, version: str
    ) -> tuple[list[float] | None, CachedAnswer | None, float]:
        """The question's vector (one embedding call) and the stored answer nearest to it. Any failure leaves the
        lookup a plain miss: the answer is then kept for exact matches only."""
        if self._embed is None:
            return None, None, 0.0
        try:
            vector = [round(x, 6) for x in self._embed(question)]
        except Exception:
            logger.warning("answer cache embedding failed; semantic match skipped", exc_info=True)
            return None, None, 0.0
        best, best_similarity = None, -1.0
        try:
            for _, raw in self.backend.entries():
                try:
                    entry = CachedAnswer.model_validate_json(raw)
                except ValueError:
                    continue  # an entry another version of the app wrote
                if entry.kb_version != version or entry.domain != domain or entry.vector is None:
                    continue
                similarity = cosine(vector, entry.vector)
                if similarity > best_similarity:
                    best, best_similarity = entry, similarity
        except Exception:
            logger.warning("answer cache scan failed; semantic match skipped", exc_info=True)
        return vector, best, best_similarity

    def _read(self, key: str, version: str) -> CachedAnswer | None:
        raw = self.backend.get(key)
        if raw is None:
            return None
        entry = CachedAnswer.model_validate_json(raw)
        return entry if entry.kb_version == version else None

    def store(self, lookup: CacheLookup, part: PartResult, compute_ms: float) -> None:
        """Keep `part` under the key of the lookup that missed, if it is a shown, grounded answer."""
        if lookup.kind != "miss" or not lookup.key or not lookup.kb_version or not is_cacheable_part(part):
            return
        plain = part.model_copy(update={"retried": False, "cache": None, "cache_similarity": None,
                                        "cache_saved_ms": None})
        entry = CachedAnswer(kb_version=lookup.kb_version, domain=lookup.domain, vector=lookup.vector,
                             compute_ms=compute_ms, part=plain)
        try:
            self.backend.put(lookup.key, entry.model_dump_json())
        except Exception:
            logger.warning("answer cache store failed", exc_info=True)

    def clear(self) -> None:
        """The KB changed (FR-H6): drop every answer."""
        clear_answers(self.backend, self._log)

    def _record(self, event, **fields) -> None:
        if self._log is not None:
            self._log.record(event, **fields)
