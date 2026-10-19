"""The answer cache's event log: one line per KB lookup (hit or miss) and per clear, for the Monitoring tab (FR-H7).

A line holds the time, the event and numbers only (the saved latency and the similarity), never a question, an answer or
a key, so no PII reaches the log (FR-G11). A missing file is an empty log.
"""
import json
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError

CACHE_LOG_FILE = "cache.jsonl"  # in the data folder; the Monitoring tab reads it

CacheEvent = Literal["hit_exact", "hit_semantic", "miss", "clear"]


class CacheRecord(BaseModel):
    ts: str
    event: CacheEvent
    saved_ms: float | None = None  # hits: the search and answer time the cached answer spared
    similarity: float | None = None  # semantic hits


class CacheLog:
    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self.records: list[CacheRecord] = []

    def record(self, event: CacheEvent, *, saved_ms: float | None = None, similarity: float | None = None) -> None:
        rec = CacheRecord(ts=datetime.now(timezone.utc).isoformat(timespec="seconds"), event=event,
                          saved_ms=None if saved_ms is None else round(saved_ms, 1),
                          similarity=None if similarity is None else round(similarity, 4))
        with self._lock:
            self.records.append(rec)
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a") as handle:
                    handle.write(rec.model_dump_json() + "\n")


def load_cache_records(path: Path | str) -> list[CacheRecord]:
    path = Path(path)
    if not path.is_file():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(CacheRecord.model_validate(json.loads(line)))
        except (ValueError, ValidationError):
            continue  # a half-written last line
    return records


class CacheStats(BaseModel):
    lookups: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    hit_rate: float | None = None  # None before the first lookup
    saved_ms_total: float = 0.0
    saved_ms_per_hit: float | None = None
    clears: int = 0


def summarize_cache(records: Iterable[CacheRecord]) -> CacheStats:
    stats = CacheStats()
    for rec in records:
        if rec.event == "clear":
            stats.clears += 1
            continue
        stats.lookups += 1
        if rec.event == "miss":
            stats.misses += 1
            continue
        stats.exact_hits += rec.event == "hit_exact"
        stats.semantic_hits += rec.event == "hit_semantic"
        stats.saved_ms_total += rec.saved_ms or 0.0
    hits = stats.exact_hits + stats.semantic_hits
    if stats.lookups:
        stats.hit_rate = hits / stats.lookups
    if hits:
        stats.saved_ms_per_hit = stats.saved_ms_total / hits
    stats.saved_ms_total = round(stats.saved_ms_total, 1)
    return stats
