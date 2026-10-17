"""Rate limiting (FR-H13): a rolling window (`rate_limits.window_seconds`, one hour) with three
independent counters: questions, ticket creations, ticket lookups.

Counters live in SQLite, so they survive a restart. Questions and creations are keyed by session ID;
lookups are keyed by session and by email, and both must be within the limit. Emails are stored only as a
masked hash. `rate_limits.bypass` turns everything off for evals and tests.
"""
import math
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from smartdesk_ai.config.settings import RateLimits
from smartdesk_ai.ops.runtime import masked_user_id

Counter = Literal["questions", "ticket_creations", "ticket_lookups"]


def usage_limit_message(wait_seconds: float) -> str:
    minutes = max(math.ceil(wait_seconds / 60), 1)
    return f"Usage limit reached. Please try again in {minutes} {'minute' if minutes == 1 else 'minutes'}."


class LimitDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    wait_seconds: int = 0
    message: str | None = None


class RateLimiter:
    def __init__(self, cfg: RateLimits, path: Path | str, *, clock: Callable[[], float] = time.time) -> None:
        self._cfg = cfg
        self._clock = clock
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.execute("CREATE TABLE IF NOT EXISTS rate_events (counter TEXT, key TEXT, ts REAL)")
            self._db.execute("CREATE INDEX IF NOT EXISTS rate_events_lookup ON rate_events (counter, key, ts)")

    def _keys(self, counter: Counter, session_id: str, email: str | None) -> list[str]:
        keys = [f"session:{session_id}"]
        if counter == "ticket_lookups" and email:
            keys.append(f"user:{masked_user_id(email)}")
        return keys

    def _limit(self, counter: Counter) -> int:
        return getattr(self._cfg, counter)

    def count(self, counter: Counter, session_id: str, email: str | None = None) -> int:
        """The most-used key's count inside the window."""
        floor = self._clock() - self._cfg.window_seconds
        with self._lock:
            return max((self._db.execute(
                "SELECT COUNT(*) FROM rate_events WHERE counter = ? AND key = ? AND ts > ?",
                (counter, key, floor)).fetchone()[0] for key in self._keys(counter, session_id, email)), default=0)

    def check(self, counter: Counter, session_id: str, email: str | None = None) -> LimitDecision:
        """Whether one more use is allowed now. Does not record it."""
        if self._cfg.bypass:
            return LimitDecision(allowed=True)
        now, window = self._clock(), self._cfg.window_seconds
        wait = 0.0
        with self._lock, self._db:
            self._db.execute("DELETE FROM rate_events WHERE ts <= ?", (now - window,))
            for key in self._keys(counter, session_id, email):
                oldest = self._db.execute(
                    "SELECT ts FROM rate_events WHERE counter = ? AND key = ? ORDER BY ts LIMIT ? ",
                    (counter, key, self._limit(counter))).fetchall()
                if len(oldest) >= self._limit(counter):  # full: free again when the oldest use leaves the window
                    wait = max(wait, oldest[0][0] + window - now)
        if wait > 0:
            return LimitDecision(allowed=False, wait_seconds=math.ceil(wait), message=usage_limit_message(wait))
        return LimitDecision(allowed=True)

    def record(self, counter: Counter, session_id: str, email: str | None = None) -> None:
        if self._cfg.bypass:
            return
        now = self._clock()
        with self._lock, self._db:
            self._db.executemany("INSERT INTO rate_events (counter, key, ts) VALUES (?, ?, ?)",
                                 [(counter, key, now) for key in self._keys(counter, session_id, email)])

    def hit(self, counter: Counter, session_id: str, email: str | None = None) -> LimitDecision:
        """Check and, when allowed, record one use."""
        decision = self.check(counter, session_id, email)
        if decision.allowed:
            self.record(counter, session_id, email)
        return decision
