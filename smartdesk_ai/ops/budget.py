"""Daily budget cutoffs (FR-H8).

Spend is added per day (the day starts at midnight in `budget.timezone`) and kept in SQLite, so a restart
does not reset it. Cutoff 1 (`budget.cutoff_1_usd`): every new LLM call uses Light-Model. Cutoff 2
(`budget.cutoff_2_usd`): no LLM calls, and the chat shows a graceful message. Each crossing is logged once
per day. With `budget.eval_bypass` on, spend under the task "eval" is not counted and eval runs are never
cut off; the real service keeps its limits.
"""
import logging
import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from smartdesk_ai.config.settings import Settings

logger = logging.getLogger(__name__)

EVAL_TASK = "eval"


class SpendStore:
    """USD spent per day. One row per day, so the 'counter' resets by simply reading a new day's row."""

    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._db:
            self._db.execute("CREATE TABLE IF NOT EXISTS spend (day TEXT PRIMARY KEY, usd REAL NOT NULL)")

    def add(self, day: str, usd: float) -> float:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO spend (day, usd) VALUES (?, ?) ON CONFLICT(day) DO UPDATE SET usd = usd + excluded.usd",
                (day, usd))
            return self._db.execute("SELECT usd FROM spend WHERE day = ?", (day,)).fetchone()[0]

    def get(self, day: str) -> float:
        with self._lock:
            row = self._db.execute("SELECT usd FROM spend WHERE day = ?", (day,)).fetchone()
        return row[0] if row else 0.0


class BudgetController:
    def __init__(
        self, settings: Settings, store: SpendStore, *, clock: Callable[[], datetime] | None = None
    ) -> None:
        self._cfg = settings.budget
        self._store = store
        tz = ZoneInfo(self._cfg.timezone)
        self._clock = clock or (lambda: datetime.now(tz))
        self._tz = tz
        self._logged: set[tuple[str, int]] = set()

    def _today(self) -> str:
        return self._clock().astimezone(self._tz).date().isoformat()

    def spend_today(self) -> float:
        return self._store.get(self._today())

    def _level_for(self, spend: float) -> int:
        if spend >= self._cfg.cutoff_2_usd:
            return 2
        return 1 if spend >= self._cfg.cutoff_1_usd else 0

    def level(self, task: str | None = None) -> int:
        """0 normal, 1 all-Light, 2 stopped. Eval runs get 0 when the bypass flag is on."""
        if self._cfg.eval_bypass and task == EVAL_TASK:
            return 0
        return self._level_for(self.spend_today())

    def add(self, usd: float, task: str | None = None) -> None:
        if self._cfg.eval_bypass and task == EVAL_TASK:
            return  # tracked in the cost log under the task "eval", not against the daily limit
        day = self._today()
        total = self._store.add(day, usd)
        level = self._level_for(total)
        for crossed in range(1, level + 1):
            if (day, crossed) not in self._logged:
                self._logged.add((day, crossed))
                limit = self._cfg.cutoff_1_usd if crossed == 1 else self._cfg.cutoff_2_usd
                logger.warning("budget cutoff %d reached: $%.2f spent today (limit $%.2f)", crossed, total, limit)
