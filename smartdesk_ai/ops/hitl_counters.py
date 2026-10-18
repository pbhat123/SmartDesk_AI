"""Human-in-the-loop counters (FR-H4): what employees decide at the ticket approval gate (FR-G12).

Approvals (yes), rejections (no) and edits are counted when the gate reads a final decision. An unclear reply is
not a decision, so it is not counted, and neither is a gate that gave up after two of them. A record holds the
decision and a timestamp only, never the reply or any ticket text (FR-G11). The file is shared by every process
(the app, the CLI), and readers never create it.
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

Decision = Literal["yes", "no", "edit"]
_FIELD: dict[str, str] = {"yes": "approvals", "no": "rejections", "edit": "edits"}


class HitlCounts(BaseModel):
    approvals: int = 0
    rejections: int = 0
    edits: int = 0

    @property
    def total(self) -> int:
        return self.approvals + self.rejections + self.edits


def _count(lines: list[str]) -> HitlCounts:
    counts = HitlCounts()
    for line in lines:
        try:
            decision = json.loads(line).get("decision")
        except (ValueError, AttributeError):
            continue  # a half-written last line
        if decision in _FIELD:
            setattr(counts, _FIELD[decision], getattr(counts, _FIELD[decision]) + 1)
    return counts


def load_counts(path: Path | str) -> HitlCounts:
    """The counts in a log file; a missing file is zero, and nothing is created."""
    path = Path(path)
    return _count(path.read_text().splitlines()) if path.is_file() else HitlCounts()


class HitlCounters:
    """Records gate decisions to `path` (one JSON line each), or in memory when `path` is None."""

    def __init__(self, path: Path | str | None) -> None:
        self._path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._memory: list[str] = []

    def record(self, decision: str) -> None:
        if decision not in _FIELD:
            raise ValueError(f"not a gate decision: {decision!r}")
        line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "decision": decision})
        with self._lock:
            if self._path is None:
                self._memory.append(line)
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a") as handle:
                handle.write(line + "\n")

    def counts(self) -> HitlCounts:
        with self._lock:
            return _count(self._memory) if self._path is None else load_counts(self._path)
