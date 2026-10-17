"""Per-tool-call metrics (FR-G9).

A record holds the tool, the calling agent, the outcome, attempts used and latencies, and never
the arguments, so no email or other PII reaches the local log (FR-G11). The per-tool summary
(FR-G10) is built from these records by `summarize_tools`.
"""
import json
import math
import threading
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


AttemptOutcome = Literal["success", "transient_failure", "validation_error", "error"]


class ToolCallRecord(BaseModel):
    tool: str
    agent: str
    outcome: str  # success, validation_error, unavailable, error
    error_code: str | None = None
    attempts: int
    attempt_latencies_ms: list[float]
    attempt_outcomes: list[AttemptOutcome]  # one per attempt, same order as the latencies
    total_latency_ms: float


class MetricsRecorder:
    """Keeps records in memory and, if given a path, appends each as one JSON line."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self.records: list[ToolCallRecord] = []

    def record(self, rec: ToolCallRecord) -> None:
        with self._lock:
            self.records.append(rec)
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a") as f:
                    f.write(json.dumps(rec.model_dump()) + "\n")

    def summary(self) -> "list[ToolSummary]":
        with self._lock:
            return summarize_tools(self.records)


def load_records(path: Path | str) -> list[ToolCallRecord]:
    """The records in a local log file; a missing file is an empty log."""
    path = Path(path)
    if not path.exists():
        return []
    return [ToolCallRecord.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


class ToolSummary(BaseModel):
    """One row of the per-tool summary (FR-G10)."""

    tool: str
    calls: int
    success_rate: float
    failures: dict[str, int]  # failed calls by outcome: validation_error, unavailable, error
    retry_rate: float  # share of calls that needed more than one attempt
    p50_ms: float  # total latency per call, nearest-rank percentiles
    p95_ms: float


def percentile(values: Iterable[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value with at least a share `q` of the values at or below it."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[max(math.ceil(q * len(ordered)) - 1, 0)]


def summarize_tools(records: Iterable[ToolCallRecord]) -> list[ToolSummary]:
    grouped: dict[str, list[ToolCallRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.tool, []).append(rec)
    rows = []
    for tool, recs in sorted(grouped.items()):
        failed = Counter(r.outcome for r in recs if r.outcome != "success")
        rows.append(ToolSummary(
            tool=tool, calls=len(recs), success_rate=sum(r.outcome == "success" for r in recs) / len(recs),
            failures=dict(failed), retry_rate=sum(r.attempts > 1 for r in recs) / len(recs),
            p50_ms=percentile((r.total_latency_ms for r in recs), 0.50),
            p95_ms=percentile((r.total_latency_ms for r in recs), 0.95),
        ))
    return rows
