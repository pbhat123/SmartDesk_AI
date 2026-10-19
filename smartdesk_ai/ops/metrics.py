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
from datetime import datetime, timezone
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
    ts: str | None = None  # UTC time of the call; records written before this field existed have none


class MetricsRecorder:
    """Keeps records in memory and, if given a path, appends each as one JSON line."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self.records: list[ToolCallRecord] = []

    def record(self, rec: ToolCallRecord) -> None:
        if rec.ts is None:
            rec = rec.model_copy(update={"ts": datetime.now(timezone.utc).isoformat(timespec="seconds")})
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


def date_from_trace(records: list[ToolCallRecord], trace_path: Path | str) -> list[ToolCallRecord]:
    """Give records written before `ts` existed the time of their `tool_call` event in the trace log.

    The two logs get one line per tool call, in the same order, so the undated records (always the oldest ones) pair
    with the first trace events. The pairing is used only if every pair agrees on the tool and the outcome; otherwise
    the records stay undated rather than get a guessed time.
    """
    undated = [i for i, r in enumerate(records) if r.ts is None]
    path = Path(trace_path)
    if not undated or not path.is_file():
        return records
    events: list[dict] = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("kind") == "tool_call" and event.get("ts"):
            events.append(event)
        if len(events) == len(undated):
            break
    if len(events) < len(undated) or any(
            records[i].tool != e.get("tool") or records[i].outcome != e.get("outcome") for i, e in zip(undated, events)):
        return records
    dated = list(records)
    for i, event in zip(undated, events):
        dated[i] = records[i].model_copy(update={"ts": str(event["ts"])})
    return dated
