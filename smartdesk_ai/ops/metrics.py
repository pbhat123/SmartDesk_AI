"""Per-tool-call metrics (FR-G9).

A record holds the tool, the calling agent, the outcome, attempts used and latencies, and never
the arguments, so no email or other PII reaches the local log (FR-G11). The per-tool summary
(FR-G10) is built from these records later.
"""
import json
import threading
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
