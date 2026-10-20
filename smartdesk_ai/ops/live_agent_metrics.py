"""Agent metrics from the app's own trace and cost files (FR-H11, FR-G10): what the real sessions did, not a scripted run.

A task is one session (all its turns that started inside the window), a turn is one request. Read-only; the numbers
use the same definitions as `evals/agent_metrics.py`:
- a turn fails when its reply says to try again later, it has no reply at all, or a tool call in it ended in an error;
  a blocked reply (a guard decision) is not a failure;
- completion rate: tasks with no failed turn; error rate: failed turns over all turns;
- steps to completion: graph steps (the leading "start" is not one) of the completed tasks, averaged;
- latency: reply time minus request time, p50 and p95 per turn; cost per task: the cost tracker's USD per session.
No text of any message is read into the result.
"""
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from smartdesk_ai.ops.metrics import percentile

FAILURE_PHRASES = ("try again later", "try again in a few minutes", "temporarily unavailable")


class LiveAgentMetrics(BaseModel):
    n_tasks: int
    n_turns: int
    completion_rate: float
    error_rate: float
    steps_to_completion: float | None
    turn_p50_ms: float | None
    turn_p95_ms: float | None
    usd_per_task: float


def load_events(path: Path) -> list[dict[str, Any]]:
    """The JSON lines of a log; a missing file is empty and a damaged line is skipped."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _when(ts: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def failed_reply(text: str) -> bool:
    return any(phrase in text.lower() for phrase in FAILURE_PHRASES)


class _Turn:
    def __init__(self, session: str, started: datetime) -> None:
        self.session, self.started = session, started
        self.replied: datetime | None = None
        self.failed = False
        self.steps = 0


def compute_live_metrics(
    trace: Iterable[dict[str, Any]], costs: Iterable[dict[str, Any]], *, since: datetime | None
) -> LiveAgentMetrics | None:
    """Metrics over the turns that started at or after `since` (None: all of them); None when there are none."""
    turns: dict[str, _Turn] = {}
    events = list(trace)
    for e in events:
        if e.get("kind") == "request" and (when := _when(e.get("ts"))) and e.get("request_id") and e.get("session_id"):
            if since is None or when >= since:
                turns[e["request_id"]] = _Turn(e["session_id"], when)
    if not turns:
        return None
    replied: set[str] = set()
    for e in events:
        turn = turns.get(e.get("request_id"))
        if turn is None:
            continue
        kind = e.get("kind")
        if kind == "reply":
            replied.add(e["request_id"])
            turn.replied = _when(e.get("ts"))
            turn.failed |= failed_reply(str(e.get("text") or ""))
        elif kind == "agent_step" and e.get("step") != "start":
            turn.steps += 1
        elif kind == "tool_call" and e.get("outcome") == "error":
            turn.failed = True
    for rid, turn in turns.items():
        turn.failed |= rid not in replied
    tasks: dict[str, list[_Turn]] = defaultdict(list)
    for turn in turns.values():
        tasks[turn.session].append(turn)
    done = [ts for ts in tasks.values() if not any(t.failed for t in ts)]
    latencies = [(t.replied - t.started).total_seconds() * 1000 for t in turns.values() if t.replied]
    spend = sum(float(c.get("usd") or 0.0) for c in costs if c.get("request_id") in turns)
    return LiveAgentMetrics(
        n_tasks=len(tasks), n_turns=len(turns), completion_rate=len(done) / len(tasks),
        error_rate=sum(t.failed for t in turns.values()) / len(turns),
        steps_to_completion=(sum(t.steps for ts in done for t in ts) / len(done)) if done else None,
        turn_p50_ms=percentile(latencies, 0.50) if latencies else None,
        turn_p95_ms=percentile(latencies, 0.95) if latencies else None,
        usd_per_task=spend / len(tasks))
