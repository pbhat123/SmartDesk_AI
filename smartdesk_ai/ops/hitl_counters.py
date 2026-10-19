"""Human-in-the-loop counters (FR-H4): what employees decide at the ticket approval gate (FR-G12).

Approvals (yes), rejections (no) and edits are counted when the gate reads a final decision. An unclear reply is
not a decision, so it is not counted, and neither is a gate that gave up after two of them. After an approval the
ticket step records what happened (`created` or `failed`), and the counters join the two, so an approval is either
"ticket created", "creation failed" or, when no outcome was ever recorded, "unknown".

A line holds a timestamp, the decision (or outcome) and the conversation's random thread ID, never the reply or any
ticket text (FR-G11). The log is append-only and shared by every process (the app, the CLI); readers never create or
change it. Lines written before outcomes existed carry no session: their outcome is read from the trace log, which
records whether `create_ticket` ran.
"""
import json
import threading
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

Decision = Literal["yes", "no", "edit"]
Outcome = Literal["created", "failed"]
_FIELD: dict[str, str] = {"yes": "approvals", "no": "rejections", "edit": "edits"}
TRACE_JOIN_SECONDS = 15  # a ticket is created within moments of the approval
TRACE_COVER_SECONDS = 120  # the trace must have seen something this close to the approval to say "no ticket was made"


class HitlCounts(BaseModel):
    approvals: int = 0  # every "yes" at the gate
    approved_created: int = 0  # of those, the ticket was created
    approved_failed: int = 0  # of those, creating it failed
    approved_unknown: int = 0  # of those, no outcome was recorded and the trace cannot tell
    rejections: int = 0
    edits: int = 0

    @property
    def total(self) -> int:
        return self.approvals + self.rejections + self.edits


def _parse(ts: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def read_events(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue  # a half-written last line
        if isinstance(event, dict):
            events.append(event)
    return events


def _trace_view(trace_path: Path | str | None) -> tuple[list[tuple[datetime, str]], list[datetime]]:
    """(create_ticket outcomes with their time, the time of every trace event) from the trace log."""
    calls: list[tuple[datetime, str]] = []
    seen: list[datetime] = []
    if trace_path is None or not Path(trace_path).is_file():
        return calls, seen
    for line in Path(trace_path).read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        when = _parse(event.get("ts", "")) if isinstance(event, dict) else None
        if when is None:
            continue
        seen.append(when)
        if event.get("kind") == "tool_call" and event.get("tool") == "create_ticket":
            calls.append((when, str(event.get("outcome"))))
    return calls, seen


def _legacy_outcome(when: datetime, calls: list[tuple[datetime, str]], seen: list[datetime]) -> str:
    near = [outcome for at, outcome in calls if timedelta(0) <= at - when <= timedelta(seconds=TRACE_JOIN_SECONDS)]
    if near:
        return "created" if "success" in near else "failed"
    covered = any(abs(at - when) <= timedelta(seconds=TRACE_COVER_SECONDS) for at in seen)
    return "failed" if covered else "unknown"  # the trace was running and saw no ticket being made


def summarize(
    events: Sequence[dict], *, since: datetime | None = None, sessions: Sequence[str] | None = None,
    trace_path: Path | str | None = None,
) -> HitlCounts:
    """Counts over the events inside the window. `since` keeps events at or after that time; `sessions` keeps only
    those conversations' events (old lines have no session, so they never belong to "this session")."""
    outcomes: dict[str, list[str]] = defaultdict(list)
    for event in events:
        if event.get("event") == "outcome" and event.get("session") and event.get("outcome") in ("created", "failed"):
            outcomes[event["session"]].append(event["outcome"])
    seen_approvals: dict[str, int] = defaultdict(int)
    calls, trace_times = _trace_view(trace_path) if trace_path is not None else ([], [])
    counts = HitlCounts()
    for event in events:
        decision = event.get("decision")
        if decision not in _FIELD:
            continue
        session = event.get("session")
        when = _parse(event.get("ts", ""))
        outcome = None
        if decision == "yes":
            if session:  # the n-th approval of a conversation is followed by its n-th outcome
                index = seen_approvals[session]
                seen_approvals[session] += 1
                outcome = outcomes[session][index] if index < len(outcomes[session]) else "unknown"
            elif when is not None:
                outcome = _legacy_outcome(when, calls, trace_times) if trace_path is not None else "unknown"
            else:
                outcome = "unknown"
        if since is not None and (when is None or when < since):
            continue
        if sessions is not None and session not in sessions:
            continue
        setattr(counts, _FIELD[decision], getattr(counts, _FIELD[decision]) + 1)
        if outcome:
            attribute = {"created": "approved_created", "failed": "approved_failed", "unknown": "approved_unknown"}[outcome]
            setattr(counts, attribute, getattr(counts, attribute) + 1)
    return counts


def load_counts(
    path: Path | str, *, since: datetime | None = None, sessions: Sequence[str] | None = None,
    trace_path: Path | str | None = None,
) -> HitlCounts:
    """The counts in a log file; a missing file is zero, and nothing is created."""
    return summarize(read_events(path), since=since, sessions=sessions, trace_path=trace_path)


class HitlCounters:
    """Records gate decisions to `path` (one JSON line each), or in memory when `path` is None."""

    def __init__(self, path: Path | str | None) -> None:
        self._path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._memory: list[dict] = []

    def _write(self, fields: dict) -> None:
        event = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
        with self._lock:
            if self._path is None:
                self._memory.append(event)
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a") as handle:
                handle.write(json.dumps(event) + "\n")

    def record(self, decision: str, session: str | None = None) -> None:
        if decision not in _FIELD:
            raise ValueError(f"not a gate decision: {decision!r}")
        self._write({"decision": decision} | ({"session": session} if session else {}))

    def record_outcome(self, outcome: str, session: str | None = None) -> None:
        """What happened to the ticket after an approval: `created` or `failed`."""
        if outcome not in ("created", "failed"):
            raise ValueError(f"not a ticket outcome: {outcome!r}")
        self._write({"event": "outcome", "outcome": outcome} | ({"session": session} if session else {}))

    def counts(self) -> HitlCounts:
        with self._lock:
            events = list(self._memory) if self._path is None else None
        return summarize(events) if events is not None else load_counts(self._path)
