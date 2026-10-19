"""The guard every live eval run goes through (CLAUDE.md working rules, FR-H7, FR-H8, FR-H12).

`LiveRun` opens a spend-ledger run (source `eval`), stops the run once the ledger shows more than the ceiling (twice the
estimate), and aborts a run that makes no progress (a thread stuck in a network call cannot be interrupted any other way).
`Checkpoint` stores each finished item, so a run that was stopped carries on where it left off instead of paying twice.
`run_items` is the loop that uses both. The ceiling reads the ledger, not a tracker of the run's own, so it counts every
paid call, including the ones the models make inside a worker thread.
"""
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.judge_run import EXIT_STALLED, STALL_SECONDS, Watchdog
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.ops.spend_ledger import current_run, end_run, start_run

CEILING_MULTIPLIER = 2.0

__all__ = ["CEILING_MULTIPLIER", "EXIT_STALLED", "Checkpoint", "Estimate", "LiveRun", "RunCost", "ceiling_for", "describe", "run_items",
           "signature"]


def ceiling_for(estimate_usd: float) -> float:
    """The spend ceiling of a run: twice its estimate."""
    return estimate_usd * CEILING_MULTIPLIER


@dataclass(frozen=True)
class Estimate:
    """What a live run is expected to cost, shown before it runs. The ceiling is twice the estimate."""

    name: str
    usd: float
    detail: str = ""

    @property
    def ceiling_usd(self) -> float:
        return ceiling_for(self.usd)


def describe(estimate: Estimate) -> list[str]:
    """The lines printed before a live run."""
    return [f"Estimate for {estimate.name}: about ${estimate.usd:.3f}" + (f" ({estimate.detail})" if estimate.detail else "") + ".",
            f"The run stops once the ledger shows more than ${estimate.ceiling_usd:.2f} spent (the ceiling, 2x the estimate) or when it makes "
            "no progress for five minutes; it is recorded in data/spend_ledger.jsonl under source `eval` and resumes from its "
            "checkpoint if it is run again."]


@dataclass(frozen=True)
class RunCost:
    usd: float
    calls: int
    input_tokens: int
    output_tokens: int


class LiveRun:
    """A paid run: `with LiveRun(settings, argv, ceiling_usd=...) as run:` then call `run.beat()` after each item and
    `run.stop_reason()` before the next one. The ledger run ends with the status "ok", "ceiling", "stalled" or "crashed"."""

    def __init__(self, settings: Settings, argv: Sequence[str], *, ceiling_usd: float, stall_seconds: float = STALL_SECONDS,
                 poll_seconds: float = 1.0, ledger_path: Path | None = None, on_stall: Callable[[], None] | None = None) -> None:
        self.settings, self.argv, self.ceiling_usd, self.ledger_path = settings, list(argv), ceiling_usd, ledger_path
        self.status: str | None = None
        self._on_stall = on_stall or self._abort_process
        self.watchdog = Watchdog(stall_seconds, self._stalled, poll_seconds=poll_seconds)
        self._ledger = None

    def __enter__(self) -> "LiveRun":
        self._ledger = start_run(self.settings, self.argv, "eval", self.ledger_path)
        self._base = (self._ledger.usd, self._ledger.calls, self._ledger.input_tokens, self._ledger.output_tokens)
        self._ledger.on_spend = self.watchdog.beat  # every paid call is progress, not only a finished item
        self.watchdog.start()
        return self

    def __exit__(self, exc_type: object, *exc: object) -> None:
        self.watchdog.stop()
        end_run(self.status or ("crashed" if exc_type is not None else "ok"))

    def beat(self) -> None:
        self.watchdog.beat()

    def cost(self) -> RunCost:
        """What this run has spent according to the ledger."""
        run = self._ledger
        return RunCost(usd=run.usd - self._base[0], calls=run.calls - self._base[1],
                       input_tokens=run.input_tokens - self._base[2], output_tokens=run.output_tokens - self._base[3])

    def stop_reason(self) -> str | None:
        spent = self.cost().usd
        if spent > self.ceiling_usd:
            self.status = "ceiling"
            return f"ledger spend ${spent:.4f} is over the ${self.ceiling_usd:.2f} ceiling"
        return None

    def _stalled(self) -> None:
        self.status = "stalled"
        self._on_stall()

    @staticmethod
    def _abort_process() -> None:
        print("error: the run made no progress and was aborted", file=sys.stderr, flush=True)
        end_run("stalled")
        os._exit(EXIT_STALLED)


def signature(parts: Sequence[Any]) -> str:
    """Identifies the inputs of a run (its questions and settings), so a checkpoint is only reused for the same ones."""
    return hashlib.sha1(json.dumps(list(parts), sort_keys=True, default=str).encode()).hexdigest()[:16]


class Checkpoint:
    """Finished items of one run, saved after each one. With no path it lives in memory only."""

    def __init__(self, path: Path | None, sig: str) -> None:
        self.path, self.sig = path, sig
        self.items: dict[str, Any] = {}
        self.resumed = 0
        if path is not None and path.exists():
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                data = {}
            if data.get("signature") == sig:
                self.items = dict(data.get("items", {}))
        self.resumed = len(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def done(self, item_id: str) -> bool:
        return item_id in self.items

    def get(self, item_id: str) -> Any:
        return self.items[item_id]

    def put(self, item_id: str, payload: Any) -> None:
        self.items[item_id] = payload
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=self.path.parent, suffix=".tmp", delete=False) as handle:
            json.dump({"signature": self.sig, "items": self.items}, handle)
        os.replace(handle.name, self.path)


def run_items(run: LiveRun, checkpoint: Checkpoint, ids: Sequence[str], work: Callable[[str], Any],
              *, stop: Callable[[dict[str, Any]], str | None] | None = None) -> tuple[dict[str, Any], str | None]:
    """Do `work(id)` for every id not yet in the checkpoint, saving each result. Before each item it asks whether to stop: the
    ceiling, or the caller's own rule (`stop` sees the results so far). Returns every result so far and the reason it stopped."""
    results = {i: checkpoint.get(i) for i in ids if checkpoint.done(i)}
    for item_id in ids:
        if item_id in results:
            continue
        reason = run.stop_reason() or (stop(results) if stop else None)
        if reason:
            return results, reason
        results[item_id] = work(item_id)
        checkpoint.put(item_id, results[item_id])
        run.beat()
    return results, run.stop_reason()
