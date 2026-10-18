"""Resumable, observable scoring for the paid Ragas judge (FR-H12, FR-H8).

A full judge pass is about 300 sequential model calls. `CheckpointedJudge` wraps any `Judge` so that a run
- appends a progress line (done/total, spend so far) to a log file after every scored item,
- saves its scores to a checkpoint every `batch` new items, so a rerun resumes instead of restarting,
- stops at once when the spend ceiling is passed, even if the exception was swallowed in a worker thread,
- is aborted by a watchdog when nothing has happened for `stall_seconds`.
"""
import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHECKPOINT_FILE = "judge_checkpoint.json"
PROGRESS_LOG = "judge_progress.log"
BATCH = 10
STALL_SECONDS = 300
EXIT_STALLED = 3

FATAL_MARKERS = ("insufficient_quota", "credit_balance", "invalid_api_key", "authenticationerror")


class SpendCeilingExceeded(RuntimeError):
    """The judge's spend passed the ceiling the run was given; the run stops."""


class FatalJudgeError(RuntimeError):
    """An error that retrying cannot fix (no credits, a bad key); the run stops on the first one."""


def is_fatal(error: BaseException) -> bool:
    seen: list[BaseException] = []
    while error is not None and error not in seen:
        seen.append(error)
        text = f"{type(error).__name__} {error}".lower()
        if any(marker in text for marker in FATAL_MARKERS):
            return True
        error = error.__cause__ or error.__context__
    return False


class Watchdog:
    """Calls `on_stall` once if `beat()` has not been called for `stall_seconds`."""

    def __init__(self, stall_seconds: float, on_stall: Callable[[], None], *, poll_seconds: float = 1.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._stall, self._on_stall, self._poll, self._clock = stall_seconds, on_stall, poll_seconds, clock
        self._last = clock()
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="judge-watchdog", daemon=True)
        self.stalled = False

    def start(self) -> None:
        self._last = self._clock()
        self._thread.start()

    def stop(self) -> None:
        self._done.set()

    def beat(self) -> None:
        self._last = self._clock()

    def _loop(self) -> None:
        while not self._done.wait(self._poll):
            if self._clock() - self._last > self._stall:
                self.stalled = True
                self._on_stall()
                return


def outputs_signature(outputs: Sequence[Any]) -> str:
    """Identifies the saved pipeline answers, so a checkpoint is only reused for the same answers."""
    blob = json.dumps([o.model_dump() for o in outputs], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


def planned_calls(outputs: Sequence[Any]) -> int:
    """Judge items for these outputs: four metrics per answered question, two per question with contexts."""
    answered = sum(1 for o in outputs if not o.insufficient and o.answer.strip())
    return 4 * answered + 2 * sum(1 for o in outputs if o.contexts)


def _key(method: str, args: Sequence[Any]) -> str:
    return f"{method}:{hashlib.sha1(json.dumps(list(args), sort_keys=True, default=str).encode()).hexdigest()[:16]}"


class CheckpointedJudge:
    """A `Judge` that adds progress, checkpoints, the ceiling check and a watchdog around another judge.

    `counter` is the `UsageCounter` the inner judge reports to (its `check()` raises past the ceiling). A failed
    sample (the inner judge returned None and counted an error) is not saved, so a rerun tries it again."""

    def __init__(self, inner: Any, *, total: int, signature: str, counter: Any, settings: Any,
                 checkpoint: Path | None = None, progress_log: Path | None = None, batch: int = BATCH,
                 stall_seconds: float = STALL_SECONDS, abort: Callable[[], None] | None = None,
                 poll_seconds: float = 1.0) -> None:
        self.inner, self.total, self.signature = inner, total, signature
        self.counter, self.settings = counter, settings
        self.checkpoint, self.progress_log, self.batch = checkpoint, progress_log, batch
        self.name = getattr(inner, "name", type(inner).__name__)
        self.done = self.new_since_flush = self.resumed = 0
        self._lock = threading.Lock()
        self.scores: dict[str, float | None] = {}
        self._load()
        self._abort = abort or self._abort_process
        self.watchdog = Watchdog(stall_seconds, self._stalled, poll_seconds=poll_seconds)
        if hasattr(counter, "on_call"):
            counter.on_call = self.watchdog.beat  # every model call is progress, not only a finished item

    @property
    def errors(self) -> dict[str, int]:
        return dict(getattr(self.inner, "errors", {}))

    # --- lifecycle ---------------------------------------------------------------------------------

    def __enter__(self) -> "CheckpointedJudge":
        self.watchdog.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.watchdog.stop()
        self.flush()

    # --- the Judge protocol ------------------------------------------------------------------------

    def faithfulness(self, question: str, answer: str, contexts: Sequence[str]) -> float | None:
        return self._item("faithfulness", question, answer, list(contexts))

    def answer_correctness(self, question: str, answer: str, reference: str) -> float | None:
        return self._item("answer_correctness", question, answer, reference)

    def factual_precision(self, answer: str, reference: str) -> float | None:
        return self._item("factual_precision", answer, reference)

    def answer_relevancy(self, question: str, answer: str) -> float | None:
        return self._item("answer_relevancy", question, answer)

    def context_precision(self, question: str, contexts: Sequence[str], reference: str) -> float | None:
        return self._item("context_precision", question, list(contexts), reference)

    def context_recall(self, question: str, contexts: Sequence[str], reference: str) -> float | None:
        return self._item("context_recall", question, list(contexts), reference)

    # --- one item ---------------------------------------------------------------------------------

    def _item(self, method: str, *args: Any) -> float | None:
        key = _key(method, args)
        if key in self.scores:
            self.done += 1
            return self.scores[key]
        self.counter.check()
        errors_before = sum(self.errors.values())
        try:
            value = getattr(self.inner, method)(*args)
        except BaseException:
            self.flush()
            raise
        self.counter.check()  # a breach raised in a worker thread may have been swallowed on the way here
        self.done += 1
        self.watchdog.beat()
        if sum(self.errors.values()) == errors_before:
            self.scores[key] = value
            self.new_since_flush += 1
            if self.new_since_flush >= self.batch:
                self.flush()
        self._log(f"[{self.done}/{self.total}] {method}" + ("" if key in self.scores else " FAILED (will retry on resume)"))
        return value

    # --- persistence and logging ----------------------------------------------------------------------

    def _load(self) -> None:
        if self.checkpoint is None or not self.checkpoint.exists():
            return
        try:
            data = json.loads(self.checkpoint.read_text())
        except (OSError, ValueError):
            return
        if data.get("signature") != self.signature:
            self._log("checkpoint ignored: it was made for different pipeline answers")
            return
        self.scores = dict(data.get("scores", {}))
        self.resumed = len(self.scores)
        if hasattr(self.counter, "base_usd"):
            self.counter.base_usd = float(data.get("spend_usd", 0.0))
        self._log(f"resumed {self.resumed}/{self.total} scored items from the checkpoint "
                  f"(${getattr(self.counter, 'base_usd', 0.0):.4f} already spent)")

    def flush(self) -> None:
        if self.checkpoint is None:
            return
        with self._lock:
            self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            payload = {"signature": self.signature, "total": self.total, "scores": self.scores,
                       "spend_usd": round(self.counter.cost_usd(self.settings), 6), "calls": self.counter.calls,
                       "updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            tmp = self.checkpoint.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=1))
            os.replace(tmp, self.checkpoint)
            self.new_since_flush = 0

    def _log(self, message: str) -> None:
        spend = self.counter.cost_usd(self.settings) if self.counter is not None else 0.0
        line = f"{datetime.now().isoformat(timespec='seconds')} {message}  spend ${spend:.4f} ({self.counter.calls} calls)"
        if self.progress_log is not None:
            self.progress_log.parent.mkdir(parents=True, exist_ok=True)
            with self.progress_log.open("a") as handle:
                handle.write(line + "\n")

    def _stalled(self) -> None:
        self.flush()
        self._log(f"WATCHDOG: no progress for {self.watchdog._stall:.0f}s; aborting (checkpoint saved)")
        self._abort()

    @staticmethod
    def _abort_process() -> None:
        print("error: the Ragas judge made no progress and was aborted; the checkpoint is saved, rerun to resume",
              file=sys.stderr, flush=True)
        from smartdesk_ai.ops.spend_ledger import end_run

        end_run("stalled")  # os._exit skips atexit
        os._exit(EXIT_STALLED)  # a thread blocked in a network call cannot be interrupted any other way
