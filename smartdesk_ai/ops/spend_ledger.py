"""A persistent, append-only record of every paid model call (FR-H7, FR-H8, FR-G11).

One fixed file for the whole repo, `data/spend_ledger.jsonl` (override with `SMARTDESK_LEDGER_PATH`). It does
not follow the data dir an app or eval stack happens to use, because eval stacks run in temp dirs and their
spend must not be lost. Each line is one event of one run: `start`, `spend` (one per API call, written as it
happens, so a crash or a hard kill loses at most the call in flight) and `end`. Lines hold numbers and names
only, never prompt or completion text; the command line has emails and the home folder masked.

Hooks sit where a call really leaves the process, once each: the chat callback built into `openai_builder`
(also given to the Ragas judge model) and `LedgeredEmbeddings` around the real embedder. Nothing is written
unless a run was started, so mock mode and the tests never touch the file. `CostTracker` is separate: it is the
per-request analytics, this is the audit trail to compare with the provider's billing page.

`python -m smartdesk_ai.ops.spend_ledger [--day YYYY-MM-DD]` prints the spend per day (UTC, as the billing page
does) and per run.
"""
import argparse
import atexit
import json
import os
import re
import sys
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smartdesk_ai.config.settings import Settings

LEDGER_ENV = "SMARTDESK_LEDGER_PATH"
DEFAULT_LEDGER = Path(__file__).resolve().parents[2] / "data" / "spend_ledger.jsonl"  # data/ is gitignored
_EMBEDDING_ENCODING = "cl100k_base"
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_KEY = re.compile(r"sk-[A-Za-z0-9_-]{8,}")


def ledger_path(environ: dict[str, str] | None = None) -> Path:
    override = (environ if environ is not None else os.environ).get(LEDGER_ENV)
    return Path(override) if override else DEFAULT_LEDGER


def mask_command(argv: list[str] | str) -> str:
    text = argv if isinstance(argv, str) else " ".join(argv)
    text = _KEY.sub("[key]", _EMAIL.sub("[email]", text))
    home = str(Path.home())
    return text.replace(home, "~") if home and home != "/" else text


@dataclass
class Usage:
    """Tokens of one model result. `output_tokens` already includes the reasoning tokens (the provider bills them
    as output); `reasoning_tokens` is kept only so the ledger shows them."""

    calls: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    model: str | None = None


def usage_from_result(response: Any) -> Usage:
    """Tokens of a LangChain `LLMResult`, each billed call counted once. With n > 1 completions every generation's
    message repeats the same call-level usage, so summing the generations would count it n times; the call-level
    `llm_output["token_usage"]` is used when present, else the first generation of each prompt."""
    prompts = list(getattr(response, "generations", None) or [])
    output = getattr(response, "llm_output", None) or {}
    model = output.get("model_name") if isinstance(output, dict) else None
    token_usage = output.get("token_usage") if isinstance(output, dict) else None
    if token_usage:
        cached = (token_usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        reasoning = (token_usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        return Usage(max(len(prompts), 1), token_usage.get("prompt_tokens") or 0, cached,
                     token_usage.get("completion_tokens") or 0, reasoning, model)
    total = Usage(model=model)
    for generations in prompts:
        message = getattr(generations[0], "message", None) if generations else None
        meta = getattr(message, "usage_metadata", None) or {}
        total.calls += 1
        total.input_tokens += meta.get("input_tokens", 0)
        total.cached_tokens += (meta.get("input_token_details") or {}).get("cache_read", 0)
        total.output_tokens += meta.get("output_tokens", 0)
        total.reasoning_tokens += (meta.get("output_token_details") or {}).get("reasoning", 0)
    return total


class LedgerRun:
    def __init__(self, path: Path, settings: Settings, command: str, source: str, run_id: str | None = None) -> None:
        self.path, self.settings, self.source = Path(path), settings, source
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.command = mask_command(command)
        self.calls = self.input_tokens = self.cached_tokens = self.output_tokens = self.reasoning_tokens = 0
        self.usd = 0.0
        self.finished = False
        self._lock = threading.Lock()
        self._write({"event": "start", "command": self.command, "pid": os.getpid()})

    # --- pricing ---------------------------------------------------------------------------------

    def _price(self, model: str | None) -> tuple[Any, bool]:
        """The config price for a model id (a dated model name matches its configured id as a prefix).
        An unknown model is priced as the Main model and flagged, so a missing price never reads as free."""
        models = self.settings.models
        known = [m for m in (models.main, models.light, models.embedding) if model and model.startswith(m.id)]
        if known:
            return max(known, key=lambda m: len(m.id)).price, False
        return models.main.price, True

    def chat_usd(self, usage: Usage, model: str | None) -> tuple[float, bool]:
        price, assumed = self._price(model)
        fresh = max(usage.input_tokens - usage.cached_tokens, 0)
        usd = (fresh * price.input + usage.cached_tokens * (price.cached_input or price.input)
               + usage.output_tokens * price.output) / 1_000_000
        return usd, assumed

    # --- recording -------------------------------------------------------------------------------

    def add_chat(self, usage: Usage, model_hint: str | None = None, ok: bool = True) -> float:
        model = usage.model or model_hint
        usd, assumed = self.chat_usd(usage, model)
        return self._spend("chat", model, usage, usd, ok, assumed)

    def add_embedding(self, tokens: int, model: str) -> float:
        price, assumed = self._price(model)
        usd = tokens * price.input / 1_000_000
        return self._spend("embedding", model, Usage(1, tokens, 0, 0, 0), usd, True, assumed)

    def _spend(self, kind: str, model: str | None, usage: Usage, usd: float, ok: bool, assumed: bool) -> float:
        with self._lock:
            self.calls += usage.calls
            self.input_tokens += usage.input_tokens
            self.cached_tokens += usage.cached_tokens
            self.output_tokens += usage.output_tokens
            self.reasoning_tokens += usage.reasoning_tokens
            self.usd += usd
        record = {"event": "spend", "kind": kind, "model": model, "ok": ok, "calls": usage.calls,
                  "input_tokens": usage.input_tokens, "cached_tokens": usage.cached_tokens,
                  "output_tokens": usage.output_tokens, "reasoning_tokens": usage.reasoning_tokens,
                  "usd": round(usd, 8)}
        if assumed:
            record["price_assumed"] = True
        self._write(record)
        return usd

    def finish(self, status: str = "ok") -> None:
        if self.finished:
            return
        self.finished = True
        self._write({"event": "end", "status": status, "calls": self.calls, "input_tokens": self.input_tokens,
                     "cached_tokens": self.cached_tokens, "output_tokens": self.output_tokens,
                     "reasoning_tokens": self.reasoning_tokens, "usd": round(self.usd, 8)})

    def _write(self, fields: dict[str, Any]) -> None:
        line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "run_id": self.run_id,
                           "source": self.source, **fields})
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:  # append mode: lines from two processes never overwrite each other
                handle.write(line + "\n")


# --- the process-wide run -------------------------------------------------------------------------

_run: LedgerRun | None = None
_atexit_registered = False


def start_run(settings: Settings, command: list[str] | str | None = None, source: str = "script",
              path: Path | None = None) -> LedgerRun:
    """Begin recording this process's paid calls. A second call returns the run already active."""
    global _run, _atexit_registered
    if _run is not None and not _run.finished:
        return _run
    _run = LedgerRun(path or ledger_path(), settings, mask_command(command if command is not None else sys.argv),
                     source)
    if not _atexit_registered:
        atexit.register(lambda: end_run("exited without end_run"))
        _atexit_registered = True
    return _run


def current_run() -> LedgerRun | None:
    return _run if _run is not None and not _run.finished else None


def end_run(status: str = "ok") -> None:
    if _run is not None:
        _run.finish(status)


def reset_for_tests() -> None:
    global _run
    _run = None


def record_chat(usage: Usage, model_hint: str | None = None, ok: bool = True) -> None:
    run = current_run()
    if run is not None:
        run.add_chat(usage, model_hint, ok)


def make_callback(model_id: str) -> Any:
    """A LangChain callback that reports each real chat call to the ledger. Mock models never get one."""
    from langchain_core.callbacks import BaseCallbackHandler

    class LedgerCallback(BaseCallbackHandler):
        raise_error = False  # bookkeeping never breaks a model call

        def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            record_chat(usage_from_result(response), model_id)

        def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
            record_chat(Usage(calls=1), model_id, ok=False)  # the call may still have been billed; tokens unknown

    return LedgerCallback()


class LedgeredEmbeddings:
    """Wraps the real embedder: each API request is counted with a local token count (as `CostedEmbeddings` does)."""

    def __init__(self, inner: Any, model_id: str) -> None:
        self._inner, self._model = inner, model_id

    def __getattr__(self, name: str) -> Any:  # name, model, ... of the wrapped embedder (embedder_id reads them)
        return getattr(self._inner, name)

    def _count(self, texts: list[str]) -> None:
        run = current_run()
        if run is None or not texts:
            return
        import tiktoken

        encoding = tiktoken.get_encoding(_EMBEDDING_ENCODING)
        run.add_embedding(sum(len(encoding.encode(t)) for t in texts), self._model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._inner.embed_documents(texts)
        self._count(texts)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vector = self._inner.embed_query(text)
        self._count([text])
        return vector


# --- report ---------------------------------------------------------------------------------------

def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue  # a half-written last line after a hard kill
    return events


def summarize(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per run: command, first day (UTC), calls, tokens, USD, status (`running or killed` when there is no end line)."""
    runs: dict[str, dict[str, Any]] = {}
    for event in events:
        run = runs.setdefault(event["run_id"], {"command": "", "day": event["ts"][:10], "calls": 0, "input_tokens": 0,
                                                "output_tokens": 0, "usd": 0.0, "status": "running or killed"})
        if event["event"] == "start":
            run["command"] = event.get("command", "")
        elif event["event"] == "spend":
            run["calls"] += event.get("calls", 0)
            run["input_tokens"] += event.get("input_tokens", 0)
            run["output_tokens"] += event.get("output_tokens", 0)
            run["usd"] += event.get("usd", 0.0)
        elif event["event"] == "end":
            run["status"] = event.get("status", "ok")
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m smartdesk_ai.ops.spend_ledger")
    parser.add_argument("--day", help="only this UTC day, YYYY-MM-DD (the billing page's days are UTC)")
    parser.add_argument("--path", type=Path, default=None)
    args = parser.parse_args(argv)
    runs = summarize(read_events(args.path or ledger_path()))
    per_day: dict[str, float] = defaultdict(float)
    for run in runs.values():
        per_day[run["day"]] += run["usd"]
        if args.day is None or run["day"] == args.day:
            print(f"{run['day']}  ${run['usd']:.4f}  {run['calls']:>4} calls  {run['status']:<22} {run['command'][:70]}")
    for day, usd in sorted(per_day.items()):
        if args.day is None or day == args.day:
            print(f"total {day}: ${usd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
