"""Local trace log (FR-G8, FR-G11, FR-H9 local part): one JSON line per event, written to a local file and
kept working with no tracing keys. Every string in an event passes through the masker first, so a raw
email, phone, card or ID never reaches the file. Callers pass facts (tool, outcome, latencies); they never
pass prompt or reply text unmasked, because masking is applied here rather than left to each caller.
"""
import json
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smartdesk_ai.guardrails.pii import PiiRedactor, mask_emails_in

Masker = Callable[[str], str]


def make_masker(redactor: PiiRedactor) -> Masker:
    """Redact phone, card and ID values, then mask every email address ("j***@example.com")."""
    return lambda text: mask_emails_in(redactor.redact(text).text)


class TraceLog:
    def __init__(self, path: Path | str | None, masker: Masker) -> None:
        self.path = Path(path) if path is not None else None
        self._mask = masker
        self._lock = threading.Lock()
        self.events: list[dict[str, Any]] = []  # masked copies, for the Monitoring tab and tests

    def _clean(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._mask(value)
        if isinstance(value, dict):
            return {self._mask(str(k)): self._clean(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [self._clean(v) for v in value]
        if value is None or isinstance(value, bool | int | float):
            return value
        return self._mask(str(value))

    def event(self, kind: str, **fields: Any) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, **self._clean(fields)}
        with self._lock:
            self.events.append(record)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
