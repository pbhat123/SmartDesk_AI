"""Local trace log (FR-G8, FR-G11, FR-H9 local part): one JSON line per event, written to a local file and
kept working with no tracing keys. Every string in an event passes through the masker first, so a raw
email, phone, card or ID never reaches the file. Callers pass facts (tool, outcome, latencies); they never
pass prompt or reply text unmasked, because masking is applied here rather than left to each caller.
"""
import json
import threading
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

from smartdesk_ai.guardrails.pii import PiiRedactor, mask_emails_in
from smartdesk_ai.llm.fallback import LLMCall
from smartdesk_ai.ops.runtime import RequestContext

Masker = Callable[[str], str]

# The prompt file each LLM task uses (FR-H14); the active version comes from config.yaml `prompts`.
TASK_PROMPTS = {
    "ORCHESTRATOR": "orchestrator", "WORKER_ANSWER": "answer", "SYNTHESIZER": "synthesizer",
    "GROUNDING_JUDGE": "grounding_judge", "TOOL_SELECT": "tool_select", "TICKET_DRAFT": "ticket_summary",
    "INJECTION_CLASSIFIER": "injection_classifier", "COMPACTION": "compaction", "CONFIRM_PARSE": "confirm_parse",
    "CHITCHAT": "chitchat",
}


def prompt_text(prompt: object) -> str:
    """A prompt (messages or text) as one string, one line group per message."""
    if isinstance(prompt, str):
        return prompt
    parts = []
    for message in prompt if isinstance(prompt, list | tuple) else [prompt]:
        role = message.type if isinstance(message, BaseMessage) else "message"
        parts.append(f"[{role}] {getattr(message, 'content', message)}")
    return "\n".join(parts)


def make_masker(redactor: PiiRedactor) -> Masker:
    """Redact phone, card and ID values, then mask every email address ("j***@example.com")."""
    return lambda text: mask_emails_in(redactor.redact(text).text)


class TraceLog:
    def __init__(
        self, path: Path | str | None, masker: Masker, *, prompt_versions: Mapping[str, int] | None = None
    ) -> None:
        self.path = Path(path) if path is not None else None
        self._mask = masker
        self._prompt_versions = dict(prompt_versions or {})
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

    def request_fields(self, ctx: RequestContext | None) -> dict[str, Any]:
        """The IDs every event of a request carries: session, request and the masked user ID."""
        if ctx is None:
            return {}
        return {"session_id": ctx.session_id, "request_id": ctx.request_id, "user_id": ctx.user_id}

    def llm_record(self, ctx: RequestContext | None, agent: str, call: LLMCall) -> None:
        """One record per LLM call (FR-H9). Prompt and completion are redacted by the masker."""
        prompt_name = TASK_PROMPTS.get(call.task)
        version = self._prompt_versions.get(prompt_name) if prompt_name else None
        self.event(
            "llm_call", agent=agent, task=call.task,
            prompt_version=f"{prompt_name}_v{version}" if prompt_name and version else None,
            prompt=prompt_text(call.prompt), completion=call.completion, model=call.model_id,
            model_version=call.model_version, tier=call.tier, input_tokens=call.input_tokens,
            cached_tokens=call.cached_tokens, output_tokens=call.output_tokens, latency_ms=call.latency_ms,
            steps=[{"tier": s.tier, "latency_ms": s.latency_ms, "ok": s.ok} for s in call.steps],
            fell_back=call.fell_back, ok=call.ok, **self.request_fields(ctx),
        )
