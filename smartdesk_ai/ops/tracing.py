"""Optional tracing integrations: Langfuse and LangSmith (FR-H9, FR-F6, FR-G11).

The T0.8 spike showed the SDK masking hooks cannot be trusted alone (span names, tags, error text and
more are not covered), so the design is: only the local trace log's already-masked records are sent. Each
record is turned into one Langfuse observation or one LangSmith run with a fixed name (its kind), and the
SDK hooks stay on as a second layer. Nothing is sent without keys; a missing or failing SDK degrades to the
local log (FR-F7). LangChain auto-tracing is never switched on: `LANGSMITH_TRACING` is never set here, and
a copy set in the environment is reported, because LangChain would then build an unmasked client.
"""
import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from smartdesk_ai.config.env import EnvSettings
from smartdesk_ai.ops.health import Component, HealthRegistry
from smartdesk_ai.ops.trace_log import Masker, TraceLog, mask_value

logger = logging.getLogger(__name__)

_NAME_FIELDS = ("kind", "ts")


class TraceSink(Protocol):
    def emit(self, event: dict[str, Any]) -> None: ...


def split_event(event: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """(input, output, metadata) of a masked trace event, by kind."""
    kind = event["kind"]
    used: set[str] = set(_NAME_FIELDS)
    inputs: Any = None
    outputs: Any = None
    if kind == "llm_call":
        inputs, outputs, used = event.get("prompt"), event.get("completion"), used | {"prompt", "completion"}
    elif kind == "request":
        inputs, used = event.get("text"), used | {"text"}
    elif kind == "reply":
        outputs, used = event.get("text"), used | {"text"}
    elif kind == "tool_call":
        inputs = event.get("args")
        outputs = {k: event.get(k) for k in ("outcome", "error_code", "error", "ticket")}
        used |= {"args", "outcome", "error_code", "error", "ticket"}
    metadata = {k: v for k, v in event.items() if k not in used}
    return inputs, outputs, metadata


# --- hooks (second layer of defence) --------------------------------------------------------

def langfuse_hooks(masker: Masker) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """`mask=` (observation input, output, metadata) and `mask_otel_spans=` (every span attribute)."""
    from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult, OtelSpanPatch

    def mask(*, data: Any, **kwargs: Any) -> Any:
        return mask_value(data, masker)

    def mask_otel_spans(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult:
        patches: dict[Any, OtelSpanPatch] = {}
        for identifier, span in params.spans.items():
            changed: dict[str, Any] = {}
            for key, value in span.attributes.items():
                if isinstance(value, str):
                    masked: Any = masker(value)
                elif isinstance(value, list | tuple) and all(isinstance(v, str) for v in value):
                    masked = tuple(masker(v) for v in value)
                else:
                    continue
                if masked != value:
                    changed[key] = masked
            if changed:
                patches[identifier] = OtelSpanPatch(set_attributes=changed)
        return MaskOtelSpansResult(span_patches=patches)

    return mask, mask_otel_spans


def langsmith_hooks(masker: Masker) -> dict[str, Callable[[Any], Any]]:
    """`hide_inputs`, `hide_outputs`, `hide_metadata` for the LangSmith client."""
    def hide(value: Any) -> Any:
        return mask_value(value, masker)

    return {"hide_inputs": hide, "hide_outputs": hide, "hide_metadata": hide}


# --- sinks ------------------------------------------------------------------------------------

class LangfuseSink:
    def __init__(self, client: Any) -> None:
        self._client = client

    def flush(self) -> None:
        self._client.flush()

    def emit(self, event: dict[str, Any]) -> None:
        from langfuse import propagate_attributes

        kind = event["kind"]
        inputs, outputs, metadata = split_event(event)
        extra: dict[str, Any] = {}
        if kind == "llm_call":
            as_type = "generation"
            extra = {"model": event.get("model"), "usage_details": {
                "input": event.get("input_tokens") or 0, "output": event.get("output_tokens") or 0}}
        else:
            as_type = "tool" if kind == "tool_call" else "span"
        ids = {k: v for k, v in (("user_id", event.get("user_id")), ("session_id", event.get("session_id"))) if v}
        with propagate_attributes(**ids):
            with self._client.start_as_current_observation(
                name=kind, as_type=as_type, input=inputs, output=outputs, metadata=metadata, **extra
            ):
                pass


class LangSmithSink:
    def __init__(self, client: Any) -> None:
        self._client = client

    def flush(self) -> None:
        self._client.flush()

    def emit(self, event: dict[str, Any]) -> None:
        kind = event["kind"]
        inputs, outputs, metadata = split_event(event)
        run_type = "llm" if kind == "llm_call" else "tool" if kind == "tool_call" else "chain"
        now = datetime.now(timezone.utc)
        self._client.create_run(
            name=kind, inputs={"input": inputs}, run_type=run_type, outputs={"output": outputs},
            id=uuid.uuid4(), start_time=now, end_time=now, extra={"metadata": metadata},
        )


# --- wiring -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class TracingFactories:
    """How to build each SDK client from the env and the masker. Tests inject offline clients."""

    langfuse: Callable[[EnvSettings, Masker], Any] | None = None
    langsmith: Callable[[EnvSettings, Masker], Any] | None = None


def default_langfuse_client(env: EnvSettings, masker: Masker) -> Any:
    from langfuse import Langfuse

    mask, mask_otel_spans = langfuse_hooks(masker)
    host = {"base_url": env.langfuse_host} if env.langfuse_host else {}  # unset: the SDK's default (EU cloud)
    return Langfuse(
        public_key=env.langfuse_public_key.get_secret_value(), secret_key=env.langfuse_secret_key.get_secret_value(),
        mask=mask, mask_otel_spans=mask_otel_spans, **host)


def default_langsmith_client(env: EnvSettings, masker: Masker) -> Any:
    from langsmith import Client

    return Client(api_key=env.langsmith_api_key.get_secret_value(), **langsmith_hooks(masker))


def build_tracing(
    env: EnvSettings, trace: TraceLog, registry: HealthRegistry, factories: TracingFactories | None = None
) -> list[TraceSink]:
    """Attach a sink to `trace` for each integration whose keys are present and whose SDK loads. The local
    log is always kept. Returns the sinks that were attached."""
    factories = factories or TracingFactories()
    if os.environ.get("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes"):
        logger.warning("LANGSMITH_TRACING is set in the environment; unset it. SmartDesk AI sends traces "
                       "only through its own masked client, and LangChain auto-tracing is not masked")
    masker = trace.mask
    sinks: list[TraceSink] = []
    if env.langfuse_enabled:
        client = registry.try_load(Component.LANGFUSE, lambda: (factories.langfuse or default_langfuse_client)(env, masker))
        if client is not None:
            sinks.append(LangfuseSink(client))
    if env.langsmith_enabled:
        client = registry.try_load(Component.LANGSMITH, lambda: (factories.langsmith or default_langsmith_client)(env, masker))
        if client is not None:
            sinks.append(LangSmithSink(client))
    for sink in sinks:
        trace.add_sink(sink)
    return sinks
