"""T0.8 spike: Langfuse and LangSmith masking hooks (FR-G11).

Throwaway proof, not production code. It shows what each SDK's masking hook covers,
using only local mocks: no keys, no network. The email regex here stands in for the
real redaction that arrives with the guardrails (T5.x).
"""
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from langfuse import Langfuse
from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult, OtelSpanPatch
from langsmith import Client
from pydantic import BaseModel

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
EMAIL_MASK = "[EMAIL]"
PROBE_EMAIL = "jane.doe@example.com"


# --- local masking -----------------------------------------------------------

def mask_text(text: str) -> str:
    return EMAIL_RE.sub(EMAIL_MASK, text)


def mask_value(value: Any) -> Any:
    """Recursively mask emails. The result is always JSON-serializable."""
    if isinstance(value, str):
        return mask_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, BaseModel):
        return mask_value(value.model_dump())
    if isinstance(value, Mapping):
        return {mask_text(str(key)): mask_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [mask_value(item) for item in value]
    return mask_text(str(value))


def redact_uncovered(**fields: Any) -> dict[str, Any]:
    """Mask values for fields the SDK hooks do not reach (names, tags, user and
    session IDs, model names, status messages, versions). Callers pass these
    through here before handing them to an SDK."""
    return {key: mask_value(value) for key, value in fields.items()}


# --- Langfuse ----------------------------------------------------------------

def langfuse_mask(*, data: Any, **kwargs: Any) -> Any:
    """`mask=`: runs on observation input, output and metadata only."""
    return mask_value(data)


def langfuse_mask_otel_spans(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult:
    """`mask_otel_spans=`: runs at export on every span attribute (not name or events)."""
    patches: dict[Any, OtelSpanPatch] = {}
    for identifier, span in params.spans.items():
        changed: dict[str, Any] = {}
        for key, value in span.attributes.items():
            if isinstance(value, str):
                masked = mask_text(value)
            elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
                masked = tuple(mask_text(v) for v in value)
            else:
                continue
            if masked != value:
                changed[key] = masked
        if changed:
            patches[identifier] = OtelSpanPatch(set_attributes=changed)
    return MaskOtelSpansResult(span_patches=patches)


def make_langfuse(
    span_exporter: Any,
    public_key: str,
    use_mask: bool = True,
    use_mask_otel_spans: bool = True,
    mask: Callable[..., Any] | None = None,
    mask_otel_spans: Callable[..., Any] | None = None,
) -> Langfuse:
    """A Langfuse client that exports to `span_exporter` and never to the network.

    Langfuse caches clients by public key, so callers needing isolation pass a unique key.
    """
    return Langfuse(
        public_key=public_key,
        secret_key="sk-test",
        base_url="http://localhost:9",
        span_exporter=span_exporter,
        mask=mask if mask is not None else (langfuse_mask if use_mask else None),
        mask_otel_spans=(
            mask_otel_spans
            if mask_otel_spans is not None
            else (langfuse_mask_otel_spans if use_mask_otel_spans else None)
        ),
    )


# --- LangSmith ---------------------------------------------------------------

def make_langsmith_client(
    session: Any,
    hide_inputs: Callable[[Any], Any] = mask_value,
    hide_outputs: Callable[[Any], Any] = mask_value,
    hide_metadata: Callable[[Any], Any] = mask_value,
) -> Client:
    """A LangSmith client with masking hooks, sending only through `session`."""
    return Client(
        api_url="http://localhost:9",
        api_key="ls-test",
        session=session,
        auto_batch_tracing=False,
        hide_inputs=hide_inputs,
        hide_outputs=hide_outputs,
        hide_metadata=hide_metadata,
    )


def _capture_default_tracer_posts() -> list[str]:
    """Run a LangChain call with whatever tracing the environment enables, capturing
    request bodies instead of sending them. Meant to run in its own process."""
    import socket
    from unittest import mock

    import requests
    from langchain_core.runnables import RunnableLambda

    def deny(*args: Any, **kwargs: Any) -> None:
        raise OSError("network blocked")

    socket.socket.connect = deny  # type: ignore[method-assign]
    socket.create_connection = deny  # type: ignore[assignment]

    bodies: list[str] = []

    def fake_request(self: Any, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        body = kwargs.get("data") or kwargs.get("json") or b""
        if not isinstance(body, (bytes, str)):
            body = json.dumps(body, default=str)
        if method == "POST":
            bodies.append(body.decode("latin1") if isinstance(body, bytes) else body)
        response = mock.MagicMock()
        response.status_code = 200
        response.json.return_value = {}
        response.text = "{}"
        response.headers = {}
        response.content = b"{}"
        return response

    requests.Session.request = fake_request  # type: ignore[method-assign]
    RunnableLambda(lambda x: f"reply to {x['q']}").invoke({"q": f"hi {PROBE_EMAIL}"})
    from langsmith.run_trees import get_cached_client

    try:
        get_cached_client().flush()
    except Exception:  # no client was created because tracing is off
        pass
    return bodies


def default_tracer_payload_leaks() -> bool:
    return any(PROBE_EMAIL in body for body in _capture_default_tracer_posts())


def default_tracer_payload_count() -> int:
    return len(_capture_default_tracer_posts())
