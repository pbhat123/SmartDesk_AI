"""T0.8 spike: Langfuse and LangSmith masking hooks (FR-G11). No keys, no network.

Langfuse spans go to an in-memory exporter (after Langfuse's own export-stage
transform). LangSmith requests go to a mock HTTP session. A socket guard makes any
real connection attempt fail the test. The tests also pin which fields the hooks do
NOT cover, because those decide whether masking can be guaranteed.
"""
import json
import os
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from unittest import mock

import pytest
import requests
from langchain_core.runnables import RunnableLambda
from langsmith import traceable, tracing_context
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from langfuse import propagate_attributes
from spikes.tracing_masking_spike import (
    mask_text,
    mask_value,
    make_langfuse,
    make_langsmith_client,
    redact_uncovered,
)

ROOT = Path(__file__).resolve().parent.parent
EMAIL = "jane.doe@example.com"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


# --- the local masking function --------------------------------------------

def test_mask_text_replaces_emails_only():
    assert mask_text(f"mail {EMAIL} now") == "mail [EMAIL] now"
    assert mask_text("no address here") == "no address here"


def test_mask_value_walks_nested_data_keys_and_values():
    data = {"q": f"hi {EMAIL}", "list": [EMAIL, 3, None], "nested": {EMAIL: ("a", EMAIL)}, "n": 1.5}
    masked = mask_value(data)
    assert EMAIL not in json.dumps(masked)
    assert masked["n"] == 1.5 and masked["list"][1:] == [3, None]


def test_mask_value_output_is_json_serializable_for_odd_objects():
    class Odd:
        def __str__(self):
            return f"odd {EMAIL}"

    assert EMAIL not in json.dumps(mask_value({"x": Odd()}))


# --- Langfuse ---------------------------------------------------------------

@pytest.fixture()
def langfuse_env():
    """Fresh Langfuse client per test. Clients are cached by public key, so use a unique one."""
    created = []

    def factory(**kwargs):
        exporter = InMemorySpanExporter()
        key = f"pk-{uuid.uuid4().hex}"
        client = make_langfuse(exporter, public_key=key, **kwargs)
        client.test_public_key = key
        created.append(client)
        return client, exporter

    yield factory
    for client in created:
        client.shutdown()


def leaking_attributes(exporter):
    return {
        (span.name, key)
        for span in exporter.get_finished_spans()
        for key, value in dict(span.attributes).items()
        if EMAIL in str(value)
    }


def emit_everything(client):
    """Put the email into every field a careless caller might."""
    with client.start_as_current_observation(
        name=f"op {EMAIL}", input={"q": f"hi {EMAIL}"}, metadata={"who": EMAIL}
    ) as span:
        span.update(output=f"reply to {EMAIL}", status_message=f"err {EMAIL}", version=f"v{EMAIL}")
        with propagate_attributes(
            user_id=EMAIL, session_id=f"s-{EMAIL}", tags=[EMAIL], metadata={"k": EMAIL}, trace_name=f"tn-{EMAIL}"
        ):
            with client.start_as_current_observation(
                name="gen", as_type="generation", model=f"m-{EMAIL}", input=[{"role": "user", "content": EMAIL}]
            ) as gen:
                gen.update(output=f"o {EMAIL}", model_parameters={"x": EMAIL})
    client.flush()


def test_langfuse_mask_hook_masks_input_output_and_metadata(langfuse_env):
    client, exporter = langfuse_env(use_mask=True, use_mask_otel_spans=False)
    emit_everything(client)
    attrs = {k: str(v) for s in exporter.get_finished_spans() for k, v in dict(s.attributes).items()}
    assert attrs["langfuse.observation.input"] == '{"q": "hi [EMAIL]"}'
    assert attrs["langfuse.observation.output"] == "reply to [EMAIL]"
    assert attrs["langfuse.observation.metadata.who"] == '"[EMAIL]"'


def test_langfuse_mask_hook_alone_leaves_other_fields_unmasked(langfuse_env):
    """Documents the gap: mask= only touches observation input, output and metadata."""
    client, exporter = langfuse_env(use_mask=True, use_mask_otel_spans=False)
    emit_everything(client)
    leaked_keys = {key for _, key in leaking_attributes(exporter)}
    assert {
        "user.id",
        "session.id",
        "langfuse.trace.tags",
        "langfuse.trace.name",
        "langfuse.trace.metadata.k",
        "langfuse.observation.model.name",
        "langfuse.observation.model.parameters",
        "langfuse.observation.status_message",
        "langfuse.version",
    } <= leaked_keys
    assert not leaked_keys & {"langfuse.observation.input", "langfuse.observation.output"}


def test_langfuse_mask_otel_spans_hook_cleans_every_attribute(langfuse_env):
    client, exporter = langfuse_env(use_mask=True, use_mask_otel_spans=True)
    emit_everything(client)
    assert leaking_attributes(exporter) == set()


def test_langfuse_span_names_and_exception_events_are_not_covered_by_either_hook(langfuse_env):
    client, exporter = langfuse_env(use_mask=True, use_mask_otel_spans=True)
    emit_everything(client)
    try:
        with client.start_as_current_observation(name="boom"):
            raise ValueError(f"failed for {EMAIL}")
    except ValueError:
        pass
    client.flush()
    spans = exporter.get_finished_spans()
    assert any(EMAIL in span.name for span in spans)
    boom = next(s for s in spans if s.name == "boom")
    assert any(EMAIL in str(e.attributes.get("exception.message")) for e in boom.events)


def test_langfuse_caller_side_redaction_of_names_closes_the_gap_for_names(langfuse_env):
    client, exporter = langfuse_env(use_mask=True, use_mask_otel_spans=True)
    safe = redact_uncovered(name=f"op {EMAIL}")
    with client.start_as_current_observation(name=safe["name"], input=EMAIL):
        pass
    client.flush()
    assert all(EMAIL not in span.name for span in exporter.get_finished_spans())
    assert leaking_attributes(exporter) == set()


def test_langfuse_failing_mask_function_fails_closed(langfuse_env):
    def broken(*, data, **kwargs):
        raise RuntimeError("mask broke")

    client, exporter = langfuse_env(mask=broken, use_mask_otel_spans=False)
    with client.start_as_current_observation(name="n", input=f"hi {EMAIL}", output=EMAIL, metadata={"a": EMAIL}):
        pass
    client.flush()
    assert len(exporter.get_finished_spans()) == 1
    assert leaking_attributes(exporter) == set()


def test_langfuse_failing_mask_otel_spans_drops_the_whole_batch(langfuse_env):
    def broken(*, params):
        raise RuntimeError("mask broke")

    client, exporter = langfuse_env(use_mask=False, mask_otel_spans=broken)
    with client.start_as_current_observation(name="n", input=EMAIL):
        pass
    client.flush()
    assert exporter.get_finished_spans() == ()


def test_langfuse_langchain_handler_masks_io_but_not_trace_tags_or_run_names(langfuse_env):
    from langfuse.langchain import CallbackHandler

    client, exporter = langfuse_env(use_mask=True, use_mask_otel_spans=False)
    chain = RunnableLambda(lambda x: f"reply to {x['q']}").with_config(run_name=f"chain-{EMAIL}")
    handler = CallbackHandler(public_key=client.test_public_key)
    chain.invoke({"q": f"hi {EMAIL}"}, config={"callbacks": [handler], "tags": [EMAIL], "metadata": {"who": EMAIL}})
    client.flush()
    attrs = {k: str(v) for s in exporter.get_finished_spans() for k, v in dict(s.attributes).items()}
    assert EMAIL not in attrs["langfuse.observation.input"]
    assert EMAIL not in attrs["langfuse.observation.output"]
    leaked = {key for _, key in leaking_attributes(exporter)}
    assert "langfuse.trace.tags" in leaked and "langfuse.trace.metadata.who" in leaked
    assert any(EMAIL in s.name for s in exporter.get_finished_spans())


# --- LangSmith --------------------------------------------------------------

@pytest.fixture()
def langsmith_env():
    sent: list[tuple[str, str]] = []

    def record(url, *args, **kwargs):
        body = kwargs.get("data") or kwargs.get("json") or b""
        text = body.decode("latin1") if isinstance(body, (bytes, bytearray)) else (
            body if isinstance(body, str) else json.dumps(body, default=str)
        )
        sent.append((url, text))
        response = mock.MagicMock()
        response.status_code = 200
        response.json.return_value = {}
        response.text = "{}"
        return response

    session = mock.MagicMock(spec=requests.Session)
    session.post.side_effect = record
    session.patch.side_effect = record
    session.request.side_effect = lambda method, url, *a, **k: record(url, *a, **k)

    def factory(**kwargs):
        return make_langsmith_client(session, **kwargs)

    return factory, sent


def leak_paths(sent):
    paths = set()

    def walk(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                walk(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                walk(value, f"{path}[{i}]")
        elif EMAIL in str(obj):
            paths.add(path)

    for _, text in sent:
        walk(json.loads(text))
    return paths


def test_langsmith_hide_hooks_mask_inputs_outputs_and_metadata(langsmith_env):
    factory, sent = langsmith_env
    client = factory()

    @traceable(client=client, metadata={"who": EMAIL})
    def op(q):
        return f"reply to {q}"

    with tracing_context(enabled=True):
        op(f"hi {EMAIL}")
    client.flush()
    assert sent, "expected runs to be sent to the mock session"
    paths = leak_paths(sent)
    assert not [p for p in paths if p.startswith((".inputs", ".outputs", ".extra"))]


def test_langsmith_hooks_do_not_cover_run_name_tags_or_error_text(langsmith_env):
    factory, sent = langsmith_env
    client = factory()

    @traceable(client=client, name=f"op {EMAIL}", tags=[EMAIL])
    def op(q):
        return q

    @traceable(client=client, name="boom")
    def boom(q):
        raise ValueError(f"failed for {EMAIL}")

    with tracing_context(enabled=True):
        op("x")
        with pytest.raises(ValueError):
            boom("x")
    client.flush()
    paths = leak_paths(sent)
    assert ".name" in paths and ".tags[0]" in paths and ".error" in paths


def test_langsmith_caller_side_redaction_closes_the_gap(langsmith_env):
    factory, sent = langsmith_env
    client = factory()
    safe = redact_uncovered(name=f"op {EMAIL}", tags=[EMAIL])

    @traceable(client=client, name=safe["name"], tags=safe["tags"])
    def op(q):
        raise ValueError(mask_text(f"failed for {EMAIL}"))

    with tracing_context(enabled=True):
        with pytest.raises(ValueError):
            op(f"hi {EMAIL}")
    client.flush()
    assert leak_paths(sent) == set()


def test_langsmith_failing_hide_function_fails_closed_and_does_not_break_the_app(langsmith_env):
    factory, sent = langsmith_env

    def broken(value):
        raise RuntimeError("hide broke")

    client = factory(hide_inputs=broken, hide_outputs=broken, hide_metadata=broken)

    @traceable(client=client)
    def op(q):
        return f"reply to {q}"

    with tracing_context(enabled=True):
        assert op(f"hi {EMAIL}") == f"reply to hi {EMAIL}"  # the app still works
    client.flush()
    assert not [text for _, text in sent if EMAIL in text]


def test_langsmith_langchain_tracer_path_uses_the_clients_hooks(langsmith_env):
    from langchain_core.tracers.langchain import LangChainTracer

    factory, sent = langsmith_env
    client = factory()
    chain = RunnableLambda(lambda x: f"reply to {x['q']}").with_config(run_name="c")
    chain.invoke({"q": f"hi {EMAIL}"}, config={"callbacks": [LangChainTracer(client=client)]})
    client.flush()
    assert sent
    assert not [p for p in leak_paths(sent) if p.startswith((".inputs", ".outputs"))]


def run_probe(code: str, env_extra: dict[str, str]) -> str:
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT), **env_extra}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_env_enabled_langsmith_tracing_bypasses_our_hooks():
    """Hazard: LANGSMITH_TRACING=true makes LangChain build its own default client, with no hooks."""
    out = run_probe(
        "from spikes.tracing_masking_spike import default_tracer_payload_leaks; print(default_tracer_payload_leaks())",
        {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "ls-test"},
    )
    assert out == "True"


def test_langsmith_hide_env_switches_suppress_inputs_and_outputs_entirely():
    out = run_probe(
        "from spikes.tracing_masking_spike import default_tracer_payload_leaks; print(default_tracer_payload_leaks())",
        {
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_API_KEY": "ls-test",
            "LANGSMITH_HIDE_INPUTS": "true",
            "LANGSMITH_HIDE_OUTPUTS": "true",
        },
    )
    assert out == "False"


def test_without_the_tracing_env_flag_nothing_is_sent():
    out = run_probe(
        "from spikes.tracing_masking_spike import default_tracer_payload_count; print(default_tracer_payload_count())",
        {},
    )
    assert out == "0"
