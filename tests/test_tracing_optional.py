"""T5.16: optional Langfuse and LangSmith tracing with masking (FR-H9, FR-F6, FR-G11). No keys: the local
log only, and the app runs. With mock keys: the spans and runs carry masked data only. Both SDKs are
pointed at offline sinks (an in-memory span exporter, a mock HTTP session) and a socket guard fails any real
connection."""
import json
import logging
import socket
import uuid
from unittest import mock

import pytest
import requests
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from smartdesk_ai.agent.chat_service import build_service
from smartdesk_ai.agent.factory import TRACE_LOG, build_responder
from smartdesk_ai.config.env import load_env
from smartdesk_ai.guardrails.pii import PiiRedactor
from smartdesk_ai.ops.health import Component, HealthRegistry, Status
from smartdesk_ai.ops.trace_log import TraceLog, make_masker
from smartdesk_ai.ops.tracing import (
    LangfuseSink,
    TracingFactories,
    build_tracing,
    langfuse_hooks,
    langsmith_hooks,
    split_event,
)
from smartdesk_ai.tickets import MockTicketClient

EMAIL = "jane.roe@example.com"
PHONE = "+91 98765 43210"
KEYS = {"LANGFUSE_PUBLIC_KEY": "pk-mock", "LANGFUSE_SECRET_KEY": "sk-mock", "LANGSMITH_API_KEY": "ls-mock"}
NO_FILE = "/nonexistent/.env"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)


class Offline:
    """Offline SDK clients that record what they would send."""

    def __init__(self):
        self.exporter = InMemorySpanExporter()
        self.sent: list[str] = []
        self.clients = []

    def langfuse(self, env, masker):
        from langfuse import Langfuse

        mask, mask_otel_spans = langfuse_hooks(masker)
        client = Langfuse(public_key=f"pk-{uuid.uuid4().hex}", secret_key="sk-test", base_url="http://localhost:9",
                          span_exporter=self.exporter, mask=mask, mask_otel_spans=mask_otel_spans)
        self.clients.append(client)
        return client

    def langsmith(self, env, masker):
        from langsmith import Client

        def record(url, *args, **kwargs):
            body = kwargs.get("data") or kwargs.get("json") or b""
            self.sent.append(body.decode("latin1") if isinstance(body, bytes | bytearray) else
                             body if isinstance(body, str) else json.dumps(body, default=str))
            response = mock.MagicMock()
            response.status_code, response.text = 200, "{}"
            response.json.return_value = {}
            return response

        session = mock.MagicMock(spec=requests.Session)
        session.post.side_effect = record
        session.patch.side_effect = record
        session.request.side_effect = lambda method, url, *a, **k: record(url, *a, **k)
        client = Client(api_url="http://localhost:9", api_key="ls-test", session=session, auto_batch_tracing=False,
                        **langsmith_hooks(masker))
        self.clients.append(client)
        return client

    def factories(self) -> TracingFactories:
        return TracingFactories(langfuse=self.langfuse, langsmith=self.langsmith)

    def flush(self):
        for client in self.clients:
            if hasattr(client, "flush"):
                client.flush()

    def span_text(self) -> str:
        return json.dumps([{"name": s.name, "attrs": {k: str(v) for k, v in dict(s.attributes).items()}}
                           for s in self.exporter.get_finished_spans()])


@pytest.fixture
def offline():
    fake = Offline()
    yield fake
    for client in fake.clients:
        if hasattr(client, "shutdown"):
            client.shutdown()


def new_trace(tmp_path):
    return TraceLog(tmp_path / "trace.jsonl", make_masker(PiiRedactor(None, 0.5)))


def converse(settings, tmp_path, env, factories=None, registry=None):
    registry = registry or HealthRegistry()
    responder = build_responder(settings, environ=env, dotenv_path=None, data_dir=tmp_path, registry=registry,
                                client=MockTicketClient(), tracing_factories=factories)
    from pathlib import Path

    service = build_service(environ=env, dotenv_path=Path(NO_FILE), registry=registry, responder=responder)
    replies = [service.handle("sess-t", text) for text in (
        "How do I reset my forgotten password?", "My monitor keeps flickering", EMAIL,
        f"Please call me on {PHONE}", "Yes, create it")]
    return replies, registry


# --- no keys ---------------------------------------------------------------------------------

def test_without_keys_only_the_local_log_is_used_and_the_app_runs(settings, tmp_path):
    replies, registry = converse(settings, tmp_path, env={})
    assert all(r.text for r in replies)
    assert (tmp_path / TRACE_LOG).read_text().count("\n") > 10
    assert {h.component for h in registry.degraded_modes()} >= {Component.LANGFUSE, Component.LANGSMITH}


def test_build_tracing_with_no_keys_attaches_nothing(tmp_path):
    registry = HealthRegistry()
    trace = new_trace(tmp_path)
    assert build_tracing(load_env(environ={}, dotenv_path=None), trace, registry) == [] and trace.sinks == []


# --- mock keys ---------------------------------------------------------------------------------

def test_with_keys_spans_and_runs_carry_masked_data_only(settings, tmp_path, offline):
    converse(settings, tmp_path, env=KEYS, factories=offline.factories())
    offline.flush()
    spans, runs = offline.span_text(), "\n".join(offline.sent)
    assert spans != "[]" and runs
    for blob in (spans, runs):
        assert EMAIL not in blob and "98765" not in blob and "jane.roe" not in blob
    assert "j***@example.com" in spans and "[PHONE]" in spans


def test_langfuse_gets_generations_tool_spans_and_masked_ids(settings, tmp_path, offline):
    converse(settings, tmp_path, env=KEYS, factories=offline.factories())
    offline.flush()
    spans = offline.exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert {"request", "reply", "agent_step", "llm_call", "tool_call"} <= names
    generation = next(s for s in spans if s.name == "llm_call")
    assert dict(generation.attributes)["langfuse.observation.type"] == "generation"
    assert dict(generation.attributes)["langfuse.observation.model.name"] == settings.models.main.id
    users = {dict(s.attributes).get("user.id") for s in spans} - {None}
    assert users and all(u.startswith("user-") for u in users)
    assert {dict(s.attributes).get("session.id") for s in spans} >= {"sess-t"}


def test_langsmith_gets_one_run_per_event_with_fixed_names(settings, tmp_path, offline):
    converse(settings, tmp_path, env=KEYS, factories=offline.factories())
    offline.flush()
    bodies = [json.loads(b) for b in offline.sent if b.startswith("{")]
    names = {b.get("name") for b in bodies}
    assert {"request", "reply", "llm_call", "tool_call"} <= names
    assert not any(EMAIL in b.get("name", "") for b in bodies)


def test_only_one_sdk_when_only_its_keys_are_present(settings, tmp_path, offline):
    registry = HealthRegistry()
    trace = new_trace(tmp_path)
    env = load_env(environ={"LANGSMITH_API_KEY": "ls-mock"}, dotenv_path=None)
    sinks = build_tracing(env, trace, registry, offline.factories())
    assert [type(s).__name__ for s in sinks] == ["LangSmithSink"]


def test_the_sdk_hooks_mask_too(offline):
    masker = make_masker(PiiRedactor(None, 0.5))
    mask, _ = langfuse_hooks(masker)
    assert mask(data={"q": f"hi {EMAIL}", "n": 1}) == {"q": "hi j***@example.com", "n": 1}
    hide = langsmith_hooks(masker)["hide_inputs"]
    assert hide({"q": EMAIL, "list": [PHONE]}) == {"q": "j***@example.com", "list": ["[PHONE]"]}


def test_split_event_puts_text_in_input_and_output_and_the_rest_in_metadata():
    inputs, outputs, meta = split_event({"kind": "llm_call", "ts": "t", "prompt": "p", "completion": "c", "model": "m"})
    assert (inputs, outputs, meta) == ("p", "c", {"model": "m"})
    inputs, outputs, meta = split_event({"kind": "tool_call", "ts": "t", "args": {"a": 1}, "outcome": "success",
                                         "tool": "search_kb", "ticket": None, "error": None, "error_code": None})
    assert inputs == {"a": 1} and outputs["outcome"] == "success" and meta == {"tool": "search_kb"}


# --- failures never break the app ------------------------------------------------------------

def test_an_sdk_that_fails_to_build_degrades_to_the_local_log(settings, tmp_path, caplog):
    def broken(env, masker):
        raise RuntimeError("bad key")

    with caplog.at_level(logging.WARNING):
        replies, registry = converse(settings, tmp_path, env=KEYS,
                                     factories=TracingFactories(langfuse=broken, langsmith=broken))
    assert all(r.text for r in replies) and len(registry.degraded_modes()) >= 2
    assert registry.get(Component.LANGFUSE).status is Status.DEGRADED
    assert (tmp_path / TRACE_LOG).exists()


def test_a_sink_that_raises_is_skipped_and_logged(tmp_path, caplog):
    class Bad:
        def emit(self, event):
            raise ConnectionError("down " + EMAIL)

    trace = new_trace(tmp_path)
    trace.add_sink(Bad())
    with caplog.at_level(logging.WARNING):
        trace.event("step", note="hello")
    assert trace.events[0]["note"] == "hello" and "Bad failed (ConnectionError)" in caplog.text
    assert EMAIL not in caplog.text


def test_sinks_get_a_copy_so_they_cannot_change_the_log(tmp_path):
    class Mutator:
        def emit(self, event):
            event["note"] = "changed"

    trace = new_trace(tmp_path)
    trace.add_sink(Mutator())
    trace.event("step", note="original")
    assert trace.events[0]["note"] == "original"


def test_langsmith_auto_tracing_in_the_environment_is_reported_and_never_set(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with caplog.at_level(logging.WARNING):
        build_tracing(load_env(environ={}, dotenv_path=None), new_trace(tmp_path), HealthRegistry())
    assert "LANGSMITH_TRACING" in caplog.text


def test_building_tracing_never_sets_the_langsmith_env_flag(settings, tmp_path, offline, monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    converse(settings, tmp_path, env=KEYS, factories=offline.factories())
    import os

    assert "LANGSMITH_TRACING" not in os.environ
