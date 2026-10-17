"""T0.6: app skeleton starts and answers a stub reply with no keys, no Redis,
no tracing keys (FR-F6, FR-F7). Guard hooks are no-ops here (filled in T5.x).

The real .env is never read: in-process tests pass a missing dotenv path, and the
subprocess test runs from an empty temp directory with a bare environment.
"""
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from smartdesk_ai.agent.__main__ import main
from smartdesk_ai.agent.chat_service import (
    EMPTY_REPLY,
    STUB_REPLY,
    ChatReply,
    ChatService,
    GuardDecision,
    build_service,
    stub_responder,
)
from smartdesk_ai.config.settings import load_settings
from smartdesk_ai.ops.health import Component, HealthRegistry

ROOT = Path(__file__).resolve().parent.parent
NO_FILE = Path("/nonexistent/.env")


@pytest.fixture()
def settings():
    return load_settings(ROOT / "config.yaml")


# --- ChatService ------------------------------------------------------------

def test_default_service_returns_the_stub_reply(settings):
    reply = ChatService(settings).handle("s1", "hello")
    assert isinstance(reply, ChatReply)
    assert reply.text == STUB_REPLY
    assert reply.ticket is None


def test_stub_reply_does_not_echo_user_text(settings):
    assert "unlikely-marker" not in ChatService(settings).handle("s1", "unlikely-marker").text


def test_empty_session_id_is_rejected(settings):
    with pytest.raises(ValueError, match="session_id"):
        ChatService(settings).handle("", "hello")


def test_input_guard_output_feeds_the_responder_and_output_guard_runs_last(settings):
    seen = {}

    def input_guard(text):
        seen["guard_in"] = text
        return GuardDecision(text=text.upper())

    def responder(session_id, text):
        seen["responder_in"] = (session_id, text)
        return "answer"

    def output_guard(text):
        seen["output_in"] = text
        return text + " [checked]"

    svc = ChatService(settings, input_guard=input_guard, responder=responder, output_guard=output_guard)
    reply = svc.handle("s9", "hi there")
    assert seen == {"guard_in": "hi there", "responder_in": ("s9", "HI THERE"), "output_in": "answer"}
    assert reply.text == "answer [checked]"


def test_blocked_input_skips_the_responder_and_returns_the_refusal(settings):
    calls = []

    def responder(session_id, text):
        calls.append(text)
        return "should not happen"

    svc = ChatService(
        settings,
        input_guard=lambda t: GuardDecision(text=t, blocked=True, refusal="Sorry, I can't help with that.", reason="test"),
        responder=responder,
        output_guard=lambda t: t + " [checked]",
    )
    reply = svc.handle("s1", "bad")
    assert calls == []
    assert reply.text == "Sorry, I can't help with that. [checked]"
    assert reply.debug["blocked_reason"] == "test"


def test_raw_user_text_never_reaches_logs(settings, caplog):
    with caplog.at_level(logging.DEBUG):
        ChatService(settings).handle("s1", "my email is jane.doe@example.com")
    assert "jane.doe@example.com" not in caplog.text


# --- startup with nothing configured ---------------------------------------

def test_build_service_with_no_keys_no_redis_no_tracing_registers_degraded_modes():
    registry = HealthRegistry()
    service = build_service(environ={}, dotenv_path=NO_FILE, registry=registry)
    assert service.handle("s1", "hello").text == STUB_REPLY
    degraded = {h.component for h in registry.degraded_modes()}
    assert degraded == {Component.REDIS, Component.LANGSMITH, Component.LANGFUSE}


# --- CLI --------------------------------------------------------------------

def run_cli(lines, argv=()):
    feed = iter(lines)

    def input_fn(prompt=""):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    out = []
    code = main(
        list(argv),
        input_fn=input_fn,
        output=out.append,
        environ={},
        dotenv_path=NO_FILE,
        registry=HealthRegistry(),
        responder=stub_responder,  # these tests are about startup; the graph has its own tests
    )
    return code, out


def test_cli_answers_then_quits():
    code, out = run_cli(["hello", "quit"])
    assert code == 0
    assert out.count(STUB_REPLY) == 1


def test_cli_exits_cleanly_on_eof_and_answers_blank_lines_with_a_prompt():
    code, out = run_cli(["", "hello"])
    assert code == 0
    assert out.count(STUB_REPLY) == 1 and out.count(EMPTY_REPLY) == 1


def test_cli_banner_lists_degraded_modes():
    _, out = run_cli(["quit"])
    banner = "\n".join(out)
    for name in ("redis", "langsmith", "langfuse"):
        assert name in banner


def test_cli_debug_flag_shows_debug_info():
    _, plain = run_cli(["hello", "quit"])
    _, debug = run_cli(["hello", "quit"], argv=["--debug"])
    assert not any(line.startswith("[debug]") for line in plain)
    assert any(line.startswith("[debug]") for line in debug)


def test_module_entry_point_runs_with_a_bare_environment(tmp_path):
    """python -m smartdesk_ai.agent: no keys, no Redis, no tracing, no .env."""
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-m", "smartdesk_ai.agent"],
        input="zzqx wvvk plorf\nquit\n",
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "[debug]" not in result.stdout and "Traceback" not in result.stderr
    # the real graph answers in mock mode: an unknown topic gets "I don't know" and a ticket offer
    assert "I don't know" in result.stdout and "ticket" in result.stdout
