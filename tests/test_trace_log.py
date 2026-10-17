"""T5.9: local trace log with masking (FR-G8, FR-G11, FR-H9 local part). Every tool call and every retry
attempt is logged; the file never holds a raw email or other redacted PII."""
import json
from pathlib import Path

import pytest

from smartdesk_ai.agent.factory import build_responder
from smartdesk_ai.guardrails.pii import PiiRedactor
from smartdesk_ai.ops.trace_log import TraceLog, make_masker
from smartdesk_ai.tickets import MockTicketClient
from smartdesk_ai.tools.errors import TransientToolError
from smartdesk_ai.tools.executor import ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry

EMAIL = "jane.roe@example.com"
PHONE = "+91 98765 43210"


def new_log(tmp_path) -> TraceLog:
    return TraceLog(tmp_path / "trace.jsonl", make_masker(PiiRedactor(None, 0.5)))


def lines(log: TraceLog) -> list[dict]:
    return [json.loads(line) for line in log.path.read_text().splitlines()]


def test_events_are_written_as_json_lines_with_a_kind_and_a_time(tmp_path):
    log = new_log(tmp_path)
    log.event("step", agent="router", note="hello")
    log.event("step", agent="answer")
    first, second = lines(log)
    assert first["kind"] == "step" and first["agent"] == "router" and "ts" in first and second["agent"] == "answer"


def test_strings_anywhere_in_an_event_are_masked(tmp_path):
    log = new_log(tmp_path)
    log.event("x", text=f"mail {EMAIL} or call {PHONE}", nested={"to": [EMAIL, {"deep": EMAIL}]})
    raw = log.path.read_text()
    assert EMAIL not in raw and "98765" not in raw
    [event] = lines(log)
    assert "j***@example.com" in event["text"] and "[PHONE]" in event["text"]
    assert event["nested"]["to"][1]["deep"] == "j***@example.com"


def test_non_string_values_pass_through_and_unknown_objects_become_text(tmp_path):
    log = new_log(tmp_path)
    log.event("x", count=3, ok=True, nothing=None, path=Path("/tmp/a"))
    [event] = lines(log)
    assert (event["count"], event["ok"], event["nothing"], event["path"]) == (3, True, None, "/tmp/a")


def test_a_log_without_a_path_keeps_events_in_memory_only():
    log = TraceLog(None, make_masker(PiiRedactor(None, 0.5)))
    log.event("x", text=EMAIL)
    assert log.events[0]["text"] == "j***@example.com"


class Flaky:
    def __init__(self, *errors):
        self.errors = list(errors)

    def __call__(self, args):
        if self.errors:
            raise self.errors.pop(0)
        return "done"


def executor(settings, fn, log, name="search_kb"):
    registry = ToolRegistry()
    registry.register(name, fn)
    return ToolExecutor(registry, settings, sleep=lambda _: None, trace=log)


def test_every_attempt_and_the_call_itself_are_logged(settings, tmp_path):
    log = new_log(tmp_path)
    ex = executor(settings, Flaky(TransientToolError("503"), ConnectionError("net")), log)
    ex.execute("search_kb", {"query": "vpn"}, agent="it_worker")
    events = lines(log)
    attempts = [e for e in events if e["kind"] == "tool_attempt"]
    assert [(a["attempt"], a["outcome"]) for a in attempts] == [
        (1, "transient_failure"), (2, "transient_failure"), (3, "success")]
    [call] = [e for e in events if e["kind"] == "tool_call"]
    assert call["tool"] == "search_kb" and call["agent"] == "it_worker" and call["outcome"] == "success"
    assert call["attempts"] == 3 and call["args"] == {"query": "vpn"}
    assert all(isinstance(a["latency_ms"], float) for a in attempts)


def test_unavailable_error_and_rejected_calls_are_logged_too(settings, tmp_path):
    log = new_log(tmp_path)
    ex = executor(settings, Flaky(*[ConnectionError("net")] * 3), log)
    ex.execute("search_kb", {"query": "vpn"}, agent="a")
    ex.execute("search_kb", {}, agent="a")  # fails argument validation: no attempt is made
    ex2 = executor(settings, Flaky(ValueError("boom")), log)
    ex2.execute("search_kb", {"query": "vpn"}, agent="a")
    calls = [e for e in lines(log) if e["kind"] == "tool_call"]
    assert [c["outcome"] for c in calls] == ["unavailable", "validation_error", "error"]
    assert calls[1]["error_code"] == "schema" and calls[2]["error"] == "ValueError"
    assert len([e for e in lines(log) if e["kind"] == "tool_attempt"]) == 4  # 3 transient + 1 permanent


def test_an_email_in_tool_arguments_never_reaches_the_file(settings, tmp_path):
    log = new_log(tmp_path)
    ex = executor(settings, lambda args: [], log, name="get_tickets")
    ex.execute("get_tickets", {"email": EMAIL}, agent="status_flow")
    raw = log.path.read_text()
    assert EMAIL not in raw and "j***@example.com" in raw


def test_the_whole_mock_conversation_leaves_a_log_with_no_raw_email(settings, tmp_path):
    respond = build_responder(settings, environ={}, dotenv_path=None, data_dir=tmp_path, client=MockTicketClient())
    sid = "sess-1"
    for text in ("My monitor keeps flickering", EMAIL, "Yes, create it"):
        respond(sid, text)
    log_text = (tmp_path / "trace.jsonl").read_text()
    assert EMAIL not in log_text
    events = [json.loads(line) for line in log_text.splitlines()]
    tools = [e["tool"] for e in events if e["kind"] == "tool_call"]
    assert "search_kb" in tools and "create_ticket" in tools
    assert any(e["kind"] == "tool_call" and e["tool"] == "create_ticket" and e["args"].get("email") == "j***@example.com"
               for e in events)
