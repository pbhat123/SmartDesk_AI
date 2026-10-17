"""T5.15: FR-H9 in the local trace log: every user request and reply, every agent step, every tool call and
every LLM call is logged, with a masked user ID and the session ID, and prompt and completion text pass
through redaction. Mock models, a temp log file."""
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from smartdesk_ai.agent.chat_service import ChatService, build_service
from smartdesk_ai.agent.factory import TRACE_LOG, build_responder
from smartdesk_ai.guardrails.pii import PiiRedactor
from smartdesk_ai.llm.fallback import FallbackChat
from smartdesk_ai.ops.runtime import RequestContext, Services, masked_user_id, request_scope
from smartdesk_ai.ops.trace_log import TASK_PROMPTS, TraceLog, make_masker
from smartdesk_ai.tickets import MockTicketClient

EMAIL = "jane.roe@example.com"
PHONE = "+91 98765 43210"


def new_trace(tmp_path, settings):
    return TraceLog(tmp_path / "trace.jsonl", make_masker(PiiRedactor(None, 0.5)), prompt_versions=settings.prompts)


def events(trace, kind=None):
    rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
    return [r for r in rows if kind is None or r["kind"] == kind]


class Model:
    def __init__(self, text="the answer", usage=None, meta=None):
        self.text, self.usage, self.meta = text, usage or {"input_tokens": 120, "output_tokens": 15}, meta or {}

    def invoke(self, messages):
        return SimpleNamespace(content=self.text, usage_metadata=self.usage, response_metadata=self.meta)


def call_llm(settings, trace, task="WORKER_ANSWER", model=None, messages=None, email=None, builder=None):
    ctx = RequestContext(session_id="sess-1", request_id="req-1")
    if email:
        ctx.set_email(email)
    chat = FallbackChat(task, settings, builder=builder or (lambda m, c: model or Model()), sleep=lambda _: None)
    with request_scope(ctx, Services(trace=trace)):
        chat.invoke(messages or [SystemMessage("be helpful"), HumanMessage("question")])
    return ctx


def test_an_llm_record_holds_everything_fr_h9_asks_for(settings, tmp_path):
    trace = new_trace(tmp_path, settings)
    meta = {"model_name": "gpt-5.4-mini-2026-09-01"}
    call_llm(settings, trace, model=Model(meta=meta), email=EMAIL)
    [rec] = events(trace, "llm_call")
    assert rec["agent"] == "worker_answer" and rec["task"] == "WORKER_ANSWER"
    assert rec["prompt_version"] == f"answer_v{settings.prompts['answer']}"
    assert "be helpful" in rec["prompt"] and "question" in rec["prompt"] and rec["completion"] == "the answer"
    assert (rec["model"], rec["model_version"], rec["tier"]) == (settings.models.main.id, "gpt-5.4-mini-2026-09-01", "main")
    assert (rec["input_tokens"], rec["output_tokens"]) == (120, 15)
    assert rec["latency_ms"] >= 0 and rec["steps"][0]["tier"] == "main" and rec["steps"][0]["ok"] is True
    assert rec["session_id"] == "sess-1" and rec["request_id"] == "req-1"
    assert rec["user_id"] == masked_user_id(EMAIL) and EMAIL not in rec["user_id"] and rec["ok"] is True


def test_prompt_and_completion_pass_through_redaction(settings, tmp_path):
    trace = new_trace(tmp_path, settings)
    messages = [SystemMessage("system"), HumanMessage(f"my email is {EMAIL} and my phone is {PHONE}")]
    call_llm(settings, trace, model=Model(text=f"I will write to {EMAIL}"), messages=messages)
    raw = trace.path.read_text()
    assert EMAIL not in raw and "98765" not in raw
    [rec] = events(trace, "llm_call")
    assert "j***@example.com" in rec["prompt"] and "[PHONE]" in rec["prompt"] and "j***@example.com" in rec["completion"]


def test_an_unknown_user_has_no_user_id_yet(settings, tmp_path):
    trace = new_trace(tmp_path, settings)
    call_llm(settings, trace)
    assert events(trace, "llm_call")[0]["user_id"] is None


def test_a_fallback_shows_each_step_and_a_total_latency(settings, tmp_path):
    trace = new_trace(tmp_path, settings)

    def builder(model_id, cfg):
        if model_id == settings.models.main.id:
            raise ValueError("main is down")
        return Model()

    call_llm(settings, trace, builder=builder)
    [rec] = events(trace, "llm_call")
    assert [(s["tier"], s["ok"]) for s in rec["steps"]] == [("main", False), ("light", True)]
    assert rec["tier"] == "light" and rec["fell_back"] is True and rec["latency_ms"] >= sum(s["latency_ms"] for s in rec["steps"]) - 1


def test_a_failed_call_is_logged_too(settings, tmp_path):
    from smartdesk_ai.llm.fallback import LLMUnavailableError

    trace = new_trace(tmp_path, settings)

    def always_down(model_id, cfg):
        raise ValueError("down")

    with pytest.raises(LLMUnavailableError):
        call_llm(settings, trace, builder=always_down)
    [rec] = events(trace, "llm_call")
    assert rec["ok"] is False and rec["completion"] == "" and len(rec["steps"]) == 2


def test_every_task_names_a_configured_prompt(settings):
    assert set(TASK_PROMPTS) == set(settings.tiers)
    assert set(TASK_PROMPTS.values()) <= set(settings.prompts)


# --- the whole of FR-H9 for one conversation ---------------------------------------------

@pytest.fixture
def conversation(settings, tmp_path):
    from pathlib import Path

    from smartdesk_ai.ops.health import HealthRegistry

    responder = build_responder(settings, environ={}, dotenv_path=None, data_dir=tmp_path, client=MockTicketClient())
    service = build_service(environ={}, dotenv_path=Path("/nonexistent/.env"), registry=HealthRegistry(), responder=responder)
    for text in ("How do I reset my forgotten password?", "My monitor keeps flickering",
                 f"{EMAIL}", f"Please call me on {PHONE}", "Yes, create it"):
        service.handle("sess-9", text)
    rows = [json.loads(line) for line in (tmp_path / TRACE_LOG).read_text().splitlines()]
    return rows, (tmp_path / TRACE_LOG).read_text()


def test_requests_replies_steps_tool_calls_and_llm_calls_are_all_logged(conversation):
    rows, _ = conversation
    kinds = {r["kind"] for r in rows}
    assert {"request", "reply", "agent_step", "tool_call", "tool_attempt", "llm_call"} <= kinds
    requests = [r for r in rows if r["kind"] == "request"]
    replies = [r for r in rows if r["kind"] == "reply"]
    assert len(requests) == len(replies) == 5
    assert {r["session_id"] for r in rows} == {"sess-9"}


def test_a_request_event_holds_the_guarded_text_not_the_raw_text(conversation):
    rows, _ = conversation
    texts = [r["text"] for r in rows if r["kind"] == "request"]
    assert any("[PHONE]" in t for t in texts) and not any("98765" in t for t in texts)


def test_steps_are_ordered_per_request(conversation):
    rows, _ = conversation
    first = next(r["request_id"] for r in rows if r["kind"] == "request")
    steps = [r["step"] for r in rows if r["kind"] == "agent_step" and r["request_id"] == first]
    assert steps[:3] == ["start", "route", "search_kb"] and [r["index"] for r in rows if r["kind"] == "agent_step"][:3] == [0, 1, 2]


def test_the_user_id_appears_once_the_email_is_known_and_never_as_an_email(conversation):
    rows, raw = conversation
    ids = {r["user_id"] for r in rows if r.get("user_id")}
    assert ids == {masked_user_id(EMAIL)}
    assert EMAIL not in raw and "98765" not in raw
    assert any(r["kind"] == "llm_call" and r["user_id"] for r in rows)


def test_every_line_is_valid_json_with_a_timestamp(conversation):
    rows, _ = conversation
    assert all("ts" in r and "kind" in r for r in rows)
