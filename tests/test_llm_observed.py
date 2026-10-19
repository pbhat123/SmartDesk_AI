"""Every LLM call is observed (FR-H7, H8, H9, F2): cost entry, trace record, budget gate, retry and Main/Light
fallback. The intent classifier and the tool picker used to call the model directly and skipped all of them."""
import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest

from smartdesk_ai.agent.chat_service import build_service
from smartdesk_ai.agent.factory import build_responder
from smartdesk_ai.agent.intents import ClassifierError, make_classifier
from smartdesk_ai.agent.router import ToolPickError, make_tool_picker
from smartdesk_ai.llm import fallback as fallback_module
from smartdesk_ai.llm import mock as mock_models
from smartdesk_ai.ops.budget import BudgetController, SpendStore
from smartdesk_ai.ops.cost import CostTracker
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.ops.runtime import RequestContext, Services, request_scope
from smartdesk_ai.rag.ingest import ROOT
from smartdesk_ai.tickets import MockTicketClient
from tests.pii_helpers import has_phone

PACKAGE = ROOT / "smartdesk_ai"
# Only these two modules may obtain a chat model: factory.py defines the builders, fallback.py is the one caller.
MAY_BUILD_MODELS = {"smartdesk_ai/llm/factory.py", "smartdesk_ai/llm/fallback.py"}
MODEL_CONSTRUCTORS = {"get_llm", "get_llm_for_tier", "openai_builder", "ChatOpenAI", "init_chat_model"}


def test_no_module_outside_the_fallback_wrapper_builds_or_calls_a_chat_model():
    """Fails when new code gets a model with `get_llm(...)`, `ChatOpenAI(...)` and so on instead of `FallbackChat`:
    such a call would skip the cost tracker, the trace, the budget gate and the retry rules."""
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        relative = str(path.relative_to(ROOT))
        if relative in MAY_BUILD_MODELS:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
                if name in MODEL_CONSTRUCTORS:
                    offenders.append(f"{relative}:{node.lineno} calls {name}()")
    assert offenders == []


def counting(cls, counter):
    """The mock model class, counting every model call including calls on a tool-bound model."""

    class Counting(cls):
        def invoke(self, messages):
            counter.append(cls.__name__)
            return super().invoke(messages)

        if hasattr(cls, "bind_tools"):
            def bind_tools(self, tools):
                bound = super().bind_tools(tools)
                return SimpleNamespace(invoke=lambda messages: (counter.append(cls.__name__ + ".tools"), bound.invoke(messages))[1])

    return Counting


def test_every_model_call_in_a_whole_conversation_has_a_cost_entry_and_a_trace_record(settings, tmp_path, monkeypatch):
    calls: list[str] = []
    from smartdesk_ai.agent import factory
    from smartdesk_ai.guardrails import input_guard

    for name in ("MockAnswerModel", "MockRoutingModel", "MockDraftModel", "MockConfirmModel", "MockChitchatModel",
                 "MockGroundingModel", "MockCompactionModel", "MockInjectionModel"):
        monkeypatch.setattr(factory, name, counting(getattr(mock_models, name), calls), raising=False)
    monkeypatch.setattr(input_guard, "MockInjectionModel", counting(mock_models.MockInjectionModel, calls))
    responder = build_responder(settings, environ={}, dotenv_path=None, data_dir=tmp_path, client=MockTicketClient())
    service = build_service(environ={}, dotenv_path=Path("/nonexistent/.env"), registry=HealthRegistry(),
                            responder=responder)
    for text in ("How do I reset my forgotten password?", "hello there", "I want to raise a ticket for my broken monitor",
                 "yes", "jane.roe@example.com", "what is the status of my tickets", "the first one"):
        service.handle("s-all", text)
    records = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text().splitlines()]
    traced = [r for r in records if r.get("kind") == "llm_call"]
    costed = [json.loads(l) for l in (tmp_path / "cost.jsonl").read_text().splitlines()]
    costed = [c for c in costed if c["step"] == "llm"]
    assert len(calls) > 7  # the conversation really exercised several tasks
    assert len(costed) == len(calls) == len(traced), (len(costed), len(calls), len(traced))
    assert {"orchestrator", "it_worker"} <= {c["agent"] for c in costed}


def scope(settings, tmp_path, *, trace=None, budget=None):
    cost = CostTracker(settings, priced=False)
    services = Services(cost=cost, trace=trace, budget=budget)
    return cost, services, RequestContext(session_id="s1", request_id="r1")


ROUTING_JSON = json.dumps({"intent": "kb_query", "confidence": 0.9, "domain": "IT"})


class Model:
    def __init__(self, text=ROUTING_JSON, fail=None):
        self.text, self.fail, self.calls = text, fail, 0

    def invoke(self, messages):
        self.calls += 1
        if self.fail and self.calls <= self.fail[0]:
            raise self.fail[1]
        return SimpleNamespace(content=self.text, tool_calls=[{"name": "search_kb", "args": {"domain": "IT"}}])

    def bind_tools(self, tools):
        return self


def test_classifier_and_tool_picker_calls_reach_the_cost_tracker_and_the_trace_log(settings, tmp_path):
    seen = []
    trace = SimpleNamespace(llm_record=lambda ctx, agent, call: seen.append((agent, call.task, call.prompt)))
    cost, services, ctx = scope(settings, tmp_path, trace=trace)
    model = Model()
    classify = make_classifier(settings, builder=lambda m, c: model)
    pick = make_tool_picker(settings, builder=lambda m, c: model)
    with request_scope(ctx, services):
        classify("How do I reset my password?")
        pick("How do I reset my password?")
    assert [e.agent for e in cost.entries] == ["orchestrator", "tool_select"]
    assert [(a, t) for a, t, _ in seen] == [("orchestrator", "ORCHESTRATOR"), ("tool_select", "TOOL_SELECT")]
    assert all(entry.tier for entry in cost.entries) and cost.entries[0].tier == "main" and cost.entries[1].tier == "light"


def test_the_trace_log_masks_the_classifier_prompt(settings, tmp_path):
    from smartdesk_ai.guardrails.pii import PiiRedactor
    from smartdesk_ai.ops.trace_log import TraceLog, make_masker

    trace = TraceLog(tmp_path / "trace.jsonl", make_masker(PiiRedactor(None, settings.guardrails.pii_min_score)),
                     prompt_versions=settings.prompts)
    cost, services, ctx = scope(settings, tmp_path, trace=trace)
    classify = make_classifier(settings, builder=lambda m, c: Model())
    with request_scope(ctx, services):
        classify("my email is jane.roe@example.com and my phone is +91 98765 43210, vpn is down")
    text = (tmp_path / "trace.jsonl").read_text()
    assert "orchestrator" in text and "jane.roe@example.com" not in text and not has_phone(text)


def test_the_budget_gate_applies_to_the_classifier_and_the_tool_picker(settings, tmp_path):
    store = SpendStore(tmp_path / "ops.db")
    budget = BudgetController(settings, store)
    store.add(budget._today(), settings.budget.cutoff_2_usd + 1)
    cost, services, ctx = scope(settings, tmp_path, budget=budget)
    model = Model()
    with request_scope(ctx, services):
        with pytest.raises(ClassifierError):
            make_classifier(settings, builder=lambda m, c: model)("hello")
        with pytest.raises(ToolPickError):
            make_tool_picker(settings, builder=lambda m, c: model)("hello")
    assert model.calls == 0 and cost.entries == []  # cutoff 2: no model call at all (FR-H8)


def test_at_cutoff_1_the_classifier_uses_the_light_model(settings, tmp_path):
    store = SpendStore(tmp_path / "ops.db")
    budget = BudgetController(settings, store)
    store.add(budget._today(), settings.budget.cutoff_1_usd + 0.5)
    cost, services, ctx = scope(settings, tmp_path, budget=budget)
    used = []
    classify = make_classifier(settings, builder=lambda model_id, cfg: used.append(model_id) or Model())
    with request_scope(ctx, services):
        classify("hello")
    assert used == [settings.models.light.id] and cost.entries[0].tier == "light"


def rate_limit():
    return openai.RateLimitError("slow down", response=httpx.Response(429, request=httpx.Request("POST", "http://x")), body=None)


def test_a_transient_error_is_retried_for_the_classifier_and_the_tool_picker(settings, monkeypatch):
    monkeypatch.setattr(fallback_module.time, "sleep", lambda s: None)
    model = Model(fail=(1, rate_limit()))
    assert make_classifier(settings, builder=lambda m, c: model)("hello").intent == "kb_query" and model.calls == 2
    model = Model(fail=(2, openai.APIConnectionError(request=httpx.Request("POST", "http://x"))))
    assert make_tool_picker(settings, builder=lambda m, c: model)("hello").tool == "search_kb" and model.calls == 3


def test_a_permanent_error_is_not_retried_and_main_falls_back_to_light(settings, monkeypatch):
    monkeypatch.setattr(fallback_module.time, "sleep", lambda s: None)
    broken, light = Model(fail=(99, ValueError("bad request"))), Model()
    models = {settings.models.main.id: broken, settings.models.light.id: light}
    classify = make_classifier(settings, builder=lambda model_id, cfg: models[model_id])
    assert classify("hello").intent == "kb_query"
    assert broken.calls == 1 and light.calls == 1  # no retry of a permanent error, then the Light model answers


def test_the_tool_picker_returns_the_tool_calls_of_the_model(settings):
    pick = make_tool_picker(settings, builder=lambda m, c: Model())
    result = pick("my vpn is down")
    assert (result.tool, result.domain) == ("search_kb", "IT")


def test_rate_limits_count_questions_not_model_calls(settings):
    """The rate limiter counts questions, ticket creations and lookups per session (FR-H13), so it applies once per
    message in ChatService and needs no hook in the model wrapper."""
    text = (PACKAGE / "agent" / "chat_service.py").read_text()
    assert re.search(r"limiter|rate", text, re.I) and "limiter" not in (PACKAGE / "llm" / "fallback.py").read_text()
