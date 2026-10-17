"""T5.12: cost tracking (FR-H7): token counts times the config price table, recorded per request, per task,
per agent, per model tier and per agent step (LLM call, embedding call, tool call). Mock models; the
mock-mode app counts tokens but prices nothing."""
import json
from types import SimpleNamespace

import pytest

from smartdesk_ai.agent.chat_service import ChatService
from smartdesk_ai.llm.fallback import FallbackChat
from smartdesk_ai.ops.cost import CostedEmbeddings, CostTracker
from smartdesk_ai.ops.runtime import RequestContext, Services, request_scope
from smartdesk_ai.rag.embeddings import HashingEmbeddings


class UsageModel:
    """A chat model whose reply carries provider-style usage metadata."""

    def __init__(self, usage=None, text="ok"):
        self.usage, self.text = usage, text

    def invoke(self, messages):
        reply = SimpleNamespace(content=self.text)
        if self.usage:
            reply.usage_metadata = self.usage
        return reply


def tracker(settings, **kw):
    return CostTracker(settings, priced=True, **kw)


def chat(settings, task, model):
    return FallbackChat(task, settings, builder=lambda m, c: model, sleep=lambda _: None)


def run_in_scope(services, ctx, fn):
    with request_scope(ctx, services):
        return fn()


def test_llm_cost_is_tokens_times_the_config_prices(settings):
    cost = tracker(settings)
    services = Services(cost=cost)
    ctx = RequestContext(session_id="s1", request_id="r1")
    usage = {"input_tokens": 1000, "output_tokens": 300, "input_token_details": {"cache_read": 200}}
    run_in_scope(services, ctx, lambda: chat(settings, "WORKER_ANSWER", UsageModel(usage)).invoke(["hi"]))
    [entry] = cost.entries
    price = settings.models.main.price
    expected = (800 * price.input + 200 * price.cached_input + 300 * price.output) / 1_000_000
    assert (entry.step, entry.tier, entry.model, entry.agent) == ("llm", "main", settings.models.main.id, "worker_answer")
    assert (entry.input_tokens, entry.cached_tokens, entry.output_tokens) == (1000, 200, 300)
    assert entry.usd == pytest.approx(expected) and expected == pytest.approx(0.001965)


def test_the_light_tier_uses_the_light_prices(settings):
    cost = tracker(settings)
    ctx = RequestContext(session_id="s1", request_id="r1")
    usage = {"input_tokens": 2000, "output_tokens": 500}
    run_in_scope(Services(cost=cost), ctx, lambda: chat(settings, "CHITCHAT", UsageModel(usage)).invoke(["hi"]))
    [entry] = cost.entries
    p = settings.models.light.price
    assert entry.tier == "light" and entry.usd == pytest.approx((2000 * p.input + 500 * p.output) / 1e6)


def test_without_provider_usage_the_tokens_are_counted_locally(settings):
    cost = tracker(settings)
    ctx = RequestContext(session_id="s1", request_id="r1")
    run_in_scope(Services(cost=cost), ctx, lambda: chat(settings, "WORKER_ANSWER", UsageModel(text="a b c d e")).invoke(
        [SimpleNamespace(content="one two three four")]))
    [entry] = cost.entries
    assert entry.input_tokens > 0 and entry.output_tokens > 0 and entry.usd > 0


def test_an_embedding_call_is_counted_with_the_embedding_price(settings):
    cost = tracker(settings)
    ctx = RequestContext(session_id="s1", request_id="r1")
    embeddings = CostedEmbeddings(HashingEmbeddings(), cost, settings)
    run_in_scope(Services(cost=cost), ctx, lambda: embeddings.embed_query("how do I reset my password"))
    [entry] = cost.entries
    assert entry.step == "embedding" and entry.tier is None and entry.input_tokens > 0
    assert entry.usd == pytest.approx(entry.input_tokens * settings.models.embedding.price.input / 1e6)
    assert len(embeddings.embed_documents(["a", "b"])) == 2  # documents pass through (ingest is not a request)


def test_tool_calls_are_recorded_as_free_steps(settings, tmp_path):
    from smartdesk_ai.tools.executor import ToolExecutor
    from smartdesk_ai.tools.registry import ToolRegistry

    cost = tracker(settings)
    registry = ToolRegistry()
    registry.register("search_kb", lambda args: [])
    ctx = RequestContext(session_id="s1", request_id="r1")
    executor = ToolExecutor(registry, settings)
    run_in_scope(Services(cost=cost), ctx, lambda: executor.execute("search_kb", {"query": "vpn"}, agent="it_worker"))
    [entry] = cost.entries
    assert (entry.step, entry.agent, entry.usd, entry.tool) == ("tool", "it_worker", 0.0, "search_kb")


def test_totals_by_request_task_agent_tier_and_step(settings):
    cost = tracker(settings)
    services = Services(cost=cost)
    one = RequestContext(session_id="s1", request_id="r1", task="kb_answer")
    two = RequestContext(session_id="s1", request_id="r2", task="ticket_flow")
    big, small = {"input_tokens": 1000, "output_tokens": 100}, {"input_tokens": 100, "output_tokens": 10}
    run_in_scope(services, one, lambda: chat(settings, "WORKER_ANSWER", UsageModel(big)).invoke(["x"]))
    run_in_scope(services, one, lambda: chat(settings, "CHITCHAT", UsageModel(small)).invoke(["x"]))
    run_in_scope(services, two, lambda: chat(settings, "TICKET_DRAFT", UsageModel(small)).invoke(["x"]))
    by = cost.totals
    assert set(by("request")) == {"r1", "r2"} and set(by("task")) == {"kb_answer", "ticket_flow"}
    assert set(by("agent")) == {"worker_answer", "chitchat", "ticket_draft"}
    assert set(by("tier")) == {"main", "light"} and set(by("step")) == {"llm"}
    assert sum(by("request").values()) == pytest.approx(cost.total) == pytest.approx(sum(by("tier").values()))
    assert by("request")["r1"] > by("request")["r2"]


def test_eval_cost_is_tracked_under_the_task_eval(settings):
    cost = tracker(settings)
    services = Services(cost=cost)
    ctx = RequestContext(session_id="eval", request_id="e1", task="eval")
    run_in_scope(services, ctx, lambda: chat(settings, "GROUNDING_JUDGE", UsageModel({"input_tokens": 50, "output_tokens": 5})).invoke(["x"]))
    services.finish_request(ctx, goal="kb_answer")  # a goal found later never overwrites an explicit task
    assert set(cost.totals("task")) == {"eval"}


def test_the_goal_of_a_chat_request_names_its_task(settings):
    cost = tracker(settings)
    services = Services(cost=cost)

    def responder(session_id, text):
        chat(settings, "WORKER_ANSWER", UsageModel({"input_tokens": 100, "output_tokens": 10})).invoke(["x"])
        from smartdesk_ai.agent.chat_service import Turn
        return Turn(text="answer", debug={"steps": ["start", "route", "search_kb", "answer", "reply"]})

    reply = ChatService(settings, responder=responder, services=services).handle("s1", "how do I reset my password")
    assert set(cost.totals("task")) == {"kb_answer"} and reply.debug["cost_usd"] == pytest.approx(cost.total)


@pytest.mark.parametrize("steps,goal", [
    (["start", "route", "t_collect"], "ticket_flow"), (["start", "route", "s_lookup"], "ticket_status"),
    (["start", "route", "chitchat"], "smalltalk"), (["start", "search_kb", "answer"], "kb_answer"),
])
def test_goal_from_steps(steps, goal):
    from smartdesk_ai.ops.runtime import goal_from_steps

    assert goal_from_steps(steps) == goal


def test_the_cost_log_holds_numbers_only(settings, tmp_path):
    path = tmp_path / "cost.jsonl"
    cost = tracker(settings, path=path)
    ctx = RequestContext(session_id="s1", request_id="r1")
    run_in_scope(Services(cost=cost), ctx, lambda: chat(settings, "WORKER_ANSWER", UsageModel(
        {"input_tokens": 10, "output_tokens": 1}, text="jane.roe@example.com")).invoke(["call +91 98765 43210"]))
    services = Services(cost=cost)
    services.finish_request(ctx, goal="kb_answer")
    raw = path.read_text()
    assert "jane.roe" not in raw and "98765" not in raw
    row = json.loads(raw.splitlines()[0])
    assert row["task"] == "kb_answer" and row["request_id"] == "r1" and "ts" in row


def test_a_mock_mode_tracker_counts_tokens_but_prices_nothing(settings):
    cost = CostTracker(settings, priced=False)
    ctx = RequestContext(session_id="s1", request_id="r1")
    run_in_scope(Services(cost=cost), ctx, lambda: chat(settings, "WORKER_ANSWER", UsageModel(
        {"input_tokens": 1000, "output_tokens": 100})).invoke(["x"]))
    [entry] = cost.entries
    assert entry.input_tokens == 1000 and entry.usd == 0.0


def test_the_app_records_costs_per_request_in_mock_mode(settings, tmp_path):
    from pathlib import Path

    from smartdesk_ai.agent.chat_service import build_service
    from smartdesk_ai.agent.factory import COST_LOG, build_responder
    from smartdesk_ai.ops.health import HealthRegistry
    from smartdesk_ai.tickets import MockTicketClient

    responder = build_responder(settings, environ={}, dotenv_path=None, data_dir=tmp_path, client=MockTicketClient())
    service = build_service(environ={}, dotenv_path=Path("/nonexistent/.env"), registry=HealthRegistry(),
                            responder=responder)
    reply = service.handle("s1", "How do I reset my forgotten password?")
    rows = [json.loads(line) for line in (tmp_path / COST_LOG).read_text().splitlines()]
    assert {r["step"] for r in rows} >= {"llm", "embedding", "tool"} and {r["task"] for r in rows} == {"kb_answer"}
    assert all(r["usd"] == 0.0 for r in rows) and reply.debug["cost_usd"] == 0.0  # mock mode prices nothing
    assert len({r["request_id"] for r in rows}) == 1
