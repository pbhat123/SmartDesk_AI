"""T5.13: daily budget cutoffs (FR-H8): $5 moves every task to Light-Model, $10 stops all LLM calls with a
graceful message, the counter resets each day in the configured timezone, and evals can bypass. Mocked
spend and a fake clock: no real model is called."""
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from smartdesk_ai.agent.chat_service import ChatService
from smartdesk_ai.llm.fallback import SERVICE_UNAVAILABLE_MESSAGE, FallbackChat, LLMUnavailableError
from smartdesk_ai.ops.budget import BudgetController, SpendStore
from smartdesk_ai.ops.cost import CostTracker
from smartdesk_ai.ops.runtime import RequestContext, Services, request_scope

NY = ZoneInfo("America/New_York")


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def controller(settings, store=None, clock=None):
    clock = clock or Clock(datetime(2026, 10, 17, 12, 0, tzinfo=NY))
    return BudgetController(settings, store or SpendStore(":memory:"), clock=clock), clock


class Models:
    """A builder that remembers which model IDs were asked for."""

    def __init__(self):
        self.ids = []

    def __call__(self, model_id, cfg):
        self.ids.append(model_id)
        return type("M", (), {"invoke": lambda self, messages: type("R", (), {"content": "ok"})()})()


def test_below_five_dollars_nothing_changes(settings):
    budget, _ = controller(settings)
    budget.add(4.99, "kb_answer")
    assert budget.level() == 0 and budget.spend_today() == pytest.approx(4.99)


def test_crossing_five_dollars_switches_every_task_to_light_and_logs(settings, caplog):
    budget, _ = controller(settings)
    models = Models()
    with caplog.at_level(logging.WARNING):
        budget.add(4.99, "kb_answer")
        budget.add(0.02, "kb_answer")
    assert budget.level() == 1 and "cutoff 1" in caplog.text
    ctx = RequestContext(session_id="s", request_id="r")
    with request_scope(ctx, Services(budget=budget)):
        for task in ("WORKER_ANSWER", "ORCHESTRATOR", "GROUNDING_JUDGE", "CHITCHAT"):
            FallbackChat(task, settings, builder=models, sleep=lambda _: None).invoke(["x"])
    assert set(models.ids) == {settings.models.light.id}  # even the Main-Model tasks


def test_before_the_cutoff_main_tasks_still_use_main(settings):
    budget, _ = controller(settings)
    models = Models()
    with request_scope(RequestContext(session_id="s", request_id="r"), Services(budget=budget)):
        FallbackChat("WORKER_ANSWER", settings, builder=models, sleep=lambda _: None).invoke(["x"])
    assert models.ids == [settings.models.main.id]


def test_crossing_ten_dollars_stops_llm_calls_and_logs(settings, caplog):
    budget, _ = controller(settings)
    models = Models()
    with caplog.at_level(logging.WARNING):
        budget.add(10.0, "kb_answer")
    assert budget.level() == 2 and "cutoff 2" in caplog.text
    with request_scope(RequestContext(session_id="s", request_id="r"), Services(budget=budget)):
        with pytest.raises(LLMUnavailableError) as stopped:
            FallbackChat("CHITCHAT", settings, builder=models, sleep=lambda _: None).invoke(["x"])
    assert models.ids == [] and stopped.value.message == SERVICE_UNAVAILABLE_MESSAGE


def test_each_cutoff_is_logged_once_per_day(settings, caplog):
    budget, _ = controller(settings)
    with caplog.at_level(logging.WARNING):
        for _ in range(6):
            budget.add(2.0, "kb_answer")
    assert caplog.text.count("cutoff 1") == 1 and caplog.text.count("cutoff 2") == 1


def test_at_level_two_the_chat_service_answers_unavailable_without_calling_anything(settings):
    budget, _ = controller(settings)
    budget.add(10.0, "kb_answer")
    called = []
    service = ChatService(settings, responder=lambda sid, text: called.append(text) or "ran",
                          services=Services(budget=budget))
    reply = service.handle("s1", "How do I reset my password?")
    assert called == [] and reply.text == SERVICE_UNAVAILABLE_MESSAGE and reply.debug["edge"] == "budget_cutoff_2"


def test_the_counter_resets_on_a_new_day_in_the_configured_timezone(settings):
    budget, clock = controller(settings)
    budget.add(11.0, "kb_answer")
    assert budget.level() == 2
    clock.now = datetime(2026, 10, 17, 23, 59, tzinfo=NY)  # same day in New York...
    assert budget.level() == 2
    clock.now = clock.now.astimezone(timezone.utc) + timedelta(minutes=2)  # ...past midnight there
    assert budget.level() == 0 and budget.spend_today() == 0.0
    budget.add(5.5, "kb_answer")
    assert budget.level() == 1


def test_the_day_follows_the_config_timezone_not_utc(settings):
    budget, clock = controller(settings)
    clock.now = datetime(2026, 10, 18, 2, 0, tzinfo=timezone.utc)  # still 17 Oct, 22:00 in New York
    budget.add(6.0, "kb_answer")
    clock.now = datetime(2026, 10, 18, 3, 30, tzinfo=timezone.utc)  # 23:30 in New York: same day
    assert budget.level() == 1
    clock.now = datetime(2026, 10, 18, 4, 30, tzinfo=timezone.utc)  # 00:30 on the 18th in New York
    assert budget.level() == 0


def test_spend_survives_a_restart(settings, tmp_path):
    clock = Clock(datetime(2026, 10, 17, 12, 0, tzinfo=NY))
    first = BudgetController(settings, SpendStore(tmp_path / "ops.db"), clock=clock)
    first.add(5.5, "kb_answer")
    second = BudgetController(settings, SpendStore(tmp_path / "ops.db"), clock=clock)
    assert second.level() == 1 and second.spend_today() == pytest.approx(5.5)


def with_bypass(settings, on=True):
    return settings.model_copy(update={"budget": settings.budget.model_copy(update={"eval_bypass": on})})


def test_the_eval_bypass_flag_keeps_eval_runs_going_and_out_of_the_counter(settings):
    cfg = with_bypass(settings)
    budget, _ = controller(cfg)
    budget.add(11.0, "kb_answer")  # the real service is over the limit
    budget.add(50.0, "eval")  # eval spend is tracked elsewhere and does not count
    assert budget.spend_today() == pytest.approx(11.0)
    eval_ctx = RequestContext(session_id="eval", request_id="e", task="eval")
    chat_ctx = RequestContext(session_id="s", request_id="r")
    models = Models()
    services = Services(budget=budget)
    with request_scope(eval_ctx, services):
        FallbackChat("WORKER_ANSWER", cfg, builder=models, sleep=lambda _: None).invoke(["x"])
    assert models.ids == [cfg.models.main.id]  # no cutoff for the eval run
    with request_scope(chat_ctx, services), pytest.raises(LLMUnavailableError):
        FallbackChat("WORKER_ANSWER", cfg, builder=models, sleep=lambda _: None).invoke(["x"])


def test_without_the_flag_eval_runs_are_stopped_like_everything_else(settings):
    budget, _ = controller(with_bypass(settings, on=False))
    budget.add(10.5, "eval")
    assert budget.level() == 2
    with request_scope(RequestContext(session_id="e", request_id="e", task="eval"), Services(budget=budget)):
        with pytest.raises(LLMUnavailableError):
            FallbackChat("WORKER_ANSWER", settings, builder=Models(), sleep=lambda _: None).invoke(["x"])


def test_priced_llm_cost_feeds_the_daily_counter(settings):
    budget, _ = controller(settings)
    cost = CostTracker(settings, priced=True, on_spend=budget.add)
    services = Services(cost=cost, budget=budget)
    usage_model = type("M", (), {"invoke": lambda self, m: type("R", (), {
        "content": "x", "usage_metadata": {"input_tokens": 0, "output_tokens": 1_200_000}})()})()
    with request_scope(RequestContext(session_id="s", request_id="r"), services):
        FallbackChat("WORKER_ANSWER", settings, builder=lambda m, c: usage_model, sleep=lambda _: None).invoke(["x"])
    assert budget.spend_today() == pytest.approx(1_200_000 * settings.models.main.price.output / 1e6)
    assert budget.level() == 1


def test_compaction_measures_against_the_light_window_once_the_budget_switched(settings):
    from smartdesk_ai.ops.runtime import current_budget_level

    budget, _ = controller(settings)
    assert current_budget_level() == 0  # outside a request scope
    budget.add(6.0, "kb_answer")
    with request_scope(RequestContext(session_id="s", request_id="r"), Services(budget=budget)):
        assert current_budget_level() == 1
