"""T3.2: LLM retry with backoff, Main/Light outage fallback, prompt-size check (FR-F2). Mocks only."""
from types import SimpleNamespace

import pytest

from smartdesk_ai.llm.fallback import GRACEFUL_MESSAGE, invoke_with_fallback
from smartdesk_ai.llm.retry import call_with_retry


class FakeModel:
    def __init__(self, name, fail=False):
        self.name, self.fail, self.calls = name, fail, 0

    def invoke(self, prompt):
        self.calls += 1
        if self.fail:
            raise ConnectionError("down")
        return SimpleNamespace(content=f"reply from {self.name}")


class Models:
    """Builder that hands out one fake per model ID."""

    def __init__(self, settings, main_fails=False, light_fails=False):
        self.by_id = {
            settings.models.main.id: FakeModel("main", main_fails),
            settings.models.light.id: FakeModel("light", light_fails),
        }

    def __call__(self, model_id, cfg):
        return self.by_id[model_id]

    def model(self, name):
        return next(m for m in self.by_id.values() if m.name == name)


def run(settings, models, task="ORCHESTRATOR", prompt="hello", budget_level=0):
    sleeps = []
    result = invoke_with_fallback(
        task, prompt, settings, budget_level=budget_level, builder=models, sleep=sleeps.append
    )
    return result, sleeps


# --- retry ------------------------------------------------------------------

def test_retry_waits_with_doubling_backoff_then_raises():
    sleeps, calls = [], []

    def boom():
        calls.append(1)
        raise TimeoutError

    with pytest.raises(TimeoutError):
        call_with_retry(boom, attempts=3, backoff=0.5, sleep=sleeps.append)
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]  # no wait after the last attempt


def test_retry_returns_on_first_success_without_sleeping():
    sleeps = []
    assert call_with_retry(lambda: 7, attempts=3, backoff=0.5, sleep=sleeps.append) == 7
    assert sleeps == []


# --- fallback ---------------------------------------------------------------

def test_healthy_primary_is_used_with_no_fallback(settings):
    models = Models(settings)
    result, _ = run(settings, models)
    assert (result.ok, result.tier, result.fell_back) == (True, "main", False)
    assert result.text == "reply from main"
    assert models.model("light").calls == 0


def test_main_failure_falls_back_to_light(settings):
    models = Models(settings, main_fails=True)
    result, _ = run(settings, models)
    assert (result.ok, result.tier, result.fell_back) == (True, "light", True)
    assert models.model("main").calls == settings.llm.retry_attempts


def test_light_failure_falls_back_to_main(settings):
    models = Models(settings, light_fails=True)
    result, _ = run(settings, models, task="CHITCHAT")
    assert (result.ok, result.tier, result.fell_back) == (True, "main", True)


def test_both_failing_gives_graceful_message(settings):
    models = Models(settings, main_fails=True, light_fails=True)
    result, _ = run(settings, models)
    assert result.ok is False
    assert result.text == GRACEFUL_MESSAGE


def test_budget_level_one_never_falls_back_to_main(settings):
    models = Models(settings, light_fails=True)
    result, _ = run(settings, models, task="ORCHESTRATOR", budget_level=1)
    assert result.ok is False and result.text == GRACEFUL_MESSAGE
    assert models.model("main").calls == 0


def test_oversize_prompt_is_not_sent_to_light(settings):
    models = Models(settings, main_fails=True)
    # Fits Main (400k window) but not Light (128k window): ~4 chars per token.
    big = "x" * (settings.models.light.context_window * 4 + 400)
    result, _ = run(settings, models, prompt=big)
    assert result.ok is False and result.text == GRACEFUL_MESSAGE
    assert models.model("light").calls == 0


def test_oversize_check_does_not_block_light_to_main(settings):
    models = Models(settings, light_fails=True)
    big = "x" * (settings.models.light.context_window * 4 + 400)
    result, _ = run(settings, models, task="CHITCHAT", prompt=big)
    assert result.ok is True and result.tier == "main"
