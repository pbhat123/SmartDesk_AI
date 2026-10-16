"""T3.2: LLM retry with backoff, Main/Light outage fallback, prompt-size check (FR-F2). Mocks only."""
from types import SimpleNamespace

import httpx
import openai
import pytest

from smartdesk_ai.llm.fallback import GRACEFUL_MESSAGE, invoke_with_fallback
from smartdesk_ai.llm.retry import call_with_retry


_REQ = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status(cls, code):
    return cls("boom", response=httpx.Response(code, request=_REQ), body=None)


TRANSIENT = [
    _status(openai.RateLimitError, 429),
    openai.APITimeoutError(request=_REQ),
    openai.APIConnectionError(request=_REQ),
    _status(openai.InternalServerError, 500),
]
PERMANENT = [
    _status(openai.AuthenticationError, 401),
    _status(openai.PermissionDeniedError, 403),
    _status(openai.BadRequestError, 400),
]


class FakeModel:
    def __init__(self, name, fail=False):
        self.name, self.fail, self.calls = name, fail, 0

    def invoke(self, prompt):
        self.calls += 1
        if self.fail:
            raise (self.fail if isinstance(self.fail, Exception) else TRANSIENT[2])
        return SimpleNamespace(content=f"reply from {self.name}")


class Models:
    """Builder that hands out one fake per model ID."""

    def __init__(self, settings, main_fails=False, light_fails=False):  # True or an exception
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
        raise TRANSIENT[1]

    with pytest.raises(openai.APITimeoutError):
        call_with_retry(boom, attempts=3, backoff=0.5, sleep=sleeps.append)
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]  # no wait after the last attempt


def test_retry_returns_on_first_success_without_sleeping():
    sleeps = []
    assert call_with_retry(lambda: 7, attempts=3, backoff=0.5, sleep=sleeps.append) == 7
    assert sleeps == []


@pytest.mark.parametrize("error", TRANSIENT, ids=lambda e: type(e).__name__)
def test_transient_errors_are_retried(error):
    calls = []

    def boom():
        calls.append(1)
        raise error

    with pytest.raises(type(error)):
        call_with_retry(boom, attempts=3, backoff=0, sleep=lambda s: None)
    assert len(calls) == 3


@pytest.mark.parametrize("error", PERMANENT, ids=lambda e: type(e).__name__)
def test_permanent_errors_are_not_retried(error):
    calls = []

    def boom():
        calls.append(1)
        raise error

    with pytest.raises(type(error)):
        call_with_retry(boom, attempts=3, backoff=0, sleep=lambda s: None)
    assert len(calls) == 1


@pytest.mark.parametrize("error", PERMANENT, ids=lambda e: type(e).__name__)
def test_permanent_error_tries_once_then_goes_to_fallback_tier(settings, error):
    models = Models(settings, main_fails=error)
    result, sleeps = run(settings, models)
    assert models.model("main").calls == 1 and sleeps == []
    assert (result.ok, result.tier, result.fell_back) == (True, "light", True)


@pytest.mark.parametrize("error", TRANSIENT, ids=lambda e: type(e).__name__)
def test_transient_error_uses_all_attempts_then_falls_back(settings, error):
    models = Models(settings, main_fails=error)
    result, _ = run(settings, models)
    assert models.model("main").calls == settings.llm.retry_attempts
    assert result.tier == "light" and result.fell_back


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


def _prompt_of(tokens):
    return "hello" + " hello" * (tokens - 1)  # one token each under o200k_base


def test_oversize_prompt_is_not_sent_to_light(settings):
    light = settings.models.light
    models = Models(settings, main_fails=True)
    # Over Light's window minus its max output, but well inside Main's window.
    result, _ = run(settings, models, prompt=_prompt_of(light.context_window - light.max_output + 5))
    assert result.ok is False and result.text == GRACEFUL_MESSAGE
    assert models.model("light").calls == 0


def test_prompt_that_fits_light_after_max_output_is_sent(settings):
    light = settings.models.light
    models = Models(settings, main_fails=True)
    result, _ = run(settings, models, prompt=_prompt_of(light.context_window - light.max_output - 50))
    assert result.ok is True and result.tier == "light"


def test_oversize_check_does_not_block_light_to_main(settings):
    light = settings.models.light
    models = Models(settings, light_fails=True)
    big = _prompt_of(light.context_window)
    result, _ = run(settings, models, task="CHITCHAT", prompt=big)
    assert result.ok is True and result.tier == "main"
