"""T3.5: executor retry, max 3 attempts, transient only, then "unavailable" (FR-G4)."""
import pytest

from smartdesk_ai.tools.errors import TransientToolError
from smartdesk_ai.tools.executor import Outcome, ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry


class Flaky:
    """A tool function that raises the given errors in turn, then returns 'done'."""

    def __init__(self, *errors):
        self.errors, self.calls = list(errors), 0

    def __call__(self, args):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "done"


def build(settings, fn):
    registry = ToolRegistry()
    registry.register("search_kb", fn)
    sleeps = []
    return ToolExecutor(registry, settings, sleep=sleeps.append), sleeps


def run(executor, args=None):
    return executor.execute("search_kb", args or {"query": "vpn"}, agent="it_worker")


def test_success_first_try(settings):
    executor, sleeps = build(settings, Flaky())
    result = run(executor)
    assert (result.outcome, result.attempts, result.data) == (Outcome.SUCCESS, 1, "done")
    assert sleeps == []


@pytest.mark.parametrize("error", [TransientToolError("503"), ConnectionError("net"), TimeoutError("slow")])
def test_transient_failure_attempts_exactly_three_then_unavailable(settings, error):
    fn = Flaky(error, error, error, error)
    executor, sleeps = build(settings, fn)
    result = run(executor)
    assert fn.calls == settings.tools.max_attempts == 3
    assert (result.outcome, result.attempts) == (Outcome.UNAVAILABLE, 3)
    assert len(sleeps) == 2  # short backoff between attempts, none after the last


def test_transient_then_success_stops_retrying(settings):
    fn = Flaky(TransientToolError("blip"))
    executor, _ = build(settings, fn)
    result = run(executor)
    assert (result.outcome, result.attempts, fn.calls) == (Outcome.SUCCESS, 2, 2)


def test_bad_args_attempted_zero_times_and_not_retried(settings):
    fn = Flaky()
    executor, sleeps = build(settings, fn)
    result = run(executor, {"query": "", "domain": "FINANCE"})
    assert result.outcome is Outcome.VALIDATION_ERROR
    assert fn.calls == 0 and result.attempts == 0 and sleeps == []


def test_permanent_error_in_the_function_is_attempted_once(settings):
    fn = Flaky(ValueError("bad request"))
    executor, sleeps = build(settings, fn)
    result = run(executor)
    assert (result.outcome, result.attempts, fn.calls) == (Outcome.ERROR, 1, 1)
    assert sleeps == []


def test_unknown_tool_is_an_error_result_not_a_crash(settings):
    executor, _ = build(settings, Flaky())
    result = executor.execute("drop_tables", {}, agent="x")
    assert result.outcome is Outcome.ERROR and result.attempts == 0
