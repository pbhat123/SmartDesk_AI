"""T3.9: per-call metrics recorded by the executor (FR-G9). Fake clock, no real waiting."""
import json

import pytest

from smartdesk_ai.ops.metrics import MetricsRecorder
from smartdesk_ai.tools.errors import TransientToolError
from smartdesk_ai.tools.executor import Outcome, ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def build(settings, fn, metrics, clock):
    registry = ToolRegistry()
    registry.register("search_kb", fn)
    return ToolExecutor(registry, settings, sleep=clock.advance, metrics=metrics, clock=clock)


def test_failing_mock_run_records_outcome_attempts_and_latencies(settings):
    clock, metrics = Clock(), MetricsRecorder()

    def slow_failure(args):
        clock.advance(0.5)  # each attempt takes 0.5 s
        raise TransientToolError("503")

    result = build(settings, slow_failure, metrics, clock).execute("search_kb", {"query": "vpn"}, agent="it_worker")
    (rec,) = metrics.records
    assert result.outcome is Outcome.UNAVAILABLE
    assert (rec.tool, rec.agent, rec.outcome, rec.attempts) == ("search_kb", "it_worker", "unavailable", 3)
    assert rec.attempt_latencies_ms == [500.0, 500.0, 500.0]
    backoff = settings.tools.backoff_seconds
    assert rec.total_latency_ms == pytest.approx(1500 + 1000 * (backoff + 2 * backoff))


def test_success_record(settings):
    clock, metrics = Clock(), MetricsRecorder()

    def ok(args):
        clock.advance(0.2)
        return "x"

    build(settings, ok, metrics, clock).execute("search_kb", {"query": "vpn"}, agent="orchestrator")
    (rec,) = metrics.records
    assert (rec.outcome, rec.attempts, rec.attempt_latencies_ms) == ("success", 1, [200.0])
    assert rec.total_latency_ms == pytest.approx(200.0)


def test_validation_error_is_recorded_with_zero_attempts(settings):
    clock, metrics = Clock(), MetricsRecorder()
    build(settings, lambda a: 1, metrics, clock).execute("search_kb", {"query": ""}, agent="it_worker")
    (rec,) = metrics.records
    assert (rec.outcome, rec.attempts, rec.attempt_latencies_ms) == ("validation_error", 0, [])
    assert rec.error_code == "schema"


def test_unknown_tool_is_recorded_too(settings):
    metrics = MetricsRecorder()
    build(settings, lambda a: 1, metrics, Clock()).execute("nope", {}, agent="x")
    assert metrics.records[0].outcome == "error"


def test_records_hold_no_arguments_or_email(settings, tmp_path):
    log = tmp_path / "tool_calls.jsonl"
    clock, metrics = Clock(), MetricsRecorder(log)
    registry = ToolRegistry()
    registry.register("get_tickets", lambda a: [])
    ToolExecutor(registry, settings, metrics=metrics, clock=clock).execute(
        "get_tickets", {"email": "jane.roe@example.com"}, agent="ticket"
    )
    text = log.read_text()
    assert "jane.roe" not in text and "example.com" not in text
    assert json.loads(text.splitlines()[0])["tool"] == "get_tickets"


def test_executor_works_without_a_recorder(settings):
    registry = ToolRegistry()
    registry.register("search_kb", lambda a: 1)
    assert ToolExecutor(registry, settings).execute("search_kb", {"query": "x"}, agent="a").outcome is Outcome.SUCCESS
