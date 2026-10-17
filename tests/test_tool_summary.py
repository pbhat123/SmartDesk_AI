"""T5.10: per-tool summary from the call records: count, success rate, failures by type, retry rate and
latency p50 and p95 (FR-G10, data layer). Fake clock, no real waiting."""
import math

import pytest

from smartdesk_ai.ops.metrics import MetricsRecorder, ToolCallRecord, load_records, percentile, summarize_tools
from smartdesk_ai.tools.errors import TransientToolError
from smartdesk_ai.tools.executor import ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry

EMAIL = {"email": "jane.roe@example.com"}


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Script:
    """A tool function that follows one scripted behaviour per attempt: (seconds taken, error or None)."""

    def __init__(self, clock, *steps):
        self.clock, self.steps = clock, list(steps)

    def __call__(self, args):
        seconds, error = self.steps.pop(0)
        self.clock.advance(seconds)
        if error:
            raise error
        return []


def test_percentile_is_nearest_rank():
    assert percentile([], 0.5) == 0.0
    assert percentile([7.0], 0.95) == 7.0
    assert percentile(list(range(1, 11)), 0.5) == 5 and percentile(list(range(1, 11)), 0.95) == 10
    assert percentile([30, 10, 20], 0.5) == 20  # order of the input does not matter


@pytest.fixture
def scripted(settings):
    """search_kb: ten successes of 10 ms to 100 ms. get_tickets: success, retried success, unavailable,
    a rejected call (no email) and a permanent error."""
    clock, metrics = Clock(), MetricsRecorder()
    transient, boom = TransientToolError("503"), ValueError("boom")
    registry = ToolRegistry()
    registry.register("search_kb", Script(clock, *[(0.01 * n, None) for n in range(1, 11)]))
    registry.register("get_tickets", Script(
        clock,
        (0.020, None),                                                    # call 1: fine
        (0.010, transient), (0.010, None),                                # call 2: one retry
        (0.010, transient), (0.010, transient), (0.010, transient),       # call 3: gives up
        (0.005, boom),                                                    # call 5: permanent error
    ))
    ex = ToolExecutor(registry, settings, sleep=clock.advance, metrics=metrics, clock=clock)
    for _ in range(10):
        ex.execute("search_kb", {"query": "vpn"}, agent="it_worker")
    for args in (EMAIL, EMAIL, EMAIL, {}, EMAIL):
        ex.execute("get_tickets", args, agent="status_flow")
    return metrics, settings.tools.backoff_seconds


def by_tool(summaries):
    return {s.tool: s for s in summaries}


def test_a_scripted_run_with_one_failing_tool_gives_the_right_summary(scripted):
    metrics, backoff = scripted
    summary = by_tool(summarize_tools(metrics.records))
    ok, bad = summary["search_kb"], summary["get_tickets"]
    assert (ok.calls, ok.success_rate, ok.failures, ok.retry_rate) == (10, 1.0, {}, 0.0)
    assert ok.p50_ms == pytest.approx(50) and ok.p95_ms == pytest.approx(100)
    assert bad.calls == 5 and bad.success_rate == pytest.approx(0.4)
    assert bad.failures == {"unavailable": 1, "validation_error": 1, "error": 1}
    assert bad.retry_rate == pytest.approx(0.4)  # the retried success and the unavailable call
    retried_ms = 20 + 1000 * backoff
    unavailable_ms = 30 + 1000 * (backoff + 2 * backoff)
    assert bad.p50_ms == pytest.approx(20)  # sorted: 0, 5, 20, retried, unavailable
    assert bad.p95_ms == pytest.approx(unavailable_ms) and retried_ms < unavailable_ms


def test_summaries_are_sorted_by_tool_name(scripted):
    assert [s.tool for s in summarize_tools(scripted[0].records)] == ["get_tickets", "search_kb"]


def test_an_empty_log_gives_an_empty_summary():
    assert summarize_tools([]) == []


def test_the_recorder_summarises_itself(scripted):
    metrics, _ = scripted
    assert metrics.summary() == summarize_tools(metrics.records)


def test_the_summary_can_be_built_from_the_local_log_file(settings, tmp_path):
    path = tmp_path / "tool_metrics.jsonl"
    clock = Clock()
    registry = ToolRegistry()
    registry.register("search_kb", Script(clock, (0.01, None), (0.01, TransientToolError("x")), (0.01, None)))
    ex = ToolExecutor(registry, settings, sleep=clock.advance, metrics=MetricsRecorder(path), clock=clock)
    ex.execute("search_kb", {"query": "a"}, agent="a")
    ex.execute("search_kb", {"query": "b"}, agent="a")
    records = load_records(path)
    assert [type(r) for r in records] == [ToolCallRecord] * 2
    [row] = summarize_tools(records)
    assert row.calls == 2 and row.retry_rate == 0.5 and row.success_rate == 1.0
    assert math.isclose(row.p50_ms, 10.0)


def test_load_records_of_a_missing_file_is_empty(tmp_path):
    assert load_records(tmp_path / "none.jsonl") == []


def test_the_app_writes_the_tool_log_the_monitoring_tab_reads(settings, tmp_path):
    from smartdesk_ai.agent.factory import TOOL_METRICS_LOG, build_responder
    from smartdesk_ai.tickets import MockTicketClient

    respond = build_responder(settings, environ={}, dotenv_path=None, data_dir=tmp_path, client=MockTicketClient())
    respond("s1", "How do I reset my forgotten password?")
    [row] = summarize_tools(load_records(tmp_path / TOOL_METRICS_LOG))
    assert row.tool == "search_kb" and row.calls == 1 and row.success_rate == 1.0
