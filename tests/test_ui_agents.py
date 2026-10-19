"""T7.9: the Monitoring tab shows cost and calls by agent, from the agent spans in the local trace (FR-I9, FR-H7, FR-G10).
No keys needed: the table is read from `trace.jsonl`."""
import json

import pytest

from smartdesk_ai.ui.monitoring import load_monitoring
from tests.ui_helpers import element_text, mock_context, monitoring_app


def span(agent, usd, calls=1, tiers=("main",), tokens=(100, 10), part_id=None, tool_calls=0):
    return {"kind": "agent_span", "ts": "2026-10-19T10:00:00+00:00", "agent": agent, "part_id": part_id,
            "tiers": list(tiers), "calls": calls, "tool_calls": tool_calls, "input_tokens": tokens[0],
            "output_tokens": tokens[1], "usd": usd, "latency_ms": 5.0, "session_id": "s", "request_id": "r"}


def write_trace(ctx, *events):
    ctx.data_dir.mkdir(parents=True, exist_ok=True)
    (ctx.data_dir / "trace.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def load(ctx):
    return load_monitoring(ctx.settings, ctx.data_dir, ctx.kb_dir, ctx.registry, results_dir=ctx.results_dir)


@pytest.fixture
def ctx(tmp_path):
    return mock_context(tmp_path)


def test_the_spans_are_summed_per_agent(ctx):
    write_trace(ctx, span("orchestrator", 0.002), span("orchestrator", 0.003), span("it_worker", 0.004, calls=2),
                span("hr_worker", 0.001, tiers=("light", "main")), {"kind": "llm_call", "agent": "orchestrator"},
                span("ticket_agent", 0.0005, tiers=("light",), tool_calls=1))
    rows = {r.agent: r for r in load(ctx).agents}
    assert list(rows) == ["orchestrator", "it_worker", "hr_worker", "ticket_agent"]
    assert rows["orchestrator"].spans == 2 and rows["orchestrator"].usd == pytest.approx(0.005)
    assert rows["it_worker"].calls == 2 and rows["hr_worker"].tiers == ["light", "main"]
    assert rows["ticket_agent"].tool_calls == 1 and rows["it_worker"].input_tokens == 100


def test_the_rows_follow_the_order_of_the_agents_of_fr_i1(ctx):
    write_trace(ctx, span("ticket_agent", 0.1, tiers=("light",)), span("synthesizer", 0.1), span("orchestrator", 0.1),
                span("hr_worker", 0.1), span("it_worker", 0.1))
    assert [r.agent for r in load(ctx).agents] == ["orchestrator", "it_worker", "hr_worker", "synthesizer", "ticket_agent"]


def test_no_trace_means_no_rows_and_no_crash(ctx):
    assert load(ctx).agents == []
    write_trace(ctx)
    assert load(ctx).agents == []


def test_a_damaged_line_is_skipped(ctx):
    ctx.data_dir.mkdir(parents=True, exist_ok=True)
    (ctx.data_dir / "trace.jsonl").write_text("not json\n" + json.dumps(span("it_worker", 0.01)) + "\n{\"kind\": \"agent_span\"}\n")
    assert [r.agent for r in load(ctx).agents] == ["it_worker"]


def test_the_page_has_a_cost_by_agent_table(ctx):
    write_trace(ctx, span("it_worker", 0.004), span("synthesizer", 0.002))
    shown = element_text(monitoring_app(ctx).main)
    assert "Cost by agent" in shown and "it_worker" in shown and "synthesizer" in shown


def test_the_page_says_so_when_there_are_no_spans_yet(ctx):
    shown = element_text(monitoring_app(ctx).main)
    assert "Cost by agent" in shown and "No agent spans yet" in shown
