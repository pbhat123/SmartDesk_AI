"""Monitoring review (2026-10-19): Tool stats say which time window they cover, the last error time per tool, and can
be filtered (since app start / today / all). The human-check counters get the same kind of filter (this session / today /
all) and split approvals into "ticket created" and "creation failed". The logs stay cumulative: nothing is deleted."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from smartdesk_ai.ops.hitl_counters import HitlCounters, HitlCounts, load_counts
from smartdesk_ai.ops.metrics import MetricsRecorder, ToolCallRecord, load_records
from smartdesk_ai.tickets import MockTicketClient
from smartdesk_ai.ui.monitoring import load_monitoring
from tests.ui_helpers import element_text, mock_context, monitoring_app

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)
EMAIL = "priya.k@smartdesk.example"


def iso(delta_hours: float) -> str:
    return (NOW + timedelta(hours=delta_hours)).isoformat(timespec="seconds")


def rec(tool="get_tickets", outcome="success", ts=None, code=None):
    return ToolCallRecord(tool=tool, agent="a", outcome=outcome, error_code=code, attempts=1,
                          attempt_latencies_ms=[1.0], attempt_outcomes=["success"], total_latency_ms=1.0, ts=ts)


def write_metrics(path, records):
    path.write_text("".join(r.model_dump_json() + "\n" for r in records))


def load(ctx, **kw):
    return load_monitoring(ctx.settings, ctx.data_dir, ctx.kb_dir, ctx.registry, results_dir=ctx.results_dir, **kw)


# --- tool stats: timestamps, windows, last error -------------------------------------------------------------------

def test_new_records_are_stamped_with_a_utc_time(tmp_path):
    recorder = MetricsRecorder(tmp_path / "m.jsonl")
    recorder.record(rec())
    (row,) = load_records(tmp_path / "m.jsonl")
    assert row.ts is not None and datetime.fromisoformat(row.ts).utcoffset() == timedelta(0)


def test_an_old_log_without_times_still_loads(tmp_path):
    (tmp_path / "m.jsonl").write_text(rec().model_dump_json(exclude={"ts"}) + "\n")
    assert load_records(tmp_path / "m.jsonl")[0].ts is None


def test_windows_select_by_time(tmp_path):
    ctx = mock_context(tmp_path)
    write_metrics(tmp_path / "tool_metrics.jsonl", [
        rec(ts=iso(-72), outcome="error", code="error"),   # three days ago
        rec(ts=iso(-6)),                                      # earlier today, before the app started
        rec(ts=iso(-1)),                                      # after the app started
    ])
    started = NOW - timedelta(hours=2)
    everything = load(ctx, window="all", now=NOW, app_started=started)
    today = load(ctx, window="today", now=NOW, app_started=started)
    since_start = load(ctx, window="app", now=NOW, app_started=started)
    assert [r.calls for r in everything.tools] == [3] and [r.calls for r in today.tools] == [2]
    assert [r.calls for r in since_start.tools] == [1]
    assert since_start.tools[0].failures == 0 and everything.tools[0].failures == 1  # the old error is not "now"


def test_the_window_says_what_it_covers(tmp_path):
    ctx = mock_context(tmp_path)
    write_metrics(tmp_path / "tool_metrics.jsonl", [rec(ts=iso(-30)), rec(ts=iso(-1))])
    info = load(ctx, window="all", now=NOW, app_started=NOW).tool_window
    assert (info.calls, info.first_ts, info.last_ts) == (2, iso(-30), iso(-1))
    assert info.since == iso(-30) and info.label == "All time"
    today = load(ctx, window="today", now=NOW, app_started=NOW).tool_window
    assert today.since == "2026-10-19T00:00:00+00:00" and today.label == "Today (UTC)"


def test_the_last_error_time_and_kind_per_tool(tmp_path):
    ctx = mock_context(tmp_path)
    write_metrics(tmp_path / "tool_metrics.jsonl", [
        rec("get_tickets", "error", iso(-5), "error"), rec("get_tickets", "unavailable", iso(-3), "unavailable"),
        rec("get_tickets", ts=iso(-1)), rec("search_kb", ts=iso(-1))])
    rows = {r.tool: r for r in load(ctx, now=NOW, app_started=NOW).tools}
    assert rows["get_tickets"].last_error_ts == iso(-3) and rows["get_tickets"].last_error_type == "unavailable"
    assert rows["search_kb"].last_error_ts is None


def test_old_records_get_their_time_from_the_trace_when_the_two_logs_line_up(tmp_path):
    """data/tool_metrics.jsonl had no times before this change; the trace has one tool_call event per record, in order."""
    ctx = mock_context(tmp_path)
    old = [rec("search_kb"), rec("get_tickets", "error", code="error"), rec("get_tickets")]
    (tmp_path / "tool_metrics.jsonl").write_text("".join(r.model_dump_json(exclude={"ts"}) + "\n" for r in old))
    trace = [{"ts": iso(-50), "kind": "tool_call", "tool": "search_kb", "outcome": "success"},
             {"ts": iso(-49), "kind": "reply"},
             {"ts": iso(-10), "kind": "tool_call", "tool": "get_tickets", "outcome": "error"},
             {"ts": iso(-9), "kind": "tool_call", "tool": "get_tickets", "outcome": "success"}]
    (tmp_path / "trace.jsonl").write_text("".join(json.dumps(e) + "\n" for e in trace))
    rows = {r.tool: r for r in load(ctx, window="all", now=NOW, app_started=NOW).tools}
    assert rows["get_tickets"].last_error_ts == iso(-10)
    assert [r.calls for r in load(ctx, window="today", now=NOW, app_started=NOW).tools] == [2]  # search_kb was 2 days ago


def test_old_records_that_do_not_line_up_with_the_trace_stay_undated_and_only_show_in_all(tmp_path):
    ctx = mock_context(tmp_path)
    (tmp_path / "tool_metrics.jsonl").write_text(rec("search_kb").model_dump_json(exclude={"ts"}) + "\n")
    (tmp_path / "trace.jsonl").write_text(json.dumps({"ts": iso(-1), "kind": "tool_call", "tool": "get_tickets",
                                                       "outcome": "success"}) + "\n")
    assert [r.calls for r in load(ctx, window="all", now=NOW, app_started=NOW).tools] == [1]
    assert load(ctx, window="today", now=NOW, app_started=NOW).tools == []
    assert load(ctx, window="all", now=NOW, app_started=NOW).tool_window.undated == 1


def test_the_page_has_a_window_filter_the_window_line_and_the_last_error_column(tmp_path):
    ctx = mock_context(tmp_path)
    write_metrics(tmp_path / "tool_metrics.jsonl", [rec(ts=iso(-72), outcome="error", code="error")])
    page = monitoring_app(ctx)
    assert not page.exception
    toggle = next(g for g in page.main.get("button_group") if g.label == "Time window")
    assert toggle.options == ["This session", "Today (UTC)", "All time"]
    toggle.set_value("All time").run()
    shown = element_text(page.main)
    assert "All time" in shown and "since 2026-10-16" in shown and "last error" in shown
    page.main.get("button_group")[0].set_value("This session").run()
    assert "No tool calls yet" in element_text(page.main) or "since" in element_text(page.main)


# --- human checks: windows, outcome split, cumulative log ----------------------------------------------------------

def write_hitl(path, *lines):
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def test_approvals_split_by_the_ticket_outcome_joined_on_the_session(tmp_path):
    path = tmp_path / "hitl.jsonl"
    counters = HitlCounters(path)
    counters.record("yes", session="s1")
    counters.record_outcome("created", session="s1")
    counters.record("yes", session="s2")
    counters.record_outcome("failed", session="s2")
    counters.record("yes", session="s3")  # no outcome ever arrived: the process died in between
    counts = load_counts(path)
    assert (counts.approvals, counts.approved_created, counts.approved_failed, counts.approved_unknown) == (3, 1, 1, 1)


def test_a_session_that_approves_twice_is_joined_in_order(tmp_path):
    path = tmp_path / "hitl.jsonl"
    counters = HitlCounters(path)
    for decision, outcome in (("failed", None), ("created", None)):
        counters.record("yes", session="s1")
        counters.record_outcome(decision, session="s1")
    counts = load_counts(path)
    assert (counts.approved_created, counts.approved_failed) == (1, 1)


def test_this_session_today_and_all(tmp_path):
    path = tmp_path / "hitl.jsonl"
    write_hitl(path,
               {"ts": iso(-72), "decision": "no"},
               {"ts": iso(-6), "decision": "edit", "session": "old"},
               {"ts": iso(-5), "decision": "yes", "session": "old"}, {"ts": iso(-5), "event": "outcome", "outcome": "created", "session": "old"},
               {"ts": iso(-1), "decision": "edit", "session": "mine"},
               {"ts": iso(-1), "decision": "yes", "session": "mine"}, {"ts": iso(-1), "event": "outcome", "outcome": "failed", "session": "mine"})
    allc = load_counts(path)
    today = load_counts(path, since=datetime(2026, 10, 19, tzinfo=timezone.utc))
    mine = load_counts(path, sessions=["mine"])
    assert (allc.rejections, allc.edits, allc.approvals) == (1, 2, 2)
    assert (today.rejections, today.edits, today.approvals) == (0, 2, 2)
    assert (mine.edits, mine.approvals, mine.approved_failed, mine.approved_created) == (1, 1, 1, 0)


def test_old_approvals_are_joined_to_their_outcome_through_the_trace(tmp_path):
    """The 2026-10-18 log has `yes` lines with no session and no outcome: the trace says whether a ticket was created."""
    path = tmp_path / "hitl.jsonl"
    write_hitl(path, {"ts": iso(-10), "decision": "yes"}, {"ts": iso(-9), "decision": "yes"}, {"ts": iso(-8), "decision": "yes"},
               {"ts": iso(-400), "decision": "yes"})
    trace = tmp_path / "trace.jsonl"
    events = [{"ts": iso(-10) , "kind": "request"}, {"ts": iso(-10), "kind": "tool_call", "tool": "create_ticket", "outcome": "success"},
              {"ts": iso(-9), "kind": "request"},                       # the trace saw the request, but no ticket was created
              {"ts": iso(-8), "kind": "request"}, {"ts": iso(-8), "kind": "tool_call", "tool": "create_ticket", "outcome": "unavailable"}]
    trace.write_text("".join(json.dumps(e) + "\n" for e in events))
    counts = load_counts(path, trace_path=trace)
    assert (counts.approved_created, counts.approved_failed, counts.approved_unknown) == (1, 2, 1)  # the -400h one: no trace there


def test_a_decision_log_is_append_only(tmp_path):
    path = tmp_path / "hitl.jsonl"
    HitlCounters(path).record("no")
    before = path.read_text()
    HitlCounters(path).record("yes", session="s")
    assert path.read_text().startswith(before)  # earlier lines are never rewritten or removed


def test_the_app_records_created_and_failed_outcomes(tmp_path):
    client = MockTicketClient()
    ctx = mock_context(tmp_path, client=client)
    start = ("My laptop screen is cracked, please raise a ticket", EMAIL)
    for text in (*start, "yes"):
        ctx.service.handle("a", text)
    client.unreachable = True
    for text in (*start, "yes"):
        ctx.service.handle("b", text)
    counts = load_counts(tmp_path / "hitl.jsonl")
    assert (counts.approvals, counts.approved_created, counts.approved_failed) == (2, 1, 1)


def test_an_exception_while_creating_still_records_a_failed_outcome(tmp_path, monkeypatch):
    ctx = mock_context(tmp_path)
    for text in ("My laptop screen is cracked, please raise a ticket", EMAIL):
        ctx.service.handle("a", text)
    from smartdesk_ai.tools.executor import ToolExecutor

    real = ToolExecutor.execute

    def boom(self, name, *a, **k):
        if name == "create_ticket":
            raise RuntimeError("store exploded")
        return real(self, name, *a, **k)

    monkeypatch.setattr(ToolExecutor, "execute", boom)
    with pytest.raises(RuntimeError):
        ctx.service.handle("a", "yes")
    counts = load_counts(tmp_path / "hitl.jsonl")
    assert (counts.approvals, counts.approved_failed) == (1, 1)


def test_the_page_shows_the_split_and_a_window_filter(tmp_path):
    client = MockTicketClient()
    ctx = mock_context(tmp_path, client=client)
    start = ("My laptop screen is cracked, please raise a ticket", EMAIL)
    for text in (*start, "yes"):
        ctx.service.handle("a", text)
    client.unreachable = True
    for text in (*start, "yes"):
        ctx.service.handle("b", text)
    page = monitoring_app(ctx)
    toggle = next(g for g in page.main.get("button_group") if g.label == "Time window")
    toggle.set_value("All time").run()
    metrics = {m.label: m.value for m in page.main.metric}
    assert metrics["Approved, ticket created"] == "1" and metrics["Approved, creation failed"] == "1"
    assert metrics["Approvals"] == "2" and metrics["Rejections"] == "0"


def test_one_window_toggle_sits_at_the_top_and_drives_both_filters(tmp_path):
    page = monitoring_app(mock_context(tmp_path))

    def flat(node):
        for child in node.children.values():
            yield child
            if hasattr(child, "children"):
                yield from flat(child)

    shown = [(e.type, getattr(e, "label", None) or getattr(e, "value", None)) for e in flat(page.main)]
    kinds = [(t, v) for t, v in shown if t in ("subheader", "button_group", "metric")]
    assert [t for t, _ in kinds].index("button_group") < [t for t, _ in kinds].index("subheader")  # the toggle comes first
    assert [t for t, v in kinds if t == "button_group"] == ["button_group"]  # and there is only one
    assert kinds.index(("subheader", "Cost by tier")) < kinds.index(("subheader", "Human checks at the ticket approval gate"))
    labels = [v for t, v in kinds if t == "metric"][:4]
    assert labels == ["Spend today (ledger)", "Cache hit rate", "Approval gate", "Error rate"]  # the four KPI tiles


def test_the_kpi_tiles_and_the_component_badges_show_the_current_state(tmp_path):
    page = monitoring_app(mock_context(tmp_path))
    assert not page.exception
    tiles = {m.label: m.value for m in page.main.metric}
    assert tiles["Spend today (ledger)"] == "$0.0000" and tiles["Approval gate"] == "0 / 0 / 0"
    assert tiles["Cache hit rate"] == "n/a" and tiles["Error rate"] == "n/a"
    notes = " ".join(c.value for c in page.main.caption)
    assert "Budget cutoffs" in notes and "p95 latency n/a" in notes and "Approved / edited / rejected" in notes
    badges = " ".join(m.value for m in page.main.markdown)
    assert ":gray-badge[skipped]" in badges and ":orange-badge[degraded]" in badges  # reranker skipped, redis not set


def test_the_drift_card_shows_the_latest_check_as_tiles_with_the_change(tmp_path):
    from smartdesk_ai.ops.drift import DriftHistory, DriftPoint

    def point(ts, faithfulness):
        return DriftPoint(ts=ts, mode="mock", faithfulness=faithfulness, recall_at_k=0.9, escalation_rate=0.3,
                          avg_rerank_score=None, k=5, n_kb=2, n_gap=1, retrieval_strategy="hybrid",
                          reranker_status="skipped", prompt_versions={"answer": 4}, models={"main": "m"}, usd=0.0)

    history = DriftHistory(tmp_path / "drift.jsonl")
    history.append(point("2026-10-19T10:00:00+00:00", 0.90))
    history.append(point("2026-10-20T10:00:00+00:00", 0.95))
    tiles = {m.label: (m.value, m.delta) for m in monitoring_app(mock_context(tmp_path)).main.metric}
    assert tiles["Faithfulness"] == ("0.95", "+0.05") and tiles["Recall@5"][0] == "0.90"
    assert tiles["Avg rerank score"][0] == "n/a"
