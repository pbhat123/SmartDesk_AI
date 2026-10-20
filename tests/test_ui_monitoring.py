"""T6.6: the Monitoring tab (FR-G10 tool stats, FR-F7 degraded modes, FR-H7 cost by tier, FR-H8 daily budget,
FR-H6 stale chunks, FR-H11 agent metrics). Mock mode, temp folders; nothing reads or writes the repo's data/."""
import json
from datetime import datetime, timezone

from smartdesk_ai.ops.budget import BudgetController, SpendStore
from smartdesk_ai.ops.health import Component, HealthRegistry
from smartdesk_ai.ops.spend_ledger import ledger_path
from smartdesk_ai.tickets import MockTicketClient
from smartdesk_ai.ui.monitoring import load_monitoring
from tests.ui_helpers import monitoring_app, element_text, mock_context

EMAIL = "priya.k@smartdesk.example"


def _run_failing_ticket_flow(ctx, session="s1"):
    for text in ("My monitor keeps flickering", EMAIL, "yes", "yes"):
        ctx.service.handle(session, text)


def _load(ctx):
    return load_monitoring(ctx.settings, ctx.data_dir, ctx.kb_dir, ctx.registry, results_dir=ctx.results_dir)


def _ledger(*events):
    path = ledger_path()
    day = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [{"ts": day, "run_id": "r1", "source": "test", "event": "start", "command": "x"}]
    lines += [{"ts": day, "run_id": "r1", "source": "test", "event": "spend", **e} for e in events]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


# --- tool stats (FR-G10) --------------------------------------------------------------------------

def test_a_failing_tool_shows_failures_retries_and_latency_percentiles(tmp_path):
    client = MockTicketClient()
    ctx = mock_context(tmp_path, client=client)
    client.unreachable = True  # after the demo tickets were seeded: every create_ticket attempt fails
    _run_failing_ticket_flow(ctx)
    row = next(r for r in _load(ctx).tools if r.tool == "create_ticket")
    assert row.calls == 1 and row.failures == 1 and row.failure_types == {"unavailable": 1}
    assert row.retries == ctx.settings.tools.max_attempts - 1 == 2
    assert row.success_rate == 0.0 and row.p95_ms >= row.p50_ms > 0


def test_the_tab_renders_those_numbers_and_no_raw_email(tmp_path):
    client = MockTicketClient()
    ctx = mock_context(tmp_path, client=client)
    client.unreachable = True
    _run_failing_ticket_flow(ctx)
    at = monitoring_app(ctx)
    assert not at.exception
    tab = at.main
    shown = element_text(tab)
    assert "create_ticket" in shown and "unavailable" in shown
    assert EMAIL not in shown and EMAIL not in _load(ctx).model_dump_json()


def test_a_fresh_install_shows_empty_states_not_errors(tmp_path):
    at = monitoring_app(mock_context(tmp_path))
    assert not at.exception
    shown = element_text(at.main)
    assert "No tool calls yet" in shown and "No sessions in this window yet" in shown


def test_loading_never_creates_or_changes_files(tmp_path):
    data = tmp_path / "empty"
    data.mkdir()
    ctx = mock_context(tmp_path / "ctx")
    load_monitoring(ctx.settings, data, ctx.kb_dir, HealthRegistry(), results_dir=tmp_path / "none")
    assert list(data.iterdir()) == []


# --- degraded modes (FR-F7) -----------------------------------------------------------------------

def test_degraded_components_are_listed_with_the_substitute_in_use(tmp_path):
    ctx = mock_context(tmp_path)
    ctx.registry.mark_degraded(Component.RERANKER, "reranker not loaded")
    rows = {d.component: d for d in _load(ctx).degraded}
    assert rows["reranker"].mode.startswith("RRF order") and "not loaded" in rows["reranker"].reason
    assert "redis" in rows  # REDIS_URL is not set in the mock context
    shown = element_text(monitoring_app(ctx).main)
    assert "reranker" in shown and "RRF order" in shown


# --- cost by tier (FR-H7, H8): the spend ledger and ops.db ---------------------------------------

def test_cost_by_tier_comes_from_the_ledger_with_dated_model_names_mapped(tmp_path):
    ctx = mock_context(tmp_path)
    _ledger(
        {"kind": "chat", "model": "gpt-5.4-mini-2026-03-17", "calls": 2, "input_tokens": 1000, "output_tokens": 100, "usd": 0.10},
        {"kind": "chat", "model": "gpt-4o-mini", "calls": 3, "input_tokens": 500, "output_tokens": 50, "usd": 0.02},
        {"kind": "embedding", "model": "text-embedding-3-small", "calls": 1, "input_tokens": 30, "output_tokens": 0, "usd": 0.001},
        {"kind": "chat", "model": "mystery-model", "calls": 1, "input_tokens": 1, "output_tokens": 1, "usd": 0.5, "price_assumed": True},
    )
    data = _load(ctx)
    tiers = {t.tier: t for t in data.tiers}
    assert tiers["main"].ledger_usd == 0.10 and tiers["main"].ledger_calls == 2
    assert tiers["light"].ledger_usd == 0.02 and tiers["embedding"].ledger_usd == 0.001
    assert tiers["other"].ledger_usd == 0.5  # an unknown model is shown, never dropped
    assert data.ledger_total_usd == round(0.10 + 0.02 + 0.001 + 0.5, 6)
    assert data.ledger_today_usd == data.ledger_total_usd  # all events are from today (UTC)


def test_the_daily_budget_is_read_from_ops_db(tmp_path):
    ctx = mock_context(tmp_path)
    BudgetController(ctx.settings, SpendStore(tmp_path / "ops.db")).add(5.5)
    budget = _load(ctx).budget
    assert budget.spent_today_usd == 5.5 and budget.level == 1
    assert budget.cutoff_1_usd == ctx.settings.budget.cutoff_1_usd


def test_the_tracker_log_adds_its_own_column_per_tier(tmp_path):
    ctx = mock_context(tmp_path)
    entry = {"ts": "t", "request_id": "r", "session_id": "s", "task": "kb_answer", "agent": "answer", "step": "llm",
             "tier": "main", "model": "gpt-5.4-mini", "input_tokens": 10, "cached_tokens": 0, "output_tokens": 5, "usd": 0.25}
    (tmp_path / "cost.jsonl").write_text(json.dumps(entry) + "\n")
    tiers = {t.tier: t for t in _load(ctx).tiers}
    assert tiers["main"].tracker_usd == 0.25 and tiers["main"].tracker_steps == 1


def test_cost_table_is_on_screen(tmp_path):
    ctx = mock_context(tmp_path)
    _ledger({"kind": "chat", "model": "gpt-4o-mini", "calls": 1, "input_tokens": 5, "output_tokens": 5, "usd": 0.0123})
    shown = element_text(monitoring_app(ctx).main)
    assert "0.0123" in shown and "light" in shown


# --- stale chunks (FR-H6) -------------------------------------------------------------------------

def test_stale_chunks_are_listed_after_a_kb_file_changes(tmp_path, kb_copy):
    ctx = mock_context(tmp_path, kb_dir=kb_copy)
    assert _load(ctx).stale.stale_chunk_ids == []
    with (kb_copy / "it_support.md").open("a") as f:
        f.write("\nAn extra line that changes the file hash.\n")
    stale = _load(ctx).stale
    assert stale.stale_files == ["it_support.md"] and stale.stale_chunk_ids
    assert all(cid.startswith("IT-") for cid in stale.stale_chunk_ids)
    shown = element_text(monitoring_app(ctx).main)
    assert "it_support.md" in shown


# --- agent metrics (FR-H11) -----------------------------------------------------------------------

def test_agent_metrics_are_read_from_the_saved_eval_result(tmp_path):
    ctx = mock_context(tmp_path)
    ctx.results_dir.mkdir()
    (ctx.results_dir / "agent_metrics.json").write_text(json.dumps({
        "mock": True, "n_tasks": 12, "n_turns": 30, "completion_rate": 0.9167, "task_accuracy": 1.0,
        "steps_to_completion": 6.5, "turns_to_completion": 2.5, "task_latency_ms": 120.0, "turn_p50_ms": 40.0,
        "turn_p95_ms": 90.0, "usd_per_task": 0.0, "tokens_per_task": 1500.0, "error_rate": 0.0333, "tasks": []}))
    metrics = _load(ctx).agent_metrics
    assert metrics.n_tasks == 12 and metrics.mock and metrics.completion_rate == 0.9167
    shown = element_text(monitoring_app(ctx).main)
    assert "0.917" in shown or "91.7" in shown
    assert "load_test" in shown  # throughput is not measured by this tab


def test_a_broken_metrics_file_degrades_to_the_empty_state(tmp_path):
    ctx = mock_context(tmp_path)
    ctx.results_dir.mkdir()
    (ctx.results_dir / "agent_metrics.json").write_text("{not json")
    assert _load(ctx).agent_metrics is None


# --- live agent metrics and the saved T7.11 scenario results (FR-H11) -------------------------------

def _tier_comparison(ctx, mock=False):
    ctx.results_dir.mkdir(exist_ok=True)
    (ctx.results_dir / "tier_comparison.json").write_text(json.dumps({"mock": mock, "kb_runs": 1, "results": [
        {"name": "tier_map", "n_tasks": 12, "completion_rate": 0.9167, "task_accuracy": 0.9167,
         "usd_per_task": 0.00193, "tokens_per_task": 3897.0, "by_agent": {}, "by_tier": {}, "tasks": []},
        {"name": "all_main", "n_tasks": 12, "completion_rate": 0.9167, "task_accuracy": 0.9167,
         "usd_per_task": 0.00245, "tokens_per_task": 3896.0, "by_agent": {}, "by_tier": {}, "tasks": []}]}))


def test_live_metrics_come_from_the_trace_and_follow_the_tool_stats_window(tmp_path):
    ctx = mock_context(tmp_path)
    ctx.service.handle("s1", "How do I connect to the VPN from home?")
    ctx.service.handle("s1", "What is the sick leave policy?")
    live = _load(ctx).live_metrics
    assert live.n_tasks == 1 and live.n_turns == 2 and live.completion_rate == 1.0 and live.error_rate == 0.0
    assert live.steps_to_completion > 0 and live.turn_p95_ms >= live.turn_p50_ms > 0
    future = datetime.now(timezone.utc).replace(year=2099)  # the window starts after every turn: nothing in it
    window = load_monitoring(ctx.settings, ctx.data_dir, ctx.kb_dir, ctx.registry, results_dir=ctx.results_dir,
                             window="app", app_started=future)
    assert window.live_metrics is None


def test_saved_scenario_results_are_labelled_with_run_date_and_source(tmp_path):
    ctx = mock_context(tmp_path)
    _tier_comparison(ctx)
    runs = _load(ctx).scenario_results
    assert [r.name for r in runs.rows] == ["tier_map", "all_main"] and runs.mock is False
    assert runs.source.endswith("tier_comparison.json") and runs.saved_on == datetime.now(timezone.utc).date().isoformat()
    shown = element_text(monitoring_app(ctx).main)
    assert "tier_comparison.json" in shown and runs.saved_on in shown and "live" in shown.lower()
    assert "No sessions in this window yet" not in shown  # no live turn yet, but the saved run exists
    assert "Tier map" in shown and "All Main" in shown


def test_the_empty_message_shows_only_when_both_the_live_trace_and_the_saved_run_are_empty(tmp_path):
    ctx = mock_context(tmp_path)
    empty = element_text(monitoring_app(ctx).main)
    assert "No sessions in this window yet" in empty
    _tier_comparison(ctx)
    ctx.service.handle("s1", "How do I connect to the VPN from home?")
    both = element_text(monitoring_app(ctx).main)
    assert "No sessions in this window yet" not in both and "Live sessions in the tool stats window" in both
    assert "100.0%" in both and "Tier map" in both


def test_a_mock_scenario_run_is_labelled_mock_and_a_broken_file_is_ignored(tmp_path):
    ctx = mock_context(tmp_path)
    _tier_comparison(ctx, mock=True)
    assert _load(ctx).scenario_results.mock is True
    (ctx.results_dir / "tier_comparison.json").write_text("{nope")
    assert _load(ctx).scenario_results is None
