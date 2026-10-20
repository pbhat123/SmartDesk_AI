"""The Time window changes every windowed section: Spend, Cache hit rate, Cost by agent (and, as before, the tool stats).
Old and new records are written; each window must keep exactly its own (FR-G10, FR-H7)."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from smartdesk_ai.ops.spend_ledger import ledger_path
from smartdesk_ai.ui.monitoring import load_monitoring
from tests.ui_helpers import element_text, mock_context, monitoring_app

NOW = datetime(2026, 10, 20, 15, 0, tzinfo=timezone.utc)
APP_STARTED = NOW - timedelta(hours=1)  # 14:00
OLD, EARLIER_TODAY, SINCE_START = "2026-10-17T10:00:00+00:00", "2026-10-20T08:00:00+00:00", "2026-10-20T14:30:00+00:00"


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def seed(ctx, old, earlier, new):
    """One record of each age in the ledger, the cache log and the trace; the cost grows tenfold per step."""
    spend = lambda ts, usd: {"ts": ts, "run_id": "r", "source": "app", "event": "spend", "model": "gpt-4o-mini",
                             "calls": 1, "input_tokens": 1, "output_tokens": 1, "usd": usd}
    write_jsonl(ledger_path(), [spend(old, 1.0), spend(earlier, 10.0), spend(new, 100.0)])
    write_jsonl(ctx.data_dir / "cache.jsonl", [{"ts": old, "event": "miss"},
                                               {"ts": earlier, "event": "hit_exact", "saved_ms": 5.0},
                                               {"ts": new, "event": "miss"}])
    span = lambda ts, usd: {"kind": "agent_span", "ts": ts, "agent": "it_worker", "tiers": ["main"], "calls": 1,
                            "tool_calls": 0, "input_tokens": 10, "output_tokens": 1, "usd": usd}
    write_jsonl(ctx.data_dir / "trace.jsonl", [span(old, 0.001), span(earlier, 0.01), span(new, 0.1)])


def load(ctx, window):
    return load_monitoring(ctx.settings, ctx.data_dir, ctx.kb_dir, ctx.registry, results_dir=ctx.results_dir,
                           window=window, now=NOW, app_started=APP_STARTED)


@pytest.fixture
def ctx(tmp_path):
    return mock_context(tmp_path)


def test_spend_cache_and_cost_by_agent_each_keep_only_their_window(ctx):
    seed(ctx, OLD, EARLIER_TODAY, SINCE_START)
    app, today, everything = load(ctx, "app"), load(ctx, "today"), load(ctx, "all")
    assert [d.ledger_window_usd for d in (app, today, everything)] == [100.0, 110.0, 111.0]
    assert [d.cache.lookups for d in (app, today, everything)] == [1, 2, 3]
    assert [d.cache.hit_rate for d in (app, today, everything)] == [0.0, 0.5, pytest.approx(1 / 3)]
    assert [sum(a.usd for a in d.agents) for d in (app, today, everything)] == [
        pytest.approx(0.1), pytest.approx(0.11), pytest.approx(0.111)]
    assert [sum(a.spans for a in d.agents) for d in (app, today, everything)] == [1, 2, 3]
    assert [d.window_label for d in (app, today, everything)] == ["Since app start", "Today (UTC)", "All time"]


def test_the_all_time_and_today_figures_of_the_cost_section_do_not_move(ctx):
    seed(ctx, OLD, EARLIER_TODAY, SINCE_START)
    for window in ("app", "today", "all"):
        assert load(ctx, window).ledger_total_usd == 111.0


def test_an_empty_window_shows_zero_not_the_other_windows_numbers(ctx):
    seed(ctx, OLD, OLD, OLD)
    data = load(ctx, "app")
    assert data.ledger_window_usd == 0.0 and data.cache.lookups == 0 and data.agents == []


def test_the_page_tiles_follow_the_toggle(ctx):
    now = datetime.now(timezone.utc)
    seed(ctx, (now - timedelta(days=3)).isoformat(), (now - timedelta(seconds=1)).isoformat(), now.isoformat())
    page = monitoring_app(ctx)
    toggle = page.main.get("button_group")[0]
    tiles = {}
    for window in ("All time", "This session"):
        toggle.set_value(window).run()
        tiles[window] = {m.label: m.value for m in page.main.metric}
        assert "Cost by agent" in element_text(page.main)
    assert tiles["All time"]["Spend (window)"] == "$111.0000" and tiles["This session"]["Spend (window)"] == "$110.0000"
    assert tiles["All time"]["Cache hit rate"] == "33.3%" and tiles["This session"]["Cache hit rate"] == "50.0%"
    assert "Not affected by the Time window" in element_text(page.main)
