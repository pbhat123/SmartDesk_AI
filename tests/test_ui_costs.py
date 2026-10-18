"""UI review fix 5: the Monitoring cost section. The ledger number says what it covers (today in UTC, every source)
and is split by who spent it (app, eval, script, drift); this browser session's own cost is shown separately; dated
model ids are grouped under the tier they belong to (FR-H7, FR-H8)."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from smartdesk_ai.llm.fallback import LLMCall
from smartdesk_ai.ops.cost import CostTracker
from smartdesk_ai.ops.runtime import RequestContext
from smartdesk_ai.ops.spend_ledger import ledger_path
from smartdesk_ai.ui.monitoring import load_monitoring, session_cost
from tests.ui_helpers import app_test, element_text, mock_context, monitoring_app

NOW = datetime.now(timezone.utc)


def write_ledger(*runs):
    """runs: (run_id, source, days_ago, [(model, usd, calls)])."""
    lines = []
    for run_id, source, days_ago, spends in runs:
        ts = (NOW - timedelta(days=days_ago)).isoformat(timespec="seconds")
        lines.append({"ts": ts, "run_id": run_id, "source": source, "event": "start", "command": "x"})
        for model, usd, calls in spends:
            lines.append({"ts": ts, "run_id": run_id, "source": source, "event": "spend", "kind": "chat", "model": model,
                          "calls": calls, "input_tokens": 100, "output_tokens": 10, "usd": usd})
    ledger_path().write_text("".join(json.dumps(line) + "\n" for line in lines))


def load(ctx):
    return load_monitoring(ctx.settings, ctx.data_dir, ctx.kb_dir, ctx.registry, results_dir=ctx.results_dir)


@pytest.fixture
def ctx(tmp_path):
    return mock_context(tmp_path)


# --- today, all sources, split ---------------------------------------------------------------------------

def test_today_is_split_by_source_and_adds_up(ctx):
    write_ledger(("a", "app", 0, [("gpt-5.4-mini-2026-03-17", 0.10, 2)]),
                 ("e", "eval", 0, [("gpt-5.4-mini-2026-03-17", 0.30, 5), ("text-embedding-3-small", 0.01, 3)]),
                 ("s", "script", 0, [("gpt-4o-mini", 0.05, 1)]),
                 ("old", "eval", 3, [("gpt-5.4-mini-2026-03-17", 9.00, 50)]))  # another day: not in "today"
    data = load(ctx)
    today = {row.source: row for row in data.sources_today}
    assert [r.source for r in data.sources_today] == ["app", "eval", "script"]
    assert today["app"].usd == 0.10 and today["eval"].usd == 0.31 and today["script"].usd == 0.05
    assert today["eval"].calls == 8
    assert data.ledger_today_usd == pytest.approx(0.46) and data.ledger_total_usd == pytest.approx(9.46)


def test_the_page_labels_the_number_and_shows_each_source(ctx):
    write_ledger(("a", "app", 0, [("gpt-4o-mini", 0.0123, 1)]), ("e", "eval", 0, [("gpt-4o-mini", 0.0456, 1)]))
    page = monitoring_app(ctx).main
    labels = [m.label for m in page.metric]
    assert "Today (UTC), all sources" in labels and "Ledger, today (UTC)" not in labels
    shown = element_text(page)
    assert "app" in shown and "eval" in shown and "0.0123" in shown and "0.0456" in shown


def test_an_unusual_source_is_listed_after_the_usual_ones(ctx):
    write_ledger(("d", "drift", 0, [("gpt-4o-mini", 0.02, 1)]), ("a", "app", 0, [("gpt-4o-mini", 0.01, 1)]))
    assert [r.source for r in load(ctx).sources_today] == ["app", "drift"]


# --- dated model ids under their tier --------------------------------------------------------------------

def test_dated_model_ids_are_grouped_under_the_configured_model(ctx):
    write_ledger(("a", "app", 0, [("gpt-5.4-mini-2026-03-17", 0.1, 1), ("gpt-5.4-mini", 0.1, 1),
                                  ("gpt-5.4-mini-2026-04-01", 0.1, 1), ("gpt-4o-mini-2024-07-18", 0.1, 1)]))
    tiers = {t.tier: t for t in load(ctx).tiers}
    main = tiers["main"]
    assert main.model == ctx.settings.models.main.id and main.ledger_calls == 3
    assert main.seen_as == ["gpt-5.4-mini-2026-03-17", "gpt-5.4-mini-2026-04-01"]  # the plain id is not a variant
    assert tiers["light"].seen_as == ["gpt-4o-mini-2024-07-18"]
    shown = element_text(monitoring_app(ctx).main)
    assert shown.count("gpt-5.4-mini-2026-03-17") == 1  # one row for the tier, the variants inside it


# --- this session -----------------------------------------------------------------------------------------

def call(tier="main", model="gpt-5.4-mini", inp=1000, out=100):
    return LLMCall(task="WORKER_ANSWER", tier=tier, model_id=model, model_version=None, prompt="p", completion="c",
                   input_tokens=inp, cached_tokens=0, output_tokens=out, latency_ms=1.0, steps=(), fell_back=False, ok=True)


@pytest.fixture
def priced(ctx):
    tracker = CostTracker(ctx.settings, priced=True)
    ctx.service.services.cost = tracker
    return tracker


def test_session_cost_counts_only_this_browser_sessions_conversations(ctx, priced):
    for sid in ("mine-1", "mine-2", "someone-else"):
        priced.record_llm(RequestContext(session_id=sid, request_id=f"r-{sid}"), "answer", call())
    mine = session_cost(priced, ["mine-1", "mine-2"])
    one = priced.llm_usd("main", 1000, 0, 100)
    assert mine.usd == pytest.approx(2 * one) and mine.llm_calls == 2 and mine.tokens == 2 * 1100
    assert session_cost(priced, []).usd == 0 and session_cost(None, ["x"]).usd == 0


def test_the_page_shows_the_session_cost_apart_from_the_ledger(ctx, priced):
    at = app_test(ctx)
    at.chat_input[0].set_value("How do I reset my forgotten password?").run()
    sid = at.session_state["thread_id"]
    priced.record_llm(RequestContext(session_id=sid, request_id="x"), "answer", call(inp=2000, out=0))
    priced.record_llm(RequestContext(session_id="other", request_id="y"), "answer", call(inp=900000, out=0))
    at.switch_page("views/monitoring_page.py").run()
    metrics = {m.label: m.value for m in at.main.metric}
    mine = sum(e.usd for e in priced.entries if e.session_id == sid)  # the chat turn's own (mock, priced here) steps too
    assert mine > priced.llm_usd("main", 2000, 0, 0)
    assert metrics["This session (tracker estimate)"] == f"${mine:.4f}" and mine < 0.5  # "other" (0.675) excluded
    assert "Today (UTC), all sources" in metrics


def test_a_new_conversation_keeps_counting_toward_the_session(ctx, priced):
    at = app_test(ctx)
    first = at.session_state["thread_id"]
    at.sidebar.button[0].click().run()
    second = at.session_state["thread_id"]
    assert first != second and at.session_state["thread_ids"] == [first, second]
