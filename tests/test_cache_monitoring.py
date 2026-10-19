"""The cache in the UI: the debug panel says hit or miss, and the Monitoring tab shows the hit rate and the latency the
cache saved (SPEC section 4 cache rules, FR-H7). Mock models, temp data."""
import json

from smartdesk_ai.ops.cache_log import CacheLog, CacheRecord, load_cache_records, summarize_cache
from tests.ui_helpers import app_test, element_text, mock_context, monitoring_app

ASK = "How do I reset my forgotten password?"


def _panels(at):
    return [e for e in at.expander if "Retrieval debug" in e.label]


def _ask_twice(tmp_path):
    ctx = mock_context(tmp_path)
    at = app_test(ctx)
    at.chat_input[0].set_value(ASK).run()
    at.chat_input[0].set_value(ASK).run()
    return ctx, at


# --- the log -------------------------------------------------------------------------------------


def test_the_summary_counts_hits_misses_and_saved_latency():
    records = [CacheRecord(ts="t", event="miss"), CacheRecord(ts="t", event="hit_exact", saved_ms=100.0),
               CacheRecord(ts="t", event="hit_semantic", saved_ms=50.0, similarity=0.97),
               CacheRecord(ts="t", event="miss"), CacheRecord(ts="t", event="clear")]
    stats = summarize_cache(records)
    assert (stats.lookups, stats.exact_hits, stats.semantic_hits, stats.misses, stats.clears) == (4, 1, 1, 2, 1)
    assert stats.hit_rate == 0.5 and stats.saved_ms_total == 150.0 and stats.saved_ms_per_hit == 75.0


def test_an_empty_log_has_no_rate():
    stats = summarize_cache([])
    assert stats.lookups == 0 and stats.hit_rate is None and stats.saved_ms_per_hit is None


def test_the_log_holds_numbers_only_and_survives_a_half_written_line(tmp_path):
    path = tmp_path / "cache.jsonl"
    log = CacheLog(path)
    log.record("hit_exact", saved_ms=12.345)
    log.record("miss")
    path.write_text(path.read_text() + '{"ts": "x", "ev')
    assert [r.event for r in load_cache_records(path)] == ["hit_exact", "miss"]
    assert set(json.loads(path.read_text().splitlines()[0])) == {"ts", "event", "saved_ms", "similarity"}


# --- the debug panel -----------------------------------------------------------------------------


def test_the_panel_says_miss_then_exact_hit(tmp_path):
    _, at = _ask_twice(tmp_path)
    first, second = (element_text(p) for p in _panels(at))
    assert "Cache: miss" in first
    assert "Cache: hit (exact)" in second and "saved about" in second


def test_a_turn_that_did_not_use_the_cache_shows_no_cache_line(tmp_path):
    at = app_test(mock_context(tmp_path))
    at.chat_input[0].set_value("hello there").run()
    assert not _panels(at) and "Cache:" not in element_text(at.main)


# --- the Monitoring tab --------------------------------------------------------------------------


def test_monitoring_shows_the_hit_rate_and_the_saved_latency(tmp_path):
    ctx, _ = _ask_twice(tmp_path)
    ctx.service.handle("other", ASK)  # a third lookup: miss, hit, hit
    page = monitoring_app(ctx).main
    metrics = {m.label: m.value for m in page.metric}
    assert "Answer cache" in element_text(page)
    assert metrics["Hit rate"] == "66.7%"  # two hits in three lookups
    assert (metrics["Exact hits"], metrics["Semantic hits"], metrics["Misses"]) == ("2", "0", "1")
    assert metrics["Time saved"].endswith(" s") and metrics["Saved per hit"].endswith(" ms")


def test_monitoring_before_any_lookup_says_so(tmp_path):
    shown = element_text(monitoring_app(mock_context(tmp_path)).main)
    assert "Answer cache" in shown and "No cache lookups yet" in shown
