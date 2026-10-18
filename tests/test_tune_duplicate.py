"""T6.4: tuning `thresholds.duplicate_title` on near-identical and distinct title pairs (FR-F4). Free and
offline: the similarity is a string ratio. Config is never edited by the tuner."""
import json

import pytest

from evals import tune_duplicate as td
from evals.run import main
from smartdesk_ai.config.settings import DEFAULT_CONFIG


def pair(a, b, same):
    return td.TitlePair(id="x", a=a, b=b, same_issue=same)


def test_the_pair_set_is_balanced_and_labelled():
    pairs = td.load_pairs()
    assert sum(p.same_issue for p in pairs) >= 20 and sum(not p.same_issue for p in pairs) >= 20
    assert len({p.id for p in pairs}) == len(pairs) and all(p.a.lower() != p.b.lower() for p in pairs)
    assert len({(p.a, p.b) for p in pairs}) == len(pairs)


def test_hard_negatives_exist_so_the_threshold_matters():
    from smartdesk_ai.agent.ticket_agent import title_similarity

    pairs = td.load_pairs()
    assert max(title_similarity(p.a, p.b) for p in pairs if not p.same_issue) > min(
        title_similarity(p.a, p.b) for p in pairs if p.same_issue)


def test_score_counts_hand_checked_pairs():
    pairs = [pair("My monitor keeps flickering", "Monitor keeps flickering", True),   # ~0.9
             pair("VPN drops every hour", "Printer out of toner", False),             # low
             pair("Laptop will not start", "Laptop screen cracked", False),           # middling
             pair("Wifi down", "Completely different request about leave balance", True)]  # a repeat we cannot see
    low, high = td.score(pairs, 0.50), td.score(pairs, 0.95)
    assert (low.false_negatives, high.false_negatives) == (1, 2)
    assert low.recall == pytest.approx(0.5) and high.recall == 0.0 and low.precision == 1.0
    assert td.score(pairs, 0.0).false_positives == 2


def test_best_f1_prefers_the_lower_threshold_on_a_tie():
    pairs = [pair("monitor flickers", "monitor flickers!", True), pair("abc", "xyz", False)]
    best = td.best_f1(td.sweep(pairs))
    assert best.f1 == 1.0 and best.threshold == td.GRID[0]


def test_best_for_recall_takes_the_highest_threshold_that_still_catches_enough():
    points = [td.Point(threshold=t, precision=1.0, recall=r, f1=0.5, false_positives=0, false_negatives=0)
              for t, r in ((0.5, 1.0), (0.6, 0.95), (0.7, 0.9), (0.8, 0.8))]
    assert td.best_for_recall(points).threshold == 0.7
    assert td.best_for_recall(points, target=1.1) is None


def test_the_run_reports_the_current_value_and_two_proposals_without_touching_config(settings, tmp_path, capsys):
    before = DEFAULT_CONFIG.read_text()
    result = td.run(settings, tmp_path)
    assert DEFAULT_CONFIG.read_text() == before
    assert result["current"]["threshold"] == settings.thresholds.duplicate_title
    assert result["best_f1"]["f1"] >= result["current"]["f1"] - 1e-9
    assert json.loads((tmp_path / "duplicate_threshold.json").read_text())["note"].startswith("proposal only")
    out = capsys.readouterr().out
    assert "current" in out and "best F1" in out


def test_the_cli_step_runs(tmp_path):
    assert main(["--mock", "--only", "duplicate", "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "duplicate_threshold.json").exists()
