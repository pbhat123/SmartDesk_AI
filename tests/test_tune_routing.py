"""T6.4: tuning `thresholds.routing_confidence` on the intent set (FR-D8). Mock classifier here; the real one
is a paid run that waits for approval. Config is never edited."""
import json

import pytest

from evals import tune_routing as tr
from evals.run import main
from smartdesk_ai.agent.intents import ClassifierError
from smartdesk_ai.config.settings import DEFAULT_CONFIG
from smartdesk_ai.llm.mock import MockRoutingModel


def obs(id, confidence, correct):
    return tr.Observation(id=id, confidence=confidence, correct=correct)


FIXTURE = [obs("a", 0.95, True), obs("b", 0.90, True), obs("c", 0.80, True), obs("d", 0.70, False),
           obs("e", 0.55, True), obs("f", 0.40, False), obs("g", None, False)]


def test_coverage_and_accuracy_at_each_threshold_match_hand_counts():
    points = {p.threshold: p for p in tr.sweep(FIXTURE, [0.3, 0.6, 0.85])}
    assert (points[0.3].n_covered, points[0.3].accuracy_covered) == (6, pytest.approx(4 / 6))
    assert points[0.3].coverage == pytest.approx(6 / 7) and points[0.3].right_but_handed_off == 0
    assert (points[0.6].n_covered, points[0.6].accuracy_covered) == (4, pytest.approx(3 / 4))
    assert points[0.6].right_but_handed_off == 1  # e (0.55) was right but is now handed to the tool pick
    assert (points[0.85].n_covered, points[0.85].accuracy_covered) == (2, 1.0)
    assert points[0.85].right_but_handed_off == 2  # c and e


def test_nothing_covered_has_no_accuracy():
    [p] = tr.sweep(FIXTURE, [0.99])
    assert p.n_covered == 0 and p.accuracy_covered is None


def test_the_proposal_is_the_lowest_threshold_that_reaches_the_target():
    points = tr.sweep(FIXTURE, [0.3, 0.6, 0.75, 0.85])
    assert tr.propose(points, target=0.95).threshold == 0.75  # at 0.75 only a, b, c are accepted, all right
    assert tr.propose(points, target=1.1) is None


def test_failed_classifier_calls_count_as_wrong_and_uncovered():
    def classify(text, awaiting="none"):
        raise ClassifierError("down")

    rows = tr.load_rows()[:3]
    observations = tr.observe(rows, classify)
    assert all(o.confidence is None and not o.correct for o in observations)


def test_the_mock_classifier_gets_the_whole_set_right(settings):
    from smartdesk_ai.agent.intents import make_classifier

    observations = tr.observe(tr.load_rows(), make_classifier(settings, builder=lambda m, c: MockRoutingModel()))
    assert len(observations) >= 30 and all(o.correct for o in observations)


def test_the_run_reports_current_and_proposal_and_leaves_config_alone(settings, tmp_path, capsys):
    before = DEFAULT_CONFIG.read_text()
    result = tr.run(settings, tmp_path, builder=lambda m, c: MockRoutingModel(), live=False)
    assert DEFAULT_CONFIG.read_text() == before
    assert result["current"]["threshold"] == settings.thresholds.routing_confidence and result["live"] is False
    assert result["classifier_accuracy"] == 1.0 and result["note"].startswith("proposal only")
    assert json.loads((tmp_path / "routing_threshold.json").read_text())["n"] == len(tr.load_rows())
    assert "current" in capsys.readouterr().out


def test_the_cli_step_runs_in_mock_mode(tmp_path):
    assert main(["--mock", "--only", "routing", "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "routing_threshold.json").exists()


def test_a_live_run_needs_explicit_spend_confirmation(capsys):
    assert main(["--only", "routing"]) == 1
    assert "--confirm-spend" in capsys.readouterr().err
