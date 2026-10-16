"""T2.12: tune the escalation thresholds on the dev half, report the held-out test half (FR-A5, FR-A15).
Mocked embeddings and reranker; the real run happens after approval."""
import hashlib
import itertools
import json
import shutil
from pathlib import Path

import pytest

from evals import tune_threshold as tt
from evals.harness import make_resources
from evals.run import main
from smartdesk_ai.config.settings import load_settings
from smartdesk_ai.rag.index import open_index
from smartdesk_ai.rag.ingest import ingest
from tests.kb_helpers import KB_DIR

ROOT = Path(__file__).resolve().parent.parent


def pts(kb, gap, split="dev"):
    return [tt.Point(f"k{i}", "kb", split, s) for i, s in enumerate(kb)] + \
           [tt.Point(f"g{i}", "gap", split, s) for i, s in enumerate(gap)]


# --- metrics ---------------------------------------------------------------------------

def test_metrics_at_a_threshold_by_hand():
    # kb scores 0.9 0.7 0.4 -> at 0.5: two answered, one escalated. gap 0.6 0.3 0.1 -> one answered, two escalated.
    m = tt.metrics(pts([0.9, 0.7, 0.4], [0.6, 0.3, 0.1]), 0.5)
    assert m.answered_kb == pytest.approx(2 / 3)  # share of KB questions answered
    assert m.escalated_gap == pytest.approx(2 / 3)  # share of gap questions escalated
    assert m.balanced_accuracy == pytest.approx(2 / 3)
    assert m.escalation_precision == pytest.approx(2 / 3)  # of 3 escalations (0.4, 0.3, 0.1), 2 were gaps
    assert m.escalation_recall == pytest.approx(2 / 3)
    assert (m.n_kb, m.n_gap) == (3, 3)


def test_a_score_equal_to_the_threshold_is_answered():
    assert tt.metrics(pts([0.5], [0.4]), 0.5).answered_kb == 1.0


def test_auc_is_one_when_separable_zero_when_reversed_and_half_for_ties():
    assert tt.auc(pts([0.9, 0.8], [0.2, 0.1])) == 1.0
    assert tt.auc(pts([0.1, 0.2], [0.8, 0.9])) == 0.0
    assert tt.auc(pts([0.5], [0.5])) == 0.5


def test_metrics_with_an_empty_class_do_not_divide_by_zero():
    m = tt.metrics(pts([0.9], []), 0.5)
    assert m.n_gap == 0 and m.escalated_gap == 0.0 and m.escalation_precision == 0.0


# --- choosing the threshold ------------------------------------------------------------

def test_separable_scores_give_the_midpoint_between_the_classes():
    t = tt.choose_threshold(pts([0.9, 0.8, 0.7, 0.6], [0.5, 0.3, 0.2, 0.1]))
    assert t == 0.55


def test_overlapping_scores_pick_a_threshold_with_the_best_balanced_accuracy():
    dev = pts([0.9, 0.8, 0.7, 0.45, 0.3], [0.6, 0.35, 0.2, 0.1, 0.05])
    t = tt.choose_threshold(dev)
    best = max(tt.metrics(dev, c / 100).balanced_accuracy for c in range(1, 100))
    assert tt.metrics(dev, t).balanced_accuracy == pytest.approx(best)


def test_the_choice_is_in_the_open_unit_interval_even_for_extreme_scores():
    assert 0 < tt.choose_threshold(pts([0.999, 0.998], [0.001, 0.0005])) < 1
    assert 0 < tt.choose_threshold(pts([0.5], [0.5])) < 1


def test_the_strict_threshold_answers_no_gap_question():
    dev = pts([0.9, 0.8, 0.5, 0.3], [0.6, 0.35, 0.2])
    t = tt.strict_threshold(dev)
    assert tt.metrics(dev, t).escalated_gap == 1.0
    assert t > 0.6  # above the highest gap score
    assert tt.metrics(dev, t).answered_kb == pytest.approx(2 / 4)  # the price: KB questions at 0.5 and 0.3 escalate


def test_only_the_dev_half_decides_the_threshold():
    dev = pts([0.9, 0.8, 0.7], [0.3, 0.2, 0.1], "dev")
    quiet = tt.tune(dev + pts([0.9], [0.1], "test"))
    wild = tt.tune(dev + pts([0.05, 0.04], [0.99, 0.98], "test"))  # a hostile test half
    assert quiet["threshold"] == wild["threshold"]
    assert wild["test"].balanced_accuracy < quiet["test"].balanced_accuracy  # ...but it shows up in the report


def test_tune_reports_dev_and_test_metrics_at_the_chosen_value():
    out = tt.tune(pts([0.9, 0.8], [0.2, 0.1], "dev") + pts([0.85, 0.75], [0.3, 0.15], "test"))
    assert out["dev"].balanced_accuracy == 1.0 and out["test"].balanced_accuracy == 1.0
    assert out["dev_auc"] == 1.0 and out["test_auc"] == 1.0
    assert out["strict_threshold"] >= out["threshold"]


# --- end to end with mocks -------------------------------------------------------------

@pytest.fixture(scope="module")
def tuned(tmp_path_factory):
    res = make_resources(mock=True)
    d = tmp_path_factory.mktemp("tune_index")
    ingest(KB_DIR, d, res.embeddings, res.settings.chunking)
    return tt.run_tuning(res, open_index(d, res.embeddings)), res


def test_both_score_scales_are_tuned_and_split_correctly(tuned):
    out, _ = tuned
    assert set(out) >= {"reranker", "dense_cosine"}
    kb_dev, gap_dev = len(tt.kb_rows("dev")), len(tt.gap_rows("dev"))
    for kind in ("reranker", "dense_cosine"):
        section = out[kind]
        assert section["dev"].n_kb == kb_dev and section["dev"].n_gap == gap_dev
        assert section["test"].n_kb == len(tt.kb_rows("test")) and section["test"].n_gap == len(tt.gap_rows("test"))
        assert 0 < section["threshold"] < 1
        assert 0.0 <= section["test"].balanced_accuracy <= 1.0


def test_scores_come_from_the_production_retrieval_path(tuned):
    out, res = tuned
    first = tt.kb_rows("dev")[0]
    from smartdesk_ai.ops.health import HealthRegistry
    from smartdesk_ai.rag.pipeline import retrieve

    expected = {
        "reranker": retrieve(open_index_for(res), first.question, cfg=res.settings, reranker=res.reranker, health=HealthRegistry()),
    }
    pt = next(p for p in out["reranker"]["points"] if p.id == first.id)
    assert pt.score == pytest.approx(expected["reranker"].top_score)


def open_index_for(res, _cache={}):
    if "ix" not in _cache:
        import tempfile

        d = Path(tempfile.mkdtemp())
        ingest(KB_DIR, d, res.embeddings, res.settings.chunking)
        _cache["ix"] = open_index(d, res.embeddings)
    return _cache["ix"]


# --- config ----------------------------------------------------------------------------

def test_write_config_changes_only_the_two_threshold_lines(tmp_path):
    cfg = tmp_path / "config.yaml"
    shutil.copy(ROOT / "config.yaml", cfg)
    before = cfg.read_text().splitlines()
    tt.write_config(cfg, escalation=0.37, fallback=0.21)
    after = cfg.read_text().splitlines()
    changed = [(a, b) for a, b in zip(before, after) if a != b]
    assert len(before) == len(after) and len(changed) == 2
    settings = load_settings(cfg)
    assert (settings.thresholds.escalation, settings.thresholds.fallback) == (0.37, 0.21)
    assert settings.thresholds.routing_confidence == load_settings().thresholds.routing_confidence
    assert all("provisional" not in b for _, b in changed)


def test_write_config_refuses_values_outside_the_open_unit_interval(tmp_path):
    cfg = tmp_path / "config.yaml"
    shutil.copy(ROOT / "config.yaml", cfg)
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError):
            tt.write_config(cfg, escalation=bad, fallback=0.2)
    assert cfg.read_text() == (ROOT / "config.yaml").read_text()


# --- CLI -------------------------------------------------------------------------------

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_cli_proposes_values_and_leaves_config_untouched(capsys):
    before = sha(ROOT / "config.yaml")
    assert main(["--only", "threshold", "--mock"]) == 0
    out = capsys.readouterr().out
    assert "MOCK" in out and "proposed" in out.lower() and "escalation" in out and "fallback" in out
    assert "dev" in out and "test" in out
    assert sha(ROOT / "config.yaml") == before


def test_cli_write_config_applies_the_proposal_to_the_given_file(tmp_path, capsys):
    cfg = tmp_path / "config.yaml"
    shutil.copy(ROOT / "config.yaml", cfg)
    assert main(["--only", "threshold", "--mock", "--write-config", "--config", str(cfg)]) == 0
    assert load_settings(cfg).thresholds.escalation != 0.50 or load_settings(cfg).thresholds.fallback != 0.50
    assert "provisional" not in "\n".join(l for l in cfg.read_text().splitlines() if l.strip().startswith(("escalation", "fallback")))


def test_cli_writes_a_results_file_when_asked(tmp_path):
    main(["--only", "threshold", "--mock", "--out-dir", str(tmp_path)])
    data = json.loads((tmp_path / "threshold.json").read_text())
    assert data["mock"] is True
    for kind in ("reranker", "dense_cosine"):
        assert {"threshold", "strict_threshold", "dev", "test", "dev_auc", "test_auc"} <= set(data[kind])


def test_cli_refuses_to_write_mock_values_into_the_real_config(capsys):
    before = sha(ROOT / "config.yaml")
    assert main(["--only", "threshold", "--mock", "--write-config"]) == 1
    assert "refusing" in capsys.readouterr().err
    assert sha(ROOT / "config.yaml") == before
