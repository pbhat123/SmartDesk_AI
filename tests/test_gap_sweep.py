"""The live end-to-end threshold sweep (FR-A5, A15): the gap and KB sets run once per escalation value. Mock here."""
import json

import pytest

from evals import gap_sweep as gs
from smartdesk_ai.agent.factory import build_stack
from tests.llm_fakes import FakeAnswerLLM


def factory(settings, tmp_path):
    def stack_for(threshold):
        thresholds = settings.thresholds.model_copy(update={"escalation_mock": threshold})
        return build_stack(settings.model_copy(update={"thresholds": thresholds}), environ={}, dotenv_path=None,
                           data_dir=tmp_path / f"t{threshold}", mock_answer_model=FakeAnswerLLM("insufficient"))
    return stack_for


def test_one_row_per_threshold_with_the_numbers_the_report_needs(settings, tmp_path):
    rows = gs.sweep([0.1, 0.3], factory(settings, tmp_path))
    assert [r["threshold"] for r in rows] == [0.1, 0.3]
    for r in rows:
        assert {"gap_pass_rate", "kb_answer_rate", "escalation_precision", "escalation_recall", "cost_usd",
                "gap_failures", "gap_006", "kb_escalated"} <= set(r)
        assert r["gap_pass_rate"] == 1.0 and r["gap_006"] == "escalated"


def test_the_sweep_stops_when_spend_passes_the_ceiling(settings, tmp_path):
    rows = gs.sweep([0.1, 0.2, 0.3], factory(settings, tmp_path), ceiling_usd=-1.0)  # any spend is over
    assert len(rows) == 1  # the first run completes; the next is not started


def test_main_needs_spend_confirmation_for_a_live_sweep(capsys):
    assert gs.main(["--thresholds", "0.4"]) == 1
    assert "--confirm-spend" in capsys.readouterr().err


def test_a_mock_sweep_writes_a_result_file(settings, tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert gs.main(["--mock", "--thresholds", "0.1", "0.2", "--out-dir", str(out)]) == 0
    saved = json.loads((out / "gap_sweep.json").read_text())
    assert [r["threshold"] for r in saved["rows"]] == [0.1, 0.2] and saved["mock"] is True
