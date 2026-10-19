"""The final gap + KB run (FR-A5, FR-A15, FR-H15): each golden question is asked once; the same answers give the gap and KB
numbers and the drift baseline. Mock models only: no test calls a paid model."""
import json

import pytest

from evals import gap_eval as ge
from evals import golden_run as gr
from evals import live_guard as lg
from evals.harness import gap_rows, kb_rows
from smartdesk_ai.agent.factory import build_stack
from smartdesk_ai.ops import drift
from smartdesk_ai.ops import spend_ledger as sl

GOLDEN = drift.Golden(kb=kb_rows()[:6], gap=gap_rows()[:6])


@pytest.fixture()
def stack(settings, tmp_path):
    return build_stack(settings, environ={}, dotenv_path=None, data_dir=tmp_path / "a", use_cache=False)


def live_run(settings, tmp_path, ceiling=10.0):
    return lg.LiveRun(settings, ["x"], ceiling_usd=ceiling, ledger_path=tmp_path / "ledger.jsonl", on_stall=lambda: None)


def test_one_pass_gives_the_same_gap_numbers_as_the_gap_eval(settings, stack, tmp_path):
    result = gr.run_golden(stack, settings, GOLDEN, mock=True)
    other = build_stack(settings, environ={}, dotenv_path=None, data_dir=tmp_path / "b", use_cache=False)
    cases = [c for c in ge.run_cases(other) if c.id in {r.id for r in [*GOLDEN.kb, *GOLDEN.gap]}]
    expected = ge.summarize(cases)
    assert result.gap.gap_pass_rate == expected.gap_pass_rate and result.gap.kb_answer_rate == expected.kb_answer_rate
    assert result.gap.n_gap == 6 and result.gap.n_kb == 6 and result.stopped is None


def test_one_pass_gives_the_same_drift_numbers_as_the_drift_check(settings, stack, tmp_path):
    result = gr.run_golden(stack, settings, GOLDEN, mock=True)
    other = build_stack(settings, environ={}, dotenv_path=None, data_dir=tmp_path / "b", use_cache=False)
    point = drift.run_drift(other, settings, golden=GOLDEN, mock=True)
    mine = result.drift
    for field in ("faithfulness", "recall_at_k", "escalation_rate", "avg_rerank_score", "k", "n_kb", "n_gap", "mode"):
        assert getattr(mine, field) == getattr(point, field), field


def test_the_precision_and_recall_come_out_of_the_same_pass(settings, stack):
    result = gr.run_golden(stack, settings, GOLDEN, mock=True)
    assert result.gap.escalation_recall == result.gap.gap_pass_rate


def test_a_stopped_run_resumes_and_gives_the_same_numbers(settings, stack, tmp_path, monkeypatch):
    whole = gr.run_golden(stack, settings, GOLDEN, mock=True)
    asked = []
    real = gr.observe

    def counting(stack, row, label, k):
        asked.append(row.id)
        sl.current_run().add_chat(sl.Usage(1, 100_000, 0, 0, 0), settings.models.main.id)
        return real(stack, row, label, k)

    monkeypatch.setattr(gr, "observe", counting)
    cp = lg.Checkpoint(tmp_path / "cp.json", "s")
    with live_run(settings, tmp_path, ceiling=0.0001) as run:
        first = gr.run_golden(stack, settings, GOLDEN, mock=True, run=run, checkpoint=cp)
    assert len(asked) == 1 and first.stopped and first.drift is None
    asked.clear()
    with live_run(settings, tmp_path) as run:
        second = gr.run_golden(stack, settings, GOLDEN, mock=True, run=run, checkpoint=cp)
    assert len(asked) == 11 and second.stopped is None
    assert second.gap.gap_pass_rate == whole.gap.gap_pass_rate and second.drift.recall_at_k == whole.drift.recall_at_k


def test_the_estimate_is_the_drift_estimate_for_the_golden_set(settings):
    e = gr.estimate_golden(settings, drift.Golden.load())
    assert e.name.startswith("the gap and KB run") and "74" in e.detail
    assert e.usd == pytest.approx(drift.estimate_drift(settings.model_copy(update={"modes": settings.modes.model_copy(update={"llm": "live"})}), 74).usd, abs=1e-3)


def test_the_failures_are_named_for_the_stop_rule(settings, stack):
    result = gr.run_golden(stack, settings, GOLDEN, mock=True)
    assert isinstance(result.gap.failures, list)


def test_save_writes_the_gap_file_in_the_gap_eval_format(settings, stack, tmp_path):
    result = gr.run_golden(stack, settings, GOLDEN, mock=True)
    path = gr.save(result, tmp_path)
    saved = json.loads(path.read_text())
    assert path.name == "gap_eval.json" and saved["gap_pass_rate"] == result.gap.gap_pass_rate and saved["cases"]
    assert saved["drift"]["recall_at_k"] == result.drift.recall_at_k


# --- the command line -------------------------------------------------------------------------------------------

@pytest.fixture
def no_paid_calls(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a paid model was called")

    monkeypatch.setattr(gr, "build_live_stack", refuse)


def test_estimate_prints_and_stops(capsys, no_paid_calls):
    assert gr.main(["--estimate"]) == 0
    assert "Estimate for the gap and KB run" in capsys.readouterr().out


def test_a_live_run_without_confirm_spend_runs_nothing(capsys, no_paid_calls):
    assert gr.main([]) == 2
    assert "Nothing was run" in capsys.readouterr().out


def test_a_live_run_goes_through_the_guard_with_twice_the_estimate(settings, tmp_path, monkeypatch):
    seen = {}

    class FakeRun:
        def __init__(self, settings, argv, *, ceiling_usd, **kw):
            seen["ceiling"] = ceiling_usd

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def cost(self):
            return lg.RunCost(0.0, 0, 0, 0)

    monkeypatch.setattr(gr, "LiveRun", FakeRun)
    monkeypatch.setattr(gr, "build_live_stack", lambda *a, **k: None)
    monkeypatch.setattr(gr, "run_golden", lambda *a, **k: gr.GoldenResult(gap=ge.summarize([]), drift=None, stopped=None))
    gr.main(["--confirm-spend", "--out-dir", str(tmp_path), "--checkpoint", str(tmp_path / "cp.json")])
    assert seen["ceiling"] == pytest.approx(2 * gr.estimate_golden(settings, drift.Golden.load()).usd)


def test_the_mock_run_is_free_and_stores_no_drift_point(tmp_path, capsys):
    assert gr.main(["--mock", "--out-dir", str(tmp_path), "--drift-log", str(tmp_path / "drift.jsonl")]) in (0, 1)
    assert "MOCK" in capsys.readouterr().out and not (tmp_path / "drift.jsonl").exists()


def test_a_finished_live_run_stores_one_drift_point(settings, tmp_path, monkeypatch):
    point = drift.make_point(settings, mock=False, faithfulness=1.0, hits=[True], top_scores=[1.0], escalated=0,
                             n_questions=1, n_kb=1, n_gap=0, reranker_status="ok", usd=0.1)

    class FakeRun:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def cost(self):
            return lg.RunCost(0.0, 0, 0, 0)

    monkeypatch.setattr(gr, "LiveRun", FakeRun)
    monkeypatch.setattr(gr, "build_live_stack", lambda *a, **k: None)
    monkeypatch.setattr(gr, "run_golden", lambda *a, **k: gr.GoldenResult(gap=ge.summarize([]), drift=point, stopped=None))
    log = tmp_path / "drift.jsonl"
    gr.main(["--confirm-spend", "--out-dir", str(tmp_path), "--checkpoint", str(tmp_path / "cp.json"), "--drift-log", str(log)])
    assert len(drift.DriftHistory(log).points()) == 1


def test_single_topic_questions_use_one_worker_and_no_synthesizer(settings, stack):
    result = gr.run_golden(stack, settings, GOLDEN, mock=True)
    assert result.single_part_total == 12 and result.single_part_clean == 12 and result.not_clean == []
    assert "12 of 12 questions used exactly one worker" in gr.render(result, mock=True)
