"""The batch driver of the post-Phase-7 live runs: the estimate table, the order, the stop rules and the final table. The runs
themselves are replaced by a fake runner here: no test calls a paid model."""
import json

import pytest

from evals import post_phase7 as pp


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


GOOD = {
    "safety_eval.json": {"leakage_rate": 0.0, "breach_rate": 0.0, "leaked_ids": [], "breached_ids": [], "cost_usd": 0.05, "stopped": None,
                         "cases": [{}] * 27},
    "gap_eval.json": {"gap_pass_rate": 1.0, "kb_answer_rate": 0.981, "escalation_precision": 0.955, "escalation_recall": 1.0,
                      "failures": [], "n_gap": 21, "n_kb": 53, "single_part_clean": 74, "single_part_total": 74, "stopped": None,
                      "drift": {"faithfulness": 0.97, "recall_at_k": 1.0, "escalation_rate": 0.3, "avg_rerank_score": 5.0}},
    "compound_eval_live.json": {"passed": 13, "rows": [{}] * 14, "criteria": {}, "stopped": None},
    "cache_cutoff.json": {"verdict": {"action": "keep", "highest_different_answer": 0.8, "floor": 0.9, "hits_at_configured": 1}},
    "tier_comparison.json": {"mock": False, "kb_runs": 1, "results": []},
    "rag_metrics_final.json": {"answer_correctness": 0.7, "hallucination_rate": 0.05, "grounded_answer_rate": 0.97,
                               "answer_relevancy": 0.64, "context_precision": 0.86, "context_recall": 0.96, "factual_precision": 0.5},
}


def runner_writing(out, overrides=None, codes=None, cost=0.1):
    """A fake runner: 'runs' a step by writing its result file, and reports an exit code."""
    calls = []

    def run(step, argv):
        calls.append(step.name)
        data = {**GOOD, **(overrides or {})}
        for file in step.result_files:
            write(out / file, data[file])
        return (codes or {}).get(step.name, 0)

    run.calls = calls
    return run


@pytest.fixture()
def settings_():
    from smartdesk_ai.config.settings import load_settings

    return load_settings()


def test_the_steps_run_cheapest_and_most_informative_first(settings_):
    assert [s.name for s in pp.build_steps(settings_)] == ["safety", "gap_kb", "compound", "paraphrase", "tiers_ragas"]


def test_the_table_has_an_estimate_and_a_ceiling_per_run_and_a_total(settings_):
    steps = pp.build_steps(settings_)
    text = "\n".join(pp.estimate_table(steps))
    for step in steps:
        assert step.name in text
    total = sum(s.estimate.usd for s in steps)
    assert f"${total:.2f}" in text and "total" in text.lower() and "ceiling" in text.lower()
    assert all(s.estimate.ceiling_usd == pytest.approx(2 * s.estimate.usd) for s in steps)


def test_every_command_confirms_spend_and_writes_to_the_out_dir(settings_, tmp_path):
    for step in pp.build_steps(settings_):
        argv = step.argv(tmp_path)
        assert argv[:2] == ["-m", step.module] and "--confirm-spend" in argv and str(tmp_path) in argv, step.name


# --- the stop rules ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, override, expect", [
    ("safety", {"safety_eval.json": {**GOOD["safety_eval.json"], "leakage_rate": 0.04, "leaked_ids": ["safe-003"]}}, "leakage"),
    ("safety", {"safety_eval.json": {**GOOD["safety_eval.json"], "breach_rate": 0.04, "breached_ids": ["safe-009"]}}, "breach"),
    ("gap_kb", {"gap_eval.json": {**GOOD["gap_eval.json"], "gap_pass_rate": 0.952, "failures": ["gap-006: answered"]}}, "gap pass"),
    ("gap_kb", {"gap_eval.json": {**GOOD["gap_eval.json"], "kb_answer_rate": 0.92}}, "KB answer rate"),
    ("compound", {"compound_eval_live.json": {**GOOD["compound_eval_live.json"], "passed": 11}}, "11 of 14"),
])
def test_a_stop_rule_ends_the_batch_and_names_the_reason(settings_, tmp_path, name, override, expect):
    steps = pp.build_steps(settings_)
    run = runner_writing(tmp_path, override)
    outcome = pp.run_batch(steps, run, tmp_path, ledger_total=lambda: 0.0)
    assert outcome.stopped_at == name and expect in outcome.reason
    assert run.calls[-1] == name and len(run.calls) < len(steps)


def test_a_kb_answer_rate_exactly_five_points_below_is_still_allowed(settings_, tmp_path):
    ok = {"gap_eval.json": {**GOOD["gap_eval.json"], "kb_answer_rate": 0.931}}
    outcome = pp.run_batch(pp.build_steps(settings_), runner_writing(tmp_path, ok), tmp_path, ledger_total=lambda: 0.0)
    assert outcome.stopped_at is None


def test_a_run_that_hit_its_ceiling_or_the_watchdog_ends_the_batch(settings_, tmp_path):
    for code, word in ((1, "stopped"), (3, "watchdog")):
        run = runner_writing(tmp_path, codes={"gap_kb": code})
        outcome = pp.run_batch(pp.build_steps(settings_), run, tmp_path, ledger_total=lambda: 0.0)
        assert outcome.stopped_at == "gap_kb" and word in outcome.reason and run.calls == ["safety", "gap_kb"]


def test_a_stopped_result_file_ends_the_batch_even_with_exit_code_zero(settings_, tmp_path):
    stopped = {"gap_eval.json": {**GOOD["gap_eval.json"], "stopped": "ledger spend $0.80 is over the $0.74 ceiling"}}
    outcome = pp.run_batch(pp.build_steps(settings_), runner_writing(tmp_path, stopped), tmp_path, ledger_total=lambda: 0.0)
    assert outcome.stopped_at == "gap_kb" and "ceiling" in outcome.reason


def test_a_missing_result_file_ends_the_batch(settings_, tmp_path):
    class Silent:
        calls = []

        def __call__(self, step, argv):
            return 0

    outcome = pp.run_batch(pp.build_steps(settings_), Silent(), tmp_path, ledger_total=lambda: 0.0)
    assert outcome.stopped_at == "safety" and "no result" in outcome.reason


def test_a_good_batch_runs_all_five_in_order_and_records_ledger_cost_per_run(settings_, tmp_path):
    ledger = iter([0.0, 0.05, 0.05, 0.40, 0.40, 0.55, 0.55, 0.56, 0.56, 1.80])
    run = runner_writing(tmp_path)
    outcome = pp.run_batch(pp.build_steps(settings_), run, tmp_path, ledger_total=lambda: next(ledger))
    assert run.calls == ["safety", "gap_kb", "compound", "paraphrase", "tiers_ragas"] and outcome.stopped_at is None
    assert [round(r.ledger_usd, 2) for r in outcome.runs] == [0.05, 0.35, 0.15, 0.01, 1.24]
    assert outcome.total_usd == pytest.approx(1.80)


def test_the_final_table_gathers_every_number_the_batch_was_for(settings_, tmp_path):
    outcome = pp.run_batch(pp.build_steps(settings_), runner_writing(tmp_path), tmp_path, ledger_total=lambda: 0.0)
    text = pp.final_table(outcome, tmp_path, balance_before=8.06)
    for needle in ("gap pass", "KB answer rate", "precision", "recall", "leakage", "breach", "13 of 14", "faithfulness",
                   "context precision", "context recall", "answer correctness", "drift baseline", "cut-off", "keep",
                   "ledger", "remaining"):
        assert needle.lower() in text.lower(), needle


def test_a_stopped_batch_says_where_and_why_and_still_lists_what_ran(settings_, tmp_path):
    bad = {"safety_eval.json": {**GOOD["safety_eval.json"], "leakage_rate": 0.04, "leaked_ids": ["safe-003"]}}
    outcome = pp.run_batch(pp.build_steps(settings_), runner_writing(tmp_path, bad), tmp_path, ledger_total=lambda: 0.0)
    text = pp.final_table(outcome, tmp_path, balance_before=8.0)
    assert "STOPPED at safety" in text and "safe-003" in text and "not run" in text.lower()


# --- the command ----------------------------------------------------------------------------------------------------

def test_without_confirm_spend_it_prints_the_table_and_runs_nothing(capsys, monkeypatch):
    monkeypatch.setattr(pp, "subprocess_runner", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    assert pp.main([]) == 2
    out = capsys.readouterr().out
    assert "Nothing was run" in out and "safety" in out


def test_estimate_prints_the_table_and_stops(capsys):
    assert pp.main(["--estimate"]) == 0
    assert "total" in capsys.readouterr().out.lower()
