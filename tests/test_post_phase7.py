"""The batch driver of the post-Phase-7 live runs: the estimate table, the order, every stop rule, the saved state and --resume, and
the final table. The runs themselves are replaced by a fake runner: no test calls a paid model."""
import json
import os
import time

import pytest

from evals import post_phase7 as pp


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


PLAN = {"intent_accuracy": 0.98, "domain_accuracy": 0.93, "single_topic": 74, "split_single": [{"id": "kb-053"}], "errors": [],
        "split_single_raw": ["kb-045", "kb-053"], "compound_right_count": 14, "compound_right_count_raw": 14,
        "compound": [{"id": f"cmp-{i}", "correct_split": True, "right_count": True} for i in range(14)], "compound_correct": 14, "stopped": None}
COMPOUND = {"passed": 13, "rows": [{"id": f"cmp-{i}", "run": {"parts": []}} for i in range(14)], "criteria": {}, "stopped": None,
            "routing": {"parts": 29, "routed_right": 29, "routed_wrong": 0, "recovered_by_retry": 0, "unrecovered": 0, "not_aligned": 0}}
SAFETY = {"leakage_rate": 0.0, "breach_rate": 0.0, "leaked_ids": [], "breached_ids": [], "cost_usd": 0.05, "stopped": None, "cases": [{}] * 27}
GAP = {"gap_pass_rate": 1.0, "kb_answer_rate": 0.981, "kb_answered_with_citations": 0.962, "kb_not_cited": ["kb-011"], "escalation_precision": 0.955,
       "escalation_recall": 1.0, "failures": [], "n_gap": 21, "n_kb": 53, "single_part_clean": 74, "single_part_total": 74, "stopped": None,
       "drift": {"faithfulness": 0.97, "recall_at_k": 1.0, "escalation_rate": 0.3, "avg_rerank_score": 5.0}}
GOOD = {
    ("diag_compound", "compound_eval_live.json"): {**COMPOUND, "passed": 1, "rows": [
        {"id": "cmp-003", "run": {"parts": [{"part_id": "p2", "domain": "HR", "retrieval_ok": True, "worker_insufficient": [True, True],
                                             "judge_grounded": [], "retried": True, "citations": [], "answered": False}]}}]},
    ("diag_kb", "gap_eval.json"): {**GAP, "n_kb": 1, "kb_not_cited": [], "parts": {}},
    ("plan_check", "orchestrator_plan_check.json"): PLAN,
    ("compound", "compound_eval_live.json"): COMPOUND,
    ("safety", "safety_eval.json"): SAFETY,
    ("gap_kb", "gap_eval.json"): GAP,
    ("paraphrase", "cache_cutoff.json"): {"verdict": {"action": "keep", "highest_different_answer": 0.8, "floor": 0.9, "hits_at_configured": 1}},
    ("tiers_ragas", "tier_comparison.json"): {"mock": False, "kb_runs": 1, "results": []},
    ("tiers_ragas", "rag_metrics_final.json"): {"answer_correctness": 0.7, "hallucination_rate": 0.05, "grounded_answer_rate": 0.97,
                                                "answer_relevancy": 0.64, "context_precision": 0.86, "context_recall": 0.96, "factual_precision": 0.5},
}
NAMES = ["diag_compound", "diag_kb", "plan_check", "compound", "safety", "gap_kb", "paraphrase", "tiers_ragas"]


def fake_runner(root, overrides=None, codes=None):
    calls = []

    def run(step, argv):
        calls.append(step.name)
        for file in step.result_files:
            data = {**GOOD, **(overrides or {})}[(step.name, file)]
            write(step.out_dir(root) / file, data)
        return (codes or {}).get(step.name, 0)

    run.calls = calls
    return run


@pytest.fixture(scope="module")
def steps():
    from smartdesk_ai.config.settings import load_settings

    return pp.build_steps(load_settings())


def batch(steps, root, runner, **kw):
    return pp.run_batch(steps, runner, root, ledger_total=kw.pop("ledger_total", lambda: 0.0), **kw)


# --- the plan --------------------------------------------------------------------------------------------------

def test_the_runs_and_their_phases_are_in_the_agreed_order(steps):
    assert [s.name for s in steps] == NAMES and [s.phase for s in steps] == list("aabcddee")


def test_the_table_has_an_estimate_and_a_ceiling_per_run_and_a_total(steps):
    text = "\n".join(pp.estimate_table(steps))
    assert all(s.name in text for s in steps)
    total = sum(s.estimate.usd for s in steps)
    assert f"${total:.2f}" in text and "total" in text.lower() and "ceiling" in text.lower()
    assert all(s.estimate.ceiling_usd == pytest.approx(2 * s.estimate.usd) for s in steps)


def test_every_command_confirms_spend_and_writes_to_its_folder(steps, tmp_path):
    for s in steps:
        argv = s.argv(tmp_path)
        assert argv[:2] == ["-m", s.module] and "--confirm-spend" in argv and str(s.out_dir(tmp_path)) in argv, s.name
    by = {s.name: s for s in steps}
    assert "--only" in by["diag_compound"].flags and "cmp-013" in " ".join(by["diag_compound"].flags)
    assert by["diag_kb"].flags == ("--only", "kb-010")
    assert by["diag_compound"].subdir == by["diag_kb"].subdir == "diagnostics" and by["compound"].subdir == ""


def test_the_planning_check_gets_a_ceiling_of_twice_its_estimate(steps):
    plan = next(s for s in steps if s.name == "plan_check")
    assert plan.flags[0] == "--ceiling" and float(plan.flags[1]) == pytest.approx(plan.estimate.ceiling_usd, abs=1e-4)


# --- every stop rule -----------------------------------------------------------------------------------------------

def bad(name, file, base, **change):
    return {(name, file): {**base, **change}}


@pytest.mark.parametrize("name, override, expect", [
    ("plan_check", bad("plan_check", "orchestrator_plan_check.json", PLAN, split_single=[{"id": "kb-053"}, {"id": "gap-005"}, {"id": "kb-001"}]), "limit is 2"),
    ("plan_check", bad("plan_check", "orchestrator_plan_check.json", PLAN, compound_right_count=13,
                       compound=[{"id": "cmp-013", "correct_split": False, "right_count": False}] + PLAN["compound"][:13]), "13 of 14"),
    ("plan_check", bad("plan_check", "orchestrator_plan_check.json", PLAN, errors=["kb-001"]), "failed"),
    ("compound", bad("compound", "compound_eval_live.json", COMPOUND, passed=11), "11 of 14"),
    ("safety", bad("safety", "safety_eval.json", SAFETY, leakage_rate=0.04, leaked_ids=["safe-003"]), "leakage"),
    ("safety", bad("safety", "safety_eval.json", SAFETY, breach_rate=0.04, breached_ids=["safe-009"]), "breach"),
    ("gap_kb", bad("gap_kb", "gap_eval.json", GAP, gap_pass_rate=0.952, failures=["gap-006: answered"]), "gap pass"),
    ("gap_kb", bad("gap_kb", "gap_eval.json", GAP, kb_answered_with_citations=0.91), "with citations"),
    ("gap_kb", bad("gap_kb", "gap_eval.json", GAP, kb_answer_rate=0.92), "KB answer rate"),
])
def test_a_stop_rule_ends_the_batch_and_names_the_reason(steps, tmp_path, name, override, expect):
    run = fake_runner(tmp_path, override)
    outcome = batch(steps, tmp_path, run)
    assert outcome.stopped_at == name and expect in outcome.reason
    assert run.calls[-1] == name and outcome.not_run == NAMES[NAMES.index(name) + 1:]


def test_a_wrong_domain_label_on_a_compound_question_is_reported_apart_and_does_not_stop_step_b(steps, tmp_path):
    wrong_label = bad("plan_check", "orchestrator_plan_check.json", PLAN, compound_correct=13,
                      compound=[{"id": "cmp-007", "correct_split": False, "right_count": True}] + PLAN["compound"][:13])
    run = fake_runner(tmp_path, wrong_label)
    assert batch(steps, tmp_path, run).stopped_at is None
    outcome = batch(steps, tmp_path, fake_runner(tmp_path, wrong_label))
    assert "right number and domains in order 13 of 14" in pp.final_table(outcome, tmp_path, balance_before=8.0)


def test_the_limits_themselves_are_allowed(steps, tmp_path):
    edge = {**bad("plan_check", "orchestrator_plan_check.json", PLAN, split_single=[{"id": "a"}, {"id": "b"}]),
            **bad("compound", "compound_eval_live.json", COMPOUND, passed=12),
            **bad("gap_kb", "gap_eval.json", GAP, kb_answer_rate=0.931, kb_answered_with_citations=0.92)}
    assert batch(steps, tmp_path, fake_runner(tmp_path, edge)).stopped_at is None


def test_the_diagnostics_have_no_quality_rule(steps, tmp_path):
    weak = bad("diag_compound", "compound_eval_live.json", GOOD[("diag_compound", "compound_eval_live.json")], passed=0)
    assert batch(steps, tmp_path, fake_runner(tmp_path, weak, codes={"diag_compound": 1})).stopped_at is None


@pytest.mark.parametrize("name", NAMES)
def test_a_run_that_hit_its_ceiling_or_the_watchdog_ends_the_batch(steps, tmp_path, name):
    for code, word in ((1, "stopped"), (3, "watchdog")):
        if name in ("diag_compound", "compound") and code == 1:
            continue  # exit 1 is also "some rows failed"; a ceiling stop is then in the file's `stopped`
        outcome = batch(steps, tmp_path, fake_runner(tmp_path, codes={name: code}))
        assert outcome.stopped_at == name and word in outcome.reason


def test_a_stopped_result_file_ends_the_batch_even_with_exit_code_zero(steps, tmp_path):
    stopped = bad("compound", "compound_eval_live.json", COMPOUND, stopped="ledger spend $0.80 is over the $0.38 ceiling")
    outcome = batch(steps, tmp_path, fake_runner(tmp_path, stopped))
    assert outcome.stopped_at == "compound" and "ceiling" in outcome.reason


def test_a_missing_or_old_result_file_ends_the_batch(steps, tmp_path):
    silent = lambda step, argv: 0  # noqa: E731
    assert "no result" in batch(steps, tmp_path, silent).reason
    old = tmp_path / "diagnostics" / "compound_eval_live.json"
    write(old, COMPOUND)
    os.utime(old, (time.time() - 3600,) * 2)
    assert "no fresh result" in batch(steps, tmp_path, silent).reason  # a file from an earlier session is not this run's result
    stale = lambda step, argv: write(step.out_dir(tmp_path) / step.result_files[0], COMPOUND) or os.utime(  # noqa: E731
        step.out_dir(tmp_path) / step.result_files[0], (time.time() - 3600,) * 2) or 0
    assert "no fresh result" in batch(steps, tmp_path, stale).reason


# --- the saved state and --resume ------------------------------------------------------------------------------------

def test_a_good_batch_runs_everything_in_order_and_records_ledger_cost_per_run(steps, tmp_path):
    ticks = iter(sum(([i, i + 1.0] for i in range(8)), []))
    run = fake_runner(tmp_path)
    outcome = batch(steps, tmp_path, run, ledger_total=lambda: next(ticks))
    assert run.calls == NAMES and outcome.stopped_at is None
    assert [round(r.ledger_usd, 2) for r in outcome.runs] == [1.0] * 8 and outcome.total_usd == pytest.approx(8.0)


def test_the_state_is_saved_after_every_run_and_resume_skips_the_finished_ones(steps, tmp_path):
    state = pp.State(tmp_path / "state.json", "sig")
    stop = bad("compound", "compound_eval_live.json", COMPOUND, passed=9)
    first = fake_runner(tmp_path, stop)
    out1 = batch(steps, tmp_path, first, state=state)
    assert out1.stopped_at == "compound" and first.calls == NAMES[:4]
    saved = json.loads((tmp_path / "state.json").read_text())
    assert [n for n, r in saved["runs"].items() if r["passed"]] == ["diag_compound", "diag_kb", "plan_check"]
    second = fake_runner(tmp_path)
    out2 = batch(steps, tmp_path, second, state=pp.State(tmp_path / "state.json", "sig"), resume=True)
    assert second.calls == NAMES[3:] and out2.stopped_at is None  # the stopped run is tried again, finished ones are not
    assert [r.skipped for r in out2.runs] == [True] * 3 + [False] * 5


def test_resume_adds_the_cost_of_the_earlier_session_to_the_total(steps, tmp_path):
    state = pp.State(tmp_path / "state.json", "sig")
    costs = iter([0, 1, 1, 2, 2, 3, 3, 4])
    stop = bad("compound", "compound_eval_live.json", COMPOUND, passed=9)
    batch(steps, tmp_path, fake_runner(tmp_path, stop), state=state, ledger_total=lambda: next(costs))
    again = iter(sum(([0, 0.5] for _ in range(5)), []))
    out = batch(steps, tmp_path, fake_runner(tmp_path), state=pp.State(tmp_path / "state.json", "sig"), resume=True,
                ledger_total=lambda: next(again))
    by = {r.name: r.ledger_usd for r in out.runs}
    assert by["diag_compound"] == 1 and by["compound"] == pytest.approx(1.0 + 0.5)  # earlier 1.0 plus this session's 0.5


def test_without_resume_the_runs_are_all_done_again(steps, tmp_path):
    batch(steps, tmp_path, fake_runner(tmp_path), state=pp.State(tmp_path / "state.json", "sig"))
    again = fake_runner(tmp_path)
    batch(steps, tmp_path, again, state=pp.State(tmp_path / "state.json", "sig"), resume=False)
    assert again.calls == NAMES


def test_a_state_for_other_prompts_does_not_match(tmp_path):
    (tmp_path / "s.json").write_text(json.dumps({"signature": "old", "runs": {}, "balance_before": 8.0}))
    assert pp.State(tmp_path / "s.json", "old").matches and not pp.State(tmp_path / "s.json", "new").matches


def test_the_active_orchestrator_prompt_is_v9_and_part_of_the_signature(steps):
    from smartdesk_ai.config.settings import load_settings

    s = load_settings()
    assert s.prompts["orchestrator"] == 9
    older = s.model_copy(update={"prompts": {**s.prompts, "orchestrator": 8}})
    assert pp.plan_signature(s, steps) != pp.plan_signature(older, steps)


def test_the_signature_changes_with_the_prompt_versions(steps):
    from smartdesk_ai.config.settings import load_settings

    s = load_settings()
    other = s.model_copy(update={"prompts": {**s.prompts, "orchestrator": 99}})
    assert pp.plan_signature(s, steps) != pp.plan_signature(other, steps)


# --- the command ------------------------------------------------------------------------------------------------------

def test_without_confirm_spend_it_prints_the_table_and_runs_nothing(capsys):
    runner = lambda *a: (_ for _ in ()).throw(AssertionError("ran"))  # noqa: E731
    assert pp.main([], state_path=None, runner=runner) == 2
    out = capsys.readouterr().out
    assert "Nothing was run" in out and "plan_check" in out and "tiers_ragas" in out


def test_estimate_prints_the_table_and_stops(capsys):
    assert pp.main(["--estimate"]) == 0
    assert "total" in capsys.readouterr().out.lower()


def test_an_existing_state_is_refused_without_resume_or_fresh(tmp_path, capsys):
    from smartdesk_ai.config.settings import load_settings

    state_path = tmp_path / "s.json"
    st = pp.State(state_path, pp.plan_signature(load_settings(), pp.build_steps(load_settings())))
    st.runs = {"diag_compound": {"passed": True, "exit_code": 0, "ledger_usd": 0.1}}
    st.save()
    runner = lambda *a: (_ for _ in ()).throw(AssertionError("ran"))  # noqa: E731
    assert pp.main(["--confirm-spend", "--out-dir", str(tmp_path)], state_path=state_path, runner=runner) == 2
    assert "--resume" in capsys.readouterr().out
    assert pp.main(["--confirm-spend", "--resume", "--out-dir", str(tmp_path)], state_path=state_path,
                   runner=fake_runner(tmp_path)) == 0


def test_resume_refuses_a_state_made_for_another_plan(tmp_path, capsys):
    state_path = tmp_path / "s.json"
    state_path.write_text(json.dumps({"signature": "other", "runs": {"diag_kb": {"passed": True, "exit_code": 0, "ledger_usd": 0}}, "balance_before": 8}))
    assert pp.main(["--confirm-spend", "--resume", "--out-dir", str(tmp_path)], state_path=state_path, runner=fake_runner(tmp_path)) == 2
    assert "not reused" in capsys.readouterr().out


def test_fresh_discards_the_state_and_runs_everything(tmp_path):
    state_path = tmp_path / "s.json"
    state_path.write_text(json.dumps({"signature": "other", "runs": {}, "balance_before": 8}))
    run = fake_runner(tmp_path)
    assert pp.main(["--confirm-spend", "--fresh", "--out-dir", str(tmp_path)], state_path=state_path, runner=run) == 0
    assert run.calls == NAMES


# --- the final table ---------------------------------------------------------------------------------------------------

def test_the_final_table_gathers_every_number_the_batch_was_for(steps, tmp_path):
    outcome = batch(steps, tmp_path, fake_runner(tmp_path))
    text = pp.final_table(outcome, tmp_path, balance_before=8.06).lower()
    for needle in ("diagnostics (a)", "cmp-003", "worker insufficient", "planning check", "single-topic questions split 1 of 74",
                   "right number and domains in order 14 of 14", "13 of 14 rows", "routing (apart)", "leakage", "breach", "gap pass",
                   "answered with citations", "kb answer rate", "precision", "recall", "drift baseline", "cut-off verdict", "faithfulness",
                   "context precision", "context recall", "answer correctness", "ledger total", "remaining balance"):
        assert needle in text, needle


def test_the_final_table_shows_raw_and_guarded_splits_side_by_side(steps, tmp_path):
    text = pp.final_table(batch(steps, tmp_path, fake_runner(tmp_path)), tmp_path, balance_before=8.0)
    assert "split 1 of 74 (kb-053) after the guard, 2 raw (kb-045, kb-053)" in text
    assert "compound right number of parts 14 of 14 (raw 14)" in text


def test_a_result_older_than_the_batch_is_never_printed_it_says_not_run(steps, tmp_path):
    runner = fake_runner(tmp_path)
    old = time.time() - 3600
    batch(steps, tmp_path, runner)  # leaves every result file in place
    for f in tmp_path.rglob("*.json"):
        os.utime(f, (old, old))
    stop = bad("plan_check", "orchestrator_plan_check.json", PLAN, errors=["kb-001"])
    started = time.time()
    outcome = batch(steps, tmp_path, fake_runner(tmp_path, stop))  # this batch writes diagnostics and the plan check, then stops
    text = pp.final_table(outcome, tmp_path, balance_before=8.0, since=started)
    for later in ("live compound set: not run", "safety: not run", "gap + KB run and drift baseline: not run",
                  "paraphrase cut-off verdict: not run", "Ragas: not run", "tier comparison: not run"):
        assert later in text
    assert "leakage" not in text and "drift baseline: faithfulness" not in text and "diagnostics (a):" in text
    nothing = pp.final_table(pp.Outcome(), tmp_path, balance_before=8.0, since=time.time() + 60)
    assert "diagnostics (a): not run" in nothing and "orchestrator planning check: not run" in nothing


def test_the_batch_start_is_saved_in_the_state_and_survives_a_restart(tmp_path):
    state = pp.State(tmp_path / "s.json", "sig")
    state.started = 1234.5
    state.save()
    assert pp.State(tmp_path / "s.json", "sig").started == 1234.5
    state.clear()
    assert pp.State(tmp_path / "s.json", "sig").started is None


def test_a_stopped_batch_says_where_and_why_and_what_did_not_run(steps, tmp_path):
    stop = bad("safety", "safety_eval.json", SAFETY, leakage_rate=0.04, leaked_ids=["safe-003"])
    text = pp.final_table(batch(steps, tmp_path, fake_runner(tmp_path, stop)), tmp_path, balance_before=8.0)
    assert "STOPPED at safety" in text and "safe-003" in text and "not run: gap_kb, paraphrase, tiers_ragas" in text


def test_the_plan_and_the_stop_rules_are_in_plan_md():
    plan = (pp.ROOT / "PLAN.md").read_text()
    assert "post_phase7" in plan and "--resume" in plan and "92%" in plan and "orchestrator_v6" in plan


def test_a_state_saved_before_the_start_was_recorded_is_dated_from_its_first_finished_run(tmp_path):
    state = pp.State(tmp_path / "s.json", "sig")
    state.runs = {"a": {"finished": "2026-10-19T20:00:00+00:00"}, "b": {"finished": "2026-10-19T20:10:00+00:00"}}
    from datetime import datetime

    assert pp.batch_start(state) == datetime.fromisoformat("2026-10-19T20:00:00+00:00").timestamp() - 300
    assert abs(pp.batch_start(pp.State(None, "sig")) - time.time()) < 5


# --- the post-Phase-8 profile: v9 rerun, five runs, scenarios only, no Ragas ---------------------------------------------

PHASE8 = ["plan_check", "compound", "safety", "gap_kb", "tiers_scenarios"]


@pytest.fixture(scope="module")
def steps8():
    from smartdesk_ai.config.settings import load_settings

    return pp.build_steps(load_settings(), profile="phase8")


def test_the_phase8_profile_is_the_five_agreed_runs_without_ragas_or_diagnostics(steps8):
    assert [s.name for s in steps8] == PHASE8 and [s.phase for s in steps8] == list("bcdde")
    by = {s.name: s for s in steps8}
    assert by["tiers_scenarios"].flags == ("--scenarios-only",) and by["tiers_scenarios"].result_files == ("tier_comparison.json",)
    assert not any(s.name in ("diag_compound", "diag_kb", "paraphrase", "tiers_ragas") for s in steps8)


def test_the_phase8_stop_rules_are_the_phase7_ones(steps8, steps):
    old = {s.name: s for s in steps}
    for s in steps8:
        if s.name != "tiers_scenarios":
            assert s.check is old[s.name].check and s.ok_codes == old[s.name].ok_codes and s.flags == old[s.name].flags, s.name


def test_the_phase8_estimate_has_no_judge_pass_and_a_ceiling_of_twice_each_run(steps8, steps):
    by, old = {s.name: s for s in steps8}, {s.name: s for s in steps}
    assert by["tiers_scenarios"].estimate.usd < old["tiers_ragas"].estimate.usd / 5
    assert "judge" not in by["tiers_scenarios"].estimate.detail and "KB answers" not in by["tiers_scenarios"].estimate.detail and all(
        s.estimate.ceiling_usd == pytest.approx(2 * s.estimate.usd) for s in steps8)


def test_the_phase8_profile_has_its_own_state_file_and_the_phase7_state_is_left_alone(tmp_path, monkeypatch):
    seen = {}

    def fake_batch(steps, runner, root, **kw):
        seen["path"] = kw["state"].path
        return pp.Outcome()

    old = tmp_path / "post_phase7_state.json"
    old.write_text('{"signature": "x", "runs": {"gap_kb": {"passed": true}}}')
    monkeypatch.setattr(pp, "STATE_FILE", old)
    monkeypatch.setattr(pp, "STATE_FILE_PHASE8", tmp_path / "post_phase8_state.json")
    monkeypatch.setattr(pp, "run_batch", fake_batch)
    pp.main(["--profile", "phase8", "--confirm-spend"], state_path=old, runner=lambda s, a: 0)
    assert seen["path"] == tmp_path / "post_phase8_state.json" and "gap_kb" in old.read_text()


def test_the_final_table_handles_a_scenarios_only_tier_file(steps8, tmp_path):
    for name in ("plan_check", "compound", "safety", "gap_kb"):
        for file in next(s for s in steps8 if s.name == name).result_files:
            write(tmp_path / file, GOOD[(name, file)])
    write(tmp_path / "tier_comparison.json", {"mock": False, "kb_runs": 0, "scenarios_only": True, "results": [
        {"name": "tier_map", "usd_per_task": 0.002, "completion_rate": 0.9, "task_accuracy": 0.9, "kb": None}]})
    outcome = batch(steps8, tmp_path, lambda s, a: 0)
    table = pp.final_table(outcome, tmp_path, balance_before=5.0, since=0)
    assert "tier comparison, tier_map" in table and "no KB run" in table
