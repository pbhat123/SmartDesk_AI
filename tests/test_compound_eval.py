"""T7.10: the compound-set eval (mock): every acceptance criterion of SPEC section 3I over the 14 compound questions (FR-I4 to FR-I7,
FR-I3, FR-A19, FR-B6). The orchestrator is scripted from the rows, the worker model answers from the expected entries, so this
checks the orchestration, not retrieval quality or any model. No paid call."""
import json

import pytest

from evals import compound_eval as ce
from evals.common import EVALS_DIR, load_jsonl
from evals.schemas import CompoundRow

ROWS = [CompoundRow(**r) for r in load_jsonl(EVALS_DIR / "compound_questions.jsonl")]
BY_ID = {r.id: r for r in ROWS}


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    return ce.run_eval(ce.load_settings(), tmp_path_factory.mktemp("compound"))


# --- the real run ---------------------------------------------------------------------------------------------------

def test_every_row_passes_every_check(report):
    failed = {r.id: r.problems for r in report.rows if r.problems}
    assert not failed, failed
    assert report.rows and len(report.rows) == 14 and report.passed == 14


def test_every_acceptance_criterion_of_section_3i_is_exercised_and_passes(report):
    assert set(report.criteria) == {"single", "compound", "gap_part", "misroute", "misc_gap", "isolation"}
    for name, result in report.criteria.items():
        assert result.checked > 0 and result.passed == result.checked, (name, result)


def test_the_criteria_cover_the_rows_the_set_was_written_for(report):
    c = report.criteria
    assert c["misroute"].checked == sum(any(p.likely_wrong_domain for p in r.parts) for r in ROWS) == 4
    assert c["misc_gap"].checked == sum(any(p.expected_domain == "MISC" and not p.answerable for p in r.parts) for r in ROWS) == 3
    assert c["gap_part"].checked == sum(any(not p.answerable for p in r.parts) for r in ROWS)
    assert c["compound"].checked == sum(sum(p.answerable for p in r.parts) >= 2 for r in ROWS)
    assert c["single"].checked == sum(p.answerable for r in ROWS for p in r.parts)


def test_a_compound_row_runs_both_workers_in_parallel_and_the_synthesizer_once(report):
    row = next(r for r in report.rows if r.id == "cmp-001")
    assert row.run.first_pass_calls == 2 and row.run.workers_overlapped and row.run.synth_calls == 1
    assert row.run.cited_domains == ["HR", "IT"] or row.run.cited_domains == ["IT", "HR"]


def test_a_gap_part_is_named_offered_and_the_ticket_covers_only_it(report):
    row = next(r for r in report.rows if r.id == "cmp-009")
    assert row.run.synth_calls == 0 and "Bengaluru" in row.run.reply
    assert row.run.unanswered == ["Where can I park my car at the Bengaluru office?"]
    assert row.run.draft_category == "MISC" and "request new software" not in row.run.draft_description.lower()


def test_a_misrouted_part_is_answered_after_one_whole_kb_retry(report):
    for rid in ("cmp-005", "cmp-006", "cmp-007", "cmp-008"):
        row = next(r for r in report.rows if r.id == rid)
        assert row.run.retried_parts and not row.problems, (rid, row.problems)


def test_a_worker_sees_only_its_sub_question_and_chunks(report):
    for row in report.rows:
        assert row.run.worker_inputs, row.id
        assert not row.run.leaks, (row.id, row.run.leaks)


# --- the checks themselves bite (a violated criterion is reported) ------------------------------------------------

def good_run(report, rid):
    return next(r for r in report.rows if r.id == rid).run


def problems(report, rid, **change):
    run = good_run(report, rid).model_copy(update=change)
    return ce.check_row(BY_ID[rid], run)


def test_the_checks_accept_the_real_run(report):
    for row in ROWS:
        assert ce.check_row(row, good_run(report, row.id)) == []


def test_a_sequential_fan_out_is_reported(report):
    assert any("parallel" in p for p in problems(report, "cmp-001", workers_overlapped=False))


def test_a_missing_or_doubled_synthesizer_call_is_reported(report):
    assert any("synthesizer" in p for p in problems(report, "cmp-001", synth_calls=0))
    assert any("synthesizer" in p for p in problems(report, "cmp-001", synth_calls=2))
    assert any("synthesizer" in p for p in problems(report, "cmp-009", synth_calls=1))


def test_an_answer_that_misses_a_domain_is_reported(report):
    assert any("cites" in p for p in problems(report, "cmp-001", cited_domains=["IT"]))


def test_a_part_that_was_not_retried_is_reported(report):
    assert any("retry" in p for p in problems(report, "cmp-005", retried_parts=[]))


def test_an_unnamed_gap_or_a_missing_offer_is_reported(report):
    assert any("gap" in p for p in problems(report, "cmp-009", unanswered=[]))
    assert any("offer" in p for p in problems(report, "cmp-009", offered=False))


def test_a_ticket_that_covers_an_answered_part_is_reported(report):
    bad = problems(report, "cmp-009", draft_description="How do I request new software? Where can I park my car?")
    assert any("only the unanswered" in p for p in bad)


def test_a_wrong_ticket_category_for_a_misc_gap_is_reported(report):
    assert any("category" in p for p in problems(report, "cmp-009", draft_category="IT"))


def test_a_worker_that_saw_the_conversation_is_reported(report):
    assert any("isolation" in p for p in problems(report, "cmp-001", leaks=["the original message"]))


def test_the_wrong_number_of_parts_is_reported(report):
    assert any("parts" in p for p in problems(report, "cmp-001", n_parts=1))


# --- the command line -----------------------------------------------------------------------------------------------

def test_the_command_runs_in_mock_mode_and_saves_the_report(tmp_path, capsys):
    assert ce.main(["--out-dir", str(tmp_path)]) == 0
    saved = json.loads((tmp_path / "compound_eval.json").read_text())
    assert saved["passed"] == 14 and set(saved["criteria"]) >= {"single", "compound", "misroute"}
    out = capsys.readouterr().out
    assert "14 of 14" in out and "MOCK" in out


# --- the live path, exercised on the scripted mock stack ----------------------------------------------------------

@pytest.fixture(scope="module")
def mock_stack(tmp_path_factory):
    from smartdesk_ai.agent.factory import build_stack
    from smartdesk_ai.ops.health import HealthRegistry
    from smartdesk_ai.tickets import MockTicketClient

    settings = ce.load_settings()
    lenient = settings.model_copy(update={"thresholds": settings.thresholds.model_copy(update={"escalation_mock": -1.0})})
    worker = ce.RowWorker(ROWS)
    worker.reset(parallel=2)  # the barrier is only needed for the mock's own parallel check; the trace is what is read here
    worker._barrier = None
    return build_stack(lenient, environ={}, dotenv_path=None, data_dir=tmp_path_factory.mktemp("live_obs"), use_cache=False,
                       registry=HealthRegistry(), client=MockTicketClient(), tracing=False, mock_answer_model=worker,
                       mock_routing_model=ce.ScriptedOrchestrator(ROWS), mock_synth_model=ce.MockSynthesizerModel())


@pytest.mark.parametrize("row", ROWS, ids=[r.id for r in ROWS])
def test_the_trace_based_observation_passes_every_check_on_the_scripted_stack(mock_stack, row):
    run = ce.observe_row_live(row, mock_stack)
    assert ce.check_row(row, run) == [], run


def test_the_trace_shows_the_fan_out_and_one_synthesizer_call(mock_stack):
    run = ce.observe_row_live(BY_ID["cmp-001"], mock_stack)
    assert run.first_pass_calls == 2 and run.synth_calls == 1 and run.n_parts == 2 and len(run.worker_inputs) == 2


def test_the_draft_is_judged_by_what_the_drafter_was_given(mock_stack):
    run = ce.observe_row_live(BY_ID["cmp-009"], mock_stack)
    assert run.draft_description.startswith("Employee statements:") and "park my car" in run.draft_description
    assert "request new software" not in run.draft_description.lower() and run.draft_category == "MISC"


def test_a_part_text_the_model_worded_differently_is_matched_by_position(report):
    run = good_run(report, "cmp-009").model_copy(update={
        "part_texts": ["How can I get new software?", "Is there parking at the Bengaluru office?"],
        "unanswered": ["Is there parking at the Bengaluru office?"], "reply": "x Is there parking at the Bengaluru office? Sources:",
        "draft_description": "Employee statements:\n- Is there parking at the Bengaluru office?"})
    assert ce.check_row(BY_ID["cmp-009"], run) == []


def test_overlap_needs_every_window_to_share_a_moment():
    from datetime import datetime, timedelta

    t = datetime(2026, 1, 1)
    near = [(t, t + timedelta(seconds=2)), (t + timedelta(seconds=1), t + timedelta(seconds=3))]
    apart = [(t, t + timedelta(seconds=1)), (t + timedelta(seconds=2), t + timedelta(seconds=3))]
    assert ce._overlap(near) and not ce._overlap(apart) and ce._overlap(near[:1])


# --- the live command is guarded ------------------------------------------------------------------------------------

@pytest.fixture
def no_paid_calls(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a paid model was called")

    monkeypatch.setattr(ce, "build_live_stack", refuse)


def test_estimate_prints_and_stops(capsys, no_paid_calls):
    assert ce.main(["--live", "--estimate"]) == 0
    out = capsys.readouterr().out
    assert "Estimate for the live compound eval" in out and "ceiling" in out


def test_the_estimate_is_small_and_counts_parts(settings):
    e = ce.estimate_live(settings, ROWS)
    assert 0.05 < e.usd < 0.4 and "14 questions" in e.detail and ce.estimate_live(settings, ROWS[:2]).usd < e.usd


def test_a_live_run_without_confirm_spend_runs_nothing(capsys, no_paid_calls):
    assert ce.main(["--live"]) == 2
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
            from evals.live_guard import RunCost

            return RunCost(0.0, 0, 0, 0)

    monkeypatch.setattr(ce, "LiveRun", FakeRun)
    monkeypatch.setattr(ce, "build_live_stack", lambda *a, **k: None)
    monkeypatch.setattr(ce, "run_items", lambda *a, **k: ({}, None))
    ce.main(["--live", "--confirm-spend", "--out-dir", str(tmp_path), "--checkpoint", str(tmp_path / "cp.json")])
    assert seen["ceiling"] == pytest.approx(2 * ce.estimate_live(settings, ROWS).usd)


def test_only_limits_the_live_rows_and_the_estimate(capsys, no_paid_calls):
    assert ce.main(["--live", "--estimate", "--only", "cmp-003,cmp-008"]) == 0
    assert "2 questions" in capsys.readouterr().out
    assert ce.main(["--live", "--estimate", "--only", "cmp-999"]) == 2


def test_the_rendered_live_report_shows_routing_apart_from_the_criteria(mock_stack):
    runs = {r.id: ce.observe_row_live(r, mock_stack) for r in ROWS[:2]}
    results = [ce.RowResult(id=i, run=r, singles=[], problems=ce.check_row(BY_ID[i], r, live=True)) for i, r in runs.items()]
    report = ce.summarize(ROWS[:2], results)
    report.routing = ce.routing_stats(ROWS[:2], runs)
    text = ce.render(report, live=True)
    assert "routing (reported apart)" in text and "sent to the right domain" in text
