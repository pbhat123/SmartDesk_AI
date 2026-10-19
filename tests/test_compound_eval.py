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


def test_a_live_run_is_refused_for_now(capsys):
    assert ce.main(["--live"]) == 2
    assert "deferred" in capsys.readouterr().err.lower()
