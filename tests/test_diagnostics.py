"""Per-part diagnostics for the live runs (what each worker and judge said, whether a part was retried), the outcome-based live
misroute check with routing reported separately, and keeping the live trace. Mock models only."""
import json

import pytest

from evals import compound_eval as ce
from evals import diagnostics as dg
from evals import live_guard as lg
from evals.common import EVALS_DIR, load_jsonl
from evals.schemas import CompoundRow

ROWS = [CompoundRow(**r) for r in load_jsonl(EVALS_DIR / "compound_questions.jsonl")]
BY_ID = {r.id: r for r in ROWS}


@pytest.fixture(scope="module")
def mock_stack(tmp_path_factory):
    from smartdesk_ai.agent.factory import build_stack
    from smartdesk_ai.ops.health import HealthRegistry
    from smartdesk_ai.tickets import MockTicketClient

    settings = ce.load_settings()
    lenient = settings.model_copy(update={"thresholds": settings.thresholds.model_copy(update={"escalation_mock": -1.0})})
    worker = ce.RowWorker(ROWS)
    worker.reset(parallel=2)
    worker._barrier = None
    return build_stack(lenient, environ={}, dotenv_path=None, data_dir=tmp_path_factory.mktemp("diag"), use_cache=False,
                       registry=HealthRegistry(), client=MockTicketClient(), tracing=False, mock_answer_model=worker,
                       mock_routing_model=ce.ScriptedOrchestrator(ROWS), mock_synth_model=ce.MockSynthesizerModel())


def run(mock_stack, rid):
    return ce.observe_row_live(BY_ID[rid], mock_stack)


# --- the diagnostics ---------------------------------------------------------------------------------------------

def test_every_part_records_its_domain_retrieval_worker_and_judge_verdicts(mock_stack):
    parts = run(mock_stack, "cmp-001").parts
    assert [p.domain for p in parts] == ["IT", "HR"] and all(p.answered and p.retrieval_ok for p in parts)
    assert all(p.worker_insufficient == [False] and p.judge_grounded == [True] for p in parts)
    assert all(p.citations and p.top_cosine is not None and p.text for p in parts)


def test_a_retried_part_shows_both_attempts(mock_stack):
    parts = run(mock_stack, "cmp-005").parts
    retried = [p for p in parts if p.retried]
    assert retried and all(len(p.worker_insufficient) == 2 and p.worker_insufficient[0] is True for p in retried)
    assert all(p.answered for p in parts)  # recovered


def test_a_gap_part_is_not_answered_and_says_why(mock_stack):
    parts = run(mock_stack, "cmp-009").parts
    gap = next(p for p in parts if not p.answered)
    assert gap.worker_insufficient and all(gap.worker_insufficient) and gap.judge_grounded == []


def test_diagnostics_hold_no_answer_text(mock_stack):
    blob = json.dumps([p.model_dump() for p in run(mock_stack, "cmp-001").parts])
    assert "Sources:" not in blob and "\n\n" not in blob


# --- the outcome-based live misroute check ----------------------------------------------------------------------------

def test_routed_right_and_answered_passes_with_no_retry(mock_stack):
    run_ = run(mock_stack, "cmp-005").model_copy()
    expected = {q.sub_question: q.expected_entry_id for q in BY_ID["cmp-005"].parts}
    parts = [p.model_copy(update={"retried": False, "citations": [expected[p.text]]}) for p in run_.parts]
    fixed = run_.model_copy(update={"retried_parts": [], "parts": parts})
    assert ce.check_row(BY_ID["cmp-005"], fixed, live=True) == []
    assert any("retry" in p for p in ce.check_row(BY_ID["cmp-005"], fixed))  # the mock criterion still demands the retry


def test_a_misroute_part_not_answered_from_its_expected_entry_fails_live(mock_stack):
    run_ = run(mock_stack, "cmp-005")
    parts = [p.model_copy(update={"citations": ["HR-ONB-02"]}) if p.text == BY_ID["cmp-005"].parts[1].sub_question else p
             for p in run_.parts]
    problems = ce.check_row(BY_ID["cmp-005"], run_.model_copy(update={"parts": parts, "retried_parts": []}), live=True)
    assert any(p.startswith("misroute:") and "expected entry" in p for p in problems)


def test_live_still_fails_a_correctly_routed_part_that_was_retried(mock_stack):
    run_ = run(mock_stack, "cmp-001").model_copy(update={"retried_parts": [BY_ID["cmp-001"].parts[0].sub_question]})
    assert any("correctly routed part" in p for p in ce.check_row(BY_ID["cmp-001"], run_, live=True))


def test_every_other_live_check_is_as_strict_as_in_the_mock(mock_stack):
    good = run(mock_stack, "cmp-001")
    assert any("parallel" in p for p in ce.check_row(BY_ID["cmp-001"], good.model_copy(update={"workers_overlapped": False}), live=True))
    assert any("synthesizer" in p for p in ce.check_row(BY_ID["cmp-001"], good.model_copy(update={"synth_calls": 0}), live=True))
    assert any("isolation" in p for p in ce.check_row(BY_ID["cmp-001"], good.model_copy(update={"leaks": ["x"]}), live=True))


def test_routing_is_reported_separately_for_every_answerable_part(mock_stack):
    runs = {r.id: run(mock_stack, r.id) for r in ROWS}
    stats = ce.routing_stats(ROWS, runs)
    answerable = sum(p.answerable for r in ROWS for p in r.parts)
    assert stats["parts"] == answerable and stats["routed_right"] + stats["routed_wrong"] == answerable
    assert stats["routed_wrong"] == sum(bool(p.likely_wrong_domain) for r in ROWS for p in r.parts)  # the scripted misroutes
    # the mock retry sometimes cites another entry of the right domain (hashing embeddings), so recovery is only bounded here
    assert stats["recovered_by_retry"] + stats["unrecovered"] == stats["routed_wrong"] and stats["recovered_by_retry"] >= 1
    assert stats["not_aligned"] == 0


def test_a_row_with_another_part_count_is_not_aligned_in_the_routing_stats(mock_stack):
    run_ = run(mock_stack, "cmp-001").model_copy(update={"parts": run(mock_stack, "cmp-001").parts[:1]})
    assert ce.routing_stats([BY_ID["cmp-001"]], {"cmp-001": run_})["not_aligned"] == 2


# --- keeping the live trace ---------------------------------------------------------------------------------------

def test_the_trace_of_a_live_run_is_kept_in_the_results_folder(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "trace.jsonl").write_text('{"kind":"request"}\n')
    kept = lg.keep_trace(work, "compound", tmp_path / "results")
    assert kept == tmp_path / "results" / "traces" / "compound.jsonl" and kept.read_text() == '{"kind":"request"}\n'
    (work / "trace.jsonl").write_text('{"kind":"reply"}\n')  # a resumed run adds to the same file
    assert lg.keep_trace(work, "compound", tmp_path / "results").read_text().count("\n") == 2
    assert lg.keep_trace(tmp_path / "nowhere", "x", tmp_path / "results") is None


def test_kept_traces_are_not_committed():
    ignore = (EVALS_DIR.parent / ".gitignore").read_text()
    assert "evals/results/traces/" in ignore
