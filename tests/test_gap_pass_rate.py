"""T6.4: gap-set pass rate and escalation precision and recall, end to end in mock mode (SPEC section 7,
FR-A5, FR-A15, FR-D9). A gap question must end in "I don't know" and a ticket offer: never a made-up answer
and never a deflection. The pipeline is run with a mock answer model that reports insufficient context, as
a sound model does for a question the KB lacks; the lexical default mock cannot judge sufficiency, and the
last tests show the metric does catch it answering."""
import json

import pytest

from evals import gap_eval as ge
from evals.harness import gap_rows, kb_rows, make_resources
from evals.run import main
from smartdesk_ai.agent.factory import build_stack
from tests.llm_fakes import FakeAnswerLLM


def stack_with(settings, tmp_path, model):
    return build_stack(settings, environ={}, dotenv_path=None, data_dir=tmp_path, mock_answer_model=model)


@pytest.fixture(scope="module")
def insufficient_cases(settings, tmp_path_factory):
    stack = stack_with(settings, tmp_path_factory.mktemp("gap_insufficient"), FakeAnswerLLM("insufficient"))
    return ge.run_cases(stack)


def test_every_gap_question_ends_in_a_ticket_offer(insufficient_cases):
    gaps = [c for c in insufficient_cases if c.label == "gap"]
    assert len(gaps) == len(gap_rows()) >= 15
    assert [c.id for c in gaps if c.outcome != "escalated"] == []


def test_no_gap_question_is_answered_or_deflected(insufficient_cases):
    outcomes = {c.outcome for c in insufficient_cases if c.label == "gap"}
    assert outcomes == {"escalated"}


def test_the_gap_pass_rate_is_100_percent(insufficient_cases):
    report = ge.summarize(insufficient_cases)
    assert report.gap_pass_rate == 1.0 and report.failures == []
    assert report.escalation_recall == 1.0 and report.grey_zone_pass_rate == 1.0
    assert set(report.by_split) == {"dev", "test"} and all(s["gap_pass_rate"] == 1.0 for s in report.by_split.values())


def test_a_model_that_never_answers_escalates_the_kb_questions_too(insufficient_cases):
    report = ge.summarize(insufficient_cases)
    assert report.n_kb == len(kb_rows()) and report.kb_answer_rate == 0.0
    assert report.escalation_precision == pytest.approx(report.n_gap / (report.n_gap + report.n_kb))


def test_the_default_mock_model_is_lexical_and_the_metric_reports_where_it_answers(settings, tmp_path):
    report = ge.summarize(ge.run_cases(build_stack(settings, environ={}, dotenv_path=None, data_dir=tmp_path)))
    assert 0.5 < report.gap_pass_rate < 1.0  # a word-overlap mock cannot tell a grey-zone gap from an answerable question
    assert all(f.endswith("answered") for f in report.failures)


def test_a_deflected_gap_question_is_a_failure():
    from types import SimpleNamespace

    state = SimpleNamespace(intent=SimpleNamespace(intent="out_of_scope"), part_result=None, pending_offer=None)
    assert ge.classify(state) == "deflected"
    case = ge.Case(id="g", label="gap", split="dev", outcome="deflected", question="q")
    assert ge.summarize([case]).failures == ["g: deflected"]


def test_precision_and_recall_from_hand_counted_cases():
    def case(id, label, outcome, grey=False):
        return ge.Case(id=id, label=label, split="dev", grey_zone=grey, outcome=outcome, question="q")

    cases = [case("g1", "gap", "escalated"), case("g2", "gap", "escalated", True), case("g3", "gap", "answered", True),
             case("g4", "gap", "deflected"), case("k1", "kb", "answered"), case("k2", "kb", "escalated")]
    report = ge.summarize(cases)
    assert report.gap_pass_rate == pytest.approx(2 / 4) and report.escalation_recall == pytest.approx(2 / 4)
    assert report.escalation_precision == pytest.approx(2 / 3)  # three escalations, two of them gap questions
    assert report.kb_answer_rate == pytest.approx(1 / 2) and report.grey_zone_pass_rate == pytest.approx(1 / 2)
    assert report.failures == ["g3: answered", "g4: deflected"]


def test_the_cli_step_writes_the_result_file(tmp_path):
    assert main(["--mock", "--only", "gap", "--out-dir", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "gap_eval.json").read_text())
    assert data["mock"] is True and data["n_gap"] == len(gap_rows()) and "cases" in data
    assert json.loads((tmp_path / "run_record.json").read_text())["result_files"] == ["gap_eval.json"]
