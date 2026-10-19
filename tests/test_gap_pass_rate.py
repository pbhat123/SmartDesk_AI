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


def test_a_live_gap_run_needs_explicit_spend_confirmation(capsys):
    assert main(["--only", "gap"]) == 1
    assert "--confirm-spend" in capsys.readouterr().err


def test_a_confirmed_live_gap_run_uses_the_live_stack_with_mock_tickets(monkeypatch, tmp_path):
    from evals import gap_eval, run as run_module

    seen = {}
    monkeypatch.setattr(gap_eval, "build_live_stack", lambda settings, work, tracing=False: seen.update(live=True) or stack_with(
        settings, tmp_path, FakeAnswerLLM("insufficient")))
    monkeypatch.setattr(run_module, "make_resources", lambda **kw: make_resources(mock=True).__class__(
        settings=make_resources(mock=True).settings, embeddings=make_resources(mock=True).embeddings,
        reranker=None, mock=False))
    monkeypatch.setattr(run_module, "get_index", lambda res, d: None)
    assert run_module.main(["--only", "gap", "--confirm-spend", "--out-dir", str(tmp_path / "out")]) == 0
    assert seen == {"live": True}


# --- "answered with citations" is reported next to the KB answer rate, which keeps its definition ------------------

def _state(intent="kb_query", cited=(), offer=False, parts=None):
    from types import SimpleNamespace

    result = lambda c: SimpleNamespace(citations=list(c))  # noqa: E731
    return SimpleNamespace(intent=SimpleNamespace(intent=intent), part_result=result(cited) if cited else None,
                           part_results={f"p{i}": result(c) for i, c in enumerate(parts, 1)} if parts else {},
                           pending_offer=object() if offer else None)


def test_a_cited_answer_with_no_offer_counts_as_answered_with_citations():
    assert ge.answered_with_citations(_state(cited=["IT-1"])) is True


def test_a_deflection_an_offer_or_an_empty_result_does_not():
    assert ge.answered_with_citations(_state(intent="out_of_scope", cited=["IT-1"])) is False
    assert ge.answered_with_citations(_state(cited=["IT-1"], offer=True)) is False
    assert ge.answered_with_citations(_state()) is False


def test_a_split_question_counts_only_when_every_part_is_cited():
    assert ge.answered_with_citations(_state(parts=[["IT-1"], ["HR-1"]])) is True
    assert ge.answered_with_citations(_state(parts=[["IT-1"], []])) is False


def test_the_report_gives_both_rates_and_the_old_one_is_unchanged():
    def kb(id, outcome, cited):
        return ge.Case(id=id, label="kb", split="dev", outcome=outcome, cited=cited, question="q")

    cases = [kb("a", "answered", True), kb("b", "escalated", False), kb("c", "deflected", False), kb("d", "other", False)]
    report = ge.summarize(cases)
    assert report.kb_answer_rate == pytest.approx(3 / 4)  # anything but escalated, as before
    assert report.kb_answered_with_citations == pytest.approx(1 / 4)
    assert report.kb_not_cited == ["b", "c", "d"]
