"""T6.2: RAG metrics on the KB set (FR-H12): answer correctness, retrieval precision/recall/F1 at k,
hallucination rate, citation accuracy and factual precision. The judge is a lexical mock here; the Ragas
judge is checked structurally with stand-in metric objects, so no model is called."""
import json
from pathlib import Path

import pytest

from evals import rag_metrics as rm
from evals.harness import make_resources
from evals.schemas import KBQuestion
from smartdesk_ai.rag.index import open_index
from smartdesk_ai.rag.ingest import ingest
from tests.kb_helpers import KB_DIR


def row(i, expected, reference="The portal resets your password after you approve the prompt on your phone."):
    return KBQuestion(id=f"q{i}", question=f"question number {i} here?", expected_entry_id=expected,
                      expected_domain="IT", type="simple", reference_answer=reference, split="dev")


def out(i, retrieved, citations=(), answer="", contexts=("ctx",)):
    return rm.RagOutput(question_id=f"q{i}", answer=answer, insufficient=not answer, citations=list(citations),
                        retrieved=list(retrieved), contexts=list(contexts))


# --- retrieval precision, recall and F1 at k ---------------------------------------------------

def test_precision_recall_f1_at_k_against_the_expected_entry():
    rows = [row(1, "A"), row(2, "B"), row(3, "C")]
    outs = [out(1, ["A", "X", "Y"]), out(2, ["X", "B", "Y"]), out(3, ["X", "Y", "Z"])]
    scores = rm.retrieval_scores(outs, rows, ks=(1, 3))
    # k=1: hits 1,0,0 -> P=R=1/3 ; k=3: each question retrieves 3, hits 1,1,0 -> P=2/9, R=2/3
    assert scores[1] == pytest.approx({"precision": 1 / 3, "recall": 1 / 3, "f1": 1 / 3})
    assert scores[3]["precision"] == pytest.approx(2 / 9) and scores[3]["recall"] == pytest.approx(2 / 3)
    assert scores[3]["f1"] == pytest.approx(1 / 3)  # (0.5 + 0.5 + 0) / 3


def test_fewer_results_than_k_divides_by_what_was_retrieved():
    scores = rm.retrieval_scores([out(1, ["A"])], [row(1, "A")], ks=(3,))
    assert scores[3] == pytest.approx({"precision": 1.0, "recall": 1.0, "f1": 1.0})


def test_nothing_retrieved_scores_zero():
    scores = rm.retrieval_scores([out(1, [])], [row(1, "A")], ks=(1,))
    assert scores[1] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_repeated_entries_count_once():
    scores = rm.retrieval_scores([out(1, ["A", "A", "B"])], [row(1, "A")], ks=(2,))
    assert scores[2]["precision"] == pytest.approx(0.5)  # the top two distinct entries are A and B


# --- citation accuracy ---------------------------------------------------------------------------

def test_citation_accuracy_counts_only_retrieved_and_supporting_ids():
    rows = [row(1, "A"), row(2, "B"), row(3, "C")]
    outs = [
        out(1, ["A", "X"], citations=["A"], answer="a"),        # retrieved and supports: correct
        out(2, ["X", "Y"], citations=["B"], answer="b"),        # supports but was not retrieved: wrong
        out(3, ["C", "X"], citations=["C", "X"], answer="c"),   # C correct, X retrieved but does not support: wrong
    ]
    accuracy, cited = rm.citation_accuracy(outs, rows)
    assert (accuracy, cited) == (pytest.approx(2 / 4), 4)


def test_citation_accuracy_is_none_when_nothing_was_cited():
    assert rm.citation_accuracy([out(1, ["A"])], [row(1, "A")]) == (None, 0)


def test_unanswered_questions_are_ignored_by_citation_accuracy():
    outs = [out(1, ["A"], citations=["A"], answer="a"), out(2, ["X"], citations=["B"], answer="")]
    assert rm.citation_accuracy(outs, [row(1, "A"), row(2, "B")]) == (1.0, 1)


# --- judge-based metrics, with the mock judge ----------------------------------------------------------

REF = "The portal resets your password after you approve the prompt on your phone."


def test_mock_judge_scores_a_faithful_answer_high_and_an_invented_one_low():
    judge = rm.MockJudge()
    ctx = ["Reset your password in the portal after you approve the prompt on your phone."]
    assert judge.faithfulness("q", "Reset your password in the portal.", ctx) == 1.0
    assert judge.faithfulness("q", "Reset your password in the portal. Call the director immediately tonight.", ctx) == 0.5
    assert judge.answer_correctness("q", REF, REF) == 1.0 and judge.answer_correctness("q", "bananas oranges", REF) == 0.0
    assert judge.factual_precision("The portal resets your password.", REF) == 1.0
    assert judge.factual_precision("Bananas are yellow fruit.", REF) == 0.0


def test_metrics_are_computed_for_a_tiny_set_with_the_mock_judge():
    rows = [row(1, "A"), row(2, "B")]
    outs = [
        out(1, ["A", "X"], ["A"], answer="The portal resets your password.", contexts=[REF]),
        out(2, ["B"], ["B"], answer="Call the director immediately tonight about bananas.", contexts=[REF]),
    ]
    report = rm.compute_metrics(outs, rows, rm.MockJudge(), ks=(1, 3))
    assert report.n_questions == 2 and report.n_answered == 2
    assert report.hallucination_rate == 0.5 and report.grounded_answer_rate == pytest.approx(0.5)
    assert report.citation_accuracy == 1.0 and report.n_cited == 2
    assert 0 < report.answer_correctness < 1 and 0 < report.factual_precision < 1
    assert report.retrieval[1]["recall"] == 1.0


def test_unanswered_questions_count_in_retrieval_but_not_in_answer_metrics():
    rows = [row(1, "A"), row(2, "B")]
    outs = [out(1, ["A"], ["A"], answer="The portal resets your password.", contexts=[REF]), out(2, ["B"])]
    report = rm.compute_metrics(outs, rows, rm.MockJudge(), ks=(1,))
    assert (report.n_questions, report.n_answered) == (2, 1) and report.retrieval[1]["recall"] == 1.0
    assert report.hallucination_rate == 0.0


def test_no_answers_at_all_gives_none_not_a_crash():
    report = rm.compute_metrics([out(1, ["A"])], [row(1, "A")], rm.MockJudge(), ks=(1,))
    assert report.n_answered == 0 and report.answer_correctness is None and report.hallucination_rate is None


# --- the real pipeline in mock mode -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    res = make_resources(mock=True)
    index_dir = tmp_path_factory.mktemp("rag_index")
    ingest(KB_DIR, index_dir, res.embeddings, res.settings.chunking)
    index = open_index(index_dir, res.embeddings)
    out_dir = tmp_path_factory.mktemp("rag_out")
    report = rm.run(res, index, out_dir)
    return report, out_dir


def test_the_mock_run_covers_the_whole_kb_set_and_writes_a_result_file(mock_run):
    report, out_dir = mock_run
    data = json.loads((out_dir / "rag_metrics.json").read_text())
    assert report.n_questions == 53 == data["n_questions"] and data["mock"] is True
    assert 0 < report.n_answered <= 53
    assert set(data["retrieval"]) == {"1", "3", "5"} and data["judge"] == "mock (lexical)"
    assert 0 <= report.hallucination_rate <= 1 and 0 <= report.citation_accuracy <= 1


def test_the_answers_cite_only_retrieved_entries(mock_run):
    report, _ = mock_run
    assert report.citation_accuracy is not None and report.n_cited > 0
    assert all(o.citations and set(o.citations) <= set(o.retrieved) for o in report.outputs if not o.insufficient)


# --- the Ragas judge (structure only) --------------------------------------------------------------------------

class FakeMetric:
    def __init__(self, score):
        self.score, self.samples, self.kwargs = score, [], {}

    def single_turn_score(self, sample):
        self.samples.append(sample)
        return self.score


def test_the_ragas_judge_maps_our_fields_onto_ragas_samples():
    faith, correct, factual = FakeMetric(0.75), FakeMetric(0.5), FakeMetric(0.25)
    judge = rm.RagasJudge(faithfulness=faith, answer_correctness=correct, factual_precision=factual)
    assert judge.faithfulness("the question", "the answer", ["c1", "c2"]) == 0.75
    assert judge.answer_correctness("the question", "the answer", "the reference") == 0.5
    assert judge.factual_precision("the answer", "the reference") == 0.25
    s = faith.samples[0]
    assert (s.user_input, s.response, s.retrieved_contexts) == ("the question", "the answer", ["c1", "c2"])
    assert correct.samples[0].reference == "the reference" and factual.samples[0].response == "the answer"


def test_a_ragas_score_that_is_nan_counts_as_missing():
    judge = rm.RagasJudge(faithfulness=FakeMetric(float("nan")), answer_correctness=FakeMetric(1.0),
                          factual_precision=FakeMetric(1.0))
    assert judge.faithfulness("q", "a", ["c"]) is None


def test_the_ragas_metrics_are_built_with_the_pinned_classes_and_precision_mode(settings):
    metrics = rm.build_ragas_metrics(llm=object(), embeddings=object())
    assert type(metrics["faithfulness"]).__name__ == "Faithfulness"
    assert type(metrics["answer_correctness"]).__name__ == "AnswerCorrectness"
    assert type(metrics["factual_precision"]).__name__ == "FactualCorrectness"
    assert metrics["factual_precision"].mode == "precision"


def test_ragas_imports_in_this_environment():
    from evals.ragas_compat import import_ragas

    assert import_ragas().__version__ == "0.4.3"


def test_a_real_judge_run_needs_explicit_spend_confirmation(capsys):
    from evals.run import main

    assert main(["--only", "rag", "--judge", "ragas"]) == 1
    assert "--confirm-spend" in capsys.readouterr().err


def test_the_readme_names_every_metric_and_any_substituted_ragas_class():
    text = (Path(rm.__file__).parent / "README.md").read_text().lower()
    for needle in ("answercorrectness", "faithfulness", "factualcorrectness", "precision", "citation accuracy",
                   "hallucination rate", "ragas_compat"):
        assert needle in text


# --- the real run's plumbing (no model is called) ------------------------------------------------------------

def test_the_usage_counter_adds_up_tokens_and_prices_them_with_the_main_model(settings):
    from types import SimpleNamespace

    counter = rm.UsageCounter()
    message = SimpleNamespace(usage_metadata={"input_tokens": 1000, "output_tokens": 200})
    response = SimpleNamespace(generations=[[SimpleNamespace(message=message)], [SimpleNamespace(message=message)]])
    counter.on_llm_end(response)
    assert (counter.calls, counter.input_tokens, counter.output_tokens) == (2, 2000, 400)
    price = settings.models.main.price
    assert counter.cost_usd(settings) == pytest.approx((2000 * price.input + 400 * price.output) / 1e6)


def test_the_usage_counter_stops_the_run_past_its_spend_ceiling(settings):
    from types import SimpleNamespace

    counter = rm.UsageCounter(settings, ceiling_usd=0.001)
    message = SimpleNamespace(usage_metadata={"input_tokens": 1000, "output_tokens": 200})
    response = SimpleNamespace(generations=[[SimpleNamespace(message=message)]])
    with pytest.raises(rm.SpendCeilingExceeded):
        counter.on_llm_end(response)
    assert counter.raise_error is True  # LangChain would otherwise swallow the exception


def test_a_confirmed_real_run_goes_to_run_real(monkeypatch, tmp_path):
    from evals import run as run_module

    seen = {}
    monkeypatch.setattr(run_module.rag_metrics, "run_real",
                        lambda res, index, out_dir, **kw: seen.update(out_dir=out_dir, **kw) or None)
    monkeypatch.setattr(run_module, "make_resources", lambda **kw: make_resources(mock=True))
    monkeypatch.setattr(run_module, "get_index", lambda res, d: None)
    assert run_module.main(["--only", "rag", "--judge", "ragas", "--confirm-spend", "--out-dir", str(tmp_path)]) == 0
    assert seen["out_dir"] == tmp_path and seen["ceiling_usd"] is None


def test_the_ceiling_option_reaches_the_real_run(monkeypatch, tmp_path):
    from evals import run as run_module

    seen = {}
    monkeypatch.setattr(run_module.rag_metrics, "run_real", lambda res, index, out_dir, **kw: seen.update(kw) or None)
    monkeypatch.setattr(run_module, "make_resources", lambda **kw: make_resources(mock=True))
    monkeypatch.setattr(run_module, "get_index", lambda res, d: None)
    run_module.main(["--only", "rag", "--judge", "ragas", "--confirm-spend", "--ceiling-usd", "1.4", "--out-dir", str(tmp_path)])
    assert seen["ceiling_usd"] == 1.4


def test_a_real_rag_run_with_the_mock_judge_is_refused(capsys):
    from evals.run import main

    assert main(["--only", "rag"]) == 1
    assert "--judge ragas" in capsys.readouterr().err


# --- answer relevancy, context precision and context recall; judge errors; the AnswerSimilarity fix --------------------

def test_the_mock_judge_scores_the_three_added_metrics():
    judge = rm.MockJudge()
    assert judge.answer_relevancy("how do I reset my password", "Reset your password in the portal.") > 0.0
    assert judge.answer_relevancy("how do I reset my password", "Bananas are yellow fruit.") == 0.0
    reference = "Reset the password in the portal."
    assert judge.context_recall("q", ["The portal resets a password for you."], reference) > 0.5
    assert judge.context_recall("q", ["Unrelated parking text."], reference) == 0.0
    assert judge.context_precision("q", ["Reset the password in the portal.", "Parking text about cars."], reference) == 0.5
    assert judge.context_precision("q", [], reference) == 0.0


def test_the_report_carries_the_three_added_metrics_over_the_right_questions():
    rows = [row(1, "A"), row(2, "A")]
    outs = [out(1, ["A"], ["A"], answer="Reset your password in the portal.", contexts=["Reset your password in the portal."]),
            out(2, ["A"], contexts=["Reset your password in the portal."])]  # escalated: no answer shown
    report = rm.compute_metrics(outs, rows, rm.MockJudge(), ks=(1,))
    assert report.answer_relevancy is not None and report.context_precision is not None and report.context_recall is not None
    assert report.context_recall > 0.0  # retrieval metrics cover both questions, answer relevancy only the answered one


def test_the_ragas_judge_maps_the_added_metrics_onto_ragas_samples():
    rel, prec, rec = FakeMetric(0.6), FakeMetric(0.7), FakeMetric(0.8)
    judge = rm.RagasJudge(faithfulness=FakeMetric(1.0), answer_correctness=FakeMetric(1.0), factual_precision=FakeMetric(1.0),
                          answer_relevancy=rel, context_precision=prec, context_recall=rec)
    assert judge.answer_relevancy("the question", "the answer") == 0.6
    assert judge.context_precision("the question", ["c1"], "the reference") == 0.7
    assert judge.context_recall("the question", ["c1"], "the reference") == 0.8
    assert (prec.samples[0].reference, prec.samples[0].retrieved_contexts) == ("the reference", ["c1"])
    assert rel.samples[0].response == "the answer"


def test_the_ragas_metrics_include_the_added_classes_and_answer_correctness_has_its_similarity_metric():
    metrics = rm.build_ragas_metrics(llm=object(), embeddings=object())
    assert type(metrics["answer_relevancy"]).__name__ == "ResponseRelevancy"
    assert type(metrics["context_precision"]).__name__ == "LLMContextPrecisionWithReference"
    assert type(metrics["context_recall"]).__name__ == "LLMContextRecall"
    assert metrics["answer_correctness"].answer_similarity is not None  # "AnswerSimilarity must be set" in the first live run


class Raises:
    def __init__(self, error):
        self.error, self.samples = error, []

    def single_turn_score(self, sample):
        raise self.error


def test_one_failing_sample_is_counted_not_fatal_but_a_systematic_failure_stops_the_run():
    judge = rm.RagasJudge(faithfulness=FakeMetric(1.0), answer_correctness=Raises(ValueError("parse")),
                          factual_precision=FakeMetric(1.0))
    assert judge.answer_correctness("q", "a", "r") is None and judge.errors == {"answer_correctness": 1}
    judge.answer_correctness("q", "a", "r")
    with pytest.raises(RuntimeError, match="answer_correctness"):
        judge.answer_correctness("q", "a", "r")  # the first three calls all failed: not flaky, broken


def test_the_spend_ceiling_is_never_swallowed_by_the_error_counter(settings):
    judge = rm.RagasJudge(faithfulness=Raises(rm.SpendCeilingExceeded("over")), answer_correctness=FakeMetric(1.0),
                          factual_precision=FakeMetric(1.0))
    with pytest.raises(rm.SpendCeilingExceeded):
        judge.faithfulness("q", "a", ["c"])


def test_the_pipeline_answers_are_saved_before_the_judge_runs(settings, tmp_path):
    class Boom:
        name = "boom"

        def __getattr__(self, name):
            raise RuntimeError("judge down")

    res = make_resources(mock=True)

    def answer_fn(r):
        o = out(1, ["A"], ["A"], answer="An answer.")
        o.question_id = r.id
        return o

    with pytest.raises(RuntimeError):
        rm.run(res, None, tmp_path, judge=Boom(), answer_fn=answer_fn, split="dev")
    saved = json.loads((tmp_path / "rag_outputs.json").read_text())
    assert len(saved) >= 1  # the paid part of the run is not lost when the judge fails


# --- expected entry cited (FR-H12) -----------------------------------------------------------------------------

def test_expected_entry_cited_counts_answers_that_cite_the_entry_the_question_was_built_from():
    rows = [row(1, "A"), row(2, "B"), row(3, "C"), row(4, "D")]
    outs = [out(1, ["A", "X"], citations=["A", "X"], answer="a"),   # also cites an extra entry: still counts
            out(2, ["B"], citations=["X"], answer="a"),             # cites something else
            out(3, ["C"], citations=[], answer="a"),                # cites nothing
            out(4, ["D"])]                                          # escalated: no answer, not counted
    assert rm.expected_entry_cited(outs, rows) == 1
    assert rm.citation_accuracy(outs, rows)[0] == pytest.approx(1 / 3)  # 3 cited ids, 1 is the expected entry: the strict metric penalises the extras


def test_the_report_and_the_result_file_carry_expected_entry_cited_next_to_citation_accuracy(mock_run, capsys):
    report, out_dir = mock_run
    data = json.loads((out_dir / "rag_metrics.json").read_text())
    assert 0 <= report.expected_cited <= report.n_answered and data["expected_cited"] == report.expected_cited
    rm._print(report)
    assert f"expected entry cited {report.expected_cited}/{report.n_answered}" in capsys.readouterr().out
