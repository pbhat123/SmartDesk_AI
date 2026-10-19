"""The orchestrator-only planning check (FR-I3, FR-I4, FR-D7): scoring, the estimate, the spend controls. Mock and scripted
planners only: no test calls a paid model."""
import json

import pytest

from evals import orchestrator_plan_check as pc
from smartdesk_ai.agent.intents import ClassifierError
from smartdesk_ai.agent.state import IntentResult, SubQuestion

CASES = pc.load_cases()
BY_ID = {c.id: c for c in CASES}


def result(domains, intent="kb_query", texts=None):
    parts = [SubQuestion(part_id=f"p{i}", text=(texts or [f"part number {i} of this question?"] * len(domains))[i - 1], domain=d)
             for i, d in enumerate(domains, 1)]
    return IntentResult(intent=intent, confidence=0.95, domain=domains[0] if intent == "kb_query" else None,
                        parts=parts if intent == "kb_query" else [])


class Oracle:
    """A planner that knows the expected answer for every question."""

    def __init__(self, **override):
        self.by_question = {c.question: c for c in CASES}
        self.override = override

    def __call__(self, question):
        case = self.by_question[question]
        if case.id in self.override:
            return self.override[case.id](case)
        return result(case.expected_parts)


def run(planner, cases=CASES, **kw):
    outcomes, stopped = pc.run_cases(cases, planner, **kw)
    return pc.score(cases, outcomes, stopped)


# --- the cases ----------------------------------------------------------------------------------------------------

def test_the_cases_are_the_74_golden_questions_and_the_14_compound_questions():
    kinds = [c.kind for c in CASES]
    assert (kinds.count("kb"), kinds.count("gap"), kinds.count("compound")) == (53, 21, 14)
    assert all(len(c.expected_parts) == 1 for c in CASES if c.kind != "compound")
    assert all(len(c.expected_parts) >= 2 for c in CASES if c.kind == "compound")
    assert len({c.id for c in CASES}) == 88


# --- scoring ------------------------------------------------------------------------------------------------------

def test_a_perfect_orchestrator_scores_everything_right():
    report = run(Oracle())
    assert (report.intent_accuracy, report.domain_accuracy, report.split_rate) == (1.0, 1.0, 0.0)
    assert report.compound_correct == 14 and report.split_single == [] and report.errors == []
    assert report.single_topic == 74 and report.n_answered == 88


def test_a_wrong_intent_is_counted_and_listed():
    report = run(Oracle(**{"kb-001": lambda c: result(["IT"], intent="smalltalk")}))
    assert report.intent_accuracy == pytest.approx(87 / 88, abs=1e-3) and report.wrong_intent == ["kb-001"]


def test_a_wrong_domain_on_a_single_topic_question_is_counted():
    wrong = lambda c: result(["HR" if c.expected_parts[0] == "IT" else "IT"])  # noqa: E731
    report = run(Oracle(**{"kb-001": wrong, "gap-001": wrong}))
    assert report.wrong_domain == ["kb-001", "gap-001"] and report.domain_accuracy == pytest.approx(72 / 74, abs=1e-3)
    assert report.domain_accuracy_by_kind["kb"] == pytest.approx(52 / 53, abs=1e-3) and report.domain_accuracy_by_kind["gap"] == pytest.approx(20 / 21, abs=1e-3)


def test_the_domain_of_a_single_question_comes_from_its_first_part_or_else_the_message():
    no_parts = IntentResult(intent="kb_query", confidence=0.9, domain=BY_ID["kb-001"].expected_parts[0], parts=[])
    report = run(Oracle(**{"kb-001": lambda c: no_parts}))
    assert report.domain_accuracy == 1.0


def test_a_single_topic_question_split_in_two_is_a_split_and_is_listed_with_its_parts():
    two = lambda c: result([c.expected_parts[0], c.expected_parts[0]], texts=["what is the first half of this?", "what is the second half of this?"])  # noqa: E731
    report = run(Oracle(**{"kb-045": two, "gap-005": two}))
    assert [o.id for o in report.split_single] == ["kb-045", "gap-005"]
    assert report.split_rate == pytest.approx(2 / 74, abs=1e-3)
    assert [p.text for p in report.split_single[0].parts] == ["what is the first half of this?", "what is the second half of this?"]
    assert "split kb-045" in pc.render(report) and "what is the first half of this?" in pc.render(report)


def test_compound_questions_are_judged_on_count_and_domains_in_order():
    def swapped(case):
        return result(list(reversed(case.expected_parts)))

    def merged(case):
        return result([case.expected_parts[0]])

    def extra(case):
        return result([*case.expected_parts, "MISC"])

    report = run(Oracle(**{"cmp-001": swapped, "cmp-002": merged, "cmp-003": extra}))
    rows = {c.id: c for c in report.compound}
    assert rows["cmp-001"].correct_split is False and rows["cmp-001"].same_domains and rows["cmp-001"].right_count
    assert rows["cmp-002"].correct_split is False and not rows["cmp-002"].right_count
    assert rows["cmp-003"].correct_split is False and not rows["cmp-003"].right_count
    assert report.compound_correct == 11 and rows["cmp-004"].correct_split


def test_compound_questions_do_not_count_toward_the_single_topic_numbers():
    report = run(Oracle(**{"cmp-001": lambda c: result(["HR"])}))
    assert report.single_topic == 74 and report.domain_accuracy == 1.0 and report.split_rate == 0.0


def test_a_failed_call_is_an_error_not_a_wrong_answer():
    def boom(case):
        raise ClassifierError("bad reply")

    report = run(Oracle(**{"kb-001": boom, "cmp-001": boom}))
    assert report.errors == ["kb-001", "cmp-001"] and report.n_answered == 86
    assert report.intent_accuracy == 1.0 and report.single_topic == 73


# --- stopping -----------------------------------------------------------------------------------------------------

def test_a_run_stops_before_the_next_question_when_told_to():
    asked = []
    planner = Oracle()

    def counting(question):
        asked.append(question)
        return planner(question)

    outcomes, stopped = pc.run_cases(CASES, counting, stop=lambda: "enough" if len(asked) >= 5 else None)
    assert len(outcomes) == 5 and stopped == "enough" and len(asked) == 5
    report = pc.score(CASES[:5], outcomes, stopped)
    assert report.n_cases == 5 and report.stopped == "enough" and "stopped: enough" in pc.render(report)


# --- the estimate -------------------------------------------------------------------------------------------------

def test_the_estimate_is_priced_from_config_and_small(settings):
    e = pc.estimate(settings, CASES, 0.25)
    assert e.calls == 88 and 0.05 < e.usd < 0.25 and e.ceiling_usd == 0.25
    main = settings.models.main
    assert e.usd == pytest.approx((e.input_tokens * main.price.input + e.output_tokens * main.price.output) / 1e6, abs=1e-4)
    assert 88 * 800 < e.input_tokens < 88 * 1500, "about the orchestrator prompt plus the question and the schema per call"
    text = "\n".join(pc.describe(e))
    assert "88 Main-tier calls" in text and "$0.25 spent" in text and "spend_ledger" in text


# --- the command line ---------------------------------------------------------------------------------------------

@pytest.fixture
def no_paid_calls(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a paid model was called")

    monkeypatch.setattr(pc, "live_planner", refuse)
    monkeypatch.setattr(pc, "run_live", refuse)


def test_estimate_prints_and_stops(capsys, no_paid_calls):
    assert pc.main(["--estimate"]) == 0
    assert "Estimate for the live orchestrator planning check" in capsys.readouterr().out


def test_a_live_run_without_confirm_spend_runs_nothing(capsys, no_paid_calls):
    assert pc.main([]) == 2
    out = capsys.readouterr().out
    assert "Nothing was run" in out and "--confirm-spend" in out


def test_the_ceiling_option_reaches_the_estimate(capsys, no_paid_calls):
    pc.main(["--estimate", "--ceiling", "0.40"])
    assert "$0.40 spent" in capsys.readouterr().out


def test_the_default_ceiling_is_25_cents():
    assert pc.DEFAULT_CEILING_USD == 0.25


def test_mock_runs_every_question_with_the_mock_orchestrator(capsys, no_paid_calls):
    assert pc.main(["--mock"]) == 0
    out = capsys.readouterr().out
    assert "88 of 88 questions answered" in out and "MOCK run" in out and "intent accuracy" in out


def test_the_mock_orchestrator_keeps_single_topic_questions_whole(settings):
    outcomes, _ = pc.run_cases(CASES, pc.mock_planner(settings))
    report = pc.score(CASES, outcomes)
    assert report.split_rate == 0.0 and report.intent_accuracy > 0.9  # keyword rules: plumbing, not accuracy


def test_the_report_is_saved_as_json(tmp_path):
    report = run(Oracle())
    path = pc.save(report, tmp_path)
    assert path.name == "orchestrator_plan_check.json"
    saved = json.loads(path.read_text())
    assert saved["compound_correct"] == 14 and saved["split_rate"] == 0.0


# --- the live path is wired to the spend controls (never exercised here) -----------------------------------------

def test_the_live_path_starts_a_ledger_run_and_uses_the_watchdog_and_the_ceiling():
    import inspect

    source = inspect.getsource(pc.run_live)
    for needle in ("start_run(", '"eval"', "Watchdog(", "over_ceiling", "end_run(status)", "current_run()"):
        assert needle in source
    assert "openai_builder" in inspect.getsource(pc.live_planner)


# --- the check applies the same guard as the app (fragments merged back), raw and guarded side by side ----------------

def fragments(case):
    return result([case.expected_parts[0]] * 2, texts=["What are the possible performance scores?", "when are promotions decided?"])


def test_a_fragment_split_is_merged_back_by_the_app_guard_and_listed_only_as_raw():
    report = run(Oracle(**{"kb-045": fragments}))
    assert report.split_single == [] and report.split_rate == 0.0
    assert report.split_single_raw == ["kb-045"] and report.split_rate_raw == pytest.approx(1 / 74, abs=1e-4)
    assert "raw (before the guard" in pc.render(report) and "kb-045" in pc.render(report)


def test_a_real_two_part_split_counts_in_both_lists():
    two = lambda c: result([c.expected_parts[0]] * 2, texts=["What is the probation period?", "Does the company offer a sabbatical?"])  # noqa: E731
    report = run(Oracle(**{"kb-001": two}))
    assert [o.id for o in report.split_single] == ["kb-001"] and report.split_single_raw == ["kb-001"]


def test_the_guarded_parts_are_what_is_scored_for_compound_questions():
    merged = lambda c: result(list(c.expected_parts) + [c.expected_parts[-1]], texts=[  # noqa: E731
        *[f"A complete question number {i} here?" for i in range(len(c.expected_parts))], "and for how long?"])
    report = run(Oracle(**{"cmp-001": merged}))
    row = next(c for c in report.compound if c.id == "cmp-001")
    assert row.right_count and row.correct_split and not row.raw_right_count  # the app merges the extra fragment away
    assert report.compound_right_count == 14 and report.compound_right_count_raw == 13


def test_compound_count_and_domains_are_reported_apart():
    swapped = lambda c: result(list(reversed(c.expected_parts)))  # noqa: E731
    report = run(Oracle(**{"cmp-001": swapped}))
    assert report.compound_right_count == 14 and report.compound_correct == 13


def test_a_failed_call_has_no_raw_parts():
    boom = lambda c: (_ for _ in ()).throw(ClassifierError("bad"))  # noqa: E731
    report = run(Oracle(**{"kb-001": boom}))
    assert report.errors == ["kb-001"] and "kb-001" not in report.split_single_raw
