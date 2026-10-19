"""Orchestrator-only planning check (FR-I3, FR-I4, FR-D7): does the orchestrator classify, tag and split the golden
questions as expected? No workers, no retrieval, no tools, no tickets: one orchestrator call per question, each on its own
(no history).

The 74 golden questions (53 KB, 21 gap) are single-topic questions: the expected intent is `kb_query`, the expected domain
the row's, and the expected number of parts is one (a two-clause question about one topic must not be split). The 14
compound questions each have an expected list of part domains: a correct split has that many parts with those domains in
order. Reported: intent accuracy, domain accuracy, the split rate on single-topic questions (and which were split), and
the correct splits.

    python -m evals.orchestrator_plan_check --mock                      free, mock orchestrator (a plumbing check)
    python -m evals.orchestrator_plan_check --estimate                  print the cost estimate and stop
    python -m evals.orchestrator_plan_check --confirm-spend [--ceiling 0.25]   the paid run

A live run needs `--confirm-spend`, runs under a spend ceiling read from the spend ledger (`data/spend_ledger.jsonl`, source
`eval`) after every question, and a stall watchdog. Every call goes through `FallbackChat` (the orchestrator task).
"""
import argparse
import json
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import tiktoken
from pydantic import BaseModel

from evals.common import EVALS_DIR, load_jsonl
from evals.harness import RESULTS_DIR, gap_rows, kb_rows, live_settings
from evals.judge_run import EXIT_STALLED, Watchdog
from evals.schemas import CompoundRow
from smartdesk_ai.agent.intents import ClassifierError, make_classifier
from smartdesk_ai.agent.orchestrator import plan_parts
from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.agent.state import Category, IntentResult
from smartdesk_ai.config.settings import Settings, load_settings

RESULT_FILE = "orchestrator_plan_check.json"
DEFAULT_CEILING_USD = 0.25
STALL_SECONDS = 120  # one orchestrator call takes seconds; two minutes with no progress is a hang
FALLBACK_ENCODING = "o200k_base"
# The structured-output schema the live client sends with every call (the reply shape with parts). An assumption: the
# ledger gives the real figure after the run.
STRUCTURED_SCHEMA_TOKENS = 150
Kind = Literal["kb", "gap", "compound"]
Planner = Callable[[str], IntentResult]  # one message, no history -> the orchestrator's classification


class Case(BaseModel):
    id: str
    kind: Kind
    question: str
    expected_parts: list[Category]  # one domain for a single-topic question, one per part for a compound one


class PartOut(BaseModel):
    text: str
    domain: Category


class Outcome(BaseModel):
    id: str
    kind: Kind
    question: str
    intent: str | None  # None: the call failed or its reply was invalid
    domain: Category | None  # the message-level domain
    parts: list[PartOut]  # what the system answers: the orchestrator's parts after the same `plan_parts` guard the app applies
    raw_parts: list[PartOut] = []  # the orchestrator's own parts, before that guard (fragments not yet merged back)

    @property
    def predicted_domains(self) -> list[Category | None]:
        return [p.domain for p in self.parts] or [self.domain]


class CompoundResult(BaseModel):
    id: str
    expected: list[Category]
    got: list[Category | None]
    correct_split: bool  # the same number of parts with the same domains in order
    right_count: bool
    raw_right_count: bool = False  # the orchestrator's own count, before the guard
    same_domains: bool  # the same domains, in any order


class PlanReport(BaseModel):
    n_cases: int
    n_answered: int
    intent_accuracy: float
    domain_accuracy: float  # single-topic questions, by the first part's domain (or the message's)
    domain_accuracy_by_kind: dict[str, float]
    wrong_domain: list[str]
    wrong_intent: list[str]
    single_topic: int
    split_single: list[Outcome]  # single-topic questions that still have 2+ parts after the guard (what the app does)
    split_rate: float
    split_single_raw: list[str] = []  # ids the orchestrator itself split, before the guard (a fragment merged back is in this list only)
    split_rate_raw: float = 0.0
    compound: list[CompoundResult]
    compound_correct: int  # right count and right domains in order
    compound_right_count: int = 0  # right number of parts (the step b rule); domains are reported apart by `compound_correct`
    compound_right_count_raw: int = 0
    errors: list[str]  # question ids whose call failed or returned an invalid reply
    stopped: str | None = None  # why the run ended early, if it did


# --- the cases ----------------------------------------------------------------------------------------------------


def load_cases() -> list[Case]:
    cases = [Case(id=r.id, kind="kb", question=r.question, expected_parts=[r.expected_domain]) for r in kb_rows()]
    cases += [Case(id=r.id, kind="gap", question=r.question, expected_parts=[r.expected_category]) for r in gap_rows()]
    for row in (CompoundRow(**r) for r in load_jsonl(EVALS_DIR / "compound_questions.jsonl")):
        cases.append(Case(id=row.id, kind="compound", question=row.question,
                          expected_parts=[p.expected_domain for p in row.parts]))
    return cases


# --- running and scoring ------------------------------------------------------------------------------------------


def run_cases(cases: Sequence[Case], planner: Planner, *, after: Callable[[Outcome], None] | None = None,
              stop: Callable[[], str | None] = lambda: None, max_parts: int = 4) -> tuple[list[Outcome], str | None]:
    """Ask the orchestrator each question. `stop` is checked before each one and may name a reason to end the run."""
    outcomes: list[Outcome] = []
    for case in cases:
        reason = stop()
        if reason:
            return outcomes, reason
        try:
            result = planner(case.question)
            raw = [PartOut(text=p.text, domain=p.domain) for p in result.parts]
            kept = (plan_parts(case.question, None, result, max_parts=max_parts).parts
                    if result.intent == "kb_query" and result.parts else [])
            outcome = Outcome(id=case.id, kind=case.kind, question=case.question, intent=result.intent, domain=result.domain,
                              parts=[PartOut(text=p.text, domain=p.domain) for p in kept] if result.parts else raw, raw_parts=raw)
        except ClassifierError:
            outcome = Outcome(id=case.id, kind=case.kind, question=case.question, intent=None, domain=None, parts=[])
        outcomes.append(outcome)
        if after is not None:
            after(outcome)
    return outcomes, None


def _rate(hits: int, total: int) -> float:
    return round(hits / total, 4) if total else 0.0


def score(cases: Sequence[Case], outcomes: Sequence[Outcome], stopped: str | None = None) -> PlanReport:
    by_id = {c.id: c for c in cases}
    answered = [o for o in outcomes if o.intent is not None]
    wrong_intent = [o.id for o in answered if o.intent != "kb_query"]  # every golden question is a workplace question
    singles = [o for o in answered if by_id[o.id].kind != "compound"]
    domain_ok = {o.id: o.predicted_domains[0] == by_id[o.id].expected_parts[0] for o in singles}
    by_kind = {kind: _rate(sum(domain_ok[o.id] for o in singles if o.kind == kind), sum(o.kind == kind for o in singles))
               for kind in ("kb", "gap")}
    split = [o for o in singles if len(o.parts) >= 2]
    compound = []
    for o in (o for o in outcomes if o.kind == "compound"):
        expected, got = by_id[o.id].expected_parts, o.predicted_domains
        compound.append(CompoundResult(
            id=o.id, expected=expected, got=got, correct_split=got == expected,
            right_count=len(got) == len(expected), same_domains=sorted(map(str, got)) == sorted(expected),
            raw_right_count=len(o.raw_parts or [None]) == len(expected)))
    return PlanReport(
        n_cases=len(cases), n_answered=len(answered),
        intent_accuracy=_rate(len(answered) - len(wrong_intent), len(answered)),
        domain_accuracy=_rate(sum(domain_ok.values()), len(singles)), domain_accuracy_by_kind=by_kind,
        wrong_domain=[i for i, ok in domain_ok.items() if not ok], wrong_intent=wrong_intent,
        single_topic=len(singles), split_single=split, split_rate=_rate(len(split), len(singles)),
        split_single_raw=[o.id for o in singles if len(o.raw_parts) >= 2], split_rate_raw=_rate(sum(len(o.raw_parts) >= 2 for o in singles), len(singles)),
        compound=compound, compound_correct=sum(c.correct_split for c in compound),
        compound_right_count=sum(c.right_count for c in compound), compound_right_count_raw=sum(c.raw_right_count for c in compound),
        errors=[o.id for o in outcomes if o.intent is None], stopped=stopped)


def render(report: PlanReport) -> str:
    lines = [
        f"Orchestrator planning check: {report.n_answered} of {report.n_cases} questions answered"
        + (f" (stopped: {report.stopped})" if report.stopped else ""),
        f"  intent accuracy (all are kb_query): {report.intent_accuracy:.1%}"
        + (f"; not kb_query: {', '.join(report.wrong_intent)}" if report.wrong_intent else ""),
        f"  domain accuracy on {report.single_topic} single-topic questions: {report.domain_accuracy:.1%} "
        f"(KB {report.domain_accuracy_by_kind['kb']:.1%}, gap {report.domain_accuracy_by_kind['gap']:.1%})"
        + (f"; wrong: {', '.join(report.wrong_domain)}" if report.wrong_domain else ""),
        f"  split rate on single-topic questions: {len(report.split_single)} of {report.single_topic} "
        f"({report.split_rate:.1%})",
    ]
    for o in report.split_single:
        lines.append(f"    split {o.id}: {o.question!r} -> " + " | ".join(f"{p.domain}: {p.text}" for p in o.parts))
    if report.split_single_raw != [o.id for o in report.split_single]:
        lines.append(f"  raw (before the guard that merges fragments back): {len(report.split_single_raw)} of {report.single_topic} split "
                     f"({', '.join(report.split_single_raw) or 'none'})")
    lines.append(f"  compound questions with the right number of parts: {report.compound_right_count} of {len(report.compound)} "
                 f"(raw {report.compound_right_count_raw}); right number and domains in order: {report.compound_correct} of {len(report.compound)}")
    for c in report.compound:
        lines.append(f"    {c.id}: expected {c.expected}, got {c.got}" + ("" if c.correct_split else "  <-- not as expected"))
    if report.errors:
        lines.append(f"  failed calls: {', '.join(report.errors)}")
    return "\n".join(lines)


# --- the estimate -------------------------------------------------------------------------------------------------


class Estimate(BaseModel):
    calls: int
    input_tokens: int
    output_tokens: int
    usd: float
    ceiling_usd: float


def estimate(settings: Settings, cases: Sequence[Case], ceiling_usd: float) -> Estimate:
    """Priced from config: the orchestrator prompt and each question counted with tiktoken, the structured-output schema
    as assumed above, and the live average output per call (`tier_comparison.output_tokens_per_call`; doubled for a compound
    question, whose reply lists its parts)."""
    main = settings.models.main
    try:
        encoding = tiktoken.encoding_for_model(main.id)
    except KeyError:
        encoding = tiktoken.get_encoding(FALLBACK_ENCODING)
    prompt = len(encoding.encode(load_prompt("orchestrator", settings).text))
    inputs = sum(prompt + len(encoding.encode(c.question)) + STRUCTURED_SCHEMA_TOKENS + 12 for c in cases)  # +12: the labels
    per_call = settings.tier_comparison.output_tokens_per_call
    outputs = sum(per_call * (2 if c.kind == "compound" else 1) for c in cases)
    usd = (inputs * main.price.input + outputs * main.price.output) / 1_000_000
    return Estimate(calls=len(cases), input_tokens=inputs, output_tokens=outputs, usd=round(usd, 4), ceiling_usd=ceiling_usd)


def describe(e: Estimate) -> list[str]:
    return [f"Estimate for the live orchestrator planning check: {e.calls} Main-tier calls, about {e.input_tokens:,} input "
            f"and {e.output_tokens:,} output tokens, about ${e.usd:.3f}.",
            f"The run stops once the ledger shows more than ${e.ceiling_usd:.2f} spent (ceiling); it is recorded in "
            "data/spend_ledger.jsonl under source `eval`."]


# --- planners -----------------------------------------------------------------------------------------------------


def mock_planner(settings: Settings) -> Planner:
    from smartdesk_ai.llm.mock import MockRoutingModel

    classify = make_classifier(settings, builder=lambda model_id, cfg: MockRoutingModel())
    return lambda text: classify(text, "none", None)


def live_planner(settings: Settings) -> Planner:
    from smartdesk_ai.llm.factory import openai_builder

    classify = make_classifier(settings, builder=openai_builder)  # through FallbackChat: retry, fallback, ledger
    return lambda text: classify(text, "none", None)


def save(report: PlanReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RESULT_FILE
    path.write_text(json.dumps(report.model_dump(), indent=1) + "\n")
    return path


def run_live(settings: Settings, cases: Sequence[Case], est: Estimate, out_dir: Path) -> int:
    from smartdesk_ai.ops.spend_ledger import current_run, end_run, start_run

    live = live_settings(settings)
    start_run(live, ["python", "-m", "evals.orchestrator_plan_check", *sys.argv[1:]], "eval")

    def stalled() -> None:
        print("error: the check made no progress and was aborted", file=sys.stderr, flush=True)
        end_run("stalled")
        os._exit(EXIT_STALLED)  # a thread blocked in a network call cannot be interrupted any other way

    watchdog = Watchdog(STALL_SECONDS, stalled)

    def over_ceiling() -> str | None:
        run = current_run()
        return f"ledger spend ${run.usd:.4f} is over the ${est.ceiling_usd:.2f} ceiling" if run and run.usd > est.ceiling_usd else None

    status = "crashed"
    try:
        watchdog.start()
        outcomes, stopped = run_cases(cases, live_planner(live), after=lambda o: watchdog.beat(), stop=over_ceiling,
                                       max_parts=live.orchestrator.max_parts)
        report = score(cases[:len(outcomes)] if stopped else cases, outcomes, stopped)
        print(render(report))
        run = current_run()
        print(f"ledger cost of this run: ${run.usd:.4f} over {run.calls} calls "
              f"({run.input_tokens:,} input, {run.output_tokens:,} output tokens)" if run else "")
        print(f"saved: {save(report, out_dir)}")
        status = "ceiling" if stopped else "ok"
        return 1 if stopped else 0
    finally:
        watchdog.stop()
        end_run(status)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.orchestrator_plan_check", description=__doc__.split("\n\n")[0])
    parser.add_argument("--mock", action="store_true", help="free run with the mock orchestrator")
    parser.add_argument("--estimate", action="store_true", help="print the cost estimate and stop")
    parser.add_argument("--confirm-spend", action="store_true", help="allow the live run, which calls a paid model")
    parser.add_argument("--ceiling", type=float, default=DEFAULT_CEILING_USD, help="stop once the ledger spend passes this (USD)")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="where to save orchestrator_plan_check.json")
    args = parser.parse_args(argv)
    settings = load_settings()
    cases = load_cases()
    if args.mock:
        outcomes, stopped = run_cases(cases, mock_planner(settings), max_parts=settings.orchestrator.max_parts)
        report = score(cases, outcomes, stopped)
        print(render(report))
        print("(MOCK run: the mock orchestrator splits by keyword; this only exercises the code.)")
        return 0
    est = estimate(settings, cases, args.ceiling)
    print("\n".join(describe(est)))
    if args.estimate:
        return 0
    if not args.confirm_spend:
        print("Nothing was run. Add --mock for a free dry run, or --confirm-spend once you have approved the cost.")
        return 2
    return run_live(settings, cases, est, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
