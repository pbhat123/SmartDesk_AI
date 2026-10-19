"""Compound-set eval, mock (T7.10; SPEC section 3I acceptance criteria, FR-I3 to FR-I7, FR-A19, FR-B6).

Each of the 14 questions in `evals/compound_questions.jsonl` goes through the whole multi-agent app with mock models and mock
tickets. The orchestrator is scripted from the row (its parts, and the wrong domain where the row says the wording invites
a misroute), and the worker model answers a part from the expected entry (or an entry of the expected domain), so this checks
the orchestration, not retrieval quality and not any real model. The retrieval threshold is switched off for the same reason.

For every row it records what happened (`RowRun`) and checks it (`check_row`), against these criteria:

  single      a question with one part in one domain calls exactly one worker and no synthesizer call (each answerable
              part of each row, asked on its own)
  compound    two or more answered parts: the workers run in parallel, the synthesizer once, and the answer cites entries
              of every answered domain
  gap_part    a part the KB cannot answer is named, the answered parts are still answered, a ticket is offered and its draft
              covers only the unanswered part(s)
  misroute    a part sent to the wrong worker is answered after one whole-KB retry; a part sent right is not retried
  misc_gap    a MISC gap reaches the ticket agent with category MISC
  isolation   a worker's model input holds only its sub-question and its chunks, nothing of the conversation

    python -m evals.compound_eval [--out-dir evals/results]

Mock only: a live run is deferred to the final runs.
"""
import argparse
import json
import re
import sys
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel

from evals.common import EVALS_DIR, load_jsonl
from evals.harness import RESULTS_DIR
from evals.schemas import CompoundRow
from smartdesk_ai.agent.factory import build_stack
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.config.settings import Settings, load_settings
from smartdesk_ai.llm.mock import MockSynthesizerModel
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.tickets import MockTicketClient

RESULT_FILE = "compound_eval.json"
EMAIL = "eval.user@smartdesk.example"
CRITERIA = ("single", "compound", "gap_part", "misroute", "misc_gap", "isolation")
BARRIER_SECONDS = 5  # the workers of a fan-out must be in flight together within this long


# --- what a run records -------------------------------------------------------------------------------------------


class RowRun(BaseModel):
    n_parts: int
    first_pass_calls: int  # worker model calls made by the first pass (one per part)
    workers_overlapped: bool  # the first-pass workers were in flight at the same time
    synth_calls: int
    cited_domains: list[str]  # the domain of the first citation of each answered part
    retried_parts: list[str]  # sub-questions of the parts that were retried against the whole KB
    unanswered: list[str]  # sub-questions of the parts left unanswered
    offered: bool
    reply: str
    draft_description: str
    draft_category: str | None
    worker_inputs: list[str]  # the human message of every worker model call
    leaks: list[str]  # what a worker was given beyond its sub-question and chunks


class SingleRun(BaseModel):
    sub_question: str
    first_pass_calls: int
    synth_calls: int
    answered: bool


class RowResult(BaseModel):
    id: str
    run: RowRun
    singles: list[SingleRun]
    problems: list[str]  # "<criterion>: <what is wrong>"


class Criterion(BaseModel):
    checked: int = 0
    passed: int = 0


class EvalReport(BaseModel):
    rows: list[RowResult]
    passed: int  # rows with no problem
    criteria: dict[str, Criterion]


# --- the scripted models ------------------------------------------------------------------------------------------


class ScriptedOrchestrator:
    """The orchestrator for the compound set: the row's parts, tagged with the wrong domain where the row invites a
    misroute. A message that is a part's own sub-question (a control) gets that one part in its right domain."""

    def __init__(self, rows: Sequence[CompoundRow]) -> None:
        self.rows = {r.question: r for r in rows}
        self.parts = {p.sub_question: p for r in rows for p in r.parts}

    def invoke(self, messages):
        text = re.search(r"^Message: (.*)$", str(messages[-1].content), re.M | re.S).group(1).strip()
        if text in self.rows:
            parts = [{"text": p.sub_question, "domain": p.likely_wrong_domain or p.expected_domain, "follow_up": False}
                     for p in self.rows[text].parts]
        else:
            part = self.parts[text]
            parts = [{"text": part.sub_question, "domain": part.expected_domain, "follow_up": False}]
        reply = {"intent": "kb_query", "confidence": 0.95, "domain": parts[0]["domain"], "parts": parts}
        return SimpleNamespace(content=json.dumps(reply))


class RowWorker:
    """The worker model: answers a part of the set from the entry the row expects (or the first entry of the expected
    domain), says it cannot for a gap part, and records every input. The first `parallel` calls of a turn wait for each
    other, which only works if they are in flight together."""

    def __init__(self, rows: Sequence[CompoundRow]) -> None:
        self.parts = {p.sub_question: p for r in rows for p in r.parts}
        self.inputs: list[str] = []
        self.overlapped = True
        self._barrier: threading.Barrier | None = None
        self._first = 0
        self._lock = threading.Lock()

    def reset(self, parallel: int) -> None:
        self.inputs, self.overlapped, self._first = [], True, 0
        self._barrier = threading.Barrier(parallel) if parallel >= 2 else None

    def invoke(self, messages):
        human = str(messages[-1].content)
        question = human.split("\n\nContext:")[0].removeprefix("Question: ")
        with self._lock:
            self.inputs.append(human)
            first = self._first < (self._barrier.parties if self._barrier else 0)
            self._first += 1
        if first and self._barrier is not None:
            try:
                self._barrier.wait(timeout=BARRIER_SECONDS)
            except threading.BrokenBarrierError:
                self.overlapped = False
        part = self.parts.get(question)
        entries = re.findall(r"^\[([A-Z]+-[A-Z0-9-]+)\] [^\n]*\n(.*?)(?=\n\n\[|\Z)", human.split("Context:\n", 1)[-1], re.M | re.S)
        chosen = None
        if part is not None and part.answerable:
            chosen = next((e for e in entries if e[0] == part.expected_entry_id), None) or next(
                (e for e in entries if e[0].startswith(f"{part.expected_domain}-")), None)
        if chosen is None:
            body = {"answer": "", "insufficient": True, "citations": []}
        else:
            sentence = re.split(r"(?<=[.!?])\s", " ".join(chosen[1].split()), maxsplit=1)[0]
            body = {"answer": sentence, "insufficient": False, "citations": [chosen[0]]}
        return SimpleNamespace(content=json.dumps(body))


class CountingSynth:
    def __init__(self) -> None:
        self.calls = 0
        self._inner = MockSynthesizerModel()

    def invoke(self, messages):
        self.calls += 1
        return self._inner.invoke(messages)


# --- running ------------------------------------------------------------------------------------------------------


def load_rows() -> list[CompoundRow]:
    return [CompoundRow(**r) for r in load_jsonl(EVALS_DIR / "compound_questions.jsonl")]


def _leaks(row: CompoundRow, inputs: Sequence[str]) -> list[str]:
    found = []
    for human in inputs:
        question = human.split("\n\nContext:")[0]
        if not question.startswith("Question: ") or "\n\nContext:" not in human:
            found.append("a worker input that is not 'Question: ... Context: ...'")
        if row.question in human:
            found.append("the original message")
        for part in row.parts:
            if part.sub_question != question.removeprefix("Question: ") and part.sub_question in human:
                found.append(f"another part ({part.sub_question})")
    return sorted(set(found))


def observe_row(row: CompoundRow, stack, worker: RowWorker, synth: CountingSynth) -> tuple[RowRun, list[SingleRun]]:
    n = len(row.parts)
    worker.reset(parallel=n)
    synth.calls = 0
    tid = new_thread_id()
    state = run_turn(stack.graph, tid, row.question)
    results = [state.part_results[p.part_id] for p in state.parts]
    retried = [r.sub_question for r in results if r.retried]
    run = {
        "n_parts": len(state.parts), "first_pass_calls": len(worker.inputs) - len(retried),
        "workers_overlapped": worker.overlapped, "synth_calls": synth.calls,
        "cited_domains": [r.citations[0].split("-")[0] for r in results if r.citations],
        "retried_parts": retried,
        "unanswered": list(state.pending_offer.unanswered_parts) if state.pending_offer else [],
        "offered": state.pending_offer is not None and "ticket" in state.reply.lower(),
        "reply": state.reply, "worker_inputs": list(worker.inputs), "leaks": _leaks(row, worker.inputs),
        "draft_description": "", "draft_category": None,
    }
    if state.pending_offer is not None:  # the employee agrees by giving the email: the draft is shown at the gate
        drafted = run_turn(stack.graph, tid, EMAIL)
        if drafted.ticket is not None:
            run["draft_description"], run["draft_category"] = drafted.ticket.description, drafted.ticket.category
    singles = []
    for part in (p for p in row.parts if p.answerable):  # each answerable part asked on its own
        worker.reset(parallel=1)
        synth.calls = 0
        alone = run_turn(stack.graph, new_thread_id(), part.sub_question)
        singles.append(SingleRun(sub_question=part.sub_question, first_pass_calls=len(worker.inputs) - int(
            any(r.retried for r in alone.part_results.values())), synth_calls=synth.calls,
            answered=bool(alone.part_result and alone.part_result.citations)))
    return RowRun(**run), singles


# --- checking -----------------------------------------------------------------------------------------------------


def check_row(row: CompoundRow, run: RowRun) -> list[str]:
    """What is wrong with one row's run, as "<criterion>: <problem>". Pure: it reads only `row` and `run`."""
    answered = [p for p in row.parts if p.answerable]
    gaps = [p for p in row.parts if not p.answerable]
    out: list[str] = []
    home = "compound" if len(answered) >= 2 else "gap_part"
    if run.n_parts != len(row.parts):
        out.append(f"{home}: the orchestrator made {run.n_parts} parts, expected {len(row.parts)}")
    if len(answered) >= 2:
        if not run.workers_overlapped or run.first_pass_calls != len(row.parts):
            out.append("compound: the workers did not run in parallel")
        if run.synth_calls != 1:
            out.append(f"compound: the synthesizer ran {run.synth_calls} times, expected once")
        if set(run.cited_domains) != {p.expected_domain for p in answered}:
            out.append(f"compound: the answer cites {sorted(set(run.cited_domains))}, expected every answered domain "
                       f"{sorted({p.expected_domain for p in answered})}")
    if gaps:
        if len(answered) < 2 and run.synth_calls != 0:
            out.append(f"gap_part: the synthesizer ran {run.synth_calls} times with fewer than two answered parts")
        if run.unanswered != [p.sub_question for p in gaps]:
            out.append(f"gap_part: the unanswered part(s) are {run.unanswered}, expected {[p.sub_question for p in gaps]}")
        if not all(p.sub_question in run.reply for p in gaps):
            out.append("gap_part: the reply does not name every unanswered part")
        if answered and "Sources:" not in run.reply:
            out.append("gap_part: the answered part(s) were not answered with sources")
        if not run.offered:
            out.append("gap_part: no ticket offer for the unanswered part(s)")
        if not all(p.sub_question in run.draft_description for p in gaps) or any(
                p.sub_question in run.draft_description for p in answered):
            out.append("gap_part: the ticket must cover only the unanswered part(s) (it is missing one or holds an answered part)")
    if any(p.expected_domain == "MISC" for p in gaps) and run.draft_category != "MISC":
        out.append(f"misc_gap: the ticket category is {run.draft_category}, expected MISC")
    for part in row.parts:
        if part.likely_wrong_domain and part.sub_question not in run.retried_parts:
            out.append(f"misroute: no whole-KB retry for the misrouted part {part.sub_question!r}")
        if part.likely_wrong_domain and part.sub_question in run.unanswered:
            out.append(f"misroute: the misrouted part {part.sub_question!r} was not recovered")
        if part.answerable and not part.likely_wrong_domain and part.sub_question in run.retried_parts:
            out.append(f"misroute: the correctly routed part {part.sub_question!r} was retried")
    if run.leaks or not run.worker_inputs:
        out.append(f"isolation: a worker was given more than its sub-question and chunks: {run.leaks or 'no worker input'}")
    return out


def check_single(single: SingleRun) -> list[str]:
    out = []
    if single.first_pass_calls != 1:
        out.append(f"single: {single.first_pass_calls} worker calls for one part, expected one")
    if single.synth_calls != 0:
        out.append("single: the synthesizer ran for a question with one part")
    if not single.answered:
        out.append("single: the part was not answered")
    return out


def _problem_criterion(problem: str) -> str:
    return problem.split(":", 1)[0]


def applicable(row: CompoundRow) -> set[str]:
    gaps = [p for p in row.parts if not p.answerable]
    found = {"isolation"}
    if sum(p.answerable for p in row.parts) >= 2:
        found.add("compound")
    if gaps:
        found.add("gap_part")
    if any(p.likely_wrong_domain for p in row.parts):
        found.add("misroute")
    if any(p.expected_domain == "MISC" for p in gaps):
        found.add("misc_gap")
    return found


def summarize(rows: Sequence[CompoundRow], results: Sequence[RowResult]) -> EvalReport:
    criteria = {name: Criterion() for name in CRITERIA}
    for row, result in zip(rows, results):
        failed = {_problem_criterion(p) for p in result.problems}
        for name in applicable(row):
            criteria[name].checked += 1
            criteria[name].passed += name not in failed
        for single in result.singles:
            criteria["single"].checked += 1
            criteria["single"].passed += not check_single(single)
    return EvalReport(rows=list(results), passed=sum(not r.problems for r in results), criteria=criteria)


def run_eval(settings: Settings, work: Path) -> EvalReport:
    rows = load_rows()
    worker, synth = RowWorker(rows), CountingSynth()
    # The retrieval threshold is switched off: this checks the orchestration, and hashing embeddings are not a retriever.
    lenient = settings.model_copy(update={"thresholds": settings.thresholds.model_copy(update={"escalation_mock": -1.0})})
    stack = build_stack(lenient, environ={}, dotenv_path=None, data_dir=work, use_cache=False, registry=HealthRegistry(),
                        client=MockTicketClient(), tracing=False, mock_answer_model=worker,
                        mock_routing_model=ScriptedOrchestrator(rows), mock_synth_model=synth)
    results = []
    for row in rows:
        run, singles = observe_row(row, stack, worker, synth)
        results.append(RowResult(id=row.id, run=run, singles=singles,
                                 problems=[*check_row(row, run), *(p for s in singles for p in check_single(s))]))
    return summarize(rows, results)


def render(report: EvalReport) -> str:
    lines = [f"Compound-set eval (MOCK: scripted orchestrator and workers; this checks the orchestration, not any model): "
             f"{report.passed} of {len(report.rows)} rows pass every check"]
    for name in CRITERIA:
        c = report.criteria[name]
        lines.append(f"  {name:10} {c.passed} of {c.checked}")
    for r in report.rows:
        for problem in r.problems:
            lines.append(f"  {r.id}: {problem}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.compound_eval", description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="where to save compound_eval.json")
    parser.add_argument("--live", action="store_true", help="not available yet")
    args = parser.parse_args(argv)
    if args.live:
        print("A live compound run is deferred to the final runs (T7.10 live); this eval is mock only.", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="compound_eval_") as work:
        report = run_eval(load_settings(), Path(work))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / RESULT_FILE).write_text(json.dumps(report.model_dump(), indent=1) + "\n")
    print(render(report))
    return 0 if report.passed == len(report.rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
