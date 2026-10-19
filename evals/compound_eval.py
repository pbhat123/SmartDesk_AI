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

With `--live` the real models run (mock tickets): the orchestrator splits the message itself, so its wording is the row's
`part_texts`, parallelism and call counts are read from the trace, and the ticket draft is judged by what the drafter was
given. `--estimate` prints the cost; `--confirm-spend` runs it under the live guard (ceiling 2x the estimate, watchdog,
resumable).
"""
import argparse
import json
import re
import sys
import tempfile
import threading
from datetime import datetime, timedelta
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel

from evals.common import EVALS_DIR, load_jsonl
from evals.diagnostics import PartDiag, part_diagnostics
from evals.gap_eval import build_live_stack
from evals.harness import RESULTS_DIR, ROOT
from evals.live_guard import Checkpoint, Estimate, LiveRun, describe, keep_trace, run_items, signature
from evals.schemas import CompoundRow
from smartdesk_ai.agent.factory import build_stack
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.config.settings import Settings, load_settings
from smartdesk_ai.llm.mock import MockSynthesizerModel
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.ops.runtime import RequestContext, request_scope
from smartdesk_ai.tickets import MockTicketClient

RESULT_FILE = "compound_eval.json"
LIVE_RESULT_FILE = "compound_eval_live.json"
CHECKPOINT = ROOT / "data" / "eval_checkpoints" / "compound.json"
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
    part_texts: list[str] = []  # the orchestrator's wording of each part, in order (empty: the row's own wording)
    parts: list[PartDiag] = []  # live runs: what happened to each part (domain, retrieval, worker, judge, retry, citations)


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
    routing: dict[str, int] | None = None  # live runs: where the parts were sent, reported apart from the criteria


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


# --- running live: what happened is read from the trace ---------------------------------------------------------


def _human(prompt: str) -> str:
    return prompt.rsplit("[human] ", 1)[-1]


def _span_window(event: dict) -> tuple[datetime, datetime]:
    end = datetime.fromisoformat(event["ts"])
    return end - timedelta(milliseconds=event.get("latency_ms", 0.0)), end


def _overlap(windows: Sequence[tuple[datetime, datetime]]) -> bool:
    """Every window shares a moment with every other: the workers were in flight together."""
    return len(windows) < 2 or max(w[0] for w in windows) < min(w[1] for w in windows)


def run_from_trace(row: CompoundRow, state, events: Sequence[dict], draft_events: Sequence[dict] = (), drafted=None) -> RowRun:
    """A `RowRun` for a live turn: parts and answers from the final state, the worker fan-out and the synthesizer from the
    request's trace events. The ticket draft is judged by what the drafter was given (its statements), because its own wording
    is a paraphrase."""
    results = [state.part_results[p.part_id] for p in state.parts]
    first_pass: dict[str, dict] = {}
    for event in sorted((e for e in events if e.get("kind") == "agent_span" and e.get("part_id")), key=lambda e: e["ts"]):
        first_pass.setdefault(event["part_id"], event)  # a retry is the part's second span
    worker_inputs = [_human(e["prompt"]) for e in events if e.get("kind") == "llm_call" and e.get("task") == "WORKER_ANSWER"]
    draft_input = next((_human(e["prompt"]) for e in draft_events if e.get("kind") == "llm_call" and e.get("task") == "TICKET_DRAFT"), "")
    texts = [r.sub_question for r in results]
    return RowRun(
        n_parts=len(state.parts), first_pass_calls=len(first_pass),
        workers_overlapped=_overlap([_span_window(e) for e in first_pass.values()]),
        synth_calls=sum(e.get("kind") == "llm_call" and e.get("task") == "SYNTHESIZER" for e in events),
        cited_domains=[r.citations[0].split("-")[0] for r in results if r.citations],
        retried_parts=[r.sub_question for r in results if r.retried],
        unanswered=list(state.pending_offer.unanswered_parts) if state.pending_offer else [],
        offered=state.pending_offer is not None and "ticket" in state.reply.lower(), reply=state.reply,
        draft_description=draft_input, draft_category=drafted.category if drafted is not None else None,
        worker_inputs=worker_inputs, leaks=_leaks(row, worker_inputs), part_texts=texts, parts=part_diagnostics(state, events))


def observe_row_live(row: CompoundRow, stack) -> RowRun:
    """One compound question through the whole app, as a live run sees it (the single-part controls are left to the
    gap and KB run, which asks 74 single-topic questions)."""
    trace = stack.services.trace
    tid = new_thread_id()

    def turn(text: str, suffix: str):
        rid = f"{row.id}-{tid[:8]}{suffix}"  # unique per run, so a stack asked twice never mixes its events
        with request_scope(RequestContext(session_id=tid, request_id=rid, task="eval"), stack.services):
            state = run_turn(stack.graph, tid, text)
        return state, [e for e in trace.events if e.get("request_id") == rid]

    state, events = turn(row.question, "")
    draft_events, drafted = [], None
    if state.pending_offer is not None:
        after, draft_events = turn(EMAIL, "-draft")
        drafted = after.ticket
    return run_from_trace(row, state, events, draft_events, drafted)


def estimate_live(settings: Settings, rows: Sequence[CompoundRow]) -> Estimate:
    """Per row: the orchestrator call, one worker call and one judge call per part, one synthesizer call when two or more parts
    can be answered, and for a row with a gap the ticket draft. Priced like a golden question (`drift.tokens_per_question`)
    for each worker part, plus the orchestrator at its measured size."""
    models, tokens = settings.models, settings.drift.tokens_per_question
    worker = (tokens.main_input * models.main.price.input + tokens.main_output * models.main.price.output) / 1_000_000
    orchestrator = (1500 * models.main.price.input + 200 * models.main.price.output) / 1_000_000
    synth = (1200 * models.main.price.input + 250 * models.main.price.output) / 1_000_000
    draft = (800 * models.light.price.input + 150 * models.light.price.output) / 1_000_000
    parts = sum(len(r.parts) for r in rows)
    synths = sum(sum(p.answerable for p in r.parts) >= 2 for r in rows)
    drafts = sum(any(not p.answerable for p in r.parts) for r in rows)
    usd = len(rows) * orchestrator + parts * worker + synths * synth + drafts * draft
    return Estimate("the live compound eval", round(usd, 4), f"{len(rows)} questions, {parts} parts, {synths} synthesized, {drafts} drafts")


def run_live(settings: Settings, rows: Sequence[CompoundRow], estimate: Estimate, out_dir: Path, checkpoint_path: Path) -> int:
    live = settings.model_copy(update={"modes": settings.modes.model_copy(update={"llm": "live"})})
    sig = signature([[r.id, r.question] for r in rows] + [live.models.main.id, live.models.light.id, dict(live.prompts)])
    checkpoint = Checkpoint(checkpoint_path, sig)
    if checkpoint.resumed:
        print(f"resuming: {checkpoint.resumed} of {len(rows)} questions are already answered")
    by_id = {r.id: r for r in rows}
    work = Path(tempfile.mkdtemp(prefix="compound_live_"))
    with LiveRun(live, ["python", "-m", "evals.compound_eval", *sys.argv[1:]], ceiling_usd=estimate.ceiling_usd) as run:
        stack = build_live_stack(live, work)
        done, stopped = run_items(run, checkpoint, [r.id for r in rows],
                                  lambda i: observe_row_live(by_id[i], stack).model_dump())
        results = []
        for row in rows:
            if row.id in done:
                run_ = RowRun(**done[row.id])
                results.append(RowResult(id=row.id, run=run_, singles=[], problems=check_row(row, run_, live=True)))
        report = summarize([r for r in rows if r.id in done], results)
        report.routing = routing_stats(rows, {r.id: r.run for r in results})
        cost = run.cost()
    kept = keep_trace(work, "compound_live", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / LIVE_RESULT_FILE).write_text(json.dumps({**report.model_dump(), "stopped": stopped, "usd": round(cost.usd, 6)}, indent=1) + "\n")
    print(render(report, live=True))
    if kept:
        print(f"trace kept: {kept}")
    print(f"ledger cost of this run: ${cost.usd:.4f} over {cost.calls} calls ({cost.input_tokens:,} input, {cost.output_tokens:,} output tokens)")
    if stopped:
        print(f"STOPPED after {len(results)} of {len(rows)} rows: {stopped}")
    return 1 if stopped or report.passed < len(rows) else 0


# --- checking -----------------------------------------------------------------------------------------------------


def check_row(row: CompoundRow, run: RowRun, *, live: bool = False) -> list[str]:
    """What is wrong with one row's run, as "<criterion>: <problem>". Pure: it reads only `row` and `run`. In a live run the
    misroute criterion is judged by outcome: a part the row says may be misrouted must be answered from its expected entry,
    whether the orchestrator routed it right or the whole-KB retry recovered it (where it went is reported by `routing_stats`)."""
    texts = run.part_texts if len(run.part_texts) == len(row.parts) else [p.sub_question for p in row.parts]
    text_of = {p.sub_question: t for p, t in zip(row.parts, texts)}
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
        if run.unanswered != [text_of[p.sub_question] for p in gaps]:
            out.append(f"gap_part: the unanswered part(s) are {run.unanswered}, expected {[text_of[p.sub_question] for p in gaps]}")
        if not all(text_of[p.sub_question] in run.reply for p in gaps):
            out.append("gap_part: the reply does not name every unanswered part")
        if answered and "Sources:" not in run.reply:
            out.append("gap_part: the answered part(s) were not answered with sources")
        if not run.offered:
            out.append("gap_part: no ticket offer for the unanswered part(s)")
        if not all(text_of[p.sub_question] in run.draft_description for p in gaps) or any(
                text_of[p.sub_question] in run.draft_description for p in answered):
            out.append("gap_part: the ticket must cover only the unanswered part(s) (it is missing one or holds an answered part)")
    if any(p.expected_domain == "MISC" for p in gaps) and run.draft_category != "MISC":
        out.append(f"misc_gap: the ticket category is {run.draft_category}, expected MISC")
    for i, part in enumerate(row.parts):
        text = text_of[part.sub_question]
        if live and part.likely_wrong_domain:
            diag = run.parts[i] if len(run.parts) == len(row.parts) else None
            if part.expected_entry_id and (diag is None or part.expected_entry_id not in diag.citations):
                out.append(f"misroute: the part {part.sub_question!r} was not answered from its expected entry {part.expected_entry_id}")
        else:
            if part.likely_wrong_domain and text not in run.retried_parts:
                out.append(f"misroute: no whole-KB retry for the misrouted part {part.sub_question!r}")
            if part.likely_wrong_domain and text in run.unanswered:
                out.append(f"misroute: the misrouted part {part.sub_question!r} was not recovered")
        if part.answerable and not part.likely_wrong_domain and text in run.retried_parts:
            out.append(f"misroute: the correctly routed part {part.sub_question!r} was retried")
    if run.leaks or not run.worker_inputs:
        out.append(f"isolation: a worker was given more than its sub-question and chunks: {run.leaks or 'no worker input'}")
    return out


def routing_stats(rows: Sequence[CompoundRow], runs: dict[str, RowRun]) -> dict[str, int]:
    """Where the live orchestrator sent each answerable part, apart from whether the part was answered: right domain, wrong
    domain, and of the wrong ones how many the whole-KB retry recovered (answered from the expected entry). A row whose part
    count changed cannot be matched part by part and is counted as not aligned."""
    stats = dict(parts=0, routed_right=0, routed_wrong=0, recovered_by_retry=0, unrecovered=0, not_aligned=0)
    for row in rows:
        run = runs.get(row.id)
        if run is None:
            continue
        answerable = [(i, p) for i, p in enumerate(row.parts) if p.answerable]
        stats["parts"] += len(answerable)
        if len(run.parts) != len(row.parts):
            stats["not_aligned"] += len(answerable)
            continue
        for i, part in answerable:
            diag = run.parts[i]
            if diag.domain == part.expected_domain:
                stats["routed_right"] += 1
                continue
            stats["routed_wrong"] += 1
            recovered = diag.retried and bool(part.expected_entry_id) and part.expected_entry_id in diag.citations
            stats["recovered_by_retry" if recovered else "unrecovered"] += 1
    return stats


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


def render(report: EvalReport, *, live: bool = False) -> str:
    head = ("Compound-set eval (live: real models; the single-part control is the gap and KB run)" if live else
            "Compound-set eval (MOCK: scripted orchestrator and workers; this checks the orchestration, not any model)")
    lines = [f"{head}: {report.passed} of {len(report.rows)} rows pass every check"]
    for name in CRITERIA:
        c = report.criteria[name]
        lines.append(f"  {name:10} {c.passed} of {c.checked}")
    if report.routing:
        r = report.routing
        lines.append(f"  routing (reported apart): {r['routed_right']} of {r['parts']} answerable parts sent to the right domain, "
                     f"{r['routed_wrong']} to the wrong one ({r['recovered_by_retry']} recovered by the whole-KB retry, "
                     f"{r['unrecovered']} not), {r['not_aligned']} in rows whose part count changed")
    for r in report.rows:
        for problem in r.problems:
            lines.append(f"  {r.id}: {problem}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.compound_eval", description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="where to save compound_eval.json")
    parser.add_argument("--live", action="store_true", help="the real models (costs money; needs --confirm-spend)")
    parser.add_argument("--estimate", action="store_true", help="with --live: print the cost estimate and stop")
    parser.add_argument("--confirm-spend", action="store_true", help="allow the live run")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT, help="where a live run saves finished rows, to resume")
    parser.add_argument("--only", default=None, help="live: only these rows, comma separated (a diagnostic re-run)")
    args = parser.parse_args(argv)
    if args.live or args.estimate:
        settings, rows = load_settings(), load_rows()
        if args.only:
            wanted = set(args.only.split(","))
            rows = [r for r in rows if r.id in wanted]
            if {r.id for r in rows} != wanted:
                print(f"error: unknown row(s): {sorted(wanted - {r.id for r in rows})}", file=sys.stderr)
                return 2
        estimate = estimate_live(settings, rows)
        print("\n".join(describe(estimate)))
        if args.estimate:
            return 0
        if not args.confirm_spend:
            print("Nothing was run. Add --confirm-spend once you have approved the cost; without --live the eval is free (mock).")
            return 2
        return run_live(settings, rows, estimate, args.out_dir, args.checkpoint)
    with tempfile.TemporaryDirectory(prefix="compound_eval_") as work:
        report = run_eval(load_settings(), Path(work))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / RESULT_FILE).write_text(json.dumps(report.model_dump(), indent=1) + "\n")
    print(render(report))
    return 0 if report.passed == len(report.rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
