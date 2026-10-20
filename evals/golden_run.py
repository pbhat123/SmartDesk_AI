"""The final gap + KB run with the drift baseline taken from the same answers (SPEC section 7, FR-A5, FR-A15, FR-H15).

The gap eval and the drift check both ask every golden question (53 KB, 21 gap) through the whole app. Run apart they pay for
the same 74 answers twice, so this asks each question once and reads from it: the gap pass rate, the KB answer rate and
escalation precision and recall (as `gap_eval` defines them), and one drift point (as `ops.drift` defines it). It goes through the
live guard: estimate, `--confirm-spend`, a ceiling of twice the estimate, a watchdog and a per-question checkpoint that resumes.

    python -m evals.golden_run [--estimate | --mock | --confirm-spend] [--out-dir DIR]
"""
import argparse
from collections import Counter
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from evals.diagnostics import PartDiag, part_diagnostics
from evals.gap_eval import Case, GapReport, answered_with_citations, build_live_stack, classify, summarize
from evals.harness import RESULTS_DIR, ROOT
from evals.moderation_record import ModerationDegraded, RecordingModerator, eval_moderator, row_status
from evals.live_guard import Checkpoint, Estimate, LiveRun, describe, keep_trace, run_items, signature
from smartdesk_ai.agent.factory import AgentStack, build_stack
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.config.env import DEFAULT_DOTENV
from smartdesk_ai.config.settings import Settings, load_settings
from smartdesk_ai.ops import drift
from smartdesk_ai.ops.health import Component
from smartdesk_ai.ops.runtime import RequestContext, request_scope

CHECKPOINT = ROOT / "data" / "eval_checkpoints" / "golden.json"
DRIFT_LOG = ROOT / "data" / drift.DRIFT_LOG
RESULT_FILE = "gap_eval.json"


class Observation(BaseModel):
    """What one golden question showed: the gap outcome and what the drift numbers read."""

    id: str
    label: Literal["kb", "gap"]
    split: str
    grey_zone: bool = False
    question: str
    moderation: str = "not_recorded"  # "ok" (the Moderation API checked the question), "mock" (keyword stand-in)
    outcome: str
    cited: bool = False
    top_score: float | None
    hit: bool | None  # KB questions: the expected entry was among the k retrieved
    grounded: int  # the grounding judge's verdicts for this question
    verdicts: int
    workers: int = 0  # worker spans of the question (one part should mean one worker)
    synth_calls: int = 0  # synthesizer calls (a single-part question should make none)
    parts: list[PartDiag] = []  # what happened to each part: domain, retrieval, worker and judge verdicts, retry, citations


class GoldenResult(BaseModel):
    gap: GapReport
    single_part_clean: int = 0  # questions that used exactly one worker and no synthesizer call
    single_part_total: int = 0
    not_clean: list[str] = []
    drift: drift.DriftPoint | None  # None when the run stopped before the last question
    stopped: str | None = None
    usd: float = 0.0
    observations: list[Observation] = []


def observe(stack: AgentStack, row: Any, label: str, k: int, moderator: RecordingModerator | None = None) -> Observation:
    """Ask one golden question on its own session, under the cost task "eval". With a `moderator`, the question is checked
    first, as the app's input guard would, and the status is recorded; a failed check raises before the turn is paid for."""
    moderation = "not_recorded"
    if moderator is not None:
        before = moderator.counts()
        try:
            moderator.check(row.question)
        except Exception:  # the recorder has counted it; row_status raises the plain error
            pass
        moderation, _ = row_status(row.id, before, moderator, blocked_reason=None)
    trace = stack.services.trace
    before = len(trace.events) if trace is not None else 0
    session = new_thread_id()
    ctx = RequestContext(session_id=session, request_id=f"golden-{row.id}", task="eval")
    with request_scope(ctx, stack.services):
        state = run_turn(stack.graph, session, row.question)
    part = state.part_result
    chunks = part.chunks if part is not None else []
    new_events = trace.events[before:] if trace is not None else []
    verdicts = drift.verdicts_from_events(new_events)
    return Observation(
        id=row.id, label=label, split=row.split, grey_zone=getattr(row, "grey_zone", False), question=row.question,
        moderation=moderation, outcome=classify(state), cited=label == "kb" and answered_with_citations(state), top_score=chunks[0].score if chunks else None,
        hit=row.expected_entry_id in {c.entry_id for c in chunks[:k]} if label == "kb" else None,
        grounded=sum(verdicts), verdicts=len(verdicts),
        workers=len({e.get("part_id") for e in new_events if e.get("kind") == "agent_span" and e.get("part_id")}),
        synth_calls=sum(e.get("kind") == "llm_call" and e.get("task") == "SYNTHESIZER" for e in new_events),
        parts=part_diagnostics(state, new_events))


def run_golden(stack: AgentStack, settings: Settings, golden: drift.Golden, *, mock: bool, run: LiveRun | None = None,
               checkpoint: Checkpoint | None = None, partial: bool = False, moderator: RecordingModerator | None = None) -> GoldenResult:
    k = settings.retrieval.top_n
    items = [(r, "kb") for r in golden.kb] + [(r, "gap") for r in golden.gap]
    by_id = {r.id: (r, label) for r, label in items}
    ids = [r.id for r, _ in items]
    if run is None:
        done, stopped = {i: observe(stack, *by_id[i], k, moderator).model_dump() for i in ids}, None
    else:
        done, stopped = run_items(run, checkpoint if checkpoint is not None else Checkpoint(None, ""), ids,
                                  lambda i: observe(stack, *by_id[i], k, moderator).model_dump())
    seen = [Observation(**done[i]) for i in ids if i in done]
    gap = summarize([Case(id=o.id, label=o.label, split=o.split, grey_zone=o.grey_zone, outcome=o.outcome, cited=o.cited, question=o.question)
                     for o in seen])
    usd = run.cost().usd if run is not None else stack.services.cost.total
    point = None
    if stopped is None and len(seen) == len(ids) and not partial:  # a subset is a diagnostic, never a drift point
        judged = sum(o.verdicts for o in seen)
        point = drift.make_point(
            settings, mock=mock, faithfulness=sum(o.grounded for o in seen) / judged if judged else None,
            hits=[o.hit for o in seen if o.hit is not None], top_scores=[o.top_score for o in seen if o.top_score is not None],
            escalated=sum(o.outcome == "escalated" for o in seen), n_questions=len(seen), n_kb=len(golden.kb),
            n_gap=len(golden.gap), reranker_status=stack.registry.get(Component.RERANKER).status.value, usd=usd)
    clean = [o for o in seen if o.workers == 1 and o.synth_calls == 0]
    return GoldenResult(gap=gap, drift=point, stopped=stopped, usd=usd, observations=seen, single_part_clean=len(clean), single_part_total=len(seen),
                        not_clean=[o.id for o in seen if o not in clean])


def estimate_golden(settings: Settings, golden: drift.Golden) -> Estimate:
    """The golden questions at the measured per-question tokens of the drift check (`drift.tokens_per_question`)."""
    live = settings.model_copy(update={"modes": settings.modes.model_copy(update={"llm": "live"})})
    e = drift.estimate_drift(live, golden.size)
    return Estimate("the gap and KB run (with the drift baseline)", round(e.usd, 4),
                    f"{golden.size} questions: {len(golden.kb)} KB, {len(golden.gap)} gap")


def render(result: GoldenResult, *, mock: bool) -> str:
    g = result.gap
    precision = "n/a" if g.escalation_precision is None else f"{g.escalation_precision:.3f}"
    lines = [f"gap and KB run ({'MOCK' if mock else 'live'}): gap pass rate {g.gap_pass_rate:.3f} ({g.n_gap} questions), "
             f"KB answer rate {g.kb_answer_rate:.3f} ({g.n_kb} questions), "
             f"KB answered with citations {g.kb_answered_with_citations:.3f}" + (f" (not: {', '.join(g.kb_not_cited)})" if g.kb_not_cited else ""),
             f"  escalation precision {precision}  recall {g.escalation_recall:.3f}"]
    lines += [f"  FAIL {f}" for f in g.failures]
    lines.append(f"  single-topic check: {result.single_part_clean} of {result.single_part_total} questions used exactly one worker "
                 f"and no synthesizer call" + (f"; not: {', '.join(result.not_clean)}" if result.not_clean else ""))
    if result.drift is not None:
        d = result.drift
        shown = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(f"  drift point: faithfulness {shown(d.faithfulness)}, recall@{d.k} {shown(d.recall_at_k)}, "
                     f"escalation rate {shown(d.escalation_rate)}, avg rerank score {shown(d.avg_rerank_score)}")
    recorded = Counter(o.moderation for o in result.observations)
    if recorded.get("ok") or recorded.get("mock"):
        lines.append(f"  moderation: checked {recorded['ok'] + recorded['mock']} of {len(result.observations)} questions "
                     f"({'API' if recorded.get('ok') else 'keyword stand-in'}), none degraded")
    if result.stopped:
        lines.append(f"  STOPPED: {result.stopped}")
    return "\n".join(lines)


def save(result: GoldenResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"mock": result.drift.mode == "mock" if result.drift else False, **result.gap.model_dump(),
               "cases": [c.model_dump() for c in result.gap.cases],
               "drift": result.drift.model_dump() if result.drift else None, "stopped": result.stopped,
               "moderation": {o.id: o.moderation for o in result.observations},
               "parts": {o.id: [p.model_dump() for p in o.parts] for o in result.observations if not o.cited or len(o.parts) != 1},
               "single_part_clean": result.single_part_clean, "single_part_total": result.single_part_total, "not_clean": result.not_clean}
    path = out_dir / RESULT_FILE
    path.write_text(json.dumps(payload, indent=2))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.golden_run", description=__doc__.split("\n\n")[0])
    parser.add_argument("--mock", action="store_true", help="free run with mock models (stores nothing in the drift history)")
    parser.add_argument("--estimate", action="store_true", help="print the cost estimate and stop")
    parser.add_argument("--confirm-spend", action="store_true", help="allow the live run, which calls paid models")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="where to save gap_eval.json")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT, help="where a live run saves finished questions, to resume")
    parser.add_argument("--only", default=None, help="only these questions, comma separated (a diagnostic re-run; no drift point)")
    parser.add_argument("--drift-log", type=Path, default=DRIFT_LOG, help="the drift history a finished live run appends to")
    args = parser.parse_args(argv)
    settings = load_settings()
    golden = drift.Golden.load()
    if args.only:
        wanted = set(args.only.split(","))
        golden = drift.Golden(kb=[r for r in golden.kb if r.id in wanted], gap=[r for r in golden.gap if r.id in wanted])
        found = {r.id for r in [*golden.kb, *golden.gap]}
        if found != wanted:
            print(f"error: unknown question(s): {sorted(wanted - found)}", file=sys.stderr)
            return 2
    if args.mock:
        with tempfile.TemporaryDirectory(prefix="golden_run_") as work:
            stack = build_stack(settings, environ={}, dotenv_path=None, data_dir=Path(work), use_cache=False)
            result = run_golden(stack, settings, golden, mock=True, partial=bool(args.only),
                                moderator=eval_moderator(settings, environ={}, dotenv_path=None))
        print(render(result, mock=True))
        print("(MOCK run: it only exercises the code.)")
        return 0
    estimate = estimate_golden(settings, golden)
    print("\n".join(describe(estimate)))
    if args.estimate:
        return 0
    if not args.confirm_spend:
        print("Nothing was run. Add --mock for a free dry run, or --confirm-spend once you have approved the cost.")
        return 2
    live = settings.model_copy(update={"modes": settings.modes.model_copy(update={"llm": "live"})})
    sig = signature([[r.id, r.question] for r in [*golden.kb, *golden.gap]] + [live.models.main.id, live.models.light.id,
                                                                           dict(live.prompts), "moderation-recorded"])
    checkpoint = Checkpoint(args.checkpoint, sig)  # the questions are in the signature, so a subset never reuses the full run's
    if checkpoint.resumed:
        print(f"resuming: {checkpoint.resumed} of {golden.size} questions are already answered")
    with LiveRun(live, ["python", "-m", "evals.golden_run", *(sys.argv[1:] if argv is None else argv)],
                 ceiling_usd=estimate.ceiling_usd) as run:
        work = Path(tempfile.mkdtemp(prefix="golden_run_"))
        stack = build_live_stack(live, work)
        try:
            result = run_golden(stack, live, golden, mock=False, run=run, checkpoint=checkpoint, partial=bool(args.only),
                                moderator=eval_moderator(live, environ=None, dotenv_path=DEFAULT_DOTENV))
        except ModerationDegraded as error:
            print(f"error: {error}", file=sys.stderr)
            return 3
        print(render(result, mock=False))
        cost = run.cost()
        print(f"ledger cost of this run: ${cost.usd:.4f} over {cost.calls} calls ({cost.input_tokens:,} input, "
              f"{cost.output_tokens:,} output tokens)")
    kept = keep_trace(work, "golden_run" if not args.only else "golden_diagnostic", args.out_dir)
    print(f"saved: {save(result, args.out_dir)}" + (f"\ntrace kept: {kept}" if kept else ""))
    if result.drift is not None:
        drift.DriftHistory(args.drift_log).append(result.drift)
        print(f"drift baseline stored in {args.drift_log}")
    return 1 if result.stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
