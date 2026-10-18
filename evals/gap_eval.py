"""Gap-set pass rate and escalation precision and recall, end to end (SPEC section 7, FR-A5, FR-A15).

Every gap question and every KB question goes through the whole app on its own session. A gap question
passes when it ends in "I don't know" with a ticket offer: never a made-up answer (it cites an entry or
is shown as answered) and never a deflection (the intent is out of scope). A KB question counts as
answered when no ticket is offered. Escalation recall is the share of gap questions escalated (equal to
the gap pass rate); escalation precision is the share of all escalations that were gap questions.
"""
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from evals.harness import Resources, gap_rows, kb_rows
from smartdesk_ai.agent.factory import AgentStack, build_stack
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import AgentState, new_thread_id
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.ops.runtime import RequestContext, request_scope

Outcome = Literal["escalated", "answered", "deflected", "other"]


class Case(BaseModel):
    id: str
    label: Literal["kb", "gap"]
    split: str
    grey_zone: bool = False
    outcome: Outcome
    question: str = Field(exclude=True)  # eval questions are public data, but the result file stays small


def classify(state: AgentState) -> Outcome:
    """What the app did with a question. Escalation wins only if nothing was shown as an answer."""
    if state.intent is not None and state.intent.intent == "out_of_scope":
        return "deflected"
    part = state.part_result
    if part is not None and part.citations and state.pending_offer is None:
        return "answered"
    if state.pending_offer is not None and not (part is not None and part.citations):
        return "escalated"
    return "answered" if part is not None and part.answerable and not part.llm_insufficient else "other"


class GapReport(BaseModel):
    n_gap: int
    n_kb: int
    gap_pass_rate: float
    kb_answer_rate: float
    escalation_precision: float | None
    escalation_recall: float
    by_split: dict[str, dict[str, float]]
    grey_zone_pass_rate: float | None
    failures: list[str] = Field(default_factory=list)  # gap question IDs that did not pass, with what happened
    cases: list[Case] = Field(default_factory=list, exclude=True)


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def summarize(cases: Sequence[Case]) -> GapReport:
    gaps = [c for c in cases if c.label == "gap"]
    kbs = [c for c in cases if c.label == "kb"]
    passed = [c for c in gaps if c.outcome == "escalated"]
    escalations = [c for c in cases if c.outcome == "escalated"]
    grey = [c for c in gaps if c.grey_zone]
    by_split = {}
    for split in sorted({c.split for c in cases}):
        g = [c for c in gaps if c.split == split]
        k = [c for c in kbs if c.split == split]
        by_split[split] = {"gap_pass_rate": _rate(sum(c.outcome == "escalated" for c in g), len(g)),
                           "kb_answer_rate": _rate(sum(c.outcome != "escalated" for c in k), len(k))}
    return GapReport(
        n_gap=len(gaps), n_kb=len(kbs), gap_pass_rate=_rate(len(passed), len(gaps)),
        kb_answer_rate=_rate(sum(c.outcome != "escalated" for c in kbs), len(kbs)),
        escalation_precision=(len(passed) / len(escalations)) if escalations else None,
        escalation_recall=_rate(len(passed), len(gaps)), by_split=by_split,
        grey_zone_pass_rate=_rate(sum(c.outcome == "escalated" for c in grey), len(grey)) if grey else None,
        failures=[f"{c.id}: {c.outcome}" for c in gaps if c.outcome != "escalated"], cases=list(cases),
    )


def run_cases(stack: AgentStack, split: str | None = None) -> list[Case]:
    """Ask every gap and KB question on its own session, under the cost task "eval"."""
    cases = []
    items = [(r, "gap") for r in gap_rows(split)] + [(r, "kb") for r in kb_rows(split)]
    for row, label in items:
        session = new_thread_id()
        ctx = RequestContext(session_id=session, request_id=f"gap-{row.id}", task="eval")
        with request_scope(ctx, stack.services):
            state = run_turn(stack.graph, session, row.question)
        cases.append(Case(id=row.id, label=label, split=row.split, grey_zone=getattr(row, "grey_zone", False),
                          outcome=classify(state), question=row.question))
    return cases


def run(res: Resources, out_dir: Path | None, work_dir: Path, *, settings: Settings | None = None,
        stack: AgentStack | None = None) -> GapReport:
    stack = stack or build_stack(settings or res.settings, environ={}, dotenv_path=None, data_dir=work_dir)
    report = summarize(run_cases(stack))
    print(f"gap eval ({'mock' if res.mock else 'live'}): gap pass rate {report.gap_pass_rate:.3f} "
          f"({report.n_gap} questions), KB answer rate {report.kb_answer_rate:.3f} ({report.n_kb} questions)")
    precision = "n/a" if report.escalation_precision is None else f"{report.escalation_precision:.3f}"
    print(f"  escalation precision {precision}  recall {report.escalation_recall:.3f}")
    for failure in report.failures:
        print(f"  FAIL {failure}")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"mock": res.mock, **report.model_dump(),
                   "cases": [c.model_dump() for c in report.cases]}
        (out_dir / "gap_eval.json").write_text(json.dumps(payload, indent=2))
    return report
