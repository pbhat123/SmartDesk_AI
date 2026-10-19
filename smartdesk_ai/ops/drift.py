"""Drift monitoring (FR-H15): re-run the golden set (the KB set plus the gap set) on demand and keep the numbers.

Each run stores one `DriftPoint` in `drift.jsonl`: Faithfulness, Recall@k, escalation rate and the average top
rerank score, with a timestamp, the prompt versions and the model IDs. The Monitoring tab charts them over time.
- Faithfulness is the share of the runtime grounding judge's verdicts (FR-H2) that said "grounded", read from the
  run's own trace. It is the cheap, always-on judge, not the Ragas faithfulness of `evals/rag_metrics.py`.
- Recall@k is the share of KB questions whose expected entry is among the k retrieved (k is `retrieval.top_n`).
- Escalation rate is the share of all golden questions that ended in a ticket offer.
- Average rerank score is the mean score of the top retrieved entry. Its scale depends on the retrieval strategy and
  on whether the reranker was loaded, so both are stored with it.

A run calls the judge model, so nobody runs one blindly: the command and the tab button both show the cost estimate
first and ask for confirmation, and a live run stops (and stores nothing) once its spend passes the ceiling.
`python -m smartdesk_ai.ops.drift [--estimate] [--yes]` (`--yes` skips the prompt in mock mode only).
"""
import argparse
import json
import re
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from smartdesk_ai.agent.factory import AgentStack, LiveModeUnavailable, build_stack, resolve_data_dir
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.config.env import DEFAULT_DOTENV
from smartdesk_ai.config.settings import DEFAULT_CONFIG, Settings, load_settings
from smartdesk_ai.ops.health import Component
from smartdesk_ai.ops.runtime import RequestContext, request_scope
from smartdesk_ai.ops.spend_ledger import current_run, end_run, start_run

DRIFT_LOG = "drift.jsonl"
JUDGE_TASK = "GROUNDING_JUDGE"
_GROUNDED = re.compile(r'"grounded"\s*:\s*(true|false)')


class DriftPoint(BaseModel):
    ts: str
    mode: str  # mock or live: the two are never charted together
    faithfulness: float | None  # None when the judge never ran
    recall_at_k: float | None  # None when the golden set has no KB questions
    escalation_rate: float
    avg_rerank_score: float | None
    k: int
    n_kb: int
    n_gap: int
    retrieval_strategy: str
    reranker_status: str  # the score scale changes when the reranker is not loaded
    prompt_versions: dict[str, int]
    models: dict[str, str]
    usd: float


class DriftEstimate(BaseModel):
    n_questions: int
    mode: str
    usd: float
    ceiling_usd: float
    free: bool


class DriftCeilingExceeded(RuntimeError):
    def __init__(self, spent: float, ceiling: float) -> None:
        super().__init__(f"spend ${spent:.2f} passed the ceiling of ${ceiling:.2f}")
        self.spent, self.ceiling = spent, ceiling


@dataclass(frozen=True)
class Golden:
    """The golden set: the KB questions (with the entry each was built from) and the gap questions."""

    kb: Sequence[Any]
    gap: Sequence[Any]

    @property
    def size(self) -> int:
        return len(self.kb) + len(self.gap)

    @classmethod
    def load(cls) -> "Golden":
        from evals.harness import gap_rows, kb_rows  # the data sets live with the evals

        return cls(kb=kb_rows(), gap=gap_rows())


def estimate_drift(settings: Settings, n_questions: int) -> DriftEstimate:
    """What a run costs: questions times the measured per-question tokens at the config prices. Mock is free."""
    mock = settings.modes.llm == "mock"
    tokens, models = settings.drift.tokens_per_question, settings.models
    per_question = (tokens.main_input * models.main.price.input + tokens.main_output * models.main.price.output
                    + tokens.embedding * models.embedding.price.input) / 1_000_000
    usd = 0.0 if mock else n_questions * per_question
    return DriftEstimate(n_questions=n_questions, mode="mock" if mock else "live", usd=usd,
                         ceiling_usd=usd * settings.drift.ceiling_multiplier, free=mock)


def describe_estimate(estimate: DriftEstimate, golden: Golden) -> list[str]:
    """The lines shown before a run, by the command and by the tab."""
    lines = [f"Drift check: re-runs the {estimate.n_questions} golden questions ({len(golden.kb)} KB, "
             f"{len(golden.gap)} gap) through the whole app."]
    if estimate.free:
        lines.append("Mode: mock models. Estimated cost: $0.00 (no paid calls).")
    else:
        lines.append(f"Mode: live. Estimated cost: about ${estimate.usd:.2f}; the run stops and stores nothing if "
                     f"it passes ${estimate.ceiling_usd:.2f}. It calls the judge model.")
    return lines


def faithfulness_from_events(events: Sequence[Mapping[str, Any]]) -> float | None:
    """Grounded verdicts over all verdicts of the grounding judge in a run's trace events."""
    verdicts = []
    for event in events:
        if event.get("kind") != "llm_call" or event.get("task") != JUDGE_TASK:
            continue
        found = _GROUNDED.search(str(event.get("completion", "")))
        if found:
            verdicts.append(found.group(1) == "true")
    return sum(verdicts) / len(verdicts) if verdicts else None


class DriftHistory:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    def append(self, point: DriftPoint) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a") as handle:
                handle.write(point.model_dump_json() + "\n")

    def points(self) -> list[DriftPoint]:
        """Every stored point, oldest first. A missing file is empty; a damaged line is skipped. Nothing is created."""
        if not self._path.is_file():
            return []
        points = []
        for line in self._path.read_text().splitlines():
            try:
                points.append(DriftPoint.model_validate_json(line))
            except ValidationError:
                continue  # a half-written last line
        return sorted(points, key=lambda p: p.ts)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ledger_spend() -> float:
    run = current_run()
    return run.usd if run is not None else 0.0


def run_drift(
    stack: AgentStack,
    settings: Settings,
    *,
    golden: Golden | None = None,
    mock: bool,
    ceiling_usd: float | None = None,
    spent: Callable[[], float] | None = None,
    clock: Callable[[], datetime] = _utcnow,
) -> DriftPoint:
    """Ask every golden question on its own session (cost task "eval") and compute the four numbers.
    `spent` reads the cumulative spend (default: this process's ledger run, else the cost tracker); the run
    raises `DriftCeilingExceeded`, after finishing the question in flight, once it spent more than `ceiling_usd`."""
    from evals.gap_eval import classify  # one definition of "escalated" for the evals and for drift

    golden = golden or Golden.load()
    spent = spent or (lambda: _ledger_spend() if current_run() is not None else stack.services.cost.total)
    base = spent()
    k = settings.retrieval.top_n
    hits: list[bool] = []
    top_scores: list[float] = []
    escalated = 0
    items = [(row, True) for row in golden.kb] + [(row, False) for row in golden.gap]
    for row, is_kb in items:
        session = new_thread_id()
        ctx = RequestContext(session_id=session, request_id=f"drift-{row.id}", task="eval")
        with request_scope(ctx, stack.services):
            state = run_turn(stack.graph, session, row.question)
        part = state.part_result
        chunks = part.chunks if part is not None else []
        if chunks:
            top_scores.append(chunks[0].score)
        if is_kb:
            hits.append(row.expected_entry_id in {c.entry_id for c in chunks[:k]})
        escalated += classify(state) == "escalated"
        if ceiling_usd is not None and spent() - base > ceiling_usd:
            raise DriftCeilingExceeded(spent() - base, ceiling_usd)
    models = settings.models
    events = stack.services.trace.events if stack.services.trace is not None else []
    return DriftPoint(
        ts=clock().isoformat(timespec="seconds"), mode="mock" if mock else "live",
        faithfulness=faithfulness_from_events(events),
        recall_at_k=sum(hits) / len(hits) if hits else None,
        escalation_rate=escalated / len(items) if items else 0.0,
        avg_rerank_score=sum(top_scores) / len(top_scores) if top_scores else None,
        k=k, n_kb=len(golden.kb), n_gap=len(golden.gap), retrieval_strategy=settings.retrieval.strategy,
        reranker_status=stack.registry.get(Component.RERANKER).status.value,
        prompt_versions=dict(settings.prompts),
        models={"main": models.main.id, "light": models.light.id, "embedding": models.embedding.id,
                "reranker": models.reranker.id},
        usd=round(spent() - base, 6),
    )


def build_live_stack(settings: Settings, work_dir: Path) -> AgentStack:
    """The real models over the real index with mock tickets, in a scratch folder (see `evals/gap_eval.py`)."""
    from evals.gap_eval import build_live_stack as build

    return build(settings, work_dir)


DriftRunner = Callable[[], DriftPoint]


def make_runner(
    settings: Settings,
    *,
    golden: Golden | None = None,
    command: Sequence[str] = ("python", "-m", "smartdesk_ai.ops.drift"),
) -> DriftRunner:
    """A function that runs one drift check in a scratch folder and returns its point (the caller stores it).
    A live run is metered in the spend ledger and stops at the ceiling."""

    # The golden questions never create tickets, whatever modes.tickets says: the run always uses mock tickets.
    run_settings = settings.model_copy(update={"modes": settings.modes.model_copy(update={"tickets": "mock"})})

    def runner() -> DriftPoint:
        chosen = golden or Golden.load()
        mock = run_settings.modes.llm == "mock"
        estimate = estimate_drift(run_settings, chosen.size)
        with tempfile.TemporaryDirectory(prefix="drift_") as work:
            if mock:
                stack = build_stack(run_settings, environ={}, dotenv_path=None, data_dir=Path(work), use_cache=False)
                return run_drift(stack, run_settings, golden=chosen, mock=True)
            started_here = current_run() is None
            start_run(run_settings, list(command), "drift")
            try:
                stack = build_live_stack(run_settings, Path(work))
                return run_drift(stack, run_settings, golden=chosen, mock=False, ceiling_usd=estimate.ceiling_usd)
            finally:
                if started_here:  # never end a run the app itself opened
                    end_run()

    return runner


def main(
    argv: Sequence[str] | None = None,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
    environ: Mapping[str, str] | None = None,
    dotenv_path: Path | None = DEFAULT_DOTENV,
    data_dir: Path | None = None,
    config_path: Path = DEFAULT_CONFIG,
    golden_set: Golden | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m smartdesk_ai.ops.drift", description="Drift check on the golden set")
    parser.add_argument("--estimate", action="store_true", help="print the cost estimate and stop")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt (mock mode only)")
    args = parser.parse_args(argv)
    settings = load_settings(config_path, environ=environ)
    golden = golden_set or Golden.load()
    estimate = estimate_drift(settings, golden.size)
    for line in describe_estimate(estimate, golden):
        output(line)
    if args.estimate:
        return 0
    if args.yes and not estimate.free:
        output("[error] --yes is not accepted for live runs: they need the interactive confirmation.")
        return 2
    if not args.yes and input_fn("Run it? Type 'yes' to confirm: ").strip().lower() != "yes":
        output("Cancelled. Nothing was run.")
        return 0
    runner = make_runner(settings, golden=golden)
    try:
        point = runner()
    except DriftCeilingExceeded as stop:
        output(f"[stopped] {stop}. Nothing was stored.")
        return 1
    except LiveModeUnavailable as error:
        output(f"[error] Cannot run in live mode: {error}")
        return 2
    DriftHistory(resolve_data_dir(data_dir, environ) / DRIFT_LOG).append(point)
    shown = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    output(f"Stored: faithfulness {shown(point.faithfulness)}, recall@{point.k} {shown(point.recall_at_k)}, "
           f"escalation rate {shown(point.escalation_rate)}, avg rerank score {shown(point.avg_rerank_score)} "
           f"(${point.usd:.4f}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
