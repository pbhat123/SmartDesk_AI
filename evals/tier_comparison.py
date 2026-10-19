"""Model right-sizing comparison (FR-H17): the task-to-tier map against every task on Main-Model.

The same scenario scripts (`evals/scenarios`, ticket, status and KB flows through the whole app) and the same KB set
run twice, once with the map in `config.yaml` and once with every task on the Main tier. The report puts them side by
side: cost per task, tokens by tier, task completion and accuracy, and on the KB set Ragas Faithfulness (the
grounded-answer rate), hallucination rate and answer correctness.

Cost is the tokens of each call times the config price of the tier it ran on, in mock mode too (a mock run has real
prompt sizes and scripted replies, so its dollars are what those tokens would cost, not a result). Quality numbers from
a mock run only exercise the code.

The KB answer path (orchestrator, worker, grounding judge, synthesizer) is on Main-Model in the default map, so both
configurations run the same models there. The KB set then runs once and both columns carry its numbers (`kb_shared`);
`--kb-both` runs it for each anyway. It runs for each as soon as the map puts one of those tasks on another tier.

Nothing paid runs by accident: with no flag the command prints the estimate and stops. `--mock` is a free dry run,
`--estimate` prints the estimate, and a live run needs `--confirm-spend`, runs under a spend ceiling (estimate x
`tier_comparison.ceiling_multiplier`) and a stall watchdog, and is metered in the spend ledger. The live run is built
but has not been run: it is combined with T7.11 after Phase 7.
"""
import argparse
import dataclasses
import json
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from evals import agent_metrics
from evals.harness import Resources, kb_rows, link_live_index, live_settings, make_resources
from evals.judge_run import EXIT_STALLED, STALL_SECONDS, SpendCeilingExceeded, Watchdog
from evals.rag_metrics import AnswerFn, MockJudge, RagOutput, RagReport, compute_metrics, make_answer_fn, mock_answerer
from evals.scenario_schema import Scenario, load_scenario
from smartdesk_ai.config.settings import Settings, load_settings
from smartdesk_ai.ops.cost import CostEntry, CostTracker, price_entries
from smartdesk_ai.ops.runtime import RequestContext, Services, request_scope

CONFIGS = ("tier_map", "all_main")
KB_PATH_TASKS = ("ORCHESTRATOR", "WORKER_ANSWER", "GROUNDING_JUDGE", "SYNTHESIZER")
RESULT_FILE = "tier_comparison.json"
MOCK_NOTE = ("MOCK run: prompts are the real ones, replies are scripted, and prices come from config.yaml. The cost "
             "columns show what these tokens would cost on each configuration; the quality columns only exercise the code.")


# --- the two configurations -----------------------------------------------------------------------------


def settings_for(name: str, settings: Settings) -> Settings:
    """`tier_map`: the settings as they are. `all_main`: every task on the Main tier."""
    if name == "tier_map":
        return settings
    if name == "all_main":
        return settings.model_copy(update={"tiers": {task: "main" for task in settings.tiers}})
    raise ValueError(f"unknown configuration {name!r}; known: {list(CONFIGS)}")


def kb_path_differs(settings: Settings) -> bool:
    """True when the map runs a task of the KB answer path on a tier other than Main, so all-Main changes that path."""
    return any(settings.tiers[task] != "main" for task in KB_PATH_TASKS)


def mock_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"modes": settings.modes.model_copy(update={"llm": "mock", "tickets": "mock"})})


# --- the spend ceiling -----------------------------------------------------------------------------------


class SpendGuard:
    """Adds up what a run has cost and raises `SpendCeilingExceeded` once it passes the ceiling. It is checked between
    tasks and questions, so a run stops after the one in flight. `on_beat` tells the watchdog the run is alive."""

    def __init__(self, ceiling_usd: float | None, on_beat: Callable[[], None] | None = None) -> None:
        self.ceiling_usd, self.spent, self.on_beat = ceiling_usd, 0.0, on_beat

    def add(self, usd: float) -> None:
        self.spent += usd
        if self.on_beat is not None:
            self.on_beat()
        if self.ceiling_usd is not None and self.spent > self.ceiling_usd:
            raise SpendCeilingExceeded(f"spend ${self.spent:.2f} is over the ${self.ceiling_usd:.2f} ceiling")


# --- results ---------------------------------------------------------------------------------------------


class TierUse(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0


class TaskCost(BaseModel):
    id: str
    usd: float
    tokens: int
    completed: bool
    accurate: bool


class KbResult(BaseModel):
    n_questions: int
    n_answered: int
    faithfulness: float | None  # Ragas Faithfulness, as the grounded-answer rate (FR-H12)
    hallucination_rate: float | None
    answer_correctness: float | None
    usd: float
    usd_per_question: float
    judge: str


class ConfigResult(BaseModel):
    name: str
    n_tasks: int
    completion_rate: float | None
    task_accuracy: float | None
    usd_per_task: float
    tokens_per_task: float
    by_tier: dict[str, TierUse]
    tasks: list[TaskCost]
    kb: KbResult
    kb_shared: bool = False  # the KB numbers come from one run that both configurations share


class Comparison(BaseModel):
    mock: bool
    kb_runs: int
    results: list[ConfigResult] = Field(default_factory=list)

    @property
    def by_name(self) -> dict[str, ConfigResult]:
        return {r.name: r for r in self.results}


def _by_tier(entries: Sequence[CostEntry], settings: Settings) -> dict[str, TierUse]:
    use: dict[str, TierUse] = defaultdict(TierUse)
    for entry in entries:
        if entry.step != "llm" or entry.tier is None:
            continue
        row = use[entry.tier]
        row.calls += 1
        row.input_tokens += entry.input_tokens
        row.output_tokens += entry.output_tokens
        row.usd += price_entries([entry], settings)
    return dict(use)


def _task_entries(task: agent_metrics.TaskRun) -> list[CostEntry]:
    return [e for turn in task.turns for e in turn.entries]


def scenario_result(name: str, variant: Settings, tasks: Sequence[agent_metrics.TaskRun]) -> dict:
    entries = [e for t in tasks for e in _task_entries(t)]
    costs = [TaskCost(id=t.id, usd=price_entries(_task_entries(t), variant), tokens=t.tokens, completed=t.completed,
                      accurate=t.accurate) for t in tasks]
    n = len(tasks)
    return dict(name=name, n_tasks=n, completion_rate=sum(t.completed for t in tasks) / n if n else None,
                task_accuracy=sum(t.accurate for t in tasks) / n if n else None,
                usd_per_task=sum(c.usd for c in costs) / n if n else 0.0,
                tokens_per_task=sum(c.tokens for c in costs) / n if n else 0.0,
                by_tier=_by_tier(entries, variant), tasks=costs)


# --- the KB set ------------------------------------------------------------------------------------------

KbPipeline = Callable[[Settings], AnswerFn]  # the answer function for one configuration's settings
Score = Callable[[str, Sequence[RagOutput], Sequence], RagReport]  # (configuration, outputs, rows) -> metrics


def mock_pipeline(res: Resources, index) -> KbPipeline:
    def build(variant: Settings) -> AnswerFn:
        search = variant.model_copy(update={"thresholds": variant.thresholds.model_copy(
            update={"escalation": variant.thresholds.escalation_mock})})  # hashing vectors score lower (as the app does)
        return make_answer_fn(search, index, res.reranker, mock_answerer(variant))

    return build


def run_kb_set(variant: Settings, answer_fn: AnswerFn, rows: Sequence, score: Score, name: str,
               guard: SpendGuard) -> KbResult:
    """Answer the KB questions through the agent's own search and answer path and score them. Every call runs under a
    request scope, so its tokens are counted and priced at the tiers' config prices."""
    tracker = CostTracker(variant, priced=False)
    outputs = []
    for row in rows:
        before = len(tracker.entries)
        ctx = RequestContext(session_id=f"tiers-{name}", request_id=f"tiers-{name}-{row.id}", task="eval")
        with request_scope(ctx, Services(cost=tracker)):
            outputs.append(answer_fn(row))
        guard.add(price_entries(tracker.entries[before:], variant))
    report = score(name, outputs, rows)
    usd = price_entries(tracker.entries, variant)
    return KbResult(n_questions=report.n_questions, n_answered=report.n_answered,
                    faithfulness=report.grounded_answer_rate, hallucination_rate=report.hallucination_rate,
                    answer_correctness=report.answer_correctness, usd=usd,
                    usd_per_question=usd / report.n_questions if report.n_questions else 0.0, judge=report.judge)


def all_scenarios() -> list[Scenario]:
    return [load_scenario(p) for p in sorted(agent_metrics.SCENARIO_DIR.glob("*.yaml"))]


def compare(
    res: Resources,
    work_dir: Path,
    *,
    index=None,
    scenarios: Sequence[Scenario] | None = None,
    kb_limit: int | None = None,
    kb_both: bool = False,
    judge=None,
    guard: SpendGuard | None = None,
    scenario_builder: agent_metrics.StackBuilder = agent_metrics.mock_stack,
    kb_pipeline: KbPipeline | None = None,
    score: Score | None = None,
) -> Comparison:
    """Run both configurations. The defaults are the free mock run; the live run passes live builders, a pipeline and a
    scorer. A `guard` stops the run between tasks and questions once its ceiling is passed."""
    guard = guard or SpendGuard(None)
    scenarios = list(scenarios) if scenarios is not None else all_scenarios()
    rows = kb_rows()[:kb_limit]
    if index is None and kb_pipeline is None:
        from evals.run import get_index  # a throwaway index in mock mode

        index = get_index(res, None)
    pipeline = kb_pipeline or mock_pipeline(res, index)
    judge = judge or MockJudge()
    score = score or (lambda name, outputs, rows: compute_metrics(outputs, rows, judge))
    shared = not (kb_both or kb_path_differs(res.settings))
    comparison = Comparison(mock=res.mock, kb_runs=1 if shared else 2)
    kb_done: KbResult | None = None
    for name in CONFIGS:
        variant = settings_for(name, res.settings)
        tasks = agent_metrics.run_scenarios(
            scenarios, variant, work_dir / name, build=scenario_builder,
            after_task=lambda task, v=variant: guard.add(price_entries(_task_entries(task), v)))
        if kb_done is None or not shared:
            kb_done = run_kb_set(variant, pipeline(variant), rows, score, name, guard)
        comparison.results.append(ConfigResult(kb=kb_done, kb_shared=shared, **scenario_result(name, variant, tasks)))
    return comparison


# --- the estimate ----------------------------------------------------------------------------------------


class Estimate(BaseModel):
    n_scenarios: int
    n_kb: int
    kb_runs: int
    scenario_usd: dict[str, float]  # all scenarios, per configuration
    kb_usd_per_run: float
    judge_usd_per_pass: float
    total_usd: float
    ceiling_usd: float


def _with_output_floor(entries: Sequence[CostEntry], floor: int) -> list[CostEntry]:
    """A mock reply is a few words; a live one averages `floor` tokens, so no call is priced below that."""
    return [e.model_copy(update={"output_tokens": max(e.output_tokens, floor)}) if e.step == "llm" else e for e in entries]


def estimate_comparison(settings: Settings, scenarios: Sequence[Scenario], *, n_kb: int, kb_runs: int) -> Estimate:
    """Scenarios: a free mock run measures the real prompt sizes of each configuration, priced with the config prices.
    KB answers: the per-question tokens measured in the live gap run (`drift.tokens_per_question`). Judge: one measured
    Ragas pass per KB run. All config numbers; nothing paid is called."""
    cfg, prices = settings.tier_comparison, settings.models
    base = mock_settings(settings)
    scenario_usd = {}
    with tempfile.TemporaryDirectory(prefix="tiers_estimate_") as work:
        for name in CONFIGS:
            variant = settings_for(name, base)
            tasks = agent_metrics.run_scenarios(list(scenarios), variant, Path(work) / name)
            entries = [e for t in tasks for e in _task_entries(t)]
            scenario_usd[name] = price_entries(_with_output_floor(entries, cfg.output_tokens_per_call), variant)
    q = settings.drift.tokens_per_question
    kb_per_question = (q.main_input * prices.main.price.input + q.main_output * prices.main.price.output
                       + q.embedding * prices.embedding.price.input) / 1_000_000
    judge_pass = (cfg.ragas_judge_tokens.main_input * prices.main.price.input
                  + cfg.ragas_judge_tokens.main_output * prices.main.price.output) / 1_000_000
    kb_run = n_kb * kb_per_question
    total = sum(scenario_usd.values()) + kb_runs * (kb_run + judge_pass)
    return Estimate(n_scenarios=len(scenarios), n_kb=n_kb, kb_runs=kb_runs, scenario_usd=scenario_usd,
                    kb_usd_per_run=kb_run, judge_usd_per_pass=judge_pass, total_usd=total,
                    ceiling_usd=total * cfg.ceiling_multiplier)


def describe_estimate(estimate: Estimate) -> list[str]:
    shared = ("The KB answer path is on Main-Model in both configurations, so the KB set runs once and both columns "
              "carry its numbers (--kb-both runs it for each)." if estimate.kb_runs == 1 else
              "The KB set runs once for each configuration.")
    return [
        f"Right-sizing comparison: {estimate.n_scenarios} scenario scripts on each configuration, plus "
        f"{estimate.n_kb} KB questions scored by the Ragas judge, {estimate.kb_runs} time(s).",
        shared,
        f"Estimated cost: about ${estimate.total_usd:.2f} = scenarios ${estimate.scenario_usd['tier_map']:.3f} (tier map) + "
        f"${estimate.scenario_usd['all_main']:.3f} (all Main) + {estimate.kb_runs} x (KB answers "
        f"${estimate.kb_usd_per_run:.3f} + Ragas judge ${estimate.judge_usd_per_pass:.3f}).",
        f"A live run stops after the task or question in flight once it passes the ceiling of ${estimate.ceiling_usd:.2f}.",
    ]


# --- the report ------------------------------------------------------------------------------------------


def _n(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_table(comparison: Comparison) -> list[str]:
    a, b = comparison.results
    lines = [("" if not comparison.mock else MOCK_NOTE), "",
             f"{'':34}{a.name:>14}{b.name:>14}"]

    def row(label: str, left: str, right: str) -> None:
        lines.append(f"{label:34}{left:>14}{right:>14}")

    row("scenario tasks", str(a.n_tasks), str(b.n_tasks))
    row("task completion", _n(a.completion_rate), _n(b.completion_rate))
    row("task accuracy", _n(a.task_accuracy), _n(b.task_accuracy))
    row("cost per task (USD)", _n(a.usd_per_task, 5), _n(b.usd_per_task, 5))
    row("tokens per task", f"{a.tokens_per_task:.0f}", f"{b.tokens_per_task:.0f}")
    for tier in ("main", "light"):
        left, right = a.by_tier.get(tier, TierUse()), b.by_tier.get(tier, TierUse())
        row(f"  {tier} tokens in / out (all tasks)", f"{left.input_tokens}/{left.output_tokens}",
            f"{right.input_tokens}/{right.output_tokens}")
        row(f"  {tier} cost (all tasks, USD)", _n(left.usd, 5), _n(right.usd, 5))
    row("KB faithfulness (grounded rate)", _n(a.kb.faithfulness), _n(b.kb.faithfulness))
    row("KB hallucination rate", _n(a.kb.hallucination_rate), _n(b.kb.hallucination_rate))
    row("KB answer correctness", _n(a.kb.answer_correctness), _n(b.kb.answer_correctness))
    row("KB answered / asked", f"{a.kb.n_answered}/{a.kb.n_questions}", f"{b.kb.n_answered}/{b.kb.n_questions}")
    row("cost per KB question (USD)", _n(a.kb.usd_per_question, 5), _n(b.kb.usd_per_question, 5))
    if a.kb_shared:
        lines.append("The KB columns come from one run: the answer path uses the same models in both configurations.")
    lines.append(f"Quality judge: {a.kb.judge}.")
    return lines


def save(comparison: Comparison, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / RESULT_FILE
    path.write_text(json.dumps(comparison.model_dump(mode="json"), indent=2))
    return path


# --- the live run (built, not yet run) ------------------------------------------------------------------------


def _live_stack(settings: Settings, data_dir: Path, client, registry):
    from smartdesk_ai.agent.factory import build_stack
    from smartdesk_ai.config.env import DEFAULT_DOTENV

    link_live_index(data_dir)
    return build_stack(live_settings(settings), environ=None, dotenv_path=DEFAULT_DOTENV, data_dir=data_dir,
                       use_cache=False, registry=registry, client=client)


def run_live(settings: Settings, args: argparse.Namespace, estimate: Estimate) -> int:
    """The paid run: real models, the real index, mock tickets, a ledger run, a spend ceiling and a stall watchdog.
    Not run yet; it is combined with T7.11 after Phase 7."""
    import os

    from evals.rag_metrics import judge_saved
    from evals.run import get_index
    from smartdesk_ai.agent.answer import make_answerer
    from smartdesk_ai.agent.grounding import make_grounding_judge
    from smartdesk_ai.llm.factory import openai_builder
    from smartdesk_ai.ops.spend_ledger import end_run, start_run

    live = live_settings(settings)
    res = dataclasses.replace(make_resources(mock=False), settings=live)
    start_run(live, ["python", "-m", "evals.tier_comparison", *sys.argv[1:]], "eval")
    guard = SpendGuard(estimate.ceiling_usd)

    def stalled() -> None:
        print("error: the comparison made no progress and was aborted", file=sys.stderr, flush=True)
        end_run("stalled")
        os._exit(EXIT_STALLED)  # a thread blocked in a network call cannot be interrupted any other way

    watchdog = Watchdog(STALL_SECONDS, stalled)
    guard.on_beat = watchdog.beat
    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="tier_comparison_"))
    status = "crashed"
    try:
        index = get_index(res, None)

        def pipeline(variant: Settings) -> AnswerFn:
            answerer = make_answerer(variant, builder=openai_builder,
                                     judge=make_grounding_judge(variant, builder=openai_builder))
            return make_answer_fn(variant, index, res.reranker, answerer)

        def live_score(name: str, outputs: Sequence[RagOutput], rows: Sequence) -> RagReport:
            variant_res = dataclasses.replace(res, settings=settings_for(name, live))
            return judge_saved(variant_res, outputs, rows, out_dir / name,
                               ceiling_usd=max(estimate.ceiling_usd - guard.spent, 0.0))

        watchdog.start()
        comparison = compare(res, out_dir, index=index, kb_limit=args.kb_limit, kb_both=args.kb_both, guard=guard,
                             scenario_builder=_live_stack, kb_pipeline=pipeline, score=live_score)
        print("\n".join(format_table(comparison)))
        print(f"saved: {save(comparison, out_dir)}")
        status = "ok"
        return 0
    except SpendCeilingExceeded as stop:
        print(f"[stopped] {stop}. Nothing was saved.", file=sys.stderr)
        status = "ceiling"
        return 1
    finally:
        watchdog.stop()
        end_run(status)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.tier_comparison", description=__doc__.split("\n\n")[0])
    parser.add_argument("--mock", action="store_true", help="free dry run with mock models")
    parser.add_argument("--estimate", action="store_true", help="print the cost estimate and stop")
    parser.add_argument("--confirm-spend", action="store_true", help="allow the live run, which calls paid models")
    parser.add_argument("--kb-both", action="store_true", help="run the KB set for each configuration even if the models match")
    parser.add_argument("--kb-limit", type=int, default=None, help="use only the first N KB questions")
    parser.add_argument("--out-dir", type=Path, default=None, help="write tier_comparison.json here")
    args = parser.parse_args(argv)
    settings = load_settings()
    if args.mock:
        res = make_resources(mock=True, need_reranker=False)
        with tempfile.TemporaryDirectory(prefix="tier_comparison_") as work:
            comparison = compare(res, Path(work), kb_limit=args.kb_limit, kb_both=args.kb_both)
        print("\n".join(format_table(comparison)))
        if args.out_dir:
            print(f"saved: {save(comparison, args.out_dir)}")
        return 0
    n_kb = len(kb_rows()[:args.kb_limit])
    kb_runs = 2 if args.kb_both or kb_path_differs(settings) else 1
    estimate = estimate_comparison(settings, all_scenarios(), n_kb=n_kb, kb_runs=kb_runs)
    print("\n".join(describe_estimate(estimate)))
    if args.estimate:
        return 0
    if not args.confirm_spend:
        print("Nothing was run. Add --mock for a free dry run, or --confirm-spend once you have approved the cost.")
        return 2
    return run_live(settings, args, estimate)


if __name__ == "__main__":
    raise SystemExit(main())
