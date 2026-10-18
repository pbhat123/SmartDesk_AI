"""Agent metrics from the scenario scripts (FR-H11). Each script in `evals/scenarios` is one task, run
through the whole app (the real graph with mock models and mock tickets, or live ones later).

- Task completion rate: tasks in which no turn failed, meaning no exception, no failed ticket creation and
  no "try again later" outage reply. A cancelled ticket is a completed task (the employee chose it).
- Task accuracy: tasks in which every turn matched its script (`evals/scenario_check.py`).
- Steps to completion: graph steps across a completed task's turns, averaged (the leading "start" is not a step).
- End-to-end latency: mean wall time per task, and p50 and p95 per turn.
- Cost per task: mean USD and tokens from the cost tracker (mock runs price nothing but count tokens).
- Error rate: failed turns over all turns.
Throughput (requests per second) comes from `scripts/load_test.py`.
"""
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from evals.common import EVALS_DIR
from evals.harness import Resources
from evals.scenario_check import check_turn
from evals.scenario_schema import Scenario, load_scenario
from smartdesk_ai.agent.factory import build_stack
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.ops.health import Component, HealthRegistry
from smartdesk_ai.ops.metrics import percentile
from smartdesk_ai.ops.runtime import RequestContext, request_scope
from smartdesk_ai.tickets import MockTicketClient

SCENARIO_DIR = EVALS_DIR / "scenarios"
FAILURE_PHRASES = ("try again later", "try again in a few minutes", "temporarily unavailable")
_DEGRADE = {"reranker": Component.RERANKER, "presidio": Component.PRESIDIO, "redis": Component.REDIS,
            "tracing": Component.LANGFUSE}


class TurnRun(BaseModel):
    matched: bool
    failed: bool
    latency_ms: float
    steps: int
    tokens: int = 0
    usd: float = 0.0
    problems: list[str] = Field(default_factory=list)


class TaskRun(BaseModel):
    id: str
    kind: str
    turns: list[TurnRun]

    @property
    def completed(self) -> bool:
        return not any(t.failed for t in self.turns)

    @property
    def accurate(self) -> bool:
        return all(t.matched for t in self.turns)

    @property
    def steps(self) -> int:
        return sum(t.steps for t in self.turns)

    @property
    def latency_ms(self) -> float:
        return sum(t.latency_ms for t in self.turns)

    @property
    def usd(self) -> float:
        return sum(t.usd for t in self.turns)

    @property
    def tokens(self) -> int:
        return sum(t.tokens for t in self.turns)


class AgentReport(BaseModel):
    n_tasks: int
    n_turns: int
    completion_rate: float | None
    task_accuracy: float | None
    steps_to_completion: float | None
    turns_to_completion: float | None
    task_latency_ms: float | None
    turn_p50_ms: float | None
    turn_p95_ms: float | None
    usd_per_task: float | None
    tokens_per_task: float | None
    error_rate: float | None
    tasks: list[dict] = Field(default_factory=list)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(tasks: Sequence[TaskRun]) -> AgentReport:
    turns = [t for task in tasks for t in task.turns]
    done = [t for t in tasks if t.completed]
    latencies = [t.latency_ms for t in turns]
    return AgentReport(
        n_tasks=len(tasks), n_turns=len(turns),
        completion_rate=(len(done) / len(tasks)) if tasks else None,
        task_accuracy=(sum(t.accurate for t in tasks) / len(tasks)) if tasks else None,
        steps_to_completion=_mean([t.steps for t in done]),
        turns_to_completion=_mean([len(t.turns) for t in done]),
        task_latency_ms=_mean([t.latency_ms for t in tasks]),
        turn_p50_ms=percentile(latencies, 0.50) if latencies else None,
        turn_p95_ms=percentile(latencies, 0.95) if latencies else None,
        usd_per_task=_mean([t.usd for t in tasks]), tokens_per_task=_mean([t.tokens for t in tasks]),
        error_rate=(sum(t.failed for t in turns) / len(turns)) if turns else None,
        tasks=[{"id": t.id, "completed": t.completed, "accurate": t.accurate, "steps": t.steps,
                "latency_ms": t.latency_ms, "usd": t.usd, "tokens": t.tokens} for t in tasks],
    )


def _failed_reply(reply: str) -> bool:
    return any(phrase in reply.lower() for phrase in FAILURE_PHRASES)


def run_scenarios(
    scenarios: Sequence[Scenario], settings: Settings, work_dir: Path, *, clock: Callable[[], float] = time.perf_counter
) -> list[TaskRun]:
    """Run each script on its own fresh app (mock models and mock tickets) and time every turn."""
    runs = []
    for scenario in scenarios:
        client = MockTicketClient()
        registry = HealthRegistry()
        for component in scenario.setup.degraded:
            registry.mark_degraded(_DEGRADE[component], "scripted degraded mode")
        stack = build_stack(settings, environ={}, dotenv_path=None, data_dir=work_dir / scenario.id,
                            registry=registry, client=client)
        client.unreachable = scenario.setup.github == "down"  # after the demo tickets were seeded
        session = new_thread_id()
        turns = []
        for index, scripted in enumerate(scenario.turns):
            ctx = RequestContext(session_id=session, request_id=f"{scenario.id}-{index}", task=scenario.id)
            issues_before, started = len(client._issues), clock()
            problems: list[str] = []
            failed = False
            steps = 0
            try:
                with request_scope(ctx, stack.services):
                    state = run_turn(stack.graph, session, scripted.user,
                                     email=scenario.setup.email_known if index == 0 else None)
                latency = (clock() - started) * 1000
                steps = len([s for s in state.steps if s != "start"])
                problems = check_turn(scripted.expect, state, client=client, issues_before=issues_before,
                                      issues_after=len(client._issues), searched="search_kb" in state.steps)
                failed = _failed_reply(state.reply) or "create_failed" in state.steps
            except Exception as error:  # a crash is a failed turn; its message could echo user text, so only the type is kept
                latency = (clock() - started) * 1000
                problems, failed = [f"turn raised {type(error).__name__}"], True
            mine = [e for e in stack.services.cost.entries if e.request_id == ctx.request_id]
            turns.append(TurnRun(
                matched=not problems, failed=failed, latency_ms=latency, steps=steps,
                tokens=sum(e.input_tokens + e.output_tokens for e in mine), usd=sum(e.usd for e in mine),
                problems=problems))
        runs.append(TaskRun(id=scenario.id, kind=scenario.kind, turns=turns))
    return runs


def run(res: Resources, out_dir: Path | None, work_dir: Path) -> AgentReport:
    scenarios = [load_scenario(p) for p in sorted(SCENARIO_DIR.glob("*.yaml"))]
    report = summarize(run_scenarios(scenarios, res.settings, work_dir))
    print(f"agent metrics: {report.n_tasks} tasks, {report.n_turns} turns")
    print(f"  completion {report.completion_rate:.3f}  accuracy {report.task_accuracy:.3f}  error rate {report.error_rate:.3f}")
    print(f"  steps to completion {report.steps_to_completion:.1f}  mean task latency {report.task_latency_ms:.0f} ms"
          f"  turn p50 {report.turn_p50_ms:.0f} ms p95 {report.turn_p95_ms:.0f} ms")
    print(f"  cost per task ${report.usd_per_task:.4f} ({report.tokens_per_task:.0f} tokens)")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "agent_metrics.json").write_text(json.dumps({"mock": res.mock, **report.model_dump()}, indent=2))
    return report
