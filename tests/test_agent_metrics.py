"""T6.3: agent metrics from the scenario scripts (FR-H11): task completion rate, task accuracy, steps to
completion, end-to-end latency, cost per task and error rate. Throughput comes from the load test later.
Mock mode throughout; a fake clock makes latency exact."""
import json

import pytest

from evals import agent_metrics as am
from evals.scenario_schema import load_scenario
from evals.run import main
from evals.common import EVALS_DIR

SCENARIO_FILES = sorted((EVALS_DIR / "scenarios").glob("*.yaml"))


def turn(matched=True, failed=False, ms=10.0, steps=3, tokens=0, usd=0.0):
    return am.TurnRun(matched=matched, failed=failed, latency_ms=ms, steps=steps, tokens=tokens, usd=usd, problems=[])


def task(id, *turns):
    return am.TaskRun(id=id, kind="scenario", turns=list(turns))


# --- the aggregates against hand-computed fixtures -------------------------------------------------

FIXTURE = [
    task("a", turn(ms=100, steps=4, tokens=1000, usd=0.002), turn(ms=300, steps=6, tokens=500, usd=0.001)),
    task("b", turn(ms=200, steps=5, tokens=200, usd=0.0004)),
    task("c", turn(matched=False, failed=True, ms=400, steps=2), turn(matched=False, ms=50, steps=1)),
    task("d", turn(ms=150, steps=3, tokens=300, usd=0.0006), turn(matched=False, ms=250, steps=2)),
]


def test_completion_rate_is_tasks_without_a_failed_turn():
    assert am.summarize(FIXTURE).completion_rate == pytest.approx(3 / 4)  # only c has a failed turn


def test_task_accuracy_is_tasks_whose_turns_all_matched_the_script():
    assert am.summarize(FIXTURE).task_accuracy == pytest.approx(2 / 4)  # a and b


def test_steps_to_completion_is_the_mean_over_completed_tasks_only():
    report = am.summarize(FIXTURE)
    assert report.steps_to_completion == pytest.approx((4 + 6 + 5 + 3 + 2) / 3)  # a=10, b=5, d=5 -> 20/3
    assert report.turns_to_completion == pytest.approx((2 + 1 + 2) / 3)


def test_latency_is_the_mean_task_time_and_the_turn_percentiles():
    report = am.summarize(FIXTURE)
    assert report.task_latency_ms == pytest.approx((400 + 200 + 450 + 400) / 4)
    assert report.turn_p50_ms == pytest.approx(200.0)  # sorted: 50 100 150 200 250 300 400 -> nearest rank 4th
    assert report.turn_p95_ms == pytest.approx(400.0)


def test_cost_per_task_is_the_mean_over_all_tasks():
    report = am.summarize(FIXTURE)
    assert report.usd_per_task == pytest.approx((0.003 + 0.0004 + 0.0 + 0.0006) / 4)
    assert report.tokens_per_task == pytest.approx((1500 + 200 + 0 + 300) / 4)


def test_error_rate_is_failed_turns_over_all_turns():
    assert am.summarize(FIXTURE).error_rate == pytest.approx(1 / 7)


def test_an_empty_run_has_no_rates():
    report = am.summarize([])
    assert report.n_tasks == 0 and report.completion_rate is None and report.task_accuracy is None


def test_no_completed_task_gives_no_step_average():
    assert am.summarize([task("x", turn(failed=True))]).steps_to_completion is None


# --- the scenario scripts, in mock mode ------------------------------------------------------------------

class TickClock:
    """Every reading advances 5 ms, so a timed turn takes exactly 5 ms."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.005
        return self.now


@pytest.fixture(scope="module")
def mock_tasks(settings, tmp_path_factory):
    scenarios = [load_scenario(p) for p in SCENARIO_FILES]
    return scenarios, am.run_scenarios(scenarios, settings, tmp_path_factory.mktemp("agent"), clock=TickClock())


def test_every_scenario_script_is_run(mock_tasks):
    scenarios, tasks = mock_tasks
    assert [t.id for t in tasks] == [s.id for s in scenarios] and len(tasks) == 12
    assert [len(t.turns) for t in tasks] == [len(s.turns) for s in scenarios]


def test_all_scripts_behave_as_written(mock_tasks):
    _, tasks = mock_tasks
    wrong = {t.id: [p for r in t.turns for p in r.problems] for t in tasks if not all(r.matched for r in t.turns)}
    assert wrong == {}
    assert am.summarize(tasks).task_accuracy == 1.0


def test_only_the_api_down_script_fails_to_complete(mock_tasks):
    _, tasks = mock_tasks
    assert [t.id for t in tasks if not t.completed] == ["api_down"]
    report = am.summarize(tasks)
    total_turns = sum(len(t.turns) for t in tasks)
    assert report.completion_rate == pytest.approx(11 / 12)
    assert report.error_rate == pytest.approx(1 / total_turns)  # one failed turn: the creation that could not reach GitHub


def test_latency_comes_from_the_injected_clock(mock_tasks):
    _, tasks = mock_tasks
    assert all(r.latency_ms == pytest.approx(5.0) for t in tasks for r in t.turns)
    assert am.summarize(tasks).turn_p95_ms == pytest.approx(5.0)


def test_steps_are_counted_from_the_graph_trace(mock_tasks):
    _, tasks = mock_tasks
    flow_a = next(t for t in tasks if t.id == "flow_a")
    assert flow_a.turns[0].steps == 4  # route, search_kb, answer, reply (the leading "start" is not a step)
    assert all(r.steps >= 1 for t in tasks for r in t.turns)


def test_mock_cost_is_zero_dollars_but_tokens_are_counted(mock_tasks):
    _, tasks = mock_tasks
    report = am.summarize(tasks)
    assert report.usd_per_task == 0.0 and report.tokens_per_task > 0


def test_a_turn_that_raises_counts_as_failed_and_unmatched(settings, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("graph exploded")

    monkeypatch.setattr(am, "run_turn", boom)
    [run] = am.run_scenarios([load_scenario(EVALS_DIR / "scenarios" / "flow_a.yaml")], settings, tmp_path)
    assert not run.completed and not run.accurate and run.turns[0].failed and "RuntimeError" in run.turns[0].problems[0]
    assert "graph exploded" not in run.turns[0].problems[0]


def test_the_runner_writes_a_result_file_and_the_cli_step_works(tmp_path):
    assert main(["--mock", "--only", "agent", "--out-dir", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "agent_metrics.json").read_text())
    assert data["mock"] is True and data["n_tasks"] == 12 and data["task_accuracy"] == 1.0
    assert len(data["tasks"]) == 12 and (tmp_path / "run_record.json").exists()
    assert json.loads((tmp_path / "run_record.json").read_text())["result_files"] == ["agent_metrics.json"]


def test_a_real_agent_run_is_not_offered_yet(capsys):
    assert main(["--only", "agent"]) == 1
    assert "--mock" in capsys.readouterr().err
