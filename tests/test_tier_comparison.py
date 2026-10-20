"""T6.12: right-sizing comparison, the tier map against every task on Main-Model (FR-H17). Mock models only: no test
here calls a paid model, and the live run is not part of this task."""
import json

import pytest

from evals import tier_comparison as tc
from evals.common import EVALS_DIR
from evals.harness import make_resources
from evals.judge_run import SpendCeilingExceeded
from evals.scenario_schema import load_scenario
from smartdesk_ai.ops.cost import CostEntry, price_entries

SCENARIOS = [load_scenario(EVALS_DIR / "scenarios" / f"{name}.yaml")
             for name in ("kb_answer", "ticket_create_yes", "status_1")]


@pytest.fixture(scope="module")
def mock_comparison(tmp_path_factory):
    res = make_resources(mock=True, need_reranker=False)
    return tc.compare(res, tmp_path_factory.mktemp("tiers"), scenarios=SCENARIOS, kb_limit=8)


# --- the two configurations ------------------------------------------------------------------------


def test_the_all_main_configuration_puts_every_task_on_the_main_tier(settings):
    changed = tc.settings_for("all_main", settings)
    assert set(changed.tiers) == set(settings.tiers) and set(changed.tiers.values()) == {"main"}
    assert "light" in settings.tiers.values()  # the real map is untouched
    assert tc.settings_for("tier_map", settings).tiers == settings.tiers


def test_an_unknown_configuration_is_refused(settings):
    with pytest.raises(ValueError):
        tc.settings_for("everything_light", settings)


# --- pricing -----------------------------------------------------------------------------------------


def entry(tier, input_tokens, output_tokens, step="llm", cached=0):
    return CostEntry(ts="t", request_id="r", session_id="s", task="eval", agent="a", step=step, tier=tier,
                     input_tokens=input_tokens, cached_tokens=cached, output_tokens=output_tokens)


def test_entries_are_priced_from_the_config_tier_prices_even_in_mock_mode(settings):
    main, light, emb = settings.models.main.price, settings.models.light.price, settings.models.embedding.price
    entries = [entry("main", 1000, 200), entry("light", 1000, 200, cached=400),
               entry(None, 500, 0, step="embedding"), entry(None, 0, 0, step="tool")]
    expected = ((1000 * main.input + 200 * main.output) + (600 * light.input + 400 * light.cached_input + 200 * light.output)
                + 500 * emb.input) / 1e6
    assert price_entries(entries, settings) == pytest.approx(expected)


# --- the mock comparison -------------------------------------------------------------------------------


def test_both_configurations_are_reported_side_by_side(mock_comparison):
    assert [r.name for r in mock_comparison.results] == ["tier_map", "all_main"] and mock_comparison.mock is True
    for result in mock_comparison.results:
        assert result.n_tasks == len(SCENARIOS) and result.usd_per_task > 0 and result.tokens_per_task > 0
        assert result.kb.faithfulness is not None and result.kb.answer_correctness is not None
        assert result.kb.n_questions == 8 and result.kb.usd_per_question > 0


def test_only_the_price_differs_when_the_models_behave_alike(mock_comparison):
    tier_map, all_main = mock_comparison.by_name["tier_map"], mock_comparison.by_name["all_main"]
    assert tier_map.tokens_per_task == pytest.approx(all_main.tokens_per_task)  # same prompts, same replies
    assert (tier_map.completion_rate, tier_map.task_accuracy) == (all_main.completion_rate, all_main.task_accuracy)
    assert all_main.usd_per_task > tier_map.usd_per_task  # light-tier tasks are priced as Main-Model in the second run
    assert tier_map.by_tier["light"].input_tokens > 0
    assert all_main.by_tier.get("light") is None or all_main.by_tier["light"].input_tokens == 0
    assert all_main.by_tier["main"].input_tokens == pytest.approx(
        tier_map.by_tier["main"].input_tokens + tier_map.by_tier["light"].input_tokens)


def test_a_task_that_uses_a_light_model_costs_more_on_all_main_and_one_that_does_not_costs_the_same(mock_comparison):
    per_task = {name: {t.id: t.usd for t in mock_comparison.by_name[name].tasks} for name in tc.CONFIGS}
    assert per_task["all_main"]["ticket_create_yes"] > per_task["tier_map"]["ticket_create_yes"]
    for task_id in per_task["tier_map"]:
        assert per_task["all_main"][task_id] >= per_task["tier_map"][task_id]


def test_the_kb_set_runs_once_when_both_configurations_use_the_same_models_on_that_path(mock_comparison):
    assert mock_comparison.kb_runs == 1
    one, two = (r.kb for r in mock_comparison.results)
    assert one == two and mock_comparison.results[0].kb_shared and mock_comparison.results[1].kb_shared


def test_the_kb_set_runs_for_each_configuration_when_the_path_differs_or_when_asked(settings, tmp_path):
    res = make_resources(mock=True, need_reranker=False)
    both = tc.compare(res, tmp_path / "both", scenarios=SCENARIOS[:1], kb_limit=4, kb_both=True)
    assert both.kb_runs == 2 and not any(r.kb_shared for r in both.results)
    lighter = settings.model_copy(update={"tiers": {**settings.tiers, "WORKER_ANSWER": "light"}})
    assert tc.kb_path_differs(lighter) and not tc.kb_path_differs(settings)


def test_the_judge_is_injectable_and_decides_the_quality_columns(tmp_path):
    from evals.rag_metrics import MockJudge

    class Harsh(MockJudge):
        name = "harsh"

        def faithfulness(self, question, answer, contexts):
            return 0.25

    res = make_resources(mock=True, need_reranker=False)
    result = tc.compare(res, tmp_path, scenarios=SCENARIOS[:1], kb_limit=4, judge=Harsh())
    assert all(r.kb.faithfulness == pytest.approx(0.25) for r in result.results)


def test_the_comparison_stops_between_tasks_once_the_spend_ceiling_is_passed(tmp_path):
    res = make_resources(mock=True, need_reranker=False)
    guard = tc.SpendGuard(ceiling_usd=1e-9)
    with pytest.raises(SpendCeilingExceeded):
        tc.compare(res, tmp_path, scenarios=SCENARIOS, kb_limit=4, guard=guard)
    assert guard.spent > 0


def test_the_guard_adds_up_and_raises_past_the_ceiling():
    guard = tc.SpendGuard(ceiling_usd=0.10)
    guard.add(0.06)
    with pytest.raises(SpendCeilingExceeded):
        guard.add(0.05)
    assert guard.spent == pytest.approx(0.11)
    tc.SpendGuard(ceiling_usd=None).add(1e6)  # no ceiling, no stop


# --- the estimate ----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def estimate(settings):
    return tc.estimate_comparison(settings, SCENARIOS, n_kb=53, kb_runs=1)


def test_the_estimate_prices_scenarios_kb_answers_and_the_ragas_judge_per_configuration(estimate, settings):
    assert estimate.scenario_usd["all_main"] > estimate.scenario_usd["tier_map"] > 0
    price = settings.models.main.price
    tokens = settings.tier_comparison.ragas_judge_tokens
    assert estimate.judge_usd_per_pass == pytest.approx(
        (tokens.main_input * price.input + tokens.main_output * price.output) / 1e6)
    assert estimate.kb_usd_per_run > 0
    assert estimate.total_usd == pytest.approx(
        sum(estimate.scenario_usd.values()) + estimate.kb_usd_per_run + estimate.judge_usd_per_pass)
    assert estimate.ceiling_usd == pytest.approx(estimate.total_usd * settings.tier_comparison.ceiling_multiplier)


def test_running_the_kb_set_twice_costs_exactly_one_more_kb_run_and_judge_pass(settings, estimate):
    twice = tc.estimate_comparison(settings, SCENARIOS, n_kb=53, kb_runs=2)
    assert twice.total_usd - estimate.total_usd == pytest.approx(estimate.kb_usd_per_run + estimate.judge_usd_per_pass)


# --- the command ------------------------------------------------------------------------------------------


def no_paid_call(*args, **kwargs):
    raise AssertionError("a paid run was started")


def test_without_flags_the_command_prints_the_estimate_and_runs_nothing(capsys, monkeypatch):
    monkeypatch.setattr(tc, "run_live", no_paid_call)
    monkeypatch.setattr(tc, "compare", no_paid_call)
    assert tc.main([]) == 2
    out = capsys.readouterr().out
    assert "Estimated cost" in out and "--confirm-spend" in out and "--mock" in out


def test_estimate_only_prints_the_estimate_and_exits_cleanly(capsys, monkeypatch):
    monkeypatch.setattr(tc, "run_live", no_paid_call)
    assert tc.main(["--estimate"]) == 0
    assert "ceiling" in capsys.readouterr().out.lower()


def test_a_live_run_needs_confirm_spend_and_gets_the_estimates_ceiling(capsys, monkeypatch):
    seen = {}
    monkeypatch.setattr(tc, "run_live", lambda settings, args, estimate: seen.update(ceiling=estimate.ceiling_usd) or 0)
    assert tc.main(["--confirm-spend"]) == 0 and seen["ceiling"] > 0


def test_the_mock_run_prints_both_columns_and_writes_the_result_file(capsys, tmp_path):
    assert tc.main(["--mock", "--out-dir", str(tmp_path), "--kb-limit", "4"]) == 0
    out = capsys.readouterr().out
    for text in ("tier_map", "all_main", "cost per task", "faithfulness", "answer correctness", "MOCK"):
        assert text in out, text
    saved = json.loads((tmp_path / "tier_comparison.json").read_text())
    assert saved["mock"] is True and [r["name"] for r in saved["results"]] == ["tier_map", "all_main"]


# --- T7.11: cost by agent, the ledger ceiling, and the Ragas report from the same judge pass ---------------------------

def test_every_configuration_reports_cost_by_agent_and_it_adds_up_to_the_tiers(mock_comparison):
    for result in mock_comparison.results:
        assert result.by_agent, result.name
        assert "orchestrator" in result.by_agent and any(a.endswith("_worker") or a == "misc_search" for a in result.by_agent)
        by_agent_usd = sum(u.usd for name, u in result.by_agent.items() if name != "retrieval")
        by_tier_usd = sum(u.usd for u in result.by_tier.values())
        assert by_agent_usd == pytest.approx(by_tier_usd), result.name  # the same LLM calls grouped two ways


def test_the_table_has_a_row_per_agent_for_both_configurations(mock_comparison):
    text = "\n".join(tc.format_table(mock_comparison))
    assert "cost by agent (all tasks, USD)" in text and "  orchestrator" in text


def test_the_spend_guard_also_reads_the_ledger():
    spent = [0.0]
    guard = tc.SpendGuard(0.10, ledger=lambda: spent[0])
    guard.add(0.01)
    spent[0] = 0.25  # the ledger counts calls the run's own tracker did not see
    with pytest.raises(SpendCeilingExceeded):
        guard.add(0.0)


def test_the_live_run_keeps_the_ragas_report_of_the_shared_judge_pass(tmp_path, monkeypatch):
    seen = {}
    out = tmp_path / "tier_map"
    out.mkdir()
    (out / "rag_metrics.json").write_text(json.dumps({"context_precision": 0.9}))
    path = tc.save_ragas(tmp_path, tmp_path / "results")
    assert path.name == "rag_metrics_final.json" and json.loads(path.read_text())["context_precision"] == 0.9
    assert tc.save_ragas(tmp_path / "nowhere", tmp_path / "results") is None
    assert not seen


def test_a_live_run_without_confirm_spend_runs_nothing(capsys, monkeypatch):
    monkeypatch.setattr(tc, "run_live", lambda *a, **k: (_ for _ in ()).throw(AssertionError("paid run")))
    assert tc.main([]) == 2 and "Nothing was run" in capsys.readouterr().out


def test_kept_answers_are_scored_again_on_a_rerun_without_asking_the_models(settings, tmp_path):
    from evals.harness import kb_rows
    from evals.rag_metrics import MockJudge, RagOutput, compute_metrics

    rows = kb_rows()[:3]
    asked = []

    def answer(row):
        asked.append(row.id)
        return RagOutput(question_id=row.id, answer="a", contexts=["c"], retrieved=[row.expected_entry_id],
                         citations=[row.expected_entry_id], insufficient=False)

    score = lambda name, outputs, rows: compute_metrics(outputs, rows, MockJudge())  # noqa: E731
    guard = tc.SpendGuard(None)
    first = tc.run_kb_set(settings, answer, rows, score, "tier_map", guard, cache_dir=tmp_path)
    again = tc.run_kb_set(settings, answer, rows, score, "tier_map", guard, cache_dir=tmp_path)
    assert asked == [r.id for r in rows] and again.usd == first.usd and again.n_questions == 3
    other = tc.run_kb_set(settings, answer, rows[:2], score, "tier_map", guard, cache_dir=tmp_path)  # other questions: asked again
    assert len(asked) == 5 and other.n_questions == 2


# --- scenarios only (the post-Phase-8 rerun: no KB answers, no Ragas judge pass) --------------------------------------


def test_scenarios_only_skips_the_kb_set_and_the_judge(tmp_path):
    class NoJudge:
        def __getattr__(self, name):
            raise AssertionError("the Ragas judge must not be called in a scenarios-only run")

    res = make_resources(mock=True, need_reranker=False)
    comparison = tc.compare(res, tmp_path, scenarios=SCENARIOS, scenarios_only=True, judge=NoJudge())
    assert comparison.scenarios_only is True and [r.name for r in comparison.results] == ["tier_map", "all_main"]
    assert all(r.kb is None and r.n_tasks == len(SCENARIOS) for r in comparison.results)
    text = "\n".join(tc.format_table(comparison))
    assert "cost per task" in text and "KB faithfulness" not in text and "Quality judge" not in text
    saved = json.loads(tc.save(comparison, tmp_path / "out").read_text())
    assert saved["scenarios_only"] is True and all(r["kb"] is None for r in saved["results"])


def test_the_scenarios_only_estimate_has_no_kb_or_judge_cost(settings):
    full = tc.estimate_comparison(settings, SCENARIOS, n_kb=10, kb_runs=1)
    only = tc.estimate_comparison(settings, SCENARIOS, n_kb=0, kb_runs=0)
    assert only.total_usd == pytest.approx(sum(full.scenario_usd.values())) and only.kb_usd_per_run == 0.0
    assert only.ceiling_usd == pytest.approx(only.total_usd * settings.tier_comparison.ceiling_multiplier)
    text = "\n".join(tc.describe_estimate(only))
    assert "Ragas" not in text and "scenario scripts" in text


def test_the_command_line_has_a_scenarios_only_estimate_that_is_free(capsys):
    assert tc.main(["--scenarios-only", "--estimate"]) == 0
    out = capsys.readouterr().out
    assert "scenario" in out and "Ragas" not in out
