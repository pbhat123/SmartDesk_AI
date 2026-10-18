"""T6.8: drift monitoring (FR-H15). The golden set (KB set plus gap set) is re-run on demand; Faithfulness, Recall@k,
escalation rate and average rerank score are stored with a timestamp, prompt versions and model IDs, and charted
in the Monitoring tab. A run only starts after the employee has seen the cost estimate and confirmed it.
Mock models and temp folders throughout: no test spends money or calls a live stack."""
import json
import re
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from evals.harness import gap_rows, kb_rows
from smartdesk_ai.agent.factory import build_stack
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.ops import drift
from smartdesk_ai.ops.drift import (
    DriftCeilingExceeded,
    DriftHistory,
    DriftPoint,
    Golden,
    estimate_drift,
    faithfulness_from_events,
    run_drift,
)
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.ui.monitoring import load_monitoring
from tests.ui_helpers import monitoring_app, charts, element_text, mock_context

FIXED = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)
PASSWORD = "How do I reset my forgotten password?"
WIFI = "Which Wi-Fi network should I use in the office?"


class Scripted:
    """An answer model keyed on the question: grounded text, ungrounded text, or insufficient context."""

    def invoke(self, messages):
        text = str(messages[-1].content)
        question = re.search(r"Question: (.*)", text).group(1).lower()
        ids = re.findall(r"^\[([A-Z]+-[A-Z0-9-]+)\]", text, re.M)
        if "tractor" in question:
            body = {"answer": "", "insufficient": True, "citations": []}
        elif "wi-fi" in question:
            body = {"answer": "zebra quantum widgets", "insufficient": False, "citations": ids[:1]}
        else:
            body = {"answer": "You can reset a forgotten or expired password without contacting the help desk.",
                    "insufficient": False, "citations": ids[:1]}
        return SimpleNamespace(content=json.dumps(body))


def golden(*rows) -> Golden:
    """rows: ("kb", question, expected_entry_id) or ("gap", question)."""
    kb = [SimpleNamespace(id=f"k{i}", question=r[1], expected_entry_id=r[2], split="dev")
          for i, r in enumerate(rows) if r[0] == "kb"]
    gap = [SimpleNamespace(id=f"g{i}", question=r[1], split="dev", grey_zone=False)
           for i, r in enumerate(rows) if r[0] == "gap"]
    return Golden(kb=kb, gap=gap)


def stack_for(settings, tmp_path, model=None):
    return build_stack(settings, environ={}, dotenv_path=None, data_dir=tmp_path, mock_answer_model=model or Scripted(),
                       registry=HealthRegistry())


def live(settings: Settings) -> Settings:
    return settings.model_copy(update={"modes": settings.modes.model_copy(update={"llm": "live"})})


# --- the four numbers, on a tiny set with known answers --------------------------------------------

def test_a_mock_run_stores_the_four_metrics_with_timestamp_versions_and_models(settings, tmp_path):
    rows = golden(("kb", PASSWORD, "IT-PWD-01"), ("gap", "How do I register my tractor for parking?"))
    point = run_drift(stack_for(settings, tmp_path), settings, golden=rows, mock=True, clock=lambda: FIXED)
    assert point.ts == FIXED.isoformat(timespec="seconds") and point.mode == "mock"
    assert point.prompt_versions == settings.prompts
    assert point.models == {"main": settings.models.main.id, "light": settings.models.light.id,
                            "embedding": settings.models.embedding.id, "reranker": settings.models.reranker.id}
    assert point.faithfulness == 1.0  # the one drafted answer was accepted by the grounding judge
    assert point.recall_at_k == 1.0 and point.k == settings.retrieval.top_n
    assert point.escalation_rate == 0.5  # the gap question was escalated, the KB question answered
    assert point.avg_rerank_score is not None and point.avg_rerank_score > 0
    assert (point.n_kb, point.n_gap) == (1, 1) and point.retrieval_strategy == settings.retrieval.strategy


def test_recall_counts_a_question_when_its_expected_entry_is_not_retrieved(settings, tmp_path):
    rows = golden(("kb", PASSWORD, "IT-PWD-01"), ("kb", PASSWORD, "HR-LEAVE-01"))  # the second is deliberately wrong
    point = run_drift(stack_for(settings, tmp_path), settings, golden=rows, mock=True, clock=lambda: FIXED)
    assert point.recall_at_k == 0.5


def test_an_answer_the_judge_rejects_lowers_faithfulness_and_counts_as_escalated(settings, tmp_path):
    rows = golden(("kb", PASSWORD, "IT-PWD-01"), ("kb", WIFI, "IT-WIFI-02"))
    point = run_drift(stack_for(settings, tmp_path), settings, golden=rows, mock=True, clock=lambda: FIXED)
    assert 0.0 < point.faithfulness < 1.0
    assert point.escalation_rate == 0.5  # the Wi-Fi draft was never shown


def test_faithfulness_is_the_share_of_judge_verdicts_that_were_grounded():
    judge = lambda grounded: {"kind": "llm_call", "task": "GROUNDING_JUDGE",  # noqa: E731
                              "completion": json.dumps({"grounded": grounded, "unsupported": []})}
    events = [judge(True), judge(True), judge(False), {"kind": "llm_call", "task": "ORCHESTRATOR", "completion": "x"},
              {"kind": "reply"}, {"kind": "llm_call", "task": "GROUNDING_JUDGE", "completion": "not json"}]
    assert faithfulness_from_events(events) == pytest.approx(2 / 3)
    assert faithfulness_from_events([]) is None


# --- the cost estimate and the spend ceiling ----------------------------------------------------------

def test_the_estimate_is_questions_times_per_question_tokens_at_config_prices(settings):
    tokens, price = settings.drift.tokens_per_question, settings.models
    per_question = (tokens.main_input * price.main.price.input + tokens.main_output * price.main.price.output
                    + tokens.embedding * price.embedding.price.input) / 1_000_000
    estimate = estimate_drift(live(settings), 74)
    assert estimate.usd == pytest.approx(74 * per_question) and not estimate.free
    assert estimate.ceiling_usd == pytest.approx(estimate.usd * settings.drift.ceiling_multiplier)
    assert 0.2 < estimate.usd < 0.6  # about what the live gap run of 2026-10-18 cost ($0.34 for 74 questions)


def test_a_mock_run_is_free(settings):
    estimate = estimate_drift(settings, 74)
    assert estimate.usd == 0.0 and estimate.free and estimate.n_questions == 74


def test_the_estimate_counts_the_real_golden_set():
    assert len(kb_rows()) + len(gap_rows()) == 74 == drift.Golden.load().size


def test_a_run_stops_and_stores_nothing_when_spend_passes_the_ceiling(settings, tmp_path):
    spent = iter(x * 0.1 for x in range(100))
    rows = golden(*[("kb", PASSWORD, "IT-PWD-01")] * 6)
    with pytest.raises(DriftCeilingExceeded, match="0.25"):
        run_drift(stack_for(settings, tmp_path), settings, golden=rows, mock=True, ceiling_usd=0.25,
                  spent=lambda: next(spent))


# --- history ---------------------------------------------------------------------------------------

def _point(**kw) -> DriftPoint:
    base = dict(ts="2026-10-18T12:00:00+00:00", mode="mock", faithfulness=1.0, recall_at_k=0.9, escalation_rate=0.3,
                avg_rerank_score=2.5, k=5, n_kb=53, n_gap=21, retrieval_strategy="hybrid_rerank",
                reranker_status="unknown", prompt_versions={"answer": 4}, models={"main": "m"}, usd=0.0)
    return DriftPoint(**{**base, **kw})


def test_history_round_trips_in_time_order_and_skips_damaged_lines(tmp_path):
    history = DriftHistory(tmp_path / "drift.jsonl")
    assert history.points() == []
    history.append(_point(ts="2026-10-19T00:00:00+00:00"))
    history.append(_point(ts="2026-10-18T00:00:00+00:00"))
    with (tmp_path / "drift.jsonl").open("a") as f:
        f.write('{"ts": "broken\n')
    assert [p.ts[:10] for p in history.points()] == ["2026-10-18", "2026-10-19"]


def test_reading_history_creates_nothing(tmp_path):
    DriftHistory(tmp_path / "sub" / "drift.jsonl").points()
    assert list(tmp_path.iterdir()) == []


def test_the_runner_never_uses_live_tickets_even_if_config_says_so(settings, tmp_path, monkeypatch):
    live_tickets = settings.model_copy(update={"modes": settings.modes.model_copy(update={"tickets": "live"})})
    seen = []
    real = drift.build_stack
    monkeypatch.setattr(drift, "build_stack", lambda s, **kw: (seen.append(s.modes.tickets), real(s, **kw))[1])
    point = drift.make_runner(live_tickets, golden=golden(("kb", PASSWORD, "IT-PWD-01")))()
    assert seen == ["mock"] and point.mode == "mock"


# --- the command: estimate, confirm, then run --------------------------------------------------------

def _run_main(tmp_path, argv, answers=(), settings_path=None, golden_set=None):
    out, feed = [], iter(answers)
    kwargs = {"config_path": settings_path} if settings_path else {}
    code = drift.main(argv, input_fn=lambda prompt: (out.append(prompt), next(feed))[1], output=out.append,
                      environ={}, dotenv_path=None, data_dir=tmp_path, golden_set=golden_set, **kwargs)
    return code, "\n".join(out)


@pytest.fixture
def live_config(tmp_path):
    from smartdesk_ai.config.settings import DEFAULT_CONFIG

    path = tmp_path / "live.yaml"
    path.write_text(DEFAULT_CONFIG.read_text().replace("  llm: mock", "  llm: live"))
    return path


@pytest.fixture
def no_live_stack(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a live stack was built")

    monkeypatch.setattr(drift, "build_live_stack", boom, raising=False)


def test_the_command_shows_the_estimate_and_runs_only_after_yes(tmp_path):
    small = golden(("kb", PASSWORD, "IT-PWD-01"), ("gap", "How do I register my tractor for parking?"))
    code, shown = _run_main(tmp_path, [], ["yes"], golden_set=small)
    assert code == 0 and "2 golden questions" in shown and "$0.00" in shown
    assert len(DriftHistory(tmp_path / "drift.jsonl").points()) == 1


def test_no_at_the_prompt_runs_nothing(tmp_path):
    code, shown = _run_main(tmp_path, [], ["no"], golden_set=golden(("kb", PASSWORD, "IT-PWD-01")))
    assert code == 0 and "Cancelled" in shown
    assert not (tmp_path / "drift.jsonl").exists()


def test_estimate_only_prints_and_never_asks(tmp_path):
    code, shown = _run_main(tmp_path, ["--estimate"], [], golden_set=golden(("kb", PASSWORD, "IT-PWD-01")))
    assert code == 0 and "1 golden questions" in shown and "Run it?" not in shown
    assert not (tmp_path / "drift.jsonl").exists()


def test_a_live_estimate_shows_dollars_and_builds_no_stack(tmp_path, live_config, no_live_stack):
    code, shown = _run_main(tmp_path, ["--estimate"], [], settings_path=live_config)
    assert code == 0 and "live" in shown.lower() and re.search(r"\$0\.[1-9]", shown)


def test_a_live_run_answered_no_builds_no_stack(tmp_path, live_config, no_live_stack):
    code, shown = _run_main(tmp_path, [], ["no"], settings_path=live_config)
    assert code == 0 and "Cancelled" in shown and not (tmp_path / "drift.jsonl").exists()


def test_yes_on_the_command_line_is_refused_for_live_runs(tmp_path, live_config, no_live_stack):
    code, shown = _run_main(tmp_path, ["--yes"], [], settings_path=live_config)
    assert code == 2 and "interactive" in shown


# --- the button in the Monitoring tab -----------------------------------------------------------------

def _tab(at):
    return at.main


def _button(at, label):
    return next(b for b in _tab(at).button if b.label == label)


class Runner:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return _point(ts=f"2026-10-1{self.calls}T12:00:00+00:00", faithfulness=0.9 + self.calls / 100)


def test_the_button_shows_the_estimate_and_runs_nothing_until_confirmed(tmp_path):
    runner = Runner()
    at = monitoring_app(mock_context(tmp_path), drift_runner=runner)
    _button(at, "Check for drift").click().run()
    shown = element_text(_tab(at))
    assert "74 golden questions" in shown and "$0.00" in shown and "mock" in shown.lower()
    assert runner.calls == 0 and not (tmp_path / "drift.jsonl").exists()
    _button(at, "Cancel").click().run()
    assert runner.calls == 0 and [b.label for b in _tab(at).button if b.label == "Confirm and run"] == []


def test_confirming_runs_once_stores_the_point_and_charts_it(tmp_path):
    runner = Runner()
    ctx = mock_context(tmp_path)
    at = monitoring_app(ctx, drift_runner=runner)
    _button(at, "Check for drift").click().run()
    _button(at, "Confirm and run").click().run()
    assert runner.calls == 1 and not at.exception
    assert len(DriftHistory(tmp_path / "drift.jsonl").points()) == 1
    assert len(charts(_tab(at))) == 2  # the rates, and the average rerank score on its own scale
    _button(at, "Check for drift").click().run()
    _button(at, "Confirm and run").click().run()
    assert len(load_monitoring(ctx.settings, ctx.data_dir, ctx.kb_dir, ctx.registry, results_dir=ctx.results_dir).drift) == 2
    rates, rerank = charts(_tab(at))
    assert len(rates) == 2 and set(rates.columns) == {"Faithfulness", "Recall@k", "Escalation rate"}
    assert list(rerank.columns) == ["Average rerank score"]
    assert rates["Faithfulness"].round(2).tolist() == [0.91, 0.92]  # the two runs, oldest first


def test_a_failed_run_stores_nothing_and_says_so_without_details(tmp_path):
    def failing():
        raise RuntimeError("boom with priya.k@smartdesk.example inside")

    at = monitoring_app(mock_context(tmp_path), drift_runner=failing)
    _button(at, "Check for drift").click().run()
    _button(at, "Confirm and run").click().run()
    shown = element_text(_tab(at)) + " ".join(str(e.value) for e in at.error)
    assert "did not finish" in shown and "RuntimeError" in shown
    assert "priya.k@smartdesk.example" not in shown and not (tmp_path / "drift.jsonl").exists()


def test_a_run_that_hit_the_ceiling_says_how_much_was_spent(tmp_path):
    def over():
        raise DriftCeilingExceeded(0.7, 0.5)

    at = monitoring_app(mock_context(tmp_path), drift_runner=over)
    _button(at, "Check for drift").click().run()
    _button(at, "Confirm and run").click().run()
    shown = element_text(_tab(at)) + " ".join(str(e.value) for e in at.error)
    assert "ceiling" in shown and "0.70" in shown


def test_the_chart_shows_only_points_of_the_current_mode(tmp_path):
    ctx = mock_context(tmp_path)
    DriftHistory(tmp_path / "drift.jsonl").append(_point(mode="live", ts="2026-10-17T00:00:00+00:00"))
    DriftHistory(tmp_path / "drift.jsonl").append(_point(mode="mock", ts="2026-10-18T00:00:00+00:00"))
    at = monitoring_app(ctx)
    assert len(charts(_tab(at))[0]) == 1
    assert "1 point" in element_text(_tab(at)) and "live" in element_text(_tab(at))


def test_run_details_list_versions_and_models(tmp_path):
    DriftHistory(tmp_path / "drift.jsonl").append(_point())
    shown = element_text(_tab(monitoring_app(mock_context(tmp_path))))
    assert "answer_v4" in shown or "answer v4" in shown


def test_without_history_the_tab_says_there_is_none(tmp_path):
    assert "No drift checks yet" in element_text(_tab(monitoring_app(mock_context(tmp_path))))


def test_the_default_runner_does_a_real_mock_run_end_to_end(tmp_path):
    ctx = mock_context(tmp_path)
    at = monitoring_app(ctx)
    _button(at, "Check for drift").click().run()
    _button(at, "Confirm and run").click().run()
    assert not at.exception
    (point,) = DriftHistory(tmp_path / "drift.jsonl").points()
    assert point.mode == "mock" and (point.n_kb, point.n_gap) == (53, 21)
    assert 0.0 <= point.recall_at_k <= 1.0 and 0.0 <= point.escalation_rate <= 1.0
