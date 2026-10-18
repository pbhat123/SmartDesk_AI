"""The persistent spend ledger (FR-H7, FR-H8, FR-G11): one fixed file, append-only, numbers only, crash-safe,
and priced from config.yaml. No model is called."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from smartdesk_ai.ops import spend_ledger as sl
from smartdesk_ai.ops.budget import BudgetController, SpendStore
from smartdesk_ai.rag.ingest import ROOT


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


@pytest.fixture()
def ledger(tmp_path):
    return tmp_path / "ledger.jsonl"


def result(prompt=1000, completion=200, cached=0, reasoning=0, n=1, model="m"):
    """What LangChain hands a callback: with n completions every generation repeats the call's usage."""
    meta = {"input_tokens": prompt, "output_tokens": completion,
            "input_token_details": {"cache_read": cached}, "output_token_details": {"reasoning": reasoning}}
    gen = SimpleNamespace(message=SimpleNamespace(usage_metadata=meta))
    usage = {"prompt_tokens": prompt, "completion_tokens": completion,
             "prompt_tokens_details": {"cached_tokens": cached},
             "completion_tokens_details": {"reasoning_tokens": reasoning}}
    return SimpleNamespace(generations=[[gen] * n], llm_output={"token_usage": usage, "model_name": model})


# --- where it is written ---------------------------------------------------------------------------

def test_the_ledger_is_one_fixed_file_in_the_repo_with_an_env_override():
    assert sl.DEFAULT_LEDGER == ROOT / "data" / "spend_ledger.jsonl"
    assert sl.ledger_path({}) == sl.DEFAULT_LEDGER
    assert sl.ledger_path({sl.LEDGER_ENV: "/x/y.jsonl"}) == Path("/x/y.jsonl")


def test_an_eval_stack_in_a_temp_data_dir_still_writes_to_the_fixed_ledger(settings, tmp_path, monkeypatch):
    from smartdesk_ai.agent.factory import build_stack

    fixed = tmp_path / "fixed" / "ledger.jsonl"
    monkeypatch.setenv(sl.LEDGER_ENV, str(fixed))
    live = settings.model_copy(update={"modes": settings.modes.model_copy(update={"llm": "live"})})
    with pytest.raises(Exception):  # live needs a key and an index; the ledger run starts before that matters
        build_stack(live, environ={"OPENAI_API_KEY": "sk-test-key-123456"}, dotenv_path=None,
                    data_dir=tmp_path / "temp_data_dir_of_an_eval")
    events = lines(fixed)
    assert events[0]["event"] == "start" and events[0]["source"] == "app"
    assert not (tmp_path / "temp_data_dir_of_an_eval" / "spend_ledger.jsonl").exists()


def test_a_second_start_in_the_same_process_returns_the_active_run(settings, ledger):
    first = sl.start_run(settings, "cmd", "eval", ledger)
    assert sl.start_run(settings, "other", "app", ledger) is first


def test_without_a_started_run_nothing_is_written_and_mock_mode_never_starts_one(settings, tmp_path, monkeypatch):
    from smartdesk_ai.agent.factory import build_stack

    fixed = tmp_path / "ledger.jsonl"
    monkeypatch.setenv(sl.LEDGER_ENV, str(fixed))
    sl.record_chat(sl.Usage(1, 10, 0, 10))
    sl.LedgeredEmbeddings(SimpleNamespace(embed_query=lambda t: [0.0]), "emb").embed_query("hello")
    build_stack(settings, environ={}, dotenv_path=None, data_dir=tmp_path / "d")  # config.yaml is mock mode
    assert not fixed.exists()


def test_the_repo_ledger_is_not_touched_by_this_test_suite():
    before = sl.DEFAULT_LEDGER.stat().st_mtime_ns if sl.DEFAULT_LEDGER.exists() else None
    sl.reset_for_tests()
    assert (sl.DEFAULT_LEDGER.stat().st_mtime_ns if sl.DEFAULT_LEDGER.exists() else None) == before
    assert Path(sl.ledger_path()) != sl.DEFAULT_LEDGER  # conftest points every test at a temp file


# --- content: numbers only, masked command ---------------------------------------------------------------

def test_lines_hold_numbers_and_names_only_and_the_command_is_masked(settings, ledger):
    run = sl.start_run(settings, f"python -m evals.run --user a.b@corp.com --key sk-abcdef123456 --out {Path.home()}/x",
                       "eval", ledger)
    run.add_chat(sl.usage_from_result(result()), "m")
    run.finish("ok")
    text = ledger.read_text()
    assert "a.b@corp.com" not in text and "sk-abcdef123456" not in text and str(Path.home()) not in text
    assert "[email]" in text and "[key]" in text and "~/x" in text
    allowed = {"ts", "run_id", "source", "event", "command", "pid", "kind", "model", "ok", "calls", "input_tokens",
               "cached_tokens", "output_tokens", "reasoning_tokens", "usd", "status", "price_assumed"}
    assert all(set(e) <= allowed for e in lines(ledger))


def test_the_ledger_is_append_only_across_runs(settings, ledger):
    sl.start_run(settings, "one", "eval", ledger).finish("ok")
    sl.reset_for_tests()
    sl.start_run(settings, "two", "eval", ledger).finish("ok")
    events = lines(ledger)
    assert [e["event"] for e in events] == ["start", "end", "start", "end"]
    assert events[0]["run_id"] != events[2]["run_id"]


# --- cost accuracy -----------------------------------------------------------------------------------

def test_reasoning_tokens_are_billed_once_as_output(settings, ledger):
    run = sl.start_run(settings, "c", "eval", ledger)
    usage = sl.usage_from_result(result(prompt=1000, completion=500, reasoning=300, model=settings.models.main.id))
    assert (usage.output_tokens, usage.reasoning_tokens) == (500, 300)  # completion_tokens already includes them
    price = settings.models.main.price
    assert run.add_chat(usage) == pytest.approx((1000 * price.input + 500 * price.output) / 1e6)
    spend = [e for e in lines(ledger) if e["event"] == "spend"][0]
    assert spend["output_tokens"] == 500 and spend["reasoning_tokens"] == 300


def test_cached_input_is_priced_at_the_cached_rate(settings, ledger):
    run = sl.start_run(settings, "c", "eval", ledger)
    usd = run.add_chat(sl.usage_from_result(result(prompt=1000, completion=0, cached=600, model=settings.models.main.id)))
    price = settings.models.main.price
    assert usd == pytest.approx((400 * price.input + 600 * price.cached_input) / 1e6)


def test_a_call_with_several_completions_is_counted_once_not_n_times():
    usage = sl.usage_from_result(result(prompt=1000, completion=200, n=3))
    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (1, 1000, 200)
    no_output = SimpleNamespace(generations=result(n=3).generations, llm_output=None)  # fall back to the first generation
    assert sl.usage_from_result(no_output).input_tokens == 1000


def test_each_model_is_priced_from_config_and_an_unknown_one_is_flagged_not_free(settings, ledger):
    run = sl.start_run(settings, "c", "eval", ledger)
    m = settings.models
    light = run.add_chat(sl.Usage(1, 1_000_000, 0, 0), m.light.id)
    dated = run.add_chat(sl.Usage(1, 1_000_000, 0, 0), m.main.id + "-2026-01-01")
    unknown = run.add_chat(sl.Usage(1, 1_000_000, 0, 0), "mystery")
    emb = run.add_embedding(1_000_000, m.embedding.id)
    assert (light, dated, unknown, emb) == pytest.approx((m.light.price.input, m.main.price.input,
                                                          m.main.price.input, m.embedding.price.input))
    flagged = [e for e in lines(ledger) if e.get("price_assumed")]
    assert len(flagged) == 1 and flagged[0]["model"] == "mystery"


def test_the_chat_callback_and_the_embedding_wrapper_report_real_calls(settings, ledger):
    sl.start_run(settings, "c", "eval", ledger)
    callback = sl.make_callback(settings.models.main.id)
    callback.on_llm_end(result(model=settings.models.main.id))
    callback.on_llm_error(RuntimeError("boom"))
    inner = SimpleNamespace(embed_query=lambda t: [0.0], embed_documents=lambda ts: [[0.0]] * len(ts), name="emb")
    wrapped = sl.LedgeredEmbeddings(inner, settings.models.embedding.id)
    wrapped.embed_documents(["alpha beta", "gamma"])
    assert wrapped.name == "emb"  # embedder_id still sees the wrapped embedder
    spend = [e for e in lines(ledger) if e["event"] == "spend"]
    assert [(e["kind"], e["ok"]) for e in spend] == [("chat", True), ("chat", False), ("embedding", True)]
    assert spend[1]["usd"] == 0 and spend[2]["input_tokens"] > 0


def test_the_real_chat_builder_attaches_the_ledger_callback(settings, monkeypatch):
    from smartdesk_ai.llm.factory import openai_builder

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123456")
    model = openai_builder(settings.models.main.id, settings.models.main)
    assert any(type(cb).__name__ == "LedgerCallback" for cb in model.callbacks)


def test_the_judge_counter_prices_cached_tokens_and_counts_n_completions_once(settings):
    from evals import rag_metrics as rm

    counter = rm.UsageCounter(settings)
    counter.on_llm_end(result(prompt=1000, completion=200, cached=400, n=3))
    price = settings.models.main.price
    assert counter.calls == 1
    assert counter.cost_usd(settings) == pytest.approx((600 * price.input + 400 * price.cached_input + 200 * price.output) / 1e6)


# --- crash safety ------------------------------------------------------------------------------------

def test_spend_lines_survive_a_run_that_never_finishes(settings, ledger):
    run = sl.start_run(settings, "c", "eval", ledger)
    run.add_chat(sl.usage_from_result(result()), "m")
    summary = sl.summarize(lines(ledger))[run.run_id]  # no end line: as after kill -9
    assert summary["status"] == "running or killed" and summary["calls"] == 1 and summary["usd"] > 0


def test_an_exception_ends_the_run_as_crashed_in_the_eval_entry_point(settings, ledger, monkeypatch):
    from evals import run as run_module
    from evals.harness import make_resources

    monkeypatch.setenv(sl.LEDGER_ENV, str(ledger))
    monkeypatch.setattr(run_module, "make_resources", lambda **kw: make_resources(mock=True))
    monkeypatch.setattr(run_module, "get_index", lambda *a: (_ for _ in ()).throw(KeyError("boom")))
    with pytest.raises(KeyError):
        run_module.main(["--only", "rag", "--judge", "ragas", "--confirm-spend", "--out-dir", str(ledger.parent)])
    events = lines(ledger)
    assert events[0]["event"] == "start" and events[-1] == {**events[-1], "event": "end", "status": "crashed"}


def test_a_process_that_dies_without_ending_the_run_is_closed_by_atexit(settings, ledger):
    code = ("import sys; from smartdesk_ai.config.settings import load_settings; from smartdesk_ai.ops import spend_ledger as s;"
            "r = s.start_run(load_settings(), 'x', 'eval', __import__('pathlib').Path(sys.argv[1])); raise SystemExit(0)")
    subprocess.run([sys.executable, "-c", code, str(ledger)], check=True, cwd=ROOT)
    assert lines(ledger)[-1]["event"] == "end" and "exited without end_run" in lines(ledger)[-1]["status"]


def test_the_watchdog_abort_ends_the_run_as_stalled_before_exiting(settings, ledger, monkeypatch):
    from evals import judge_run as jr

    sl.start_run(settings, "c", "eval", ledger)
    monkeypatch.setattr(jr.os, "_exit", lambda code: None)
    jr.CheckpointedJudge._abort_process()
    assert lines(ledger)[-1]["status"] == "stalled"


def test_the_report_totals_spend_per_run_and_day(settings, ledger, capsys):
    run = sl.start_run(settings, "python -m evals.run", "eval", ledger)
    run.add_chat(sl.Usage(1, 1_000_000, 0, 0), settings.models.main.id)
    run.finish("ok")
    sl.main(["--path", str(ledger)])
    out = capsys.readouterr().out
    assert "$0.7500" in out and "total" in out


# --- the daily budget reads persisted spend, not process memory ---------------------------------------------

def test_the_daily_cutoff_reads_the_database_on_every_call_not_a_total_held_in_memory(settings, tmp_path):
    path = tmp_path / "ops.db"
    controller = BudgetController(settings, SpendStore(path))
    assert controller.level() == 0
    SpendStore(path).add(controller._today(), settings.budget.cutoff_1_usd + 0.5)  # another process, another connection
    assert controller.level() == 1
    SpendStore(path).add(controller._today(), settings.budget.cutoff_2_usd)
    assert controller.level() == 2
    assert BudgetController(settings, SpendStore(path)).level() == 2  # and a restarted process sees it too


def test_the_ragas_judge_model_and_the_real_embedder_both_report_to_the_ledger(settings, monkeypatch):
    from smartdesk_ai.config import env as env_module
    from smartdesk_ai.rag.embeddings import get_embeddings

    fake_env = env_module.load_env(environ={"OPENAI_API_KEY": "sk-test-key-123456"}, dotenv_path=None)
    monkeypatch.setattr(env_module, "load_env", lambda *a, **k: fake_env)
    from evals import rag_metrics as rm

    judge, counter = rm.build_real_judge(settings, ceiling_usd=1.0)
    chat = judge._metrics["faithfulness"].llm.langchain_llm
    names = {type(cb).__name__ for cb in chat.callbacks}
    assert "LedgerCallback" in names and any(isinstance(cb, rm.UsageCounter) for cb in chat.callbacks)
    assert (chat.max_retries, chat.request_timeout) == (0, 60)
    assert isinstance(get_embeddings(settings, fake_env), sl.LedgeredEmbeddings)
