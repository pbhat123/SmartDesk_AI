"""The Ragas judge run is resumable, capped, observable and stoppable (FR-H12, FR-H8). No model is called."""
import asyncio
import json
import threading
import time
from types import SimpleNamespace

import httpx
import openai
import pytest

from evals import judge_run as jr
from evals import rag_metrics as rm
from evals.harness import make_resources
from evals.run import main
from evals.schemas import KBQuestion
from tests.test_rag_metrics import FakeMetric, Raises

pytestmark = pytest.mark.filterwarnings("ignore")


def outputs(n_answered=3, n_escalated=1):
    made = [rm.RagOutput(question_id=f"kb-{i:03d}", answer=f"answer {i}", insufficient=False, citations=["E"],
                         retrieved=["E"], contexts=[f"context {i}"]) for i in range(n_answered)]
    made += [rm.RagOutput(question_id=f"kb-esc{i}", answer="", insufficient=True, retrieved=["E"], contexts=["c"])
             for i in range(n_escalated)]
    return made


def inner_judge(**overrides):
    metrics = {name: FakeMetric(0.5) for name in ("faithfulness", "answer_correctness", "factual_precision",
                                                  "answer_relevancy", "context_precision", "context_recall")}
    metrics.update(overrides)
    return rm.RagasJudge(**metrics)


def wrap(inner, outs, settings, tmp_path, *, ceiling=None, counter=None, **kw):
    counter = counter or rm.UsageCounter(settings, ceiling)
    return jr.CheckpointedJudge(inner, total=jr.planned_calls(outs), signature=jr.outputs_signature(outs),
                                counter=counter, settings=settings, checkpoint=tmp_path / jr.CHECKPOINT_FILE,
                                progress_log=tmp_path / jr.PROGRESS_LOG, **kw), counter


def score(judge, outs, rows):
    return rm.compute_metrics(outs, rows, judge)


def fake_rows(outs):
    return [KBQuestion(id=o.question_id, question=f"question {o.question_id}?", expected_entry_id="E", expected_domain="IT",
                       type="simple", reference_answer="the reference answer", split="dev") for o in outs]


# --- planning -----------------------------------------------------------------------------------------

def test_planned_calls_counts_four_metrics_per_answer_and_two_per_context_set():
    assert jr.planned_calls(outputs(3, 1)) == 4 * 3 + 2 * 4


# --- progress log and checkpoint ---------------------------------------------------------------------------

def test_every_item_logs_progress_with_spend_and_a_checkpoint_is_saved_each_batch(settings, tmp_path):
    outs = outputs()
    judge, _ = wrap(inner_judge(), outs, settings, tmp_path, batch=5)
    with judge:
        score(judge, outs, fake_rows(outs))
    lines = (tmp_path / jr.PROGRESS_LOG).read_text().splitlines()
    total = jr.planned_calls(outs)
    assert len(lines) == total and f"[{total}/{total}]" in lines[-1] and "spend $" in lines[-1]
    saved = json.loads((tmp_path / jr.CHECKPOINT_FILE).read_text())
    assert len(saved["scores"]) == total and saved["total"] == total


def test_the_checkpoint_is_written_after_each_batch_not_only_at_the_end(settings, tmp_path):
    outs = outputs()
    judge, _ = wrap(inner_judge(), outs, settings, tmp_path, batch=4)
    sizes = []
    original = judge.flush

    def spy():
        original()
        sizes.append(len(json.loads((tmp_path / jr.CHECKPOINT_FILE).read_text())["scores"]))

    judge.flush = spy
    with judge:
        score(judge, outs, fake_rows(outs))
    assert sizes[:3] == [4, 8, 12]


# --- resume -------------------------------------------------------------------------------------------

def test_a_rerun_resumes_from_the_checkpoint_and_calls_the_metrics_only_for_the_missing_items(settings, tmp_path):
    outs = outputs()
    rows = fake_rows(outs)
    first = inner_judge(context_recall=Raises(jr.FatalJudgeError("no credits")))
    judge, _ = wrap(first, outs, settings, tmp_path)
    with pytest.raises(jr.FatalJudgeError):
        with judge:
            score(judge, outs, rows)
    done = len(json.loads((tmp_path / jr.CHECKPOINT_FILE).read_text())["scores"])
    assert done == 4 * 3 + 2 * 4 - 4  # everything but context_recall (the last metric) was scored and saved

    again = inner_judge()
    judge2, _ = wrap(again, outs, settings, tmp_path)
    with judge2:
        report = score(judge2, outs, rows)
    assert judge2.resumed == done
    assert len(again._metrics["faithfulness"].samples) == 0  # not re-judged
    assert len(again._metrics["context_recall"].samples) == 4  # only the missing ones
    assert report.context_recall == 0.5 and report.grounded_answer_rate == 0.5


def test_resumed_spend_counts_towards_the_ceiling(settings, tmp_path):
    outs = outputs()
    path = tmp_path / jr.CHECKPOINT_FILE
    path.write_text(json.dumps({"signature": jr.outputs_signature(outs), "scores": {}, "spend_usd": 0.9}))
    judge, counter = wrap(inner_judge(), outs, settings, tmp_path, ceiling=1.0)
    assert counter.base_usd == 0.9 and counter.cost_usd(settings) == pytest.approx(0.9)
    counter.base_usd = 1.1
    with pytest.raises(rm.SpendCeilingExceeded), judge:
        score(judge, outs, fake_rows(outs))


def test_a_checkpoint_for_different_answers_is_ignored(settings, tmp_path):
    outs = outputs()
    (tmp_path / jr.CHECKPOINT_FILE).write_text(json.dumps({"signature": "other", "scores": {"x": 1.0}, "spend_usd": 5}))
    judge, counter = wrap(inner_judge(), outs, settings, tmp_path)
    assert judge.scores == {} and counter.base_usd == 0.0
    assert "ignored" in (tmp_path / jr.PROGRESS_LOG).read_text()


def test_a_failed_sample_is_not_checkpointed_so_a_rerun_retries_it(settings, tmp_path):
    outs = outputs(2, 0)
    flaky = inner_judge(faithfulness=Raises(ValueError("bad json")))
    judge, _ = wrap(flaky, outs, settings, tmp_path)
    with judge:
        score(judge, outs, fake_rows(outs))
    saved = json.loads((tmp_path / jr.CHECKPOINT_FILE).read_text())["scores"]
    assert not any(k.startswith("faithfulness") for k in saved)
    assert "FAILED" in (tmp_path / jr.PROGRESS_LOG).read_text()


# --- the spend ceiling stops the run ---------------------------------------------------------------------------

def over_ceiling_response():
    message = SimpleNamespace(usage_metadata={"input_tokens": 10_000_000, "output_tokens": 0})
    return SimpleNamespace(generations=[[SimpleNamespace(message=message)]])


def test_a_ceiling_hit_in_a_worker_thread_still_stops_the_run_even_if_the_exception_is_swallowed(settings, tmp_path):
    outs = outputs()
    counter = rm.UsageCounter(settings, ceiling_usd=0.5)

    class Threaded:
        """Its metric runs the model callback on another thread and swallows whatever it raises."""

        def single_turn_score(self, sample):
            def work():
                try:
                    counter.on_llm_end(over_ceiling_response())
                except Exception:
                    pass  # swallowed, as an executor or a retry wrapper might

            thread = threading.Thread(target=work)
            thread.start()
            thread.join()
            return 0.5

    inner = inner_judge(faithfulness=Threaded())
    judge, _ = wrap(inner, outs, settings, tmp_path, counter=counter, ceiling=0.5)
    with pytest.raises(rm.SpendCeilingExceeded):
        with judge:
            score(judge, outs, fake_rows(outs))
    assert counter.exceeded
    assert len(inner._metrics["answer_correctness"].samples) == 0  # nothing was judged after the breach
    assert (tmp_path / jr.CHECKPOINT_FILE).exists()  # and what was done is saved


def test_the_ceiling_exception_passes_through_the_ragas_judge_and_the_wrapper(settings, tmp_path):
    outs = outputs()
    inner = inner_judge(faithfulness=Raises(rm.SpendCeilingExceeded("over")))
    judge, _ = wrap(inner, outs, settings, tmp_path)
    with pytest.raises(rm.SpendCeilingExceeded), judge:
        score(judge, outs, fake_rows(outs))


# --- fatal errors stop at once -------------------------------------------------------------------------------

def quota_error():
    response = httpx.Response(429, request=httpx.Request("POST", "http://x"))
    return openai.RateLimitError("Error code: 429 - insufficient_quota credit_balance_exhausted", response=response, body=None)


def test_a_no_credits_error_stops_the_run_on_the_first_call_not_the_third():
    judge = inner_judge(faithfulness=Raises(quota_error()))
    with pytest.raises(jr.FatalJudgeError, match="insufficient_quota"):
        judge.faithfulness("q", "a", ["c"])
    assert judge.errors == {}


def test_ordinary_errors_are_not_fatal():
    assert not jr.is_fatal(ValueError("bad json")) and jr.is_fatal(quota_error())


# --- per-call timeout and capped retries ------------------------------------------------------------------------

def test_the_real_judge_runs_with_a_60s_timeout_and_at_most_three_attempts(monkeypatch):
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    run_config = rm.judge_run_config()
    assert (run_config.timeout, run_config.max_retries) == (60, 3) and run_config.max_wait <= 10
    chat = ChatOpenAI(model="m", api_key="x", timeout=rm.JUDGE_TIMEOUT_S, max_retries=0)
    wrapper = LangchainLLMWrapper(chat, run_config=run_config)
    rm.apply_retry_policy(run_config)
    assert chat.request_timeout == 60 and chat.max_retries == 0  # no hidden SDK retries on top
    assert issubclass(openai.APITimeoutError, run_config.exception_types)  # a timeout is retried, within the cap
    assert issubclass(openai.RateLimitError, run_config.exception_types)
    assert wrapper.run_config is run_config


def test_a_failing_call_is_attempted_three_times_and_no_more():
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    calls = []

    class Failing(ChatOpenAI):
        async def agenerate_prompt(self, *args, **kwargs):
            calls.append(1)
            raise quota_error()

    run_config = rm.judge_run_config()
    run_config.max_wait = 0
    wrapper = LangchainLLMWrapper(Failing(model="m", api_key="x", max_retries=0), run_config=run_config)
    rm.apply_retry_policy(run_config)
    with pytest.raises(openai.RateLimitError):
        asyncio.run(wrapper.generate(SimpleNamespace(to_messages=lambda: [], to_string=lambda: "p")))
    assert len(calls) == 3


def test_a_timed_out_sample_is_a_failed_sample_not_a_hang_and_three_in_a_row_stop_the_run():
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "http://x"))
    judge = inner_judge(faithfulness=Raises(timeout))
    assert judge.faithfulness("q", "a", ["c"]) is None and judge.errors == {"faithfulness": 1}
    judge.faithfulness("q", "a", ["c"])
    with pytest.raises(RuntimeError, match="faithfulness failed 3 times"):
        judge.faithfulness("q", "a", ["c"])


# --- the watchdog ---------------------------------------------------------------------------------------------

def test_the_watchdog_fires_when_nothing_happens_for_the_stall_time():
    fired = threading.Event()
    dog = jr.Watchdog(0.15, fired.set, poll_seconds=0.02)
    dog.start()
    assert fired.wait(2) and dog.stalled
    dog.stop()


def test_the_watchdog_stays_quiet_while_there_is_progress():
    fired = threading.Event()
    dog = jr.Watchdog(0.3, fired.set, poll_seconds=0.02)
    dog.start()
    for _ in range(10):
        time.sleep(0.05)
        dog.beat()
    dog.stop()
    assert not fired.is_set()


def test_a_stall_saves_the_checkpoint_logs_it_and_aborts(settings, tmp_path):
    outs = outputs()
    aborted = threading.Event()

    class Hangs:
        def single_turn_score(self, sample):
            aborted.wait(5)  # a call that never returns, until the watchdog's abort
            return 0.5

    judge, _ = wrap(inner_judge(faithfulness=FakeMetric(1.0), answer_correctness=Hangs()), outs, settings, tmp_path,
                    stall_seconds=0.2, abort=aborted.set, poll_seconds=0.02)
    with judge:
        score(judge, outs, fake_rows(outs))
    assert aborted.is_set()
    assert "WATCHDOG" in (tmp_path / jr.PROGRESS_LOG).read_text()
    assert len(json.loads((tmp_path / jr.CHECKPOINT_FILE).read_text())["scores"]) >= 3


# --- judge-only mode --------------------------------------------------------------------------------------------

def test_judge_only_needs_the_saved_outputs(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")  # a fresh clone has no key; the run stops on the missing outputs first
    assert main(["--only", "rag", "--judge", "ragas", "--judge-only", "--confirm-spend", "--out-dir", str(tmp_path)]) == 1
    assert "rag_outputs.json" in capsys.readouterr().err


def test_judge_only_without_ragas_or_an_out_dir_is_refused(capsys):
    assert main(["--only", "rag", "--judge-only", "--mock"]) == 1
    assert "--judge-only needs" in capsys.readouterr().err


def test_judge_only_skips_the_pipeline_and_passes_the_ceiling_and_stall_time(monkeypatch, tmp_path):
    from evals import run as run_module

    seen = {}
    monkeypatch.setattr(run_module.rag_metrics, "run_judge_only", lambda res, out_dir, **kw: seen.update(kw, out=out_dir))
    monkeypatch.setattr(run_module.rag_metrics, "run_real", lambda *a, **k: pytest.fail("the pipeline must not run"))
    monkeypatch.setattr(run_module, "make_resources", lambda **kw: make_resources(mock=True))
    monkeypatch.setattr(run_module, "get_index", lambda *a: pytest.fail("no index is needed"))
    assert main(["--only", "rag", "--judge", "ragas", "--judge-only", "--confirm-spend", "--ceiling-usd", "0.9",
                 "--stall-seconds", "120", "--out-dir", str(tmp_path)]) == 0
    assert seen == {"ceiling_usd": 0.9, "stall_seconds": 120.0, "out": tmp_path}


def test_judge_only_reads_the_saved_outputs_and_resumes(monkeypatch, settings, tmp_path, capsys):
    from evals.harness import kb_rows

    rows = kb_rows(None)
    outs = [rm.RagOutput(question_id=r.id, answer=f"a {r.id}", insufficient=False, citations=["E"], retrieved=["E"],
                         contexts=["c"]) for r in rows[:4]]
    (tmp_path / rm.OUTPUTS_FILE).write_text(json.dumps([o.model_dump() for o in outs]))
    counter = rm.UsageCounter(settings)
    monkeypatch.setattr(rm, "build_real_judge", lambda s, ceiling_usd=None: (inner_judge(), counter))
    monkeypatch.setattr(rm, "kb_rows", lambda split: rows[:4])
    res = make_resources(mock=True)
    report = rm.run_judge_only(res, tmp_path, ceiling_usd=1.0)
    assert report.context_recall == 0.5 and (tmp_path / "rag_metrics.json").exists()
    assert "judge spend:" in capsys.readouterr().out
    saved = json.loads((tmp_path / jr.CHECKPOINT_FILE).read_text())
    assert len(saved["scores"]) == jr.planned_calls(outs)
    again = inner_judge()
    monkeypatch.setattr(rm, "build_real_judge", lambda s, ceiling_usd=None: (again, rm.UsageCounter(settings)))
    rm.run_judge_only(res, tmp_path)
    assert len(again._metrics["faithfulness"].samples) == 0
