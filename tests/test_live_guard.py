"""The common guard of the live eval runs (CLAUDE.md working rules, FR-H7, FR-H8): a ledger run, a spend ceiling of twice the
estimate, a stall watchdog and a resume checkpoint. No model is called: spend is added to the ledger run by hand."""
import json
import threading
import time
from types import SimpleNamespace

import pytest

from evals import live_guard as lg
from smartdesk_ai.ops import spend_ledger as sl


@pytest.fixture()
def ledger(tmp_path):
    return tmp_path / "ledger.jsonl"


def usage(tokens=1000):
    return sl.Usage(1, tokens, 0, 0, 0)


def events(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


# --- the ceiling ------------------------------------------------------------------------------------------------

def test_the_ceiling_is_twice_the_estimate():
    assert lg.ceiling_for(0.37) == pytest.approx(0.74)
    assert lg.ceiling_for(0.0) == 0.0 and lg.CEILING_MULTIPLIER == 2.0


def test_a_run_records_a_ledger_run_as_an_eval_and_ends_it(settings, ledger):
    with lg.LiveRun(settings, ["python", "-m", "evals.x"], ceiling_usd=1.0, ledger_path=ledger, on_stall=lambda: None) as run:
        sl.current_run().add_chat(usage(), settings.models.main.id)
        assert run.cost().calls == 1 and run.cost().usd > 0
    first, *_, last = events(ledger)
    assert first["event"] == "start" and first["source"] == "eval" and last["event"] == "end" and last["status"] == "ok"


def test_stop_reason_is_none_until_the_ledger_passes_the_ceiling(settings, ledger):
    with lg.LiveRun(settings, ["x"], ceiling_usd=0.001, ledger_path=ledger, on_stall=lambda: None) as run:
        assert run.stop_reason() is None
        sl.current_run().add_chat(usage(10_000), settings.models.main.id)
        reason = run.stop_reason()
        assert reason and "ceiling" in reason and "$0.00" in reason
        assert run.status == "ceiling"


def test_a_ceiling_stop_is_recorded_in_the_ledger_status(settings, ledger):
    with lg.LiveRun(settings, ["x"], ceiling_usd=0.0001, ledger_path=ledger, on_stall=lambda: None) as run:
        sl.current_run().add_chat(usage(10_000), settings.models.main.id)
        run.stop_reason()
    assert events(ledger)[-1]["status"] == "ceiling"


def test_an_exception_leaves_the_status_crashed(settings, ledger):
    with pytest.raises(RuntimeError):
        with lg.LiveRun(settings, ["x"], ceiling_usd=1.0, ledger_path=ledger, on_stall=lambda: None):
            raise RuntimeError("boom")
    assert events(ledger)[-1]["status"] == "crashed"


def test_the_ceiling_counts_only_this_runs_spend_not_earlier_runs(settings, ledger):
    with lg.LiveRun(settings, ["a"], ceiling_usd=1.0, ledger_path=ledger, on_stall=lambda: None):
        sl.current_run().add_chat(usage(100_000), settings.models.main.id)
    with lg.LiveRun(settings, ["b"], ceiling_usd=0.5, ledger_path=ledger, on_stall=lambda: None) as run:
        assert run.cost().usd == 0.0 and run.stop_reason() is None


# --- the watchdog ----------------------------------------------------------------------------------------------

def test_a_run_with_no_progress_is_stopped_by_the_watchdog(settings, ledger):
    stalled = threading.Event()
    with lg.LiveRun(settings, ["x"], ceiling_usd=1.0, ledger_path=ledger, stall_seconds=0.05, poll_seconds=0.01,
                    on_stall=stalled.set):
        assert stalled.wait(2)


def test_beats_keep_the_watchdog_quiet(settings, ledger):
    stalled = threading.Event()
    with lg.LiveRun(settings, ["x"], ceiling_usd=1.0, ledger_path=ledger, stall_seconds=0.3, poll_seconds=0.01,
                    on_stall=stalled.set) as run:
        for _ in range(8):
            time.sleep(0.1)
            run.beat()
    assert not stalled.is_set()


def test_a_model_call_counts_as_progress(settings, ledger):
    stalled = threading.Event()
    with lg.LiveRun(settings, ["x"], ceiling_usd=1.0, ledger_path=ledger, stall_seconds=0.3, poll_seconds=0.01,
                    on_stall=stalled.set):
        for _ in range(8):
            time.sleep(0.1)
            sl.current_run().add_chat(usage(10), settings.models.main.id)
    assert not stalled.is_set()


def test_the_default_stall_action_ends_the_ledger_run_and_exits(monkeypatch, settings, ledger):
    exits = []
    monkeypatch.setattr(lg.os, "_exit", exits.append)
    with lg.LiveRun(settings, ["x"], ceiling_usd=1.0, ledger_path=ledger, stall_seconds=0.05, poll_seconds=0.01):
        deadline = time.time() + 2
        while not exits and time.time() < deadline:
            time.sleep(0.01)
    assert exits == [lg.EXIT_STALLED] and "stalled" in [e.get("status") for e in events(ledger)]


# --- the checkpoint --------------------------------------------------------------------------------------------

def test_a_checkpoint_remembers_finished_items_across_processes(tmp_path):
    path = tmp_path / "cp.json"
    first = lg.Checkpoint(path, "sig")
    first.put("q1", {"outcome": "answered"})
    first.put("q2", {"outcome": "escalated"})
    again = lg.Checkpoint(path, "sig")
    assert again.done("q1") and again.get("q2") == {"outcome": "escalated"} and not again.done("q3")
    assert again.resumed == 2 and len(again) == 2


def test_a_checkpoint_for_other_inputs_is_ignored(tmp_path):
    path = tmp_path / "cp.json"
    lg.Checkpoint(path, "old").put("q1", {})
    assert not lg.Checkpoint(path, "new").done("q1")


def test_a_damaged_checkpoint_starts_empty(tmp_path):
    path = tmp_path / "cp.json"
    path.write_text("{not json")
    assert len(lg.Checkpoint(path, "sig")) == 0


def test_a_checkpoint_write_is_atomic(tmp_path):
    path = tmp_path / "cp.json"
    cp = lg.Checkpoint(path, "sig")
    cp.put("q1", {"a": 1})
    assert not list(tmp_path.glob("*.tmp")) and json.loads(path.read_text())["items"] == {"q1": {"a": 1}}


def test_a_checkpoint_without_a_path_is_memory_only():
    cp = lg.Checkpoint(None, "sig")
    cp.put("q1", {})
    assert cp.done("q1")


def test_signature_depends_on_the_content():
    assert lg.signature(["a", "b"]) == lg.signature(["a", "b"]) != lg.signature(["a", "c"])


# --- running items ---------------------------------------------------------------------------------------------

def test_run_items_skips_finished_ones_and_checkpoints_each(settings, ledger, tmp_path):
    cp = lg.Checkpoint(tmp_path / "cp.json", "s")
    cp.put("a", {"v": 1})
    seen = []
    with lg.LiveRun(settings, ["x"], ceiling_usd=1.0, ledger_path=ledger, on_stall=lambda: None) as run:
        done, stopped = lg.run_items(run, cp, ["a", "b", "c"], lambda i: seen.append(i) or {"v": i})
    assert seen == ["b", "c"] and stopped is None and done == {"a": {"v": 1}, "b": {"v": "b"}, "c": {"v": "c"}}
    assert lg.Checkpoint(tmp_path / "cp.json", "s").done("c")


def test_run_items_stops_before_the_next_item_over_the_ceiling(settings, ledger, tmp_path):
    cp = lg.Checkpoint(tmp_path / "cp.json", "s")
    seen = []

    def work(i):
        seen.append(i)
        sl.current_run().add_chat(usage(100_000), settings.models.main.id)
        return {}

    with lg.LiveRun(settings, ["x"], ceiling_usd=0.0001, ledger_path=ledger, on_stall=lambda: None) as run:
        done, stopped = lg.run_items(run, cp, ["a", "b", "c"], work)
    assert seen == ["a"] and "ceiling" in stopped and set(done) == {"a"}


def test_run_items_stops_when_the_item_says_so(settings, ledger, tmp_path):
    cp = lg.Checkpoint(None, "s")
    with lg.LiveRun(settings, ["x"], ceiling_usd=1.0, ledger_path=ledger, on_stall=lambda: None) as run:
        done, stopped = lg.run_items(run, cp, ["a", "b"], lambda i: {}, stop=lambda results: "enough" if "a" in results else None)
    assert set(done) == {"a"} and stopped == "enough"
