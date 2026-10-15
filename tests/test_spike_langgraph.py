"""T0.7 spike: LangGraph interrupt + SqliteSaver + Pydantic state + Send (FR-G12, H10, I4).

No network and no LLM: nodes are plain functions. The spike graph is a slice of the
PLAN section 4.3 design: plan -> Send fan-out to workers -> retry_gate (join, retry
loop) -> confirm (the only interrupt) -> create or cancel.
"""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from spikes.langgraph_spike import (
    Draft,
    PartResult,
    build_graph,
    open_checkpointer,
    resume,
    start,
)

ROOT = Path(__file__).resolve().parent.parent


def read_log(path: Path) -> list[str]:
    return path.read_text().split() if path.exists() else []


@pytest.fixture()
def env(tmp_path):
    return {"db": tmp_path / "spike.db", "log": tmp_path / "calls.log"}


def run_module(*args: str) -> dict:
    """Run the spike CLI in a fresh Python process and parse its JSON output."""
    result = subprocess.run(
        # `-c` (not `-m`) so state classes are tagged spikes.langgraph_spike.X, not __main__.X
        [sys.executable, "-c", "import sys; from spikes.langgraph_spike import main; sys.exit(main())", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT)},
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


# --- Send fan-out joins -----------------------------------------------------

def test_parallel_send_branches_join_before_the_gate_continues(env):
    with open_checkpointer(env["db"]) as saver:
        graph = build_graph(saver)
        start(graph, "t1", "question", env["log"])
        steps = graph.get_state({"configurable": {"thread_id": "t1"}}).values["steps"]
    first_gate = steps.index("retry_gate")
    # all three first-round workers finished before the gate ran
    assert steps[:first_gate].count("worker") == 3
    assert steps[first_gate + 1 :].count("worker") == 1  # the retry
    assert steps.count("retry_gate") == 2
    # the run stopped inside confirm: its own step is not recorded until it finishes
    assert steps[-1] == "retry_gate" and "confirm" not in steps


def test_retry_overwrites_by_key_and_never_duplicates(env):
    with open_checkpointer(env["db"]) as saver:
        graph = build_graph(saver)
        start(graph, "t1", "question", env["log"])
        results = graph.get_state({"configurable": {"thread_id": "t1"}}).values["part_results"]
    assert set(results) == {"p1", "p2", "p3"}
    assert all(isinstance(r, PartResult) for r in results.values())
    assert results["p3"].retried is True and results["p3"].answerable is True
    assert results["p1"].retried is False


# --- interrupt --------------------------------------------------------------

def test_graph_pauses_at_the_interrupt_before_any_side_effect(env):
    with open_checkpointer(env["db"]) as saver:
        graph = build_graph(saver)
        result = start(graph, "t1", "question", env["log"])
        snapshot = graph.get_state({"configurable": {"thread_id": "t1"}})
    assert "__interrupt__" in result
    assert snapshot.next == ("confirm",)
    assert "create" not in read_log(env["log"])


def test_threads_in_one_database_are_independent(env):
    with open_checkpointer(env["db"]) as saver:
        graph = build_graph(saver)
        start(graph, "a", "question", env["log"])
        other = graph.get_state({"configurable": {"thread_id": "b"}})
    assert other.values == {} and other.next == ()


# --- resume in a new process ------------------------------------------------

def test_resume_in_a_new_process_with_the_same_thread_id_creates_once(env):
    first = run_module("start", "--db", str(env["db"]), "--thread", "t1", "--log", str(env["log"]))
    assert first["interrupted"] is True and first["next"] == ["confirm"]
    assert "create" not in read_log(env["log"])

    second = run_module("resume", "--db", str(env["db"]), "--thread", "t1", "--log", str(env["log"]), "--answer", "yes")
    assert second["interrupted"] is False
    assert second["reply"] == "created"
    assert read_log(env["log"]).count("create") == 1


def test_answer_no_cancels_and_creates_nothing(env):
    run_module("start", "--db", str(env["db"]), "--thread", "t1", "--log", str(env["log"]))
    out = run_module("resume", "--db", str(env["db"]), "--thread", "t1", "--log", str(env["log"]), "--answer", "no")
    assert out["reply"] == "cancelled"
    assert "create" not in read_log(env["log"])


def test_resumed_thread_cannot_be_resumed_again(env):
    run_module("start", "--db", str(env["db"]), "--thread", "t1", "--log", str(env["log"]))
    run_module("resume", "--db", str(env["db"]), "--thread", "t1", "--log", str(env["log"]), "--answer", "yes")
    with open_checkpointer(env["db"]) as saver:
        graph = build_graph(saver)
        assert graph.get_state({"configurable": {"thread_id": "t1"}}).next == ()
    assert read_log(env["log"]).count("create") == 1


def test_node_reruns_from_its_top_on_resume(env):
    """Gotcha behind PLAN 4.4: code before interrupt() runs again after resume."""
    run_module("start", "--db", str(env["db"]), "--thread", "t1", "--log", str(env["log"]))
    run_module("resume", "--db", str(env["db"]), "--thread", "t1", "--log", str(env["log"]), "--answer", "yes")
    assert read_log(env["log"]).count("confirm_top") == 2


# --- Pydantic state through the checkpointer -------------------------------

def test_pydantic_state_survives_a_process_restart_as_models(env):
    run_module("start", "--db", str(env["db"]), "--thread", "t1", "--log", str(env["log"]))
    with open_checkpointer(env["db"]) as saver:  # a different connection than the one that wrote it
        graph = build_graph(saver)
        values = graph.get_state({"configurable": {"thread_id": "t1"}}).values
    assert isinstance(values["draft"], Draft)
    assert values["draft"].title == "VPN not connecting"
    assert all(isinstance(r, PartResult) for r in values["part_results"].values())


def test_registered_state_types_load_without_serializer_warnings(env, caplog):
    run_module("start", "--db", str(env["db"]), "--thread", "t1", "--log", str(env["log"]))
    with caplog.at_level(logging.WARNING):
        with open_checkpointer(env["db"]) as saver:
            build_graph(saver).get_state({"configurable": {"thread_id": "t1"}})
    assert "unregistered type" not in caplog.text


def test_edit_reply_is_passed_through_the_resume_value(env):
    run_module("start", "--db", str(env["db"]), "--thread", "t1", "--log", str(env["log"]))
    out = run_module("resume", "--db", str(env["db"]), "--thread", "t1", "--log", str(env["log"]), "--answer", "edit: Laptop will not boot")
    assert out["interrupted"] is True  # edit loops back to the confirm gate
    assert out["payload"]["title"] == "Laptop will not boot"
    done = run_module("resume", "--db", str(env["db"]), "--thread", "t1", "--log", str(env["log"]), "--answer", "yes")
    assert done["reply"] == "created"
    assert read_log(env["log"]).count("create") == 1


def test_configurable_values_are_persisted_in_checkpoint_metadata(env):
    """Gotcha: anything put in config["configurable"] lands in the SQLite file.

    So only thread_id may go there. Emails and other PII stay out (CLAUDE.md, FR-G11).
    """
    import sqlite3

    marker = env["log"].with_name("CONFIG-MARKER.log")
    with open_checkpointer(env["db"]) as saver:
        start(build_graph(saver), "t1", "question", marker)
    rows = sqlite3.connect(env["db"]).execute("select metadata from checkpoints").fetchall()
    assert any(b"CONFIG-MARKER" in bytes(row[0]) for row in rows)
