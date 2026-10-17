"""T4.1: SQLite checkpointer keeps session state and the pending draft across a restart
(FR-H10, FR-G12). No network, no LLM; a tiny graph stands in for the agent."""
import logging
import sqlite3
from typing import Annotated

import pytest
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from smartdesk_ai.memory.checkpointer import make_checkpointer


class Draft(BaseModel):
    title: str


class MiniState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = []
    draft: Draft | None = None
    reply: str = ""


def _draft(state: MiniState) -> dict:
    return {"draft": Draft(title="VPN down")}


def _confirm(state: MiniState) -> dict:
    answer = interrupt(state.draft.model_dump())
    return {"reply": "created" if str(answer).lower() == "yes" else "cancelled"}


def build(saver):
    builder = StateGraph(MiniState)
    builder.add_node("draft", _draft)
    builder.add_node("confirm", _confirm)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "confirm")
    builder.add_edge("confirm", END)
    return builder.compile(checkpointer=saver)


def cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def open_saver(path):
    return make_checkpointer(path, state_types=[Draft])


def test_pending_draft_and_history_survive_a_restart(tmp_path):
    db = tmp_path / "chat.db"
    first = open_saver(db)
    result = build(first).invoke({"messages": [HumanMessage("my vpn is down")]}, cfg("t1"))
    assert result["__interrupt__"][0].value == {"title": "VPN down"}
    first.conn.close()  # the process ends here

    second = open_saver(db)  # a new process opens the same file
    graph = build(second)
    snapshot = graph.get_state(cfg("t1"))
    assert snapshot.next == ("confirm",)
    assert snapshot.values["draft"] == Draft(title="VPN down")
    assert [m.content for m in snapshot.values["messages"]] == ["my vpn is down"]
    done = graph.invoke(Command(resume="yes"), cfg("t1"))
    assert done["reply"] == "created"
    assert graph.get_state(cfg("t1")).next == ()


def test_threads_in_one_database_do_not_mix(tmp_path):
    saver = open_saver(tmp_path / "chat.db")
    graph = build(saver)
    graph.invoke({"messages": [HumanMessage("first")]}, cfg("a"))
    graph.invoke({"messages": [HumanMessage("second")]}, cfg("b"))
    assert [m.content for m in graph.get_state(cfg("a")).values["messages"]] == ["first"]
    assert [m.content for m in graph.get_state(cfg("b")).values["messages"]] == ["second"]
    assert graph.get_state(cfg("never-used")).values == {}


def test_creates_missing_parent_directory(tmp_path):
    saver = open_saver(tmp_path / "data" / "nested" / "chat.db")
    assert (tmp_path / "data" / "nested" / "chat.db").exists()
    saver.conn.close()


def test_registered_state_types_load_without_a_warning(tmp_path, caplog):
    db = tmp_path / "chat.db"
    build(open_saver(db)).invoke({"messages": [HumanMessage("hi")]}, cfg("t"))
    with caplog.at_level(logging.WARNING):
        build(open_saver(db)).get_state(cfg("t"))
    assert "unregistered type" not in caplog.text


def test_checkpoint_metadata_holds_no_pii(tmp_path):
    db = tmp_path / "chat.db"
    build(open_saver(db)).invoke({"messages": [HumanMessage("hi")]}, cfg("t"))
    with sqlite3.connect(db) as conn:
        metadata = [bytes(row[0]).decode() for row in conn.execute("select metadata from checkpoints")]
    assert metadata
    assert not any("@" in m for m in metadata)


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_path_is_rejected(bad):
    with pytest.raises(ValueError):
        make_checkpointer(bad)
