"""T0.7 spike: LangGraph interrupt + SqliteSaver + Pydantic state + Send.

Throwaway proof, not production code. It mirrors the shape of PLAN section 4.3:
plan -> Send fan-out (worker) -> retry_gate (join + retry loop) -> confirm
(the only interrupt) -> create / cancel, with edit looping back to confirm.

CLI, so a second process can resume the first one's thread:
  python -m spikes.langgraph_spike start  --db DB --thread T --log LOG
  python -m spikes.langgraph_spike resume --db DB --thread T --log LOG --answer yes
"""
import argparse
import json
import operator
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, Send, interrupt
from pydantic import BaseModel


class Part(BaseModel):
    part_id: str
    text: str
    retry: bool = False


class PartResult(BaseModel):
    answer: str
    answerable: bool
    retried: bool = False


class Draft(BaseModel):
    title: str
    description: str


def merge_by_key(left: dict | None, right: dict | None) -> dict:
    """Keyed merge: a later result for the same part overwrites, never duplicates."""
    return {**(left or {}), **(right or {})}


class SpikeState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = []
    parts: list[Part] = []
    part_results: Annotated[dict[str, PartResult], merge_by_key] = {}
    draft: Draft | None = None
    decision: str | None = None
    edit_text: str | None = None
    reply: str = ""
    steps: Annotated[list[str], operator.add] = []


def _log(config: RunnableConfig, event: str) -> None:
    """Append an event to the side-effect log, so tests can count node executions."""
    path = config["configurable"].get("log_path")
    if path:
        with open(path, "a") as handle:
            handle.write(event + "\n")


def plan(state: SpikeState) -> dict[str, Any]:
    parts = [
        Part(part_id="p1", text="vpn"),
        Part(part_id="p2", text="mfa"),
        Part(part_id="p3", text="unknown topic"),
    ]
    draft = Draft(title="VPN not connecting", description="Employee cannot connect to the VPN.")
    return {"parts": parts, "draft": draft, "steps": ["plan"]}


def fan_out(state: SpikeState) -> list[Send]:
    return [Send("worker", {"part": part}) for part in state.parts]


def worker(payload: dict[str, Any]) -> dict[str, Any]:
    part: Part = payload["part"]
    answerable = part.retry or part.part_id != "p3"  # p3 fails its first try
    result = PartResult(answer=f"answer to {part.text}", answerable=answerable, retried=part.retry)
    return {"part_results": {part.part_id: result}, "steps": ["worker"]}


def retry_gate(state: SpikeState) -> dict[str, Any]:
    return {"steps": ["retry_gate"]}


def after_gate(state: SpikeState) -> list[Send] | str:
    todo = [
        p
        for p in state.parts
        if not state.part_results[p.part_id].answerable and not state.part_results[p.part_id].retried
    ]
    if todo:
        return [Send("worker", {"part": p.model_copy(update={"retry": True})}) for p in todo]
    return "confirm"


def confirm(state: SpikeState, config: RunnableConfig) -> dict[str, Any]:
    _log(config, "confirm_top")  # runs again on resume: the node restarts from its top
    answer = interrupt(state.draft.model_dump())
    text = str(answer).strip()
    lowered = text.lower()
    if lowered == "yes":
        return {"decision": "yes", "steps": ["confirm"]}
    if lowered == "no":
        return {"decision": "no", "steps": ["confirm"]}
    if lowered.startswith("edit:"):
        return {"decision": "edit", "edit_text": text[5:].strip(), "steps": ["confirm"]}
    return {"decision": "unclear", "steps": ["confirm"]}


def after_confirm(state: SpikeState) -> str:
    return {"yes": "create", "edit": "apply_edit"}.get(state.decision or "", "cancel")


def apply_edit(state: SpikeState) -> dict[str, Any]:
    return {"draft": state.draft.model_copy(update={"title": state.edit_text}), "steps": ["apply_edit"]}


def create(state: SpikeState, config: RunnableConfig) -> dict[str, Any]:
    _log(config, "create")
    return {"reply": "created", "steps": ["create"]}


def cancel(state: SpikeState) -> dict[str, Any]:
    return {"reply": "cancelled", "steps": ["cancel"]}


def build_graph(checkpointer):
    builder = StateGraph(SpikeState)
    for name, fn in [
        ("plan", plan),
        ("worker", worker),
        ("retry_gate", retry_gate),
        ("confirm", confirm),
        ("apply_edit", apply_edit),
        ("create", create),
        ("cancel", cancel),
    ]:
        builder.add_node(name, fn)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", fan_out, ["worker"])
    builder.add_edge("worker", "retry_gate")
    builder.add_conditional_edges("retry_gate", after_gate, ["worker", "confirm"])
    builder.add_conditional_edges("confirm", after_confirm, ["create", "apply_edit", "cancel"])
    builder.add_edge("apply_edit", "confirm")
    builder.add_edge("create", END)
    builder.add_edge("cancel", END)
    return builder.compile(checkpointer=checkpointer)


# Pydantic types stored in checkpoints must be allow-listed, or LangGraph warns on every
# load ("Deserializing unregistered type ...") and will block them in a future version.
STATE_TYPES = [(__name__, cls.__name__) for cls in (Part, PartResult, Draft)]


@contextmanager
def open_checkpointer(db_path: Path | str) -> Iterator[SqliteSaver]:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        yield SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES))
    finally:
        conn.close()


def _config(thread_id: str, log_path: Path | str | None) -> dict:
    return {"configurable": {"thread_id": thread_id, "log_path": str(log_path) if log_path else None}}


def start(graph, thread_id: str, text: str, log_path: Path | str | None = None) -> dict:
    return graph.invoke({"messages": [HumanMessage(text)]}, _config(thread_id, log_path))


def resume(graph, thread_id: str, answer: str, log_path: Path | str | None = None) -> dict:
    return graph.invoke(Command(resume=answer), _config(thread_id, log_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "resume"])
    parser.add_argument("--db", required=True)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--log", default=None)
    parser.add_argument("--answer", default=None)
    args = parser.parse_args(argv)

    with open_checkpointer(args.db) as saver:
        graph = build_graph(saver)
        if args.action == "start":
            result = start(graph, args.thread, "my vpn is down", args.log)
        else:
            result = resume(graph, args.thread, args.answer, args.log)
        snapshot = graph.get_state({"configurable": {"thread_id": args.thread}})
    pending = result.get("__interrupt__")
    print(
        json.dumps(
            {
                "interrupted": bool(pending),
                "payload": pending[0].value if pending else None,
                "next": list(snapshot.next),
                "reply": snapshot.values.get("reply"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
