"""T4.2: Pydantic graph state, thread ID per session (FR-D2, FR-G12)."""
import inspect
import logging

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ValidationError

from smartdesk_ai.agent import state as state_module
from smartdesk_ai.agent.state import (
    STATE_MODELS,
    AgentState,
    IntentResult,
    PendingOffer,
    TicketDraft,
    new_thread_id,
    thread_config,
)
from smartdesk_ai.memory.checkpointer import make_checkpointer


def fill(state: AgentState) -> dict:
    """Stand-in node that writes every field the agent keeps between turns."""
    return {
        "email": "priya.k@smartdesk.example",
        "last_topic": "vpn",
        "awaiting": "offer_reply",
        "intent": IntentResult(intent="kb_query", confidence=0.9, domain="IT"),
        "pending_offer": PendingOffer(unanswered_text="vpn token expired"),
        "ticket": TicketDraft(title="VPN token", description="Token expired.", category="IT"),
        "reply": "I could not find that. Shall I open a ticket?",
    }


def build(saver):
    builder = StateGraph(AgentState)
    builder.add_node("fill", fill)
    builder.add_edge(START, "fill")
    builder.add_edge("fill", END)
    return builder.compile(checkpointer=saver)


def test_defaults_are_an_empty_session():
    s = AgentState()
    assert s.messages == [] and s.email is None and s.last_topic is None
    assert s.awaiting == "none" and s.intent is None
    assert s.pending_offer is None and s.ticket is None and s.reply == ""


def test_awaiting_accepts_only_the_four_dialog_states():
    for value in ("none", "offer_reply", "email", "ticket_pick"):
        assert AgentState(awaiting=value).awaiting == value
    with pytest.raises(ValidationError):
        AgentState(awaiting="maybe")


def test_ticket_draft_priority_and_category_are_closed_sets():
    assert TicketDraft(title="t", description="d", category="MISC", priority="critical").priority == "critical"
    with pytest.raises(ValidationError):
        TicketDraft(title="t", description="d", category="FACILITIES")
    with pytest.raises(ValidationError):
        TicketDraft(title="t", description="d", category="IT", priority="urgent")


def test_intent_confidence_is_between_zero_and_one():
    with pytest.raises(ValidationError):
        IntentResult(intent="kb_query", confidence=1.5)
    with pytest.raises(ValidationError):
        IntentResult(intent="chat", confidence=0.5)


def test_state_round_trips_through_the_sqlite_checkpointer(tmp_path):
    db = tmp_path / "chat.db"
    tid = new_thread_id()
    first = make_checkpointer(db, STATE_MODELS)
    build(first).invoke({"messages": [HumanMessage("vpn broken")]}, thread_config(tid))
    first.conn.close()

    values = build(make_checkpointer(db, STATE_MODELS)).get_state(thread_config(tid)).values
    restored = AgentState(**values)
    assert restored.email == "priya.k@smartdesk.example"
    assert restored.awaiting == "offer_reply"
    assert restored.intent == IntentResult(intent="kb_query", confidence=0.9, domain="IT")
    assert restored.pending_offer == PendingOffer(unanswered_text="vpn token expired")
    assert restored.ticket.title == "VPN token"
    assert [m.content for m in restored.messages] == ["vpn broken"]


def test_two_thread_ids_never_mix(tmp_path):
    graph = build(make_checkpointer(tmp_path / "chat.db", STATE_MODELS))
    a, b = new_thread_id(), new_thread_id()
    graph.invoke({"messages": [HumanMessage("from a")]}, thread_config(a))
    graph.invoke({"messages": [HumanMessage("from b")], "email": None}, thread_config(b))
    assert [m.content for m in graph.get_state(thread_config(a)).values["messages"]] == ["from a"]
    assert [m.content for m in graph.get_state(thread_config(b)).values["messages"]] == ["from b"]
    assert graph.get_state(thread_config(new_thread_id())).values == {}


def test_thread_ids_are_unique_and_opaque():
    ids = {new_thread_id() for _ in range(200)}
    assert len(ids) == 200
    assert all("@" not in i and len(i) >= 16 for i in ids)


def test_thread_config_carries_only_the_thread_id():
    assert thread_config("abc") == {"configurable": {"thread_id": "abc"}}
    for bad in ("", "  ", "priya.k@smartdesk.example"):
        with pytest.raises(ValueError):
            thread_config(bad)


def test_every_state_model_is_registered_with_the_serializer():
    defined = {
        cls
        for _, cls in inspect.getmembers(state_module, inspect.isclass)
        if issubclass(cls, BaseModel) and cls.__module__ == state_module.__name__
    }
    assert defined == set(STATE_MODELS)


def test_checkpoint_load_logs_no_unregistered_type_warning(tmp_path, caplog):
    db = tmp_path / "chat.db"
    tid = new_thread_id()
    build(make_checkpointer(db, STATE_MODELS)).invoke({"messages": []}, thread_config(tid))
    with caplog.at_level(logging.WARNING):
        build(make_checkpointer(db, STATE_MODELS)).get_state(thread_config(tid))
    assert "unregistered type" not in caplog.text
