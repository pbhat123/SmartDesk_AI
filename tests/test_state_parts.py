"""T7.1: the multi-agent state fields (FR-I3, FR-I4): sub-questions, keyed part results, per-part traces."""
from langgraph.graph import END, START, StateGraph

from smartdesk_ai.agent.state import STATE_MODELS, AgentState, PartResult, PendingOffer, SubQuestion, thread_config
from smartdesk_ai.memory.checkpointer import make_checkpointer


def part(pid="p1", **kw):
    return PartResult(part_id=pid, sub_question="How do I reset my password?", agent="it_worker", answer="A\n\nSources: X",
                      body="A", citations=["IT-PWD-01"], answerable=True, **kw)


def test_new_models_are_registered_for_the_checkpoint_serializer():
    assert SubQuestion in STATE_MODELS and PartResult in STATE_MODELS


def test_the_new_fields_default_to_a_turn_with_no_parts():
    s = AgentState()
    assert s.parts == [] and s.part_results == {} and s.part_steps == {}
    assert PendingOffer(unanswered_text="x").unanswered_parts == []


def test_a_sub_question_has_a_part_id_text_and_one_of_three_domains():
    q = SubQuestion(part_id="p1", text="How many sick days?", domain="HR")
    assert (q.part_id, q.domain) == ("p1", "HR")


def two_writers(state):
    return {"part_results": {"a": part("a")}, "part_steps": {"a": ["search_kb", "answer"]}}


def other_writer(state):
    return {"part_results": {"b": part("b")}, "part_steps": {"b": ["search_kb", "answer"]}}


def reset(state):
    return {"part_results": None, "part_steps": None}


def test_parallel_writers_merge_by_key_and_none_resets(tmp_path):
    """Two nodes in one superstep both write part_results: no conflict, both kept; a later None clears them."""
    builder = StateGraph(AgentState)
    for name, fn in (("a", two_writers), ("b", other_writer), ("r", reset)):
        builder.add_node(name, fn)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge(["a", "b"], "r")
    builder.add_edge("r", END)
    saver = make_checkpointer(tmp_path / "c.db", STATE_MODELS)
    graph = builder.compile(checkpointer=saver)
    graph.invoke({}, thread_config("t1"))
    assert graph.get_state(thread_config("t1")).values["part_results"] == {}


def test_keyed_merge_overwrites_a_part_for_the_retry(tmp_path):
    builder = StateGraph(AgentState)
    builder.add_node("first", lambda s: {"part_results": {"a": part("a")}})
    builder.add_node("retry", lambda s: {"part_results": {"a": part("a", retried=True)}})
    builder.add_edge(START, "first")
    builder.add_edge("first", "retry")
    builder.add_edge("retry", END)
    graph = builder.compile(checkpointer=make_checkpointer(tmp_path / "c.db", STATE_MODELS))
    graph.invoke({}, thread_config("t2"))
    results = graph.get_state(thread_config("t2")).values["part_results"]
    assert list(results) == ["a"] and results["a"].retried is True


def test_part_results_survive_the_sqlite_checkpoint(tmp_path):
    db = tmp_path / "c.db"
    builder = StateGraph(AgentState)
    builder.add_node("w", two_writers)
    builder.add_edge(START, "w")
    builder.add_edge("w", END)
    builder.compile(checkpointer=make_checkpointer(db, STATE_MODELS)).invoke({}, thread_config("t3"))
    again = builder.compile(checkpointer=make_checkpointer(db, STATE_MODELS))
    assert again.get_state(thread_config("t3")).values["part_results"]["a"].body == "A"
