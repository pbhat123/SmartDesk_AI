"""T4.22: the status trigger phrases route to ticket_status (FR-C5), through the intent set and
into the status flow of the graph. Mock models, no network."""
import pytest

from evals.common import EVALS_DIR, load_jsonl
from evals.schemas import IntentRow
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.intents import ClassifierError
from smartdesk_ai.agent.router import ToolPick, make_router, route_message
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.llm.mock import MockRoutingModel
from tests.agent_helpers import assemble

PHRASES = [
    "What is the status of my ticket?",
    "Did anyone look at my VPN issue?",
    "Any updates on my monitor ticket?",
]
EMAIL = "one.ticket@smartdesk.example"


@pytest.fixture(scope="module")
def route(settings):
    return make_router(settings, builder=lambda model_id, cfg: MockRoutingModel())


@pytest.mark.parametrize("phrase", PHRASES)
def test_the_example_phrases_route_to_ticket_status(route, phrase):
    decision = route(phrase, "none", None)
    assert decision.intent == "ticket_status" and decision.route == "ticket_agent"
    assert decision.source == "classifier" and not decision.clarify and not decision.deflect


def test_the_intent_set_status_rows_route_to_ticket_status(route):
    rows = [IntentRow(**r) for r in load_jsonl(EVALS_DIR / "intents.jsonl")]
    status_rows = [r for r in rows if r.expected_intent == "ticket_status"]
    assert len(status_rows) >= 4 and any(r.awaiting == "ticket_pick" for r in status_rows)
    for r in status_rows:
        decision = route(r.utterance, r.awaiting, None)
        assert (decision.intent, decision.route) == ("ticket_status", "ticket_agent"), r.id


@pytest.mark.parametrize("text", [
    "Please raise a ticket for my broken VPN",
    "How do I reset my forgotten password?",
    "Hi there",
    "I need some help",
])
def test_other_messages_do_not_route_to_status(route, text):
    assert route(text, "none", None).intent != "ticket_status"


def test_the_tool_pick_fallback_maps_get_tickets_to_the_status_route():
    def broken(text, awaiting="none", last_topic=None):
        raise ClassifierError("forced")

    decision = route_message(PHRASES[0], "none", None, broken,
                             lambda text, awaiting="none": ToolPick(tool="get_tickets"), 0.6)
    assert (decision.intent, decision.route, decision.source) == ("ticket_status", "ticket_agent", "tool_pick")


@pytest.mark.parametrize("phrase", PHRASES)
def test_the_graph_sends_the_phrases_to_the_status_flow(built_index_dir, settings, tmp_path, phrase):
    stack = assemble(built_index_dir, settings, tmp_path, routed=True)
    state = run_turn(stack.graph, new_thread_id(), phrase, email=EMAIL)
    assert state.route_kind == "status" and state.intent.intent == "ticket_status"
    assert "s_lookup" in state.steps
    assert stack.spy.calls == [], "a status question is not a KB search"
    assert state.pending_offer is None and state.ticket is None


def test_a_status_question_while_an_offer_is_pending_clears_the_offer(built_index_dir, settings, tmp_path):
    from tests.llm_fakes import FakeAnswerLLM

    stack = assemble(built_index_dir, settings, tmp_path, routed=True, llm=FakeAnswerLLM("insufficient"))
    tid = new_thread_id()
    run_turn(stack.graph, tid, "My monitor keeps flickering", email=EMAIL)
    state = run_turn(stack.graph, tid, PHRASES[0])
    assert state.route_kind == "status" and state.pending_offer is None and state.awaiting == "none"
