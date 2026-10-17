"""T4.10: escalation handoff (FR-A16). One merged message offers a ticket and asks for the email;
a known email makes it a plain yes/no; the email counts as agreeing; a refusal ends the flow."""
import pytest

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import AgentState, PendingOffer, new_thread_id
from smartdesk_ai.agent.ticket_agent import escalation_offer, handle_offer_reply
from tests.agent_helpers import build_test_graph
from tests.llm_fakes import FakeAnswerLLM

EMAIL = "priya.k@smartdesk.example"


def offered(email=None, awaiting="offer_reply"):
    return AgentState(email=email, awaiting=awaiting, pending_offer=PendingOffer(unanswered_text="parking?"))


# --- the merged offer ---------------------------------------------------------------------


def test_unknown_email_the_offer_and_the_email_request_are_one_message():
    text = escalation_offer(email_known=False)
    assert "don't know" in text.lower() and "ticket" in text.lower() and "email" in text.lower()


def test_known_email_the_offer_is_a_plain_yes_no_question_with_no_email_request():
    text = escalation_offer(email_known=True)
    assert "don't know" in text.lower() and "ticket" in text.lower()
    assert "email" not in text.lower() and "yes or no" in text.lower() and text.count("?") == 1


def test_the_graph_builds_the_offer_from_the_session_email(built_index_dir, settings, tmp_path):
    graph, _ = build_test_graph(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"))
    unknown = run_turn(graph, new_thread_id(), "Is there parking near the office?")
    known = run_turn(graph, new_thread_id(), "Is there parking near the office?", email=EMAIL)
    assert unknown.reply.endswith(escalation_offer(False)) and "email" in unknown.reply.lower()
    assert known.reply.endswith(escalation_offer(True)) and "email" not in known.reply.lower()
    assert known.email == EMAIL and known.awaiting == "offer_reply"


# --- the employee's reply -------------------------------------------------------------------


@pytest.mark.parametrize("text", [EMAIL, f"sure, it's {EMAIL}", f"My email is {EMAIL}.", f"<{EMAIL}> please"])
def test_giving_the_email_counts_as_agreeing(text):
    out = handle_offer_reply(offered(), text)
    assert out.kind == "agreed" and out.update["email"] == EMAIL
    assert out.update["awaiting"] == "none"
    assert "pending_offer" not in out.update  # the unanswered question is kept for the ticket


def test_the_email_is_never_echoed_back():
    assert EMAIL not in handle_offer_reply(offered(), EMAIL).reply


@pytest.mark.parametrize("text", ["yes", "Yes!", "yes please", "sure", "ok", "yep"])
def test_yes_with_a_known_email_agrees_without_asking_again(text):
    out = handle_offer_reply(offered(email=EMAIL), text)
    assert out.kind == "agreed" and "email" not in out.reply.lower()
    assert out.update == {"awaiting": "none"}


def test_yes_without_an_email_asks_for_it_once():
    out = handle_offer_reply(offered(), "yes")
    assert out.kind == "ask_email" and out.update == {"awaiting": "email"}
    assert "email" in out.reply.lower() and out.reply.count("?") == 1


def test_the_email_reply_after_the_ask_agrees():
    out = handle_offer_reply(offered(awaiting="email"), EMAIL)
    assert out.kind == "agreed" and out.update["email"] == EMAIL


@pytest.mark.parametrize("text", ["no", "No.", "no thanks", "No thank you", "nope", "not now", "never mind"])
def test_a_refusal_ends_the_flow_with_no_ticket(text):
    state = offered(email=EMAIL)
    out = handle_offer_reply(state, text)
    assert out.kind == "declined"
    assert out.update == {"awaiting": "none", "pending_offer": None, "ticket": None}
    assert "won't open a ticket" in out.reply  # closes the flow, does not offer again


def test_a_refusal_also_ends_an_email_request():
    assert handle_offer_reply(offered(awaiting="email"), "no thanks").kind == "declined"


@pytest.mark.parametrize("text", ["maybe", "what does that mean", "priya@", "how many sick days do I get?"])
def test_anything_else_is_left_for_the_router(text):
    out = handle_offer_reply(offered(), text)
    assert out.kind == "unclear" and out.update == {} and out.reply == ""


def test_nothing_pending_means_nothing_to_handle():
    assert handle_offer_reply(AgentState(), "yes").kind == "unclear"
