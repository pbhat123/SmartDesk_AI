"""T4.28: multi-turn conversation (FR-D6, FR-D2). A follow-up is answered with the earlier topic in
mind, and the email, once given, is remembered for the whole session. Mock models, no network."""
import re

import pytest

from smartdesk_ai.agent.confirm import make_confirm_parser
from smartdesk_ai.agent.followup import contextual_question, is_follow_up
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.llm.mock import MockConfirmModel, MockDraftModel
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM

EMAIL = "one.ticket@smartdesk.example"
ASKS_FOR_EMAIL = re.compile(r"(share|what'?s|what is|need)\b.*\bemail", re.I)


def build(built_index_dir, settings, tmp_path, modes, seeded=False):
    parser = make_confirm_parser(settings, builder=lambda m, c: MockConfirmModel())
    return assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM(modes), routed=True,
                    drafter_llm=MockDraftModel(), parse_confirm=parser, seeded=seeded)


# --- follow-ups -----------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["what about sick leave?", "What about for contractors?", "and sick leave?",
                                  "How about the VPN?", "Also for interns?", "what if I'm part time?"])
def test_a_leading_follow_up_phrase_is_recognised(text):
    assert is_follow_up(text)


@pytest.mark.parametrize("text", ["How do I reset my password?", "Is there parking near the office?",
                                  "Anyone know what about this means", "My monitor keeps flickering"])
def test_a_fresh_question_is_not_a_follow_up(text):
    assert not is_follow_up(text)


def test_the_contextual_question_carries_both_the_follow_up_and_the_topic():
    text = contextual_question("what about sick leave?", "How many days of annual leave do I get?")
    assert "what about sick leave?" in text and "annual leave" in text


def test_without_a_topic_or_follow_up_the_question_is_unchanged():
    assert contextual_question("what about sick leave?", None) == "what about sick leave?"
    assert contextual_question("How do I reset my password?", "annual leave") == "How do I reset my password?"


def test_a_follow_up_is_searched_and_answered_with_the_prior_topic(built_index_dir, settings, tmp_path):
    llm = FakeAnswerLLM("cite_top")
    stack = assemble(built_index_dir, settings, tmp_path, llm=llm, routed=True)
    tid = new_thread_id()
    first = run_turn(stack.graph, tid, "How many days of annual leave do I get?")
    assert first.last_topic == "How many days of annual leave do I get?"
    second = run_turn(stack.graph, tid, "what about sick leave?")
    query = stack.spy.calls[-1][0]
    assert "sick leave" in query and "annual leave" in query
    assert "annual leave" in llm.calls[-1][-1].content  # the answer model sees the topic too
    assert second.last_topic == first.last_topic  # the topic stays until a new question replaces it
    assert second.steps[:3] == ["start", "route", "search_kb"]


def test_a_new_question_replaces_the_topic_and_is_searched_as_asked(built_index_dir, settings, tmp_path):
    stack = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("cite_top"), routed=True)
    tid = new_thread_id()
    run_turn(stack.graph, tid, "How many days of annual leave do I get?")
    state = run_turn(stack.graph, tid, "How do I reset my password?")
    assert stack.spy.calls[-1][0] == "How do I reset my password?"
    assert state.last_topic == "How do I reset my password?"


def test_a_follow_up_chain_keeps_resolving_against_the_first_topic(built_index_dir, settings, tmp_path):
    stack = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("cite_top"), routed=True)
    tid = new_thread_id()
    for text in ["How many days of annual leave do I get?", "what about sick leave?", "and parental leave?"]:
        state = run_turn(stack.graph, tid, text)
    assert "parental leave" in stack.spy.calls[-1][0] and "annual leave" in stack.spy.calls[-1][0]
    assert stack.spy.calls[-1][0].count("sick leave") == 0
    assert state.last_topic == "How many days of annual leave do I get?"


def test_a_follow_up_with_no_earlier_topic_is_searched_as_asked(built_index_dir, settings, tmp_path):
    stack = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("cite_top"), routed=True)
    run_turn(stack.graph, new_thread_id(), "what about sick leave?")
    assert stack.spy.calls[-1][0] == "what about sick leave?"


def test_the_topic_survives_other_turns_such_as_thanks(built_index_dir, settings, tmp_path):
    stack = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("cite_top"), routed=True)
    tid = new_thread_id()
    run_turn(stack.graph, tid, "How many days of annual leave do I get?")
    run_turn(stack.graph, tid, "Thanks!")
    run_turn(stack.graph, tid, "what about sick leave?")
    assert "annual leave" in stack.spy.calls[-1][0]


# --- the email is remembered for the whole session ------------------------------------------


def assert_never_asked_again(states):
    for state in states:
        assert state.awaiting != "email" and not state.status_pending_email, state.steps
        assert not ASKS_FOR_EMAIL.search(state.reply), state.reply


def test_an_email_given_for_a_status_check_is_not_asked_again_for_a_ticket_or_a_status(
        built_index_dir, settings, tmp_path):
    stack = build(built_index_dir, settings, tmp_path, ["cite_top", "insufficient"], seeded=True)
    tid = new_thread_id()
    first = run_turn(stack.graph, tid, "What's the status of my ticket?")
    assert first.awaiting == "email"  # asked once
    run_turn(stack.graph, tid, EMAIL)
    states = [
        run_turn(stack.graph, tid, "How do I reset my password?"),
        run_turn(stack.graph, tid, "My monitor keeps flickering"),  # a KB gap: a plain yes or no offer
    ]
    assert states[-1].awaiting == "offer_reply"
    states.append(run_turn(stack.graph, tid, "yes"))
    states.append(run_turn(stack.graph, tid, "yes"))  # the approval gate
    states.append(run_turn(stack.graph, tid, "Any update on my ticket?"))
    assert_never_asked_again(states)
    assert states[-1].email == EMAIL
    assert any("has been created" in s.reply for s in states)


def test_an_email_given_for_a_ticket_is_not_asked_again_for_a_status_check(built_index_dir, settings, tmp_path):
    stack = build(built_index_dir, settings, tmp_path, "insufficient")
    tid = new_thread_id()
    offer = run_turn(stack.graph, tid, "My monitor keeps flickering")
    assert offer.awaiting == "email"  # asked once
    states = [run_turn(stack.graph, tid, "priya.k@smartdesk.example")]
    states.append(run_turn(stack.graph, tid, "no"))  # the gate: cancel
    states.append(run_turn(stack.graph, tid, "What's the status of my ticket?"))
    states.append(run_turn(stack.graph, tid, "Is there parking near the office?"))
    states.append(run_turn(stack.graph, tid, "Please raise a ticket for my broken VPN"))
    assert_never_asked_again(states)
    assert states[-1].email == "priya.k@smartdesk.example" and "s_lookup" in states[2].steps


def test_a_declined_offer_keeps_the_email(built_index_dir, settings, tmp_path):
    stack = build(built_index_dir, settings, tmp_path, "insufficient")
    tid = new_thread_id()
    run_turn(stack.graph, tid, "Is there parking near the office?", email=EMAIL)
    declined = run_turn(stack.graph, tid, "no thanks")
    later = run_turn(stack.graph, tid, "Please raise a ticket for my broken VPN")
    assert declined.email == EMAIL and later.email == EMAIL
    assert_never_asked_again([declined, later])
