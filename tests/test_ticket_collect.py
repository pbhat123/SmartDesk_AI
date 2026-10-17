"""T4.12: ticket flow, collect the email, the summary and the description. A known email is not
re-asked, a missing one is asked, an invalid one is re-asked (FR-B1, FR-B5, FR-C1)."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import AgentState, PendingOffer, new_thread_id
from smartdesk_ai.agent.ticket_agent import collect_step, employee_statements, handle_offer_reply
from tests.agent_helpers import build_test_graph
from tests.llm_fakes import FakeAnswerLLM

EMAIL = "priya.k@smartdesk.example"
ISSUE = "My monitor keeps flickering"


def state(email=None, awaiting="none", messages=(ISSUE,), pending=True, **extra):
    return AgentState(
        email=email, awaiting=awaiting,
        messages=[HumanMessage(m) for m in messages],
        pending_offer=PendingOffer(unanswered_text=ISSUE) if pending else None,
        **extra,
    )


# --- collect_step -------------------------------------------------------------------------


def test_a_known_email_is_not_asked_again():
    out = collect_step(state(email=EMAIL))
    assert out.kind == "ready" and out.reply == "" and out.update == {}


def test_a_missing_email_is_asked_explicitly():
    out = collect_step(state())
    assert out.kind == "need_email" and "email" in out.reply.lower()
    assert out.update["awaiting"] == "email" and out.update["pending_offer"] is not None


def test_a_direct_request_keeps_a_pending_flow_marker_so_the_email_reply_is_understood():
    out = collect_step(state(pending=False))
    assert out.update["pending_offer"].unanswered_text == ISSUE


def test_an_email_inside_an_earlier_message_is_never_guessed():
    s = state(messages=("My colleague priya.k@smartdesk.example is affected too", ISSUE))
    assert collect_step(s).kind == "need_email"


def test_no_issue_text_asks_what_the_ticket_is_about():
    out = collect_step(state(email=EMAIL, messages=("Please raise a ticket",), pending=False))
    assert out.kind == "need_issue" and out.reply.count("?") == 1
    assert out.update == {}


@pytest.mark.parametrize("text", ["I want to raise a ticket", "Can you open a support request for me?", "I need help"])
def test_a_bare_ticket_request_has_no_issue_in_it(text):
    assert employee_statements(state(messages=(text,), pending=False)) == []


def test_a_request_that_names_the_problem_has_an_issue():
    assert employee_statements(state(messages=("Please raise a ticket for my broken VPN",), pending=False))


def test_the_collect_replies_never_ask_about_priority():
    bare = state(email=EMAIL, messages=("Please raise a ticket",), pending=False)
    for out in (collect_step(state()), collect_step(bare)):
        assert "priority" not in out.reply.lower()


# --- the reply to an email request --------------------------------------------------------


@pytest.mark.parametrize("text", ["priya@", "priya.k@smartdesk", "@smartdesk.example", "priya k@smartdesk"])
def test_an_invalid_email_is_asked_again(text):
    out = handle_offer_reply(state(awaiting="email"), text)
    assert out.kind == "invalid_email" and "email" in out.reply.lower() and out.update == {}
    assert text not in out.reply  # the bad input is not echoed back


def test_a_valid_email_after_an_invalid_one_is_accepted():
    s = state(awaiting="email")
    assert handle_offer_reply(s, "priya@").kind == "invalid_email"
    assert handle_offer_reply(s, EMAIL).update["email"] == EMAIL


def test_a_non_email_reply_during_an_email_request_is_left_for_the_router():
    assert handle_offer_reply(state(awaiting="email"), "how many sick days do I get?").kind == "unclear"


# --- statements -----------------------------------------------------------------------------


def test_statements_skip_yes_no_and_email_only_replies():
    s = state(messages=(ISSUE, "yes please", EMAIL, "It started after the update"), pending=False)
    assert employee_statements(s) == [ISSUE, "It started after the update"]


def test_statements_start_at_the_ticket_context_and_skip_assistant_text():
    s = AgentState(
        messages=[HumanMessage("old question"), AIMessage("old answer"), HumanMessage(ISSUE)],
        ticket_context_start=2,
    )
    assert employee_statements(s) == [ISSUE]


def test_emails_are_removed_from_statements():
    s = state(messages=(f"Contact me on {EMAIL} about the monitor",), pending=False)
    assert EMAIL not in employee_statements(s)[0] and "monitor" in employee_statements(s)[0]


# --- through the graph --------------------------------------------------------------------


def gap_graph(built_index_dir, settings, tmp_path):
    graph, _ = build_test_graph(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"))
    return graph


def test_yes_with_the_email_unknown_asks_for_it_once(built_index_dir, settings, tmp_path):
    graph, tid = gap_graph(built_index_dir, settings, tmp_path), new_thread_id()
    run_turn(graph, tid, ISSUE)
    asked = run_turn(graph, tid, "yes")
    assert asked.awaiting == "email" and "email" in asked.reply.lower()


def test_invalid_then_valid_email_through_the_graph(built_index_dir, settings, tmp_path):
    graph, tid = gap_graph(built_index_dir, settings, tmp_path), new_thread_id()
    run_turn(graph, tid, ISSUE)
    run_turn(graph, tid, "yes")
    again = run_turn(graph, tid, "priya@")
    assert again.awaiting == "email" and again.email is None and "valid" in again.reply.lower()
    done = run_turn(graph, tid, EMAIL)
    assert done.email == EMAIL and done.awaiting == "none" and "t_collect" in done.steps


def test_giving_the_email_straight_after_the_offer_counts_as_agreeing(built_index_dir, settings, tmp_path):
    graph, tid = gap_graph(built_index_dir, settings, tmp_path), new_thread_id()
    run_turn(graph, tid, ISSUE)
    done = run_turn(graph, tid, EMAIL)
    assert done.email == EMAIL and "t_collect" in done.steps and "email" not in done.reply.lower().split("?")[0]


def test_a_known_session_email_is_not_asked_through_the_graph(built_index_dir, settings, tmp_path):
    graph, tid = gap_graph(built_index_dir, settings, tmp_path), new_thread_id()
    first = run_turn(graph, tid, ISSUE, email=EMAIL)
    assert "email" not in first.reply.lower()
    done = run_turn(graph, tid, "yes")
    assert "t_collect" in done.steps and done.awaiting == "none"
    assert not any(isinstance(m, AIMessage) and "email address" in m.content.lower() for m in done.messages)


def test_a_refusal_through_the_graph_ends_the_flow(built_index_dir, settings, tmp_path):
    graph, tid = gap_graph(built_index_dir, settings, tmp_path), new_thread_id()
    run_turn(graph, tid, ISSUE)
    done = run_turn(graph, tid, "no thanks")
    assert done.pending_offer is None and done.awaiting == "none" and done.ticket is None
    assert "t_collect" not in done.steps


def test_an_unrelated_reply_to_the_offer_is_handled_as_a_new_question(built_index_dir, settings, tmp_path):
    graph, tid = gap_graph(built_index_dir, settings, tmp_path), new_thread_id()
    run_turn(graph, tid, ISSUE)
    other = run_turn(graph, tid, "How do I reset my forgotten password?")
    assert "search_kb" in other.steps and "t_collect" not in other.steps


# --- the direct request path (routing in the graph) ----------------------------------------


def routed(built_index_dir, settings, tmp_path):
    graph, _ = build_test_graph(built_index_dir, settings, tmp_path, routed=True)
    return graph


def test_a_direct_ticket_request_asks_for_the_missing_email(built_index_dir, settings, tmp_path):
    graph, tid = routed(built_index_dir, settings, tmp_path), new_thread_id()
    out = run_turn(graph, tid, "Please raise a ticket for my broken VPN")
    assert out.route_kind == "ticket" and out.awaiting == "email" and "email" in out.reply.lower()
    assert "search_kb" not in out.steps
    done = run_turn(graph, tid, EMAIL)
    assert done.email == EMAIL and "t_collect" in done.steps and done.awaiting == "none"


def test_a_direct_ticket_request_with_a_known_email_asks_nothing(built_index_dir, settings, tmp_path):
    graph = routed(built_index_dir, settings, tmp_path)
    out = run_turn(graph, new_thread_id(), "Please raise a ticket for my broken VPN", email=EMAIL)
    assert out.awaiting == "none" and "email" not in out.reply.lower()


def test_a_bare_ticket_request_asks_what_it_is_about(built_index_dir, settings, tmp_path):
    graph = routed(built_index_dir, settings, tmp_path)
    out = run_turn(graph, new_thread_id(), "I want to raise a ticket", email=EMAIL)
    assert out.reply == "What would you like the ticket to be about?" and out.awaiting == "none"


def test_a_new_topic_while_the_email_is_awaited_clears_the_flow(built_index_dir, settings, tmp_path):
    graph, tid = routed(built_index_dir, settings, tmp_path), new_thread_id()
    run_turn(graph, tid, "Please raise a ticket for my broken VPN")
    other = run_turn(graph, tid, "How do I reset my forgotten password?")
    assert other.awaiting == "none" and other.pending_offer is None and "search_kb" in other.steps


def test_routes_for_chitchat_and_deflection_do_not_start_a_ticket(built_index_dir, settings, tmp_path):
    graph = routed(built_index_dir, settings, tmp_path)
    for text, kind in (("Hi there", "smalltalk"), ("What's the weather like?", "deflect"), ("I need some help", "clarify")):
        out = run_turn(graph, new_thread_id(), text)
        assert out.route_kind == kind and out.ticket is None and out.pending_offer is None, text
