"""T8.2: cross-session recall on a status check (FR-H10, FR-C1). Session 1 creates a ticket; a new session (new thread, empty
conversation) with the same email gets it back without naming it. The number comes from the long-term store, the title and status
are read live from GitHub (mock here), no session-1 message reaches session 2, and another email recalls nothing. Mock only."""
import pytest

from smartdesk_ai.agent.graph import run_turn, thread_config
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.agent.status_flow import NO_TICKETS
from smartdesk_ai.tickets import MockTicketClient, TicketStatus
from smartdesk_ai.tools.definitions import GetTicketsArgs
from smartdesk_ai.tools.ticket_tools import make_get_tickets
from tests.scenario_runner import build, load, run_scenario

EMAIL = "priya.k@smartdesk.example"  # the email flow_b's session 1 gives
OTHER = "arjun.m@smartdesk.example"
SESSION_1_TEXT = "My monitor keeps flickering"


class NoSearchClient(MockTicketClient):
    """GitHub search finds nothing, so a ticket can only come back through a stored number (not the FR-C11 fallback)."""

    def search_by_email(self, email):
        return []


@pytest.fixture
def stack(built_index_dir, settings, tmp_path):
    """Session 1 is the real flow B: a flickering monitor, the email, 'yes'. Then the email-to-ticket map is wiped, so what is
    left of the employee is the long-term store."""
    scenario = load("flow_b")
    stack = build(scenario, built_index_dir, settings, tmp_path)
    stack.client.__class__ = NoSearchClient
    run_scenario(scenario, stack)
    assert len(stack.client._issues) == 1
    stack.store._conn.execute("DELETE FROM email_tickets")
    stack.number = next(iter(stack.client._issues))
    return stack


def status_in_new_session(stack, email=EMAIL, ask="What's the status of my ticket?"):
    tid = new_thread_id()
    return tid, run_turn(stack.graph, tid, ask, email=email)


# --- recall ---------------------------------------------------------------------------------------------------------------

def test_a_new_session_with_the_same_email_recalls_the_ticket_without_it_being_named(stack):
    tid, state = status_in_new_session(stack)
    ticket = stack.client._issues[stack.number]
    assert f"#{stack.number}" in state.reply and ticket.title in state.reply and "Status: Open" in state.reply
    assert state.steps[-1] == "s_lookup" and stack.memory.ticket_numbers(EMAIL) == [stack.number]


def test_without_an_email_the_new_session_asks_for_one_and_then_recalls(stack):
    tid = new_thread_id()
    asked = run_turn(stack.graph, tid, "Did anyone look at my monitor issue?")
    assert asked.awaiting == "email" and "get_tickets" not in asked.steps
    recalled = run_turn(stack.graph, tid, EMAIL.upper())
    assert f"#{stack.number}" in recalled.reply and stack.client._issues[stack.number].title in recalled.reply


def test_title_and_status_are_read_live_not_from_the_store(stack):
    stack.client.update_issue(stack.number, state="closed", labels=["resolved"])
    state = status_in_new_session(stack)[1]
    assert "Status: Resolved" in state.reply and stack.client._issues[stack.number].status is TicketStatus.RESOLVED
    stack.client._issues[stack.number] = stack.client._issues[stack.number].model_copy(update={"title": "Monitor replaced on floor two"})
    assert "Monitor replaced on floor two" in status_in_new_session(stack)[1].reply


def test_several_remembered_tickets_are_listed_with_live_titles_and_statuses(stack):
    second = stack.client.add_issue("Second request about the dock", f"Employee: {EMAIL}", labels=["IT"])
    stack.memory.remember(EMAIL, second.number, "IT")
    state = status_in_new_session(stack, ask="Any updates on my tickets?")[1]
    assert state.awaiting == "ticket_pick" and sorted(c.number for c in state.status_candidates) == [1, second.number]
    assert "Second request about the dock" in state.reply and state.reply.count("#") == 2


def test_a_ticket_in_both_the_map_and_the_store_is_listed_once(stack):
    stack.store.link_email(EMAIL, stack.number)
    state = status_in_new_session(stack)[1]
    assert state.reply.count(f"#{stack.number}") == 1 and state.awaiting == "none"


# --- other people, other sessions ---------------------------------------------------------------------------------------

def test_a_different_email_in_the_new_session_recalls_nothing(stack):
    state = status_in_new_session(stack, email=OTHER)[1]
    assert state.reply == NO_TICKETS and str(stack.number) not in state.reply and "monitor" not in state.reply.lower()


def test_a_stored_number_that_is_another_employees_ticket_is_never_shown(stack):
    """A number kept for one email can later point at someone else's issue (a mock restart renumbers); the issue's own
    `Employee:` line must match the email, or it is skipped."""
    theirs = stack.client.add_issue("Payslip question", f"Employee: {OTHER}", labels=["HR"])
    stack.memory.remember(EMAIL, theirs.number, "HR")
    state = status_in_new_session(stack)[1]
    assert "Payslip question" not in state.reply and f"#{theirs.number}" not in state.reply and f"#{stack.number}" in state.reply


def test_a_stored_number_whose_issue_is_gone_is_skipped(stack):
    stack.memory.remember(EMAIL, 99, "IT")
    state = status_in_new_session(stack)[1]
    assert "#99" not in state.reply and f"#{stack.number}" in state.reply


def test_no_message_from_session_1_is_in_session_2(stack):
    tid, state = status_in_new_session(stack)
    texts = [str(m.content) for m in stack.graph.get_state(thread_config(tid)).values["messages"]]
    assert len(texts) == 2 and texts[0] == "What's the status of my ticket?"
    assert not any(SESSION_1_TEXT.lower() in t.lower() and "Ticket #" not in t for t in texts)
    assert state.status_pending_email is False and state.pending_offer is None
    assert "Yes, create it" not in texts and EMAIL not in " ".join(texts[:1])


def test_the_email_is_not_in_the_thread_config_or_the_reply(stack):
    tid, state = status_in_new_session(stack)
    assert thread_config(tid) == {"configurable": {"thread_id": tid}} and EMAIL not in state.reply


def test_the_recall_works_with_the_long_term_store_alone_and_not_at_all_without_it(stack):
    """The store is what carries the number; an app built without one finds nothing once the map is empty and search is off."""
    assert stack.memory.ticket_numbers(EMAIL)
    without = make_get_tickets(stack.client, stack.store)(GetTicketsArgs(email=EMAIL))
    assert without.tickets == []
    with_store = make_get_tickets(stack.client, stack.store, memory=stack.memory)(GetTicketsArgs(email=EMAIL))
    assert [t.number for t in with_store.tickets] == [stack.number]


def test_a_failed_search_still_returns_the_remembered_ticket(stack):
    client = MockTicketClient()
    ticket = client.add_issue("Printer jams on floor two", f"Employee: {EMAIL}", labels=["IT"])
    stack.memory.remember(EMAIL, ticket.number, "IT")
    client.fail_searches = 1
    found = make_get_tickets(client, stack.store, memory=stack.memory)(GetTicketsArgs(email=EMAIL))
    assert [t.number for t in found.tickets] == [ticket.number]
