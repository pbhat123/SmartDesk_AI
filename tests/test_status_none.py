"""T4.25: no tickets on file gets a clear message that offers nothing invented (FR-C4)."""
import re

import pytest

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.agent.status_flow import NO_TICKETS
from tests.agent_helpers import assemble
from tests.scenario_runner import build, load, run_scenario
from tests.seed_helper import ZERO_EMAIL


@pytest.fixture
def stack(built_index_dir, settings, tmp_path):
    return assemble(built_index_dir, settings, tmp_path, routed=True, seeded=True)


def ask(stack, text="What is the status of my ticket?", email=ZERO_EMAIL):
    return run_turn(stack.graph, new_thread_id(), text, email=email)


def test_zero_tickets_says_so_clearly(stack):
    state = ask(stack)
    assert state.reply == NO_TICKETS and "no tickets" in state.reply.lower()


def test_it_invents_no_ticket_number_link_status_or_update(stack):
    reply = ask(stack).reply
    assert not re.search(r"#\d|https?://|\bin progress\b|\bresolved\b|\bopen\b|update", reply, re.I)


def test_it_offers_no_ticket_and_starts_no_flow(stack):
    state = ask(stack)
    assert state.awaiting == "none" and state.pending_offer is None and state.ticket is None
    assert state.status_candidates == [] and "ticket" not in state.reply.lower().replace("no tickets", "")
    assert len(stack.client._issues) == 4  # only the seeded demo tickets: nothing was created


def test_it_does_not_echo_the_email(stack):
    assert ZERO_EMAIL not in ask(stack).reply


def test_the_same_message_for_each_way_of_asking(stack):
    replies = {ask(stack, t).reply for t in ("What is the status of my ticket?", "Any updates on my monitor ticket?",
                                              "Did anyone look at my VPN issue?")}
    assert replies == {NO_TICKETS}


def test_after_the_email_is_given_in_the_chat(stack):
    tid = new_thread_id()
    run_turn(stack.graph, tid, "What is the status of my ticket?")
    state = run_turn(stack.graph, tid, ZERO_EMAIL)
    assert state.reply == NO_TICKETS and state.email == ZERO_EMAIL


def test_the_employee_can_still_start_a_ticket_afterwards(stack):
    tid = new_thread_id()
    run_turn(stack.graph, tid, "What is the status of my ticket?", email=ZERO_EMAIL)
    state = run_turn(stack.graph, tid, "Please raise a ticket for my broken VPN")
    assert state.route_kind == "ticket"


def test_the_status_0_scenario_passes(built_index_dir, settings, tmp_path):
    scenario = load("status_0")
    run_scenario(scenario, build(scenario, built_index_dir, settings, tmp_path))
