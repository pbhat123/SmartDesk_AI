"""T4.24: several tickets are listed briefly and the employee picks one (FR-C3, FR-D2). Seeded mock."""
import re

import pytest

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import TicketRef, new_thread_id
from smartdesk_ai.agent.status_flow import pick_ticket
from tests.agent_helpers import assemble
from tests.seed_helper import MANY_EMAIL, ONE_EMAIL

REFS = [
    TicketRef(number=11, title="VPN drops every hour from home", status="Resolved", url="https://x/issues/11"),
    TicketRef(number=14, title="Question about a payslip deduction", status="Open", url="https://x/issues/14"),
    TicketRef(number=20, title="Monitor flickers after docking", status="In Progress", url="https://x/issues/20"),
]


@pytest.fixture
def stack(built_index_dir, settings, tmp_path):
    return assemble(built_index_dir, settings, tmp_path, routed=True, seeded=True)


def listed(stack, tid=None):
    tid = tid or new_thread_id()
    return tid, run_turn(stack.graph, tid, "Any updates on my tickets?", email=MANY_EMAIL)


# --- the list -----------------------------------------------------------------------------------


def test_two_or_more_tickets_produce_a_pick_list_and_wait_for_the_choice(stack):
    _, state = listed(stack)
    numbers = stack.store.numbers_for(MANY_EMAIL)
    assert state.awaiting == "ticket_pick" and [c.number for c in state.status_candidates] == numbers
    for ref in state.status_candidates:
        assert f"#{ref.number}" in state.reply and ref.title in state.reply and f"({ref.status})" in state.reply


def test_the_list_is_brief_one_line_per_ticket_and_one_question(stack):
    _, state = listed(stack)
    item_lines = [line for line in state.reply.splitlines() if re.match(r"^\d+\. ", line)]
    assert len(item_lines) == 2 and all(len(line) < 120 for line in item_lines)
    assert state.reply.count("?") == 1 and "second one" in state.reply
    assert "http" not in state.reply, "the links and updates come with the chosen ticket"


def test_no_team_comment_is_read_for_the_list(stack):
    seen = []
    original = stack.executor.execute
    stack.executor.execute = lambda name, args, **kw: (seen.append(args), original(name, args, **kw))[1]
    listed(stack)
    assert seen == [{"email": MANY_EMAIL}]


def test_the_email_is_not_shown_in_the_list(stack):
    assert MANY_EMAIL not in listed(stack)[1].reply


# --- the pick ----------------------------------------------------------------------------------------


def test_the_second_one_selects_the_second_ticket(stack):
    tid, first = listed(stack)
    second = first.status_candidates[1]
    done = run_turn(stack.graph, tid, "The second one")
    assert f"#{second.number}" in done.reply and second.title in done.reply
    assert f"#{first.status_candidates[0].number}" not in done.reply
    assert done.awaiting == "none" and done.status_candidates == []
    assert done.intent.intent == "ticket_status" or "s_pick" in " ".join(done.steps)


def test_the_choice_is_looked_up_live_for_that_ticket_only(stack):
    seen = []
    original = stack.executor.execute
    stack.executor.execute = lambda name, args, **kw: (seen.append(args), original(name, args, **kw))[1]
    tid, first = listed(stack)
    run_turn(stack.graph, tid, "the first one")
    assert seen[-1] == {"email": MANY_EMAIL, "ticket_number": first.status_candidates[0].number}


@pytest.mark.parametrize("text, index", [
    ("The first one", 0), ("second", 1), ("the 2nd one", 1), ("1", 0), ("2", 1), ("the last one", 1),
])
def test_the_ways_of_picking(stack, text, index):
    tid, first = listed(stack)
    done = run_turn(stack.graph, tid, text)
    assert f"#{first.status_candidates[index].number}" in done.reply


def test_picking_by_the_ticket_number(stack):
    tid, first = listed(stack)
    number = first.status_candidates[1].number
    assert f"#{number}" in run_turn(stack.graph, tid, f"#{number}").reply


def test_picking_by_a_word_from_the_title(stack):
    tid, first = listed(stack)
    vpn = next(c for c in first.status_candidates if "VPN" in c.title)
    assert f"#{vpn.number}" in run_turn(stack.graph, tid, "the VPN one").reply


def test_a_choice_past_the_end_of_the_list_asks_again(stack):
    tid, _ = listed(stack)
    again = run_turn(stack.graph, tid, "the third one")
    assert again.awaiting == "ticket_pick" and "only found 2" in again.reply and again.status_candidates


def test_a_ticket_that_was_not_listed_cannot_be_picked(stack):
    tid, _ = listed(stack)
    foreign = stack.store.numbers_for(ONE_EMAIL)[0]
    seen = []
    original = stack.executor.execute
    stack.executor.execute = lambda name, args, **kw: (seen.append(args), original(name, args, **kw))[1]
    run_turn(stack.graph, tid, f"#{foreign}")
    assert not any(a.get("ticket_number") == foreign for a in seen)


def test_a_new_topic_instead_of_a_pick_leaves_the_list(stack):
    tid, _ = listed(stack)
    other = run_turn(stack.graph, tid, "How do I reset my forgotten password?")
    assert other.awaiting == "none" and other.status_candidates == [] and "search_kb" in other.steps


def test_asking_again_while_a_list_is_open_lists_again(stack):
    tid, _ = listed(stack)
    again = run_turn(stack.graph, tid, "What is the status of my tickets?")
    assert again.awaiting == "ticket_pick" and len(again.status_candidates) == 2


def test_the_list_survives_a_restart(built_index_dir, settings, tmp_path):
    first = assemble(built_index_dir, settings, tmp_path, routed=True, seeded=True)
    tid, state = listed(first)
    first.saver.conn.close()
    second = assemble(built_index_dir, settings, tmp_path, routed=True, client=first.client)
    done = run_turn(second.graph, tid, "the second one")
    assert f"#{state.status_candidates[1].number}" in done.reply


def test_a_lookup_outage_on_the_pick_keeps_the_list_so_the_employee_can_retry(stack):
    tid, _ = listed(stack)
    stack.client.unreachable = True
    down = run_turn(stack.graph, tid, "the first one")
    assert "try again later" in down.reply.lower() and down.awaiting == "ticket_pick"
    stack.client.unreachable = False
    assert "Ticket #" in run_turn(stack.graph, tid, "the first one").reply


# --- pick_ticket ------------------------------------------------------------------------------------


@pytest.mark.parametrize("text, number", [
    ("the first one", 11), ("third", 20), ("#14", 14), ("ticket 20", 20), ("14", 14), ("2", 14), ("the payslip one", 14),
    ("last", 20),
])
def test_pick_ticket_resolves(text, number):
    assert pick_ticket(text, REFS).ref.number == number


@pytest.mark.parametrize("text", ["hmm", "what is the weather", "the one", "both"])
def test_pick_ticket_leaves_what_it_cannot_read(text):
    assert pick_ticket(text, REFS).kind == "none"


def test_pick_ticket_flags_a_number_past_the_end():
    assert pick_ticket("the fifth one", REFS).kind == "out_of_range"
    assert pick_ticket("7", REFS).kind == "out_of_range"


def test_a_word_in_two_titles_is_not_a_choice():
    twins = [REFS[0].model_copy(update={"title": "Monitor broken"}), REFS[1].model_copy(update={"title": "Monitor loose"})]
    assert pick_ticket("the monitor one", twins).kind == "none"
