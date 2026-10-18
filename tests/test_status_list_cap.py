"""T4.26 follow-up: the status list is capped at `status.max_listed`, newest first, followed by
"and N older ones" (FR-C3). Mock tickets; no network."""
import re

import pytest
import yaml

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.agent.status_flow import older_line
from smartdesk_ai.config.settings import load_settings
from smartdesk_ai.tickets import Category, TicketCreate
from tests.agent_helpers import assemble

EMAIL = "busy.person@smartdesk.example"


def stack_with(n, built_index_dir, settings, tmp_path):
    stack = assemble(built_index_dir, settings, tmp_path, routed=True)
    for i in range(1, n + 1):
        ticket = stack.client.create(TicketCreate(
            title=f"Request number {i}", description="d", category=Category.IT, email=EMAIL, request_id=f"req-{i}"))
        stack.store.link_email(EMAIL, ticket.number)
    return stack


def ask(stack, text="Any updates on my tickets?"):
    tid = new_thread_id()
    return tid, run_turn(stack.graph, tid, text, email=EMAIL)


def lines(state):
    return [line for line in state.reply.splitlines() if re.match(r"^\d+\. ", line)]


def test_the_cap_is_a_config_value():
    assert load_settings().status.max_listed == 10


@pytest.mark.parametrize("bad", [0, -1, "ten"])
def test_a_bad_cap_is_rejected_by_config(tmp_path, bad):
    from pathlib import Path

    data = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    data["status"]["max_listed"] = bad
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(Exception, match="max_listed"):
        load_settings(path)


def test_twelve_tickets_list_the_newest_ten_and_say_how_many_are_older(built_index_dir, settings, tmp_path):
    stack = stack_with(12, built_index_dir, settings, tmp_path)
    _, state = ask(stack)
    assert len(lines(state)) == settings.status.max_listed == 10
    assert lines(state)[0].startswith("1. #12 ") and lines(state)[-1].startswith("10. #3 ")
    assert "and 2 older ones." in state.reply
    assert "I found 12 tickets" in state.reply
    assert [c.number for c in state.status_candidates] == list(range(12, 2, -1))
    assert state.awaiting == "ticket_pick"


def test_the_older_line_comes_after_the_list_and_before_the_question(built_index_dir, settings, tmp_path):
    _, state = ask(stack_with(12, built_index_dir, settings, tmp_path))
    text = state.reply
    assert text.index("10. #3") < text.index("and 2 older ones.") < text.index("Which one")
    assert text.count("?") == 1


def test_one_older_ticket_is_singular(built_index_dir, settings, tmp_path):
    _, state = ask(stack_with(11, built_index_dir, settings, tmp_path))
    assert "and 1 older one." in state.reply and "older ones" not in state.reply


def test_exactly_the_cap_has_no_older_line(built_index_dir, settings, tmp_path):
    _, state = ask(stack_with(10, built_index_dir, settings, tmp_path))
    assert len(lines(state)) == 10 and "older" not in state.reply


def test_fewer_than_the_cap_lists_them_all(built_index_dir, settings, tmp_path):
    _, state = ask(stack_with(3, built_index_dir, settings, tmp_path))
    assert len(lines(state)) == 3 and "older" not in state.reply


def test_the_cap_follows_the_config_value(built_index_dir, settings, tmp_path):
    small = settings.model_copy(update={"status": settings.status.model_copy(update={"max_listed": 3})})
    stack = stack_with(5, built_index_dir, small, tmp_path)
    _, state = ask(stack)
    assert len(lines(state)) == 3 and "and 2 older ones." in state.reply


def test_a_listed_ticket_can_still_be_picked(built_index_dir, settings, tmp_path):
    stack = stack_with(12, built_index_dir, settings, tmp_path)
    tid, _ = ask(stack)
    assert "Ticket #3 (mock): Request number 3" in run_turn(stack.graph, tid, "the last one").reply
    tid, _ = ask(stack)
    assert "Ticket #12 (mock): Request number 12" in run_turn(stack.graph, tid, "the first one").reply


def test_an_older_ticket_cannot_be_picked_from_the_capped_list(built_index_dir, settings, tmp_path):
    stack = stack_with(12, built_index_dir, settings, tmp_path)
    tid, _ = ask(stack)
    again = run_turn(stack.graph, tid, "#1")  # not on the list, so not looked up
    assert "Ticket #1:" not in again.reply


def test_the_older_line_wording():
    assert older_line(1) == "and 1 older one." and older_line(2) == "and 2 older ones."
