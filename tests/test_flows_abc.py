"""T4.30: the three sample flows of the brief, end to end in mock mode (SPEC section 7: Flow A, B, C).
A: password reset answered, then thanks and goodbye. B: monitor flickering, ticket offered, ID returned.
C: status check, one open ticket with its team update. Scripts are evals/scenarios/flow_*.yaml."""
import re

import pytest

from smartdesk_ai.tickets import TicketStatus
from tests.scenario_runner import build, load, run_scenario


def play(name, built_index_dir, settings, tmp_path, answer_mode="insufficient"):
    scenario = load(name)
    stack = build(scenario, built_index_dir, settings, tmp_path, answer_mode=answer_mode)
    return scenario, stack, run_scenario(scenario, stack)


def test_flow_a_answers_the_password_reset_then_says_goodbye(built_index_dir, settings, tmp_path):
    _, stack, states = play("flow_a", built_index_dir, settings, tmp_path, answer_mode="cite_top")
    answer, thanks, goodbye = states
    assert "IT-PWD-01" in answer.part_result.citations and "IT-PWD-01" in answer.reply
    assert answer.steps[:3] == ["start", "route", "search_kb"] and "offer" not in answer.steps
    assert thanks.steps[-1] == goodbye.steps[-1] == "chitchat"
    assert len(stack.client._issues) == 0  # nothing was created
    assert all(s.pending_offer is None for s in states)
    assert len(stack.spy.calls) == 1  # only the question searched; thanks and bye made no tool call


def test_flow_b_offers_a_ticket_collects_the_email_and_returns_the_id(built_index_dir, settings, tmp_path):
    scenario, stack, states = play("flow_b", built_index_dir, settings, tmp_path)
    offer, summary, created = states
    assert offer.awaiting == "email" and offer.reply.lower().index("sorry") < offer.reply.lower().index("ticket")
    for label in ("Title:", "Description:", "Category:", "Email:"):
        assert label in summary.reply
    assert len(stack.client._issues) == 1
    [issue] = stack.client._issues.values()
    assert f"#{issue.number}" in created.reply and issue.url in created.reply
    assert issue.title.lower().startswith("my monitor")


def test_flow_c_lists_the_one_open_ticket_with_its_team_update(built_index_dir, settings, tmp_path):
    scenario, stack, states = play("flow_c", built_index_dir, settings, tmp_path)
    asked, listed = states
    assert asked.awaiting == "email" and "get_tickets" not in asked.steps
    [number] = re.findall(r"#(\d+)", listed.reply)
    ticket = stack.client._issues[int(number)]
    assert ticket.title in listed.reply and ticket.status is not TicketStatus.CLOSED
    assert stack.client.latest_update(ticket.number).text in listed.reply
    assert listed.awaiting == "none"


@pytest.mark.parametrize("name", ["flow_a", "flow_b", "flow_c"])
def test_each_flow_is_deterministic_across_two_runs(name, built_index_dir, settings, tmp_path_factory):
    mode = "cite_top" if name == "flow_a" else "insufficient"
    replies = []
    for run in range(2):
        _, _, states = play(name, built_index_dir, settings, tmp_path_factory.mktemp(f"{name}{run}"), mode)
        replies.append([re.sub(r"#\d+|https?://\S+", "", s.reply) for s in states])
    assert replies[0] == replies[1]
