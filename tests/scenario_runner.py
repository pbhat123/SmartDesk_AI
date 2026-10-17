"""Runs the scripted conversations in evals/scenarios against the agent graph (mock LLMs, mock
tickets). Each turn's `expect` block is checked against the reply, the state and the tool traffic."""
import re

from evals.common import EVALS_DIR
from evals.scenario_schema import Expect, Scenario, load_scenario
from smartdesk_ai.agent.confirm import make_confirm_parser
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.llm.mock import MockConfirmModel, MockDraftModel
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM

SCENARIOS = EVALS_DIR / "scenarios"
FIELD_LABELS = {"title": "Title:", "description": "Description:", "category": "Category:", "email": "Email:",
                "priority": "Priority:"}


def load(name: str) -> Scenario:
    return load_scenario(SCENARIOS / f"{name}.yaml")


STATUS_STEPS = {"tickets_none", "ticket_list", "ticket_status_single"}


def build(scenario: Scenario, built_index_dir, settings, tmp_path, answer_mode="insufficient"):
    parser = make_confirm_parser(settings, builder=lambda m, c: MockConfirmModel())
    seeded = any(t.expect.step in STATUS_STEPS for t in scenario.turns)  # the demo tickets of the seed script
    return assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM(answer_mode), routed=True,
                    drafter_llm=MockDraftModel(), parse_confirm=parser, seeded=seeded)


def check_step(step: str, state, expect: Expect, issues_before: int, issues_after: int) -> None:
    reply = state.reply
    if step == "ticket_offer":
        assert state.pending_offer is not None and "yes or no" in reply.lower(), reply
    elif step == "ticket_offer_with_email_request":
        assert state.pending_offer is not None and "email" in reply.lower() and "don't know" in reply.lower(), reply
    elif step == "email_requested":
        assert state.awaiting == "email" and "email" in reply.lower(), reply
    elif step == "ticket_summary":
        assert "Shall I go ahead and create this?" in reply, reply
    elif step == "ticket_created":
        assert issues_after == issues_before + 1, "exactly one ticket must be created"
    elif step == "ticket_cancelled":
        assert "won't create" in reply and issues_after == issues_before, reply
    elif step == "tickets_none":
        assert "no tickets" in reply.lower() and not re.search(r"#\d+|https?://", reply), reply
    elif step == "ticket_list":
        assert state.awaiting == "ticket_pick" and len(re.findall(r"^\d+\. #\d+ ", reply, re.M)) >= 2, reply
    elif step == "ticket_status_single":
        assert len(re.findall(r"#\d+", reply)) == 1, reply
    elif step == "ticket_unavailable":
        assert "try again later" in reply.lower(), reply
    else:
        raise NotImplementedError(f"scenario step {step!r} is not covered yet")


def run_scenario(scenario: Scenario, stack) -> list:
    tid = new_thread_id()
    email = scenario.setup.email_known
    states = []
    for index, turn in enumerate(scenario.turns):
        issues_before = len(stack.client._issues)
        searches_before = len(stack.spy.calls)
        state = run_turn(stack.graph, tid, turn.user, email=email if index == 0 else None)
        issues_after = len(stack.client._issues)
        expect = turn.expect
        check_step(expect.step, state, expect, issues_before, issues_after)
        for text in expect.reply_contains:
            assert text in state.reply, (text, state.reply)
        for field in expect.summary_fields:
            assert FIELD_LABELS[field] in state.reply, (field, state.reply)
        if expect.ticket_created is not None:
            assert (issues_after > issues_before) == expect.ticket_created, expect.step
        if expect.intent is not None:
            assert state.intent is not None and state.intent.intent == expect.intent, (expect.intent, state.intent)
        if expect.tool == "search_kb":
            assert len(stack.spy.calls) > searches_before
        if expect.tool == "create_ticket":
            assert issues_after == issues_before + 1
        if expect.tool == "none":
            assert len(stack.spy.calls) == searches_before and issues_after == issues_before
        if expect.awaiting_after is not None:
            assert state.awaiting == expect.awaiting_after, (state.awaiting, expect.awaiting_after)
        if (expect.ticket_id_in_reply or expect.ticket_link_in_reply or expect.status_in_reply
                or expect.team_update_in_reply):
            [number] = re.findall(r"#(\d+)", state.reply)  # exactly one ticket is shown
            ticket = stack.client._issues[int(number)]
            if expect.ticket_link_in_reply:
                assert ticket.url in state.reply
            if expect.status_in_reply:
                assert ticket.status.value in state.reply, (ticket.status, state.reply)
            if expect.team_update_in_reply:
                update = stack.client.latest_update(ticket.number)
                assert update is not None and update.text in state.reply, state.reply
        states.append(state)
    return states
