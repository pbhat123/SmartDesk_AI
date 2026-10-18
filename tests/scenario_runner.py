"""Runs the scripted conversations in evals/scenarios against the agent graph (mock LLMs, mock
tickets). Each turn's `expect` block is checked against the reply, the state and the tool traffic."""
from evals.common import EVALS_DIR
from evals.scenario_check import check_turn
from evals.scenario_schema import Scenario, load_scenario
from smartdesk_ai.agent.confirm import make_confirm_parser
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.llm.mock import MockConfirmModel, MockDraftModel
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM

SCENARIOS = EVALS_DIR / "scenarios"


def load(name: str) -> Scenario:
    return load_scenario(SCENARIOS / f"{name}.yaml")


STATUS_STEPS = {"tickets_none", "ticket_list", "ticket_status_single"}


def build(scenario: Scenario, built_index_dir, settings, tmp_path, answer_mode="insufficient"):
    parser = make_confirm_parser(settings, builder=lambda m, c: MockConfirmModel())
    seeded = any(t.expect.step in STATUS_STEPS for t in scenario.turns)  # the demo tickets of the seed script
    return assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM(answer_mode), routed=True,
                    drafter_llm=MockDraftModel(), parse_confirm=parser, seeded=seeded)


def run_scenario(scenario: Scenario, stack) -> list:
    tid = new_thread_id()
    email = scenario.setup.email_known
    states = []
    for index, turn in enumerate(scenario.turns):
        issues_before = len(stack.client._issues)
        searches_before = len(stack.spy.calls)
        state = run_turn(stack.graph, tid, turn.user, email=email if index == 0 else None)
        problems = check_turn(turn.expect, state, client=stack.client, issues_before=issues_before,
                              issues_after=len(stack.client._issues), searched=len(stack.spy.calls) > searches_before)
        assert not problems, (turn.expect.step, problems, state.reply)
        states.append(state)
    return states
