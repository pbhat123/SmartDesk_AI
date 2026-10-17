"""T4.19: scripted ticket scenarios (FR-B2, FR-B4). A scenario creates exactly one issue after
"yes" and none after "no"; the expectations live in evals/scenarios/*.yaml."""
import pytest

from tests.scenario_runner import build, load, run_scenario

TICKET_SCENARIOS = ["ticket_create_yes", "ticket_create_no", "ticket_create_edit", "flow_b"]


@pytest.mark.parametrize("name", TICKET_SCENARIOS)
def test_the_scenario_passes(name, built_index_dir, settings, tmp_path):
    scenario = load(name)
    stack = build(scenario, built_index_dir, settings, tmp_path)
    run_scenario(scenario, stack)


def test_the_yes_scenario_creates_exactly_one_issue(built_index_dir, settings, tmp_path):
    scenario = load("ticket_create_yes")
    stack = build(scenario, built_index_dir, settings, tmp_path)
    run_scenario(scenario, stack)
    assert len(stack.client._issues) == 1


def test_the_no_scenario_creates_none(built_index_dir, settings, tmp_path):
    scenario = load("ticket_create_no")
    stack = build(scenario, built_index_dir, settings, tmp_path)
    run_scenario(scenario, stack)
    assert len(stack.client._issues) == 0


def test_the_edit_scenario_creates_one_issue_with_the_edited_priority(built_index_dir, settings, tmp_path):
    scenario = load("ticket_create_edit")
    stack = build(scenario, built_index_dir, settings, tmp_path)
    run_scenario(scenario, stack)
    [issue] = stack.client._issues.values()
    assert "high" in issue.labels and "IT" in issue.labels


def test_flow_b_creates_the_issue_for_the_email_given_in_the_chat(built_index_dir, settings, tmp_path):
    scenario = load("flow_b")
    stack = build(scenario, built_index_dir, settings, tmp_path)
    run_scenario(scenario, stack)
    [issue] = stack.client._issues.values()
    assert "Employee: priya.k@smartdesk.example" in issue.body


def test_every_ticket_scenario_carries_the_fr_ids_it_covers():
    for name in TICKET_SCENARIOS:
        assert load(name).fr
