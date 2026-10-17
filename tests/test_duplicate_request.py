"""T4.20: a near-identical open ticket for the same email triggers an "open another?" question
before the approval gate (FR-F4). Separate from the request-ID protection of FR-G6."""
import pytest

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.agent.ticket_agent import find_duplicate, title_similarity
from smartdesk_ai.llm.mock import MockDraftModel
from smartdesk_ai.tickets import Ticket
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM

EMAIL = "priya.k@smartdesk.example"
ISSUE = "My monitor keeps flickering"


def ticket(number=1, title="My monitor keeps flickering", state="open", labels=()):
    return Ticket(number=number, url=f"https://github.com/o/r/issues/{number}", title=title, state=state,
                  labels=tuple(labels))


# --- the title comparison ----------------------------------------------------------------------


@pytest.mark.parametrize("other", [
    "My monitor keeps flickering",
    "my monitor keeps flickering!",
    "My monitor keeps on flickering",
    "Monitor keeps flickering",
])
def test_near_identical_titles_match(settings, other):
    assert title_similarity("My monitor keeps flickering", other) >= settings.thresholds.duplicate_title


@pytest.mark.parametrize("other", ["VPN down", "Unable to connect to VPN from home", "My laptop battery drains fast"])
def test_different_titles_do_not_match(settings, other):
    assert title_similarity("My monitor keeps flickering", other) < settings.thresholds.duplicate_title


def test_only_open_tickets_count(settings):
    threshold = settings.thresholds.duplicate_title
    assert find_duplicate(ISSUE, [ticket(state="closed")], threshold) is None
    assert find_duplicate(ISSUE, [ticket(state="open", labels=("in-progress",))], threshold).number == 1


def test_the_closest_open_ticket_is_the_one_reported(settings):
    tickets = [ticket(1, "My monitor keeps flickering badly today"), ticket(2, "My monitor keeps flickering")]
    assert find_duplicate(ISSUE, tickets, settings.thresholds.duplicate_title).number == 2


def test_the_threshold_comes_from_config(settings):
    assert 0.5 < settings.thresholds.duplicate_title <= 1.0


# --- through the graph ---------------------------------------------------------------------------


@pytest.fixture
def stack(built_index_dir, settings, tmp_path):
    return assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"),
                    drafter_llm=MockDraftModel())


def seed(stack, title=ISSUE, email=EMAIL, state="open"):
    return stack.client.add_issue(title, f"Something broke.\n\nEmployee: {email}\nRequest ID: req-seed", state=state)


def reach_the_check(stack, tid=None):
    tid = tid or new_thread_id()
    run_turn(stack.graph, tid, ISSUE, email=EMAIL)
    return tid, run_turn(stack.graph, tid, "yes")


def paused(stack, tid):
    return stack.graph.get_state({"configurable": {"thread_id": tid}}).next


def test_a_near_identical_open_ticket_triggers_the_question_before_the_gate(stack):
    existing = seed(stack)
    tid, state = reach_the_check(stack)
    assert paused(stack, tid) == ("t_dup_ask",)
    assert "open another" in state.reply.lower() and state.reply.count("?") == 1
    assert "Shall I go ahead" not in state.reply and "t_present" not in state.steps
    assert f"#{existing.number}" in state.reply and existing.url in state.reply and ISSUE in state.reply
    assert len(stack.client._issues) == 1


def test_the_question_names_the_ticket_exactly_as_github_returned_it(stack):
    seed(stack, title="My monitor keeps flickering")
    _, state = reach_the_check(stack)
    assert "(Open)" in state.reply
    assert EMAIL not in state.reply


def test_yes_goes_on_to_the_gate_and_a_second_ticket_can_be_created(stack):
    seed(stack)
    tid, _ = reach_the_check(stack)
    gate = run_turn(stack.graph, tid, "yes")
    assert "Shall I go ahead and create this?" in gate.reply and paused(stack, tid) == ("t_confirm",)
    assert len(stack.client._issues) == 1  # nothing created yet
    run_turn(stack.graph, tid, "yes")
    assert len(stack.client._issues) == 2


def test_no_ends_the_flow_and_creates_nothing(stack):
    seed(stack)
    tid, _ = reach_the_check(stack)
    done = run_turn(stack.graph, tid, "no")
    assert paused(stack, tid) == () and len(stack.client._issues) == 1
    assert done.ticket is None and done.duplicate is None and "t_cancel" in done.steps


def test_an_unclear_reply_is_asked_once_more_then_cancels(stack):
    seed(stack)
    tid, first = reach_the_check(stack)
    again = run_turn(stack.graph, tid, "hmm")
    assert paused(stack, tid) == ("t_dup_ask",) and first.reply in again.reply
    done = run_turn(stack.graph, tid, "what is the weather")
    assert paused(stack, tid) == () and len(stack.client._issues) == 1 and "t_cancel" in done.steps


def test_an_unclear_reply_then_yes_still_proceeds(stack):
    seed(stack)
    tid, _ = reach_the_check(stack)
    run_turn(stack.graph, tid, "hmm")
    gate = run_turn(stack.graph, tid, "yes")
    assert paused(stack, tid) == ("t_confirm",) and "Shall I go ahead" in gate.reply


def test_the_gate_after_a_duplicate_question_still_has_its_own_unclear_allowance(stack):
    seed(stack)
    tid, _ = reach_the_check(stack)
    run_turn(stack.graph, tid, "hmm")
    run_turn(stack.graph, tid, "yes")
    run_turn(stack.graph, tid, "hmm")  # first unclear at the gate, not the second
    assert paused(stack, tid) == ("t_confirm",)


def test_the_pause_at_the_question_survives_a_restart(built_index_dir, settings, tmp_path):
    first = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"),
                     drafter_llm=MockDraftModel())
    seed(first)
    tid, _ = reach_the_check(first)
    first.saver.conn.close()
    second = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"),
                      drafter_llm=MockDraftModel(), client=first.client)
    assert paused(second, tid) == ("t_dup_ask",)


@pytest.mark.parametrize("make", [
    lambda s: seed(s, title="VPN down"),                       # a different issue
    lambda s: seed(s, state="closed"),                         # no longer open
    lambda s: seed(s, email="someone.else@smartdesk.example"),  # another employee's ticket
])
def test_no_question_when_there_is_no_open_near_identical_ticket_of_theirs(stack, make):
    make(stack)
    tid, state = reach_the_check(stack)
    assert paused(stack, tid) == ("t_confirm",) and "Shall I go ahead" in state.reply
    assert "t_dup_check" in state.steps and "t_dup_ask" not in state.steps


def test_no_tickets_at_all_goes_straight_to_the_gate(stack):
    tid, state = reach_the_check(stack)
    assert "Shall I go ahead" in state.reply and paused(stack, tid) == ("t_confirm",)


def test_the_lookup_goes_through_the_executor_as_get_tickets(stack, settings):
    seen = []
    original = stack.executor.execute
    stack.executor.execute = lambda name, args, **kw: (seen.append(name), original(name, args, **kw))[1]
    reach_the_check(stack)
    assert "get_tickets" in seen


def test_a_failed_lookup_skips_the_check_instead_of_blocking_the_ticket(stack):
    seed(stack)
    stack.client.fail_searches = 10  # every attempt fails
    tid, state = reach_the_check(stack)
    assert "dup_check_skipped" in state.steps and "Shall I go ahead" in state.reply
    assert paused(stack, tid) == ("t_confirm",)


def test_a_direct_request_is_checked_too(built_index_dir, settings, tmp_path):
    stack = assemble(built_index_dir, settings, tmp_path, routed=True, drafter_llm=MockDraftModel())
    seed(stack, title="Please raise a ticket for my broken VPN")  # the mock drafter titles it with their sentence
    tid = new_thread_id()
    state = run_turn(stack.graph, tid, "Please raise a ticket for my broken VPN", email=EMAIL)
    assert paused(stack, tid) == ("t_dup_ask",) and "open another" in state.reply.lower()
