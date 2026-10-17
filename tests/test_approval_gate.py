"""T4.16: the approval gate is a LangGraph interrupt (FR-B2, FR-G12). A proposed create_ticket
pauses the graph; yes creates exactly one ticket, no creates none, an edit shows the updated summary
first, the pause survives several turns and a restart, and a second unclear reply cancels."""
import pytest
from langchain_core.messages import AIMessage

from smartdesk_ai.agent.confirm import ConfirmDecision, apply_edit, rule_parser
from smartdesk_ai.agent.graph import make_t_create_node, run_turn
from smartdesk_ai.agent.state import AgentState, TicketDraft, new_thread_id
from smartdesk_ai.agent.ticket_agent import TicketDeps
from smartdesk_ai.llm.mock import MockDraftModel
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM

EMAIL = "priya.k@smartdesk.example"
ISSUE = "My monitor keeps flickering"


def scripted(replies):
    """A confirm parser that reads the reply the way a model would: from a table; yes/no exact."""

    def parse(reply, draft):
        exact = rule_parser(reply, draft)
        if exact.decision != "unclear":
            return exact
        return replies.get(reply.strip().lower(), ConfirmDecision(decision="unclear"))

    return parse


EDITS = {
    "make the priority high": ConfirmDecision(decision="edit", priority="high"),
    "change the title to monitor flicker": ConfirmDecision(decision="edit", title="Monitor flicker"),
    "also it is on the second floor": ConfirmDecision(decision="edit", details="it is on the second floor"),
    "this is for hr": ConfirmDecision(decision="edit", category="HR"),
    "hmm": ConfirmDecision(decision="unclear"),
    "tweak it": ConfirmDecision(decision="edit"),
}


@pytest.fixture
def stack(built_index_dir, settings, tmp_path):
    return assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"),
                    drafter_llm=MockDraftModel(), parse_confirm=scripted(EDITS))


def at_the_gate(stack, tid=None):
    tid = tid or new_thread_id()
    run_turn(stack.graph, tid, ISSUE, email=EMAIL)
    state = run_turn(stack.graph, tid, "yes")
    return tid, state


def issues(stack):
    return list(stack.client._issues.values())


def paused(stack, tid):
    return stack.graph.get_state({"configurable": {"thread_id": tid}}).next


# --- pause, yes, no -------------------------------------------------------------------------


def test_a_proposed_create_ticket_pauses_the_graph_before_anything_is_created(stack):
    tid, state = at_the_gate(stack)
    assert paused(stack, tid) == ("t_confirm",)
    assert "Shall I go ahead and create this?" in state.reply
    assert issues(stack) == []


def test_yes_resumes_and_creates_exactly_one_ticket(stack):
    tid, _ = at_the_gate(stack)
    done = run_turn(stack.graph, tid, "yes")
    assert len(issues(stack)) == 1 and paused(stack, tid) == ()
    assert done.ticket is None and done.pending_offer is None and "t_create" in done.steps
    assert issues(stack)[0].title == "My monitor keeps flickering"


def test_a_second_yes_does_not_create_another(stack):
    tid, _ = at_the_gate(stack)
    run_turn(stack.graph, tid, "yes")
    run_turn(stack.graph, tid, "yes")
    assert len(issues(stack)) == 1


def test_no_cancels_and_creates_nothing(stack):
    tid, _ = at_the_gate(stack)
    done = run_turn(stack.graph, tid, "no")
    assert issues(stack) == [] and paused(stack, tid) == ()
    assert done.ticket is None and "t_cancel" in done.steps and "won't create" in done.reply


def test_the_creation_receives_the_confirmed_draft(stack):
    tid, _ = at_the_gate(stack)
    run_turn(stack.graph, tid, "yes")
    ticket = issues(stack)[0]
    assert f"Employee: {EMAIL}" in ticket.body and "IT" in ticket.labels


# --- edit ----------------------------------------------------------------------------------


def test_an_edit_shows_the_updated_summary_before_anything_is_created(stack):
    tid, _ = at_the_gate(stack)
    edited = run_turn(stack.graph, tid, "make the priority high")
    assert "Priority: High" in edited.reply and "Shall I go ahead and create this?" in edited.reply
    assert issues(stack) == [] and paused(stack, tid) == ("t_confirm",)
    assert edited.steps[-2:] == ["t_apply_edit", "t_present"]


def test_edits_change_the_title_category_and_details(stack):
    tid, _ = at_the_gate(stack)
    run_turn(stack.graph, tid, "change the title to monitor flicker")
    run_turn(stack.graph, tid, "this is for hr")
    out = run_turn(stack.graph, tid, "also it is on the second floor")
    assert "Title: Monitor flicker" in out.reply and "Category: HR" in out.reply
    assert "it is on the second floor" in out.reply
    run_turn(stack.graph, tid, "yes")
    created = issues(stack)[0]
    assert created.title == "Monitor flicker" and "second floor" in created.body and "HR" in created.labels


def test_the_pause_survives_several_turns(stack):
    tid, _ = at_the_gate(stack)
    for text in ("make the priority high", "also it is on the second floor", "hmm", "this is for hr"):
        run_turn(stack.graph, tid, text)
        assert paused(stack, tid) == ("t_confirm",) and issues(stack) == []
    run_turn(stack.graph, tid, "yes")
    assert len(issues(stack)) == 1


def test_the_pause_survives_a_restart(built_index_dir, settings, tmp_path):
    first = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"),
                     drafter_llm=MockDraftModel(), parse_confirm=scripted(EDITS))
    tid, _ = at_the_gate(first)
    first.saver.conn.close()  # the process ends
    second = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"),
                      drafter_llm=MockDraftModel(), parse_confirm=scripted(EDITS), client=first.client)
    assert paused(second, tid) == ("t_confirm",)
    run_turn(second.graph, tid, "yes")
    assert len(issues(second)) == 1


# --- unclear replies -------------------------------------------------------------------------


def test_an_unclear_reply_is_asked_once_more_with_the_same_summary(stack):
    tid, first = at_the_gate(stack)
    again = run_turn(stack.graph, tid, "hmm")
    assert paused(stack, tid) == ("t_confirm",) and issues(stack) == []
    assert first.reply in again.reply and again.reply != first.reply


def test_an_unrelated_reply_counts_as_unclear(stack):
    tid, first = at_the_gate(stack)
    again = run_turn(stack.graph, tid, "what is the weather today")
    assert first.reply in again.reply and paused(stack, tid) == ("t_confirm",)


def test_a_second_unclear_reply_cancels_and_creates_nothing(stack):
    tid, _ = at_the_gate(stack)
    run_turn(stack.graph, tid, "hmm")
    done = run_turn(stack.graph, tid, "what is the weather today")
    assert paused(stack, tid) == () and issues(stack) == []
    assert done.ticket is None and "t_cancel" in done.steps


def test_an_edit_between_two_unclear_replies_resets_the_count(stack):
    tid, _ = at_the_gate(stack)
    run_turn(stack.graph, tid, "hmm")
    run_turn(stack.graph, tid, "make the priority high")
    run_turn(stack.graph, tid, "hmm")
    assert paused(stack, tid) == ("t_confirm",)
    run_turn(stack.graph, tid, "yes")
    assert len(issues(stack)) == 1


def test_an_edit_that_changes_nothing_counts_as_unclear(stack):
    tid, _ = at_the_gate(stack)
    run_turn(stack.graph, tid, "tweak it")
    assert paused(stack, tid) == ("t_confirm",)
    done = run_turn(stack.graph, tid, "tweak it")
    assert issues(stack) == [] and paused(stack, tid) == () and "t_cancel" in done.steps


# --- guards ----------------------------------------------------------------------------------


def test_the_create_node_refuses_to_run_without_a_yes(stack):
    node = make_t_create_node(TicketDeps(drafter=lambda s: None, executor=stack.executor, parse_confirm=rule_parser))
    state = AgentState(email=EMAIL, ticket=TicketDraft(title="t", description="d", category="IT"))
    with pytest.raises(RuntimeError, match="confirm"):
        node(state)
    assert issues(stack) == []


def test_the_history_records_the_replies_in_order(stack):
    tid, _ = at_the_gate(stack)
    out = run_turn(stack.graph, tid, "no")
    roles = [m.type for m in out.messages]
    assert roles.count("human") >= 3 and isinstance(out.messages[-1], AIMessage)
    assert [m.content for m in out.messages if m.type == "human"][-1] == "no"


def test_the_gate_pauses_for_a_ticket_made_by_a_direct_request(built_index_dir, settings, tmp_path):
    s = assemble(built_index_dir, settings, tmp_path, routed=True, drafter_llm=MockDraftModel(),
                 parse_confirm=scripted(EDITS))
    tid = new_thread_id()
    state = run_turn(s.graph, tid, "Please raise a ticket for my broken VPN", email=EMAIL)
    assert "Shall I go ahead" in state.reply and paused(s, tid) == ("t_confirm",)
    run_turn(s.graph, tid, "yes")
    assert len(issues(s)) == 1


def test_apply_edit_reports_whether_anything_changed():
    draft = TicketDraft(title="t", description="d", category="IT")
    changed, did = apply_edit(draft, ConfirmDecision(decision="edit", priority="low"))
    assert did and changed.priority == "low"
    same, did = apply_edit(draft, ConfirmDecision(decision="edit"))
    assert not did and same == draft
