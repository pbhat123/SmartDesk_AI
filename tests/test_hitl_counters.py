"""T6.7: human-in-the-loop counters at the ticket approval gate (FR-H4, FR-G12). The gate's final decisions are
counted: approvals (yes), rejections (no) and edits. An unclear reply is not a decision, and nothing but the
decision and a timestamp is stored (FR-G11)."""
import json

import pytest

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.ops.hitl_counters import HitlCounters, HitlCounts, load_counts
from smartdesk_ai.ui.monitoring import load_monitoring
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM
from tests.test_approval_gate import EDITS, EMAIL, ISSUE, scripted
from tests.ui_helpers import monitoring_app, element_text, mock_context
from smartdesk_ai.llm.mock import MockDraftModel


# --- the counter itself ---------------------------------------------------------------------------

def test_decisions_increment_their_own_counter():
    counters = HitlCounters(None)
    for decision in ("yes", "yes", "no", "edit"):
        counters.record(decision)
    assert counters.counts() == HitlCounts(approvals=2, rejections=1, edits=1)
    assert counters.counts().total == 4


def test_only_decisions_are_counted():
    counters = HitlCounters(None)
    with pytest.raises(ValueError):
        counters.record("unclear")
    assert counters.counts() == HitlCounts()


def test_counts_survive_a_restart_and_are_shared_by_every_reader(tmp_path):
    path = tmp_path / "hitl.jsonl"
    HitlCounters(path).record("yes")
    HitlCounters(path).record("no")
    assert HitlCounters(path).counts() == HitlCounts(approvals=1, rejections=1, edits=0)
    assert load_counts(path) == HitlCounts(approvals=1, rejections=1, edits=0)


def test_a_missing_file_is_zero_and_reading_creates_nothing(tmp_path):
    assert load_counts(tmp_path / "nope.jsonl") == HitlCounts()
    assert list(tmp_path.iterdir()) == []


def test_a_half_written_or_foreign_line_is_skipped(tmp_path):
    path = tmp_path / "hitl.jsonl"
    HitlCounters(path).record("edit")
    with path.open("a") as f:
        f.write('{"ts": "t", "decis\n{"ts": "t", "decision": "maybe"}\n')
    assert load_counts(path) == HitlCounts(edits=1)


def test_a_record_holds_the_decision_and_a_timestamp_only(tmp_path):
    path = tmp_path / "hitl.jsonl"
    HitlCounters(path).record("yes")
    (row,) = [json.loads(line) for line in path.read_text().splitlines()]
    assert set(row) == {"ts", "decision"}


# --- at the gate, in the graph --------------------------------------------------------------------

@pytest.fixture
def gate(built_index_dir, settings, tmp_path):
    counters = HitlCounters(tmp_path / "hitl.jsonl")
    stack = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"),
                     drafter_llm=MockDraftModel(), parse_confirm=scripted(EDITS), hitl=counters)
    return stack, counters


def at_the_gate(stack):
    tid = new_thread_id()
    run_turn(stack.graph, tid, ISSUE, email=EMAIL)
    run_turn(stack.graph, tid, "yes")
    return tid


def test_yes_at_the_gate_is_an_approval(gate):
    stack, counters = gate
    run_turn(stack.graph, at_the_gate(stack), "yes")
    assert counters.counts() == HitlCounts(approvals=1)


def test_no_at_the_gate_is_a_rejection(gate):
    stack, counters = gate
    run_turn(stack.graph, at_the_gate(stack), "no")
    assert counters.counts() == HitlCounts(rejections=1)


def test_an_edit_then_a_yes_counts_both(gate):
    stack, counters = gate
    tid = at_the_gate(stack)
    run_turn(stack.graph, tid, "make the priority high")
    assert counters.counts() == HitlCounts(edits=1)
    run_turn(stack.graph, tid, "yes")
    assert counters.counts() == HitlCounts(approvals=1, edits=1)


def test_unclear_replies_and_a_give_up_are_not_decisions(gate):
    stack, counters = gate
    tid = at_the_gate(stack)
    run_turn(stack.graph, tid, "hmm")
    run_turn(stack.graph, tid, "tweak it")  # an edit that changes nothing is unclear (FR-G12)
    assert counters.counts() == HitlCounts()  # the second unclear reply cancelled the ticket: still no decision


def test_the_approval_is_counted_once_even_when_the_gate_asked_again(gate):
    stack, counters = gate
    tid = at_the_gate(stack)
    run_turn(stack.graph, tid, "hmm")
    run_turn(stack.graph, tid, "yes")
    assert counters.counts() == HitlCounts(approvals=1)


def test_a_stack_without_counters_behaves_as_before(built_index_dir, settings, tmp_path):
    stack = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"),
                     drafter_llm=MockDraftModel(), parse_confirm=scripted(EDITS))
    run_turn(stack.graph, at_the_gate(stack), "yes")
    assert len(stack.client._issues) == 1


# --- through the whole app, and in the tab ---------------------------------------------------------

def _flow(ctx, session, *turns):
    for text in turns:
        ctx.service.handle(session, text)


def test_scripted_yes_no_edit_runs_through_the_app_show_in_the_tab(tmp_path):
    ctx = mock_context(tmp_path)
    start = ("My monitor keeps flickering", EMAIL, "yes")  # ends at the approval gate
    _flow(ctx, "a", *start, "yes")
    _flow(ctx, "b", *start, "no")
    _flow(ctx, "c", *start, "Make the priority high", "yes")
    data = load_monitoring(ctx.settings, ctx.data_dir, ctx.kb_dir, ctx.registry, results_dir=ctx.results_dir)
    assert data.hitl == HitlCounts(approvals=2, rejections=1, edits=1)
    assert EMAIL not in (tmp_path / "hitl.jsonl").read_text()
    tab = monitoring_app(ctx).main
    metrics = {m.label: m.value for m in tab.metric}
    assert metrics["Approvals"] == "2" and metrics["Rejections"] == "1" and metrics["Edits"] == "1"
    assert "approval gate" in element_text(tab)


def test_the_tab_shows_zeros_before_any_ticket_flow(tmp_path):
    tab = monitoring_app(mock_context(tmp_path)).main
    metrics = {m.label: m.value for m in tab.metric}
    assert metrics["Approvals"] == metrics["Rejections"] == metrics["Edits"] == "0"


# --- UI review fix 6: the counters in a browser-style run -----------------------------------------------------

def _turns(ctx, session, *texts):
    return [ctx.service.handle(session, t) for t in texts]


def test_a_title_change_at_the_gate_counts_as_one_edit_and_the_yes_after_it_as_the_approval(tmp_path):
    ctx = mock_context(tmp_path)
    replies = _turns(ctx, "a", "My laptop screen is cracked, please raise a ticket", EMAIL,
                     "change the title to Cracked laptop screen", "yes")
    assert "Title: Cracked laptop screen" in replies[2].text and "Ticket #" in replies[3].text
    assert load_counts(tmp_path / "hitl.jsonl") == HitlCounts(approvals=1, edits=1)


def test_the_same_words_after_the_gate_closed_are_a_new_question_and_count_nothing(tmp_path):
    """This is what the 2026-10-18 browser test did: 'no' (1 rejection), then the title change AFTER the cancel,
    which is no longer an edit of anything, then a yes to a new ticket offer (1 approval). Edits stayed at 0."""
    ctx = mock_context(tmp_path)
    _turns(ctx, "a", "My laptop screen is cracked, please raise a ticket", EMAIL, "no")
    assert load_counts(tmp_path / "hitl.jsonl") == HitlCounts(rejections=1)
    reply = ctx.service.handle("a", "change the title to Cracked laptop screen")
    assert "t_confirm" not in reply.debug["steps"]  # no gate was open
    assert load_counts(tmp_path / "hitl.jsonl") == HitlCounts(rejections=1)


def test_a_full_browser_style_run_reads_one_rejection_one_edit_one_approval(tmp_path):
    ctx = mock_context(tmp_path)
    start = ("My laptop screen is cracked, please raise a ticket", EMAIL)
    _turns(ctx, "r", *start, "no")
    _turns(ctx, "e", *start, "change the title to Cracked laptop screen", "yes")  # the edit, then its approval
    counts = load_counts(tmp_path / "hitl.jsonl")
    assert (counts.approvals, counts.rejections, counts.edits) == (1, 1, 1)
    metrics = {m.label: m.value for m in monitoring_app(ctx).main.metric}
    assert (metrics["Approvals"], metrics["Rejections"], metrics["Edits"]) == ("1", "1", "1")


def test_an_edit_is_counted_when_it_happens_even_if_the_employee_never_approves(tmp_path):
    ctx = mock_context(tmp_path)
    _turns(ctx, "a", "My laptop screen is cracked, please raise a ticket", EMAIL)
    ctx.service.handle("a", "change the title to Cracked laptop screen")
    assert load_counts(tmp_path / "hitl.jsonl") == HitlCounts(edits=1)  # no yes, no no: the edit is already there


def test_an_approval_is_counted_at_the_gate_even_if_creating_the_ticket_then_fails(tmp_path):
    from smartdesk_ai.tickets import MockTicketClient

    client = MockTicketClient()
    ctx = mock_context(tmp_path, client=client)
    _turns(ctx, "a", "My laptop screen is cracked, please raise a ticket", EMAIL)
    client.unreachable = True
    reply = ctx.service.handle("a", "yes")
    assert "try again later" in reply.text
    assert load_counts(tmp_path / "hitl.jsonl") == HitlCounts(approvals=1)  # the decision was made; the failure is separate
