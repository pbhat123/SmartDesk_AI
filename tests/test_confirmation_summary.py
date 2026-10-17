"""T4.15: the confirmation summary shows every field and asks for an explicit yes (FR-B8, FR-B2)."""
import pytest

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import TicketDraft, new_thread_id
from smartdesk_ai.agent.ticket_agent import CONFIRM_QUESTION, confirmation_summary
from smartdesk_ai.llm.mock import MockDraftModel
from tests.agent_helpers import build_test_graph
from tests.llm_fakes import FakeAnswerLLM

EMAIL = "priya.k@smartdesk.example"
DRAFT = TicketDraft(
    title="Unable to connect to VPN from home",
    description="The VPN drops every ten minutes.\nIt started after the update.",
    category="IT",
    priority="high",
)


def test_the_summary_contains_title_description_category_email_and_priority():
    text = confirmation_summary(DRAFT, EMAIL)
    for expected in (
        "Title: Unable to connect to VPN from home",
        "The VPN drops every ten minutes.",
        "It started after the update.",
        "Category: IT",
        f"Email: {EMAIL}",
        "Priority: High",
    ):
        assert expected in text


def test_priority_is_left_out_when_it_is_not_set():
    text = confirmation_summary(DRAFT.model_copy(update={"priority": None}), EMAIL)
    assert "Priority" not in text and "Category: IT" in text


@pytest.mark.parametrize("level, shown", [("low", "Low"), ("medium", "Medium"), ("high", "High"), ("critical", "Critical")])
def test_each_priority_is_shown_in_words(level, shown):
    assert f"Priority: {shown}" in confirmation_summary(DRAFT.model_copy(update={"priority": level}), EMAIL)


def test_it_asks_for_an_explicit_confirmation_with_one_question():
    text = confirmation_summary(DRAFT, EMAIL)
    assert text.rstrip().endswith("Shall I go ahead and create this?")
    assert text.count("?") == 1 and CONFIRM_QUESTION in text


def test_it_never_asks_about_the_priority():
    text = confirmation_summary(DRAFT, EMAIL)
    question = text.rstrip().splitlines()[-1]
    assert "priority" not in question.lower()


def test_it_says_nothing_the_draft_does_not_say():
    lines = [line for line in confirmation_summary(DRAFT, EMAIL).splitlines() if line.strip()]
    labelled = {line.split(": ", 1)[0] for line in lines if ": " in line}
    assert {"Title", "Category", "Priority", "Email"} <= labelled
    assert not any(token in " ".join(lines) for token in ("#", "http", "ticket number"))


def test_the_graph_shows_the_summary_after_drafting(built_index_dir, settings, tmp_path):
    graph, _ = build_test_graph(
        built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"), drafter_llm=MockDraftModel()
    )
    tid = new_thread_id()
    run_turn(graph, tid, "I'm locked out and cannot work", email=EMAIL)
    out = run_turn(graph, tid, "yes")
    assert out.reply == confirmation_summary(out.ticket, EMAIL)
    assert "Priority: High" in out.reply and "Shall I go ahead and create this?" in out.reply
    assert out.steps[-2:] == ["t_draft", "t_present"]
    assert out.messages[-1].content == out.reply


def test_no_ticket_is_created_by_showing_the_summary(built_index_dir, settings, tmp_path):
    graph, _ = build_test_graph(
        built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"), drafter_llm=MockDraftModel()
    )
    tid = new_thread_id()
    run_turn(graph, tid, "My monitor keeps flickering", email=EMAIL)
    out = run_turn(graph, tid, "yes")
    assert "t_create" not in out.steps
