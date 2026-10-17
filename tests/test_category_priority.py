"""T4.14: category (IT, HR, MISC) and priority (low, medium, high, critical) are inferred from the
conversation; priority is never asked (FR-B7, FR-D7). Mock LLM, no network."""
import json
from types import SimpleNamespace

import pytest

from evals.common import EVALS_DIR, load_jsonl
from evals.schemas import CategoryRow
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import TicketDraft, new_thread_id
from smartdesk_ai.agent.ticket_agent import DraftReply, draft_from_statements, make_drafter
from smartdesk_ai.llm.mock import MockDraftModel, infer_mock_priority
from tests.agent_helpers import build_test_graph
from tests.llm_fakes import FakeAnswerLLM

ROWS = [CategoryRow(**r) for r in load_jsonl(EVALS_DIR / "categories.jsonl")]


class Fixed:
    """A drafting model that returns a given category and priority for whatever it is shown."""

    def __init__(self, category="IT", priority=None, **over):
        self.reply = {"title": "Monitor flickering", "description": "My monitor keeps flickering",
                      "category": category, "priority": priority, **over}

    def invoke(self, messages):
        return SimpleNamespace(content=json.dumps(self.reply))


def drafted(llm, text="My monitor keeps flickering"):
    return draft_from_statements(llm, "prompt", [text])


@pytest.mark.parametrize("row", ROWS, ids=[r.id for r in ROWS])
def test_the_category_from_the_model_reaches_the_draft(row):
    out = drafted(Fixed(category=row.expected_category, title="x", description=row.utterance), row.utterance)
    assert out.category == row.expected_category


@pytest.mark.parametrize("row", ROWS, ids=[r.id for r in ROWS])
def test_the_mock_drafter_infers_the_eval_set_category(row):
    assert drafted(MockDraftModel(), row.utterance).category == row.expected_category


def test_the_eval_set_covers_all_three_categories():
    assert {r.expected_category for r in ROWS} == {"IT", "HR", "MISC"}


@pytest.mark.parametrize("value", ["low", "medium", "high", "critical"])
def test_the_four_priorities_are_accepted(value):
    assert drafted(Fixed(priority=value)).priority == value


@pytest.mark.parametrize("given, expected", [("High", "high"), ("CRITICAL", "critical"), (" Low ", "low")])
def test_priority_case_is_normalised(given, expected):
    assert drafted(Fixed(priority=given)).priority == expected


@pytest.mark.parametrize("given, expected", [("it", "IT"), ("Hr", "HR"), ("misc", "MISC")])
def test_category_case_is_normalised(given, expected):
    assert drafted(Fixed(category=given)).category == expected


@pytest.mark.parametrize("bad", ["urgent", "P1", "severe", "normal", ""])
def test_any_other_priority_is_rejected_by_the_reply_model(bad):
    with pytest.raises(ValueError):
        DraftReply(title="t", description="d", category="IT", priority=bad)


@pytest.mark.parametrize("bad", ["FACILITIES", "Finance", ""])
def test_any_other_category_is_rejected_by_the_reply_model(bad):
    with pytest.raises(ValueError):
        DraftReply(title="t", description="d", category=bad, priority=None)


def test_a_reply_with_an_invalid_priority_is_not_used_and_falls_back():
    out = drafted(Fixed(category="IT", priority="urgent"))
    assert out.priority is None and out.category == "MISC"  # the plain draft from the employee's words


def test_the_draft_type_itself_only_accepts_the_four_values():
    with pytest.raises(ValueError):
        TicketDraft(title="t", description="d", category="IT", priority="urgent")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Nobody can log in to the system", "critical"),
        ("My laptop was stolen last night", "critical"),
        ("I'm locked out and cannot work", "high"),
        ("The VPN keeps dropping every ten minutes", "medium"),
        ("How do I set up MFA on my new phone?", None),
    ],
)
def test_the_mock_priority_follows_the_stated_impact(text, expected):
    assert infer_mock_priority(text) == expected


def test_priority_is_optional_so_a_draft_without_one_is_valid():
    assert drafted(Fixed(priority=None)).priority is None


def test_the_draft_prompt_tells_the_model_to_infer_not_ask():
    from smartdesk_ai.agent.prompt_loader import load_prompt
    from smartdesk_ai.config.settings import load_settings

    text = load_prompt("ticket_summary", load_settings()).text.lower()
    assert "never ask" in text and "infer" in text


def test_no_message_in_the_offer_to_draft_flow_asks_about_priority(built_index_dir, settings, tmp_path):
    graph, _ = build_test_graph(
        built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"), drafter_llm=MockDraftModel()
    )
    tid = new_thread_id()
    run_turn(graph, tid, "I'm locked out and cannot work")
    run_turn(graph, tid, "yes")
    out = run_turn(graph, tid, "priya.k@smartdesk.example")
    assistant = [m.content.lower() for m in out.messages if m.type == "ai"]
    assert out.ticket is not None and out.ticket.priority == "high"
    assert assistant and not any("priority" in text for text in assistant)
    assert not any("?" in text and "urgent" in text for text in assistant)


def test_make_drafter_passes_the_inferred_values_through(settings):
    drafter = make_drafter(settings, builder=lambda model_id, cfg: MockDraftModel())
    out = drafter(["The lift near reception has been out of order since Monday"])
    assert out.category == "MISC"
