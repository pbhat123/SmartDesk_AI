"""T1.9: intent eval set (FR-D1, D2, D8, D9, G2)."""
import re
from collections import Counter

import pytest

from evals.common import EVALS_DIR, load_jsonl
from evals.schemas import IntentRow

PATH = EVALS_DIR / "intents.jsonl"
ALL_INTENTS = {"kb_query", "create_ticket", "ticket_status", "smalltalk", "out_of_scope", "ambiguous"}


@pytest.fixture(scope="module")
def rows():
    return [IntentRow(**r) for r in load_jsonl(PATH)]


def test_at_least_30_rows_with_unique_ids(rows):
    assert len(rows) >= 30
    assert len({r.id for r in rows}) == len(rows)
    assert len({(r.utterance.lower(), r.awaiting) for r in rows}) == len(rows)


def test_all_six_intents_appear(rows):
    counts = Counter(r.expected_intent for r in rows)
    assert set(counts) == ALL_INTENTS
    assert counts["ambiguous"] >= 3 and counts["out_of_scope"] >= 4


def test_route_and_tool_agree_with_intent(rows):
    for r in rows:
        if r.expected_intent == "kb_query":
            assert r.expected_route in {"it_worker", "hr_worker", "misc"} and r.expected_tool == "search_kb", r.id
        elif r.expected_intent == "create_ticket":
            assert (r.expected_route, r.expected_tool) == ("ticket_agent", "create_ticket"), r.id
        elif r.expected_intent == "ticket_status":
            assert (r.expected_route, r.expected_tool) == ("ticket_agent", "get_tickets"), r.id
        else:  # smalltalk, out_of_scope, ambiguous: reply directly, no tool (FR-G2)
            assert (r.expected_route, r.expected_tool) == ("none", "none"), r.id


def test_all_three_kb_routes_are_exercised(rows):
    kb_routes = {r.expected_route for r in rows if r.expected_intent == "kb_query"}
    assert kb_routes == {"it_worker", "hr_worker", "misc"}


def test_out_of_scope_rows_cover_weather_recipes_trivia_and_coding(rows):
    kinds = {r.out_of_scope_kind for r in rows if r.expected_intent == "out_of_scope"}
    assert {"weather", "recipe", "trivia", "coding"} <= kinds
    assert all(r.out_of_scope_kind for r in rows if r.expected_intent == "out_of_scope")
    assert not any(r.out_of_scope_kind for r in rows if r.expected_intent != "out_of_scope")


def test_dialog_state_rows_exercise_awaiting(rows):
    awaiting = {r.awaiting for r in rows}
    assert {"offer_reply", "email", "ticket_pick"} <= awaiting
    topic_change = [r for r in rows if r.awaiting != "none" and r.expected_intent == "kb_query"]
    assert topic_change, "need a topic change while a flow is pending (FR-D2)"


def test_workplace_gap_topics_are_in_scope_not_deflected(rows):
    gap_rows = [r for r in rows if re.search(r"\b(parking|cafeteria|desk)\b", r.utterance.lower())]
    assert gap_rows
    assert all(r.expected_intent == "kb_query" and r.expected_route == "misc" for r in gap_rows)
