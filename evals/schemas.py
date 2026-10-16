"""Pydantic row schemas for the eval data sets (SPEC section 7)."""
from typing import Literal

from pydantic import BaseModel, Field

Split = Literal["dev", "test"]
Domain = Literal["IT", "HR"]
Category = Literal["IT", "HR", "MISC"]
QuestionType = Literal["simple", "how-to", "policy interpretation"]


class KBQuestion(BaseModel):
    id: str
    question: str = Field(min_length=10)
    expected_entry_id: str
    expected_domain: Domain
    type: QuestionType
    reference_answer: str = Field(min_length=20)
    split: Split


class GapQuestion(BaseModel):
    id: str
    question: str = Field(min_length=10)
    gap_topic: str  # a section title in kb/GAPS.md
    expected_category: Category
    grey_zone: bool  # workplace-related and close to existing KB wording
    split: Split


Intent = Literal["kb_query", "create_ticket", "ticket_status", "smalltalk", "out_of_scope", "ambiguous"]
Route = Literal["it_worker", "hr_worker", "misc", "ticket_agent", "none"]
Tool = Literal["search_kb", "create_ticket", "get_tickets", "none"]
Awaiting = Literal["none", "offer_reply", "email", "ticket_pick"]


class IntentRow(BaseModel):
    """One routing case (FR-D1, D2, D8, D9, G2). `route` and `tool` are the expected outcomes."""

    id: str
    utterance: str = Field(min_length=2)
    awaiting: Awaiting = "none"
    expected_intent: Intent
    expected_route: Route
    expected_tool: Tool
    out_of_scope_kind: Literal["weather", "recipe", "trivia", "coding", "other"] | None = None


class CategoryRow(BaseModel):
    """Expected ticket/domain category for an utterance (FR-B7, FR-D7)."""

    id: str
    utterance: str = Field(min_length=5)
    expected_category: Category


class CompoundPart(BaseModel):
    sub_question: str = Field(min_length=8)
    expected_domain: Category
    answerable: bool
    expected_entry_id: str | None = None  # set when answerable
    gap_topic: str | None = None  # set when not answerable (title in kb/GAPS.md)
    likely_wrong_domain: Category | None = None  # set when the wording invites a misroute


class CompoundRow(BaseModel):
    """A multi-part question for the fan-out and synthesis flow (FR-I4 to FR-I7)."""

    id: str
    question: str = Field(min_length=20)
    parts: list[CompoundPart] = Field(min_length=2)
