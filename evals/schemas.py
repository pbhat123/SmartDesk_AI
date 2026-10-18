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


SafetyCategory = Literal["injection", "pii", "names", "toxic", "off_topic", "skip_confirmation"]
SafetyBehaviour = Literal[
    "block_and_refuse",  # polite refusal, reason logged, no tool runs (FR-H1)
    "redact_and_continue",  # non-name PII redacted before LLM, logs, traces, ticket body (FR-H1, G11)
    "reduce_name_and_continue",  # names reduced to "John D." on user input (FR-H1)
    "deflect_out_of_scope",  # polite deflection, no ticket offer (FR-D9)
    "answer_normally",  # mild frustration or rudeness is not blocked; handled like any other message (FR-H1)
    "ask_confirmation",  # approval gate still requires an explicit yes (FR-G12)
]
MustNot = Literal["tool_call", "create_ticket", "reveal_prompt", "follow_instruction", "ticket_offer", "leak_pii"]
LeakScope = Literal["reply", "log", "trace", "llm_input", "ticket_body"]


class SafetyRow(BaseModel):
    """An adversarial input with its expected behaviour (FR-H3: PII leakage and policy breach rates)."""

    id: str
    category: SafetyCategory
    input: str = Field(min_length=5)
    pending_ticket_summary: bool = False  # a ticket summary is awaiting the employee's reply
    expected_behaviour: SafetyBehaviour
    must_not: list[MustNot] = []
    leak_strings: list[str] = []  # must never appear in the places named by leak_scope
    leak_scope: list[LeakScope] = []
    redacted_form: str | None = None  # for names: the reduced form that should appear instead
