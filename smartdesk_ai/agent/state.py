"""Graph state for one chat session (FR-D2, FR-G12; PLAN 4.1).

Everything here is written to the checkpoint database, so only redacted text may enter it.
Each Pydantic model must be listed in `STATE_MODELS`, which the checkpointer allow-lists
(PLAN 12.1). The thread ID is the only value that goes in `config["configurable"]`.
"""
import uuid
from typing import Annotated, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

Intent = Literal["kb_query", "create_ticket", "ticket_status", "smalltalk", "out_of_scope", "ambiguous"]
Awaiting = Literal["none", "offer_reply", "email", "ticket_pick"]
RouteSource = Literal["classifier", "tool_pick"]
Category = Literal["IT", "HR", "MISC"]
Priority = Literal["low", "medium", "high", "critical"]
Domain = Literal["IT", "HR"]
RouteKind = Literal["kb", "ticket", "status", "smalltalk", "deflect", "clarify"]


class IntentResult(BaseModel):
    """What the classifier decided (FR-D1, D8). `domain` is only set for kb_query."""

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    route_source: RouteSource = "classifier"
    domain: Category | None = None


class PendingOffer(BaseModel):
    """A ticket was offered after a KB gap and the employee has not answered yet."""

    unanswered_text: str


class TicketDraft(BaseModel):
    """The ticket being collected. The email lives on the state, not here (FR-B13)."""

    title: str
    description: str
    category: Category
    priority: Priority | None = None


class ConfirmDecision(BaseModel):
    """What the employee's reply at the approval gate means (FR-G12). An `edit` carries the fields
    to change; `details` are added to the description in the employee's words (FR-B6)."""

    decision: Literal["yes", "no", "edit", "unclear"]
    title: str | None = None
    description: str | None = None
    details: str | None = None
    category: Category | None = None
    priority: Priority | None = None


class ChunkRef(BaseModel):
    """A retrieved KB entry as shown in the debug view: id, title and score, never the text."""

    entry_id: str
    title: str
    score: float


class PartResult(BaseModel):
    """Outcome of answering one question from the KB (PLAN 4.1)."""

    answer: str  # what the employee sees, with sources; the I-don't-know text when unanswered
    citations: list[str] = []
    chunks: list[ChunkRef] = []
    answerable: bool  # retrieval signals 1 and 2 passed
    llm_insufficient: bool = False  # signal 3
    retried: bool = False  # the whole-KB retry ran (FR-A19)
    searched_domain: Domain | None = None  # filter used by the last search; None is the whole KB


class AgentState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    messages: Annotated[list[AnyMessage], add_messages] = []
    email: str | None = None
    last_topic: str | None = None
    awaiting: Awaiting = "none"
    intent: IntentResult | None = None
    pending_offer: PendingOffer | None = None
    ticket: TicketDraft | None = None
    route_kind: RouteKind | None = None  # where routing sent this turn
    ticket_context_start: int = 0  # index in `messages` where the current ticket's conversation begins
    gate: ConfirmDecision | None = None  # the last reply at the approval gate
    gate_unclear: int = 0  # unclear replies at the current gate; the second one cancels (FR-G12)
    search_domain: Domain | None = None  # KB filter chosen by routing for this turn
    part_result: PartResult | None = None
    steps: list[str] = []  # this turn's trace, e.g. ["search_kb", "answer", "offer"] (FR-H11)
    reply: str = ""


STATE_MODELS: tuple[type[BaseModel], ...] = (
    IntentResult, PendingOffer, TicketDraft, ConfirmDecision, ChunkRef, PartResult, AgentState,
)


def new_thread_id() -> str:
    """A fresh opaque ID per chat session; never derived from an email (FR-G11)."""
    return uuid.uuid4().hex


def thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    """The only LangGraph config this app passes: the thread ID and nothing else."""
    if not thread_id.strip() or "@" in thread_id:
        raise ValueError("thread_id must be a non-empty opaque ID, never an email")
    return {"configurable": {"thread_id": thread_id}}
