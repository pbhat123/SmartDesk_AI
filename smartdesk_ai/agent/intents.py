"""Intent classifier (FR-D1, FR-D2).

One structured Main-Model call (task ORCHESTRATOR) returns an intent, a confidence and, for
kb_query, a domain. The dialog state `awaiting` goes into the prompt so a bare "yes" or "the
second one" is routed correctly. Routing on top of this (confidence threshold, tool-pick
fallback) lives in `router.py` (FR-D8).
"""
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.agent.state import AgentState, Awaiting, Category, Intent, IntentResult, SubQuestion
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.factory import ModelBuilder, openai_builder
from smartdesk_ai.llm.fallback import FallbackChat
from smartdesk_ai.llm.structured import ReplyError, parse_model_reply

Route = Literal["it_worker", "hr_worker", "misc", "ticket_agent", "none"]

# The intent that continues each pending flow; any other firm intent is a topic change (FR-D2).
_CONTINUES: dict[str, str] = {
    "offer_reply": "create_ticket",
    "email": "create_ticket",
    "ticket_pick": "ticket_status",
}


class ClassifierError(Exception):
    """The model call failed or its answer was not a valid classification."""


def build_messages(prompt: str, text: str, awaiting: Awaiting, last_topic: str | None, history: str = "") -> list[Any]:
    """The classifier's input. `history` is the earlier conversation, masked (FR-I3); the block is left out when empty."""
    earlier = f"Conversation so far:\n{history}\n" if history else ""
    human = f"Awaiting: {awaiting}\nLast topic: {last_topic or 'none'}\n{earlier}Message: {text}"
    return [SystemMessage(prompt), HumanMessage(human)]


class PartReply(BaseModel):
    """One sub-question as the model returns it (FR-I3, FR-I4)."""

    model_config = ConfigDict(extra="forbid")

    text: str
    domain: Category
    follow_up: bool = False  # true when the question only makes sense with the earlier turns, so it was rewritten


class ClassifierReply(BaseModel):
    """The exact shape the classifier must return (FR-H2). Anything else is invalid output."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    domain: Category | None = None
    parts: list[PartReply] = Field(default_factory=list)


def _to_result(reply: ClassifierReply) -> IntentResult:
    kb = reply.intent == "kb_query"
    parts = [SubQuestion(part_id=f"p{i}", text=p.text, domain=p.domain, rewritten=p.follow_up)
             for i, p in enumerate(reply.parts, 1)] if kb else []
    return IntentResult(intent=reply.intent, confidence=reply.confidence, domain=reply.domain if kb else None, parts=parts)


def _parse(content: Any) -> IntentResult:
    try:
        return _to_result(parse_model_reply(ClassifierReply, content))
    except ReplyError as error:
        raise ClassifierError("classifier reply is not a valid classification") from error


def classify_intent(
    llm: Any,
    prompt: str,
    text: str,
    awaiting: Awaiting = "none",
    last_topic: str | None = None,
    history: str = "",
) -> IntentResult:
    """Classify one message, given the earlier conversation as masked text. Any failure is a `ClassifierError`, never a
    crash (FR-D8)."""
    try:
        response = llm.invoke(build_messages(prompt, text, awaiting, last_topic, history))
    except Exception as error:  # network, rate limit, timeout: the router falls back
        raise ClassifierError(f"classifier call failed: {type(error).__name__}") from error
    return _parse(getattr(response, "content", response))


def route_for(result: IntentResult) -> Route:
    """The agent route for a classification (kb_query without a domain searches the whole KB)."""
    if result.intent == "kb_query":
        return {"IT": "it_worker", "HR": "hr_worker"}.get(result.domain or "", "misc")  # type: ignore[return-value]
    if result.intent in ("create_ticket", "ticket_status"):
        return "ticket_agent"
    return "none"


def dialog_update_for(state: AgentState, intent: str | None) -> dict[str, Any]:
    """State changes after an intent is known: a new topic mid-flow clears the pending offer (FR-D2).

    A partly built ticket draft goes with it; the session email stays. Continuing the flow, an
    unclear reply, or no intent at all (a chitchat tool pick) keeps the state as it is.
    """
    if state.awaiting == "none" or intent is None or intent == "ambiguous":
        return {}
    if intent == _CONTINUES[state.awaiting]:
        return {}
    return {"awaiting": "none", "pending_offer": None, "ticket": None}


def dialog_update(state: AgentState, result: IntentResult) -> dict[str, Any]:
    return dialog_update_for(state, result.intent)


def make_classifier(
    settings: Settings,
    *,
    builder: ModelBuilder = openai_builder,
    budget_level: int = 0,
    structured: bool | None = None,
) -> Callable[..., IntentResult]:
    """A classifier bound to the ORCHESTRATOR model and the active orchestrator prompt.

    The live client (the default builder) uses the model's structured-output mode with
    `ClassifierReply` (through `FallbackChat`, so each call is retried, budget-gated and observed); a custom builder, i.e. the mock path, gets plain JSON unless
    `structured=True` is passed.
    """
    use_schema = structured if structured is not None else builder is openai_builder
    # Through FallbackChat like every other task: retry, Main/Light fallback, the budget gate, cost, trace and records.
    llm = FallbackChat("ORCHESTRATOR", settings, builder=builder, budget_level=budget_level,
                       schema=ClassifierReply if use_schema else None)
    prompt = load_prompt("orchestrator", settings).text

    def classify(text: str, awaiting: Awaiting = "none", last_topic: str | None = None, history: str = "") -> IntentResult:
        return classify_intent(llm, prompt, text, awaiting, last_topic, history)

    return classify
