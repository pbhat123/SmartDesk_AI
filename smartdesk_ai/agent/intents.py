"""Intent classifier (FR-D1, FR-D2).

One structured Main-Model call (task ORCHESTRATOR) returns an intent, a confidence and, for
kb_query, a domain. The dialog state `awaiting` goes into the prompt so a bare "yes" or "the
second one" is routed correctly. Routing on top of this (confidence threshold, tool-pick
fallback) lives in `router.py` (FR-D8).
"""
import json
import re
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.agent.state import AgentState, Awaiting, IntentResult
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.factory import ModelBuilder, get_llm, openai_builder

Route = Literal["it_worker", "hr_worker", "misc", "ticket_agent", "none"]

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)
# The intent that continues each pending flow; any other firm intent is a topic change (FR-D2).
_CONTINUES: dict[str, str] = {
    "offer_reply": "create_ticket",
    "email": "create_ticket",
    "ticket_pick": "ticket_status",
}


class ClassifierError(Exception):
    """The model call failed or its answer was not a valid classification."""


def build_messages(prompt: str, text: str, awaiting: Awaiting, last_topic: str | None) -> list[Any]:
    human = f"Awaiting: {awaiting}\nLast topic: {last_topic or 'none'}\nMessage: {text}"
    return [SystemMessage(prompt), HumanMessage(human)]


def _parse(content: Any) -> IntentResult:
    raw = content if isinstance(content, str) else str(content)
    raw = raw.strip()
    fenced = _FENCE.match(raw)
    if fenced:
        raw = fenced.group(1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ClassifierError("classifier did not return JSON") from error
    if not isinstance(data, dict):
        raise ClassifierError("classifier JSON is not an object")
    if isinstance(data.get("confidence"), bool) or not isinstance(data.get("confidence"), int | float):
        raise ClassifierError("classifier confidence is missing or not a number")
    try:
        result = IntentResult(
            intent=data.get("intent"),
            confidence=data["confidence"],
            domain=data.get("domain"),
        )
    except ValidationError as error:
        raise ClassifierError("classifier returned an invalid intent, confidence or domain") from error
    if result.intent != "kb_query" and result.domain is not None:
        result = result.model_copy(update={"domain": None})
    return result


def classify_intent(
    llm: Any,
    prompt: str,
    text: str,
    awaiting: Awaiting = "none",
    last_topic: str | None = None,
) -> IntentResult:
    """Classify one message. Any failure is a `ClassifierError`, never a crash (FR-D8)."""
    try:
        response = llm.invoke(build_messages(prompt, text, awaiting, last_topic))
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


def dialog_update(state: AgentState, result: IntentResult) -> dict[str, Any]:
    """State changes after classifying: a new topic mid-flow clears the pending offer (FR-D2).

    Continuing the flow, or an unclear reply, keeps the state as it is.
    """
    if state.awaiting == "none" or result.intent == "ambiguous":
        return {}
    if result.intent == _CONTINUES[state.awaiting]:
        return {}
    return {"awaiting": "none", "pending_offer": None}


def make_classifier(
    settings: Settings, *, builder: ModelBuilder = openai_builder, budget_level: int = 0
) -> Callable[..., IntentResult]:
    """A classifier bound to the ORCHESTRATOR model and the active orchestrator prompt."""
    llm = get_llm("ORCHESTRATOR", settings, budget_level=budget_level, builder=builder).model
    prompt = load_prompt("orchestrator", settings).text

    def classify(text: str, awaiting: Awaiting = "none", last_topic: str | None = None) -> IntentResult:
        return classify_intent(llm, prompt, text, awaiting, last_topic)

    return classify
