"""Routing (FR-D8, FR-G2) and the scope boundary (FR-D9).

Order: the intent classifier decides. Only if it errors, returns invalid output, or is below
the routing threshold does the route come from the model's tool pick. A confident `ambiguous`
is a valid result: one clarifying question, no fallback. The tool pick is used for routing
only; execution still goes through the ticket agent and the executor.

Scope: workplace topics are in-domain and take the KB path, which is the only path that can
end in a ticket offer. Unrelated requests are deflected with no ticket. Unsure means in-domain.
"""
from collections.abc import Callable
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from smartdesk_ai.agent.intents import ClassifierError, Route, route_for
from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.agent.state import Awaiting, Intent, IntentResult, RouteSource
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.factory import ModelBuilder, get_llm, openai_builder
from smartdesk_ai.tools.definitions import TOOLS, openai_tool_schemas

ToolName = Literal["search_kb", "create_ticket", "get_tickets"]
_KB_ROUTES: tuple[Route, ...] = ("it_worker", "hr_worker", "misc")
_TOOL_INTENT: dict[str, Intent] = {
    "search_kb": "kb_query",
    "create_ticket": "create_ticket",
    "get_tickets": "ticket_status",
}

# Placeholder wording until the CHITCHAT task writes replies (T4.27).
CLARIFYING_QUESTION = "Could you tell me a little more about what you need help with?"
SMALLTALK_REPLY = "Hello! I can answer IT and HR questions, open a support ticket, or check on a ticket. What do you need?"
DEFLECTION_REPLY = (
    "Sorry, I can't help with that one. I'm here for IT, HR and workplace questions, "
    "such as accounts, equipment, leave and office services."
)


class ToolPickError(Exception):
    """The tool-pick model call failed."""


class ToolPick(BaseModel):
    """The model's tool choice, used for routing only. `tool` is None when it called no tool."""

    model_config = ConfigDict(frozen=True)

    tool: ToolName | None = None
    domain: Literal["IT", "HR"] | None = None


class RouteDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    intent: Intent | None  # None: the tool pick named no tool (chitchat route)
    route: Route
    source: RouteSource
    confidence: float | None = None
    clarify: bool = False  # ask exactly one clarifying question
    deflect: bool = False  # out of scope: polite deflection, no ticket

    @property
    def ticket_offer_possible(self) -> bool:
        """Only the KB path can end in a ticket offer (FR-D9, FR-A16)."""
        return self.route in _KB_ROUTES


ClassifyFn = Callable[..., IntentResult]
Router = Callable[[str, Awaiting, str | None], "RouteDecision"]  # (text, awaiting, last topic) -> decision
ToolPicker = Callable[[str, Awaiting], ToolPick]


def clarifying_question() -> str:
    return CLARIFYING_QUESTION


def deflection_reply() -> str:
    return DEFLECTION_REPLY


def _from_classifier(result: IntentResult) -> RouteDecision:
    return RouteDecision(
        intent=result.intent,
        route=route_for(result),
        source="classifier",
        confidence=result.confidence,
        clarify=result.intent == "ambiguous",
        deflect=result.intent == "out_of_scope",
    )


def _from_tool_pick(pick: ToolPick, classified: IntentResult | None) -> RouteDecision:
    if pick.tool is None:
        if classified is not None and classified.intent == "out_of_scope":
            # Unsure whether this is work-related: treat as in-domain (FR-D9).
            return RouteDecision(intent="kb_query", route="misc", source="tool_pick")
        return RouteDecision(intent=None, route="none", source="tool_pick")
    if pick.tool == "search_kb":
        route: Route = {"IT": "it_worker", "HR": "hr_worker"}.get(pick.domain or "", "misc")  # type: ignore[assignment]
    else:
        route = "ticket_agent"
    return RouteDecision(intent=_TOOL_INTENT[pick.tool], route=route, source="tool_pick")


def route_message(
    text: str,
    awaiting: Awaiting,
    last_topic: str | None,
    classify: ClassifyFn,
    pick_tool: ToolPicker,
    threshold: float,
) -> RouteDecision:
    """Classifier first; tool pick on error, invalid output or confidence below `threshold`."""
    try:
        result: IntentResult | None = classify(text, awaiting, last_topic)
    except ClassifierError:
        result = None
    if result is not None and result.confidence >= threshold:
        return _from_classifier(result)
    try:
        pick = pick_tool(text, awaiting)
    except ToolPickError:
        return RouteDecision(intent="ambiguous", route="none", source="tool_pick", clarify=True)
    return _from_tool_pick(pick, result)


def make_tool_picker(
    settings: Settings, *, builder: ModelBuilder = openai_builder, budget_level: int = 0
) -> ToolPicker:
    """Tool pick on the TOOL_SELECT model with the three tool schemas bound (FR-G2)."""
    llm = get_llm("TOOL_SELECT", settings, budget_level=budget_level, builder=builder).model
    bound = llm.bind_tools(openai_tool_schemas())
    prompt = load_prompt("tool_select", settings).text
    known = {t.name for t in TOOLS}

    def pick(text: str, awaiting: Awaiting = "none") -> ToolPick:
        try:
            response: Any = bound.invoke([SystemMessage(prompt), HumanMessage(f"Awaiting: {awaiting}\nMessage: {text}")])
        except Exception as error:
            raise ToolPickError(f"tool pick failed: {type(error).__name__}") from error
        calls = getattr(response, "tool_calls", None) or []
        if not calls or calls[0].get("name") not in known:
            return ToolPick()
        name = calls[0]["name"]
        domain = (calls[0].get("args") or {}).get("domain")
        return ToolPick(tool=name, domain=domain if domain in ("IT", "HR") and name == "search_kb" else None)

    return pick


def make_router(settings: Settings, *, builder: ModelBuilder = openai_builder, budget_level: int = 0) -> Router:
    """The routing function of the agent graph: the classifier first, the tool pick as the
    fallback, with the confidence threshold from config (FR-D8)."""
    from smartdesk_ai.agent.intents import make_classifier

    classify = make_classifier(settings, builder=builder, budget_level=budget_level)
    pick = make_tool_picker(settings, builder=builder, budget_level=budget_level)
    threshold = settings.thresholds.routing_confidence

    def route(text: str, awaiting: Awaiting, last_topic: str | None) -> RouteDecision:
        return route_message(text, awaiting, last_topic, classify, pick, threshold)

    return route
