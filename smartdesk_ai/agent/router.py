"""Routing (FR-D8, FR-G2) and the scope boundary (FR-D9).

Order: the intent classifier decides. Only if it errors, returns invalid output, or is below
the routing threshold does the route come from the model's tool pick. A confident `ambiguous`
is a valid result: one clarifying question, no fallback. The tool pick is used for routing
only; execution still goes through the ticket agent and the executor.
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
_TOOL_INTENT: dict[str, Intent] = {
    "search_kb": "kb_query",
    "create_ticket": "create_ticket",
    "get_tickets": "ticket_status",
}

# Placeholder wording until the CHITCHAT task writes replies (T4.27).
CLARIFYING_QUESTION = "Could you tell me a little more about what you need help with?"


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


ClassifyFn = Callable[..., IntentResult]
ToolPicker = Callable[[str, Awaiting], ToolPick]


def clarifying_question() -> str:
    return CLARIFYING_QUESTION


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
