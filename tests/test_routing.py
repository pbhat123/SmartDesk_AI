"""T4.4: routing order (FR-D8, FR-G2): classifier first, model tool pick as the fallback."""
from types import SimpleNamespace

import pytest

from evals.common import EVALS_DIR, load_jsonl
from evals.schemas import IntentRow
from smartdesk_ai.agent.intents import ClassifierError
from smartdesk_ai.agent.router import (
    ToolPick,
    ToolPickError,
    clarifying_question,
    make_tool_picker,
    route_message,
)
from smartdesk_ai.agent.state import IntentResult

THRESHOLD = 0.60


def classifier(intent="kb_query", confidence=0.9, domain=None):
    calls = []

    def classify(text, awaiting="none", last_topic=None):
        calls.append((text, awaiting, last_topic))
        return IntentResult(intent=intent, confidence=confidence, domain=domain)

    classify.calls = calls
    return classify


def broken_classifier(text, awaiting="none", last_topic=None):
    raise ClassifierError("forced")


def picker(tool=None, domain=None):
    calls = []

    def pick(text, awaiting="none"):
        calls.append((text, awaiting))
        return ToolPick(tool=tool, domain=domain)

    pick.calls = calls
    return pick


def route(classify, pick, text="hello", awaiting="none"):
    return route_message(text, awaiting, None, classify, pick, THRESHOLD)


def test_confident_classifier_decides_and_the_tool_pick_is_not_used():
    pick = picker("get_tickets")
    decision = route(classifier("kb_query", 0.9, "HR"), pick)
    assert (decision.intent, decision.route, decision.source) == ("kb_query", "hr_worker", "classifier")
    assert pick.calls == []


def test_confidence_exactly_at_the_threshold_is_confident():
    pick = picker("get_tickets")
    decision = route(classifier("create_ticket", THRESHOLD), pick)
    assert decision.source == "classifier" and decision.route == "ticket_agent" and pick.calls == []


@pytest.mark.parametrize(
    "tool, domain, intent, expected_route",
    [
        ("search_kb", "IT", "kb_query", "it_worker"),
        ("search_kb", "HR", "kb_query", "hr_worker"),
        ("search_kb", None, "kb_query", "misc"),
        ("create_ticket", None, "create_ticket", "ticket_agent"),
        ("get_tickets", None, "ticket_status", "ticket_agent"),
        (None, None, None, "none"),
    ],
)
def test_forced_classifier_error_routes_from_the_tool_pick(tool, domain, intent, expected_route):
    pick = picker(tool, domain)
    decision = route(broken_classifier, pick, "something", "offer_reply")
    assert decision.source == "tool_pick"
    assert (decision.intent, decision.route) == (intent, expected_route)
    assert pick.calls == [("something", "offer_reply")]


def test_no_tool_maps_to_the_chitchat_route_without_clarifying():
    decision = route(broken_classifier, picker(None))
    assert decision.route == "none" and decision.intent is None
    assert not decision.clarify and not decision.deflect


def test_low_confidence_falls_back_to_the_tool_pick():
    pick = picker("search_kb", "IT")
    decision = route(classifier("smalltalk", 0.59), pick)
    assert decision.source == "tool_pick" and decision.route == "it_worker" and len(pick.calls) == 1


def test_invalid_output_is_a_classifier_error_and_falls_back(settings):
    # an unknown intent never reaches the router: classify_intent raises ClassifierError (T4.3)
    decision = route(broken_classifier, picker("create_ticket"))
    assert decision.route == "ticket_agent"


def test_confident_ambiguous_gets_one_clarifying_question_and_no_fallback():
    pick = picker("search_kb")
    decision = route(classifier("ambiguous", 0.9), pick, "It's not working")
    assert decision.clarify and decision.route == "none" and decision.source == "classifier"
    assert pick.calls == []
    assert clarifying_question().count("?") == 1


def test_low_confidence_ambiguous_is_not_confident_so_the_tool_pick_decides():
    pick = picker("search_kb")
    decision = route(classifier("ambiguous", 0.3), pick)
    assert not decision.clarify and decision.route == "misc" and len(pick.calls) == 1


def test_classifier_error_then_tool_pick_error_asks_one_clarifying_question():
    def broken_pick(text, awaiting="none"):
        raise ToolPickError("forced")

    decision = route(broken_classifier, broken_pick)
    assert decision.clarify and decision.route == "none"


def test_the_threshold_comes_from_config(settings):
    assert settings.thresholds.routing_confidence == THRESHOLD


def test_awaiting_and_last_topic_reach_the_classifier():
    classify = classifier()
    route_message("yes", "offer_reply", "vpn", classify, picker(), THRESHOLD)
    assert classify.calls == [("yes", "offer_reply", "vpn")]


# --- eval set: with the classifier down, every row routes from its tool pick ------------

ROWS = [IntentRow(**r) for r in load_jsonl(EVALS_DIR / "intents.jsonl")]
DOMAIN = {"it_worker": "IT", "hr_worker": "HR"}


@pytest.mark.parametrize("row", ROWS, ids=[r.id for r in ROWS])
def test_fallback_route_matches_the_eval_route(row):
    tool = None if row.expected_tool == "none" else row.expected_tool
    decision = route(broken_classifier, picker(tool, DOMAIN.get(row.expected_route)), row.utterance, row.awaiting)
    assert decision.route == row.expected_route


# --- the LLM-backed tool picker ---------------------------------------------------------


class FakeToolLLM:
    def __init__(self, tool_calls):
        self.tool_calls, self.bound, self.messages = tool_calls, None, None

    def bind_tools(self, tools):
        self.bound = tools
        return self

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content="", tool_calls=self.tool_calls)


def build_picker(settings, llm):
    built = []

    def builder(model_id, cfg):
        built.append(model_id)
        return llm

    return make_tool_picker(settings, builder=builder), built


def test_tool_picker_binds_the_three_tools_on_the_light_model(settings):
    llm = FakeToolLLM([{"name": "search_kb", "args": {"query": "vpn", "domain": "IT"}, "id": "1"}])
    pick, built = build_picker(settings, llm)
    assert pick("how do I use the vpn", "none") == ToolPick(tool="search_kb", domain="IT")
    assert built == [settings.models.light.id]
    assert [t["function"]["name"] for t in llm.bound] == ["search_kb", "create_ticket", "get_tickets"]
    assert "Awaiting: none" in llm.messages[-1].content


def test_tool_picker_no_tool_call_means_no_tool(settings):
    pick, _ = build_picker(settings, FakeToolLLM([]))
    assert pick("thanks", "none") == ToolPick(tool=None, domain=None)


def test_tool_picker_ignores_an_unknown_tool_name_and_a_bad_domain(settings):
    pick, _ = build_picker(settings, FakeToolLLM([{"name": "delete_everything", "args": {}, "id": "1"}]))
    assert pick("x", "none").tool is None
    pick, _ = build_picker(settings, FakeToolLLM([{"name": "search_kb", "args": {"query": "q", "domain": "LEGAL"}, "id": "1"}]))
    assert pick("x", "none") == ToolPick(tool="search_kb", domain=None)


def test_tool_picker_call_failure_is_a_tool_pick_error(settings):
    class Boom(FakeToolLLM):
        def invoke(self, messages):
            raise TimeoutError("down")

    pick, _ = build_picker(settings, Boom([]))
    with pytest.raises(ToolPickError):
        pick("x", "none")


def test_make_router_wires_the_classifier_the_tool_pick_and_the_config_threshold(settings):
    from smartdesk_ai.agent.router import make_router
    from smartdesk_ai.llm.mock import MockRoutingModel

    route = make_router(settings, builder=lambda model_id, cfg: MockRoutingModel())
    decision = route("How many sick days do I get?", "none", None)
    assert (decision.intent, decision.route, decision.source) == ("kb_query", "hr_worker", "classifier")
    assert route("It's not working", "none", None).clarify


def test_the_mock_routing_model_matches_the_whole_intent_eval_set():
    from smartdesk_ai.agent.intents import classify_intent, route_for
    from smartdesk_ai.llm.mock import MockRoutingModel

    for row in ROWS:
        result = classify_intent(MockRoutingModel(), "prompt", row.utterance, row.awaiting)
        assert (result.intent, route_for(result)) == (row.expected_intent, row.expected_route), row.id
