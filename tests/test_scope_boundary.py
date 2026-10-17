"""T4.5: scope boundary (FR-D9, FR-D3). Workplace gaps take the KB path, where a gap ends in a
ticket offer; unrelated requests are deflected with no ticket offer; unsure means in-domain.

The ticket offer itself is built by the KB-first flow (T4.8 to T4.10). Here the router must
send workplace gaps down the KB path, the only path that can offer a ticket, and must never
offer one on a deflection.
"""
import pytest

from evals.common import EVALS_DIR, load_jsonl
from evals.schemas import GapQuestion, IntentRow
from smartdesk_ai.agent.router import ToolPick, deflection_reply, route_message
from smartdesk_ai.agent.state import IntentResult

THRESHOLD = 0.60
ROWS = [IntentRow(**r) for r in load_jsonl(EVALS_DIR / "intents.jsonl")]


def by_utterance(rows_in_scope=ROWS):
    """A classifier that answers the way the eval set expects (stands in for the mock LLM)."""
    table = {(r.utterance, r.awaiting): r for r in rows_in_scope}

    def classify(text, awaiting="none", last_topic=None):
        r = table[(text, awaiting)]
        domain = {"it_worker": "IT", "hr_worker": "HR", "misc": "MISC"}.get(r.expected_route)
        return IntentResult(intent=r.expected_intent, confidence=0.9, domain=domain)

    return classify


def no_pick(text, awaiting="none"):
    raise AssertionError("tool pick must not run for a confident classification")


def route(row):
    return route_message(row.utterance, row.awaiting, None, by_utterance(), no_pick, THRESHOLD)


GAPS = [r for r in ROWS if r.expected_intent == "kb_query" and r.expected_route == "misc"]
UNRELATED = [r for r in ROWS if r.expected_intent == "out_of_scope"]


def test_the_eval_set_has_parking_cafeteria_and_desk_gaps():
    text = " ".join(r.utterance.lower() for r in GAPS)
    assert "parking" in text and "cafeteria" in text and "desk" in text


@pytest.mark.parametrize("row", GAPS, ids=[r.id for r in GAPS])
def test_workplace_gaps_take_the_kb_path_and_can_end_in_a_ticket_offer(row):
    decision = route(row)
    assert decision.intent == "kb_query" and decision.route == "misc"
    assert not decision.deflect and decision.ticket_offer_possible


@pytest.mark.parametrize("row", UNRELATED, ids=[r.id for r in UNRELATED])
def test_unrelated_requests_are_deflected_with_no_ticket_offer(row):
    decision = route(row)
    assert decision.deflect and decision.route == "none"
    assert not decision.ticket_offer_possible and not decision.clarify


def test_unrelated_kinds_include_weather_recipes_and_trivia():
    assert {"weather", "recipe", "trivia"} <= {r.out_of_scope_kind for r in UNRELATED}


def test_the_deflection_offers_no_ticket_and_stays_polite():
    reply = deflection_reply().lower()
    assert "ticket" not in reply and "email" not in reply
    assert "sorry" in reply or "can't help" in reply or "unable" in reply
    assert "it" in reply and "hr" in reply  # says what it can help with


def test_gap_questions_in_the_gap_set_are_workplace_topics_not_deflected():
    gaps = [GapQuestion(**r) for r in load_jsonl(EVALS_DIR / "gap_questions.jsonl")]
    misc = [g for g in gaps if g.expected_category == "MISC"]
    assert misc, "need MISC gap questions"
    for g in misc:
        classify = lambda text, awaiting="none", last_topic=None: IntentResult(  # noqa: E731
            intent="kb_query", confidence=0.9, domain="MISC"
        )
        decision = route_message(g.question, "none", None, classify, no_pick, THRESHOLD)
        assert decision.ticket_offer_possible and not decision.deflect


def broken(text, awaiting="none", last_topic=None):
    from smartdesk_ai.agent.intents import ClassifierError

    raise ClassifierError("forced")


def unsure_out_of_scope(text, awaiting="none", last_topic=None):
    return IntentResult(intent="out_of_scope", confidence=0.4)


def pick_of(tool, domain=None):
    return lambda text, awaiting="none": ToolPick(tool=tool, domain=domain)


def test_unsure_scope_defaults_to_in_domain_not_deflection():
    decision = route_message("Can I park my bike near the lift?", "none", None, unsure_out_of_scope, pick_of(None), THRESHOLD)
    assert decision.intent == "kb_query" and decision.route == "misc"
    assert not decision.deflect and decision.ticket_offer_possible


def test_unsure_scope_follows_the_tool_pick_when_it_names_a_tool():
    decision = route_message("x", "none", None, unsure_out_of_scope, pick_of("search_kb", "IT"), THRESHOLD)
    assert decision.route == "it_worker" and not decision.deflect


def test_a_dead_classifier_with_no_tool_pick_is_chitchat_not_a_guess_at_scope():
    decision = route_message("hi", "none", None, broken, pick_of(None), THRESHOLD)
    assert decision.route == "none" and not decision.deflect and not decision.ticket_offer_possible


def test_confident_out_of_scope_is_deflected_even_when_the_pick_would_search():
    decision = route_message("weather?", "none", None, lambda *a, **k: IntentResult(intent="out_of_scope", confidence=0.95), no_pick, THRESHOLD)
    assert decision.deflect


# --- T4.10 re-check: the full acceptance, routed through the real KB graph ----------------


def test_workplace_gaps_end_in_a_ticket_offer_and_unrelated_never_do(built_index_dir, settings, tmp_path):
    from smartdesk_ai.agent.graph import run_turn
    from smartdesk_ai.agent.state import new_thread_id
    from tests.agent_helpers import build_test_graph
    from tests.llm_fakes import FakeAnswerLLM

    graph, spy = build_test_graph(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"))
    for row in GAPS:
        decision = route(row)
        state = run_turn(graph, new_thread_id(), row.utterance)  # misc route: whole-KB search
        assert state.pending_offer is not None and "ticket" in state.reply.lower(), row.id
    searched = len(spy.calls)
    for row in UNRELATED:
        decision = route(row)
        assert decision.deflect and "ticket" not in deflection_reply().lower(), row.id
    assert len(spy.calls) == searched, "a deflection must not touch the KB or the ticket path"
