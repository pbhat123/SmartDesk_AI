"""T4.29: one agent graph runs all three capabilities, and domain classification is consistent
(FR-D7, FAQ Q15). Mock models, no network."""
import re

import pytest

from evals.common import EVALS_DIR, load_jsonl
from evals.schemas import CategoryRow
from smartdesk_ai.agent.confirm import make_confirm_parser
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.intents import make_classifier
from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.agent.ticket_agent import draft_from_statements
from smartdesk_ai.llm.mock import MockConfirmModel, MockDraftModel, MockRoutingModel
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM

ROWS = [CategoryRow(**r) for r in load_jsonl(EVALS_DIR / "categories.jsonl")]
EMAIL = "one.ticket@smartdesk.example"
AGREEMENT_THRESHOLD = 0.90  # share of the category set on which classifier and drafter agree (judgment call)


@pytest.fixture
def stack(built_index_dir, settings, tmp_path):
    parser = make_confirm_parser(settings, builder=lambda m, c: MockConfirmModel())
    return assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM(["cite_top", "insufficient"]),
                    routed=True, drafter_llm=MockDraftModel(), parse_confirm=parser, seeded=True)


def test_one_compiled_graph_has_the_kb_ticket_and_status_paths(stack):
    nodes = set(stack.graph.get_graph().nodes)
    assert {"route", "it_worker", "hr_worker", "misc_search", "kb_retry", "t_create", "s_lookup", "chitchat"} <= nodes


def test_that_one_graph_handles_kb_then_create_then_status_in_one_session(stack):
    tid, issues = new_thread_id(), len(stack.client._issues)
    kb = run_turn(stack.graph, tid, "How do I reset my forgotten password?", email=EMAIL)
    assert kb.steps[:3] == ["start", "route", "search_kb"] and kb.pending_offer is None
    gap = run_turn(stack.graph, tid, "My monitor keeps flickering")
    assert gap.pending_offer is not None
    run_turn(stack.graph, tid, "yes")  # agrees; the draft is shown
    created = run_turn(stack.graph, tid, "yes")
    assert len(stack.client._issues) == issues + 1 and "has been created" in created.reply
    status = run_turn(stack.graph, tid, "What is the status of my ticket?")
    assert "s_lookup" in status.steps and re.search(r"#\d+", status.reply)
    assert status.email == EMAIL


def test_each_capability_ends_in_the_expected_route_kind(stack):
    seen = {}
    for kind, text in [("kb", "How do I reset my forgotten password?"), ("ticket", "Please raise a ticket for my VPN"),
                       ("status", "What is the status of my ticket?"), ("smalltalk", "Thanks!"),
                       ("deflect", "What's the weather like today?"), ("clarify", "it's not working")]:
        seen[kind] = run_turn(stack.graph, new_thread_id(), text, email=EMAIL).route_kind
    assert seen == {k: k for k in seen}


# --- consistent domain classification ---------------------------------------------------------


def classifier_domains(settings):
    classify = make_classifier(settings, builder=lambda model_id, cfg: MockRoutingModel())
    return {r.id: classify(r.utterance).domain for r in ROWS}


def drafted_categories():
    return {r.id: draft_from_statements(MockDraftModel(), "prompt", [r.utterance]).category for r in ROWS}


def test_classifier_and_drafter_agree_on_the_category_set(settings):
    domains, categories = classifier_domains(settings), drafted_categories()
    agreement = sum(domains[r.id] == categories[r.id] for r in ROWS) / len(ROWS)
    assert agreement >= AGREEMENT_THRESHOLD, agreement


def test_both_match_the_expected_category_at_the_threshold(settings):
    domains, categories = classifier_domains(settings), drafted_categories()
    for name, found in (("classifier", domains), ("drafter", categories)):
        correct = sum(found[r.id] == r.expected_category for r in ROWS) / len(ROWS)
        assert correct >= AGREEMENT_THRESHOLD, (name, correct)


def test_the_classifier_and_the_ticket_prompt_define_the_categories_alike(settings):
    definition = ("IT for devices, accounts, software and networks. HR for leave, pay, benefits, policies "
                  "and onboarding. MISC only when neither fits, for example office parking or the cafeteria.")
    assert definition in load_prompt("orchestrator", settings).text
    assert definition in load_prompt("ticket_summary", settings).text
