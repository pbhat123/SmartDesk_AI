"""T4.27: tone and small talk (FR-D4, FR-D5). Greetings, thanks and goodbyes get a friendly reply
with no tool call; a ticket offer opens with an acknowledgement. Mock models, no network."""
import re
from types import SimpleNamespace

import pytest

from smartdesk_ai.agent.chitchat import (
    ACKNOWLEDGEMENT_PHRASES,
    make_chitchat,
    smalltalk_kind,
    static_chitchat,
)
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.agent.ticket_agent import escalation_offer
from smartdesk_ai.llm.mock import MockChitchatModel
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM

ACKNOWLEDGES = re.compile(r"\b(sorry|apologi[sz]e|understand|frustrating|that sounds)\b", re.I)
GREETING_WORDS = re.compile(r"\b(hello|hi|hey|welcome|good morning)\b", re.I)
THANKS_WORDS = re.compile(r"\b(welcome|glad|happy|pleasure)\b", re.I)
GOODBYE_WORDS = re.compile(r"\b(bye|goodbye|take care|have a)\b", re.I)

CASES = [
    ("Hello", "greeting", GREETING_WORDS),
    ("Good morning", "greeting", GREETING_WORDS),
    ("Thanks, that worked!", "thanks", THANKS_WORDS),
    ("Thank you so much", "thanks", THANKS_WORDS),
    ("Bye", "goodbye", GOODBYE_WORDS),
    ("Goodbye, see you", "goodbye", GOODBYE_WORDS),
]


def mock_chitchat(settings):
    return make_chitchat(settings, builder=lambda model_id, cfg: MockChitchatModel(), sleep=lambda _: None)


@pytest.mark.parametrize("text,kind,_", CASES)
def test_the_kind_of_small_talk_is_read_from_the_message(text, kind, _):
    assert smalltalk_kind(text) == kind


@pytest.mark.parametrize("text,kind,words", CASES)
def test_the_static_reply_fits_the_kind(text, kind, words):
    reply = static_chitchat("smalltalk", text)
    assert words.search(reply) and "@" not in reply


@pytest.mark.parametrize("text,kind,words", CASES)
def test_the_chitchat_model_reply_fits_the_kind(settings, text, kind, words):
    assert words.search(mock_chitchat(settings)("smalltalk", text))


def test_a_failing_model_falls_back_to_the_static_reply(settings):
    class Down:
        def invoke(self, messages):
            raise ConnectionError("down")

    chat = make_chitchat(settings, builder=lambda model_id, cfg: Down(), sleep=lambda _: None)
    assert chat("smalltalk", "Thanks!") == static_chitchat("smalltalk", "Thanks!")


@pytest.mark.parametrize("bad", ["", "   ", "write to me at a@b.example", "x" * 600])
def test_an_unusable_model_reply_falls_back_to_the_static_reply(settings, bad):
    model = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content=bad))
    chat = make_chitchat(settings, builder=lambda model_id, cfg: model, sleep=lambda _: None)
    assert chat("smalltalk", "Hello") == static_chitchat("smalltalk", "Hello")


def test_deflect_and_clarify_stay_short_and_offer_no_ticket(settings):
    chat = mock_chitchat(settings)
    deflect, clarify = chat("deflect", "What's the weather?"), chat("clarify", "it's not working")
    assert "ticket" not in deflect.lower() and clarify.count("?") == 1


# --- through the graph ------------------------------------------------------------------------


@pytest.mark.parametrize("text,kind,words", CASES)
def test_small_talk_gets_a_reply_and_calls_no_tool(built_index_dir, settings, tmp_path, text, kind, words):
    stack = assemble(built_index_dir, settings, tmp_path, routed=True, chitchat=mock_chitchat(settings))
    issues_before = len(stack.client._issues)
    state = run_turn(stack.graph, new_thread_id(), text)
    assert words.search(state.reply), state.reply
    assert state.steps == ["start", "route", "chitchat"]
    assert stack.spy.calls == [] and len(stack.client._issues) == issues_before
    assert state.pending_offer is None and state.awaiting == "none"


def test_the_graph_works_without_a_chitchat_model(built_index_dir, settings, tmp_path):
    stack = assemble(built_index_dir, settings, tmp_path, routed=True)
    assert THANKS_WORDS.search(run_turn(stack.graph, new_thread_id(), "Thanks a lot").reply)


# --- empathy before the ticket offer ----------------------------------------------------------


@pytest.mark.parametrize("email_known", [False, True])
def test_the_offer_text_starts_with_an_acknowledgement(email_known):
    text = escalation_offer(email_known)
    assert ACKNOWLEDGES.search(text.split(".")[0]) and text.startswith(tuple(ACKNOWLEDGEMENT_PHRASES))
    assert "don't know" in text.lower() and "ticket" in text.lower()


def test_a_kb_gap_reply_acknowledges_before_it_offers_a_ticket(built_index_dir, settings, tmp_path):
    stack = assemble(built_index_dir, settings, tmp_path, routed=True, llm=FakeAnswerLLM("insufficient"))
    state = run_turn(stack.graph, new_thread_id(), "My monitor keeps flickering")
    assert state.pending_offer is not None
    assert ACKNOWLEDGES.search(state.reply)
    assert state.reply.lower().index("sorry") < state.reply.lower().index("ticket")
    assert state.reply.lower().index("sorry") < state.reply.lower().index("don't know")
