"""T4.21: edge cases at the chat entry point (FR-F4): empty input, input over `input.max_chars`, and an
invalid email, which is asked for again. None of them reaches a model or creates a ticket."""
import logging

import pytest

from smartdesk_ai.agent.chat_service import (
    EMPTY_REPLY,
    ChatService,
    GuardDecision,
    too_long_reply,
)
from smartdesk_ai.agent.graph import make_kb_responder
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.llm.mock import MockDraftModel
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM


class CountingResponder:
    def __init__(self):
        self.calls = []

    def __call__(self, session_id, text):
        self.calls.append(text)
        return "ok"


@pytest.fixture
def counting(settings):
    responder = CountingResponder()
    return ChatService(settings, responder=responder), responder


@pytest.mark.parametrize("text", ["", "   ", "\n\t  \n"])
def test_empty_input_gets_a_prompt_to_type_and_reaches_nothing(counting, text):
    service, responder = counting
    reply = service.handle("s1", text)
    assert reply.text == EMPTY_REPLY and "type" in reply.text.lower()
    assert responder.calls == [] and reply.debug["edge"] == "empty"


def test_over_length_input_gets_a_polite_request_to_shorten(counting, settings):
    service, responder = counting
    limit = settings.input.max_chars
    reply = service.handle("s1", "x" * (limit + 1))
    assert reply.text == too_long_reply(limit) and str(limit) in reply.text and "shorten" in reply.text.lower()
    assert responder.calls == [] and reply.debug["edge"] == "too_long"


def test_input_of_exactly_the_limit_is_accepted(counting, settings):
    service, responder = counting
    assert service.handle("s1", "x" * settings.input.max_chars).text == "ok"
    assert len(responder.calls) == 1


def test_the_limit_comes_from_config(settings):
    tight = settings.model_copy(update={"input": settings.input.model_copy(update={"max_chars": 10})})
    responder = CountingResponder()
    service = ChatService(tight, responder=responder)
    assert service.handle("s1", "short").text == "ok"
    assert "10" in service.handle("s1", "this is far too long").text
    assert len(responder.calls) == 1


def test_the_limit_is_checked_before_the_guards(settings):
    seen = []
    service = ChatService(settings, input_guard=lambda t: seen.append(t) or GuardDecision(text=t))
    service.handle("s1", "x" * (settings.input.max_chars + 1))
    service.handle("s1", "")
    assert seen == []


def test_the_rejected_text_is_never_logged(counting, settings, caplog):
    service, _ = counting
    secret = "my email is jane.doe@example.com " + "x" * settings.input.max_chars
    with caplog.at_level(logging.DEBUG):
        service.handle("s1", secret)
    assert "jane.doe@example.com" not in caplog.text


def test_an_empty_message_does_not_disturb_the_conversation(built_index_dir, settings, tmp_path):
    stack = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"), routed=True,
                     drafter_llm=MockDraftModel())
    service = ChatService(settings, responder=make_kb_responder(stack.graph))
    sid = new_thread_id()
    service.handle(sid, "My monitor keeps flickering")
    assert service.handle(sid, "").text == EMPTY_REPLY
    assert "email" in service.handle(sid, "yes").text.lower()  # the offer is still pending


# --- invalid email --------------------------------------------------------------------------------


def chat(built_index_dir, settings, tmp_path):
    stack = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"), routed=True,
                     drafter_llm=MockDraftModel())
    return stack, ChatService(settings, responder=make_kb_responder(stack.graph)), new_thread_id()


@pytest.mark.parametrize("bad", ["priya@", "priya.smartdesk.example", "priya @smartdesk.example", "@@"])
def test_an_invalid_email_is_asked_for_again_and_nothing_is_created(built_index_dir, settings, tmp_path, bad):
    stack, service, sid = chat(built_index_dir, settings, tmp_path)
    service.handle(sid, "My monitor keeps flickering")
    service.handle(sid, "yes")
    reply = service.handle(sid, bad)
    if "@" in bad:
        assert "valid email" in reply.text.lower()
    assert not stack.client._issues
    ok = service.handle(sid, "priya.k@smartdesk.example")
    assert "Shall I go ahead and create this?" in ok.text


def test_the_invalid_email_is_not_echoed_back(built_index_dir, settings, tmp_path):
    _, service, sid = chat(built_index_dir, settings, tmp_path)
    service.handle(sid, "My monitor keeps flickering")
    service.handle(sid, "yes")
    assert "priya@" not in service.handle(sid, "priya@").text


def test_a_valid_email_is_accepted_the_first_time(built_index_dir, settings, tmp_path):
    _, service, sid = chat(built_index_dir, settings, tmp_path)
    service.handle(sid, "My monitor keeps flickering")
    reply = service.handle(sid, "priya.k@smartdesk.example")
    assert "Shall I go ahead and create this?" in reply.text


def test_an_invalid_email_given_with_a_direct_request_is_asked_for_again(built_index_dir, settings, tmp_path):
    stack, service, sid = chat(built_index_dir, settings, tmp_path)
    assert "email" in service.handle(sid, "Please raise a ticket for my broken VPN").text.lower()
    assert "valid email" in service.handle(sid, "priya@").text.lower()
    assert not stack.client._issues
