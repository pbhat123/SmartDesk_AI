"""A deterministic check for first-person self-harm statements (FR-H1): a fixed caring reply, no ticket,
the reason logged and the message never logged or traced."""
import json

import pytest

from smartdesk_ai.agent.chat_service import ChatService
from smartdesk_ai.guardrails.input_guard import InputGuard
from smartdesk_ai.guardrails.moderation import ModerationResult
from smartdesk_ai.guardrails.names import NameReducer
from smartdesk_ai.guardrails.pii import PiiRedactor
from smartdesk_ai.guardrails.self_harm import SELF_HARM_REPLY, is_self_harm

STATEMENTS = [
    "I want to kill myself",
    "I'm going to end my life tonight",
    "i am thinking about suicide",
    "I've been thinking of hurting myself",
    "I don't want to live anymore",
    "I feel like ending it all",
    "I'm suicidal",
    "Honestly I just want to die",
    "I am planning to harm myself after this call",
    "Sometimes I wish I were dead",
]
NOT_SELF_HARM = [
    "How do I kill this process on my laptop?",
    "Please kill the stuck VPN session",
    "This deadline is killing me",
    "My laptop died again, can you fix it?",
    "I want to end this call and open a ticket",
    "How do I report a colleague who said they would hurt themselves?",
    "I want to die... of boredom in this onboarding session, any shortcuts?",
    "What is the suicide prevention policy at SmartDesk?",
    "The process will die if memory runs out",
    "I'd kill for a faster laptop",
]


@pytest.mark.parametrize("text", STATEMENTS)
def test_first_person_self_harm_is_detected(text):
    assert is_self_harm(text)


@pytest.mark.parametrize("text", NOT_SELF_HARM)
def test_technical_and_third_person_wording_is_not_self_harm(text):
    assert not is_self_harm(text)


def test_the_reply_is_caring_and_names_manager_people_team_and_emergency_services():
    for part in ("manager", "People Team", "emergency"):
        assert part in SELF_HARM_REPLY
    assert "ticket" not in SELF_HARM_REPLY.lower() and "knowledge base" not in SELF_HARM_REPLY.lower()


class Calls:
    def __init__(self):
        self.seen = []

    def check(self, text):
        self.seen.append(text)
        return ModerationResult(flagged=False)


def guard(moderator):
    return InputGuard(PiiRedactor(None, 0.5), NameReducer(None, 0.5), moderator)


def test_self_harm_is_answered_with_the_fixed_reply_before_moderation_and_logs_only_the_reason(caplog):
    moderator = Calls()
    with caplog.at_level("DEBUG"):
        decision = guard(moderator)("I want to kill myself")
    assert decision.blocked and decision.refusal == SELF_HARM_REPLY and decision.reason == "self_harm"
    assert moderator.seen == []
    assert "self_harm" in caplog.text and "kill" not in caplog.text
    assert "kill" not in decision.text  # what ChatService traces is a placeholder, not the statement


def test_the_service_creates_no_ticket_runs_no_graph_and_traces_no_statement(settings, tmp_path):
    from smartdesk_ai.agent.factory import TRACE_LOG, build_responder
    from smartdesk_ai.tickets import MockTicketClient

    client = MockTicketClient()
    responder = build_responder(settings, environ={}, dotenv_path=None, data_dir=tmp_path, client=client)
    before = len(client._issues)  # the demo tickets are already seeded
    calls = []
    service = ChatService(settings, input_guard=guard(Calls()), responder=lambda s, t: calls.append(t) or responder(s, t),
                          services=responder.services)
    reply = service.handle("s1", "I want to kill myself, please raise a ticket for me")
    assert reply.text == SELF_HARM_REPLY and reply.debug["blocked_reason"] == "self_harm"
    assert calls == [] and len(client._issues) == before
    assert "kill" not in (tmp_path / TRACE_LOG).read_text()
