"""T5.6: output guard on the final reply: Moderation, a PII scan that redacts non-name PII only, and
Pydantic validation (FR-H2). Moderation is mocked; nothing here touches the network."""
import logging

import pytest

from smartdesk_ai.agent.chat_service import ChatService
from smartdesk_ai.guardrails.moderation import ModerationResult
from smartdesk_ai.guardrails.output_guard import (
    BLOCKED_REPLY,
    INVALID_REPLY,
    FinalReply,
    InvalidReply,
    OutputGuard,
    kb_contact_numbers,
    validate_reply,
)
from smartdesk_ai.guardrails.pii import PiiRedactor
from tests.kb_helpers import KB_DIR

PHONE, CARD, EMAIL = "+91 98765 43210", "4111 1111 1111 1111", "priya.k@smartdesk.example"


class StubModerator:
    def __init__(self, flagged=False, error=None):
        self.flagged, self.error, self.seen = flagged, error, []

    def check(self, text):
        self.seen.append(text)
        if self.error:
            raise self.error
        return ModerationResult(flagged=self.flagged, categories=["harassment"] if self.flagged else [])


def guard(moderator=None, keep=frozenset()):
    return OutputGuard(PiiRedactor(None, 0.5), moderator or StubModerator(), keep_numbers=keep)


def test_output_with_pii_is_redacted():
    out = guard().check(f"Your card {CARD} and phone {PHONE} are on file.")
    assert CARD not in out.text and PHONE not in out.text and "[CARD]" in out.text and "[PHONE]" in out.text
    assert out.actions == ["pii"] and not out.rejected


def test_full_names_are_not_touched_on_output():
    text = "Your manager Asha Menon approved it."
    assert guard().check(text).text == text


def test_the_employees_email_may_appear_in_a_reply():
    text = f"Email: {EMAIL}\nShall I go ahead and create this?"
    assert guard().check(text).text == text


def test_toxic_output_is_replaced():
    moderator = StubModerator(flagged=True)
    out = guard(moderator).check("some toxic generated text")
    assert out.text == BLOCKED_REPLY and out.actions == ["moderation:harassment"]


def test_moderation_sees_the_redacted_text_and_a_masked_email():
    moderator = StubModerator()
    guard(moderator).check(f"call {PHONE} or {EMAIL}")
    [seen] = moderator.seen
    assert "98765" not in seen and EMAIL not in seen


def test_a_moderation_outage_lets_the_reply_through_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        out = guard(StubModerator(error=ConnectionError("down"))).check("Hello there")
    assert out.text == "Hello there" and "moderation" in caplog.text.lower()


@pytest.mark.parametrize("bad", [None, 42, "", "   ", {"text": ""}, {"wrong": "x"}, {"text": "ok", "extra": 1}])
def test_invalid_structured_output_is_rejected(bad):
    with pytest.raises(InvalidReply):
        validate_reply(bad)
    out = guard().check(bad)
    assert out.rejected and out.text == INVALID_REPLY and out.actions == ["schema"]


def test_valid_shapes_are_accepted():
    assert validate_reply("hi") == FinalReply(text="hi")
    assert validate_reply({"text": "hi"}) == FinalReply(text="hi")
    assert validate_reply(FinalReply(text="hi")).text == "hi"


def test_the_company_numbers_of_the_kb_are_not_redacted():
    keep = kb_contact_numbers(KB_DIR)
    assert "918055500100" in keep  # the main switchboard from kb/company.md
    out = guard(keep=keep).check(f"Call the switchboard on +91-80-5550-0100, not {PHONE}.")
    assert "+91-80-5550-0100" in out.text and PHONE not in out.text


def test_chat_service_applies_the_guard_to_every_reply(settings):
    service = ChatService(settings, responder=lambda sid, text: f"Call {PHONE}", output_guard=guard())
    assert service.handle("s1", "hi").text == "Call [PHONE]"


def test_the_log_never_holds_the_reply_text(caplog):
    with caplog.at_level(logging.DEBUG):
        guard(StubModerator(flagged=True)).check(f"toxic {PHONE}")
    assert "98765" not in caplog.text and "toxic" not in caplog.text


def test_build_service_wires_the_output_guard_by_default():
    from pathlib import Path

    from smartdesk_ai.agent.chat_service import build_service
    from smartdesk_ai.ops.health import HealthRegistry

    built = build_service(environ={}, dotenv_path=Path("/nonexistent/.env"), registry=HealthRegistry(),
                          responder=lambda sid, text: f"Call {PHONE} or the switchboard on +91-80-5550-0100")
    reply = built.handle("s1", "hello").text
    assert "[PHONE]" in reply and "+91-80-5550-0100" in reply and "98765" not in reply
