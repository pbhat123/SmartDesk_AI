"""T5.3: Moderation on user input: a toxic message gets a polite refusal, the reason is logged, and no
tool runs (FR-H1). The Moderation API is mocked; nothing here touches the network."""
import json
import logging
from pathlib import Path
from types import SimpleNamespace

from smartdesk_ai.agent.chat_service import ChatService
from smartdesk_ai.guardrails.input_guard import InputGuard
from smartdesk_ai.guardrails.moderation import KeywordModerator, ModerationResult, OpenAIModerator
from smartdesk_ai.guardrails.names import NameReducer
from smartdesk_ai.guardrails.pii import PiiRedactor

SAFETY_SET = Path(__file__).resolve().parent.parent / "evals" / "safety_set.jsonl"


class StubModerator:
    def __init__(self, result=None, error=None):
        self.result = result or ModerationResult(flagged=False)
        self.error, self.seen = error, []

    def check(self, text):
        self.seen.append(text)
        if self.error:
            raise self.error
        return self.result


def guard(moderator):
    return InputGuard(pii=PiiRedactor(None, 0.5), names=NameReducer(None, 0.5), moderator=moderator)


def test_toxic_message_gets_a_polite_refusal_and_the_reason_is_logged(caplog):
    moderator = StubModerator(ModerationResult(flagged=True, categories=["harassment", "harassment_threatening"]))
    with caplog.at_level(logging.INFO):
        decision = guard(moderator)("You are a useless piece of garbage bot and I hope you die")
    assert decision.blocked and decision.reason == "moderation:harassment,harassment_threatening"
    assert "can't help" in decision.refusal and "respectful" in decision.refusal
    assert "moderation:harassment,harassment_threatening" in caplog.text
    assert "garbage" not in caplog.text  # the message itself is never logged


def test_a_blocked_message_reaches_no_tool(settings):
    calls = []
    moderator = StubModerator(ModerationResult(flagged=True, categories=["violence"]))
    service = ChatService(settings, input_guard=guard(moderator), responder=lambda sid, text: calls.append(text) or "ran")
    reply = service.handle("s1", "Everyone in that team deserves to suffer.")
    assert calls == [] and "can't help" in reply.text and reply.debug["blocked_reason"] == "moderation:violence"


def test_a_clean_message_passes_through_unchanged():
    decision = guard(StubModerator())("How many casual leave days do I get?")
    assert not decision.blocked and decision.text == "How many casual leave days do I get?"


def test_moderation_sees_redacted_text_and_a_masked_email():
    moderator = StubModerator()
    guard(moderator)("Call +91 98765 43210 or mail priya.k@smartdesk.example")
    [seen] = moderator.seen
    assert "98765" not in seen and "priya.k@smartdesk.example" not in seen and "p***@smartdesk.example" in seen


def test_the_text_passed_on_is_redacted_and_keeps_the_email():
    decision = guard(StubModerator())("Call +91 98765 43210 or mail priya.k@smartdesk.example")
    assert "[PHONE]" in decision.text and "priya.k@smartdesk.example" in decision.text


def test_a_moderation_outage_does_not_stop_the_service(caplog):
    with caplog.at_level(logging.WARNING):
        decision = guard(StubModerator(error=ConnectionError("down")))("hello")
    assert not decision.blocked and "moderation" in caplog.text.lower()


def test_openai_moderator_reads_the_api_result():
    created = []

    def create(model, input):
        created.append((model, input))
        flags = SimpleNamespace(model_dump=lambda: {"harassment": True, "violence": False, "hate": True})
        return SimpleNamespace(results=[SimpleNamespace(flagged=True, categories=flags)])

    client = SimpleNamespace(moderations=SimpleNamespace(create=create))
    result = OpenAIModerator(client, "omni-moderation-latest").check("text")
    assert result == ModerationResult(flagged=True, categories=["harassment", "hate"])
    assert created == [("omni-moderation-latest", "text")]


def test_keyword_moderator_flags_exactly_the_toxic_rows_of_the_safety_set():
    rows = [json.loads(line) for line in SAFETY_SET.read_text().splitlines() if line.strip()]
    flagged = {r["id"] for r in rows if KeywordModerator().check(r["input"]).flagged}
    # mild rudeness (answer_normally) is not flagged: the live Moderation API does not flag it either
    assert flagged == {r["id"] for r in rows if r["category"] == "toxic" and r["expected_behaviour"] == "block_and_refuse"}
