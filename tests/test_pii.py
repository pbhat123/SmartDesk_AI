"""T5.1: PII detection and redaction with a regex fallback (FR-H1, FR-F7, FR-G11).
Presidio is faked or forced to fail here; the real engine has its own opt-in test below."""
import logging
import os

import pytest

from smartdesk_ai.guardrails.pii import (
    PiiRedactor,
    PresidioEngine,
    Span,
    load_presidio_engine,
    mask_email,
    mask_emails_in,
    make_presidio_engine,
)
from smartdesk_ai.ops.health import Component, HealthRegistry, Status

PHONE, CARD, AADHAAR, PAN = "+91 98765 43210", "4111 1111 1111 1111", "1234 5678 9012", "ABCDE1234F"


def regex_only() -> PiiRedactor:
    return PiiRedactor(engine=None, min_score=0.5)


@pytest.mark.parametrize("raw,kind,placeholder", [
    (f"Please call me on {PHONE} about my ticket.", "PHONE", "[PHONE]"),
    ("My mobile is (415) 555-0132, thanks", "PHONE", "[PHONE]"),
    (f"My corporate card number is {CARD}, why was it declined?", "CARD", "[CARD]"),
    ("card 4111-1111-1111-1111 please", "CARD", "[CARD]"),
    (f"My Aadhaar is {AADHAAR}, update payroll.", "ID", "[ID]"),
    (f"My PAN is {PAN}, how do I declare tax?", "ID", "[ID]"),
    ("SSN 123-45-6789 on file", "ID", "[ID]"),
])
def test_regex_redacts_each_kind(raw, kind, placeholder):
    out = regex_only().redact(raw)
    assert placeholder in out.text and kind in out.kinds
    for secret in (PHONE, CARD, AADHAAR, PAN, "555-0132", "123-45-6789"):
        assert secret not in out.text


def test_ordinary_numbers_and_ticket_ids_are_left_alone():
    text = "Ticket #14 was raised on 2026-10-17 and my desk is 4B, floor 12. Is 2FA on?"
    assert regex_only().redact(text).text == text


def test_a_card_that_fails_the_luhn_check_is_not_called_a_card():
    out = regex_only().redact("order 4111 1111 1111 1112 shipped")
    assert "[CARD]" not in out.text


def test_email_is_allowed_and_kept():
    text = "Raise a ticket, my email is priya.k@smartdesk.example"
    out = regex_only().redact(text)
    assert out.text == text and out.kinds == []


def test_phone_next_to_an_email_is_still_redacted():
    out = regex_only().redact(f"priya.k@smartdesk.example or {PHONE}")
    assert "priya.k@smartdesk.example" in out.text and PHONE not in out.text


def test_mask_email():
    assert mask_email("priya.k@smartdesk.example") == "p***@smartdesk.example"
    assert mask_email("a@x.io") == "a***@x.io"
    assert mask_emails_in("mail priya.k@smartdesk.example now") == "mail p***@smartdesk.example now"


class FakeAnalyzer:
    def __init__(self, results):
        self.results, self.calls = results, []

    def analyze(self, text, language, entities, score_threshold=None):
        self.calls.append((text, tuple(entities), score_threshold))
        return [r for r in self.results if r.entity_type in entities]


class FakeResult:
    def __init__(self, entity_type, start, end, score=0.9):
        self.entity_type, self.start, self.end, self.score = entity_type, start, end, score


def test_presidio_hits_are_merged_with_regex_hits():
    text = "driver licence D1234567 and phone 020 7946 0958 now"
    start = text.index("D1234567")
    engine = PresidioEngine(FakeAnalyzer([FakeResult("US_DRIVER_LICENSE", start, start + 8)]))
    out = PiiRedactor(engine, min_score=0.5).redact(text)
    assert "D1234567" not in out.text and "[ID]" in out.text and "[PHONE]" in out.text


def test_presidio_is_called_with_the_configured_score_and_never_asked_for_email():
    analyzer = FakeAnalyzer([])
    PiiRedactor(PresidioEngine(analyzer), min_score=0.6).redact("hello there")
    [(_, entities, score)] = analyzer.calls
    assert score == 0.6 and "EMAIL_ADDRESS" not in entities and "PERSON" not in entities


def test_presidio_failure_at_call_time_falls_back_to_regex(caplog):
    class Boom:
        def analyze(self, *a, **k):
            raise RuntimeError("nlp exploded")

    redactor = PiiRedactor(PresidioEngine(Boom()), min_score=0.5)
    with caplog.at_level(logging.WARNING):
        out = redactor.redact(f"call {PHONE}")
    assert "[PHONE]" in out.text and PHONE not in out.text
    assert "presidio" in caplog.text.lower()


def test_forced_presidio_failure_degrades_and_regex_still_redacts(caplog):
    registry = HealthRegistry()

    def failing_loader():
        raise OSError("model en_core_web_lg is not installed")

    with caplog.at_level(logging.WARNING):
        engine = make_presidio_engine(registry, loader=failing_loader)
    assert engine is None
    health = registry.get(Component.PRESIDIO)
    assert health.status is Status.DEGRADED and "regex" in health.mode
    assert "presidio" in caplog.text and "degraded" in caplog.text
    out = PiiRedactor(engine, min_score=0.5).redact(f"Call me on {PHONE}")
    assert PHONE not in out.text and "[PHONE]" in out.text


def test_a_working_loader_marks_presidio_ok():
    registry = HealthRegistry()
    engine = make_presidio_engine(registry, loader=lambda: PresidioEngine(FakeAnalyzer([])))
    assert engine is not None and registry.get(Component.PRESIDIO).status is Status.OK


def test_loader_refuses_to_download_a_missing_model(monkeypatch):
    import spacy.util

    monkeypatch.setattr(spacy.util, "is_package", lambda name: False)
    with pytest.raises(RuntimeError, match="en_core_web_lg"):
        load_presidio_engine()


@pytest.mark.skipif(not os.environ.get("RUN_REAL_MODEL_TESTS"), reason="set RUN_REAL_MODEL_TESTS=1")
def test_real_presidio_redacts_phone_card_and_id():
    engine = load_presidio_engine()
    out = PiiRedactor(engine, min_score=0.5).redact(f"phone {PHONE}, card {CARD}, PAN {PAN}")
    for secret in (PHONE, CARD, PAN):
        assert secret not in out.text
