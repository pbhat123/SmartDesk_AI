"""T5.2: person names are reduced to first name plus last initial, on user input only (FR-H1, FR-G11).
Presidio's PERSON detection is faked here; the real engine has an opt-in test at the bottom."""
import logging
import os
import re
from pathlib import Path

import pytest

from smartdesk_ai.guardrails.names import NameReducer, reduce_name
from smartdesk_ai.guardrails.pii import PresidioEngine, load_presidio_engine


class PersonAnalyzer:
    """Reports every listed name as a PERSON, the way the NER model would."""

    def __init__(self, *names):
        self.names, self.calls = names, []

    def analyze(self, text, language, entities, score_threshold=None):
        self.calls.append((tuple(entities), score_threshold))
        hits = []
        for name in self.names:
            for m in re.finditer(re.escape(name), text):
                hits.append(type("R", (), {"entity_type": "PERSON", "start": m.start(), "end": m.end(), "score": 0.9}))
        return hits


def reducer(*names) -> NameReducer:
    return NameReducer(PresidioEngine(PersonAnalyzer(*names)), min_score=0.5)


@pytest.mark.parametrize("full,short", [
    ("John Doe", "John D."), ("Asha Menon", "Asha M."), ("John Michael Doe", "John D."),
    ("Priya", "Priya"), ("John D.", "John D."), ("John D", "John D."), ("Anne-Marie O'Neil", "Anne-Marie O."),
])
def test_reduce_name(full, short):
    assert reduce_name(full) == short


def test_two_full_names_are_both_reduced():
    out = reducer("John Doe", "Asha Menon").reduce("John Doe in my team shouts; my manager Asha Menon agrees.")
    assert out.text == "John D. in my team shouts; my manager Asha M. agrees."
    assert out.count == 2
    assert "Doe" not in out.text and "Menon" not in out.text


def test_a_single_name_stays_as_it_is():
    out = reducer("Priya").reduce("Priya from payroll told me to ask about my payslip.")
    assert out.text == "Priya from payroll told me to ask about my payslip." and out.count == 0


def test_reducing_twice_changes_nothing_more():
    r = reducer("John Doe", "John D")
    once = r.reduce("Ask John Doe, then John D. again.").text
    assert once == "Ask John D., then John D. again."
    assert r.reduce(once).text == once


def test_possessive_and_sentence_end():
    out = reducer("John Doe").reduce("It is John Doe's fault. I told John Doe.")
    assert out.text == "It is John D.'s fault. I told John D."


def test_asks_presidio_for_person_only_with_the_configured_score():
    analyzer = PersonAnalyzer()
    NameReducer(PresidioEngine(analyzer), min_score=0.7).reduce("hello")
    assert analyzer.calls == [(("PERSON",), 0.7)]


def test_without_presidio_names_are_skipped_and_nothing_raises():
    out = NameReducer(None, min_score=0.5).reduce("John Doe called")
    assert out.text == "John Doe called" and out.count == 0


def test_a_failing_engine_leaves_the_text_alone_and_warns(caplog):
    class Boom:
        def analyze(self, *a, **k):
            raise RuntimeError("nlp exploded")

    with caplog.at_level(logging.WARNING):
        out = NameReducer(PresidioEngine(Boom()), min_score=0.5).reduce("John Doe called")
    assert out.text == "John Doe called" and "names" in caplog.text.lower()


def test_names_are_not_logged_when_reduced(caplog):
    with caplog.at_level(logging.DEBUG):
        reducer("John Doe").reduce("John Doe called")
    assert "Doe" not in caplog.text


def test_kb_side_code_never_reduces_names():
    """Names are reduced on user input only; retrieved KB text is not altered (FR-H1)."""
    root = Path(__file__).resolve().parent.parent / "smartdesk_ai"
    for folder in ("rag", "tools", "tickets"):
        for path in (root / folder).glob("*.py"):
            assert "guardrails.names" not in path.read_text(), path


@pytest.mark.skipif(not os.environ.get("RUN_REAL_MODEL_TESTS"), reason="set RUN_REAL_MODEL_TESTS=1")
def test_real_presidio_reduces_two_names_and_keeps_a_single_one():
    out = NameReducer(load_presidio_engine(), min_score=0.5).reduce(
        "My manager Asha Menon approved it, but John Doe disagrees. Priya knows.")
    assert "Asha M." in out.text and "John D." in out.text and "Menon" not in out.text and "Doe" not in out.text
