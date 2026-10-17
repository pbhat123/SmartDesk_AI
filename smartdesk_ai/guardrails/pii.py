"""PII detection and redaction (FR-H1, FR-G11).

Phone numbers, card numbers and government IDs are replaced by placeholders before the LLM call, the
logs, the traces and the ticket body. Email addresses are allowed (the ticket needs one) and are left
alone here; `mask_email` hides them wherever they would be logged. Regex checks always run. Presidio,
when it loads, adds hits on top; if it is missing or fails the regex checks still work and the degraded
mode is recorded (FR-F7). Names are handled in `names.py`.
"""
import logging
import re
from functools import lru_cache
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from smartdesk_ai.ops.health import Component, HealthRegistry, get_registry

logger = logging.getLogger(__name__)

SPACY_MODEL = "en_core_web_lg"
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")

PLACEHOLDERS = {"PHONE": "[PHONE]", "CARD": "[CARD]", "ID": "[ID]"}

# Presidio entity name -> our kind. EMAIL_ADDRESS is left out on purpose: email is an allowed field.
# Aadhaar and PAN are left to the regex checks: the default English analyzer has no recognizer for them.
PRESIDIO_KINDS = {
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CARD",
    "US_SSN": "ID",
    "US_PASSPORT": "ID",
    "US_DRIVER_LICENSE": "ID",
    "IBAN_CODE": "ID",
}

_CARD = re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])")
_AADHAAR = re.compile(r"(?<![\w-])\d{4}[ -]?\d{4}[ -]?\d{4}(?![\w-])")
_PAN = re.compile(r"(?<![A-Za-z0-9])[A-Z]{5}\d{4}[A-Z](?![A-Za-z0-9])")
_SSN = re.compile(r"(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])")
_PHONE = re.compile(r"(?<![\w-])\+?\(?\d[\d ().-]{8,17}\d(?![\w-])")
_MIN_PHONE_DIGITS, _MAX_PHONE_DIGITS = 10, 15


class Span(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: int
    end: int
    kind: str


class Redaction(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    kinds: list[str] = Field(default_factory=list)  # one entry per redacted item, never the values


def mask_email(email: str) -> str:
    """"priya.k@smartdesk.example" -> "p***@smartdesk.example". For logs and traces (FR-G11)."""
    local, _, domain = email.partition("@")
    return f"{local[:1]}***@{domain}"


def mask_emails_in(text: str) -> str:
    return EMAIL.sub(lambda m: mask_email(m.group()), text)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _luhn_ok(number: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(number)):
        digit = int(char)
        if index % 2:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


def regex_spans(text: str) -> list[Span]:
    """Card, then ID, then phone; a later pattern never re-reports text an earlier one covered."""
    found: list[Span] = []

    def free(match: re.Match[str]) -> bool:
        return all(match.end() <= s.start or match.start() >= s.end for s in found)

    for match in _CARD.finditer(text):
        if 13 <= len(_digits(match.group())) <= 19 and _luhn_ok(_digits(match.group())) and free(match):
            found.append(Span(start=match.start(), end=match.end(), kind="CARD"))
    for pattern in (_AADHAAR, _PAN, _SSN):
        for match in pattern.finditer(text):
            if free(match):
                found.append(Span(start=match.start(), end=match.end(), kind="ID"))
    for match in _PHONE.finditer(text):
        if _MIN_PHONE_DIGITS <= len(_digits(match.group())) <= _MAX_PHONE_DIGITS and free(match):
            found.append(Span(start=match.start(), end=match.end(), kind="PHONE"))
    return found


class Analyzer(Protocol):
    def analyze(self, text: str, language: str, entities: list[str], score_threshold: float | None = None) -> Any: ...


class PresidioEngine:
    """A thin wrapper so the rest of the code never imports Presidio types."""

    def __init__(self, analyzer: Analyzer) -> None:
        self._analyzer = analyzer

    def find(self, text: str, entities: Sequence[str], min_score: float) -> list[tuple[str, int, int]]:
        results = self._analyzer.analyze(text=text, language="en", entities=list(entities), score_threshold=min_score)
        return [(r.entity_type, r.start, r.end) for r in results]


@lru_cache(maxsize=1)
def _analyzer_engine() -> Any:
    from presidio_analyzer import AnalyzerEngine

    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)  # it warns about every unused language recognizer
    return AnalyzerEngine()  # loads the spaCy model once per process


def load_presidio_engine() -> PresidioEngine:
    """Build the real analyzer. Raises if Presidio or the spaCy model is missing; it never downloads
    the model itself (Presidio would otherwise fetch it silently)."""
    import spacy.util

    if not spacy.util.is_package(SPACY_MODEL):
        raise RuntimeError(f"spaCy model {SPACY_MODEL} is not installed (python -m spacy download {SPACY_MODEL})")
    return PresidioEngine(_analyzer_engine())


def make_presidio_engine(
    registry: HealthRegistry | None = None, loader: Callable[[], PresidioEngine] = load_presidio_engine
) -> PresidioEngine | None:
    """The engine, or None with the degraded mode recorded and one warning logged (FR-F7)."""
    return (registry if registry is not None else get_registry()).try_load(Component.PRESIDIO, loader)


def _merge(spans: list[Span]) -> list[Span]:
    merged: list[Span] = []
    for span in sorted(spans, key=lambda s: (s.start, -s.end)):
        if merged and span.start < merged[-1].end:
            last = merged[-1]
            merged[-1] = Span(start=last.start, end=max(last.end, span.end), kind=last.kind)
        else:
            merged.append(span)
    return merged


class PiiRedactor:
    def __init__(self, engine: PresidioEngine | None, min_score: float) -> None:
        self._engine = engine
        self._min_score = min_score

    def _presidio_spans(self, text: str) -> list[Span]:
        if self._engine is None:
            return []
        try:
            hits = self._engine.find(text, list(PRESIDIO_KINDS), self._min_score)
        except Exception as error:  # a failing engine must not lose the regex protection (FR-F7)
            logger.warning("presidio failed at call time (%s); regex checks only", type(error).__name__)
            return []
        return [Span(start=s, end=e, kind=PRESIDIO_KINDS[entity]) for entity, s, e in hits if entity in PRESIDIO_KINDS]

    def _redact_segment(self, text: str) -> Redaction:
        spans = _merge(regex_spans(text) + self._presidio_spans(text))
        out, cursor = [], 0
        for span in spans:
            out += [text[cursor:span.start], PLACEHOLDERS[span.kind]]
            cursor = span.end
        out.append(text[cursor:])
        return Redaction(text="".join(out), kinds=[s.kind for s in spans])

    def redact(self, text: str) -> Redaction:
        """Redact everything but email addresses, which are cut out first and put back unchanged."""
        pieces: list[str] = []
        kinds: list[str] = []
        cursor = 0
        for match in EMAIL.finditer(text):
            part = self._redact_segment(text[cursor:match.start()])
            pieces += [part.text, match.group()]
            kinds += part.kinds
            cursor = match.end()
        tail = self._redact_segment(text[cursor:])
        return Redaction(text="".join(pieces) + tail.text, kinds=kinds + tail.kinds)
