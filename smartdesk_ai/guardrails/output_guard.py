"""The output guard (FR-H2): runs on every reply before it is shown.

Order: Pydantic validation of the reply, PII scan (non-name PII only, email allowed), Moderation. The
grounding judge is not here: it runs inside the answering step (T5.7). Names are never touched on output.
The company's own contact numbers, which the KB legitimately contains, are not treated as PII.
Reply text is never logged; only the action taken is.
"""
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from smartdesk_ai.config.env import DEFAULT_DOTENV, load_env
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.guardrails.moderation import MOCK_MODERATION_NOTE, KeywordModerator, Moderator, OpenAIModerator
from smartdesk_ai.guardrails.pii import (
    PiiRedactor,
    PresidioEngine,
    digits_of,
    make_presidio_engine,
    mask_emails_in,
    regex_spans,
)
from smartdesk_ai.ops.health import Component, HealthRegistry

logger = logging.getLogger(__name__)

BLOCKED_REPLY = "Sorry, I can't share that reply. Could you rephrase your question, or would you like me to raise a ticket?"
INVALID_REPLY = "Sorry, something went wrong while preparing my reply. Please try again."


class InvalidReply(ValueError):
    """The final reply is not a non-empty text."""


class FinalReply(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reply text is blank")
        return value


def validate_reply(candidate: Any) -> FinalReply:
    """A `FinalReply` from a string, a dict with a `text`, or a `FinalReply`; anything else raises."""
    try:
        if isinstance(candidate, FinalReply):
            return candidate
        if isinstance(candidate, str):
            return FinalReply(text=candidate)
        if isinstance(candidate, Mapping):
            return FinalReply.model_validate(dict(candidate))
    except ValidationError as error:
        raise InvalidReply("reply failed validation") from error
    raise InvalidReply(f"reply has an unsupported type: {type(candidate).__name__}")


class OutputDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    actions: list[str] = []  # what changed the reply: "pii", "moderation:<category>", "schema"
    rejected: bool = False


def kb_contact_numbers(kb_dir: Path) -> frozenset[str]:
    """Digit strings of the phone-like numbers printed in the KB (switchboard, payroll line)."""
    found: set[str] = set()
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text()
        found.update(digits_of(text[s.start:s.end]) for s in regex_spans(text) if s.kind == "PHONE")
    return frozenset(found)


class OutputGuard:
    def __init__(self, pii: PiiRedactor, moderator: Moderator, keep_numbers: frozenset[str] = frozenset(),
                 registry: HealthRegistry | None = None) -> None:
        self._registry = registry
        self._pii = pii
        self._moderator = moderator
        self._keep = keep_numbers

    def check(self, candidate: Any) -> OutputDecision:
        try:
            text = validate_reply(candidate).text
        except InvalidReply:
            logger.warning("output rejected: reason=schema")
            return OutputDecision(text=INVALID_REPLY, actions=["schema"], rejected=True)
        actions: list[str] = []
        redacted = self._pii.redact(text, self._keep)
        if redacted.kinds:
            actions.append("pii")
            logger.info("output redacted: pii=%s", sorted(set(redacted.kinds)))
        text = redacted.text
        try:
            result = self._moderator.check(mask_emails_in(text))
            if self._registry is not None:
                if isinstance(self._moderator, KeywordModerator):
                    self._registry.mark_mock(Component.MODERATION, MOCK_MODERATION_NOTE)
                else:
                    self._registry.mark_ok(Component.MODERATION)
        except Exception as error:  # an outage must not stop the service (FR-F7)
            if self._registry is not None:
                self._registry.mark_degraded(Component.MODERATION, type(error).__name__)
            logger.warning("output moderation unavailable (%s); reply allowed", type(error).__name__)
            return OutputDecision(text=text, actions=actions)
        if result.flagged:
            reason = f"moderation:{','.join(result.categories) or 'flagged'}"
            logger.info("output replaced: reason=%s", reason)
            return OutputDecision(text=BLOCKED_REPLY, actions=actions + [reason])
        return OutputDecision(text=text, actions=actions)

    def __call__(self, text: str) -> str:
        return self.check(text).text


def build_output_guard(
    settings: Settings,
    kb_dir: Path,
    *,
    environ: Mapping[str, str] | None = None,
    dotenv_path: Path | None = DEFAULT_DOTENV,
    registry: HealthRegistry | None = None,
    engine: PresidioEngine | None = None,
    moderator: Moderator | None = None,
) -> OutputGuard:
    """Same mode rules as the input guard: mock mode uses the offline moderator."""
    if engine is None:
        engine = make_presidio_engine(registry)
    if moderator is None:
        env = load_env(environ=environ, dotenv_path=dotenv_path)
        if settings.modes.llm == "live" and env.openai_api_key is not None:
            from openai import OpenAI

            moderator = OpenAIModerator(OpenAI(api_key=env.openai_api_key), settings.guardrails.moderation_model)
        else:
            moderator = KeywordModerator()
    return OutputGuard(PiiRedactor(engine, settings.guardrails.pii_min_score), moderator, kb_contact_numbers(kb_dir),
                       registry)
