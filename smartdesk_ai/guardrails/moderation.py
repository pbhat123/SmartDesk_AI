"""Toxicity check through the OpenAI Moderation API (FR-H1, FR-H2), behind a small interface.

`OpenAIModerator` is the live one; `KeywordModerator` is the offline stand-in used when `modes.llm` is
mock, so the CLI and the tests make no network call. A moderation outage never stops the service: the
guard treats it as "not flagged" and logs a warning (FR-F7).
"""
import logging
import re
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from smartdesk_ai.ops.health import Component, HealthRegistry

logger = logging.getLogger(__name__)


PLAIN_HARASSMENT = "harassment"  # the one category that is answered calmly instead of refused (FR-H1)


class ModerationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    flagged: bool
    categories: list[str] = Field(default_factory=list)

    @property
    def harassment_only(self) -> bool:
        """Flagged for plain "harassment" and nothing else: no threatening, hate, sexual, violence, self-harm or illicit category."""
        return self.flagged and self.categories == [PLAIN_HARASSMENT]


class Moderator(Protocol):
    def check(self, text: str) -> ModerationResult: ...


class OpenAIModerator:
    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def check(self, text: str) -> ModerationResult:
        result = self._client.moderations.create(model=self._model, input=text).results[0]
        if not result.flagged:
            return ModerationResult(flagged=False)
        categories = result.categories
        names = categories.model_dump() if hasattr(categories, "model_dump") else dict(categories)
        return ModerationResult(flagged=True, categories=sorted(name for name, hit in names.items() if hit))


def make_openai_moderator(api_key: SecretStr, model: str) -> OpenAIModerator:
    """The live moderator. The SDK needs the plain key: a SecretStr would be sent as its masked text and get a 401."""
    from openai import OpenAI

    return OpenAIModerator(OpenAI(api_key=api_key.get_secret_value()), model)


_KEYWORD_RULES = {
    "harassment": re.compile(r"\b(worthless|disgusting|scum|pathetic loser)\b", re.I),
    "harassment_threatening": re.compile(r"\b(go to hell|hope you die)\b", re.I),
    "violence": re.compile(r"\b(hurt|kill|punch|attack) (?:someone|somebody|him|her|them|you)\b", re.I),
}


PROBE_TEXT = "Health check: how do I reset my password?"  # harmless; the probe only needs the call to succeed


def probe_moderation(moderator: Moderator, registry: HealthRegistry) -> None:
    """At startup: one free Moderation call in live mode, so the status reads ok or degraded from the first screen. The
    keyword stand-in makes no call and is marked mock at once. A failure never raises (FR-F7); it is recorded by error type."""
    if not isinstance(moderator, OpenAIModerator):
        registry.mark_mock(Component.MODERATION, MOCK_MODERATION_NOTE)
        return
    try:
        moderator.check(PROBE_TEXT)
    except Exception as error:
        registry.mark_degraded(Component.MODERATION, type(error).__name__)
        logger.warning("moderation probe failed (%s); messages and replies will pass unchecked", type(error).__name__)
    else:
        registry.mark_ok(Component.MODERATION)


MOCK_MODERATION_NOTE = "offline keyword moderator instead of the OpenAI Moderation API"  # shown on Monitoring in mock mode


class KeywordModerator:
    """A few word rules that stand in for the API in mock mode."""

    def check(self, text: str) -> ModerationResult:
        hits = sorted(name for name, rule in _KEYWORD_RULES.items() if rule.search(text))
        return ModerationResult(flagged=bool(hits), categories=hits)
