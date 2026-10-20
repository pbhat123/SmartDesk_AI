"""Toxicity check through the OpenAI Moderation API (FR-H1, FR-H2), behind a small interface.

`OpenAIModerator` is the live one; `KeywordModerator` is the offline stand-in used when `modes.llm` is
mock, so the CLI and the tests make no network call. A moderation outage never stops the service: the
guard treats it as "not flagged" and logs a warning (FR-F7).
"""
import re
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class ModerationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    flagged: bool
    categories: list[str] = Field(default_factory=list)


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
    "harassment": re.compile(
        r"\b(worthless|disgusting|scum|pathetic loser|go to hell|hope you die)\b", re.I),
    "violence": re.compile(r"\b(hurt|kill|punch|attack) (?:someone|somebody|him|her|them|you)\b", re.I),
}


MOCK_MODERATION_NOTE = "offline keyword moderator instead of the OpenAI Moderation API"  # shown on Monitoring in mock mode


class KeywordModerator:
    """A few word rules that stand in for the API in mock mode."""

    def check(self, text: str) -> ModerationResult:
        hits = sorted(name for name, rule in _KEYWORD_RULES.items() if rule.search(text))
        return ModerationResult(flagged=bool(hits), categories=hits)
