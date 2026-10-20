"""Proof, per row, that Moderation really ran in an eval (FR-H1, FR-H2).

A moderation outage is deliberately silent in the app (the text is let through and the health registry shows
"degraded"). An eval must not pass over it: a run whose moderation failed would measure only the other layers.
`RecordingModerator` wraps the real moderator and counts what each row did; `row_status` turns the counts into the
row's recorded status or raises `ModerationDegraded`, which stops the run with a plain message.
"""
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

from smartdesk_ai.config.env import load_env
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.guardrails.moderation import KeywordModerator, Moderator, ModerationResult, make_openai_moderator

BEFORE_MODERATION = ("self_harm", "threat:violence")  # blocked by a local rule before the moderator is asked


class ModerationDegraded(RuntimeError):
    """Moderation failed (or never ran) during an eval row. The message says which row and what the error type was."""


class Counts(NamedTuple):
    calls: int
    errors: int
    flagged: int


class RecordingModerator:
    """Passes every check to `inner` and counts calls, errors (by type) and flagged results."""

    def __init__(self, inner: Moderator, *, live: bool) -> None:
        self.inner, self.live = inner, live
        self.calls = self.errors = self.flagged = 0
        self.error_types: list[str] = []

    def check(self, text: str) -> ModerationResult:
        self.calls += 1
        try:
            result = self.inner.check(text)
        except Exception as error:
            self.errors += 1
            self.error_types.append(type(error).__name__)
            raise
        self.flagged += result.flagged
        return result

    def counts(self) -> Counts:
        return Counts(self.calls, self.errors, self.flagged)


def eval_moderator(settings: Settings, *, environ: Mapping[str, str] | None, dotenv_path: Path | None) -> RecordingModerator:
    """The moderator the app would use in this mode: the OpenAI API in live mode when a key exists, else the keyword stand-in."""
    if settings.modes.llm == "live":
        key = load_env(environ=environ, dotenv_path=dotenv_path).openai_api_key
        if key is not None:
            return RecordingModerator(make_openai_moderator(key, settings.guardrails.moderation_model), live=True)
    return RecordingModerator(KeywordModerator(), live=False)


def row_status(row_id: str, before: Counts, moderator: RecordingModerator, *, blocked_reason: str | None) -> tuple[str, Counts]:
    """The status to record for one row: "ok" (the real API ran), "mock" (the keyword stand-in ran) or "not_reached"
    (a local rule blocked the row before moderation). Raises `ModerationDegraded` if a call failed or none was made."""
    now = moderator.counts()
    delta = Counts(now.calls - before.calls, now.errors - before.errors, now.flagged - before.flagged)
    if delta.errors:
        kinds = ", ".join(sorted(set(moderator.error_types)))
        raise ModerationDegraded(f"moderation was degraded on row {row_id} ({delta.errors} of {delta.calls} calls failed: {kinds}). "
                                 "The row's result would not include the moderation layer, so the run stops here.")
    if delta.calls == 0:
        if blocked_reason in BEFORE_MODERATION:
            return "not_reached", delta
        raise ModerationDegraded(f"moderation did not run on row {row_id} (no call was made), so the run stops here.")
    return ("ok" if moderator.live else "mock"), delta
