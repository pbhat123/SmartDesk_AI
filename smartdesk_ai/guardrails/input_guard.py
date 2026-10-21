"""The input guard (FR-H1): runs on every user message before the agent graph.

Order: PII redaction (email kept), name reduction, self-harm check, violent-threat rule, Moderation, injection check. Each later step sees only the redacted
text, so raw PII never reaches the Moderation API, a log or a trace. A blocked message gets a polite
refusal and the reason is logged; the text is never logged. There is no topic filter here: that is the
orchestrator's `out_of_scope` intent (FR-D9).
"""
import logging
from collections.abc import Mapping
from pathlib import Path

from smartdesk_ai.config.env import DEFAULT_DOTENV, load_env
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.guardrails.decision import GuardDecision
from smartdesk_ai.guardrails.injection import REFUSAL as INJECTION_REFUSAL
from smartdesk_ai.guardrails.injection import InjectionChecker, make_injection_checker
from smartdesk_ai.guardrails.moderation import MOCK_MODERATION_NOTE, KeywordModerator, Moderator, make_openai_moderator
from smartdesk_ai.guardrails.names import NameReducer
from smartdesk_ai.guardrails.self_harm import SELF_HARM_REPLY, WITHHELD, is_self_harm
from smartdesk_ai.guardrails.threats import is_violent_threat
from smartdesk_ai.guardrails.pii import PiiRedactor, PresidioEngine, make_presidio_engine, mask_emails_in
from smartdesk_ai.llm.factory import ModelBuilder, openai_builder
from smartdesk_ai.llm.mock import MockInjectionModel
from smartdesk_ai.ops.health import Component, HealthRegistry

logger = logging.getLogger(__name__)

MODERATION_REFUSAL = (
    "Sorry, I can't help with that message. Please keep our conversation respectful and I'll be glad to "
    "help with an IT or HR question."
)
THREAT_REFUSAL = (
    "I can't help with a message that threatens someone. If things are getting this tense, please talk to "
    "your manager or the People Team. I'm glad to help with an IT or HR question whenever you're ready."
)


class InputGuard:
    def __init__(
        self, pii: PiiRedactor, names: NameReducer, moderator: Moderator, injection: InjectionChecker | None = None,
        registry: HealthRegistry | None = None,
    ) -> None:
        self._registry = registry
        self._pii = pii
        self._names = names
        self._moderator = moderator
        self._injection = injection

    @property
    def moderator(self) -> Moderator:
        return self._moderator

    def __call__(self, text: str) -> GuardDecision:
        redacted = self._pii.redact(text)
        reduced = self._names.reduce(redacted.text)
        safe = reduced.text
        if redacted.kinds or reduced.count:
            logger.info("input redacted: pii=%s names=%d", sorted(set(redacted.kinds)), reduced.count)
        if is_self_harm(safe):  # first, and the statement itself is kept out of every log and trace
            logger.info("input blocked: reason=self_harm")
            return GuardDecision(text=WITHHELD, blocked=True, refusal=SELF_HARM_REPLY, reason="self_harm")
        if is_violent_threat(safe):
            return self._block(safe, THREAT_REFUSAL, "threat:violence")
        reason, calm = self._moderation_reason(safe)
        if reason is not None:
            return self._block(safe, MODERATION_REFUSAL, reason)
        if self._injection is not None:
            verdict = self._injection.check(mask_emails_in(safe))
            if verdict.injected:
                return self._block(safe, INJECTION_REFUSAL, verdict.reason or "injection")
        return GuardDecision(text=safe, calm=calm)

    def _moderation_reason(self, safe_text: str) -> tuple[str | None, bool]:
        """(block reason, calm). Plain harassment is not blocked: the message goes on and the reply is kept calm (FR-H1)."""
        try:
            result = self._moderator.check(mask_emails_in(safe_text))
            if self._registry is not None:
                if isinstance(self._moderator, KeywordModerator):
                    self._registry.mark_mock(Component.MODERATION, MOCK_MODERATION_NOTE)
                else:
                    self._registry.mark_ok(Component.MODERATION)
        except Exception as error:  # an outage must not stop the service (FR-F7)
            if self._registry is not None:
                self._registry.mark_degraded(Component.MODERATION, type(error).__name__)
            logger.warning("moderation unavailable (%s); message allowed", type(error).__name__)
            return None, False
        if result.harassment_only:
            logger.info("input flagged: reason=moderation:harassment (only); allowed, reply kept calm")
            return None, True
        return (f"moderation:{','.join(result.categories) or 'flagged'}" if result.flagged else None), False

    @staticmethod
    def _block(safe_text: str, refusal: str, reason: str) -> GuardDecision:
        logger.info("input blocked: reason=%s", reason)
        return GuardDecision(text=safe_text, blocked=True, refusal=refusal, reason=reason)


def build_input_guard(
    settings: Settings,
    *,
    environ: Mapping[str, str] | None = None,
    dotenv_path: Path | None = DEFAULT_DOTENV,
    registry: HealthRegistry | None = None,
    engine: PresidioEngine | None = None,
    moderator: Moderator | None = None,
    injection_builder: ModelBuilder | None = None,
) -> InputGuard:
    """The guard for the configured mode. Mock mode uses offline stand-ins for Moderation and the
    classifier; live mode uses the OpenAI Moderation API and the Light-Model classifier. A missing
    Presidio degrades to regex PII checks and no name reduction, and never raises (FR-F7).
    `engine` lets a caller (or a test) supply Presidio instead of loading it."""
    live = settings.modes.llm == "live"
    if engine is None:
        engine = make_presidio_engine(registry)
    if moderator is None:
        env = load_env(environ=environ, dotenv_path=dotenv_path)
        if live and env.openai_api_key is not None:
            moderator = make_openai_moderator(env.openai_api_key, settings.guardrails.moderation_model)
        else:
            moderator = KeywordModerator()
    if injection_builder is None:
        injection_builder = openai_builder if live else (lambda model_id, cfg: MockInjectionModel())
    min_score = settings.guardrails.pii_min_score
    return InputGuard(
        pii=PiiRedactor(engine, min_score),
        names=NameReducer(engine, min_score),
        moderator=moderator,
        injection=make_injection_checker(settings, builder=injection_builder),
        registry=registry,
    )
