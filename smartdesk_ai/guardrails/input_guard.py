"""The input guard (FR-H1): runs on every user message before the agent graph.

Order: PII redaction (email kept), name reduction, Moderation. Each later step sees only the redacted
text, so raw PII never reaches the Moderation API, a log or a trace. A blocked message gets a polite
refusal and the reason is logged; the text is never logged. There is no topic filter here: that is the
orchestrator's `out_of_scope` intent (FR-D9).
"""
import logging

from smartdesk_ai.guardrails.decision import GuardDecision
from smartdesk_ai.guardrails.moderation import Moderator
from smartdesk_ai.guardrails.names import NameReducer
from smartdesk_ai.guardrails.pii import PiiRedactor, mask_emails_in

logger = logging.getLogger(__name__)

MODERATION_REFUSAL = (
    "Sorry, I can't help with that message. Please keep our conversation respectful and I'll be glad to "
    "help with an IT or HR question."
)


class InputGuard:
    def __init__(self, pii: PiiRedactor, names: NameReducer, moderator: Moderator) -> None:
        self._pii = pii
        self._names = names
        self._moderator = moderator

    def __call__(self, text: str) -> GuardDecision:
        redacted = self._pii.redact(text)
        reduced = self._names.reduce(redacted.text)
        safe = reduced.text
        if redacted.kinds or reduced.count:
            logger.info("input redacted: pii=%s names=%d", sorted(set(redacted.kinds)), reduced.count)
        reason = self._moderation_reason(safe)
        if reason is not None:
            return self._block(safe, MODERATION_REFUSAL, reason)
        return GuardDecision(text=safe)

    def _moderation_reason(self, safe_text: str) -> str | None:
        try:
            result = self._moderator.check(mask_emails_in(safe_text))
        except Exception as error:  # an outage must not stop the service (FR-F7)
            logger.warning("moderation unavailable (%s); message allowed", type(error).__name__)
            return None
        return f"moderation:{','.join(result.categories) or 'flagged'}" if result.flagged else None

    @staticmethod
    def _block(safe_text: str, refusal: str, reason: str) -> GuardDecision:
        logger.info("input blocked: reason=%s", reason)
        return GuardDecision(text=safe_text, blocked=True, refusal=refusal, reason=reason)
