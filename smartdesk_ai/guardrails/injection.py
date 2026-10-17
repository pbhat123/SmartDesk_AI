"""Prompt-injection check (FR-H1): a cheap heuristic score first, the LLM classifier only above the low
threshold from config (`thresholds.injection_heuristic`), so most messages cost no model call.

The same check screens team-update text from ticket comments (FR-C9). That text comes from a public repo,
so it is scanned before it can reach a model or the employee, and an update that looks like an injection is
replaced by a fixed notice. Whatever passes still travels only through `TeamUpdate.quoted()`.
"""
import logging
import re
import time
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.factory import ModelBuilder, openai_builder
from smartdesk_ai.llm.fallback import FallbackChat
from smartdesk_ai.llm.structured import parse_model_reply
from smartdesk_ai.tickets import TeamUpdate

logger = logging.getLogger(__name__)

UPDATE_WITHHELD = "(This update was withheld by a safety check. Please ask the support team directly.)"
REFUSAL = (
    "Sorry, I can't act on that message because it looks like an attempt to change how I work. "
    "I'm happy to help with an IT or HR question or a ticket."
)

_WORDS = r"(?:all |any |the |your |my |every |these |those )*"
_TARGET = r"(?:previous |prior |above |earlier |preceding |original |system |company )*"
_NOUN = r"(?:instructions?|rules|guidelines|prompts?|policies|polic(?:y)|programming|knowledge base|restrictions)"

# (pattern, weight). Weights combine as 1 - product(1 - w): one strong sign clears the threshold, one weak
# sign does not, and two weak signs together do.
_SIGNS: tuple[tuple[re.Pattern[str], float], ...] = tuple(
    (re.compile(p, re.I), w) for p, w in (
        (rf"\b(?:ignore|disregard|forget|override|bypass|drop)\b {_WORDS}{_TARGET}{_NOUN}", 0.6),
        (r"\b(?:system|hidden|initial|secret|original) (?:prompt|instructions?|message)\b", 0.5),
        (r"\b(?:developer|debug|admin|god|jailbreak|dan) mode\b|\bjailbreak\b", 0.6),
        (r"\b(?:reveal|print|show|output|repeat|display|leak|tell me)\b[^.\n]{0,40}\b(?:prompt|instructions)\b", 0.5),
        (r"\b(?:output|print|repeat|show)\b[^.\n]{0,30}\b(?:text|message|everything)\b[^.\n]{0,20}\babove\b", 0.5),
        (r"^\s*[\[<(]?\s*(?:system|admin|assistant|developer)\s*[\]>):]", 0.5),
        (r"\byou are now\b|\bfrom now on\b|\bpretend (?:to be|you)\b|\bact as (?:if|an?)\b", 0.3),
        (r"\bnew (?:policy|instructions?|rules?)\s*:", 0.3),
    )
)


def heuristic_score(text: str) -> float:
    """0.0 (no sign of injection) to 1.0. Pattern matching only: no model call."""
    remaining = 1.0
    for pattern, weight in _SIGNS:
        if pattern.search(text):
            remaining *= 1.0 - weight
    return round(1.0 - remaining, 4)


class ClassifierReply(BaseModel):
    """The exact shape the INJECTION_CLASSIFIER model must return (FR-H2)."""

    model_config = ConfigDict(extra="forbid")

    injection: bool
    reason: str | None = None


Classifier = Callable[[str], bool]


def make_injection_classifier(
    settings: Settings,
    *,
    builder: ModelBuilder = openai_builder,
    budget_level: int = 0,
    sleep: Callable[[float], None] = time.sleep,
) -> Classifier:
    """True when the INJECTION_CLASSIFIER task (Light-Model) calls the text an injection. Raises when the
    model is unavailable or answers malformed JSON; `InjectionChecker` treats that as a block."""
    llm = FallbackChat(
        "INJECTION_CLASSIFIER", settings, builder=builder, budget_level=budget_level,
        schema=ClassifierReply if builder is openai_builder else None, sleep=sleep,
    )
    prompt = load_prompt("injection_classifier", settings).text

    def classify(text: str) -> bool:
        human = f"Untrusted text to judge:\n<<<TEXT\n{text}\nTEXT>>>"
        reply = llm.invoke([SystemMessage(prompt), HumanMessage(human)])
        return parse_model_reply(ClassifierReply, reply.content).injection

    return classify


class InjectionVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float
    classified: bool = False  # the LLM classifier was called
    injected: bool = False
    reason: str | None = None


class InjectionChecker:
    def __init__(self, threshold: float, classifier: Classifier) -> None:
        self._threshold = threshold
        self._classifier = classifier

    def check(self, text: str) -> InjectionVerdict:
        score = heuristic_score(text)
        if score <= self._threshold:
            return InjectionVerdict(score=score)
        try:
            injected = self._classifier(text)
        except Exception as error:  # the score was already suspicious, so an outage blocks (FR-F2)
            logger.warning("injection classifier unavailable (%s); message blocked", type(error).__name__)
            return InjectionVerdict(score=score, classified=True, injected=True,
                                    reason="injection:classifier_unavailable")
        return InjectionVerdict(score=score, classified=True, injected=injected,
                                reason="injection:classifier" if injected else None)


def screen_team_update(update: TeamUpdate | None, checker: InjectionChecker) -> TeamUpdate | None:
    """The update itself if it is clean, a fixed notice if it looks like an injection (FR-C9)."""
    if update is None:
        return None
    verdict = checker.check(update.text)
    if not verdict.injected:
        return update
    logger.warning("team update withheld: reason=%s", verdict.reason)
    return TeamUpdate(text=UPDATE_WITHHELD)
