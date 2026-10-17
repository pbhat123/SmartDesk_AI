"""Small talk, deflection and the clarifying question (FR-D3, FR-D4, FR-D5; PLAN 4.2).

The CHITCHAT task (Light-Model, FR-H17) writes the reply; no tool is called. Any model failure or
unusable reply falls back to the fixed wording below, so a greeting never fails (FR-F7). The
acknowledgement that opens a ticket offer is fixed text too: it needs no model call (FR-D4).
"""
import re
import time
from collections.abc import Callable
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.agent.router import CLARIFYING_QUESTION, DEFLECTION_REPLY, SMALLTALK_REPLY
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.factory import ModelBuilder, openai_builder
from smartdesk_ai.llm.fallback import FallbackChat

Situation = Literal["smalltalk", "deflect", "clarify"]
Kind = Literal["greeting", "thanks", "goodbye", "other", "deflect", "clarify"]
Chitchat = Callable[[Situation, str], str]  # (situation, employee message) -> reply

ACKNOWLEDGEMENT_PHRASES = ("I'm sorry you're running into this.",)
REPLY_MAX_CHARS = 400

_GOODBYE = re.compile(r"\b(bye|goodbye|see you|take care|talk later|cheers)\b", re.I)
_THANKS = re.compile(r"\b(thanks|thank you|thx|much appreciated|appreciate it)\b", re.I)
_GREETING = re.compile(r"^\W*(hi|hello|hey|good (morning|afternoon|evening)|how are you)\b", re.I)

_STATIC: dict[Kind, str] = {
    "greeting": SMALLTALK_REPLY,
    "thanks": "You're welcome! Is there anything else I can help with?",
    "goodbye": "Goodbye, and take care! Come back any time you need help.",
    "other": SMALLTALK_REPLY,
    "deflect": DEFLECTION_REPLY,
    "clarify": CLARIFYING_QUESTION,
}


def smalltalk_kind(text: str) -> Kind:
    """Which kind of small talk a message is. A goodbye wins over thanks ("thanks, bye")."""
    if _GOODBYE.search(text):
        return "goodbye"
    if _THANKS.search(text):
        return "thanks"
    return "greeting" if _GREETING.match(text) else "other"


def _kind(situation: Situation, text: str) -> Kind:
    return smalltalk_kind(text) if situation == "smalltalk" else situation


def static_chitchat(situation: Situation, text: str) -> str:
    """The fixed reply for a situation; what the model-free graph and every fallback say."""
    return _STATIC[_kind(situation, text)]


def usable(kind: Kind, reply: str) -> bool:
    """A model reply is shown only if it is short plain text with no email, no ticket offer in a
    deflection (FR-D9) and exactly one question when clarifying (FR-D3)."""
    if not reply.strip() or len(reply) > REPLY_MAX_CHARS or "@" in reply:
        return False
    if kind == "deflect" and "ticket" in reply.lower():
        return False
    return kind != "clarify" or reply.count("?") == 1


def make_chitchat(
    settings: Settings,
    *,
    builder: ModelBuilder = openai_builder,
    budget_level: int = 0,
    sleep: Callable[[float], None] = time.sleep,
) -> Chitchat:
    """A chitchat function on the CHITCHAT model and the active chitchat prompt."""
    llm = FallbackChat("CHITCHAT", settings, builder=builder, budget_level=budget_level, sleep=sleep)
    prompt = load_prompt("chitchat", settings).text

    def chitchat(situation: Situation, text: str) -> str:
        kind = _kind(situation, text)
        try:
            response = llm.invoke([SystemMessage(prompt), HumanMessage(f"Situation: {kind}\nMessage: {text}")])
            reply = str(getattr(response, "content", response)).strip()
        except Exception:  # an outage or a bad call: the fixed wording, never an error (FR-F7)
            return _STATIC[kind]
        return reply if usable(kind, reply) else _STATIC[kind]

    return chitchat
