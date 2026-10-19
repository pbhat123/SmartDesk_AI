"""The synthesizer (FR-I1, FR-I4, FR-I7): merges the answers of several parts into one reply.

It gets the employee's original message and each answered sub-question with its answer, and nothing else (FR-I3).
Its call goes through `FallbackChat` like every other model call (retry, budget gate, cost, trace).

The merged text is only used when it holds no fact the workers did not give: every number and every entry id in it
must be in the worker answers, and the fact check used for ticket drafts (`is_grounded`: numbers, codes, URLs, emails,
IDs and capitalised names) must pass against them. Anything else, a failed call or a malformed reply, and the reply
is the worker answers joined in the order of the questions. The Sources line is built from the workers' citations, never
by the model.
"""
import logging
import re
import time
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from smartdesk_ai.agent.agents import SYNTHESIZER
from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.agent.state import PartResult
from smartdesk_ai.agent.ticket_agent import is_grounded
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.factory import ModelBuilder, openai_builder
from smartdesk_ai.llm.fallback import FallbackChat
from smartdesk_ai.llm.structured import ReplyError, parse_model_reply

logger = logging.getLogger(__name__)

_NUMBER = re.compile(r"\d[\d,.:%]*\d|\d")
_ENTRY_ID = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b")


class SynthReply(BaseModel):
    """The exact shape the synthesizer must return (FR-H2)."""

    model_config = ConfigDict(extra="forbid")

    answer: str


class SynthOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str  # the reply with its Sources line
    merged: bool  # the model's merge was used; False: the worker answers were joined
    reason: str | None = None  # why the merge was not used


class Synthesizer:
    def __init__(self, llm: Any, prompt: str) -> None:
        self.llm, self.prompt = llm, prompt

    def merge(self, message: str, parts: Sequence[PartResult]) -> SynthOutcome:
        """One reply for answered `parts` (in question order). Never raises: the worker answers are already grounded
        and checked, so a failed or unsafe merge degrades to joining them."""
        bodies = [p.body for p in parts]
        sources = sources_line(parts)
        joined = "\n\n".join(bodies)
        try:
            text = self._ask(message, parts)
        except Exception as error:  # an outage or a malformed reply: the type only, the message could echo text
            logger.warning("synthesizer fell back to the joined answers (%s)", type(error).__name__)
            return SynthOutcome(text=f"{joined}\n\n{sources}", merged=False, reason=type(error).__name__)
        problem = unsupported(text, bodies)
        if problem:
            logger.warning("synthesizer fell back to the joined answers (%s)", problem)
            return SynthOutcome(text=f"{joined}\n\n{sources}", merged=False, reason=problem)
        return SynthOutcome(text=f"{text}\n\n{sources}", merged=True)

    def _ask(self, message: str, parts: Sequence[PartResult]) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        blocks = [f"Question {i}: {p.sub_question}\nAnswer {i}: {p.body}" for i, p in enumerate(parts, 1)]
        human = f"Original message: {message}\n\n" + "\n\n".join(blocks)
        response = self.llm.invoke([SystemMessage(self.prompt), HumanMessage(human)])
        return parse_model_reply(SynthReply, getattr(response, "content", response)).answer.strip()


def join_answers(parts: Sequence[PartResult]) -> SynthOutcome:
    """The reply when there is no synthesizer: the worker answers in the order of the questions, then the sources."""
    return SynthOutcome(text="\n\n".join(p.body for p in parts) + f"\n\n{sources_line(parts)}", merged=False,
                        reason="no_synthesizer")


def unsupported(text: str, bodies: Sequence[str]) -> str | None:
    """Why `text` may not be used, or None when it adds nothing to what the workers said (FR-H2)."""
    if not text:
        return "empty"
    said = " ".join(bodies)
    if any(number not in said for number in _NUMBER.findall(text)):
        return "number"
    if any(entry not in said for entry in _ENTRY_ID.findall(text)):
        return "entry_id"
    if not is_grounded(text, bodies):
        return "fact"
    return None


def sources_line(parts: Sequence[PartResult]) -> str:
    """"Sources: ID (Title); ..." from the workers' citations, in question order, without repeats."""
    titles = {c.entry_id: c.title for p in parts for c in p.chunks}
    cited = list(dict.fromkeys(i for p in parts for i in p.citations))
    return "Sources: " + "; ".join(f"{i} ({titles.get(i, i)})" for i in cited)


def make_synthesizer(
    settings: Settings,
    *,
    builder: ModelBuilder = openai_builder,
    budget_level: int = 0,
    structured: bool | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Synthesizer:
    """A synthesizer for the SYNTHESIZER task and the active synthesizer prompt. The live client uses
    structured-output mode; a custom builder (the mock path) gets plain JSON unless `structured=True`."""
    use_schema = structured if structured is not None else builder is openai_builder
    llm = FallbackChat("SYNTHESIZER", settings, builder=builder, budget_level=budget_level,
                       schema=SynthReply if use_schema else None, sleep=sleep, agent=SYNTHESIZER)
    return Synthesizer(llm, load_prompt("synthesizer", settings).text)
