"""The synthesizer (FR-I1, FR-I4, FR-I7): merges the answers of several parts into one reply.

It gets the employee's original message and each answered sub-question with its answer, and nothing else (FR-I3).
Its call goes through `FallbackChat` like every other model call (retry, budget gate, cost, trace).
"""
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from smartdesk_ai.agent.agents import SYNTHESIZER
from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.factory import ModelBuilder, openai_builder
from smartdesk_ai.llm.fallback import FallbackChat


class SynthReply(BaseModel):
    """The exact shape the synthesizer must return (FR-H2)."""

    model_config = ConfigDict(extra="forbid")

    answer: str


class Synthesizer:
    def __init__(self, llm: Any, prompt: str) -> None:
        self.llm, self.prompt = llm, prompt


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
