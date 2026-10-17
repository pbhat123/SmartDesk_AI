"""Runtime grounding judge (FR-H2): the GROUNDING_JUDGE task (Main-Model) checks a draft answer's claims
against the retrieved chunks. `answer_from_context` calls it once per answered part; an unsupported draft
becomes the insufficient-context signal, which leads to the whole-KB retry and the ticket offer.
"""
import time
from collections.abc import Callable, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from smartdesk_ai.agent.answer import Judge, format_context
from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.factory import ModelBuilder, openai_builder
from smartdesk_ai.llm.fallback import FallbackChat
from smartdesk_ai.llm.structured import parse_model_reply
from smartdesk_ai.tools.search_kb import KBChunk


class GroundingReply(BaseModel):
    """The exact shape the GROUNDING_JUDGE model must return (FR-H2)."""

    model_config = ConfigDict(extra="forbid")

    grounded: bool
    unsupported: list[str] = Field(default_factory=list)


def make_grounding_judge(
    settings: Settings,
    *,
    builder: ModelBuilder = openai_builder,
    budget_level: int = 0,
    sleep: Callable[[float], None] = time.sleep,
) -> Judge:
    """A judge for the active grounding_judge prompt. It raises when the model is unavailable or answers
    malformed JSON; the answer step treats that as ungrounded."""
    llm = FallbackChat(
        "GROUNDING_JUDGE", settings, builder=builder, budget_level=budget_level,
        schema=GroundingReply if builder is openai_builder else None, sleep=sleep,
    )
    prompt = load_prompt("grounding_judge", settings).text

    def judge(question: str, answer: str, chunks: Sequence[KBChunk]) -> bool:
        human = f"Question: {question}\n\nDraft answer:\n{answer}\n\nContext:\n{format_context(chunks)}"
        reply = llm.invoke([SystemMessage(prompt), HumanMessage(human)])
        return parse_model_reply(GroundingReply, reply.content).grounded

    return judge
