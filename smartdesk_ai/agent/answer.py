"""Answer step: answer from the retrieved context only, or say "I don't know" (FR-A4, FR-A15).

One WORKER_ANSWER call returns an `AnswerReply`. If it reports insufficient context (escalation
signal 3), or returns nothing usable, the reply is the fixed "I don't know" text and the flow
escalates even when retrieval scores passed (`should_escalate`).
"""
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.factory import ModelBuilder, get_llm, openai_builder
from smartdesk_ai.llm.structured import ReplyError, parse_model_reply
from smartdesk_ai.tools.search_kb import KBChunk, SearchKBResult

INSUFFICIENT_REPLY = "I don't know the answer to that from our knowledge base."


class AnswerError(Exception):
    """The model call failed or its reply was not a valid answer."""


class AnswerReply(BaseModel):
    """The exact shape the answer model must return (FR-H2)."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    insufficient: bool
    citations: list[str] = Field(default_factory=list)


class AnswerResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str  # what the employee sees
    answer: str  # the model's answer, empty when insufficient
    insufficient: bool  # escalation signal 3


def format_context(chunks: Sequence[KBChunk]) -> str:
    return "\n\n".join(f"[{c.entry_id}] {c.title}\n{c.text}" for c in chunks)


def _insufficient() -> AnswerResult:
    return AnswerResult(text=INSUFFICIENT_REPLY, answer="", insufficient=True)


def answer_from_context(llm: Any, prompt: str, question: str, chunks: Sequence[KBChunk]) -> AnswerResult:
    """Answer `question` from `chunks` only. Raises `AnswerError` on a failed or invalid call."""
    if not chunks:  # signal 1: nothing to answer from, so no model call
        return _insufficient()
    human = f"Question: {question}\n\nContext:\n{format_context(chunks)}"
    try:
        response = llm.invoke([SystemMessage(prompt), HumanMessage(human)])
    except Exception as error:
        raise AnswerError(f"answer call failed: {type(error).__name__}") from error
    try:
        reply = parse_model_reply(AnswerReply, getattr(response, "content", response))
    except ReplyError as error:
        raise AnswerError("answer reply is not valid") from error
    if reply.insufficient or not reply.answer.strip():
        return _insufficient()
    return AnswerResult(text=reply.answer.strip(), answer=reply.answer.strip(), insufficient=False)


def should_escalate(search: SearchKBResult, answer: AnswerResult | None) -> bool:
    """Any one signal escalates: no chunks, low score (both in `search.answerable`), or the LLM
    reporting insufficient context (FR-A15)."""
    return (not search.answerable) or (answer is not None and answer.insufficient)


def make_answerer(
    settings: Settings,
    *,
    builder: ModelBuilder = openai_builder,
    budget_level: int = 0,
    structured: bool | None = None,
) -> Callable[[str, Sequence[KBChunk]], AnswerResult]:
    """An answerer bound to the WORKER_ANSWER model and the active answer prompt. The live
    client uses structured-output mode; a custom builder (the mock path) gets plain JSON."""
    llm = get_llm("WORKER_ANSWER", settings, budget_level=budget_level, builder=builder).model
    if structured if structured is not None else builder is openai_builder:
        llm = llm.with_structured_output(AnswerReply)
    prompt = load_prompt("answer", settings).text

    def answer(question: str, chunks: Sequence[KBChunk]) -> AnswerResult:
        return answer_from_context(llm, prompt, question, chunks)

    return answer
