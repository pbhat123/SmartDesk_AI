"""Answer step: answer from the retrieved context only, or say "I don't know" (FR-A4, FR-A15).

One WORKER_ANSWER call returns an `AnswerReply`. If it reports insufficient context, or its reply
is malformed or fails the schema (escalation signal 3), the reply is the fixed "I don't know" text
and the flow escalates even when retrieval scores passed (`should_escalate`). If the call itself
fails because every model tier is down, `AnswerUnavailable` is raised instead: that is an outage
to tell the employee about, never a ticket offer (FR-F2, FR-F1).
"""
import time
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.fallback import GRACEFUL_MESSAGE, FallbackChat
from smartdesk_ai.llm.factory import ModelBuilder, openai_builder
from smartdesk_ai.llm.structured import ReplyError, parse_model_reply
from smartdesk_ai.tools.search_kb import KBChunk, SearchKBResult

INSUFFICIENT_REPLY = "I don't know the answer to that from our knowledge base."


class AnswerError(Exception):
    """The model replied, but not with a valid answer. This is escalation signal 3."""


class AnswerUnavailable(Exception):
    """The model call failed: every tier is down (FR-F2). Not a reason to offer a ticket;
    `message` is the polite try-again-later text for the employee (FR-F1)."""

    def __init__(self, message: str = GRACEFUL_MESSAGE) -> None:
        super().__init__(message)
        self.message = message


class AnswerReply(BaseModel):
    """The exact shape the answer model must return (FR-H2)."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    insufficient: bool
    citations: list[str] = Field(default_factory=list)


class AnswerResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str  # what the employee sees: the answer plus its sources
    answer: str  # the model's answer, empty when insufficient
    insufficient: bool  # escalation signal 3
    citations: list[str] = Field(default_factory=list)  # retrieved entry ids the answer used (FR-A13)


def format_context(chunks: Sequence[KBChunk]) -> str:
    return "\n\n".join(f"[{c.entry_id}] {c.title}\n{c.text}" for c in chunks)


def _insufficient() -> AnswerResult:
    return AnswerResult(text=INSUFFICIENT_REPLY, answer="", insufficient=True)


def _retrieved_citations(cited: Sequence[str], chunks: Sequence[KBChunk]) -> list[str]:
    """The cited ids that were actually retrieved, in the model's order, without repeats."""
    retrieved = {c.entry_id for c in chunks}
    return list(dict.fromkeys(i for i in cited if i in retrieved))


def _sources_line(cited: Sequence[str], chunks: Sequence[KBChunk]) -> str:
    titles = {c.entry_id: c.title for c in chunks}
    return "Sources: " + "; ".join(f"{i} ({titles[i]})" for i in cited)


def answer_from_context(llm: Any, prompt: str, question: str, chunks: Sequence[KBChunk]) -> AnswerResult:
    """Answer `question` from `chunks` only. Raises `AnswerError` on a failed or invalid call."""
    if not chunks:  # signal 1: nothing to answer from, so no model call
        return _insufficient()
    human = f"Question: {question}\n\nContext:\n{format_context(chunks)}"
    try:
        response = llm.invoke([SystemMessage(prompt), HumanMessage(human)])
    except Exception as error:  # the message is not kept: it could echo prompt text (FR-G11)
        raise AnswerUnavailable(getattr(error, "message", GRACEFUL_MESSAGE)) from error
    try:
        reply = parse_model_reply(AnswerReply, getattr(response, "content", response))
    except ReplyError as error:
        raise AnswerError("answer reply is not valid") from error
    if reply.insufficient or not reply.answer.strip():
        return _insufficient()
    cited = _retrieved_citations(reply.citations, chunks)
    if not cited:  # an answer that names no retrieved entry is never shown (FR-A13)
        return _insufficient()
    text = reply.answer.strip()
    return AnswerResult(text=f"{text}\n\n{_sources_line(cited, chunks)}", answer=text, insufficient=False, citations=cited)


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
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[str, Sequence[KBChunk]], AnswerResult]:
    """An answerer for the WORKER_ANSWER task and the active answer prompt. Every call goes
    through the Main/Light outage fallback. The live client uses structured-output mode; a custom
    builder (the mock path) gets plain JSON unless `structured=True`."""
    use_schema = structured if structured is not None else builder is openai_builder
    llm = FallbackChat(
        "WORKER_ANSWER", settings, builder=builder, budget_level=budget_level,
        schema=AnswerReply if use_schema else None, sleep=sleep,
    )
    prompt = load_prompt("answer", settings).text

    def answer(question: str, chunks: Sequence[KBChunk]) -> AnswerResult:
        return answer_from_context(llm, prompt, question, chunks)

    return answer
