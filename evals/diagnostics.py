"""What happened to each part of a live turn (FR-I3 to FR-I7, FR-A15): its domain, the retrieval signal, what the worker and the
grounding judge said on each attempt, whether it was retried, and what it cited. Read from the final state and the request's trace
events; no answer text is kept. The worker collapses "the model said insufficient", "the judge said ungrounded" and "no usable
citation" into one escalation signal, so the per-attempt verdicts here are what tells them apart."""
import json
import re
from collections.abc import Sequence

from pydantic import BaseModel

_GROUNDED = re.compile(r'"grounded"\s*:\s*(true|false)')


class PartDiag(BaseModel):
    part_id: str
    text: str  # the orchestrator's sub-question
    domain: str  # the domain the orchestrator gave it
    agent: str  # the node that gave the final result
    retrieval_ok: bool  # the retrieval signals (top cosine against the threshold) passed
    top_cosine: float | None
    threshold: float | None
    worker_insufficient: list[bool]  # the model's own "insufficient" flag, one per attempt
    judge_grounded: list[bool]  # the grounding judge's verdict, one per attempt that got that far
    retried: bool
    searched_domain: str | None
    citations: list[str]
    answered: bool


def _insufficient(completion: str) -> bool | None:
    try:
        return bool(json.loads(completion).get("insufficient"))
    except (ValueError, AttributeError, TypeError):
        return None


def _for_part(events: Sequence[dict], task: str, text: str) -> list[dict]:
    needle = f"Question: {text}\n"
    return [e for e in events if e.get("kind") == "llm_call" and e.get("task") == task and needle in str(e.get("prompt", ""))]


def part_diagnostics(state, events: Sequence[dict]) -> list[PartDiag]:
    out = []
    for part in state.parts:
        result = state.part_results.get(part.part_id)
        if result is None:
            continue
        text = result.sub_question or part.text
        worker = [v for e in _for_part(events, "WORKER_ANSWER", text) if (v := _insufficient(str(e.get("completion", "")))) is not None]
        judge = [m.group(1) == "true" for e in _for_part(events, "GROUNDING_JUDGE", text)
                 if (m := _GROUNDED.search(str(e.get("completion", ""))))]
        out.append(PartDiag(
            part_id=part.part_id, text=text, domain=str(part.domain), agent=result.agent, retrieval_ok=result.answerable,
            top_cosine=result.top_cosine, threshold=result.threshold, worker_insufficient=worker, judge_grounded=judge,
            retried=result.retried, searched_domain=str(result.searched_domain) if result.searched_domain else None,
            citations=list(result.citations),
            answered=bool(result.citations) and result.answerable and not result.llm_insufficient))
    return out
