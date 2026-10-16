"""The `search_kb` tool (FR-G7): retrieval chunks, their scores, and `answerable`.

`answerable` covers escalation signals 1 and 2 only (no chunks, top dense cosine below the
threshold, FR-A15). Signal 3, the LLM reporting insufficient context, comes from the worker.
"""
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from smartdesk_ai.config.settings import Settings
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.rag.answerable import Signals
from smartdesk_ai.rag.index import KBIndex
from smartdesk_ai.rag.pipeline import OrderedBy, retrieve
from smartdesk_ai.rag.rerank import Reranker
from smartdesk_ai.tools.definitions import SearchKBArgs


class KBChunk(BaseModel):
    entry_id: str
    domain: Literal["IT", "HR"]
    title: str
    header: str
    text: str
    score: float


class SearchKBResult(BaseModel):
    chunks: list[KBChunk]  # whole KB entries, best first
    ordered_by: OrderedBy  # which stage produced `score`
    top_cosine: float
    answerable: bool
    signals: Signals
    threshold: float


def make_search_kb(
    index: KBIndex,
    settings: Settings,
    *,
    reranker: Reranker | None,
    health: HealthRegistry | None = None,
) -> Callable[[SearchKBArgs], SearchKBResult]:
    def search_kb(args: SearchKBArgs) -> SearchKBResult:
        found = retrieve(
            index, args.query, domain=args.domain, cfg=settings, reranker=reranker, health=health
        )
        return SearchKBResult(
            chunks=[
                KBChunk(entry_id=e.entry_id, domain=e.domain, title=e.title, header=e.header,
                        text=e.text, score=e.score)
                for e in found.entries
            ],
            ordered_by=found.ordered_by,
            top_cosine=found.top_cosine,
            answerable=found.answerable,
            signals=found.signals,
            threshold=found.threshold,
        )

    return search_kb
