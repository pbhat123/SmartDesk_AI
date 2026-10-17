"""Cost tracking (FR-H7): token counts times the price table in config.yaml.

Every LLM call, embedding call and tool call inside a request becomes one `CostEntry` carrying the
request, the task (conversation goal such as a KB answer or a ticket flow, or "eval"), the agent, the
model tier and the step kind. Entries hold numbers and names only, never message text. Reasoning tokens
are part of the provider's output token count, so they are billed at the output price. In mock mode the
tokens are counted but nothing is priced, so mock runs never touch the real daily budget (FR-H8).
"""
import threading
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import tiktoken
from pydantic import BaseModel

from smartdesk_ai.config.settings import Settings, Tier
from smartdesk_ai.llm.fallback import LLMCall
from smartdesk_ai.ops.runtime import RequestContext, current_scope

Step = Literal["llm", "embedding", "tool"]
GroupBy = Literal["request", "task", "agent", "tier", "step"]
UNASSIGNED = "unassigned"
_EMBEDDING_ENCODING = "cl100k_base"  # the encoding OpenAI embedding models use


class CostEntry(BaseModel):
    ts: str
    request_id: str
    session_id: str
    task: str | None
    agent: str
    step: Step
    tier: Tier | None = None
    model: str | None = None
    tool: str | None = None
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0


class CostTracker:
    def __init__(
        self,
        settings: Settings,
        *,
        priced: bool,
        path: Path | str | None = None,
        on_spend: Callable[[float, str | None], None] | None = None,
    ) -> None:
        self._settings = settings
        self._priced = priced
        self._path = Path(path) if path is not None else None
        self._on_spend = on_spend  # (usd, task) -> None: feeds the daily budget counter (FR-H8)
        self._lock = threading.Lock()
        self.entries: list[CostEntry] = []
        self._unwritten: list[CostEntry] = []

    def llm_usd(self, tier: Tier, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
        if not self._priced:
            return 0.0
        price = getattr(self._settings.models, tier).price
        fresh = max(input_tokens - cached_tokens, 0)
        return (fresh * price.input + cached_tokens * price.cached_input + output_tokens * price.output) / 1_000_000

    def _add(self, ctx: RequestContext | None, **fields) -> CostEntry:
        entry = CostEntry(
            ts=datetime.now(timezone.utc).isoformat(),
            request_id=ctx.request_id if ctx else "-", session_id=ctx.session_id if ctx else "-",
            task=ctx.task if ctx else UNASSIGNED, **fields)
        with self._lock:
            self.entries.append(entry)
            self._unwritten.append(entry)
        if entry.usd and self._on_spend is not None:
            self._on_spend(entry.usd, entry.task)
        if ctx is None:
            self._write([entry])
        return entry

    def record_llm(self, ctx: RequestContext | None, agent: str, call: LLMCall) -> CostEntry:
        return self._add(
            ctx, agent=agent, step="llm", tier=call.tier, model=call.model_id, input_tokens=call.input_tokens,
            cached_tokens=call.cached_tokens, output_tokens=call.output_tokens,
            usd=self.llm_usd(call.tier, call.input_tokens, call.cached_tokens, call.output_tokens))

    def record_embedding(self, ctx: RequestContext | None, tokens: int) -> CostEntry:
        price = self._settings.models.embedding.price.input
        return self._add(ctx, agent="retrieval", step="embedding", model=self._settings.models.embedding.id,
                         input_tokens=tokens, usd=tokens * price / 1_000_000 if self._priced else 0.0)

    def record_tool(self, ctx: RequestContext | None, agent: str, tool: str) -> CostEntry:
        return self._add(ctx, agent=agent, step="tool", tool=tool)

    def finish_request(self, ctx: RequestContext, goal: str | None) -> float:
        with self._lock:
            mine = [e for e in self._unwritten if e.request_id == ctx.request_id]
            self._unwritten = [e for e in self._unwritten if e.request_id != ctx.request_id]
            for entry in mine:
                if entry.task is None:
                    entry.task = goal or UNASSIGNED
        self._write(mine)
        return sum(e.usd for e in mine)

    def _write(self, entries: list[CostEntry]) -> None:
        if self._path is None or not entries:
            return
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a") as handle:
                handle.writelines(entry.model_dump_json() + "\n" for entry in entries)

    @property
    def total(self) -> float:
        return sum(e.usd for e in self.entries)

    def totals(self, by: GroupBy) -> dict[str, float]:
        key = {"request": lambda e: e.request_id, "task": lambda e: e.task or UNASSIGNED,
               "agent": lambda e: e.agent, "tier": lambda e: e.tier or "-", "step": lambda e: e.step}[by]
        out: dict[str, float] = defaultdict(float)
        for entry in self.entries:
            out[key(entry)] += entry.usd
        return dict(out)


class CostedEmbeddings:
    """Wraps an embedder so each query embedding inside a request is counted (FR-H7). Document
    embedding is ingest work, not a request, and passes straight through."""

    def __init__(self, inner, tracker: CostTracker, settings: Settings) -> None:
        self._inner, self._tracker = inner, tracker
        self._encoding = tiktoken.get_encoding(_EMBEDDING_ENCODING)

    def __getattr__(self, name):  # name, model, ... of the wrapped embedder (embedder_id reads them)
        return getattr(self._inner, name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        vector = self._inner.embed_query(text)
        scope = current_scope()
        if scope is not None:
            ctx, services = scope
            services.on_embedding(ctx, len(self._encoding.encode(text)))
        return vector
