"""Per-request context and the shared ops services (cost, budget, rate limits, trace).

`ChatService` opens a `request_scope` around every message. Code deep inside a request (the LLM wrapper,
the tool executor, the embedding wrapper) finds the scope with `current_scope()` instead of every maker
having to pass these objects down. Outside a scope nothing is recorded and nothing is limited, so unit
tests and scripts need no set-up. Nothing here holds message text; the context only carries IDs.
"""
import contextvars
import hashlib
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from smartdesk_ai.llm.fallback import LLMCall
    from smartdesk_ai.ops.budget import BudgetController
    from smartdesk_ai.ops.cost import CostTracker
    from smartdesk_ai.ops.rate_limit import RateLimiter
    from smartdesk_ai.ops.trace_log import TraceLog

logger = logging.getLogger(__name__)

FALLBACK_SALT = "smartdesk-user-v1"  # used only when MASK_SALT is not set
_salt: str | None = None
_warned = False


def configure_mask_salt(salt: str | None) -> None:
    """Set the salt from the environment (`MASK_SALT`) at start-up; None or blank selects the fallback."""
    global _salt, _warned
    _salt, _warned = (salt or None), False


def _active_salt() -> str:
    global _warned
    if _salt:
        return _salt
    if not _warned:
        _warned = True
        logger.warning("MASK_SALT is not set; masked user IDs and rate-limit keys use the built-in salt")
    return FALLBACK_SALT


def masked_user_id(email: str) -> str:
    """A stable ID for an employee that is not their email (FR-G11, FR-H9). Also the rate-limit key."""
    digest = hashlib.sha256(f"{_active_salt()}:{email.strip().lower()}".encode()).hexdigest()
    return f"user-{digest[:12]}"


def salted_digest(text: str) -> str:
    """A salted SHA-256 of `text`, for keys that must not reveal an email or a title (the create-request log)."""
    return hashlib.sha256(f"{_active_salt()}:{text}".encode()).hexdigest()


@dataclass
class RequestContext:
    session_id: str
    request_id: str
    task: str | None = None  # the conversation goal, or "eval"; filled in when the request ends if still None
    user_id: str | None = None  # masked; set once the email is known

    def set_email(self, email: str | None) -> None:
        if email:
            self.user_id = masked_user_id(email)


class Services:
    """The optional ops services. Any of them may be None (they degrade away, FR-F7)."""

    def __init__(
        self,
        *,
        cost: "CostTracker | None" = None,
        budget: "BudgetController | None" = None,
        limiter: "RateLimiter | None" = None,
        trace: "TraceLog | None" = None,
    ) -> None:
        self.cost, self.budget, self.limiter, self.trace = cost, budget, limiter, trace

    def budget_level(self, ctx: "RequestContext | None" = None) -> int:
        return self.budget.level(ctx.task if ctx else None) if self.budget is not None else 0

    def on_llm_call(self, ctx: RequestContext | None, agent: str, call: "LLMCall") -> None:
        if self.cost is not None and call.ok:
            self.cost.record_llm(ctx, agent, call)
        if self.trace is not None:
            self.trace.llm_record(ctx, agent, call)

    def on_embedding(self, ctx: RequestContext | None, tokens: int) -> None:
        if self.cost is not None:
            self.cost.record_embedding(ctx, tokens)

    def on_tool(self, ctx: RequestContext | None, agent: str, tool: str) -> None:
        if self.cost is not None:
            self.cost.record_tool(ctx, agent, tool)

    def finish_request(self, ctx: RequestContext, goal: str | None) -> float:
        """Close a request: costs recorded under it get their task and are written out. Returns its cost."""
        return self.cost.finish_request(ctx, goal) if self.cost is not None else 0.0


_scope: contextvars.ContextVar[tuple[RequestContext, Services] | None] = contextvars.ContextVar(
    "smartdesk_request_scope", default=None)


@contextmanager
def request_scope(ctx: RequestContext, services: Services) -> Iterator[RequestContext]:
    token = _scope.set((ctx, services))
    try:
        yield ctx
    finally:
        _scope.reset(token)


def current_scope() -> tuple[RequestContext, Services] | None:
    return _scope.get()


def current_budget_level() -> int:
    """The budget level for the request in progress; 0 outside a request scope."""
    scope = _scope.get()
    return scope[1].budget_level(scope[0]) if scope is not None else 0


@contextmanager
def task_scope(task: str) -> Iterator[None]:
    """Run a block under an explicit cost task name, e.g. "eval" for Ragas runs and ablations."""
    current = _scope.get()
    if current is None:
        yield
        return
    ctx, _ = current
    previous, ctx.task = ctx.task, task
    try:
        yield
    finally:
        ctx.task = previous


def goal_from_steps(steps: Sequence[str]) -> str:
    """The conversation goal of a request, from the graph's step trace."""
    if any(step.startswith("t_") for step in steps):
        return "ticket_flow"
    if any(step.startswith("s_") for step in steps):
        return "ticket_status"
    if "chitchat" in steps:
        return "smalltalk"
    if "search_kb" in steps or any(step.startswith("cache_hit") for step in steps):
        return "kb_answer"
    return "other"
