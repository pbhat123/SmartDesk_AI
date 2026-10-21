"""Per-turn wrapper around the agent (PLAN section 5).

Skeleton for T0.6: the guards default to no-ops and the responder to a stub, so the app
starts and answers with no keys, no Redis and no tracing (FR-F6, FR-F7). `build_service` plugs in the
real input guard (T5.5) and, through `responder_factory`, the graph (T4.x) using the same hooks.
Raw user text is never logged here.
"""
import logging
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from smartdesk_ai.config.env import DEFAULT_DOTENV, load_env
from smartdesk_ai.config.settings import DEFAULT_CONFIG, Settings, load_settings
from smartdesk_ai.guardrails.decision import GuardDecision
from smartdesk_ai.guardrails.input_guard import build_input_guard
from smartdesk_ai.guardrails.moderation import probe_moderation
from smartdesk_ai.guardrails.output_guard import build_output_guard
from smartdesk_ai.llm.fallback import SERVICE_UNAVAILABLE_MESSAGE
from smartdesk_ai.ops.health import HealthRegistry, get_registry
from smartdesk_ai.guardrails.pii import EMAIL
from smartdesk_ai.ops.runtime import RequestContext, Services, current_scope, goal_from_steps, request_scope
from smartdesk_ai.rag.ingest import DEFAULT_KB_DIR

logger = logging.getLogger(__name__)

STUB_REPLY = "SmartDesk AI is starting up. The help desk agent is not connected yet."
EMPTY_REPLY = "It looks like your message was empty. Please type your question or request."


def too_long_reply(max_chars: int) -> str:
    return f"That message is a bit too long for me (the limit is {max_chars} characters). Could you shorten it?"


class ChatReply(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    ticket: int | None = None
    debug: dict[str, Any] = Field(default_factory=dict)


class Turn(BaseModel):
    """A responder's result when it has debug details to show (FR-A12)."""

    model_config = ConfigDict(frozen=True)

    text: str
    debug: dict[str, Any] = Field(default_factory=dict)


InputGuard = Callable[[str], GuardDecision]
OutputGuard = Callable[[str], str]
Responder = Callable[[str, str], str | Turn]  # (session_id, guarded text) -> reply text or Turn


def no_input_guard(text: str) -> GuardDecision:
    return GuardDecision(text=text)


def no_output_guard(text: str) -> str:
    return text


def stub_responder(session_id: str, text: str) -> str:
    return STUB_REPLY


class ChatService:
    def __init__(
        self,
        settings: Settings,
        input_guard: InputGuard = no_input_guard,
        responder: Responder = stub_responder,
        output_guard: OutputGuard = no_output_guard,
        services: Services | None = None,
    ) -> None:
        self.settings = settings
        self._input_guard = input_guard
        self._responder = responder
        self._output_guard = output_guard
        self._services = services or Services()

    @property
    def services(self) -> Services:
        return self._services

    def _edge_case(self, text: str) -> ChatReply | None:
        if not text.strip():
            return ChatReply(text=EMPTY_REPLY, debug={"edge": "empty"})
        limit = self.settings.input.max_chars
        if len(text) > limit:
            return ChatReply(text=too_long_reply(limit), debug={"edge": "too_long"})
        return None

    def handle(self, session_id: str, text: str) -> ChatReply:
        if not session_id:
            raise ValueError("session_id must not be empty")
        edge = self._edge_case(text)
        if edge is not None:  # cheap checks first: nothing reaches a guard, a model or a tool (FR-F4)
            return edge
        ctx = RequestContext(session_id=session_id, request_id=uuid.uuid4().hex)
        with request_scope(ctx, self._services):
            limit = self._services.limiter.hit("questions", session_id) if self._services.limiter else None
            if limit is not None and not limit.allowed:  # every message counts, even small talk (FR-H13)
                reply = ChatReply(text=limit.message or "", debug={"edge": "rate_limited"})
            elif self._services.budget_level(ctx) >= 2:  # cutoff 2: no guard, model or tool runs (FR-H8)
                reply = ChatReply(text=SERVICE_UNAVAILABLE_MESSAGE, debug={"edge": "budget_cutoff_2"})
            else:
                reply = self._handle(session_id, text)
        cost = self._services.finish_request(ctx, goal_from_steps(reply.debug.get("steps", [])))
        if self._services.cost is not None:
            reply.debug["cost_usd"] = cost
        return reply

    def _log(self, kind: str, ctx: RequestContext | None = None, **fields: Any) -> None:
        trace = self._services.trace
        if trace is not None:
            scope = current_scope()
            trace.event(kind, **fields, **trace.request_fields(scope[0] if scope else None))

    def _handle(self, session_id: str, text: str) -> ChatReply:
        decision = self._input_guard(text)
        found = EMAIL.search(decision.text)  # an email typed in this message identifies the user from here on
        if found:
            scope = current_scope()
            if scope is not None:
                scope[0].set_email(found.group())
        self._log("request", text=decision.text, blocked=decision.blocked, reason=decision.reason)
        debug: dict[str, Any] = {}
        if decision.blocked:
            reply_text = decision.refusal or ""
            debug["blocked_reason"] = decision.reason
        else:
            produced = self._responder(session_id, decision.text)
            reply_text = produced.text if isinstance(produced, Turn) else produced
            debug["responder"] = getattr(self._responder, "__name__", "custom")
            if isinstance(produced, Turn):
                debug.update(produced.debug)
        reply_text = self._output_guard(reply_text)
        for index, step in enumerate(debug.get("steps", [])):
            self._log("agent_step", step=step, index=index)
        self._log("reply", text=reply_text, blocked=decision.blocked)
        logger.debug("turn done session=%s blocked=%s", session_id, decision.blocked)
        return ChatReply(text=reply_text, debug=debug)


def build_service(
    environ: Mapping[str, str] | None = None,
    dotenv_path: Path | None = DEFAULT_DOTENV,
    registry: HealthRegistry | None = None,
    config_path: Path = DEFAULT_CONFIG,
    responder: Responder = stub_responder,
    responder_factory: Callable[[Settings], Responder] | None = None,
    input_guard: InputGuard | None = None,
    output_guard: OutputGuard | None = None,
    kb_dir: Path = DEFAULT_KB_DIR,
) -> ChatService:
    """Assemble the service. Missing keys or optional components never raise (FR-F7).

    Only an invalid config.yaml is fatal, since nothing can run without it.
    """
    env = load_env(environ=environ, dotenv_path=dotenv_path)
    registry = registry if registry is not None else get_registry()
    registry.register_env(env)
    settings = load_settings(config_path, environ=environ)
    if responder_factory is not None:
        responder = responder_factory(settings)
    if input_guard is None:  # every message passes the guard before the graph (FR-H1)
        input_guard = build_input_guard(
            settings, environ=environ, dotenv_path=dotenv_path, registry=registry)
        probe_moderation(input_guard.moderator, registry)  # one free call in live mode: ok or degraded from the first screen
    if output_guard is None:  # every reply passes the guard before it is shown (FR-H2)
        output_guard = build_output_guard(
            settings, kb_dir, environ=environ, dotenv_path=dotenv_path, registry=registry)
    return ChatService(settings, input_guard=input_guard, responder=responder, output_guard=output_guard,
                       services=getattr(responder, "services", None))
