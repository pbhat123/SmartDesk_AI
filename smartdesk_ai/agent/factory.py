"""Builds the agent graph and its responder for the CLI and the UI, from `modes.llm` and `modes.tickets`.

llm mock: hashing embeddings, scripted mock models and a separate index under data/index_mock; no keys,
no network. llm live: the real models, embeddings and index (needs OPENAI_API_KEY and a built index).
tickets mock: an in-memory client, no GitHub calls. Anything that makes live mode unusable raises
`LiveModeUnavailable`, and the caller falls back to mock with a notice (FR-F7).
"""
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from smartdesk_ai.agent.chat_service import Responder
from smartdesk_ai.agent.chitchat import make_chitchat
from smartdesk_ai.agent.confirm import make_confirm_parser
from smartdesk_ai.agent.graph import KBDeps, build_kb_graph, make_kb_responder
from smartdesk_ai.agent.grounding import make_grounding_judge
from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.agent.router import make_router
from smartdesk_ai.agent.state import STATE_MODELS
from smartdesk_ai.agent.synthesizer import make_synthesizer
from smartdesk_ai.agent.ticket_agent import TicketDeps, make_drafter
from smartdesk_ai.agent.workers import MISC_SEARCH, make_worker_answerers
from smartdesk_ai.config.env import DEFAULT_DOTENV, load_env
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.guardrails.injection import make_injection_checker, screen_team_update
from smartdesk_ai.llm.factory import ModelBuilder, openai_builder
from smartdesk_ai.llm.mock import (
    MockAnswerModel,
    MockChitchatModel,
    MockCompactionModel,
    MockConfirmModel,
    MockDraftModel,
    MockGroundingModel,
    MockInjectionModel,
    MockRoutingModel,
    MockSynthesizerModel,
)
from smartdesk_ai.memory.checkpointer import make_checkpointer
from smartdesk_ai.guardrails.pii import PiiRedactor
from smartdesk_ai.ops.cache import AnswerCache, clear_answers, open_backend
from smartdesk_ai.ops.cache_log import CACHE_LOG_FILE, CacheLog
from smartdesk_ai.ops.budget import BudgetController, SpendStore
from smartdesk_ai.ops.cost import CostedEmbeddings, CostTracker
from smartdesk_ai.ops.compaction import Compactor, make_summarizer
from smartdesk_ai.ops.health import Component, HealthRegistry, get_registry
from smartdesk_ai.ops.hitl_counters import HitlCounters
from smartdesk_ai.ops.metrics import MetricsRecorder
from smartdesk_ai.ops.rate_limit import RateLimiter
from smartdesk_ai.ops.spend_ledger import start_run
from smartdesk_ai.ops.runtime import Services, configure_mask_salt, current_budget_level
from smartdesk_ai.ops.tracing import TracingFactories, build_tracing
from smartdesk_ai.ops.trace_log import TraceLog, make_masker
from smartdesk_ai.rag.embeddings import Embeddings, HashingEmbeddings, get_embeddings
from smartdesk_ai.rag.index import IndexNotBuilt, open_index
from smartdesk_ai.rag.ingest import DEFAULT_KB_DIR, ROOT, ingest
from smartdesk_ai.rag.rerank import load_reranker
from smartdesk_ai.rag.staleness import read_state
from smartdesk_ai.tickets import TicketClient, TicketConfigError, create_ticket_client
from smartdesk_ai.tickets.seed import seed
from smartdesk_ai.tickets.sqlite_map import TicketStore
from smartdesk_ai.tools.executor import ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry
from smartdesk_ai.tools.search_kb import make_search_kb
from smartdesk_ai.tools.ticket_tools import register_ticket_tools

DEFAULT_DATA_DIR = ROOT / "data"  # gitignored
DATA_DIR_ENV = "SMARTDESK_DATA_DIR"  # lets tests and deployments move every runtime file
CHECKPOINT_DB = "checkpoints.db"
TRACE_LOG = "trace.jsonl"
COST_LOG = "cost.jsonl"
OPS_DB = "ops.db"  # daily spend and rate-limit counters
SKIPPED_IN_MOCK = "skipped in mock mode"
HITL_LOG = "hitl.jsonl"  # approvals, rejections and edits at the ticket gate (FR-H4); the Monitoring tab reads it
TOOL_METRICS_LOG = "tool_metrics.jsonl"  # per-call records; the Monitoring tab reads them
TICKET_MAP_DB = "app.db"  # the live email-to-ticket map; scripts/seed_tickets.py writes the same file


@dataclass(frozen=True)
class AgentStack:
    graph: Any
    client: TicketClient
    store: TicketStore
    executor: ToolExecutor
    services: Services
    registry: HealthRegistry


def kb_version(settings: Settings, index_dir: Path) -> Callable[[], str]:
    """What a cached answer depends on besides the question: the ingested KB (file hashes and embedder, from the index
    state) and the settings that shape an answer. A re-ingest, even in another process, makes older entries unreachable."""
    shaping = json.dumps([settings.prompts.get("answer"), settings.prompts.get("grounding_judge"),
                          settings.models.main.id, settings.retrieval.strategy])

    def version() -> str:
        return hashlib.sha256((json.dumps(read_state(index_dir), sort_keys=True) + shaping).encode()).hexdigest()[:16]

    return version


class LiveModeUnavailable(RuntimeError):
    """modes.llm or modes.tickets is `live` but something it needs is missing. The message names
    what is missing and how to fix it; there is no silent fallback to mock."""


def resolve_data_dir(data_dir: Path | None, environ: Mapping[str, str] | None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    override = (environ if environ is not None else os.environ).get(DATA_DIR_ENV)
    return Path(override) if override else DEFAULT_DATA_DIR


SDK_LOGGERS = ("openai", "httpx", "httpcore")  # at DEBUG they log full request bodies, raw emails included


def quiet_sdk_loggers() -> None:
    """Pin the HTTP and OpenAI SDK loggers to WARNING, so a DEBUG root logger cannot write prompts to a log (FR-G11)."""
    for name in SDK_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def _builder_for(model: Any) -> ModelBuilder:
    return lambda model_id, cfg: model


def build_stack(
    settings: Settings,
    *,
    environ: Mapping[str, str] | None = None,
    dotenv_path: Path | None = DEFAULT_DOTENV,
    data_dir: Path | None = None,
    kb_dir: Path = DEFAULT_KB_DIR,
    registry: HealthRegistry | None = None,
    client: TicketClient | None = None,
    tracing_factories: TracingFactories | None = None,
    mock_answer_model: Any | None = None,
    tracing: bool = True,
    use_cache: bool = True,
) -> "AgentStack":
    """Everything the app is made of: the compiled graph with its ticket client, ticket map, executor and ops
    services. `build_responder` wraps it for the CLI and the UI; the evals use the parts directly. Evals and the drift
    check pass `use_cache=False`, so a repeated question is measured, not remembered."""
    registry = registry if registry is not None else get_registry()
    env = load_env(environ=environ, dotenv_path=dotenv_path)
    data_dir = resolve_data_dir(data_dir, environ)
    quiet_sdk_loggers()
    configure_mask_salt(env.mask_salt.get_secret_value() if env.mask_salt else None)
    live = settings.modes.llm == "live"
    cache_backend = None
    if use_cache and settings.cache.enabled:
        cache_backend = open_backend(settings, env, registry, mode=settings.modes.llm)
    cache_log = CacheLog(Path(data_dir) / CACHE_LOG_FILE)
    reranker = None
    embeddings: Embeddings
    search_settings = settings
    if live:
        if env.openai_api_key is None:
            raise LiveModeUnavailable(
                "OPENAI_API_KEY is not set. Add it to .env (see .env.example) or export it."
            )
        start_run(settings, source="app")  # no-op if an eval already started this process's run
        embeddings = get_embeddings(settings, env)
        index_dir = Path(data_dir) / "index"  # what `python -m smartdesk_ai.rag.ingest` builds
        answer_b = route_b = draft_b = confirm_b = chat_b = inject_b = judge_b = compact_b = synth_b = openai_builder
    else:
        embeddings = HashingEmbeddings()
        index_dir = Path(data_dir) / "index_mock"
        # idempotent, offline; a changed KB clears the answer cache (FR-H6)
        ingest(kb_dir, index_dir, embeddings, settings.chunking,
               on_kb_changed=None if cache_backend is None else lambda report: clear_answers(cache_backend, cache_log))
        # Hashing vectors score lower than the real embeddings, so retrieval uses its own mock threshold.
        search_settings = settings.model_copy(update={"thresholds": settings.thresholds.model_copy(
            update={"escalation": settings.thresholds.escalation_mock})})
        answer_b = _builder_for(mock_answer_model or MockAnswerModel())  # evals swap in a scripted answer model
        route_b = _builder_for(MockRoutingModel())
        draft_b, confirm_b = _builder_for(MockDraftModel()), _builder_for(MockConfirmModel())
        chat_b = _builder_for(MockChitchatModel())
        inject_b = _builder_for(MockInjectionModel())
        judge_b = _builder_for(MockGroundingModel())
        compact_b = _builder_for(MockCompactionModel())
        synth_b = _builder_for(MockSynthesizerModel())
    # The local trace log works with no tracing keys; its masker needs no Presidio (FR-G8, FR-G11).
    trace = TraceLog(Path(data_dir) / TRACE_LOG, make_masker(PiiRedactor(None, settings.guardrails.pii_min_score)),
                     prompt_versions=settings.prompts)
    if tracing:
        build_tracing(env, trace, registry, tracing_factories)  # optional Langfuse and LangSmith; the local log stays
    spend = SpendStore(Path(data_dir) / OPS_DB)
    budget = BudgetController(settings, spend)
    # Mock runs count tokens but price nothing, so they never touch the real daily budget.
    cost = CostTracker(settings, priced=live, path=Path(data_dir) / COST_LOG, on_spend=budget.add)
    limiter = RateLimiter(settings.rate_limits, Path(data_dir) / OPS_DB)
    services = Services(cost=cost, budget=budget, limiter=limiter, trace=trace)
    costed_embeddings = CostedEmbeddings(embeddings, cost, settings)  # query embeddings are counted (FR-H7)
    try:
        index = open_index(index_dir, costed_embeddings)
    except IndexNotBuilt:
        raise LiveModeUnavailable(
            f"the KB index is missing in {index_dir}. Build it with: python -m smartdesk_ai.rag.ingest"
        ) from None
    if not live:
        registry.mark_skipped(Component.RERANKER, SKIPPED_IN_MOCK)  # on purpose: mock retrieval is a plumbing check
    elif settings.retrieval.strategy.endswith("_rerank"):
        reranker = load_reranker(settings, registry)  # degrades to RRF order, never fails (FR-F7)
    else:
        registry.mark_skipped(Component.RERANKER, f"not used by retrieval.strategy={settings.retrieval.strategy}")
    if client is None:  # a caller may hand in its own client, e.g. the safety eval to inspect tickets
        try:
            client = create_ticket_client(settings, env)
        except TicketConfigError as error:
            raise LiveModeUnavailable(f"{error}. Set it in .env (see .env.example).") from error

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    # Mock tickets live in memory, so their numbers never reach the live map in app.db.
    store = TicketStore(Path(data_dir) / TICKET_MAP_DB if settings.modes.tickets == "live" else ":memory:")
    if settings.modes.tickets == "mock":
        # Demo data for the status flow with no keys: the seed script's tickets, in memory only.
        seed(client, store)
    tools = ToolRegistry()
    tools.register("search_kb", make_search_kb(index, search_settings, reranker=reranker, health=registry))
    # Team comments come from a public repo: scan them before anything downstream sees them (FR-C9).
    update_checker = make_injection_checker(settings, builder=inject_b)
    register_ticket_tools(tools, client, store, screen_update=lambda update: screen_team_update(update, update_checker))
    executor = ToolExecutor(
        tools, settings, request_store=store, trace=trace, metrics=MetricsRecorder(Path(data_dir) / TOOL_METRICS_LOG))
    ticket_deps = TicketDeps(
        drafter=make_drafter(settings, builder=draft_b),
        executor=executor,
        parse_confirm=make_confirm_parser(settings, builder=confirm_b),
        duplicate_threshold=settings.thresholds.duplicate_title,
        hitl=HitlCounters(Path(data_dir) / HITL_LOG),
    )
    prompt_tokens = len(load_prompt("compaction", settings).text) // 3  # a generous size estimate
    workers = make_worker_answerers(settings, builder=answer_b, judge=make_grounding_judge(settings, builder=judge_b))
    cache = None
    if cache_backend is not None:
        cache = AnswerCache(cache_backend, kb_version=kb_version(settings, index_dir),
                            embed=costed_embeddings.embed_query, cutoff=settings.thresholds.semantic_cache_cutoff,
                            log=cache_log)
    graph = build_kb_graph(
        KBDeps(executor=executor, answer=workers[MISC_SEARCH], cache=cache, workers=workers,
               synthesizer=make_synthesizer(settings, builder=synth_b)),
        make_checkpointer(Path(data_dir) / CHECKPOINT_DB, STATE_MODELS),
        route=make_router(settings, builder=route_b),
        tickets=ticket_deps,
        status_max_listed=settings.status.max_listed,
        max_parts=settings.orchestrator.max_parts,
        chitchat=make_chitchat(settings, builder=chat_b),
        budget_level=current_budget_level,
        compactor=Compactor(settings, make_summarizer(settings, builder=compact_b), overhead_tokens=prompt_tokens),
    )
    return AgentStack(graph=graph, client=client, store=store, executor=executor, services=services,
                      registry=registry)


def build_responder(settings: Settings, **kwargs: Any) -> Responder:
    """The ChatService responder over `build_stack` (same arguments)."""
    stack = build_stack(settings, **kwargs)
    responder = make_kb_responder(stack.graph)
    responder.services = stack.services  # build_service hands them to ChatService, which opens the request scopes
    return responder
