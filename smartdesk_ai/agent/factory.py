"""Builds the agent graph and its responder for the CLI and the UI, from `modes.llm` and `modes.tickets`.

llm mock: hashing embeddings, scripted mock models and a separate index under data/index_mock; no keys,
no network. llm live: the real models, embeddings and index (needs OPENAI_API_KEY and a built index).
tickets mock: an in-memory client, no GitHub calls. Anything that makes live mode unusable raises
`LiveModeUnavailable`, and the caller falls back to mock with a notice (FR-F7).
"""
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from smartdesk_ai.agent.answer import make_answerer
from smartdesk_ai.agent.chat_service import Responder
from smartdesk_ai.agent.confirm import make_confirm_parser
from smartdesk_ai.agent.graph import KBDeps, build_kb_graph, make_kb_responder
from smartdesk_ai.agent.router import make_router
from smartdesk_ai.agent.state import STATE_MODELS
from smartdesk_ai.agent.ticket_agent import TicketDeps, make_drafter
from smartdesk_ai.config.env import DEFAULT_DOTENV, load_env
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.factory import ModelBuilder, openai_builder
from smartdesk_ai.llm.mock import MockAnswerModel, MockConfirmModel, MockDraftModel, MockRoutingModel
from smartdesk_ai.memory.checkpointer import make_checkpointer
from smartdesk_ai.ops.health import HealthRegistry, get_registry
from smartdesk_ai.rag.embeddings import Embeddings, HashingEmbeddings, get_embeddings
from smartdesk_ai.rag.index import IndexNotBuilt, open_index
from smartdesk_ai.rag.ingest import DEFAULT_KB_DIR, ROOT, ingest
from smartdesk_ai.rag.rerank import load_reranker
from smartdesk_ai.tickets import TicketConfigError, create_ticket_client
from smartdesk_ai.tickets.sqlite_map import TicketStore
from smartdesk_ai.tools.executor import ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry
from smartdesk_ai.tools.search_kb import make_search_kb
from smartdesk_ai.tools.ticket_tools import register_ticket_tools

DEFAULT_DATA_DIR = ROOT / "data"  # gitignored
CHECKPOINT_DB = "checkpoints.db"
TICKETS_DB = "tickets.db"


class LiveModeUnavailable(RuntimeError):
    """modes.llm or modes.tickets is `live` but something it needs is missing."""


def _builder_for(model: Any) -> ModelBuilder:
    return lambda model_id, cfg: model


def build_responder(
    settings: Settings,
    *,
    environ: Mapping[str, str] | None = None,
    dotenv_path: Path | None = DEFAULT_DOTENV,
    data_dir: Path = DEFAULT_DATA_DIR,
    kb_dir: Path = DEFAULT_KB_DIR,
    registry: HealthRegistry | None = None,
    force_mock: bool = False,
) -> Responder:
    registry = registry if registry is not None else get_registry()
    env = load_env(environ=environ, dotenv_path=dotenv_path)
    live = settings.modes.llm == "live" and not force_mock
    reranker = None
    embeddings: Embeddings
    search_settings = settings
    if live:
        if env.openai_api_key is None:
            raise LiveModeUnavailable("OPENAI_API_KEY is not set (see .env.example)")
        embeddings = get_embeddings(settings, env)
        index_dir = Path(data_dir) / "index"  # what `python -m smartdesk_ai.rag.ingest` builds
        reranker = load_reranker(settings, registry)
        answer_b = route_b = draft_b = confirm_b = openai_builder
    else:
        embeddings = HashingEmbeddings()
        index_dir = Path(data_dir) / "index_mock"
        ingest(kb_dir, index_dir, embeddings, settings.chunking)  # idempotent, offline
        # Hashing vectors score lower than the real embeddings, so retrieval uses its own mock threshold.
        search_settings = settings.model_copy(update={"thresholds": settings.thresholds.model_copy(
            update={"escalation": settings.thresholds.escalation_mock})})
        answer_b, route_b = _builder_for(MockAnswerModel()), _builder_for(MockRoutingModel())
        draft_b, confirm_b = _builder_for(MockDraftModel()), _builder_for(MockConfirmModel())
    try:
        index = open_index(index_dir, embeddings)
    except IndexNotBuilt as error:
        raise LiveModeUnavailable(str(error)) from error
    try:
        client = create_ticket_client(settings, env)
    except TicketConfigError as error:
        raise LiveModeUnavailable(str(error)) from error

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    store = TicketStore(Path(data_dir) / TICKETS_DB)
    tools = ToolRegistry()
    tools.register("search_kb", make_search_kb(index, search_settings, reranker=reranker, health=registry))
    register_ticket_tools(tools, client, store)
    executor = ToolExecutor(tools, settings, request_store=store)
    ticket_deps = TicketDeps(
        drafter=make_drafter(settings, builder=draft_b),
        executor=executor,
        parse_confirm=make_confirm_parser(settings, builder=confirm_b),
        duplicate_threshold=settings.thresholds.duplicate_title,
    )
    graph = build_kb_graph(
        KBDeps(executor=executor, answer=make_answerer(settings, builder=answer_b)),
        make_checkpointer(Path(data_dir) / CHECKPOINT_DB, STATE_MODELS),
        route=make_router(settings, builder=route_b),
        tickets=ticket_deps,
    )
    return make_kb_responder(graph)
