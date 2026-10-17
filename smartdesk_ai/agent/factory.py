"""Builds the agent graph and its responder for the CLI and the UI, from `modes.llm`.

mock: hashing embeddings, scripted mock models and a separate index under data/index_mock; no keys,
no network. live: the real models, embeddings and index (needs OPENAI_API_KEY and a built index).
Anything that makes live mode unusable raises `LiveModeUnavailable`, and the caller falls back to
mock with a notice (FR-F7).
"""
from collections.abc import Mapping
from pathlib import Path

from smartdesk_ai.agent.answer import make_answerer
from smartdesk_ai.agent.chat_service import Responder
from smartdesk_ai.agent.graph import KBDeps, build_kb_graph, make_kb_responder
from smartdesk_ai.agent.state import STATE_MODELS
from smartdesk_ai.config.env import DEFAULT_DOTENV, load_env
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.factory import ModelBuilder, openai_builder
from smartdesk_ai.llm.mock import MockAnswerModel
from smartdesk_ai.memory.checkpointer import make_checkpointer
from smartdesk_ai.ops.health import HealthRegistry, get_registry
from smartdesk_ai.rag.embeddings import Embeddings, HashingEmbeddings, get_embeddings
from smartdesk_ai.rag.index import IndexNotBuilt, open_index
from smartdesk_ai.rag.ingest import DEFAULT_KB_DIR, ROOT, ingest
from smartdesk_ai.rag.rerank import load_reranker
from smartdesk_ai.tools.executor import ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry
from smartdesk_ai.tools.search_kb import make_search_kb

DEFAULT_DATA_DIR = ROOT / "data"  # gitignored
CHECKPOINT_DB = "checkpoints.db"


class LiveModeUnavailable(RuntimeError):
    """modes.llm is `live` but something it needs is missing."""


def _mock_builder(model_id: str, cfg: object) -> MockAnswerModel:
    return MockAnswerModel()


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
    live = settings.modes.llm == "live" and not force_mock
    builder: ModelBuilder = openai_builder if live else _mock_builder
    reranker = None
    embeddings: Embeddings
    if live:
        env = load_env(environ=environ, dotenv_path=dotenv_path)
        if env.openai_api_key is None:
            raise LiveModeUnavailable("OPENAI_API_KEY is not set (see .env.example)")
        embeddings = get_embeddings(settings, env)
        index_dir = Path(data_dir) / "index"  # what `python -m smartdesk_ai.rag.ingest` builds
        reranker = load_reranker(settings, registry)
    else:
        embeddings = HashingEmbeddings()
        index_dir = Path(data_dir) / "index_mock"
        ingest(kb_dir, index_dir, embeddings, settings.chunking)  # idempotent, offline
    try:
        index = open_index(index_dir, embeddings)
    except IndexNotBuilt as error:
        raise LiveModeUnavailable(str(error)) from error

    tools = ToolRegistry()
    tools.register("search_kb", make_search_kb(index, settings, reranker=reranker, health=registry))
    deps = KBDeps(executor=ToolExecutor(tools, settings), answer=make_answerer(settings, builder=builder))
    saver = make_checkpointer(Path(data_dir) / CHECKPOINT_DB, STATE_MODELS)
    return make_kb_responder(build_kb_graph(deps, saver))
