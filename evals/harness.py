"""Shared set-up for the eval runs: settings, embedder, reranker, data sets, and the mock/real switch.

Mock mode (`--mock`) uses hashing embeddings and a lexical stand-in for the cross-encoder: free,
offline and deterministic, for tests and dry runs. Real mode uses OpenAI embeddings (through a disk
cache that counts every call, token and dollar) and the local bge-reranker-base.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from evals.common import EVALS_DIR, load_jsonl
from evals.embedding_cache import CachedEmbeddings
from evals.schemas import GapQuestion, KBQuestion
from smartdesk_ai.config.settings import Settings, load_settings
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.rag.embeddings import HashingEmbeddings, get_embeddings
from smartdesk_ai.rag.rerank import CrossEncoderReranker, Reranker, load_reranker

ROOT = EVALS_DIR.parent
DEFAULT_INDEX_DIR = ROOT / "data" / "index"
EMBEDDING_CACHE = ROOT / "data" / "embedding_cache.json"
RESULTS_DIR = EVALS_DIR / "results"
_WORD = re.compile(r"[a-z0-9]+")


def lexical_scorer(pairs: list[tuple[str, str]]) -> list[float]:
    """Mock cross-encoder: the logit grows with the share of query words found in the passage."""
    out = []
    for query, passage in pairs:
        q, p = set(_WORD.findall(query.lower())), set(_WORD.findall(passage.lower()))
        out.append(8.0 * len(q & p) / max(len(q), 1) - 3.0)
    return out


@dataclass
class Resources:
    settings: Settings
    embeddings: CachedEmbeddings
    reranker: Reranker | None
    mock: bool

    @property
    def banner(self) -> str:
        return "MOCK embeddings and reranker, not real results" if self.mock else (
            f"real: {self.settings.models.embedding.id} + {self.settings.models.reranker.id}"
        )


def make_resources(*, mock: bool, need_reranker: bool = True) -> Resources:
    settings = load_settings()
    if mock:
        return Resources(
            settings=settings,
            embeddings=CachedEmbeddings(HashingEmbeddings(), None, price_per_million=0.0),
            reranker=CrossEncoderReranker(lexical_scorer),
            mock=True,
        )
    embeddings = CachedEmbeddings(
        get_embeddings(settings), EMBEDDING_CACHE, price_per_million=settings.models.embedding.price.input
    )
    reranker = load_reranker(settings, HealthRegistry()) if need_reranker else None
    if need_reranker and reranker is None:
        raise RuntimeError("the reranker failed to load; a real run needs it (see the warning above)")
    return Resources(settings=settings, embeddings=embeddings, reranker=reranker, mock=False)


def kb_rows(split: str | None = None) -> list[KBQuestion]:
    rows = [KBQuestion(**r) for r in load_jsonl(EVALS_DIR / "kb_questions.jsonl")]
    return [r for r in rows if split is None or r.split == split]


def gap_rows(split: str | None = None) -> list[GapQuestion]:
    rows = [GapQuestion(**r) for r in load_jsonl(EVALS_DIR / "gap_questions.jsonl")]
    return [r for r in rows if split is None or r.split == split]


def live_settings(settings: Settings) -> Settings:
    """The same settings with the real models switched on. Tickets stay in the mock mode they are in."""
    return settings.model_copy(update={"modes": settings.modes.model_copy(update={"llm": "live"})})


def link_live_index(data_dir: Path) -> None:
    """Make the real KB index (built once by the ingest script) visible in a scratch data folder, so a live
    eval shares the index but keeps its checkpoints, logs and counters apart from the real ones."""
    from smartdesk_ai.agent.factory import DEFAULT_DATA_DIR

    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "index"
    if target.exists():
        return
    source = DEFAULT_DATA_DIR / "index"
    if not source.exists():
        raise RuntimeError("no live KB index; build it first: python -m smartdesk_ai.rag.ingest")
    target.symlink_to(source, target_is_directory=True)
