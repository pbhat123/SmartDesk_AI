"""On-disk layout and the opened index shared by ingest and the retrievers (FR-A11)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from smartdesk_ai.rag.models import Chunk, ParentEntry

if TYPE_CHECKING:  # embeddings imports STOPWORDS from here
    from smartdesk_ai.rag.embeddings import Embeddings

CHROMA_DIR = "chroma"  # dense index: a Chroma persistent store
COLLECTION = "kb_chunks"
POLICY_FILES = ("company.md", "it_support.md", "onboarding.md", "hr_policies.md", "payroll_comp.md")  # the only files ever indexed
BM25_FILE = "bm25.json"  # sparse index: chunks plus their tokens, rebuilt into BM25Okapi on load

_TOKEN = re.compile(r"[a-z0-9₹]+")
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it me my of on or so the to we what when "
    "where which who will with you your".split()
)


def bm25_tokens(text: str) -> list[str]:
    """Lower-case word tokens without stopwords. Ingest and query time must use this same function."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class IndexNotBuilt(RuntimeError):
    """The index directory has no ingested KB yet."""


@dataclass
class KBIndex:
    """Both indexes plus the embedder used to query the dense one."""

    chunks: list[Chunk]
    entries: dict[str, ParentEntry]  # full parent text per entry id
    bm25: BM25Okapi
    collection: Collection
    embeddings: Embeddings

    @cached_property
    def by_id(self) -> dict[str, Chunk]:
        return {c.chunk_id: c for c in self.chunks}


def open_index(index_dir: Path, embeddings: Embeddings) -> KBIndex:
    index_dir = Path(index_dir)
    bm25_path = index_dir / BM25_FILE
    if not bm25_path.is_file() or not (index_dir / CHROMA_DIR).is_dir():
        raise IndexNotBuilt(f"no index in {index_dir}; run: python -m smartdesk_ai.rag.ingest")
    payload = json.loads(bm25_path.read_text())
    if "entries" not in payload:
        raise IndexNotBuilt(f"index in {index_dir} is from an older version; rebuild: python -m smartdesk_ai.rag.ingest")
    client = chromadb.PersistentClient(
        path=str(index_dir / CHROMA_DIR), settings=ChromaSettings(anonymized_telemetry=False)
    )
    return KBIndex(
        chunks=[Chunk(**c) for c in payload["chunks"]],
        entries={k: ParentEntry(**v) for k, v in payload["entries"].items()},
        bm25=BM25Okapi(payload["tokens"]),
        collection=client.get_collection(COLLECTION),
        embeddings=embeddings,
    )
