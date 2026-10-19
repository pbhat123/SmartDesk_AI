"""Re-runnable KB ingest: parse, chunk, embed into Chroma, build the BM25 index (FR-A11).

Only the five policy files are read, by name. GAPS.md, SOURCES.md, qa.json and any other
file in kb/ are never indexed. Running it again leaves the same chunks: new or changed
chunks are upserted and chunks of deleted or renamed entries are removed.
"""
import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel

from smartdesk_ai.config.settings import Chunking, load_settings
from smartdesk_ai.rag.chunking import TokenCounter, chunk_entry, count_tokens
from smartdesk_ai.rag.embeddings import Embeddings, HashingEmbeddings, embedder_id, get_embeddings
from smartdesk_ai.rag.index import BM25_FILE, CHROMA_DIR, COLLECTION, POLICY_FILES, bm25_tokens
from smartdesk_ai.rag.kb_schema import Entry, KBValidationError, parse_entries, validate_entries
from smartdesk_ai.rag.models import Chunk, ParentEntry
from smartdesk_ai.rag.staleness import check_staleness, fingerprint, policy_hashes, write_state

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KB_DIR = ROOT / "kb"
DEFAULT_INDEX_DIR = ROOT / "data" / "index"  # data/ is gitignored


class IngestError(RuntimeError):
    """Nothing could be indexed."""


class IngestReport(BaseModel):
    files_read: list[str]
    missing_files: list[str]
    entries: int
    chunks: int
    added: list[str]  # chunk ids new to the index
    removed: list[str]  # chunk ids pruned (deleted or renamed entries, or a missing file)
    embedded: list[str] = []  # chunk ids embedded this run: changed, new, or everything after a config change
    kb_changed: bool = False  # chunks were embedded or removed: the answer cache must be cleared (FR-H6)


def load_policy_entries(kb_dir: Path) -> tuple[list[Entry], list[str], list[str]]:
    entries: list[Entry] = []
    read: list[str] = []
    missing: list[str] = []
    for name in POLICY_FILES:
        path = kb_dir / name
        if not path.is_file():
            missing.append(name)
            logger.warning("policy file missing, its chunks will be pruned: %s", name)
            continue
        entries += parse_entries(path.read_text(), source=name)
        read.append(name)
    if not read:
        raise IngestError(f"none of the policy files {POLICY_FILES} found in {kb_dir}")
    return validate_entries(entries), read, missing


def _write_bm25(index_dir: Path, chunks: list[Chunk], entries: list[Entry]) -> None:
    parents = {
        e.id: ParentEntry(
            entry_id=e.id, domain=e.domain, title=e.title, source_file=e.source, text=e.body,
            header=f"{e.document} > {e.heading}" if e.document else e.heading,
        ).model_dump()
        for e in entries
    }
    payload = {
        "chunks": [c.model_dump() for c in chunks],
        "entries": parents,
        "tokens": [bm25_tokens(c.embed_text) for c in chunks],
    }
    tmp = index_dir / (BM25_FILE + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False))
    os.replace(tmp, index_dir / BM25_FILE)  # readers never see a half-written file


def build_index(index_dir: Path, chunks: list[Chunk], entries: list[Entry], embeddings: Embeddings) -> None:
    """Write both indexes for an arbitrary chunk list, replacing whatever was there. Used by the
    chunking ablation; `ingest` is the incremental path for the real KB."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(index_dir / CHROMA_DIR), settings=ChromaSettings(anonymized_telemetry=False)
    )
    if COLLECTION in {c.name for c in client.list_collections()}:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings.embed_documents([c.embed_text for c in chunks]),
        documents=[c.text for c in chunks],
        metadatas=[
            {"entry_id": c.entry_id, "domain": c.domain, "title": c.title, "header": c.header,
             "source_file": c.source_file, "piece": c.piece, "pieces": c.pieces}
            for c in chunks
        ],
    )
    _write_bm25(index_dir, chunks, entries)


def ingest(
    kb_dir: Path,
    index_dir: Path,
    embeddings: Embeddings,
    cfg: Chunking,
    count: TokenCounter = count_tokens,
    on_kb_changed: Callable[[IngestReport], None] | None = None,
) -> IngestReport:
    """Build or refresh both indexes. Only chunks of new or changed files are embedded (FR-H6);
    `on_kb_changed` is called once if the KB changed, so the answer cache can be cleared."""
    kb_dir, index_dir = Path(kb_dir), Path(index_dir)
    entries, read, missing = load_policy_entries(kb_dir)  # validates before anything is written
    chunks = [c for e in entries for c in chunk_entry(e, cfg, count)]
    embedder = embedder_id(embeddings)
    stale = check_staleness(kb_dir, index_dir, cfg, embedder)  # compares with the previous ingest
    index_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(index_dir / CHROMA_DIR), settings=ChromaSettings(anonymized_telemetry=False)
    )
    exists = COLLECTION in {c.name for c in client.list_collections()}
    before = set(client.get_collection(COLLECTION).get()["ids"]) if exists else set()
    if stale.reason and exists:
        client.delete_collection(COLLECTION)  # a new embedder may use another vector size
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    kept = set() if stale.reason else before  # vectors still valid for unchanged files
    redo_files = set(read) if stale.reason else set(stale.stale_files) | set(stale.new_files)
    after = {c.chunk_id for c in chunks}
    to_embed = [c for c in chunks if c.source_file in redo_files or c.chunk_id not in kept]

    if to_embed:
        collection.upsert(
            ids=[c.chunk_id for c in to_embed],
            embeddings=embeddings.embed_documents([c.embed_text for c in to_embed]),
            documents=[c.text for c in to_embed],
            metadatas=[
                {"entry_id": c.entry_id, "domain": c.domain, "title": c.title, "header": c.header,
                 "source_file": c.source_file, "piece": c.piece, "pieces": c.pieces}
                for c in to_embed
            ],
        )
    removed = sorted(before - after)
    if removed and not stale.reason:  # after a rebuild the collection never held them
        collection.delete(ids=removed)
    _write_bm25(index_dir, chunks, entries)
    write_state(index_dir, {f: h for f, h in policy_hashes(kb_dir).items() if f in read}, fingerprint(cfg, embedder))

    embedded = sorted(c.chunk_id for c in to_embed)
    report = IngestReport(
        files_read=read, missing_files=missing, entries=len(entries), chunks=len(chunks),
        added=sorted(after - before), removed=removed, embedded=embedded,
        kb_changed=bool(embedded or removed),
    )
    if report.kb_changed and on_kb_changed is not None:
        try:
            on_kb_changed(report)
        except Exception:  # a broken cache must not fail the ingest that already succeeded
            logger.exception("KB-changed handler failed")
    return report


def _redis_cache_clearer(args: argparse.Namespace, settings, environ) -> Callable[[IngestReport], None] | None:
    """When REDIS_URL is set, a handler that clears the shared answer cache after a KB change (FR-H6). A running app's
    in-memory cache cannot be reached from here; it ignores old answers by itself (they carry the KB version)."""
    from smartdesk_ai.config.env import DEFAULT_DOTENV, load_env
    from smartdesk_ai.ops.cache import MemoryBackend, clear_answers, open_backend
    from smartdesk_ai.ops.cache_log import CACHE_LOG_FILE, CacheLog
    from smartdesk_ai.ops.health import HealthRegistry

    env = load_env(environ=environ, dotenv_path=DEFAULT_DOTENV if environ is None else None)
    if not env.redis_url:
        return None
    backend = open_backend(settings, env, HealthRegistry(), mode="mock" if args.mock else settings.modes.llm)
    if isinstance(backend, MemoryBackend):
        print("Redis is set but not reachable: the shared answer cache was not cleared.", file=sys.stderr)
        return None
    log = CacheLog(Path(args.index_dir).parent / CACHE_LOG_FILE)
    return lambda report: clear_answers(backend, log)


def main(argv: list[str] | None = None, *, environ: dict[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or refresh the KB indexes (idempotent).")
    parser.add_argument("--kb-dir", type=Path, default=DEFAULT_KB_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--mock", action="store_true", help="offline hashing embeddings, no API spend")
    args = parser.parse_args(argv)
    settings = load_settings()
    embeddings = HashingEmbeddings() if args.mock else get_embeddings(settings)
    try:
        report = ingest(args.kb_dir, args.index_dir, embeddings, settings.chunking,
                        on_kb_changed=_redis_cache_clearer(args, settings, environ))
    except (IngestError, KBValidationError) as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1
    print(f"{report.entries} entries, {report.chunks} chunks; added {len(report.added)}, removed {len(report.removed)}")
    if report.missing_files:
        print(f"missing policy files: {', '.join(report.missing_files)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
