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
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel

from smartdesk_ai.config.settings import Chunking, load_settings
from smartdesk_ai.rag.chunking import TokenCounter, chunk_entry, count_tokens
from smartdesk_ai.rag.embeddings import Embeddings, HashingEmbeddings, get_embeddings
from smartdesk_ai.rag.index import BM25_FILE, CHROMA_DIR, COLLECTION, bm25_tokens
from smartdesk_ai.rag.kb_schema import Entry, KBValidationError, parse_entries, validate_entries
from smartdesk_ai.rag.models import Chunk, ParentEntry

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KB_DIR = ROOT / "kb"
DEFAULT_INDEX_DIR = ROOT / "data" / "index"  # data/ is gitignored
POLICY_FILES = ("company.md", "it_support.md", "onboarding.md", "hr_policies.md", "payroll_comp.md")


class IngestError(RuntimeError):
    """Nothing could be indexed."""


class IngestReport(BaseModel):
    files_read: list[str]
    missing_files: list[str]
    entries: int
    chunks: int
    added: list[str]  # chunk ids new to the index
    removed: list[str]  # chunk ids pruned (deleted or renamed entries, or a missing file)


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


def ingest(
    kb_dir: Path, index_dir: Path, embeddings: Embeddings, cfg: Chunking, count: TokenCounter = count_tokens
) -> IngestReport:
    entries, read, missing = load_policy_entries(Path(kb_dir))  # validates before anything is written
    chunks = [c for e in entries for c in chunk_entry(e, cfg, count)]
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(index_dir / CHROMA_DIR), settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    before = set(collection.get()["ids"])
    after = {c.chunk_id for c in chunks}

    vectors = embeddings.embed_documents([c.embed_text for c in chunks])
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[
            {"entry_id": c.entry_id, "domain": c.domain, "title": c.title, "header": c.header,
             "source_file": c.source_file, "piece": c.piece, "pieces": c.pieces}
            for c in chunks
        ],
    )
    removed = sorted(before - after)
    if removed:
        collection.delete(ids=removed)
    _write_bm25(index_dir, chunks, entries)
    return IngestReport(
        files_read=read, missing_files=missing, entries=len(entries), chunks=len(chunks),
        added=sorted(after - before), removed=removed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or refresh the KB indexes (idempotent).")
    parser.add_argument("--kb-dir", type=Path, default=DEFAULT_KB_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--mock", action="store_true", help="offline hashing embeddings, no API spend")
    args = parser.parse_args(argv)
    settings = load_settings()
    embeddings = HashingEmbeddings() if args.mock else get_embeddings(settings)
    try:
        report = ingest(args.kb_dir, args.index_dir, embeddings, settings.chunking)
    except (IngestError, KBValidationError) as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1
    print(f"{report.entries} entries, {report.chunks} chunks; added {len(report.added)}, removed {len(report.removed)}")
    if report.missing_files:
        print(f"missing policy files: {', '.join(report.missing_files)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
