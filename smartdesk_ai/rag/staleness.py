"""KB staleness (FR-H6): a hash per policy file, and stale flags derived from it.

Ingest records a SHA-256 per file plus a fingerprint of the chunking config and the embedder
in `state.json`. `check_staleness` compares them with the files on disk, read-only: a changed or
deleted file flags the chunks that came from it, and nothing else. A changed chunking config or
embedder makes every chunk stale. Re-running ingest re-embeds the flagged chunks and the flags
clear because the stored hashes catch up.
"""
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

from smartdesk_ai.config.settings import Chunking
from smartdesk_ai.rag.index import BM25_FILE, POLICY_FILES

STATE_FILE = "state.json"


class StaleReport(BaseModel):
    stale_files: list[str] = []  # present in the index, content changed since ingest
    stale_chunk_ids: list[str] = []  # indexed chunks that need re-embedding or pruning
    new_files: list[str] = []  # on disk, never ingested
    deleted_files: list[str] = []  # ingested, no longer on disk
    reason: str | None = None  # set when everything is stale at once


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_hashes(kb_dir: Path) -> dict[str, str]:
    return {name: file_hash(kb_dir / name) for name in POLICY_FILES if (kb_dir / name).is_file()}


def fingerprint(cfg: Chunking, embedder: str) -> dict:
    return {"chunking": cfg.model_dump(), "embedder": embedder}


def read_state(index_dir: Path) -> dict | None:
    path = Path(index_dir) / STATE_FILE
    return json.loads(path.read_text()) if path.is_file() else None


def write_state(index_dir: Path, files: dict[str, str], fp: dict) -> None:
    path = Path(index_dir) / STATE_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"files": files, "fingerprint": fp}, indent=2, sort_keys=True))
    tmp.replace(path)


def _indexed(index_dir: Path) -> list[tuple[str, str]]:
    """(chunk_id, source_file) for every chunk currently in the index."""
    path = Path(index_dir) / BM25_FILE
    if not path.is_file():
        return []
    return [(c["chunk_id"], c["source_file"]) for c in json.loads(path.read_text())["chunks"]]


def check_staleness(kb_dir: Path, index_dir: Path, cfg: Chunking, embedder: str) -> StaleReport:
    indexed = _indexed(index_dir)
    if not indexed:
        return StaleReport()  # nothing ingested yet, so nothing can be stale
    all_ids = sorted(cid for cid, _ in indexed)
    all_files = sorted({src for _, src in indexed})
    state = read_state(index_dir)
    if state is None:
        return StaleReport(stale_files=all_files, stale_chunk_ids=all_ids, reason="no state recorded for this index")
    expected, stored_fp = fingerprint(cfg, embedder), state["fingerprint"]
    if stored_fp != expected:
        what = "chunking config" if stored_fp["chunking"] != expected["chunking"] else "embedder"
        return StaleReport(stale_files=all_files, stale_chunk_ids=all_ids, reason=f"{what} changed since the last ingest")

    current, stored = policy_hashes(Path(kb_dir)), state["files"]
    changed = sorted(f for f in current if f in stored and current[f] != stored[f])
    deleted = sorted(f for f in stored if f not in current)
    new = sorted(f for f in current if f not in stored)
    flagged = set(changed) | set(deleted)
    return StaleReport(
        stale_files=changed,
        stale_chunk_ids=sorted(cid for cid, src in indexed if src in flagged),
        new_files=new,
        deleted_files=deleted,
    )
