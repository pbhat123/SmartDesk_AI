"""Heading-based chunker (FR-A3).

One chunk per KB entry. An entry above `split_above_tokens` is split into pieces of at
most `target_max_tokens`, each overlapping the previous by `overlap_pct` percent, and
every piece keeps the parent entry id. The embedded text of each chunk starts with a
"Document > Section" header.
"""
import re
from collections.abc import Callable
from functools import lru_cache

import tiktoken

from smartdesk_ai.config.settings import Chunking
from smartdesk_ai.rag.kb_schema import Entry
from smartdesk_ai.rag.models import Chunk

TokenCounter = Callable[[str], int]
_UNIT = re.compile(r"\S+\s*")  # a word with its trailing whitespace, so pieces rejoin exactly
_SNAP_WINDOW = 0.3  # a piece may end early at a sentence boundary in the last 30% of its window


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def _is_sentence_end(unit: str) -> bool:
    return unit.rstrip().endswith((".", "!", "?", ":")) or unit.endswith("\n")


def _split(text: str, max_tokens: int, overlap_pct: int, count: TokenCounter) -> list[str]:
    units = _UNIT.findall(text)
    costs = [count(u) for u in units]
    overlap_tokens = round(max_tokens * overlap_pct / 100)
    pieces: list[str] = []
    start = 0
    while True:
        end, used = start, 0
        while end < len(units) and (end == start or used + costs[end] <= max_tokens):
            used += costs[end]
            end += 1
        if end < len(units):  # not the last piece: prefer to stop at a sentence end
            floor = start + int((end - start) * (1 - _SNAP_WINDOW))
            snap = next((j for j in range(end, floor, -1) if _is_sentence_end(units[j - 1])), None)
            end = snap or end
        pieces.append("".join(units[start:end]).strip())
        if end >= len(units):
            return pieces
        back, carried = 0, 0
        while end - back - 1 > start and carried < overlap_tokens:
            back += 1
            carried += costs[end - back]
        start = end - back


def chunk_entry(entry: Entry, cfg: Chunking, count: TokenCounter = count_tokens) -> list[Chunk]:
    """Turn one validated entry into one chunk, or into overlapping pieces if it is long."""
    header = f"{entry.document} > {entry.heading}" if entry.document else entry.heading
    if count(entry.body) <= cfg.split_above_tokens:
        texts = [entry.body]
    else:
        texts = _split(entry.body, cfg.target_max_tokens, cfg.overlap_pct, count)
    return [
        Chunk(
            chunk_id=f"{entry.id}#{i}", entry_id=entry.id, domain=entry.domain, title=entry.title,
            header=header, text=text, source_file=entry.source, piece=i, pieces=len(texts),
        )
        for i, text in enumerate(texts)
    ]
