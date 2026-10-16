"""On-disk layout shared by ingest and the retrievers (FR-A11)."""
import re

CHROMA_DIR = "chroma"  # dense index: a Chroma persistent store
COLLECTION = "kb_chunks"
BM25_FILE = "bm25.json"  # sparse index: chunks plus their tokens, rebuilt into BM25Okapi on load

_TOKEN = re.compile(r"[a-z0-9₹]+")


def bm25_tokens(text: str) -> list[str]:
    """Lower-case word tokens. Ingest and query time must use this same function."""
    return _TOKEN.findall(text.lower())
