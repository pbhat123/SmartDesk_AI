"""Retrieval data types shared by chunking, ingest, retrievers, fusion and rerank."""
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Chunk(BaseModel):
    """One embedded unit. A chunk is a whole KB entry, or a piece of a long one (FR-A3)."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str  # "<entry_id>#<piece>", stable across ingests
    entry_id: str  # parent entry; every piece of a split entry keeps it
    domain: Literal["IT", "HR"]
    title: str
    header: str  # "Document > Section"
    text: str
    source_file: str
    piece: int = 0
    pieces: int = 1

    @property
    def embed_text(self) -> str:
        """What gets embedded and BM25-indexed: the header, then the text (FR-A3)."""
        return f"{self.header}\n{self.text}"


class ScoredChunk(BaseModel):
    """A chunk with the score and the pipeline stage that produced it (FR-A6)."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    stage: str  # bm25, dense, rrf, rerank


class ParentEntry(BaseModel):
    """A whole KB entry as the answering step sees it (FR-A6): never just one piece of it."""

    model_config = ConfigDict(frozen=True)

    entry_id: str
    domain: Literal["IT", "HR"]
    title: str
    header: str  # "Document > Section"
    text: str  # the full entry body
    source_file: str


class RetrievedEntry(ParentEntry):
    score: float  # of the best-scoring piece
    chunk_id: str  # that piece
