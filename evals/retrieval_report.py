"""Retrieval ablation per question, from cached embeddings only (FR-A7).

`python -m evals.retrieval_report [--mock] [--out-dir DIR]` ranks the expected entry for every KB question under
dense, BM25, hybrid, hybrid + rerank and dense + rerank, and reports hit@1/3/5 and MRR overall, per split, per
question type and for the lower and upper half of the questions by lexical overlap with their expected entry. The
overlap split shows paraphrase bias: the questions are paraphrased on purpose (FR-A18), which handicaps BM25.
A real run reads `data/embedding_cache.json`; an uncached text raises instead of calling the API.
"""
import argparse
import json
import re
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evals.embedding_cache import CachedEmbeddings
from evals.harness import DEFAULT_INDEX_DIR, EMBEDDING_CACHE, Resources, kb_rows, make_resources
from evals.retrieval_eval import rank_results
from evals.schemas import KBQuestion
from smartdesk_ai.config.settings import load_settings
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.rag.index import STOPWORDS, KBIndex, open_index
from smartdesk_ai.rag.rerank import load_reranker

STRATEGIES = ["dense", "bm25", "hybrid", "hybrid+rerank", "dense+rerank"]
KS = (1, 3, 5)
MIN_WORD = 4


class OfflineEmbeddings:
    """Stands in for the real embedder behind the cache: any text that is not cached is an error, never an API call."""

    def __init__(self, name: str) -> None:
        self.name = name

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("text not in the embedding cache: this report makes no API calls")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("text not in the embedding cache: this report makes no API calls")


def rank_metrics(ranks: Sequence[int | None]) -> dict[str, float]:
    """hit@k = share of questions whose expected entry is within the top k; MRR = mean of 1 / rank (0 if missing)."""
    n = len(ranks) or 1
    out = {f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in KS}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    return out


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9₹]+", text.lower()) if len(w) >= MIN_WORD and w not in STOPWORDS}


def overlap_share(question: str, entry_text: str) -> float:
    """Share of the question's content words that also appear in its expected entry."""
    q = _words(question)
    return len(q & _words(entry_text)) / len(q) if q else 0.0


def split_by_overlap(rows: Sequence[KBQuestion], overlap: dict[str, float]) -> tuple[list[KBQuestion], list[KBQuestion]]:
    ordered = sorted(rows, key=lambda r: overlap[r.id])
    half = len(ordered) // 2
    return ordered[:half], ordered[half:]


def entry_ranks(index: KBIndex, rows: Sequence[KBQuestion], strategy: str, res: Resources) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for row in rows:
        entries = [c.entry_id for c in rank_results(strategy, index, row.question, res.settings, res.reranker)]
        out[row.id] = entries.index(row.expected_entry_id) + 1 if row.expected_entry_id in entries else None
    return out


def build_report(index: KBIndex, rows: list[KBQuestion], res: Resources) -> dict[str, Any]:
    ranks = {s: entry_ranks(index, rows, s, res) for s in STRATEGIES}
    text: dict[str, str] = {}
    for chunk in index.chunks:
        text[chunk.entry_id] = text.get(chunk.entry_id, "") + " " + chunk.text
    overlap = {r.id: overlap_share(r.question, text.get(r.expected_entry_id, "")) for r in rows}
    low, high = split_by_overlap(rows, overlap)

    def table(subset: Sequence[KBQuestion]) -> dict[str, dict[str, float]]:
        return {s: rank_metrics([ranks[s][r.id] for r in subset]) for s in STRATEGIES}

    return {
        "mock": res.mock, "n": len(rows), "strategies": STRATEGIES, "overall": table(rows),
        "by_split": {k: table([r for r in rows if r.split == k]) for k in sorted({r.split for r in rows})},
        "by_type": {k: table([r for r in rows if r.type == k]) for k in sorted({r.type for r in rows})},
        "by_overlap": {"low": table(low), "high": table(high)},
        "overlap_mean": statistics.mean(overlap.values()) if overlap else 0.0, "ranks": ranks,
    }


def _print(report: dict[str, Any]) -> None:
    def show(title: str, tables: dict[str, dict[str, float]]) -> None:
        print(f"\n{title}\n{'strategy':<15}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'MRR':>8}")
        for name, m in tables.items():
            print(f"{name:<15}{m['hit@1']:>7.3f}{m['hit@3']:>7.3f}{m['hit@5']:>7.3f}{m['mrr']:>8.3f}")

    show(f"All KB questions (n={report['n']})", report["overall"])
    for key in ("by_split", "by_type", "by_overlap"):
        for name, tables in report[key].items():
            show(f"{key[3:]} = {name}", tables)
    print(f"\nmean lexical overlap between a question and its expected entry: {report['overlap_mean']:.2f}")


def run(res: Resources, index: KBIndex, out_dir: Path | None) -> dict[str, Any]:
    report = build_report(index, kb_rows(None), res)
    _print(report)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "retrieval_report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.retrieval_report")
    parser.add_argument("--mock", action="store_true", help="hashing embeddings, lexical reranker (free)")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.mock:
        import tempfile

        from smartdesk_ai.rag.ingest import DEFAULT_KB_DIR, ingest

        res = make_resources(mock=True)
        index_dir = Path(tempfile.mkdtemp(prefix="rr_index_"))
        ingest(DEFAULT_KB_DIR, index_dir, res.embeddings, res.settings.chunking)
        index = open_index(index_dir, res.embeddings)
    else:
        settings = load_settings()
        embeddings = CachedEmbeddings(OfflineEmbeddings(settings.models.embedding.id), EMBEDDING_CACHE, price_per_million=0.0)
        reranker = load_reranker(settings, HealthRegistry())
        if reranker is None:
            print("error: the reranker failed to load", file=sys.stderr)
            return 1
        res = Resources(settings=settings, embeddings=embeddings, reranker=reranker, mock=False)
        index = open_index(DEFAULT_INDEX_DIR, embeddings)
    run(res, index, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
