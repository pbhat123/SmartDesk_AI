"""Held-out exact-term check for the retrieval strategy (FR-A7, FR-A18).

`evals/lexical_set.jsonl` holds queries typed the way a user types a literal: an error code, a network or tool name,
an address, a button label. Each literal occurs in exactly one KB entry, which is the expected one. The set is a
held-out check: nothing is tuned on it. It is scored with the same five strategies as the paraphrased KB questions
(`evals.retrieval_report`), so the two tables can sit side by side.

`python -m evals.lexical_report [--mock] [--embed-new] [--out-dir DIR]`
- default: reads only cached embeddings and raises on a miss (no API call);
- `--embed-new`: embeds the queries that are not cached yet, the only paid call (a few hundred tokens), recorded in the
  spend ledger and added to the embedding cache.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from evals.common import EVALS_DIR, load_jsonl
from evals.embedding_cache import CachedEmbeddings
from evals.harness import DEFAULT_INDEX_DIR, EMBEDDING_CACHE, Resources, make_resources
from evals.retrieval_report import STRATEGIES, OfflineEmbeddings, entry_ranks, rank_metrics
from smartdesk_ai.config.settings import load_settings
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.ops.spend_ledger import end_run, start_run
from smartdesk_ai.rag.embeddings import get_embeddings
from smartdesk_ai.rag.index import KBIndex, open_index
from smartdesk_ai.rag.rerank import load_reranker

LEXICAL_SET = EVALS_DIR / "lexical_set.jsonl"


class LexicalQuery(BaseModel):
    id: str
    question: str = Field(min_length=3)
    expected_entry_id: str
    literal: str = Field(min_length=2)  # the exact string from the KB the query is built around
    heldout: bool


def lexical_rows(path: Path = LEXICAL_SET) -> list[LexicalQuery]:
    return [LexicalQuery(**r) for r in load_jsonl(path)]


def build_report(index: KBIndex, rows: list[LexicalQuery], res: Resources) -> dict[str, Any]:
    ranks = {s: entry_ranks(index, rows, s, res) for s in STRATEGIES}
    return {"mock": res.mock, "n": len(rows), "strategies": STRATEGIES,
            "overall": {s: rank_metrics(list(ranks[s].values())) for s in STRATEGIES}, "ranks": ranks}


def _print(report: dict[str, Any], rows: list[LexicalQuery]) -> None:
    print(f"\nExact-term queries (n={report['n']}, held out)\n{'strategy':<15}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'MRR':>8}")
    for name, m in report["overall"].items():
        print(f"{name:<15}{m['hit@1']:>7.3f}{m['hit@3']:>7.3f}{m['hit@5']:>7.3f}{m['mrr']:>8.3f}")
    print("\nrank of the expected entry (None = outside the list)")
    for row in rows:
        print(f"{row.id}  {row.expected_entry_id:<12}" + "  ".join(f"{s}={report['ranks'][s][row.id]}" for s in STRATEGIES))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.lexical_report")
    parser.add_argument("--mock", action="store_true", help="hashing embeddings, lexical reranker (free)")
    parser.add_argument("--embed-new", action="store_true", help="embed uncached queries (paid, a few cents at most)")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = lexical_rows()
    if args.mock:
        import tempfile

        from smartdesk_ai.rag.ingest import DEFAULT_KB_DIR, ingest

        res = make_resources(mock=True)
        index_dir = Path(tempfile.mkdtemp(prefix="lx_index_"))
        ingest(DEFAULT_KB_DIR, index_dir, res.embeddings, res.settings.chunking)
        index = open_index(index_dir, res.embeddings)
    else:
        settings = load_settings()
        if args.embed_new:
            start_run(settings, ["python", "-m", "evals.lexical_report", *(sys.argv[1:] if argv is None else argv)], "eval")
            inner = get_embeddings(settings)
            price = settings.models.embedding.price.input
        else:
            inner, price = OfflineEmbeddings(settings.models.embedding.id), 0.0
        embeddings = CachedEmbeddings(inner, EMBEDDING_CACHE, price_per_million=price)
        reranker = load_reranker(settings, HealthRegistry())
        if reranker is None:
            print("error: the reranker failed to load", file=sys.stderr)
            return 1
        res = Resources(settings=settings, embeddings=embeddings, reranker=reranker, mock=False)
        index = open_index(DEFAULT_INDEX_DIR, embeddings)
    report = build_report(index, rows, res)
    if not args.mock:
        report["api_calls"], report["embedding_cost_usd"] = embeddings.api_calls, round(embeddings.cost_usd, 8)
        embeddings.save()  # keep the new query vectors so later runs are free
    _print(report, rows)
    if not args.mock:
        print(f"\nembedding API calls: {embeddings.api_calls}  cost ${embeddings.cost_usd:.6f}")
    end_run("ok")
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "lexical_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
