"""Eval entry point: `python -m evals.run [--only ingest|retrieval|chunking|threshold] [--mock]`.

Real runs call the OpenAI embedding API (through a counting cache) and load bge-reranker-base.
`--mock` is free and offline. `--estimate` prints the API calls, tokens and cost without calling.
"""
import argparse
import sys
import tempfile
from pathlib import Path

from evals import retrieval_eval
from evals.harness import DEFAULT_INDEX_DIR, Resources, make_resources
from smartdesk_ai.rag.index import IndexNotBuilt, KBIndex, open_index
from smartdesk_ai.rag.ingest import DEFAULT_KB_DIR, ingest

STEPS = ("ingest", "retrieval", "chunking", "threshold")


def build_index(res: Resources, index_dir: Path, kb_dir: Path = DEFAULT_KB_DIR) -> None:
    report = ingest(kb_dir, index_dir, res.embeddings, res.settings.chunking)
    res.embeddings.save()
    print(f"ingest: {report.entries} entries, {report.chunks} chunks, embedded {len(report.embedded)}")


def get_index(res: Resources, index_dir: Path | None) -> KBIndex:
    """Open the index; mock runs build a throwaway one, real runs need `--only ingest` first."""
    if res.mock and index_dir is None:
        index_dir = Path(tempfile.mkdtemp(prefix="eval_index_"))
        build_index(res, index_dir)
    try:
        return open_index(index_dir or DEFAULT_INDEX_DIR, res.embeddings)
    except IndexNotBuilt:
        raise IndexNotBuilt(
            f"no real index in {index_dir or DEFAULT_INDEX_DIR}; build it first: python -m evals.run --only ingest"
        ) from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.run")
    parser.add_argument("--only", choices=STEPS, help="run one step (default: all, in order)")
    parser.add_argument("--mock", action="store_true", help="hashing embeddings and a mock reranker: free, offline")
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="write results JSON here")
    args = parser.parse_args(argv)
    steps = [args.only] if args.only else list(STEPS)

    try:
        res = make_resources(mock=args.mock, need_reranker="retrieval" in steps or "threshold" in steps)
        if "ingest" in steps and not args.mock:
            build_index(res, args.index_dir or DEFAULT_INDEX_DIR)
        index = None
        if "retrieval" in steps:
            index = get_index(res, args.index_dir)
            retrieval_eval.run(res, index, args.out_dir)
        res.embeddings.save()
    except (IndexNotBuilt, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not args.mock:
        e = res.embeddings
        print(f"embedding usage this run: {e.api_calls} API calls, {e.tokens} tokens, ${e.cost_usd:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
