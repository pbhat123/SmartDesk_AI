"""Eval entry point: `python -m evals.run [--only ingest|retrieval|chunking|threshold] [--mock]`.

Real runs call the OpenAI embedding API (through a counting cache) and load bge-reranker-base.
`--mock` is free and offline. `--estimate` prints the API calls, tokens and cost without calling.
"""
import argparse
import sys
import tempfile
from pathlib import Path

from evals import agent_metrics, chunking_ablation, gap_eval, rag_metrics, retrieval_eval, tune_duplicate, tune_routing, tune_threshold
from evals.harness import DEFAULT_INDEX_DIR, ROOT, Resources, make_resources
from evals.run_record import build_record, write_record
from smartdesk_ai.rag.index import IndexNotBuilt, KBIndex, open_index
from smartdesk_ai.rag.ingest import DEFAULT_KB_DIR, ingest

STEPS = ("ingest", "retrieval", "chunking", "threshold", "rag", "agent", "gap", "duplicate", "routing")
FREE_STEPS = ("ingest", "retrieval", "chunking", "threshold", "duplicate")
RESULT_FILES = {"routing": "routing_threshold.json", "duplicate": "duplicate_threshold.json", "gap": "gap_eval.json", "rag": "rag_metrics.json", "agent": "agent_metrics.json", "retrieval": "retrieval.json", "chunking": "chunking.json", "threshold": "threshold.json"}


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
    parser.add_argument("--judge", choices=("mock", "ragas"), default="mock", help="rag step: the metric judge")
    parser.add_argument("--confirm-spend", action="store_true", help="allow a real run that calls paid models")
    parser.add_argument("--chosen", type=float, default=None, help="threshold step: record this escalation value's dev and test numbers")
    parser.add_argument("--write-config", action="store_true", help="threshold step: write --chosen (or the proposal) into --config")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml", help="file --write-config edits")
    args = parser.parse_args(argv)
    # With no --only: a mock run does everything; a real run does the free steps. The paid ones (Ragas judge,
    # live classifier, live agent) are asked for by name and need --confirm-spend.
    steps = [args.only] if args.only else (list(STEPS) if args.mock else list(FREE_STEPS))
    if "routing" in steps and not args.mock and not args.confirm_spend:
        print("error: the live routing run calls the paid classifier; add --confirm-spend once you have approved the cost",
              file=sys.stderr)
        return 1
    if "gap" in steps and not args.mock and not args.confirm_spend:
        print("error: the live gap run calls paid models; add --confirm-spend once you have approved the cost",
              file=sys.stderr)
        return 1
    if "agent" in steps and not args.mock:
        print("error: the agent step runs only in --mock mode for now (a live run needs a cost plan first)", file=sys.stderr)
        return 1
    if "rag" in steps and not args.mock and args.judge != "ragas":
        print("error: the rag step runs with --mock, or for real with --judge ragas --confirm-spend", file=sys.stderr)
        return 1
    if "rag" in steps and args.judge == "ragas" and not args.confirm_spend:
        print("error: --judge ragas calls paid models; add --confirm-spend once you have approved the cost", file=sys.stderr)
        return 1
    if args.write_config and args.mock and args.config.resolve() == (ROOT / "config.yaml").resolve():
        print("error: refusing to write mock values into the real config.yaml; pass --config <other file>", file=sys.stderr)
        return 1

    try:
        res = make_resources(mock=args.mock, need_reranker=bool({"retrieval", "chunking", "threshold", "rag"} & set(steps)))
        if "ingest" in steps and not args.mock:
            build_index(res, args.index_dir or DEFAULT_INDEX_DIR)
        index = None
        if {"retrieval", "chunking", "threshold", "rag"} & set(steps):
            index = get_index(res, args.index_dir)
        if "retrieval" in steps:
            retrieval_eval.run(res, index, args.out_dir)
        if "rag" in steps:
            if args.judge == "ragas":
                rag_metrics.run_real(res, index, args.out_dir)
            else:
                rag_metrics.run(res, index, args.out_dir)
        if "agent" in steps:
            agent_metrics.run(res, args.out_dir, Path(tempfile.mkdtemp(prefix="eval_agent_")))
        if "routing" in steps:
            from smartdesk_ai.llm.factory import openai_builder
            from smartdesk_ai.llm.mock import MockRoutingModel

            builder = (lambda model_id, cfg: MockRoutingModel()) if args.mock else openai_builder
            tune_routing.run(res.settings, args.out_dir, builder=builder, live=not args.mock)
        if "duplicate" in steps:
            tune_duplicate.run(res.settings, args.out_dir)
        if "gap" in steps:
            gap_eval.run(res, args.out_dir, Path(tempfile.mkdtemp(prefix="eval_gap_")))
        if "chunking" in steps:
            fixed_dir = Path(tempfile.mkdtemp(prefix="eval_fixed_")) if args.mock else DEFAULT_INDEX_DIR.parent / "index_fixed"
            chunking_ablation.run(res, index, fixed_dir, args.out_dir)
        if "threshold" in steps:
            tune_threshold.run(res, index, args.out_dir, args.config if args.write_config else None, args.chosen)
        res.embeddings.save()
        record = build_record(res.settings, mock=args.mock, steps=steps)
        if args.out_dir:
            written = [RESULT_FILES[s] for s in steps if s in RESULT_FILES and (args.out_dir / RESULT_FILES[s]).exists()]
            print(f"run record: {write_record(record, args.out_dir, written)}")
        else:
            print(record.model_dump_json(indent=2))
    except (IndexNotBuilt, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not args.mock:
        e = res.embeddings
        print(f"embedding usage this run: {e.api_calls} API calls, {e.tokens} tokens, ${e.cost_usd:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
