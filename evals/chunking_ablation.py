"""Chunking ablation: heading-based chunks vs fixed 512-token windows with 50 overlap (FR-A17)."""
import json
import tempfile
from pathlib import Path

from evals.harness import Resources
from evals.retrieval_eval import KS, STRATEGIES, load_rows, recall_at_k
from smartdesk_ai.rag.chunking import MIN_SHARE, chunk_fixed, count_tokens
from smartdesk_ai.rag.index import KBIndex, open_index
from smartdesk_ai.rag.ingest import DEFAULT_KB_DIR, build_index, ingest, load_policy_entries
from smartdesk_ai.rag.models import Chunk

WINDOW, OVERLAP = 512, 50


def _stats(chunks: list[Chunk], attribution: dict[str, set[str]]) -> dict:
    return {
        "chunks": len(chunks),
        "avg_tokens": round(sum(count_tokens(c.text) for c in chunks) / len(chunks), 1),
        "avg_entries_per_chunk": round(sum(len(attribution[c.chunk_id]) for c in chunks) / len(chunks), 2),
    }


def run_ablation(res: Resources, fixed_dir: Path, heading_index: KBIndex | None = None) -> dict:
    cfg, rows = res.settings, load_rows()
    entries, _, _ = load_policy_entries(DEFAULT_KB_DIR)

    if heading_index is None:
        heading_dir = Path(tempfile.mkdtemp(prefix="abl_heading_"))
        ingest(DEFAULT_KB_DIR, heading_dir, res.embeddings, cfg.chunking)
        heading_index = open_index(heading_dir, res.embeddings)
    heading_attr = {c.chunk_id: {c.entry_id} for c in heading_index.chunks}

    fixed_chunks, fixed_attr = chunk_fixed(entries, window=WINDOW, overlap=OVERLAP)
    build_index(fixed_dir, fixed_chunks, entries, res.embeddings)
    fixed_index = open_index(fixed_dir, res.embeddings)

    return {
        "mock": res.mock,
        "embedder": res.embeddings.name,
        "n": len(rows),
        "ks": list(KS),
        "fixed": {"window": WINDOW, "overlap": OVERLAP},
        "attribution_min_share": MIN_SHARE,
        "stats": {
            "heading": _stats(heading_index.chunks, heading_attr),
            "fixed512": _stats(fixed_chunks, fixed_attr),
        },
        "recall": {
            "heading": recall_at_k(heading_index, rows, cfg, res.reranker, STRATEGIES, attribution=lambda c: heading_attr[c.chunk_id]),
            "fixed512": recall_at_k(  # lenient: a window counts for every entry it substantially covers
                fixed_index, rows, cfg, res.reranker, STRATEGIES,
                collapse=False, attribution=lambda c: fixed_attr[c.chunk_id],
            ),
            "fixed512_primary": recall_at_k(  # strict: a window counts only for the entry covering most of it
                fixed_index, rows, cfg, res.reranker, STRATEGIES,
                collapse=False, attribution=lambda c: {c.entry_id},
            ),
        },
    }


LABELS = {"heading": "heading", "fixed512": "fixed512 (lenient)", "fixed512_primary": "fixed512 (strict)"}


def format_comparison(result: dict) -> str:
    ks = result["ks"]
    lines = [
        f"Chunking ablation, Recall@k ({result['n']} questions)",
        f"{'strategy':<16}{'chunking':<20}" + "".join(f"R@{k:<6}" for k in ks),
    ]
    for s in STRATEGIES:
        for key, label in LABELS.items():
            by_k = result["recall"][key][s]
            lines.append(f"{s if key == 'heading' else '':<16}{label:<20}" + "".join(f"{by_k[k]:<8.2f}" for k in ks))
    for name, st in result["stats"].items():
        lines.append(f"{name}: {st['chunks']} chunks, avg {st['avg_tokens']} tokens, {st['avg_entries_per_chunk']} entries per chunk")
    lines.append(
        f"fixed512 = {result['fixed']['window']}-token windows, {result['fixed']['overlap']} overlap, each window its own result. "
        f"Lenient: a window counts for every entry covering at least {int(result['attribution_min_share'] * 100)}% of it. "
        "Strict: only for the entry covering most of it."
    )
    return "\n".join(lines)


def run(res: Resources, heading_index: KBIndex, fixed_dir: Path, out_dir: Path | None) -> None:
    result = run_ablation(res, fixed_dir, heading_index)
    print(f"[{res.banner}]\n" + format_comparison(result) + "\n")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "chunking.json").write_text(json.dumps(result, indent=2, default=str))
