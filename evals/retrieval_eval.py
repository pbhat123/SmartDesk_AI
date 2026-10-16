"""Recall@1/3/5 for BM25, dense, hybrid (RRF) and hybrid + rerank on the KB set (FR-A7)."""
import json
from collections.abc import Callable
from pathlib import Path

from evals.harness import Resources, kb_rows
from evals.schemas import KBQuestion
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.rag.fusion import rrf
from smartdesk_ai.rag.index import KBIndex
from smartdesk_ai.rag.models import Chunk
from smartdesk_ai.rag.rerank import Reranker
from smartdesk_ai.rag.retrievers import bm25_search, dense_search

STRATEGIES = ["bm25", "dense", "hybrid", "hybrid+rerank"]  # the four FR-A7 strategies
# Variants tried after the first real run: rerank only the dense top-k; weight dense 2x / 3x in RRF.
EXTRA_STRATEGIES = ["dense+rerank", "hybrid 2:1", "hybrid 3:1"]
ALL_STRATEGIES = STRATEGIES + EXTRA_STRATEGIES
DENSE_WEIGHTS = {"hybrid 2:1": 2.0, "hybrid 3:1": 3.0}  # dense weight; BM25 weight is 1
KS = (1, 3, 5)


def load_rows(split: str | None = None) -> list[KBQuestion]:
    return kb_rows(split)


def rank_results(
    strategy: str, index: KBIndex, query: str, cfg: Settings, reranker: Reranker | None, *, collapse: bool = True
) -> list[Chunk]:
    """The ranked chunks one strategy returns for a query. `collapse` keeps only the best piece per entry."""
    k = cfg.retrieval.k_each
    if strategy == "bm25":
        ranked = bm25_search(index, query, k)
    elif strategy == "dense":
        ranked = dense_search(index, query, k)
    elif strategy == "dense+rerank":
        if reranker is None:
            raise ValueError("dense+rerank needs a reranker")
        ranked = reranker.rerank(query, dense_search(index, query, k))
    elif strategy in DENSE_WEIGHTS:
        ranked = rrf(
            [bm25_search(index, query, k), dense_search(index, query, k)],
            k=cfg.retrieval.rrf_k, weights=[1.0, DENSE_WEIGHTS[strategy]],
        )
    elif strategy in ("hybrid", "hybrid+rerank"):
        fused = rrf([bm25_search(index, query, k), dense_search(index, query, k)], k=cfg.retrieval.rrf_k)
        if strategy == "hybrid":
            ranked = fused
        elif reranker is None:
            raise ValueError("hybrid+rerank needs a reranker")
        else:
            ranked = reranker.rerank(query, fused)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    chunks = [s.chunk for s in ranked]
    if not collapse:
        return chunks
    seen: set[str] = set()
    return [c for c in chunks if not (c.entry_id in seen or seen.add(c.entry_id))]


def recall_at_k(
    index: KBIndex,
    rows: list[KBQuestion],
    cfg: Settings,
    reranker: Reranker | None,
    strategies: list[str] = STRATEGIES,
    *,
    ks: tuple[int, ...] = KS,
    split: str | None = None,
    collapse: bool = True,
    attribution: Callable[[Chunk], set[str]] = lambda c: {c.entry_id},
) -> dict[str, dict[int, float]]:
    """Share of questions whose expected entry is among the top-k results, per strategy and k.
    `attribution` says which entries a result counts for (one by default; several for naive chunks)."""
    rows = [r for r in rows if split is None or r.split == split]
    out: dict[str, dict[int, float]] = {}
    for strategy in strategies:
        hits = dict.fromkeys(ks, 0)
        for r in rows:
            ranked = rank_results(strategy, index, r.question, cfg, reranker, collapse=collapse)
            for k in ks:
                if any(r.expected_entry_id in attribution(c) for c in ranked[:k]):
                    hits[k] += 1
        out[strategy] = {k: hits[k] / len(rows) for k in ks}
    return out


def format_table(results: dict[str, dict[int, float]], *, title: str, n: int) -> str:
    ks = sorted(next(iter(results.values())))
    lines = [f"{title} ({n} questions)", f"{'strategy':<16}" + "".join(f"R@{k:<5}" for k in ks)]
    lines += [f"{name:<16}" + "".join(f"{by_k[k]:<7.2f}" for k in ks) for name, by_k in results.items()]
    return "\n".join(lines)


def run(res: Resources, index: KBIndex, out_dir: Path | None) -> None:
    cfg, rows = res.settings, load_rows()
    payload = {"mock": res.mock, "embedder": res.embeddings.name, "n": len(rows), "ks": list(KS)}
    print(f"[{res.banner}]")
    for label, split in (("overall", None), ("dev", "dev"), ("test", "test")):
        subset = [r for r in rows if split is None or r.split == split]
        table = recall_at_k(index, rows, cfg, res.reranker, ALL_STRATEGIES, split=split)
        payload[label] = {s: {str(k): v for k, v in by_k.items()} for s, by_k in table.items()}
        print(format_table(table, title=f"Recall@k, {label}", n=len(subset)) + "\n")
    print("hybrid 2:1 and 3:1 = RRF with the dense list weighted 2x and 3x; dense+rerank reranks only the dense top-k.\n")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "retrieval.json").write_text(json.dumps(payload, indent=2))
