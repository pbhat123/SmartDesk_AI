"""Evidence for `thresholds.semantic_cache_cutoff` (SPEC section 4 cache rules), from cached embeddings only: no API call.

Paraphrase recall comes from `paraphrase_pairs.jsonl`, 15 held-out pairs of the same question in different words. Those
vectors are embedded once with `--embed-new` (a few hundred tokens, recorded in the spend ledger) and cached; without that
flag a missing vector is an error, never a call.

Two questions that need different answers must never be a semantic hit, so the cut-off has to sit above the highest cosine
between any such pair. Different answers = different expected KB entry, or a gap question against anything outside its
topic. Same-topic gap pairs are the nearest thing to paraphrases in the cached data. A question not in the embedding
cache is skipped and counted, never embedded. `python -m evals.cache_cutoff [--out-dir DIR]`.
"""
import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evals.common import EVALS_DIR, load_jsonl
from evals.harness import EMBEDDING_CACHE
from smartdesk_ai.config.settings import load_settings


def _vectors(model: str, path: Path = EMBEDDING_CACHE):
    store = json.loads(path.read_text()).get(model, {})

    def vec(question: str):
        raw = store.get(hashlib.sha256(question.encode()).hexdigest())
        return None if raw is None else np.array(raw) / np.linalg.norm(raw)

    return vec


SAFETY_MARGIN = 0.10  # the cut-off stays at least this far above the highest different-answer pair


def paraphrase_cosines(vec, pairs) -> list[float]:
    return sorted(float(vec(p["a"]) @ vec(p["b"])) for p in pairs)


def recommend(highest_different: float, paraphrases: list[float]) -> dict:
    """The lowest allowed cut-off is the highest different-answer pair plus the margin, rounded up to 0.01. Hits only fall
    as the cut-off rises, so that floor catches the most paraphrases; among cut-offs that lose none of those, take the
    highest (the safest)."""
    floor = int((highest_different + SAFETY_MARGIN) * 100 + 0.999999) / 100
    hits = lambda c: sum(x >= c for x in paraphrases)  # noqa: E731
    best = hits(floor)
    cutoff = max(c / 100 for c in range(int(floor * 100), 100) if hits(c / 100) == best)
    return {"floor": floor, "recommended": cutoff, "paraphrase_hits_at_recommended": best,
            "paraphrase_hits_at_0.95": hits(0.95), "n_paraphrases": len(paraphrases),
            "hit_rate_by_cutoff": {f"{c / 100:.2f}": hits(c / 100) for c in range(70, 100, 5)}}


def analyse(model: str) -> dict:
    vec = _vectors(model)
    kb, lex, gap = (load_jsonl(EVALS_DIR / f) for f in ("kb_questions.jsonl", "lexical_set.jsonl", "gap_questions.jsonl"))
    skipped = 0
    answered = []
    for row in kb + lex:
        v = vec(row["question"])
        skipped += v is None
        if v is not None:
            answered.append((row["id"], row["question"], row["expected_entry_id"], v))
    gaps = [(r["id"], r["question"], r["gap_topic"], vec(r["question"])) for r in gap]
    skipped += sum(g[3] is None for g in gaps)
    gaps = [g for g in gaps if g[3] is not None]
    different = [(float(a[3] @ b[3]), a[0], b[0]) for a, b in itertools.combinations(answered, 2) if a[2] != b[2]]
    different += [(float(a[3] @ b[3]), a[0], b[0]) for a, b in itertools.combinations(gaps, 2) if a[2] != b[2]]
    different += [(float(g[3] @ a[3]), g[0], a[0]) for g in gaps for a in answered]
    same_entry = [float(a[3] @ b[3]) for a, b in itertools.combinations(answered, 2) if a[2] == b[2]]
    topics = defaultdict(list)
    for g in gaps:
        topics[g[2]].append(g[3])
    same_topic = [float(a @ b) for vs in topics.values() for a, b in itertools.combinations(vs, 2)]
    different.sort(reverse=True)
    stats = lambda xs: {"n": len(xs), "min": min(xs), "median": float(np.median(xs)), "max": max(xs)}  # noqa: E731
    return {"model": model, "skipped_not_cached": skipped, "different_answer_pairs": len(different),
            "highest_different_answer": [{"cosine": round(s, 4), "a": a, "b": b} for s, a, b in different[:5]],
            "same_entry_pairs": stats(same_entry), "same_topic_gap_pairs": stats(same_topic)}


def paraphrase_vectors(settings, embed_new: bool):
    """A vector lookup over the embedding cache; with `embed_new`, uncached texts are embedded and cached (paid)."""
    from evals.embedding_cache import CachedEmbeddings
    from evals.retrieval_report import OfflineEmbeddings

    if embed_new:
        from smartdesk_ai.rag.embeddings import get_embeddings

        inner, price = get_embeddings(settings), settings.models.embedding.price.input
    else:
        inner, price = OfflineEmbeddings(settings.models.embedding.id), 0.0
    cached = CachedEmbeddings(inner, EMBEDDING_CACHE, price_per_million=price)

    def vec(text: str):
        v = np.array(cached.embed_query(text))
        return v / np.linalg.norm(v)

    return vec, cached


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-new", action="store_true", help="embed the paraphrase pairs not cached yet (paid, tiny)")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    settings = load_settings()
    result = analyse(settings.models.embedding.id)
    pairs = load_jsonl(EVALS_DIR / "paraphrase_pairs.jsonl")
    if args.embed_new:
        from smartdesk_ai.ops.spend_ledger import end_run, start_run

        start_run(settings, ["python", "-m", "evals.cache_cutoff", "--embed-new"], "eval")
    vec, cached = paraphrase_vectors(settings, args.embed_new)
    cosines = paraphrase_cosines(vec, pairs)
    if args.embed_new:
        cached.save()
        end_run("ok")
        result["embedding_api_calls"], result["embedding_cost_usd"] = cached.api_calls, round(cached.cost_usd, 8)
    result["paraphrase_pairs"] = {"n": len(cosines), "min": cosines[0], "median": float(np.median(cosines)),
                                  "max": cosines[-1], "cosines": [round(c, 4) for c in cosines]}
    result["recommendation"] = recommend(result["highest_different_answer"][0]["cosine"], cosines)
    result["configured_cutoff"] = settings.thresholds.semantic_cache_cutoff
    print(json.dumps(result, indent=2))
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "cache_cutoff.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
