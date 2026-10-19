"""Evidence for `thresholds.semantic_cache_cutoff` (SPEC section 4 cache rules), from cached embeddings only: no API call.

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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    settings = load_settings()
    result = analyse(settings.models.embedding.id)
    result["configured_cutoff"] = settings.thresholds.semantic_cache_cutoff
    print(json.dumps(result, indent=2))
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "cache_cutoff.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
