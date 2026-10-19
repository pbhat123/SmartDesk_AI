"""Evidence for `thresholds.semantic_cache_cutoff` (SPEC section 4 cache rules), from cached embeddings only: no API call.

Paraphrase recall comes from `paraphrase_pairs.jsonl`, 15 held-out pairs of the same question in different words. Those
vectors are embedded once with `--embed-new` (a few hundred tokens, recorded in the spend ledger) and cached; without that
flag a missing vector is an error, never a call.

After Phase 7 the cache key is the orchestrator's sub-question (T6.10), so the same measurement is repeated with two more
sets: the author's standalone wording of each answerable compound part against the orchestrator's own wording of it in the live
compound run (`evals/results/compound_eval_live.json`; counted but not paired when the wording is identical), and
`near_miss_pairs.jsonl`, 16 pairs on one topic with different answers (sick against casual leave and the like), which join the
different-answer evidence. `--embed-new` embeds the new texts (about a thousand tokens), needs `--confirm-spend`, runs under the live
guard and ends with a verdict on `thresholds.semantic_cache_cutoff`.

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
from evals.harness import EMBEDDING_CACHE, RESULTS_DIR
from evals.live_guard import Estimate, LiveRun, describe
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


def near_miss_cosines(vec, pairs) -> list[dict]:
    """The cosine of each near-miss pair (same topic, different answers), in file order."""
    return [{"id": p["id"], "cosine": round(float(vec(p["a"]) @ vec(p["b"])), 4)} for p in pairs]


def rewrite_pairs(rows, live: dict) -> tuple[list[dict], int]:
    """The author's wording of each answerable compound part paired with the orchestrator's wording of it in the live run.
    A row whose part count changed is skipped; a part the orchestrator left word for word is counted, not paired."""
    texts = {r["id"]: r["run"].get("part_texts", []) for r in live.get("rows", [])}
    pairs, identical = [], 0
    for row in rows:
        got = texts.get(row.id)
        if not got or len(got) != len(row.parts):
            continue
        for n, (part, text) in enumerate(zip(row.parts, got), 1):
            if not part.answerable:
                continue
            if text == part.sub_question:
                identical += 1
            else:
                pairs.append({"id": f"{row.id}-p{n}", "a": part.sub_question, "b": text})
    return pairs, identical


def verdict(*, configured: float, highest_different: float, paraphrases: list[float]) -> dict:
    """What to do with the cut-off: raise it if it is not 0.10 above the highest different-answer pair; lower it only if that
    catches more paraphrases while staying 0.10 above; otherwise keep it."""
    rec = recommend(highest_different, paraphrases)
    hits = lambda c: sum(x >= c for x in paraphrases)  # noqa: E731
    out = {"configured": configured, "highest_different_answer": round(highest_different, 4), "floor": rec["floor"],
           "recommended": rec["recommended"], "paraphrases": len(paraphrases),
           "hits_at_configured": hits(configured), "hits_at_recommended": rec["paraphrase_hits_at_recommended"]}
    if configured < rec["floor"]:
        out["action"] = f"raise to at least {rec['floor']:.2f}"
    elif rec["paraphrase_hits_at_recommended"] > hits(configured) and rec["recommended"] < configured:
        out["action"] = f"re-tune to {rec['recommended']:.2f}"
    else:
        out["action"] = "keep"
    return out


def analyse(model: str, near_miss: list[dict] | None = None) -> dict:
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
    near = [(float(vec(p["a"]) @ vec(p["b"])), p["id"], "near-miss") for p in (near_miss or [])
            if vec(p["a"]) is not None and vec(p["b"]) is not None]
    different += near
    different.sort(reverse=True)
    stats = lambda xs: {"n": len(xs), "min": min(xs), "median": float(np.median(xs)), "max": max(xs)}  # noqa: E731
    return {"model": model, "skipped_not_cached": skipped, "different_answer_pairs": len(different),
            "highest_different_answer": [{"cosine": round(s, 4), "a": a, "b": b} for s, a, b in different[:5]],
            "near_miss_pairs": stats([c for c, *_ in near]) if near else None,
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


MIN_ESTIMATE_USD = 0.01  # the approved figure for the re-measure; the real cost is a fraction of a cent


def new_texts(settings, pairs: list[dict], near: list[dict], rewrites: list[dict]) -> list[str]:
    return sorted({t for p in [*pairs, *near, *rewrites] for t in (p["a"], p["b"])})


def estimate_embedding(settings, texts: list[str] | None = None) -> Estimate:
    """Embedding the texts at the config price, counted with tiktoken. Rounded up to the approved $0.01, so the ceiling (twice
    that) is not tripped by rounding on a cost of a few thousandths of a cent."""
    from smartdesk_ai.rag.chunking import count_tokens

    if texts is None:
        texts = new_texts(settings, load_jsonl(EVALS_DIR / "paraphrase_pairs.jsonl"),
                          load_jsonl(EVALS_DIR / "near_miss_pairs.jsonl"), [])
        texts += ["x" * 120] * 60  # room for the compound rewrites, which exist only after the live compound run
    tokens = sum(count_tokens(t) for t in texts)
    usd = tokens * settings.models.embedding.price.input / 1_000_000
    return Estimate("the paraphrase re-measure", max(round(usd, 6), MIN_ESTIMATE_USD),
                    f"embeds up to {len(texts)} texts, about {tokens:,} tokens, one call each; computed ${usd:.6f}")


def _load_live_rows() -> dict:
    path = RESULTS_DIR / "compound_eval_live.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-new", action="store_true", help="embed the texts not cached yet (paid, tiny; needs --confirm-spend)")
    parser.add_argument("--estimate", action="store_true", help="print the cost of --embed-new and stop")
    parser.add_argument("--confirm-spend", action="store_true", help="allow --embed-new")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    settings = load_settings()
    pairs = load_jsonl(EVALS_DIR / "paraphrase_pairs.jsonl")
    near = load_jsonl(EVALS_DIR / "near_miss_pairs.jsonl")
    from evals.compound_eval import load_rows

    rewrites, identical = rewrite_pairs(load_rows(), _load_live_rows())
    texts = new_texts(settings, pairs, near, rewrites)
    if args.embed_new or args.estimate:
        estimate = estimate_embedding(settings, texts)
        print("\n".join(describe(estimate)))
        if args.estimate:
            return 0
        if not args.confirm_spend:
            print("Nothing was run. Add --confirm-spend once you have approved the cost.")
            return 2
        with LiveRun(settings, ["python", "-m", "evals.cache_cutoff", *(argv if argv is not None else [])],
                     ceiling_usd=estimate.ceiling_usd) as run:
            vec, cached = paraphrase_vectors(settings, True)
            for text in texts:
                vec(text)
            cached.save()
            run.stop_reason()
            cost = run.cost()
        print(f"ledger cost of this run: ${cost.usd:.6f} ({cached.api_calls} embedding calls, {cached.tokens:,} tokens)")
    else:
        vec, cached = paraphrase_vectors(settings, False)
    have = lambda p: _has(vec, p)  # noqa: E731
    near_ok, rewrites_ok = [p for p in near if have(p)], [p for p in rewrites if have(p)]
    result = analyse(settings.models.embedding.id, near_ok)
    cosines = paraphrase_cosines(vec, pairs)
    if args.embed_new:
        result["embedding_api_calls"], result["embedding_cost_usd"] = cached.api_calls, round(cached.cost_usd, 8)
    result["paraphrase_pairs"] = {"n": len(cosines), "min": cosines[0], "median": float(np.median(cosines)),
                                  "max": cosines[-1], "cosines": [round(c, 4) for c in cosines]}
    rewrite_cos = paraphrase_cosines(vec, rewrites_ok)
    result["rewrite_pairs"] = {"n": len(rewrite_cos), "identical_not_paired": identical,
                               **({"min": rewrite_cos[0], "median": float(np.median(rewrite_cos)), "max": rewrite_cos[-1]} if rewrite_cos else {}),
                               "cosines": {p["id"]: round(float(vec(p["a"]) @ vec(p["b"])), 4) for p in rewrites_ok}}
    result["near_miss_cosines"] = near_miss_cosines(vec, near_ok)
    highest = result["highest_different_answer"][0]["cosine"]
    result["recommendation"] = recommend(highest, [*cosines, *rewrite_cos])
    result["configured_cutoff"] = settings.thresholds.semantic_cache_cutoff
    result["verdict"] = verdict(configured=settings.thresholds.semantic_cache_cutoff, highest_different=highest,
                                paraphrases=[*cosines, *rewrite_cos])
    print(json.dumps(result, indent=2))
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "cache_cutoff.json").write_text(json.dumps(result, indent=2))
    return 0


def _has(vec, pair: dict) -> bool:
    """Both texts of a pair are in the embedding cache (offline, a missing text raises rather than calling the API)."""
    try:
        return vec(pair["a"]) is not None and vec(pair["b"]) is not None
    except RuntimeError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
