"""Tune `thresholds.duplicate_title` on near-identical and distinct title pairs (FR-F4).

The similarity is the agent's own `title_similarity` (a string ratio, no model call), so this is free and
offline. A pair is flagged as a duplicate when its similarity is at or above the threshold. Flagging a
different issue only costs the employee one extra yes/no question; missing a repeat creates a second
ticket. So two proposals are reported: the threshold with the best F1, and the highest threshold that still
catches `RECALL_TARGET` of the repeats. Nothing here edits config.yaml; a value goes in only with approval.
"""
import json
from pathlib import Path

from pydantic import BaseModel

from evals.common import EVALS_DIR, load_jsonl
from smartdesk_ai.agent.ticket_agent import title_similarity
from smartdesk_ai.config.settings import Settings

RECALL_TARGET = 0.90
GRID = [round(0.40 + 0.01 * i, 2) for i in range(56)]  # 0.40 .. 0.95


class TitlePair(BaseModel):
    id: str
    a: str
    b: str
    same_issue: bool


class Point(BaseModel):
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positives: int
    false_negatives: int


def load_pairs() -> list[TitlePair]:
    return [TitlePair(**r) for r in load_jsonl(EVALS_DIR / "title_pairs.jsonl")]


def score(pairs: list[TitlePair], threshold: float) -> Point:
    flagged = [(title_similarity(p.a, p.b) >= threshold, p.same_issue) for p in pairs]
    tp = sum(f and s for f, s in flagged)
    fp = sum(f and not s for f, s in flagged)
    fn = sum(not f and s for f, s in flagged)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Point(threshold=threshold, precision=precision, recall=recall, f1=f1, false_positives=fp, false_negatives=fn)


def sweep(pairs: list[TitlePair], grid: list[float] = GRID) -> list[Point]:
    return [score(pairs, t) for t in grid]


def best_f1(points: list[Point]) -> Point:
    """The best F1; among equals the lowest threshold, which misses fewer repeats."""
    return max(points, key=lambda p: (p.f1, -p.threshold))


def best_for_recall(points: list[Point], target: float = RECALL_TARGET) -> Point | None:
    """The highest threshold (fewest false alarms) that still catches `target` of the repeats."""
    ok = [p for p in points if p.recall >= target]
    return max(ok, key=lambda p: p.threshold) if ok else None


def run(settings: Settings, out_dir: Path | None) -> dict:
    pairs = load_pairs()
    points = sweep(pairs)
    current = score(pairs, settings.thresholds.duplicate_title)
    f1_pick, recall_pick = best_f1(points), best_for_recall(points)
    result = {
        "n_same": sum(p.same_issue for p in pairs), "n_different": sum(not p.same_issue for p in pairs),
        "current": current.model_dump(), "best_f1": f1_pick.model_dump(),
        "best_for_recall": recall_pick.model_dump() if recall_pick else None, "curve": [p.model_dump() for p in points],
        "note": "proposal only: config.yaml is not changed",
    }
    print(f"duplicate_title: {result['n_same']} repeat pairs, {result['n_different']} distinct pairs")
    for label, p in (("current", current), ("best F1", f1_pick), (f"recall >= {RECALL_TARGET}", recall_pick)):
        if p is not None:
            print(f"  {label:<14} t={p.threshold:.2f}  precision {p.precision:.3f}  recall {p.recall:.3f}  "
                  f"F1 {p.f1:.3f}  false alarms {p.false_positives}  missed repeats {p.false_negatives}")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "duplicate_threshold.json").write_text(json.dumps(result, indent=2))
    return result
