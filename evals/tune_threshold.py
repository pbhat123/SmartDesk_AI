"""Tune the escalation threshold on the dev half and report the held-out test half (FR-A5, FR-A15).

Each KB question should be answered and each gap question escalated. A question is answered when its
top dense cosine similarity is at or above `thresholds.escalation` (SPEC v3.1). Scores come from the
production path (`retrieve`). The threshold is chosen on the dev half only, by balanced accuracy; the
test half is only ever scored at that value. The reranker's top score is measured too, as a reference
that shows why it is not used for the decision.
"""
import json
import logging
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from evals.harness import Resources, gap_rows, kb_rows
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.rag.index import KBIndex
from smartdesk_ai.rag.pipeline import retrieve

LOW, HIGH = 0.01, 0.99  # thresholds must stay inside the open unit interval the config requires
CURVE_GRID = [round(0.35 + 0.02 * i, 2) for i in range(11)]  # 0.35 .. 0.55 for the dense-cosine curve
MIN_KB_ANSWERED = 0.95


@dataclass(frozen=True)
class Point:
    id: str
    label: str  # "kb" (should be answered) or "gap" (should be escalated)
    split: str  # "dev" or "test"
    score: float


@dataclass(frozen=True)
class Metrics:
    threshold: float
    n_kb: int
    n_gap: int
    answered_kb: float  # share of KB questions answered
    escalated_gap: float  # share of gap questions escalated (escalation recall)
    balanced_accuracy: float
    escalation_precision: float  # of all escalations, the share that were gap questions

    @property
    def escalation_recall(self) -> float:
        """Share of gap questions that were escalated (the spec's escalation recall)."""
        return self.escalated_gap


def metrics(points: list[Point], threshold: float) -> Metrics:
    kb = [p.score for p in points if p.label == "kb"]
    gap = [p.score for p in points if p.label == "gap"]
    answered = sum(s >= threshold for s in kb) / len(kb) if kb else 0.0
    escalated = sum(s < threshold for s in gap) / len(gap) if gap else 0.0
    escalations = sum(s < threshold for s in kb + gap)
    precision = sum(s < threshold for s in gap) / escalations if escalations else 0.0
    return Metrics(threshold, len(kb), len(gap), answered, escalated, (answered + escalated) / 2, precision)


def auc(points: list[Point]) -> float:
    """Chance that a random KB question scores above a random gap question (ties count half)."""
    kb = [p.score for p in points if p.label == "kb"]
    gap = [p.score for p in points if p.label == "gap"]
    if not kb or not gap:
        return 0.5
    wins = sum(1.0 if k > g else 0.5 if k == g else 0.0 for k in kb for g in gap)
    return wins / (len(kb) * len(gap))


def _clip(x: float) -> float:
    return min(HIGH, max(LOW, round(x, 2)))


def choose_threshold(dev: list[Point]) -> float:
    """The threshold with the best balanced accuracy on `dev`; ties go to the middle of the best ones."""
    scores = sorted({p.score for p in dev})
    cuts = [(a + b) / 2 for a, b in zip(scores, scores[1:])] + [scores[0] - 0.01, scores[-1] + 0.01]
    candidates = sorted({_clip(c) for c in cuts})
    best = max(metrics(dev, c).balanced_accuracy for c in candidates)
    top = [c for c in candidates if metrics(dev, c).balanced_accuracy == best]
    return statistics.median_low(top)


def curve(points: list[Point], thresholds: list[float]) -> list[Metrics]:
    return [metrics(points, t) for t in thresholds]


def highest_meeting(points: list[Point], thresholds: list[float], min_answered: float) -> float | None:
    """The highest threshold on the grid that still answers at least `min_answered` of the KB questions."""
    ok = [t for t in thresholds if metrics(points, t).answered_kb >= min_answered]
    return max(ok) if ok else None


def strict_threshold(dev: list[Point]) -> float:
    """The lowest sensible threshold that answers no gap question on `dev`. It costs KB recall."""
    gaps = [p.score for p in dev if p.label == "gap"]
    if not gaps:
        return choose_threshold(dev)
    top_gap = max(gaps)
    above = sorted(s for s in {p.score for p in dev} if s > top_gap)
    t = _clip((top_gap + above[0]) / 2) if above else _clip(top_gap + 0.01)
    return _clip(max(t, top_gap + 0.01)) if t <= top_gap else t


def tune(points: list[Point]) -> dict:
    dev = [p for p in points if p.split == "dev"]
    test = [p for p in points if p.split == "test"]
    t, strict = choose_threshold(dev), strict_threshold(dev)
    return {
        "threshold": t,
        "strict_threshold": strict,
        "dev": metrics(dev, t),
        "test": metrics(test, t),
        "dev_strict": metrics(dev, strict),
        "test_strict": metrics(test, strict),
        "dev_auc": auc(dev),
        "test_auc": auc(test),
        "points": points,
    }


def _points(res: Resources, index: KBIndex, kind: str) -> list[Point]:
    """Top score per question: the dense cosine (the decision signal) or the top reranker score (reference)."""
    health = HealthRegistry()  # keep eval runs out of the app's shared degraded-mode registry
    health_log = logging.getLogger("smartdesk_ai.ops.health")
    previous = health_log.level
    reranker = res.reranker if kind == "reranker" else None
    if reranker is None:  # the dense pass runs without the reranker on purpose; its warning is noise
        health_log.setLevel(logging.ERROR)
    out = []
    for label, rows in (("kb", kb_rows()), ("gap", gap_rows())):
        for r in rows:
            result = retrieve(index, r.question, cfg=res.settings, reranker=reranker, health=health)
            score = result.top_cosine if kind == "dense_cosine" else (result.entries[0].score if result.entries else 0.0)
            out.append(Point(r.id, label, r.split, score))
    health_log.setLevel(previous)
    return out


def _with_curve(section: dict) -> dict:
    pts = section["points"]
    dev, test = [p for p in pts if p.split == "dev"], [p for p in pts if p.split == "test"]
    section["curve"] = [{"threshold": t, "dev": metrics(dev, t), "test": metrics(test, t)} for t in CURVE_GRID]
    best = highest_meeting(dev, CURVE_GRID, MIN_KB_ANSWERED)
    section["at_95_percent"] = None if best is None else {
        "threshold": best, "dev": metrics(dev, best), "test": metrics(test, best),
    }
    return section


def run_tuning(res: Resources, index: KBIndex) -> dict:
    return {
        "dense_cosine": _with_curve(tune(_points(res, index, "dense_cosine"))),
        "reranker": tune(_points(res, index, "reranker")),  # reference only (SPEC v3.1)
    }


def write_config(path: Path, *, escalation: float) -> None:
    """Set `thresholds.escalation` in config.yaml, keeping the rest of its line and every other line."""
    if not 0.0 < escalation < 1.0:
        raise ValueError(f"escalation threshold {escalation} is outside (0, 1)")
    text = Path(path).read_text()
    pattern = re.compile(r"^(?P<head>\s*escalation:\s*)[0-9.]+", re.MULTILINE)
    if len(pattern.findall(text)) != 1:
        raise ValueError(f"expected exactly one 'escalation' threshold line in {path}")
    Path(path).write_text(pattern.sub(lambda m: f"{m['head']}{escalation:.2f}", text))


def _line(label: str, m: Metrics, auc_value: float) -> str:
    return (f"  {label:<22}{m.answered_kb:<14.2f}{m.escalated_gap:<15.2f}{m.balanced_accuracy:<14.2f}"
            f"{m.escalation_precision:<17.2f}{auc_value:.2f}")


def _frac(rate: float, n: int) -> str:
    return f"{round(rate * n)}/{n} ({rate:.2f})"


def format_report(out: dict) -> str:
    dense, ref = out["dense_cosine"], out["reranker"]
    d, t = dense["dev"], dense["test"]
    lines = [f"Threshold tuning: dev = {d.n_kb} KB + {d.n_gap} gap questions, test = {t.n_kb} KB + {t.n_gap} gap questions",
             "A question is answered when its top dense cosine is at or above the threshold (thresholds.escalation).", "",
             f"  {'':<24}{'KB answered':<16}{'gap escalated':<17}{'balanced acc':<14}{'esc. precision':<16}AUC"]

    def line(label, m, auc_value):
        return (f"  {label:<24}{_frac(m.answered_kb, m.n_kb):<16}{_frac(m.escalated_gap, m.n_gap):<17}"
                f"{m.balanced_accuracy:<14.2f}{m.escalation_precision:<16.2f}{auc_value:.2f}")

    lines += [line(f"dev @ {dense['threshold']:.2f}", d, dense["dev_auc"]),
              line(f"test @ {dense['threshold']:.2f}", t, dense["test_auc"]),
              line(f"dev @ {dense['strict_threshold']:.2f} (strict)", dense["dev_strict"], dense["dev_auc"]),
              line(f"test @ {dense['strict_threshold']:.2f} (strict)", dense["test_strict"], dense["test_auc"])]
    if "curve" in dense:
        chosen = (dense.get("at_95_percent") or {}).get("threshold")
        lines += ["", f"Dense-cosine curve on dev ({d.n_kb} KB, {d.n_gap} gap questions)",
                  f"  {'threshold':<11}{'KB answered':<18}{'gap escalated':<18}"]
        for row in dense["curve"]:
            m = row["dev"]
            mark = "  <- highest with >= 95% KB answered" if row["threshold"] == chosen else ""
            lines.append(f"  {row['threshold']:<11.2f}{_frac(m.answered_kb, m.n_kb):<18}{_frac(m.escalated_gap, m.n_gap):<18}{mark}")
        if chosen is not None:
            tm = dense["at_95_percent"]["test"]
            lines.append(f"  test half at {chosen:.2f}: KB answered {_frac(tm.answered_kb, tm.n_kb)}, "
                         f"gap escalated {_frac(tm.escalated_gap, tm.n_gap)}, AUC {dense['test_auc']:.2f}")
        else:
            lines.append("  no grid value answers 95% of the dev KB questions")
    if "chosen" in dense:
        c = dense["chosen"]
        lines += ["", f"Chosen threshold {c['threshold']:.2f} (recorded):",
                  line("dev", c["dev"], dense["dev_auc"]), line("test", c["test"], dense["test_auc"])]
    lines += ["", f"Reference, not used for the decision: top reranker score, dev AUC {ref['dev_auc']:.2f}, test AUC {ref['test_auc']:.2f} "
              f"(dense cosine: {dense['dev_auc']:.2f}, {dense['test_auc']:.2f}).",
              "", f"Proposed (not applied): escalation = {dense['threshold']:.2f} (balanced), "
              f"{dense['strict_threshold']:.2f} (strict: answers no dev gap question)"]
    return "\n".join(lines)


def _jsonable(value):
    if isinstance(value, Metrics) or isinstance(value, Point):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def run(res: Resources, index: KBIndex, out_dir: Path | None, config_path: Path | None = None,
        chosen: float | None = None) -> dict:
    out = run_tuning(res, index)
    dense = out["dense_cosine"]
    pts = dense["points"]
    dev, test = [p for p in pts if p.split == "dev"], [p for p in pts if p.split == "test"]
    if chosen is not None:  # an explicit pick: record its dev and test numbers next to the proposal
        dense["chosen"] = {"threshold": chosen, "dev": metrics(dev, chosen), "test": metrics(test, chosen)}
    print(f"[{res.banner}]\n" + format_report(out) + "\n")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"mock": res.mock, "embedder": res.embeddings.name, **_jsonable(out)}
        (out_dir / "threshold.json").write_text(json.dumps(payload, indent=2))
    if config_path:
        value = chosen if chosen is not None else dense["threshold"]
        write_config(config_path, escalation=value)
        print(f"wrote thresholds.escalation = {value:.2f} to {config_path}")
    return out
