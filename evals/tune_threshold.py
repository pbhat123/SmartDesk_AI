"""Tune the escalation thresholds on the dev half and report the held-out test half (FR-A5, FR-A15).

Each KB question should be answered and each gap question escalated. A question is answered when its
top retrieval score is at or above the threshold. Scores come from the production path (`retrieve`):
normalised reranker scores for `thresholds.escalation`, top dense cosine similarity (reranker off) for
`thresholds.fallback`. The threshold is chosen on the dev half only, by balanced accuracy; the test
half is only ever scored at that value.
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


def _points(res: Resources, index: KBIndex, reranker) -> list[Point]:
    health = HealthRegistry()  # keep eval runs out of the app's shared degraded-mode registry
    health_log = logging.getLogger("smartdesk_ai.ops.health")
    previous = health_log.level
    if reranker is None:  # the dense-only pass is deliberate, so its "reranker degraded" warning is noise
        health_log.setLevel(logging.ERROR)
    out = []
    for label, rows in (("kb", kb_rows()), ("gap", gap_rows())):
        for r in rows:
            result = retrieve(index, r.question, cfg=res.settings, reranker=reranker, health=health)
            out.append(Point(r.id, label, r.split, result.top_score))
    health_log.setLevel(previous)
    return out


def run_tuning(res: Resources, index: KBIndex) -> dict:
    return {
        "reranker": tune(_points(res, index, res.reranker)),
        "dense_cosine": tune(_points(res, index, None)),
    }


def write_config(path: Path, *, escalation: float, fallback: float) -> None:
    """Replace the two threshold values in config.yaml, keeping every other line as it is."""
    for name, value in (("escalation", escalation), ("fallback", fallback)):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} threshold {value} is outside (0, 1)")
    text = Path(path).read_text()
    for name, value in (("escalation", escalation), ("fallback", fallback)):
        pattern = re.compile(
            rf"^(?P<head>\s*{name}:\s*)[0-9.]+(?P<pad>\s*)# (?:provisional, )?tuned in T2\.12(?: on the dev half)? (?P<rest>.*)$",
            re.MULTILINE,
        )
        if len(pattern.findall(text)) != 1:
            raise ValueError(f"expected exactly one '{name}' threshold line in {path}")
        text = pattern.sub(lambda m: f"{m['head']}{value:.2f}{m['pad']}# tuned in T2.12 on the dev half {m['rest']}", text)
    Path(path).write_text(text)


def _line(label: str, m: Metrics, auc_value: float) -> str:
    return (f"  {label:<22}{m.answered_kb:<14.2f}{m.escalated_gap:<15.2f}{m.balanced_accuracy:<14.2f}"
            f"{m.escalation_precision:<17.2f}{auc_value:.2f}")


def format_report(out: dict) -> str:
    first = out["reranker"]
    d, t = first["dev"], first["test"]
    lines = [f"Threshold tuning: dev = {d.n_kb} KB + {d.n_gap} gap questions, test = {t.n_kb} KB + {t.n_gap} gap questions",
             "A question is answered when its top score is at or above the threshold."]
    for kind, key, scale in (("reranker", "escalation", "normalised reranker score"),
                             ("dense_cosine", "fallback", "top dense cosine, reranker unavailable")):
        s = out[kind]
        lines += ["", f"thresholds.{key} ({scale})",
                  f"  {'':<22}{'KB answered':<14}{'gap escalated':<15}{'balanced acc':<14}{'esc. precision':<17}AUC",
                  _line(f"dev @ {s['threshold']:.2f}", s["dev"], s["dev_auc"]),
                  _line(f"test @ {s['threshold']:.2f}", s["test"], s["test_auc"]),
                  _line(f"dev @ {s['strict_threshold']:.2f} (strict)", s["dev_strict"], s["dev_auc"]),
                  _line(f"test @ {s['strict_threshold']:.2f} (strict)", s["test_strict"], s["test_auc"])]
    lines += ["", "Proposed (not applied): "
              f"escalation = {out['reranker']['threshold']:.2f}, fallback = {out['dense_cosine']['threshold']:.2f}",
              "Strict alternative answers no dev gap question, at the cost of KB answers: "
              f"escalation = {out['reranker']['strict_threshold']:.2f}, fallback = {out['dense_cosine']['strict_threshold']:.2f}"]
    return "\n".join(lines)


def _jsonable(section: dict) -> dict:
    return {
        k: ([asdict(p) for p in v] if k == "points" else asdict(v) if isinstance(v, Metrics) else v)
        for k, v in section.items()
    }


def run(res: Resources, index: KBIndex, out_dir: Path | None, config_path: Path | None = None) -> dict:
    out = run_tuning(res, index)
    print(f"[{res.banner}]\n" + format_report(out) + "\n")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"mock": res.mock, "embedder": res.embeddings.name, **{k: _jsonable(v) for k, v in out.items()}}
        (out_dir / "threshold.json").write_text(json.dumps(payload, indent=2))
    if config_path:
        write_config(config_path, escalation=out["reranker"]["threshold"], fallback=out["dense_cosine"]["threshold"])
        print(f"wrote thresholds.escalation and thresholds.fallback to {config_path}")
    return out
