"""Tune `thresholds.routing_confidence` on `evals/intents.jsonl` (FR-D8).

Below the threshold the router ignores the classifier and uses the model's tool pick instead. So the
threshold should sit where the classifier's confidence stops being trustworthy: raise it until the
classifications it still accepts are right at least `TARGET_ACCURACY` of the time, but no higher, since each
step up hands more messages to the fallback. Each row is classified once (one ORCHESTRATOR call), and the
sweep is computed from those results, so a sweep costs nothing extra. Nothing here edits config.yaml.
"""
import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from evals.common import EVALS_DIR, load_jsonl
from evals.schemas import IntentRow
from smartdesk_ai.agent.intents import ClassifierError, route_for
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.factory import ModelBuilder

TARGET_ACCURACY = 0.95
GRID = [round(0.30 + 0.05 * i, 2) for i in range(14)]  # 0.30 .. 0.95


class Observation(BaseModel):
    id: str
    confidence: float | None  # None: the classifier failed or answered nonsense
    correct: bool  # intent and route both as expected


class Point(BaseModel):
    threshold: float
    coverage: float  # share of messages the classifier decides alone
    accuracy_covered: float | None
    n_covered: int
    right_but_handed_off: int  # correct classifications the fallback now has to redo


def load_rows() -> list[IntentRow]:
    return [IntentRow(**r) for r in load_jsonl(EVALS_DIR / "intents.jsonl")]


def observe(rows: list[IntentRow], classify: Callable[..., object]) -> list[Observation]:
    out = []
    for row in rows:
        try:
            result = classify(row.utterance, row.awaiting)
        except ClassifierError:
            out.append(Observation(id=row.id, confidence=None, correct=False))
            continue
        ok = result.intent == row.expected_intent and route_for(result) == row.expected_route
        out.append(Observation(id=row.id, confidence=result.confidence, correct=ok))
    return out


def sweep(observations: list[Observation], grid: list[float] = GRID) -> list[Point]:
    points = []
    for t in grid:
        covered = [o for o in observations if o.confidence is not None and o.confidence >= t]
        handed = [o for o in observations if o.confidence is not None and o.confidence < t and o.correct]
        points.append(Point(
            threshold=t, coverage=len(covered) / len(observations) if observations else 0.0,
            accuracy_covered=(sum(o.correct for o in covered) / len(covered)) if covered else None,
            n_covered=len(covered), right_but_handed_off=len(handed)))
    return points


def propose(points: list[Point], target: float = TARGET_ACCURACY) -> Point | None:
    """The lowest threshold whose accepted classifications are right at least `target` of the time."""
    ok = [p for p in points if p.accuracy_covered is not None and p.accuracy_covered >= target]
    return min(ok, key=lambda p: p.threshold) if ok else None


def run(settings: Settings, out_dir: Path | None, *, builder: ModelBuilder, live: bool) -> dict:
    from smartdesk_ai.agent.intents import make_classifier

    rows = load_rows()
    observations = observe(rows, make_classifier(settings, builder=builder))
    points = sweep(observations)
    proposal = propose(points)
    current = next((p for p in sweep(observations, [settings.thresholds.routing_confidence])), None)
    accuracy = sum(o.correct for o in observations) / len(observations)
    result = {
        "live": live, "n": len(rows), "classifier_accuracy": accuracy,
        "failed_calls": sum(o.confidence is None for o in observations),
        "current": current.model_dump() if current else None, "proposal": proposal.model_dump() if proposal else None,
        "curve": [p.model_dump() for p in points], "observations": [o.model_dump() for o in observations],
        "note": "proposal only: config.yaml is not changed",
    }
    print(f"routing_confidence ({'live' if live else 'mock'} classifier): {len(rows)} utterances, "
          f"accuracy {accuracy:.3f}, {result['failed_calls']} failed calls")
    for label, p in (("current", current), ("proposal", proposal)):
        if p is not None:
            acc = "n/a" if p.accuracy_covered is None else f"{p.accuracy_covered:.3f}"
            print(f"  {label:<9} t={p.threshold:.2f}  coverage {p.coverage:.3f}  accuracy when accepted {acc}  "
                  f"correct but handed off {p.right_but_handed_off}")
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "routing_threshold.json").write_text(json.dumps(result, indent=2))
    return result
