"""Live end-to-end threshold sweep (FR-A5, FR-A15): the gap and KB sets are run through the whole app once per
`thresholds.escalation` value, so the answer model and the judge are in the loop (the offline sweep in
`tune_threshold.py` sees only the cosine). Reports gap pass rate, KB answer rate, escalation precision and
recall, spend, and the outcome of gap-006 at each value. Nothing is written to config.yaml.

`python -m evals.gap_sweep --thresholds 0.30 0.35 0.45 0.50 --confirm-spend [--ceiling-usd 2.0]`
"""
import argparse
import json
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from evals.gap_eval import build_live_stack, run_cases, summarize
from smartdesk_ai.agent.factory import AgentStack, build_stack
from smartdesk_ai.config.settings import load_settings
from smartdesk_ai.ops.spend_ledger import end_run, start_run

WATCHED_GAP = "gap-006"  # the parking-fees question that leaked once


def sweep(thresholds: Sequence[float], stack_for: Callable[[float], AgentStack], *,
          ceiling_usd: float | None = None) -> list[dict]:
    """One row per threshold. After a run, if the total spend is over `ceiling_usd` the sweep stops."""
    rows, spent = [], 0.0
    for threshold in thresholds:
        stack = stack_for(threshold)
        cases = run_cases(stack)
        report = summarize(cases)
        cost = stack.services.cost.total
        spent += cost
        watched = next((c.outcome for c in cases if c.id == WATCHED_GAP), None)
        rows.append({
            "threshold": threshold, "gap_pass_rate": report.gap_pass_rate, "kb_answer_rate": report.kb_answer_rate,
            "escalation_precision": report.escalation_precision, "escalation_recall": report.escalation_recall,
            "cost_usd": cost, "gap_failures": report.failures, "gap_006": watched,
            "kb_escalated": [c.id for c in cases if c.label == "kb" and c.outcome != "answered"],
        })
        print(f"threshold {threshold:.2f}: gap pass {report.gap_pass_rate:.3f}, KB answer {report.kb_answer_rate:.3f}, "
              f"precision {report.escalation_precision}, recall {report.escalation_recall:.3f}, "
              f"{WATCHED_GAP} {watched}, spend ${cost:.4f} (total ${spent:.4f})", flush=True)
        if ceiling_usd is not None and spent > ceiling_usd:
            print(f"stopping: spend ${spent:.4f} is over the ceiling ${ceiling_usd:.2f}", file=sys.stderr)
            break
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.gap_sweep")
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--confirm-spend", action="store_true", help="allow a live sweep (costs money)")
    parser.add_argument("--ceiling-usd", type=float, default=None, help="stop starting new runs past this total spend")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.mock and not args.confirm_spend:
        print("error: a live sweep calls paid models; add --confirm-spend once you have approved the cost", file=sys.stderr)
        return 1
    settings = load_settings()
    work = Path(tempfile.mkdtemp(prefix="gap_sweep_"))

    def stack_for(threshold: float) -> AgentStack:
        key = "escalation_mock" if args.mock else "escalation"
        tuned = settings.model_copy(update={"thresholds": settings.thresholds.model_copy(update={key: threshold})})
        folder = work / f"t{threshold}"
        if args.mock:
            from tests.llm_fakes import FakeAnswerLLM  # the scripted insufficient model, as in the mock gap test

            return build_stack(tuned, environ={}, dotenv_path=None, data_dir=folder, mock_answer_model=FakeAnswerLLM("insufficient"))
        return build_live_stack(tuned, folder)

    if not args.mock:
        start_run(settings, ["python", "-m", "evals.gap_sweep", *(sys.argv[1:] if argv is None else argv)], "eval")
    try:
        rows = sweep(args.thresholds, stack_for, ceiling_usd=args.ceiling_usd)
    finally:
        end_run("ok")
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "gap_sweep.json").write_text(json.dumps({"mock": args.mock, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
