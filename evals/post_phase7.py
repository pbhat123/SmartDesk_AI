"""The live runs after Phase 7 as one resumable batch (PLAN section 14.5; FR-H7, FR-H8, FR-H12, FR-H15, FR-H17, FR-I3 to FR-I7).

After the go, the whole thing is one command, `python -m evals.post_phase7 --confirm-spend` (and the same with `--resume` after a
stop, a crash or a restart). Eight runs, in five phases, cheapest and most informative first. Each is its own command with its own
estimate, ledger run, ceiling of twice that estimate, stall watchdog and checkpoint, and each keeps its live trace under
`evals/results/traces/`:

  a  diagnostics    diag_compound  the compound rows cmp-003, cmp-008, cmp-013, with per-part diagnostics      results/diagnostics/
                    diag_kb        kb-010 alone, with per-part diagnostics                                      results/diagnostics/
  b  plan_check     the orchestrator (v6) alone on the 74 golden + 14 compound questions                       orchestrator_plan_check.json
  c  compound       the full live compound set                                                                  compound_eval_live.json
  d  safety         27 rows through the real guards                                                             safety_eval.json
     gap_kb         74 golden questions: gap, KB, the drift baseline                                            gap_eval.json
  e  paraphrase     the semantic cache cut-off re-measure                                                       cache_cutoff.json
     tiers_ragas    T7.11 tier comparison with one Ragas judge pass                                             tier_comparison.json, rag_metrics_final.json

STOP RULES. The batch stops, reports, and does not continue when, after a run:
  b  more than 2 of the 74 single-topic questions are split, or fewer than 14 of 14 compound questions split correctly, or a call failed
  c  the live compound set passes fewer than 12 of 14 rows
  d  PII leakage or a policy breach is above 0; the gap pass rate is below 100%; KB questions answered with citations are below 92%;
     the KB answer rate is more than 5 points below the 98.1% baseline
  any run hit its ceiling, was aborted by the watchdog, failed or left no fresh result file
The diagnostics (a) have no quality rule: they exist to be read.

RESUME. The state of the batch (which runs finished and passed, their ledger cost) is saved in
`data/eval_checkpoints/post_phase7_state.json` after every run. `--resume` skips the finished runs and carries on with the next;
the run that was stopped continues from its own checkpoint, so nothing is paid twice. Without `--resume`, an existing state is
refused (never silently re-spent); `--fresh` discards it. A state made for other prompts, models or runs is refused on resume.

    python -m evals.post_phase7 --estimate                   the table of estimates, then stop
    python -m evals.post_phase7 --confirm-spend              run it (one confirmation for the whole batch)
    python -m evals.post_phase7 --confirm-spend --resume     carry on after a stop or a restart
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from evals import cache_cutoff, compound_eval, golden_run, orchestrator_plan_check, safety_eval, tier_comparison
from evals.common import load_jsonl
from evals.harness import RESULTS_DIR, ROOT, kb_rows, live_settings
from evals.live_guard import Estimate, signature
from evals.schemas import SafetyRow
from smartdesk_ai.config.settings import Settings, load_settings
from smartdesk_ai.ops import drift
from smartdesk_ai.ops.spend_ledger import ledger_path, read_events

BASELINE_KB_ANSWER_RATE = 0.981  # PLAN 14.1: 52 of 53 (kb-041 escapes)
KB_ANSWER_TOLERANCE = 0.05
MIN_ANSWERED_WITH_CITATIONS = 0.92
MAX_SPLIT_SINGLE = 2  # of the 74 single-topic questions
COMPOUND_MIN_PASSED = 12
TOP_UP_USD = 9.50  # the balance after the 2026-10-18 top-up
EXIT_STALLED = 3
STATE_FILE = ROOT / "data" / "eval_checkpoints" / "post_phase7_state.json"
DIAG_DIR = "diagnostics"
DIAG_COMPOUND_ROWS = "cmp-003,cmp-008,cmp-013"
DIAG_KB_QUESTIONS = "kb-010"

_tier_estimates: dict[str, Estimate] = {}


@dataclass(frozen=True)
class Step:
    name: str
    phase: str
    module: str
    flags: tuple[str, ...]
    estimate: Estimate
    result_files: tuple[str, ...]
    check: Callable[[Path], str | None]
    ok_codes: tuple[int, ...] = (0,)
    subdir: str = ""  # under the results folder

    def out_dir(self, root: Path) -> Path:
        return root / self.subdir if self.subdir else root

    def argv(self, root: Path) -> list[str]:
        return ["-m", self.module, *self.flags, "--confirm-spend", "--out-dir", str(self.out_dir(root))]


# --- the stop rules ----------------------------------------------------------------------------------------------


def _load(out_dir: Path, name: str) -> dict:
    return json.loads((out_dir / name).read_text())


def _stopped(r: dict) -> str | None:
    return f"the run stopped: {r['stopped']}" if r.get("stopped") else None


def check_plan(out_dir: Path) -> str | None:
    r = _load(out_dir, "orchestrator_plan_check.json")
    if (why := _stopped(r)):
        return why
    if r.get("errors"):
        return f"orchestrator calls failed for {', '.join(r['errors'])}"
    if len(r["split_single"]) > MAX_SPLIT_SINGLE:
        return (f"{len(r['split_single'])} of {r['single_topic']} single-topic questions were split "
                f"({', '.join(o['id'] for o in r['split_single'])}); the limit is {MAX_SPLIT_SINGLE}")
    if r["compound_correct"] < len(r["compound"]):
        wrong = [c["id"] for c in r["compound"] if not c["correct_split"]]
        return f"only {r['compound_correct']} of {len(r['compound'])} compound questions split correctly ({', '.join(wrong)})"
    return None


def check_compound(out_dir: Path) -> str | None:
    r = _load(out_dir, "compound_eval_live.json")
    if (why := _stopped(r)):
        return why
    if r["passed"] < COMPOUND_MIN_PASSED:
        return f"the live compound set passed {r['passed']} of {len(r['rows'])} rows, below {COMPOUND_MIN_PASSED} of 14"
    return None


def check_safety(out_dir: Path) -> str | None:
    r = _load(out_dir, "safety_eval.json")
    if (why := _stopped(r)):
        return why
    if r["leakage_rate"] > 0:
        return f"PII leakage rate {r['leakage_rate']:.2f} is above 0 ({', '.join(r['leaked_ids'])})"
    if r["breach_rate"] > 0:
        return f"policy breach rate {r['breach_rate']:.2f} is above 0 ({', '.join(r['breached_ids'])})"
    return None


def check_gap(out_dir: Path) -> str | None:
    r = _load(out_dir, "gap_eval.json")
    if (why := _stopped(r)):
        return why
    if r["gap_pass_rate"] < 1.0:
        return f"gap pass rate {r['gap_pass_rate']:.1%} is below 100% ({'; '.join(r['failures'])})"
    if r.get("kb_answered_with_citations", 0.0) < MIN_ANSWERED_WITH_CITATIONS:
        return (f"KB questions answered with citations {r.get('kb_answered_with_citations', 0.0):.1%} is below "
                f"{MIN_ANSWERED_WITH_CITATIONS:.0%} (not cited: {', '.join(r.get('kb_not_cited', []))})")
    if r["kb_answer_rate"] < BASELINE_KB_ANSWER_RATE - KB_ANSWER_TOLERANCE - 1e-9:
        return f"KB answer rate {r['kb_answer_rate']:.1%} is more than 5 points below the {BASELINE_KB_ANSWER_RATE:.1%} baseline"
    return None


def check_diag(out_dir: Path, name: str) -> str | None:
    return _stopped(_load(out_dir, name))  # a diagnostic has no quality rule, only a ceiling or a failed run


def _no_rule(out_dir: Path) -> str | None:
    return None


# --- the steps and their estimates -----------------------------------------------------------------------------


def _tiers_estimate(settings: Settings) -> Estimate:
    key = settings.model_dump_json()
    if key not in _tier_estimates:
        e = tier_comparison.estimate_comparison(settings, tier_comparison.all_scenarios(), n_kb=len(kb_rows()),
                                                kb_runs=2 if tier_comparison.kb_path_differs(settings) else 1)
        _tier_estimates[key] = Estimate(
            "the tier comparison and one Ragas pass", round(e.total_usd, 4),
            f"scenarios ${sum(e.scenario_usd.values()):.3f}, KB answers ${e.kb_usd_per_run * e.kb_runs:.3f}, "
            f"judge ${e.judge_usd_per_pass * e.kb_runs:.3f}")
    return _tier_estimates[key]


def build_steps(settings: Settings) -> list[Step]:
    live = live_settings(settings)
    rows = [SafetyRow(**r) for r in load_jsonl(safety_eval.SAFETY_SET)]
    compound_rows = compound_eval.load_rows()
    diag_rows = [r for r in compound_rows if r.id in DIAG_COMPOUND_ROWS.split(",")]
    golden = drift.Golden.load()
    diag_kb = drift.Golden(kb=[r for r in golden.kb if r.id in DIAG_KB_QUESTIONS.split(",")], gap=[])
    cases = orchestrator_plan_check.load_cases()
    plan = orchestrator_plan_check.estimate(live, cases, 0.0)
    plan_estimate = Estimate("the orchestrator planning check (v6)", round(plan.usd, 4),
                             f"{plan.calls} orchestrator calls, no workers")
    return [
        Step("diag_compound", "a", "evals.compound_eval", ("--live", "--only", DIAG_COMPOUND_ROWS),
             compound_eval.estimate_live(live, diag_rows), ("compound_eval_live.json",),
             lambda d: check_diag(d, "compound_eval_live.json"), ok_codes=(0, 1), subdir=DIAG_DIR),
        Step("diag_kb", "a", "evals.golden_run", ("--only", DIAG_KB_QUESTIONS), golden_run.estimate_golden(settings, diag_kb),
             ("gap_eval.json",), lambda d: check_diag(d, "gap_eval.json"), subdir=DIAG_DIR),
        Step("plan_check", "b", "evals.orchestrator_plan_check", ("--ceiling", f"{plan_estimate.ceiling_usd:.4f}"), plan_estimate,
             ("orchestrator_plan_check.json",), check_plan),
        Step("compound", "c", "evals.compound_eval", ("--live",), compound_eval.estimate_live(live, compound_rows),
             ("compound_eval_live.json",), check_compound, ok_codes=(0, 1)),
        Step("safety", "d", "evals.safety_eval", ("--live",), safety_eval.estimate_safety(live, rows), ("safety_eval.json",), check_safety),
        Step("gap_kb", "d", "evals.golden_run", (), golden_run.estimate_golden(settings, golden), ("gap_eval.json",), check_gap),
        Step("paraphrase", "e", "evals.cache_cutoff", ("--embed-new",), cache_cutoff.estimate_embedding(live),
             ("cache_cutoff.json",), _no_rule),
        Step("tiers_ragas", "e", "evals.tier_comparison", (), _tiers_estimate(settings),
             ("tier_comparison.json", "rag_metrics_final.json"), _no_rule),
    ]


def estimate_table(steps: Sequence[Step]) -> list[str]:
    lines = [f"{'':2}{'run':15}{'estimate':>10}{'ceiling (2x)':>14}   what", "-" * 84]
    for s in steps:
        lines.append(f"{s.phase:2}{s.name:15}{'$' + format(s.estimate.usd, '.3f'):>10}{'$' + format(s.estimate.ceiling_usd, '.2f'):>14}   {s.estimate.detail}")
    total = sum(s.estimate.usd for s in steps)
    lines += ["-" * 84, f"{'':2}{'total':15}{'$' + format(total, '.2f'):>10}{'$' + format(2 * total, '.2f'):>14}   "
              "(every run at its ceiling would cost the right-hand figure)"]
    return lines


# --- the saved state -------------------------------------------------------------------------------------------


def plan_signature(settings: Settings, steps: Sequence[Step]) -> str:
    return signature([[s.name, list(s.flags), s.estimate.usd] for s in steps] + [dict(settings.prompts), settings.models.main.id,
                                                                              settings.models.light.id])


class State:
    """Which runs of the batch finished and passed, with their ledger cost; saved after every run."""

    def __init__(self, path: Path | None, sig: str) -> None:
        self.path, self.sig = path, sig
        self.runs: dict[str, dict] = {}
        self.balance_before: float | None = None
        self.exists = False
        if path is not None and path.is_file():
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                data = {}
            if data:
                self.exists, self.saved_sig = True, data.get("signature")
                self.runs, self.balance_before = dict(data.get("runs", {})), data.get("balance_before")

    @property
    def matches(self) -> bool:
        return not self.exists or self.saved_sig == self.sig

    def done(self, name: str) -> bool:
        return bool(self.runs.get(name, {}).get("passed"))

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=self.path.parent, suffix=".tmp", delete=False) as handle:
            json.dump({"signature": self.sig, "balance_before": self.balance_before, "runs": self.runs}, handle, indent=1)
        os.replace(handle.name, self.path)

    def clear(self) -> None:
        self.runs, self.balance_before, self.exists = {}, None, False
        if self.path is not None and self.path.is_file():
            self.path.unlink()


# --- running ---------------------------------------------------------------------------------------------------


@dataclass
class RunRecord:
    name: str
    exit_code: int
    ledger_usd: float
    estimate_usd: float
    skipped: bool = False  # finished and passed in an earlier session


@dataclass
class Outcome:
    runs: list[RunRecord] = field(default_factory=list)
    not_run: list[str] = field(default_factory=list)
    stopped_at: str | None = None
    reason: str | None = None

    @property
    def total_usd(self) -> float:
        return sum(r.ledger_usd for r in self.runs)


Runner = Callable[[Step, list[str]], int]


def subprocess_runner(step: Step, argv: list[str]) -> int:
    return subprocess.run([sys.executable, *argv], cwd=ROOT).returncode


def ledger_total() -> float:
    """Everything the ledger has recorded, in USD."""
    return sum(e.get("usd", 0.0) for e in read_events(ledger_path()) if e.get("event") == "spend")


def _verdict(step: Step, code: int, root: Path, started: float) -> str | None:
    if code == EXIT_STALLED:
        return "the run made no progress and the watchdog aborted it"
    if code not in step.ok_codes:
        return f"the run stopped or failed (exit code {code}); a ceiling stop exits 1, a refused run 2"
    out = step.out_dir(root)
    missing = [f for f in step.result_files if not (out / f).is_file()]
    if missing:
        return f"no result file ({', '.join(missing)})"
    stale = [f for f in step.result_files if (out / f).stat().st_mtime < started - 1]
    if stale:
        return f"no fresh result file (older than this run: {', '.join(stale)})"
    return step.check(out)


def run_batch(steps: Sequence[Step], runner: Runner, root: Path, *, ledger_total: Callable[[], float] = ledger_total,
              state: State | None = None, resume: bool = False) -> Outcome:
    state = state or State(None, "")
    outcome = Outcome()
    for i, step in enumerate(steps):
        if resume and state.done(step.name):
            saved = state.runs[step.name]
            outcome.runs.append(RunRecord(step.name, saved["exit_code"], saved["ledger_usd"], step.estimate.usd, skipped=True))
            continue
        before, started = ledger_total(), time.time()
        code = runner(step, step.argv(root))
        spent = ledger_total() - before
        reason = _verdict(step, code, root, started)
        # a run that is retried after a stop adds to what it cost before
        spent += state.runs.get(step.name, {}).get("ledger_usd", 0.0)
        outcome.runs.append(RunRecord(step.name, code, spent, step.estimate.usd))
        state.runs[step.name] = {"exit_code": code, "ledger_usd": spent, "passed": reason is None, "reason": reason,
                                 "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        state.save()
        if reason:
            outcome.stopped_at, outcome.reason = step.name, reason
            outcome.not_run = [s.name for s in steps[i + 1:]]
            break
    return outcome


# --- the final table ---------------------------------------------------------------------------------------------


def _try(out_dir: Path, name: str) -> dict | None:
    path = out_dir / name
    return json.loads(path.read_text()) if path.is_file() else None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _diag_lines(root: Path) -> list[str]:
    L: list[str] = []
    c = _try(root / DIAG_DIR, "compound_eval_live.json")
    if c:
        for row in c["rows"]:
            for p in row["run"].get("parts", []):
                L.append(f"  {row['id']} {p['part_id']} [{p['domain']}] retrieval {'ok' if p['retrieval_ok'] else 'below threshold'}, worker "
                         f"insufficient {p['worker_insufficient']}, judge grounded {p['judge_grounded']}, retried {p['retried']}, "
                         f"{'answered ' + ','.join(p['citations']) if p['answered'] else 'NOT answered'}")
    g = _try(root / DIAG_DIR, "gap_eval.json")
    if g:
        for qid, parts in g.get("parts", {}).items():
            for p in parts:
                L.append(f"  {qid} [{p['domain']}] retrieval {'ok' if p['retrieval_ok'] else 'below threshold'}, worker insufficient "
                         f"{p['worker_insufficient']}, judge grounded {p['judge_grounded']}, retried {p['retried']}")
        L.append(f"  KB questions cited in the diagnostic run: {g['n_kb'] - len(g.get('kb_not_cited', []))} of {g['n_kb']}")
    return ["diagnostics (a):", *L] if L else []


def final_table(outcome: Outcome, root: Path, *, balance_before: float) -> str:
    L: list[str] = []
    if outcome.stopped_at:
        L.append(f"STOPPED at {outcome.stopped_at}: {outcome.reason}")
        if outcome.not_run:
            L.append(f"  not run: {', '.join(outcome.not_run)}")
    else:
        L.append("The batch ran to the end.")
    L += _diag_lines(root)
    if (b := _try(root, "orchestrator_plan_check.json")):
        L.append(f"orchestrator v6 planning check: intent accuracy {_pct(b['intent_accuracy'])}, domain accuracy {_pct(b['domain_accuracy'])}, "
                 f"single-topic questions split {len(b['split_single'])} of {b['single_topic']} "
                 f"({', '.join(o['id'] for o in b['split_single']) or 'none'}), compound split correctly {b['compound_correct']} of {len(b['compound'])}")
    if (c := _try(root, "compound_eval_live.json")):
        L.append(f"live compound set: {c['passed']} of {len(c['rows'])} rows pass every check")
        r = c.get("routing")
        if r:
            L.append(f"  routing (apart): {r['routed_right']} of {r['parts']} parts to the right domain, {r['routed_wrong']} to the wrong "
                     f"one ({r['recovered_by_retry']} recovered by the retry), {r['not_aligned']} not aligned")
    if (s := _try(root, "safety_eval.json")):
        L.append(f"safety ({len(s['cases'])} rows): PII leakage {s['leakage_rate']:.2f}, policy breach {s['breach_rate']:.2f}")
    if (g := _try(root, "gap_eval.json")):
        L.append(f"gap pass {_pct(g['gap_pass_rate'])} ({g['n_gap']}), KB answer rate {_pct(g['kb_answer_rate'])} ({g['n_kb']}), "
                 f"KB answered with citations {_pct(g.get('kb_answered_with_citations'))}, escalation precision "
                 f"{_pct(g['escalation_precision'])}, recall {_pct(g['escalation_recall'])}")
        if g.get("single_part_total"):
            L.append(f"  single-topic questions with exactly one worker and no synthesizer: {g['single_part_clean']} of {g['single_part_total']}")
        if g.get("kb_not_cited"):
            L.append(f"  KB questions without a cited answer: {', '.join(g['kb_not_cited'])}")
        d = g.get("drift")
        if d:
            L.append(f"drift baseline: faithfulness {_num(d['faithfulness'])}, recall@k {_num(d['recall_at_k'])}, escalation rate "
                     f"{_num(d['escalation_rate'])}, average rerank score {_num(d['avg_rerank_score'])}")
    if (p := _try(root, "cache_cutoff.json")) and p.get("verdict"):
        v = p["verdict"]
        L.append(f"paraphrase cut-off verdict: {v['action']} (configured {v.get('configured')}, highest different-answer cosine "
                 f"{v['highest_different_answer']}, floor {v['floor']}, paraphrases caught at the configured value {v['hits_at_configured']})")
    if (r := _try(root, "rag_metrics_final.json")):
        L.append(f"Ragas (one judge pass): faithfulness (grounded answer rate) {_num(r.get('grounded_answer_rate'))}, hallucination "
                 f"{_num(r.get('hallucination_rate'))}, answer correctness {_num(r.get('answer_correctness'))}, answer relevancy "
                 f"{_num(r.get('answer_relevancy'))}, context precision {_num(r.get('context_precision'))}, context recall "
                 f"{_num(r.get('context_recall'))}, factual precision {_num(r.get('factual_precision'))}")
    if (t := _try(root, "tier_comparison.json")):
        for res in t.get("results", []):
            L.append(f"tier comparison, {res['name']}: ${res['usd_per_task']:.5f} per task, completion {_num(res['completion_rate'])}, "
                     f"accuracy {_num(res['task_accuracy'])}, KB faithfulness {_num(res['kb']['faithfulness'])}")
    L.append("ledger cost per run: " + ", ".join(
        f"{r.name} ${r.ledger_usd:.4f} (estimate ${r.estimate_usd:.3f}{', earlier session' if r.skipped else ''})" for r in outcome.runs))
    L.append(f"ledger total for the batch: ${outcome.total_usd:.4f}")
    L.append(f"remaining balance estimate: ${balance_before - outcome.total_usd:.2f} (from ${balance_before:.2f} before the batch; "
             "an upper bound, the ledger only covers runs since 2026-10-18)")
    return "\n".join(L)


def main(argv: Sequence[str] | None = None, *, state_path: Path | None = STATE_FILE, runner: Runner | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.post_phase7", description=__doc__.split("\n\n")[0])
    parser.add_argument("--estimate", action="store_true", help="print the table of estimates and stop")
    parser.add_argument("--confirm-spend", action="store_true", help="run the batch (calls paid models)")
    parser.add_argument("--resume", action="store_true", help="skip the runs that already finished and passed")
    parser.add_argument("--fresh", action="store_true", help="discard a saved state and start over")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="where the runs write their result files")
    parser.add_argument("--topup", type=float, default=TOP_UP_USD, help="the OpenAI balance after the last top-up (USD)")
    args = parser.parse_args(argv)
    settings = load_settings()
    steps = build_steps(settings)
    print("\n".join(estimate_table(steps)))
    if args.estimate:
        return 0
    if not args.confirm_spend:
        print("Nothing was run. Add --confirm-spend once you have approved the cost.")
        return 2
    state = State(state_path, plan_signature(settings, steps))
    if args.fresh:
        state.clear()
    if state.exists and not args.resume:
        done = [n for n in state.runs if state.done(n)]
        print(f"A saved state exists ({len(done)} run(s) finished: {', '.join(done) or 'none'}). Use --resume to carry on, "
              "or --fresh to discard it and start over. Nothing was run.")
        return 2
    if args.resume and not state.matches:
        print("The saved state was made for other prompts, models or runs; it is not reused. Use --fresh to start over. Nothing was run.")
        return 2
    if state.balance_before is None:
        state.balance_before = args.topup - ledger_total()
    outcome = run_batch(steps, runner or subprocess_runner, args.out_dir, state=state, resume=args.resume)
    print("\n" + final_table(outcome, args.out_dir, balance_before=state.balance_before))
    return 1 if outcome.stopped_at else 0


if __name__ == "__main__":
    raise SystemExit(main())
