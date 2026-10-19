"""Golden snapshot of the single-question path in mock mode (Phase 7 parity, FR-I5; PLAN section 8).

`record_golden` runs the 53 KB questions, the 21 gap questions and the 12 scenario scripts through the whole
mock app (`build_stack`, hashing embeddings, scripted models, mock tickets, no cache) and keeps, for each
turn, what the employee saw and how the graph got there, plus a digest of every message list sent to the
answer model and the grounding judge. The multi-agent graph must reproduce it exactly: the same replies,
steps, citations and chunks, and the same LLM inputs. The routing model's input is left out on purpose,
because the orchestrator prompt changes in Phase 7; its decisions are in the record (`intent`, `route`).

Mock retrieval is not fully repeatable: the HR-filtered search for gap-006 ranks a near-tie differently from one
process to the next (the pinned hash seed does not remove it), which changes one context entry sent to the
answer model. The snapshot is therefore recorded five times in separate processes, and a field that differs
between those runs is listed under "volatile" and skipped when comparing. Everything else must match exactly.

Regenerate only when a behaviour change is intended and approved: `python -m tests.golden_helpers --write`.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.agent_metrics import _DEGRADE
from evals.common import EVALS_DIR
from evals.harness import gap_rows, kb_rows
from evals.scenario_schema import load_scenario
from smartdesk_ai.agent.factory import build_stack
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import AgentState, new_thread_id
from smartdesk_ai.config.settings import Settings, load_settings
from smartdesk_ai.llm import mock
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.tickets import MockTicketClient

GOLDEN_FILE = Path(__file__).parent / "golden" / "single_path.json"
RECORDED_MODELS = ("MockAnswerModel", "MockGroundingModel")


def digest(messages: Any) -> str:
    text = "\n".join(f"{getattr(m, 'type', 'message')}:{getattr(m, 'content', m)}" for m in messages)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


@contextmanager
def recorded_llm_inputs() -> Iterator[list[tuple[str, str]]]:
    """Collect (model, digest of the message list) for each call to the answer and judge mock models."""
    calls: list[tuple[str, str]] = []
    originals = {name: getattr(mock, name).invoke for name in RECORDED_MODELS}

    def wrap(name: str):
        original = originals[name]

        def invoke(self, messages):
            calls.append((name, digest(messages)))
            return original(self, messages)

        return invoke

    for name in RECORDED_MODELS:
        setattr(getattr(mock, name), "invoke", wrap(name))
    try:
        yield calls
    finally:
        for name in RECORDED_MODELS:
            setattr(getattr(mock, name), "invoke", originals[name])


def _record(state: AgentState, calls: list[tuple[str, str]]) -> dict[str, Any]:
    part = state.part_result
    return {
        "reply": state.reply,
        "steps": list(state.steps),
        "intent": state.intent.intent if state.intent else None,
        "route_kind": state.route_kind,
        "awaiting": state.awaiting,
        "offered": state.pending_offer is not None,
        "citations": list(part.citations) if part else [],
        "chunks": [[c.entry_id, round(c.score, 6)] for c in part.chunks] if part else [],
        "retried": part.retried if part else None,
        "llm_inputs": [list(c) for c in calls],
    }


def _stack(settings: Settings, work: Path, client: MockTicketClient, registry: HealthRegistry):
    return build_stack(settings, environ={}, dotenv_path=None, data_dir=work, use_cache=False, registry=registry,
                       client=client, tracing=False)


def record_questions(settings: Settings, work: Path) -> dict[str, dict[str, Any]]:
    stack = _stack(settings, work / "questions", MockTicketClient(), HealthRegistry())
    out: dict[str, dict[str, Any]] = {}
    for row in [*kb_rows(), *gap_rows()]:
        with recorded_llm_inputs() as calls:
            state = run_turn(stack.graph, new_thread_id(), row.question)
        out[row.id] = _record(state, calls)
    return out


def record_scenarios(settings: Settings, work: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((EVALS_DIR / "scenarios").glob("*.yaml")):
        scenario = load_scenario(path)
        client, registry = MockTicketClient(), HealthRegistry()
        for component in scenario.setup.degraded:
            registry.mark_degraded(_DEGRADE[component], "scripted degraded mode")
        stack = _stack(settings, work / scenario.id, client, registry)
        client.unreachable = scenario.setup.github == "down"
        session = new_thread_id()
        turns = []
        for index, turn in enumerate(scenario.turns):
            with recorded_llm_inputs() as calls:
                state = run_turn(stack.graph, session, turn.user,
                                 email=scenario.setup.email_known if index == 0 else None)
            turns.append({"user": turn.user, **_record(state, calls)})
        out[scenario.id] = turns
    return out


def record_golden(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        return {"questions": record_questions(settings, work), "scenarios": record_scenarios(settings, work)}


def record_golden_pinned() -> dict[str, Any]:
    """`record_golden` in a child process (fixed hash seed, so set ordering is not a second source of variation)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "golden.json"
        subprocess.run([sys.executable, "-m", "tests.golden_helpers", "--out", str(out)], check=True,
                       env={**os.environ, "PYTHONHASHSEED": "0"}, cwd=Path(__file__).parent.parent,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return json.loads(out.read_text())


def _records(golden: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every turn of a snapshot under one key: question id, or scenario name plus turn number."""
    flat = dict(golden["questions"])
    for name, turns in golden["scenarios"].items():
        flat.update({f"{name}#{i}": turn for i, turn in enumerate(turns)})
    return flat


def volatile_fields(runs: list[dict[str, Any]]) -> dict[str, list[str]]:
    """The fields that are not the same in every run, per turn."""
    flats = [_records(r) for r in runs]
    return {key: sorted(f for f in flats[0][key] if any(flat[key][f] != flats[0][key][f] for flat in flats[1:]))
            for key in flats[0] if any(flat[key] != flats[0][key] for flat in flats[1:])}


def write_golden(repeats: int = 5) -> None:
    runs = [record_golden_pinned() for _ in range(repeats)]
    GOLDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_FILE.write_text(json.dumps({**runs[0], "volatile": volatile_fields(runs)}, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_FILE}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--out" in args:
        Path(args[args.index("--out") + 1]).write_text(json.dumps(record_golden(), sort_keys=True))
    elif "--write" in args:
        write_golden()
    else:
        sys.exit("pass --write to overwrite the golden snapshot (only for an approved behaviour change)")
