"""Orchestrator input tokens per turn on the scripted scenarios (FR-I3, FR-H5, FR-H7), counted locally with tiktoken.

The orchestrator is a Main-tier call on every turn and, since Phase 7, it reads the whole conversation (masked and
compacted). This runs the 12 scenario scripts through the mock app, records every message list the orchestrator is
shown, and counts its tokens. No API call is made: the models are the mock ones, the count is tiktoken's.

Per turn it reports the whole input, the system prompt, the "Conversation so far" block (the part the full-conversation
decision added), and what the input would have been without that block. It also says on which turn compaction fired
(none, in scripts this short) and projects the turn at which it would, from the measured growth per turn.

    python -m evals.orchestrator_tokens [--out evals/results/orchestrator_tokens.json]
"""
import argparse
import json
import statistics
import tempfile
from pathlib import Path
from typing import Any

import tiktoken
from pydantic import BaseModel

from evals.agent_metrics import _DEGRADE
from evals.common import EVALS_DIR
from evals.harness import RESULTS_DIR
from evals.scenario_schema import load_scenario
from smartdesk_ai.agent.factory import build_stack
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.config.settings import Settings, load_settings
from smartdesk_ai.llm import mock
from smartdesk_ai.ops.compaction import is_summary
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.tickets import MockTicketClient

FALLBACK_ENCODING = "o200k_base"
HISTORY_MARKER = "Conversation so far:\n"


class TurnTokens(BaseModel):
    scenario: str
    turn: int
    total: int  # the whole orchestrator input: system prompt plus the human message
    prompt: int  # the system prompt
    history: int  # the "Conversation so far" block
    without_history: int  # the input as it was before the full-conversation decision
    compacted: bool  # the conversation held a summary at this turn


class TokenReport(BaseModel):
    model: str
    turns: list[TurnTokens]
    total_min: int
    total_median: float
    total_max: int
    history_median: float
    history_max: int
    first_compaction: str | None  # "scenario turn N", or None when no script reached the trigger
    messages_tokens_per_turn: float  # measured growth of the conversation per employee turn
    trigger_tokens: int  # 75% of the Main window (config compaction.trigger_fraction)
    projected_compaction_turn: int  # turns of that growth to reach the trigger
    usd_per_call_at_trigger: float  # one orchestrator call with the conversation at the trigger, config prices


def _encoding(model_id: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model_id)
    except KeyError:
        return tiktoken.get_encoding(FALLBACK_ENCODING)


def _count(encoding: tiktoken.Encoding, text: str) -> int:
    return len(encoding.encode(text))


def measure(settings: Settings, work: Path) -> TokenReport:
    encoding = _encoding(settings.models.main.id)
    seen: list[tuple[str, str]] = []
    original = mock.MockRoutingModel.invoke

    def recording(self, messages):
        human = str(messages[-1].content)
        if "Last topic:" in human:  # the classifier call (the tool pick has no such line)
            seen.append((str(messages[0].content), human))
        return original(self, messages)

    turns: list[TurnTokens] = []
    first_compaction: str | None = None
    growth: list[int] = []
    mock.MockRoutingModel.invoke = recording
    try:
        for path in sorted((EVALS_DIR / "scenarios").glob("*.yaml")):
            scenario = load_scenario(path)
            client, registry = MockTicketClient(), HealthRegistry()
            for component in scenario.setup.degraded:
                registry.mark_degraded(_DEGRADE[component], "scripted degraded mode")
            stack = build_stack(load_settings(), environ={}, dotenv_path=None, data_dir=work / scenario.id,
                                use_cache=False, registry=registry, client=client, tracing=False)
            client.unreachable = scenario.setup.github == "down"
            session = new_thread_id()
            before_tokens = 0
            for index, turn in enumerate(scenario.turns, 1):
                seen.clear()
                state = run_turn(stack.graph, session, turn.user, email=scenario.setup.email_known if index == 1 else None)
                now = sum(_count(encoding, str(m.content)) for m in state.messages)
                growth.append(now - before_tokens)
                before_tokens = now
                compacted = bool(state.messages) and is_summary(state.messages[0])
                if compacted and first_compaction is None:
                    first_compaction = f"{scenario.id} turn {index}"
                for system, human in seen:  # one classifier call per routed turn
                    history = human.split(HISTORY_MARKER, 1)[1].rsplit("\nMessage: ", 1)[0] if HISTORY_MARKER in human else ""
                    block = _count(encoding, HISTORY_MARKER + history + "\n") if history else 0
                    whole = _count(encoding, system) + _count(encoding, human)
                    turns.append(TurnTokens(scenario=scenario.id, turn=index, total=whole,
                                            prompt=_count(encoding, system), history=block,
                                            without_history=whole - block, compacted=compacted))
    finally:
        mock.MockRoutingModel.invoke = original
    totals = [t.total for t in turns]
    histories = [t.history for t in turns]
    per_turn = statistics.mean(growth)
    trigger = int(settings.compaction.trigger_fraction * settings.models.main.context_window)
    prompt_tokens = turns[0].prompt
    return TokenReport(
        model=settings.models.main.id, turns=turns,
        total_min=min(totals), total_median=statistics.median(totals), total_max=max(totals),
        history_median=statistics.median(histories), history_max=max(histories),
        first_compaction=first_compaction, messages_tokens_per_turn=round(per_turn, 1), trigger_tokens=trigger,
        projected_compaction_turn=int(trigger // per_turn) + 1,
        usd_per_call_at_trigger=round((trigger + prompt_tokens) * settings.models.main.price.input / 1_000_000, 4),
    )


def render(report: TokenReport) -> str:
    return "\n".join([
        f"Orchestrator input per turn, {len(report.turns)} routed turns in 12 scenarios, tiktoken ({report.model}), no API calls",
        f"  whole input   min {report.total_min}   median {report.total_median:.0f}   max {report.total_max} tokens",
        f"  of which the conversation block: median {report.history_median:.0f}, max {report.history_max} tokens",
        f"  compaction fired in the scripts: {report.first_compaction or 'never (the scripts are too short)'}",
        f"  conversation grows {report.messages_tokens_per_turn} tokens per employee turn; the trigger is "
        f"{report.trigger_tokens} tokens, reached at about turn {report.projected_compaction_turn}; one orchestrator call "
        f"there costs about ${report.usd_per_call_at_trigger} in input",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count the orchestrator's input tokens per turn on the scenarios (local).")
    parser.add_argument("--out", default=str(RESULTS_DIR / "orchestrator_tokens.json"))
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        report = measure(load_settings(), Path(tmp))
    Path(args.out).write_text(json.dumps(report.model_dump(), indent=1) + "\n")
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
