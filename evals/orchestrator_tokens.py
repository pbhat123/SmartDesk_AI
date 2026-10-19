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
from langchain_core.messages import HumanMessage
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


class LongConversation(BaseModel):
    """One synthetic conversation of many KB questions with answers of realistic length (about 250 tokens each)."""

    turns: int
    history_max_tokens: int  # the cap in force (config orchestrator.history_max_tokens)
    capped_max: int  # the largest orchestrator input with the cap
    capped_median: float
    uncapped_max: int  # what the same turns would have sent with no cap
    first_capped_turn: int | None  # the first turn whose history was cut
    usd_per_call_capped_max: float  # input cost of the largest capped call, config prices
    usd_per_call_uncapped_max: float


class TokenReport(BaseModel):
    model: str
    long_conversation: LongConversation | None = None
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


class _LongAnswer:
    """A scripted answer model: the first entry it was given, up to about 200 words, as the answer and cited. Taken from
    the context, so the mock grounding judge accepts it, and about as long as a real KB answer."""

    def invoke(self, messages):
        import re
        from types import SimpleNamespace

        context = str(messages[-1].content).split("Context:\n", 1)[1]
        first = re.match(r"\[([A-Z]+-[A-Z0-9-]+)\] [^\n]*\n(.*?)(?=\n\n\[|\Z)", context, re.S)
        body = " ".join(first.group(2).split()[:200])
        return SimpleNamespace(content=json.dumps({"answer": body, "insufficient": False, "citations": [first.group(1)]}))


LONG_QUESTIONS = ("How do I reset my forgotten password?", "How do I set up MFA on a new phone?",
                  "How do I connect to the VPN from home?", "How many casual leave days do I get?",
                  "When is my salary paid each month?", "How do I request an extra monitor?")


def measure_long(settings: Settings, work: Path, turns: int = 60) -> LongConversation:
    """The orchestrator's input over one long conversation, with the history cap and without it."""
    from smartdesk_ai.agent.orchestrator import transcript_for

    encoding = _encoding(settings.models.main.id)
    seen: list[tuple[str, str]] = []
    original = mock.MockRoutingModel.invoke

    def recording(self, messages):
        human = str(messages[-1].content)
        if "Last topic:" in human:
            seen.append((str(messages[0].content), human))
        return original(self, messages)

    capped: list[int] = []
    uncapped: list[int] = []
    first_cut: int | None = None
    mock.MockRoutingModel.invoke = recording
    try:
        stack = build_stack(load_settings(), environ={}, dotenv_path=None, data_dir=work / "long", use_cache=False,
                            registry=HealthRegistry(), client=MockTicketClient(), tracing=False,
                            mock_answer_model=_LongAnswer())
        session = new_thread_id()
        for turn in range(1, turns + 1):
            seen.clear()
            before = stack.graph.get_state({"configurable": {"thread_id": session}}).values.get("messages", [])
            run_turn(stack.graph, session, f"{LONG_QUESTIONS[turn % len(LONG_QUESTIONS)]} (question {turn})")
            system, human = seen[0]
            whole = _count(encoding, system) + _count(encoding, human)
            full = transcript_for([*before, HumanMessage("x")])
            block = _count(encoding, HISTORY_MARKER + full + "\n") if full else 0
            without = whole - (_count(encoding, HISTORY_MARKER + human.split(HISTORY_MARKER, 1)[1].rsplit("\nMessage: ", 1)[0] + "\n")
                               if HISTORY_MARKER in human else 0)
            capped.append(whole)
            uncapped.append(without + block)
            if first_cut is None and uncapped[-1] > capped[-1]:
                first_cut = turn
    finally:
        mock.MockRoutingModel.invoke = original
    price = settings.models.main.price.input / 1_000_000
    return LongConversation(
        turns=turns, history_max_tokens=settings.orchestrator.history_max_tokens, capped_max=max(capped),
        capped_median=statistics.median(capped), uncapped_max=max(uncapped), first_capped_turn=first_cut,
        usd_per_call_capped_max=round(max(capped) * price, 5), usd_per_call_uncapped_max=round(max(uncapped) * price, 5))


def measure(settings: Settings, work: Path, long_turns: int = 60) -> TokenReport:
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
        model=settings.models.main.id, long_conversation=measure_long(settings, work, long_turns), turns=turns,
        total_min=min(totals), total_median=statistics.median(totals), total_max=max(totals),
        history_median=statistics.median(histories), history_max=max(histories),
        first_compaction=first_compaction, messages_tokens_per_turn=round(per_turn, 1), trigger_tokens=trigger,
        projected_compaction_turn=int(trigger // per_turn) + 1,
        usd_per_call_at_trigger=round((trigger + prompt_tokens) * settings.models.main.price.input / 1_000_000, 4),
    )


def render(report: TokenReport) -> str:
    long = report.long_conversation
    extra = [] if long is None else [
        f"  one long conversation of {long.turns} turns (answers of about 250 tokens), history cap {long.history_max_tokens}: "
        f"whole input max {long.capped_max}, median {long.capped_median:.0f} tokens; without the cap it would reach "
        f"{long.uncapped_max}; the cap first cut the history at turn {long.first_capped_turn}; the largest call costs "
        f"${long.usd_per_call_capped_max} in input against ${long.usd_per_call_uncapped_max} uncapped"]
    return "\n".join([*extra, 
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
