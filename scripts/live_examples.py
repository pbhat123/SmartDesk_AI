#!/usr/bin/env python
"""Record the four example conversations of README section 14 (batch B, FR-B2, FR-B13, FR-H11).

    python scripts/live_examples.py                              # mock models and mock tickets, prints the transcript, free
    python scripts/live_examples.py --live                       # prints the estimate and stops
    python scripts/live_examples.py --live --confirm-spend       # runs it: real models, real GitHub issue, writes docs/examples.md

The conversations go through the same ChatService the UI builds, with one fake email. The live run uses a fresh temp data folder
(`data/` is untouched), blanks the Langfuse, LangSmith and Redis settings so nothing leaves the machine except OpenAI and GitHub
calls, and runs under `LiveRun` (ledger source `eval`, ceiling twice the estimate, watchdog). It stops instead of guessing when a
conversation does not go to plan, and the confirming "yes" is sent once, so it creates at most one issue. The reply texts are
saved exactly as the service returned them.
"""
import argparse
import os
import re
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EMAIL = "demo.employee@smartdesk.example"
MAX_TURNS = 7  # approved with the estimate
ESTIMATE_USD = 0.03
CEILING_USD = 0.06
DEFAULT_OUT = ROOT / "docs" / "examples.md"

PROMPTS = {
    "kb": "I forgot my laptop password. How can I reset it myself?",
    "gap": "Several keys on my laptop keyboard stopped responding. Can IT look at it?",
}
OFFER_ACCEPT = f"Yes please, my email is {EMAIL}"
CONFIRM = "Yes, go ahead"
STATUS = "Any update on my keyboard ticket?"
RECALL = "Any update on my ticket?"
SUMMARY_MARK = "Shall I go ahead and create this?"
TICKET = re.compile(r"Ticket #(\d+)")
LINK = re.compile(r"https://github\.com/\S+/issues/\d+")

TITLES = {
    "a": "A. A KB question answered with citations",
    "b": "B. A gap question: ticket offer, summary, approval, a real GitHub issue",
    "c": "C. A status check for the same email",
    "d": "D. A new session with the same email recalls the ticket (long-term memory)",
}


@dataclass
class Turn:
    user: str
    reply: str
    steps: list[str] = field(default_factory=list)


@dataclass
class Conversation:
    name: str
    thread: str
    turns: list[Turn] = field(default_factory=list)


@dataclass
class Outcome:
    conversations: list[Conversation] = field(default_factory=list)
    stopped: str | None = None
    ticket_number: int | None = None
    ticket_link: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def turn_count(self) -> int:
        return sum(len(c.turns) for c in self.conversations)


class _Stop(Exception):
    pass


def live_environ() -> dict[str, str]:
    """The environment of the live run: both modes live, and the optional external services blanked (blank means absent)."""
    return {"SMARTDESK_LLM_MODE": "live", "SMARTDESK_TICKETS_MODE": "live", "LANGSMITH_API_KEY": "",
            "LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": "", "LANGFUSE_HOST": "", "REDIS_URL": ""}


def run_examples(service, *, prompts: dict[str, str] | None = None, require_link: bool = True,
                 stop_check: Callable[[], str | None] | None = None,
                 on_turn: Callable[[], None] | None = None) -> Outcome:
    """Run the four conversations. Every turn is checked against the cap and `stop_check` first; a conversation that does not
    go to plan ends the run with `outcome.stopped` set, and nothing further is sent."""
    prompts = {**PROMPTS, **(prompts or {})}
    outcome = Outcome()
    run_id = os.urandom(3).hex()

    def say(conversation: Conversation, text: str) -> str:
        if outcome.turn_count >= MAX_TURNS:
            raise _Stop(f"turn cap of {MAX_TURNS} reached")
        reason = stop_check() if stop_check else None
        if reason:
            raise _Stop(reason)
        reply = service.handle(conversation.thread, text)
        conversation.turns.append(Turn(user=text, reply=reply.text, steps=list(reply.debug.get("steps", []))))
        if on_turn:
            on_turn()
        return reply.text

    def start(name: str, thread: str | None = None) -> Conversation:
        conversation = Conversation(name=name, thread=thread or f"example-{name}-{run_id}")
        outcome.conversations.append(conversation)
        return conversation

    try:
        a = start("a")
        if "Sources:" not in say(a, prompts["kb"]):
            outcome.notes.append("conversation a: the reply has no Sources line")

        b = start("b")
        offer = say(b, prompts["gap"])
        if "Sources:" in offer:
            raise _Stop("the gap question was answered from the knowledge base, not escalated; stopped before any ticket step")
        if "ticket" not in offer.lower():
            raise _Stop("the gap question got no ticket offer")
        if SUMMARY_MARK not in say(b, OFFER_ACCEPT):
            raise _Stop("no ticket summary after accepting the offer")
        created = say(b, CONFIRM)  # the only confirmation; never re-sent
        number = TICKET.search(created)
        if not number:
            raise _Stop("no ticket number in the reply to the confirmation")
        outcome.ticket_number = int(number.group(1))
        link = LINK.search(created)
        outcome.ticket_link = link.group() if link else None
        if require_link and not link:
            raise _Stop(f"ticket #{outcome.ticket_number} was created but its reply has no GitHub link")

        c = start("c", b.thread)  # the same session: the email is already known
        if f"#{outcome.ticket_number}" not in say(c, STATUS):
            outcome.notes.append("conversation c: the status reply does not show the new ticket")

        d = start("d")
        recalled = say(d, RECALL)
        if "email" in recalled.lower() and f"#{outcome.ticket_number}" not in recalled:
            recalled = say(d, EMAIL)
        if f"#{outcome.ticket_number}" not in recalled:
            outcome.notes.append("conversation d: the new session did not recall the ticket")
    except _Stop as stop:
        outcome.stopped = str(stop)
    return outcome


def fence(text: str) -> str:
    """A code fence longer than any run of backticks in `text`, so the reply is shown exactly as it is."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _block(text: str) -> str:
    marks = fence(text)
    return f"{marks}text\n{text}\n{marks}"


def render_markdown(outcome: Outcome, meta: dict[str, str]) -> str:
    lines = [
        "# Example conversations",
        "",
        f"Recorded {meta['date']} with the real service (the same `ChatService` the UI uses): LLM {meta['llm']}, tickets {meta['tickets']}. "
        f"One fake email, `{EMAIL}`. Spend by the ledger: {meta['cost']}. The replies are exactly as returned; the lines in italics "
        "are the agent steps from the debug panel.",
        "",
    ]
    if outcome.ticket_number is not None:
        link = f" ({outcome.ticket_link})" if outcome.ticket_link else ""
        lines += [f"Ticket created in conversation B: **#{outcome.ticket_number}**{link}.", ""]
    for conversation in outcome.conversations:
        lines += [f"## {TITLES[conversation.name]}", ""]
        for turn in conversation.turns:
            lines += ["**Employee:**", "", _block(turn.user), "", "**SmartDesk AI:**", "", _block(turn.reply), ""]
            if turn.steps:
                lines += ["*Steps: " + " → ".join(turn.steps) + "*", ""]
    return "\n".join(lines).rstrip() + "\n"


def run_mock() -> Outcome:
    from smartdesk_ai.ops.health import HealthRegistry
    from smartdesk_ai.ui.context import build_ui_context

    with tempfile.TemporaryDirectory(prefix="smartdesk-examples-") as tmp:
        work = Path(tmp)
        os.environ["SMARTDESK_LEDGER_PATH"] = str(work / "ledger.jsonl")
        environ = {"SMARTDESK_LLM_MODE": "mock", "SMARTDESK_TICKETS_MODE": "mock", "SMARTDESK_DATA_DIR": str(work / "data"),
                   "SMARTDESK_LEDGER_PATH": str(work / "ledger.jsonl")}
        context = build_ui_context(environ=environ, dotenv_path=None, registry=HealthRegistry())
        return run_examples(context.service, prompts={"gap": "How do I register my tractor for the office parking lot?"},
                            require_link=False)


def run_live(out: Path) -> int:
    from evals.live_guard import Estimate, LiveRun, describe
    from smartdesk_ai.config.settings import DEFAULT_CONFIG, load_settings
    from smartdesk_ai.ops.health import HealthRegistry
    from smartdesk_ai.ui.context import build_ui_context

    environ = live_environ()
    settings = load_settings(DEFAULT_CONFIG, environ=environ)
    with tempfile.TemporaryDirectory(prefix="smartdesk-examples-") as tmp:
        environ["SMARTDESK_DATA_DIR"] = str(Path(tmp) / "data")
        with LiveRun(settings, sys.argv, ceiling_usd=CEILING_USD) as run:
            context = build_ui_context(environ=environ, registry=HealthRegistry())  # .env supplies the keys; the environ above wins
            outcome = run_examples(context.service, stop_check=run.stop_reason, on_turn=run.beat)
            cost = run.cost()
    meta = {"date": date.today().isoformat(), "llm": context.settings.modes.llm, "tickets": context.settings.modes.tickets,
            "cost": f"${cost.usd:.4f} over {cost.calls} model calls"}
    target = out if outcome.stopped is None else out.with_suffix(".partial.md")
    target.write_text(render_markdown(outcome, meta))
    print(f"{outcome.turn_count} turns, {meta['cost']} (ceiling ${CEILING_USD:.2f}); transcript: {target}")
    if outcome.ticket_number is not None:
        print(f"ticket #{outcome.ticket_number} {outcome.ticket_link or ''}")
    for note in outcome.notes:
        print(f"note: {note}")
    if outcome.stopped:
        print(f"STOPPED: {outcome.stopped}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record the four example conversations (mock by default)")
    parser.add_argument("--live", action="store_true", help="real models and a real GitHub issue; prints the estimate first")
    parser.add_argument("--confirm-spend", action="store_true", help="with --live: actually run it")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where the live transcript is written")
    args = parser.parse_args(argv)
    if not args.live:
        outcome = run_mock()
        print(render_markdown(outcome, {"date": date.today().isoformat(), "llm": "mock", "tickets": "mock", "cost": "n/a"}))
        return 1 if outcome.stopped else 0
    print(f"Estimate for the live examples: about ${ESTIMATE_USD:.2f} (up to {MAX_TURNS} turns, one GitHub issue, free Moderation calls).")
    print(f"The run stops once the ledger shows more than ${CEILING_USD:.2f} spent (the ceiling, 2x the estimate) or when it makes no progress.")
    if not args.confirm_spend:
        print("Not run: add --confirm-spend to run it.")
        return 2
    return run_live(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
