"""Batch B: the live-examples driver (README section 14). Mock models and mock tickets only: no network, no spend. It runs the four
example conversations through the same ChatService the UI uses, stops instead of guessing when a conversation does not go to
plan, never sends the confirming "yes" twice, and writes the exact replies to a Markdown file (FR-B2, FR-B13, FR-H11)."""
import importlib.util
import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "live_examples.py"
FAKE = re.compile(r"[\w.+-]+@smartdesk\.example")
ANY_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


@pytest.fixture(scope="module")
def live_examples():
    spec = importlib.util.spec_from_file_location("live_examples", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mock_service(tmp_path):  # one per test: a ticket made by one run would trigger the duplicate question in the next
    work = tmp_path
    import os
    from smartdesk_ai.ops.health import HealthRegistry
    from smartdesk_ai.ui.context import build_ui_context
    os.environ["SMARTDESK_LEDGER_PATH"] = str(work / "ledger.jsonl")
    context = build_ui_context(
        environ={"SMARTDESK_LLM_MODE": "mock", "SMARTDESK_TICKETS_MODE": "mock", "SMARTDESK_DATA_DIR": str(work / "data"),
                 "SMARTDESK_LEDGER_PATH": str(work / "ledger.jsonl")},
        dotenv_path=None, registry=HealthRegistry())
    yield context.service
    os.environ.pop("SMARTDESK_LEDGER_PATH", None)


# The mock knowledge base answers the password question and escalates the tractor-parking question (see scripts/smoke_ui.py).
MOCK_PROMPTS = {"kb": "How do I reset my forgotten password?", "gap": "How do I register my tractor for the office parking lot?"}


def test_four_conversations_run_end_to_end_and_create_exactly_one_ticket(live_examples, mock_service):
    outcome = live_examples.run_examples(mock_service, prompts=MOCK_PROMPTS, require_link=False)
    assert outcome.stopped is None
    assert [c.name for c in outcome.conversations] == ["a", "b", "c", "d"]
    assert outcome.turn_count <= live_examples.MAX_TURNS
    assert outcome.ticket_number is not None
    sent = [turn.user for c in outcome.conversations for turn in c.turns]
    assert sum(1 for text in sent if text == live_examples.CONFIRM) == 1  # one confirming "yes", so one issue
    b = outcome.conversations[1]
    assert [t.user for t in b.turns] == [MOCK_PROMPTS["gap"], live_examples.OFFER_ACCEPT, live_examples.EDIT, live_examples.CONFIRM]
    assert "Priority: High" in b.turns[2].reply  # the edit is shown at the gate before anything is created
    assert outcome.turn_count == 8 == live_examples.MAX_TURNS  # a: 1, b: 4, c: 1, d: 2 (recall, then the email)
    assert f"#{outcome.ticket_number}" in outcome.conversations[2].turns[-1].reply  # the status check shows the new ticket
    assert f"#{outcome.ticket_number}" in outcome.conversations[3].turns[-1].reply  # the new session recalls it


def test_an_edit_the_gate_does_not_show_stops_the_run_before_the_confirm(live_examples, mock_service, monkeypatch):
    monkeypatch.setattr(live_examples, "EDIT", "Please call me back tomorrow")
    outcome = live_examples.run_examples(mock_service, prompts=MOCK_PROMPTS, require_link=False)
    assert outcome.stopped and "priority edit" in outcome.stopped
    sent = [turn.user for c in outcome.conversations for turn in c.turns]
    assert live_examples.CONFIRM not in sent and outcome.ticket_number is None


def test_a_gap_question_the_kb_answers_stops_the_run_before_any_ticket_step(live_examples, mock_service):
    prompts = {**MOCK_PROMPTS, "gap": MOCK_PROMPTS["kb"]}
    outcome = live_examples.run_examples(mock_service, prompts=prompts, require_link=False)
    assert outcome.stopped and "answered from the knowledge base" in outcome.stopped
    sent = [turn.user for c in outcome.conversations for turn in c.turns]
    assert live_examples.CONFIRM not in sent and live_examples.OFFER_ACCEPT not in sent
    assert outcome.ticket_number is None
    assert [c.name for c in outcome.conversations] == ["a", "b"]


def test_a_missing_link_in_live_mode_is_reported_not_papered_over(live_examples, mock_service):
    outcome = live_examples.run_examples(mock_service, prompts=MOCK_PROMPTS, require_link=True)
    assert outcome.stopped and "link" in outcome.stopped  # a mock ticket has no GitHub link
    sent = [turn.user for c in outcome.conversations for turn in c.turns]
    assert sent.count(live_examples.CONFIRM) == 1  # never re-sent


def test_the_turn_cap_stops_a_run(live_examples, mock_service, monkeypatch):
    monkeypatch.setattr(live_examples, "MAX_TURNS", 3)
    outcome = live_examples.run_examples(mock_service, prompts=MOCK_PROMPTS, require_link=False)
    assert outcome.stopped and "turn" in outcome.stopped and outcome.turn_count == 3


def test_the_ceiling_stops_a_run_before_the_next_turn(live_examples, mock_service):
    calls = []
    outcome = live_examples.run_examples(mock_service, prompts=MOCK_PROMPTS, require_link=False,
                                         stop_check=lambda: "over the ceiling" if calls.append(1) or len(calls) > 2 else None)
    assert outcome.stopped == "over the ceiling" and outcome.turn_count == 2


def test_the_markdown_has_the_exact_replies_and_only_the_fake_email(live_examples, mock_service):
    outcome = live_examples.run_examples(mock_service, prompts=MOCK_PROMPTS, require_link=False)
    text = live_examples.render_markdown(outcome, {"date": "2026-10-21", "llm": "mock", "tickets": "mock", "cost": "n/a"})
    for conversation in outcome.conversations:
        for turn in conversation.turns:
            assert turn.reply in text and turn.user in text
    assert set(ANY_EMAIL.findall(text)) <= {live_examples.EMAIL}
    assert FAKE.fullmatch(live_examples.EMAIL)


def test_a_reply_with_backticks_does_not_break_the_fence(live_examples):
    fence = live_examples.fence("run `x` then ```code``` here")
    assert fence.startswith("````") and fence.endswith("````")


def test_the_live_environment_blanks_tracing_and_redis_and_sets_both_modes(live_examples):
    env = live_examples.live_environ()
    assert env["SMARTDESK_LLM_MODE"] == "live" and env["SMARTDESK_TICKETS_MODE"] == "live"
    for key in ("LANGSMITH_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "REDIS_URL"):
        assert env[key] == ""
    assert "LANGSMITH_TRACING" not in env


def test_live_needs_confirm_spend_and_prints_the_estimate_first(live_examples, capsys, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("nothing may be built or called without --confirm-spend")

    monkeypatch.setattr(live_examples, "run_live", boom)
    assert live_examples.main(["--live"]) == 2
    out = capsys.readouterr().out
    assert "$0.03" in out and "$0.06" in out and "--confirm-spend" in out


def test_the_live_data_folder_gets_only_a_copy_of_the_built_index(live_examples, tmp_path):
    source = tmp_path / "data"
    (source / "index" / "chroma").mkdir(parents=True)
    (source / "index" / "state.json").write_text("{}")
    for name in ("app.db", "checkpoints.db", "cache.jsonl", "hitl.jsonl"):
        (source / name).write_text("old")
    target = tmp_path / "fresh"
    live_examples.copy_index(source, target)
    assert (target / "index" / "state.json").is_file() and (target / "index" / "chroma").is_dir()
    assert sorted(p.name for p in target.iterdir()) == ["index"]  # no tickets map, memory, checkpoints or cache from real use
