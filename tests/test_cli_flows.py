"""The CLI runs the whole agent graph in mock mode: KB answers, the ticket offer, the email, the
summary, the approval gate and the created ticket (FR-A12, FR-B2, FR-D8). No keys, no network."""
from pathlib import Path

from smartdesk_ai.agent.__main__ import main
from smartdesk_ai.ops.health import HealthRegistry

NO_FILE = Path("/nonexistent/.env")


def chat(lines, tmp_path, argv=()):
    feed = iter([*lines, "quit"])
    out: list[str] = []
    main(list(argv), input_fn=lambda _: next(feed), output=out.append, environ={}, dotenv_path=NO_FILE,
         data_dir=tmp_path, registry=HealthRegistry())
    return "\n".join(out)


def test_a_kb_question_is_answered_with_sources(tmp_path):
    out = chat(["How do I reset my forgotten password?"], tmp_path)
    assert "Sources:" in out


def test_flow_b_from_a_gap_to_a_created_ticket(tmp_path):
    out = chat(
        ["My monitor keeps flickering", "priya.k@smartdesk.example", "yes please"], tmp_path
    )
    assert "don't know" in out and "work email" in out
    assert "Title: My monitor keeps flickering" in out and "Category: IT" in out
    assert "Shall I go ahead and create this?" in out
    assert "Your ticket has been created. Ticket #5" in out and "ask me for updates" in out


def test_declining_at_the_gate_creates_nothing(tmp_path):
    out = chat(["Please raise a ticket for my broken VPN", "priya.k@smartdesk.example", "No, cancel that"], tmp_path)
    assert "won't create the ticket" in out and "has been created" not in out


def test_editing_the_priority_at_the_gate(tmp_path):
    out = chat(["Please raise a ticket for my broken VPN", "priya.k@smartdesk.example", "Make the priority high",
                "yes"], tmp_path)
    assert "Priority: High" in out and "has been created" in out


def test_small_talk_and_deflection(tmp_path):
    out = chat(["Hi there", "What's the weather like today?"], tmp_path)
    assert "Hello!" in out and "can't help with that" in out and "ticket" not in out.split("can't help")[1]


def test_debug_shows_the_trace_chunks_and_answer(tmp_path):
    out = chat(["How do I reset my forgotten password?"], tmp_path, argv=["--debug"])
    assert "[debug] 1. " in out and "[debug] answer:" in out and "[debug] steps=" in out
