"""T9.3: the architecture diagram has one source, docs/architecture.md (Mermaid), and exported images (FR-E, FR-I1)."""
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"


def mermaid_block() -> str:
    match = re.search(r"```mermaid\n(.*?)```", (DOCS / "architecture.md").read_text(), re.S)
    assert match, "no Mermaid block in docs/architecture.md"
    return match.group(1)


def test_the_diagram_names_every_agent_and_the_main_parts():
    block = mermaid_block()
    for name in ("Orchestrator", "IT worker", "HR worker", "Synthesizer", "Ticket agent", "HITL approval gate",
                 "Input guard", "Output guard", "search_kb", "reranker", "GitHub", "Long-term memory", "Monitoring"):
        assert name in block, name


def test_the_images_are_exported_next_to_the_source():
    assert (DOCS / "architecture.svg").read_text().lstrip().startswith(("<?xml", "<svg"))
    assert (DOCS / "architecture.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
