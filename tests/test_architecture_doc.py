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


def test_the_svg_is_plain_valid_xml_that_github_can_show():
    """GitHub shows an SVG in a README through an image proxy: it must parse as XML, and labels must be SVG text, not HTML
    inside <foreignObject> (the proxy strips those, so the boxes come out empty)."""
    from xml.dom import minidom

    text = (DOCS / "architecture.svg").read_text()
    minidom.parseString(text)  # raises on an unclosed <br> or any other non-XML markup
    assert "foreignObject" not in text and "<script" not in text and "onload" not in text
    labels = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))
    for name in ("Orchestrator", "HITL approval gate", "Tool registry and executor", "Long-term memory", "Output guard"):
        assert name in labels, name
