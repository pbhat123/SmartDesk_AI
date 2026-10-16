"""Shared loader for KB content tests (T1.2 onward)."""
from pathlib import Path

from smartdesk_ai.rag.kb_schema import Entry, parse_entries

KB_DIR = Path(__file__).resolve().parent.parent / "kb"


def load_all_entries() -> list[Entry]:
    entries: list[Entry] = []
    for path in sorted(KB_DIR.glob("*.md")):
        if path.name in {"SOURCES.md", "GAPS.md"}:
            continue
        entries.extend(parse_entries(path.read_text(), source=path.name))
    return entries
