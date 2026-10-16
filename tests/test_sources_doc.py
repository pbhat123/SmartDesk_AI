"""T1.5: kb/SOURCES.md lists a source and licence per adapted item (FR-A10)."""
import re

from tests.kb_helpers import KB_DIR, load_all_entries

DOC = KB_DIR / "SOURCES.md"
KNOWN_LICENCES = re.compile(r"\b(MIT|Apache|CC[- ]BY|CC0|BSD)\b")


def table_rows(text):
    rows = []
    for line in text.splitlines():
        if line.startswith("|") and not re.match(r"^\|[\s:|-]+\|$", line):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def test_sources_doc_exists():
    assert DOC.exists()


def test_each_adapted_item_has_source_and_licence():
    text = DOC.read_text()
    if re.search(r"no public (source|dataset)s? (was|were) used", text, re.IGNORECASE):
        return
    rows = [r for r in table_rows(text) if r[0].startswith(("HR-", "IT-"))]
    assert rows, "no adapted-item rows found"
    ids = {e.id for e in load_all_entries()}
    for entry_id, source, licence, *_ in rows:
        assert entry_id in ids, f"{entry_id} is not a KB entry"
        assert "hr-policies-qa-dataset" in source
        assert KNOWN_LICENCES.search(licence), f"{entry_id}: licence missing"


def test_both_licence_statements_are_recorded():
    text = DOC.read_text()
    assert "license: mit" in text.lower() or "MIT (dataset metadata)" in text
    assert "Open for research and educational use" in text


def test_states_how_the_rest_of_the_kb_was_produced():
    text = DOC.read_text().lower()
    assert "original" in text and "not generated" in text and "openai" in text
