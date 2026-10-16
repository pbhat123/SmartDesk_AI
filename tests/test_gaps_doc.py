"""T1.6: kb/GAPS.md lists deliberate gaps; no KB entry covers them (FR-A2)."""
import re

from tests.kb_helpers import KB_DIR, load_all_entries

DOC = KB_DIR / "GAPS.md"
REQUIRED = ["monitor flickering", "office parking", "cafeteria menu", "desk booking"]


def parse_gaps():
    """Return {gap title: [keywords]} from '### Title' sections with a 'Keywords:' line."""
    gaps, current = {}, None
    for line in DOC.read_text().splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
            gaps[current] = []
        elif current and line.lower().startswith("keywords:"):
            gaps[current] = [k.strip() for k in line.split(":", 1)[1].split(",") if k.strip()]
    return gaps


def test_gaps_doc_lists_required_gaps_with_keywords():
    gaps = parse_gaps()
    lowered = {g.lower(): kws for g, kws in gaps.items()}
    for name in REQUIRED:
        assert name in lowered, f"missing gap: {name}"
    assert len(gaps) >= 8
    assert all(kws for kws in gaps.values()), "every gap needs keywords"


def test_no_entry_matches_a_gap_keyword():
    gaps = parse_gaps()
    for entry in load_all_entries():
        haystack = f"{entry.topic}\n{entry.title}\n{entry.heading}\n{entry.body}"
        for gap, keywords in gaps.items():
            for kw in keywords:
                hit = re.search(rf"\b{re.escape(kw)}\b", haystack, re.IGNORECASE)
                assert not hit, f"{entry.id} matches gap '{gap}' via '{kw}'"
