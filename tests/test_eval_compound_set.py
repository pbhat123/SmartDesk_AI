"""T1.11: compound eval set, data only (FR-I4 to FR-I7)."""
import pytest

from evals.common import EVALS_DIR, load_jsonl
from evals.schemas import CompoundRow
from tests.kb_helpers import KB_DIR, load_all_entries

PATH = EVALS_DIR / "compound_questions.jsonl"
ENTRIES = {e.id: e for e in load_all_entries()}


def gaps_doc():
    gaps, current = {}, None
    for line in (KB_DIR / "GAPS.md").read_text().splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
        elif current and line.startswith("Expected category:"):
            gaps[current] = line.split(":", 1)[1].strip()
    return gaps


@pytest.fixture(scope="module")
def rows():
    return [CompoundRow(**r) for r in load_jsonl(PATH)]


def has_gap(row):
    return any(not p.answerable for p in row.parts)


def misroutable(row):
    return any(p.likely_wrong_domain for p in row.parts)


def test_at_least_10_unique_rows(rows):
    assert len(rows) >= 10
    assert len({r.id for r in rows}) == len(rows)
    assert len({r.question.lower() for r in rows}) == len(rows)


def test_some_rows_have_a_gap_part_and_some_are_misroutable(rows):
    assert sum(has_gap(r) for r in rows) >= 2
    assert sum(misroutable(r) for r in rows) >= 2


def test_some_rows_span_it_and_hr_with_every_part_answerable(rows):
    both = [
        r for r in rows
        if not has_gap(r) and {"IT", "HR"} <= {p.expected_domain for p in r.parts}
    ]
    assert len(both) >= 3


def test_answerable_parts_point_at_real_entries_in_the_tagged_domain(rows):
    for r in rows:
        for p in r.parts:
            if p.answerable:
                assert p.expected_entry_id in ENTRIES, r.id
                assert p.gap_topic is None
                assert ENTRIES[p.expected_entry_id].domain == p.expected_domain, (r.id, p.sub_question)


def test_gap_parts_name_a_documented_gap_with_matching_category(rows):
    doc = gaps_doc()
    for r in rows:
        for p in r.parts:
            if not p.answerable:
                assert p.expected_entry_id is None
                assert p.gap_topic in doc, (r.id, p.gap_topic)
                assert p.expected_domain == doc[p.gap_topic], (r.id, p.gap_topic)


def test_misroute_tag_differs_from_the_true_domain(rows):
    for r in rows:
        for p in r.parts:
            if p.likely_wrong_domain:
                assert p.likely_wrong_domain != p.expected_domain
                assert p.answerable, "misroute cases need an answerable part to prove the whole-KB retry"


def test_no_gap_row_is_entirely_gaps(rows):
    assert all(any(p.answerable for p in r.parts) for r in rows)
