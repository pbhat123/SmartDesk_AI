"""T1.10: category eval set, IT / HR / MISC (FR-B7, FR-D7)."""
import re
from collections import Counter

import pytest

from evals.common import EVALS_DIR, load_jsonl
from evals.schemas import CategoryRow

PATH = EVALS_DIR / "categories.jsonl"
FACILITIES = re.compile(
    r"\bpark(ing)?\b|cafeteria|canteen|desk|air conditioning|aircon|temperature|lift|elevator|"
    r"washroom|restroom|cleaning|chair|water|reception|building|lights|fire drill",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def rows():
    return [CategoryRow(**r) for r in load_jsonl(PATH)]


def test_schema_valid_unique_and_large_enough(rows):
    assert len(rows) >= 30
    assert len({r.id for r in rows}) == len(rows)
    assert len({r.utterance.lower() for r in rows}) == len(rows)


def test_all_three_categories_present_in_useful_numbers(rows):
    counts = Counter(r.expected_category for r in rows)
    assert set(counts) == {"IT", "HR", "MISC"}
    assert all(n >= 8 for n in counts.values())


def test_misc_rows_are_facilities_style(rows):
    for r in rows:
        if r.expected_category == "MISC":
            assert FACILITIES.search(r.utterance), f"{r.id}: MISC row is not facilities-style"


def test_non_misc_rows_are_not_facilities_style(rows):
    for r in rows:
        if r.expected_category != "MISC":
            assert not FACILITIES.search(r.utterance), f"{r.id}: facilities wording under {r.expected_category}"


def test_includes_boundary_cases(rows):
    text = " ".join(r.utterance.lower() for r in rows if r.expected_category == "IT")
    assert "laptop" in text  # equipment allocation is IT (FR-A8)
    hr = " ".join(r.utterance.lower() for r in rows if r.expected_category == "HR")
    assert "onboarding" in hr or "first day" in hr or "buddy" in hr  # rest of onboarding is HR
