"""T1.8: gap eval set, questions the KB must not answer (FR-A2, FR-D9, F-17)."""
import re
from collections import Counter

import pytest

from evals.common import EVALS_DIR, load_jsonl, longest_shared_span
from evals.schemas import GapQuestion
from tests.kb_helpers import KB_DIR, load_all_entries

PATH = EVALS_DIR / "gap_questions.jsonl"
CORPUS = [f"{e.heading}\n{e.title}\n{e.body}" for e in load_all_entries()]
REQUIRED_TOPICS = {"Monitor flickering", "Office parking", "Cafeteria menu", "Desk booking"}


def gaps_doc():
    """{gap title: expected category} parsed from kb/GAPS.md."""
    gaps, current = {}, None
    for line in (KB_DIR / "GAPS.md").read_text().splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
        elif current and line.startswith("Expected category:"):
            gaps[current] = line.split(":", 1)[1].strip()
    return gaps


@pytest.fixture(scope="module")
def rows():
    return [GapQuestion(**r) for r in load_jsonl(PATH)]


def test_at_least_15_unique_rows(rows):
    assert len(rows) >= 15
    assert len({r.id for r in rows}) == len(rows)
    assert len({r.question.lower() for r in rows}) == len(rows)


def test_required_gap_topics_are_covered(rows):
    assert REQUIRED_TOPICS <= {r.gap_topic for r in rows}


def test_every_gap_topic_exists_in_gaps_doc_with_matching_category(rows):
    doc = gaps_doc()
    for r in rows:
        assert r.gap_topic in doc, f"{r.id}: unknown gap topic {r.gap_topic}"
        assert r.expected_category == doc[r.gap_topic], r.id


def test_every_gaps_doc_topic_has_at_least_one_question(rows):
    assert set(gaps_doc()) <= {r.gap_topic for r in rows}


def test_grey_zone_cases_are_included(rows):
    flags = Counter(r.grey_zone for r in rows)
    assert flags[True] >= 5 and flags[False] >= 4


def test_dev_test_split_present_on_both_sides(rows):
    counts = Counter(r.split for r in rows)
    assert set(counts) == {"dev", "test"}
    assert 0.35 <= counts["dev"] / len(rows) <= 0.65


def test_gap_questions_do_not_copy_kb_text(rows):
    for r in rows:
        assert longest_shared_span(r.question, CORPUS) <= 4, r.id


def test_gap_questions_are_not_kb_questions(rows):
    kb = {re.sub(r"\W+", " ", q["question"].lower()) for q in load_jsonl(EVALS_DIR / "kb_questions.jsonl")}
    assert not any(re.sub(r"\W+", " ", r.question.lower()) in kb for r in rows)
