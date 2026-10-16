"""T1.7: KB eval set, paraphrased, with dev/test split (FR-A18, FR-A5, F-17)."""
import json
import re
from collections import Counter

import pytest

from evals.common import EVALS_DIR, load_jsonl, longest_shared_span, words
from evals.schemas import KBQuestion
from tests.kb_helpers import KB_DIR, load_all_entries

PATH = EVALS_DIR / "kb_questions.jsonl"
MAX_SHARED_WORDS = 4  # FR-A18: no 5-word span may be copied from indexed text
ENTRIES = {e.id: e for e in load_all_entries()}
CORPUS = [f"{e.heading}\n{e.title}\n{e.body}" for e in ENTRIES.values()]


@pytest.fixture(scope="module")
def rows():
    return [KBQuestion(**r) for r in load_jsonl(PATH)]


def test_at_least_40_rows_with_unique_ids_and_questions(rows):
    assert len(rows) >= 40
    assert len({r.id for r in rows}) == len(rows)
    assert len({r.question.lower() for r in rows}) == len(rows)
    assert all(r.question.endswith("?") for r in rows)


def test_expected_entry_exists_and_domain_matches(rows):
    for r in rows:
        assert r.expected_entry_id in ENTRIES, r.id
        assert r.expected_domain == ENTRIES[r.expected_entry_id].domain, r.id


def test_all_three_question_types_present(rows):
    assert {r.type for r in rows} == {"simple", "how-to", "policy interpretation"}


def test_dev_test_split_is_balanced_and_covers_both_domains(rows):
    counts = Counter(r.split for r in rows)
    assert set(counts) == {"dev", "test"}
    assert 0.4 <= counts["dev"] / len(rows) <= 0.6
    for domain in ("IT", "HR"):
        assert {r.split for r in rows if r.expected_domain == domain} == {"dev", "test"}


def test_no_question_shares_a_long_span_with_indexed_text(rows):
    for r in rows:
        span = longest_shared_span(r.question, CORPUS)
        assert span <= MAX_SHARED_WORDS, f"{r.id}: shares a {span}-word span with the KB"


def test_questions_are_not_copied_from_qa_json(rows):
    qa = {" ".join(words(p["question"])) for p in json.loads((KB_DIR / "qa.json").read_text())}
    for r in rows:
        assert " ".join(words(r.question)) not in qa, r.id


def test_reference_answers_do_not_invent_numbers(rows):
    for r in rows:
        body = ENTRIES[r.expected_entry_id].body
        for num in re.findall(r"\d[\d,]*", r.reference_answer):
            assert num.rstrip(",") in body, f"{r.id}: '{num}' not in {r.expected_entry_id}"
