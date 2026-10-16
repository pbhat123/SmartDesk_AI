"""T1.4: kb/qa.json Q&A pairs (FR-A1, FR-A10)."""
import json
import re

import pytest

from tests.kb_helpers import KB_DIR, load_all_entries

QA_PATH = KB_DIR / "qa.json"
FIELDS = {"question", "answer", "category", "type", "entry_id"}
TYPES = {"simple", "how-to", "policy interpretation"}
CATEGORY_BY_FILE = {
    "company.md": "HR Policies",
    "it_support.md": "IT Support",
    "onboarding.md": "Onboarding",
    "hr_policies.md": "HR Policies",
    "payroll_comp.md": "Payroll and Comp",
}
ENTRIES = {e.id: e for e in load_all_entries()}


@pytest.fixture(scope="module")
def pairs():
    data = json.loads(QA_PATH.read_text())
    assert isinstance(data, list) and data
    return data


def test_every_pair_has_all_fields_and_nothing_else(pairs):
    for p in pairs:
        assert set(p) == FIELDS, p
        assert all(isinstance(v, str) and v.strip() for v in p.values()), p


def test_type_is_one_of_the_three_kinds_and_all_appear(pairs):
    assert {p["type"] for p in pairs} == TYPES


def test_entry_id_exists_and_category_matches_its_policy_area(pairs):
    for p in pairs:
        assert p["entry_id"] in ENTRIES, p["entry_id"]
        assert p["category"] == CATEGORY_BY_FILE[ENTRIES[p["entry_id"]].source], p["entry_id"]


def test_every_entry_has_at_least_one_pair(pairs):
    assert set(ENTRIES) <= {p["entry_id"] for p in pairs}


def test_questions_are_unique_questions(pairs):
    questions = [p["question"] for p in pairs]
    assert len(set(questions)) == len(questions)
    assert all(q.endswith("?") for q in questions)


def test_answers_do_not_invent_numbers(pairs):
    for p in pairs:
        body = ENTRIES[p["entry_id"]].body
        for num in re.findall(r"\d[\d,]*", p["answer"]):
            num = num.rstrip(",")
            assert num in body, f"{p['entry_id']}: '{num}' not in entry text"


def test_json_is_not_a_markdown_file_so_ingest_skips_it():
    assert QA_PATH.suffix == ".json"
    assert QA_PATH not in set(KB_DIR.glob("*.md"))
