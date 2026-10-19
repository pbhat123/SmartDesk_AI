"""Single-question parity (FR-I5, PLAN section 8): the multi-agent graph reproduces the single-path app exactly.

The golden file was recorded from the single-path graph before Phase 7 (`tests/golden_helpers.py`). Every
KB, gap and scenario turn must give the same reply, steps, citations and chunks, and the answer model and
the grounding judge must receive byte-identical messages, so the live eval numbers cannot move.
"""
import json

import pytest

from tests.golden_helpers import GOLDEN_FILE, record_golden_pinned

GOLDEN = json.loads(GOLDEN_FILE.read_text())


@pytest.fixture(scope="module")
def now():
    return record_golden_pinned()


def test_golden_covers_every_question_and_scenario():
    assert len(GOLDEN["questions"]) == 74
    assert len(GOLDEN["scenarios"]) == 12


def _same(key, actual, expected):
    skipped = GOLDEN["volatile"].get(key, [])
    assert {k: v for k, v in actual.items() if k not in skipped} == {k: v for k, v in expected.items() if k not in skipped}


@pytest.mark.parametrize("qid", sorted(GOLDEN["questions"]))
def test_question_matches_golden(now, qid):
    _same(qid, now["questions"][qid], GOLDEN["questions"][qid])


@pytest.mark.parametrize("name", sorted(GOLDEN["scenarios"]))
def test_scenario_matches_golden(now, name):
    turns = GOLDEN["scenarios"][name]
    assert len(now["scenarios"][name]) == len(turns)
    for index, expected in enumerate(turns):
        _same(f"{name}#{index}", now["scenarios"][name][index], expected)


def test_only_a_few_fields_are_volatile():
    """The skipped fields are known mock-retrieval ties, not a hole in the comparison."""
    assert len(GOLDEN["volatile"]) <= 3, GOLDEN["volatile"]
    assert all(fields == ["llm_inputs"] for fields in GOLDEN["volatile"].values()), GOLDEN["volatile"]
