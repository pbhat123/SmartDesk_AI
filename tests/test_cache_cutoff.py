"""The semantic cache cut-off sits well above every different-answer pair in the cached embeddings (no API call)."""
import json

import pytest

from evals.cache_cutoff import analyse
from evals.harness import EMBEDDING_CACHE

pytestmark = pytest.mark.skipif(not EMBEDDING_CACHE.is_file(), reason="no embedding cache in this checkout")


def test_the_configured_cut_off_is_safely_above_the_highest_different_answer_pair(settings):
    result = analyse(settings.models.embedding.id)
    if result["different_answer_pairs"] == 0:
        pytest.skip("the questions are not in the embedding cache")
    highest = result["highest_different_answer"][0]["cosine"]
    assert settings.thresholds.semantic_cache_cutoff >= highest + 0.10, highest


def test_the_paraphrase_set_is_15_pairs_over_real_kb_entries():
    import json

    from evals.common import EVALS_DIR

    pairs = [json.loads(line) for line in (EVALS_DIR / "paraphrase_pairs.jsonl").read_text().splitlines()]
    entries = {json.loads(line)["expected_entry_id"] for line in (EVALS_DIR / "kb_questions.jsonl").read_text().splitlines()}
    assert len(pairs) == 15 and all(p["entry_id"] in entries and p["a"] != p["b"] for p in pairs)


def test_the_recommendation_keeps_the_margin_and_picks_the_highest_cut_off_that_loses_no_hit():
    from evals.cache_cutoff import recommend

    rec = recommend(0.734, [0.60, 0.85, 0.92, 0.93])
    assert rec["floor"] == 0.84 and rec["recommended"] == 0.85 and rec["paraphrase_hits_at_recommended"] == 3
