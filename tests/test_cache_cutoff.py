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
