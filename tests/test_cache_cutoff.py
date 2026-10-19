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


# --- the re-measure on rewritten sub-questions and near-miss pairs (T6.10) ---------------------------------------

def test_the_near_miss_set_is_pairs_of_different_kb_entries_in_the_same_topic_in_fresh_words():
    from evals.common import EVALS_DIR, longest_shared_span

    pairs = [json.loads(line) for line in (EVALS_DIR / "near_miss_pairs.jsonl").read_text().splitlines()]
    kb_text = [p.read_text() for p in (EVALS_DIR.parent / "kb").glob("*.md")]
    ids = {line.split("id: ")[1].strip() for text in kb_text for line in text.splitlines() if line.startswith("id: ")}
    assert len(pairs) >= 16 and len({p["id"] for p in pairs}) == len(pairs)
    for p in pairs:
        assert p["entry_a"] in ids and p["entry_b"] in ids and p["entry_a"] != p["entry_b"], p["id"]
        assert p["entry_a"].rsplit("-", 1)[0] == p["entry_b"].rsplit("-", 1)[0] or p["entry_a"].split("-")[1] == p["entry_b"].split("-")[1], p["id"]
        assert p["a"] != p["b"]
        assert longest_shared_span(p["a"], kb_text) < 8 and longest_shared_span(p["b"], kb_text) < 8, p["id"]  # FR-A18


def rows_and_live(parts_by_row):
    from evals.compound_eval import load_rows

    rows = load_rows()
    live = {"rows": [{"id": r.id, "run": {"part_texts": parts_by_row[r.id]}} for r in rows if r.id in parts_by_row]}
    return rows, live


def test_rewrite_pairs_match_each_answerable_part_with_the_orchestrators_wording():
    from evals.cache_cutoff import rewrite_pairs

    rows, live = rows_and_live({"cmp-001": ["How can I reset my login password?", "What is my casual leave allowance?"]})
    pairs, identical = rewrite_pairs(rows, live)
    first = next(r for r in rows if r.id == "cmp-001")
    assert [p["a"] for p in pairs] == [pt.sub_question for pt in first.parts if pt.answerable][:len(pairs)]
    assert all(p["a"] != p["b"] for p in pairs) and len(pairs) + identical == sum(pt.answerable for pt in first.parts)


def test_a_rewrite_identical_to_the_standalone_form_is_counted_not_paired():
    from evals.cache_cutoff import rewrite_pairs

    rows, live = rows_and_live({"cmp-001": [p.sub_question for p in next(r for r in rows_and_live({})[0] if r.id == "cmp-001").parts]})
    pairs, identical = rewrite_pairs(rows, live)
    assert pairs == [] and identical == 2


def test_a_row_whose_part_count_changed_is_skipped():
    from evals.cache_cutoff import rewrite_pairs

    rows, live = rows_and_live({"cmp-001": ["only one part came back"]})
    assert rewrite_pairs(rows, live) == ([], 0)


def test_the_verdict_keeps_the_cut_off_unless_rewriting_closes_the_gap():
    from evals.cache_cutoff import verdict

    keep = verdict(configured=0.95, highest_different=0.80, paraphrases=[0.5, 0.6, 0.7, 0.85])
    assert keep["action"] == "keep" and keep["floor"] == 0.90
    lower = verdict(configured=0.95, highest_different=0.55, paraphrases=[0.5, 0.7, 0.8, 0.93])
    assert lower["action"].startswith("re-tune") and lower["recommended"] < 0.95
    unsafe = verdict(configured=0.85, highest_different=0.80, paraphrases=[0.9])
    assert unsafe["action"].startswith("raise") and unsafe["floor"] == 0.90


def test_the_near_miss_pairs_raise_the_highest_different_answer_cosine_when_they_are_closer():
    import numpy as np

    from evals.cache_cutoff import near_miss_cosines

    vecs = {"x": np.array([1.0, 0.0]), "y": np.array([0.6, 0.8]), "z": np.array([0.0, 1.0])}
    vec = lambda q: vecs[q]  # noqa: E731
    got = near_miss_cosines(vec, [{"id": "n1", "a": "x", "b": "y"}, {"id": "n2", "a": "x", "b": "z"}])
    assert [g["id"] for g in got] == ["n1", "n2"] and got[0]["cosine"] == pytest.approx(0.6) and got[1]["cosine"] == pytest.approx(0.0)


def test_embedding_needs_confirm_spend_and_the_estimate_is_tiny(capsys, monkeypatch):
    from evals import cache_cutoff as cc

    monkeypatch.setattr(cc, "paraphrase_vectors", lambda *a, **k: (_ for _ in ()).throw(AssertionError("embedded")))
    assert cc.main(["--embed-new"]) == 2 and "--confirm-spend" in capsys.readouterr().out
    assert cc.main(["--embed-new", "--estimate"]) == 0
    out = capsys.readouterr().out
    assert "Estimate for the paraphrase re-measure" in out
    assert cc.estimate_embedding(cc.load_settings()).usd == 0.01


def test_without_embed_new_the_new_sets_are_used_only_when_cached(capsys, tmp_path, monkeypatch):
    from evals import cache_cutoff as cc

    monkeypatch.setattr(cc, "_load_live_rows", lambda: {})
    assert cc.main(["--out-dir", str(tmp_path)]) == 0
    saved = json.loads((tmp_path / "cache_cutoff.json").read_text())
    assert saved["verdict"]["action"] in {"keep"} or saved["verdict"]["action"].startswith(("raise", "re-tune"))
    assert saved["rewrite_pairs"]["n"] == 0 and "near_miss_cosines" in saved
