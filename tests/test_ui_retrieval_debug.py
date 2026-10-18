"""UI review fix 3: the retrieval-debug panel shows, per row, the dense cosine, the RRF score and the rerank score
under labelled columns, the escalation threshold in force and whether this turn escalated, and says that mock
retrieval is not meaningful (FR-A12, FR-A15)."""
import dataclasses

import pytest

from evals.harness import lexical_scorer
from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.index import open_index
from smartdesk_ai.rag.pipeline import retrieve
from smartdesk_ai.rag.rerank import CrossEncoderReranker
from tests.ui_helpers import app_test, element_text, mock_context

ASK = "How do I reset my forgotten password?"
OFF_TOPIC = "How do I register my tractor for the office parking lot?"
COLUMNS = ["rank", "entry_id", "title", "dense cosine", "RRF score", "rerank score"]


def with_strategy(settings, name):
    return settings.model_copy(update={"retrieval": settings.retrieval.model_copy(update={"strategy": name})})


@pytest.fixture
def index(built_index_dir):
    return open_index(built_index_dir, HashingEmbeddings())


# --- the pipeline keeps each stage's score for every entry it returns -------------------------------------

def test_hybrid_rerank_entries_carry_all_three_scores(index, settings):
    found = retrieve(index, ASK, cfg=with_strategy(settings, "hybrid_rerank"), reranker=CrossEncoderReranker(lexical_scorer))
    assert found.ordered_by == "rerank"
    for entry in found.entries:
        assert entry.rerank == entry.score and entry.rrf is not None and entry.rrf > 0
    assert any(e.dense is not None for e in found.entries)  # the dense stage found at least the top entry
    assert found.entries[0].dense is not None and -1.0 <= found.entries[0].dense <= 1.0


def test_hybrid_without_a_reranker_has_no_rerank_score(index, settings):
    found = retrieve(index, ASK, cfg=with_strategy(settings, "hybrid"), reranker=None)
    assert all(e.rerank is None and e.rrf == e.score for e in found.entries)


def test_dense_only_has_no_rrf_and_no_rerank_score(index, settings):
    found = retrieve(index, ASK, cfg=with_strategy(settings, "dense"), reranker=None)
    assert all(e.rrf is None and e.rerank is None and e.dense == e.score for e in found.entries)


def test_a_missing_reranker_leaves_the_rerank_column_empty_and_the_order_rrf(index, settings):
    found = retrieve(index, ASK, cfg=with_strategy(settings, "hybrid_rerank"), reranker=None)
    assert found.ordered_by == "rrf" and all(e.rerank is None for e in found.entries)


# --- the panel ----------------------------------------------------------------------------------------

def _ask(ctx, text):
    at = app_test(ctx)
    at.chat_input[0].set_value(text).run()
    return at


def _panel(at):
    (panel,) = [e for e in at.expander if "Retrieval debug" in e.label]
    return panel


def test_the_panel_has_labelled_columns_for_each_stage(tmp_path):
    panel = _panel(_ask(mock_context(tmp_path), ASK))
    rows = panel.dataframe[0].value
    assert list(rows.columns) == COLUMNS
    assert rows["entry_id"].iloc[0] == "IT-PWD-01"
    assert rows["dense cosine"].notna().any() and rows["RRF score"].notna().all()
    assert rows["rerank score"].isna().all()  # mock mode does not run the reranker


def test_the_panel_states_the_threshold_and_that_an_answered_turn_did_not_escalate(tmp_path):
    ctx = mock_context(tmp_path)
    shown = element_text(_panel(_ask(ctx, ASK)))
    assert f"{ctx.settings.thresholds.escalation_mock:.2f}" in shown  # mock mode applies its own threshold
    assert "Escalated this turn: no" in shown and "top dense cosine" in shown.lower()


def test_the_panel_says_when_the_turn_escalated(tmp_path):
    shown = element_text(_panel(_ask(mock_context(tmp_path), OFF_TOPIC)))
    assert "Escalated this turn: yes" in shown


def test_mock_mode_says_its_retrieval_is_not_meaningful(tmp_path):
    shown = element_text(_panel(_ask(mock_context(tmp_path), ASK)))
    assert "fake embeddings" in shown and "not meaningful" in shown


def test_live_mode_does_not_show_the_mock_caption(tmp_path):
    ctx = mock_context(tmp_path)
    live = ctx.settings.model_copy(update={"modes": ctx.settings.modes.model_copy(update={"llm": "live"})})
    shown = element_text(_panel(_ask(dataclasses.replace(ctx, settings=live), ASK)))
    assert "fake embeddings" not in shown and "Escalated this turn" in shown
