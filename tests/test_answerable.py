"""T2.8: answerable signals 1 and 2, tied to the score kind (FR-A15, FR-A5)."""
import pytest

from smartdesk_ai.rag.answerable import assess
from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.index import open_index
from smartdesk_ai.rag.pipeline import retrieve
from smartdesk_ai.rag.rerank import CrossEncoderReranker


@pytest.fixture
def th(settings):
    """escalation 0.50 (reranker scale) and fallback 0.30 (dense cosine scale)."""
    return settings.thresholds.model_copy(update={"escalation": 0.50, "fallback": 0.30})


def test_zero_chunks_is_not_answerable_and_fires_only_signal_1(th):
    a = assess(0, 0.0, "reranker", th)
    assert a.answerable is False
    assert (a.signals.no_chunks, a.signals.below_threshold) == (True, False)


def test_score_below_the_threshold_is_not_answerable_and_fires_only_signal_2(th):
    a = assess(3, 0.49, "reranker", th)
    assert a.answerable is False
    assert (a.signals.no_chunks, a.signals.below_threshold) == (False, True)


def test_score_at_or_above_the_threshold_is_answerable(th):
    assert assess(3, 0.50, "reranker", th).answerable is True
    assert assess(3, 0.93, "reranker", th).answerable is True
    assert not any(assess(3, 0.93, "reranker", th).signals.model_dump().values())


def test_reranker_scores_use_the_escalation_threshold(th):
    a = assess(3, 0.45, "reranker", th)
    assert a.threshold == 0.50 and a.answerable is False


def test_dense_cosine_scores_use_the_fallback_threshold(th):
    a = assess(3, 0.45, "dense_cosine", th)
    assert a.threshold == 0.30 and a.answerable is True
    assert assess(3, 0.25, "dense_cosine", th).answerable is False


def test_the_same_score_can_be_answerable_on_one_scale_and_not_the_other(th):
    assert assess(2, 0.40, "reranker", th).answerable is False
    assert assess(2, 0.40, "dense_cosine", th).answerable is True


def test_the_assessment_records_the_score_kind(th):
    assert assess(1, 0.9, "dense_cosine", th).score_kind == "dense_cosine"


# --- wired into retrieve ---------------------------------------------------------------

@pytest.fixture(scope="module")
def index(built_index_dir):
    return open_index(built_index_dir, HashingEmbeddings())


def scorer(logit):
    return CrossEncoderReranker(lambda pairs: [logit] * len(pairs))


def test_retrieve_marks_confident_results_answerable(index, settings):
    r = retrieve(index, "reset my forgotten password", cfg=settings, reranker=scorer(4.0))
    assert r.answerable is True and r.top_score > settings.thresholds.escalation
    assert r.threshold == settings.thresholds.escalation


def test_retrieve_marks_weak_results_unanswerable_by_signal_2(index, settings):
    r = retrieve(index, "zzqx blorptastic", cfg=settings, reranker=scorer(-4.0))
    assert r.entries and r.answerable is False
    assert r.signals.below_threshold and not r.signals.no_chunks


def test_retrieve_marks_empty_results_unanswerable_by_signal_1(index, settings):
    r = retrieve(index, "", cfg=settings, reranker=scorer(4.0))
    assert r.entries == [] and r.answerable is False and r.signals.no_chunks


def test_fallback_path_applies_the_fallback_threshold(index, settings):
    loose = settings.model_copy(update={"thresholds": settings.thresholds.model_copy(update={"fallback": 0.01})})
    strict = settings.model_copy(update={"thresholds": settings.thresholds.model_copy(update={"fallback": 0.99})})
    q = "reset my forgotten password"
    a = retrieve(index, q, cfg=loose, reranker=None)
    b = retrieve(index, q, cfg=strict, reranker=None)
    assert a.score_kind == b.score_kind == "dense_cosine"
    assert a.threshold == 0.01 and b.threshold == 0.99
    assert a.answerable is True and b.answerable is False
    assert a.top_score == b.top_score
