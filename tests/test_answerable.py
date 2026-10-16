"""T2.8: answerable signals 1 and 2 on the top dense cosine, one threshold (FR-A5, FR-A15, SPEC v3.1)."""
import pytest

from smartdesk_ai.rag.answerable import assess
from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.index import open_index
from smartdesk_ai.rag.pipeline import retrieve
from smartdesk_ai.rag.rerank import CrossEncoderReranker


@pytest.fixture
def th(settings):
    return settings.thresholds.model_copy(update={"escalation": 0.50})


def test_zero_chunks_is_not_answerable_and_fires_only_signal_1(th):
    a = assess(0, 0.0, th)
    assert a.answerable is False
    assert (a.signals.no_chunks, a.signals.below_threshold) == (True, False)


def test_a_top_cosine_below_the_threshold_is_not_answerable_and_fires_only_signal_2(th):
    a = assess(3, 0.49, th)
    assert a.answerable is False
    assert (a.signals.no_chunks, a.signals.below_threshold) == (False, True)


def test_a_top_cosine_at_or_above_the_threshold_is_answerable(th):
    assert assess(3, 0.50, th).answerable is True
    assert assess(3, 0.93, th).answerable is True
    assert not any(assess(3, 0.93, th).signals.model_dump().values())


def test_the_assessment_records_the_one_threshold_it_used(th):
    assert assess(3, 0.8, th).threshold == 0.50


def test_there_is_no_second_threshold_in_the_config(settings):
    assert not hasattr(settings.thresholds, "fallback")


# --- wired into retrieve ---------------------------------------------------------------

@pytest.fixture(scope="module")
def index(built_index_dir):
    return open_index(built_index_dir, HashingEmbeddings())


def scorer(logit):
    return CrossEncoderReranker(lambda pairs: [logit] * len(pairs))


def with_threshold(settings, value):
    return settings.model_copy(update={"thresholds": settings.thresholds.model_copy(update={"escalation": value})})


def test_retrieve_applies_the_threshold_to_the_top_dense_cosine(index, settings):
    q = "reset my forgotten password"
    loose = retrieve(index, q, cfg=with_threshold(settings, 0.01), reranker=scorer(0.0))
    strict = retrieve(index, q, cfg=with_threshold(settings, 0.99), reranker=scorer(0.0))
    assert loose.answerable is True and strict.answerable is False
    assert loose.threshold == 0.01 and strict.threshold == 0.99
    assert loose.top_cosine == strict.top_cosine and 0 < loose.top_cosine <= 1


def test_a_confident_reranker_cannot_rescue_a_low_cosine(index, settings):
    r = retrieve(index, "reset my forgotten password", cfg=with_threshold(settings, 0.99), reranker=scorer(20.0))
    assert r.entries and r.entries[0].score > 0.99  # the reranker is sure...
    assert r.answerable is False and r.signals.below_threshold  # ...and it makes no difference


def test_a_pessimistic_reranker_cannot_sink_a_good_cosine(index, settings):
    r = retrieve(index, "reset my forgotten password", cfg=with_threshold(settings, 0.01), reranker=scorer(-20.0))
    assert r.entries[0].score < 0.01 and r.answerable is True


def test_the_answer_does_not_depend_on_whether_the_reranker_is_available(index, settings):
    q = "reset my forgotten password"
    with_rr = retrieve(index, q, cfg=settings, reranker=scorer(2.0))
    without = retrieve(index, q, cfg=settings, reranker=None)
    assert (with_rr.answerable, with_rr.threshold, with_rr.top_cosine) == (without.answerable, without.threshold, without.top_cosine)


def test_retrieve_marks_empty_results_unanswerable_by_signal_1(index, settings):
    r = retrieve(index, "", cfg=settings, reranker=scorer(4.0))
    assert r.entries == [] and r.answerable is False and r.signals.no_chunks and r.top_cosine == 0.0
