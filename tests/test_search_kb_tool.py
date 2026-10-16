"""T3.8: search_kb tool returns chunks, scores and `answerable` (FR-G7, FR-A15)."""
import pytest

from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.index import open_index
from smartdesk_ai.tools.executor import Outcome, ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry
from smartdesk_ai.tools.search_kb import make_search_kb

IT_QUERY = "how do I reset my forgotten password"


@pytest.fixture
def tool(built_index_dir, settings):
    index = open_index(built_index_dir, HashingEmbeddings())
    return make_search_kb(index, settings, reranker=None, health=HealthRegistry())


@pytest.fixture
def executor(tool, settings):
    registry = ToolRegistry()
    registry.register("search_kb", tool)
    return ToolExecutor(registry, settings)


def call(executor, **args):
    return executor.execute("search_kb", args, agent="it_worker")


def test_result_carries_chunks_scores_and_answerable(executor, settings):
    result = call(executor, query=IT_QUERY)
    assert result.outcome is Outcome.SUCCESS
    data = result.data
    assert 0 < len(data.chunks) <= settings.retrieval.top_n
    assert all(c.text and c.entry_id and isinstance(c.score, float) for c in data.chunks)
    assert data.top_cosine > 0
    assert isinstance(data.answerable, bool)


def test_answerable_follows_signals_1_and_2(executor, settings):
    data = call(executor, query=IT_QUERY).data
    assert data.answerable == (not data.signals.no_chunks and data.top_cosine >= settings.thresholds.escalation)
    assert data.threshold == settings.thresholds.escalation


def test_gibberish_is_not_answerable(executor):
    data = call(executor, query="zzqx wvvk plorf").data
    assert data.answerable is False


@pytest.mark.parametrize("domain", ["IT", "HR"])
def test_domain_filter_is_honoured(executor, domain):
    data = call(executor, query="policy and access", domain=domain).data
    assert data.chunks and {c.domain for c in data.chunks} == {domain}


def test_no_domain_searches_the_whole_kb(executor):
    domains = set()
    for q in ("password reset vpn laptop", "annual leave payroll onboarding benefits"):
        domains |= {c.domain for c in call(executor, query=q).data.chunks}
    assert domains == {"IT", "HR"}


def test_bad_domain_is_a_validation_error(executor):
    assert call(executor, query="x", domain="MISC").outcome is Outcome.VALIDATION_ERROR
