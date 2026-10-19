"""T6.10: semantic cache with a config cut-off; each semantic lookup costs one counted embedding
(SPEC section 4 cache AC, FR-H7, FR-H16). Mock models and fake embedders; no network."""
import json
import math
from types import SimpleNamespace

import pytest

from smartdesk_ai.agent.state import ChunkRef, PartResult
from smartdesk_ai.ops import spend_ledger as sl
from smartdesk_ai.ops.cache import AnswerCache
from smartdesk_ai.ops.cache_log import CacheLog
from smartdesk_ai.ops.cost import CostedEmbeddings, CostTracker
from smartdesk_ai.ops.runtime import RequestContext, Services, request_scope
from tests.ui_helpers import mock_context

CUTOFF = 0.95
BASE = "How do I reset my forgotten password?"


def at(degrees: float) -> list[float]:
    """A unit vector `degrees` away from [1, 0]."""
    return [math.cos(math.radians(degrees)), math.sin(math.radians(degrees))]


def unit(cosine_to_base: float) -> list[float]:
    """A unit vector whose cosine to [1, 0] is `cosine_to_base`."""
    return [cosine_to_base, math.sqrt(1 - cosine_to_base ** 2)]


class FakeEmbedder:
    """One vector per known text; records every call (each is one paid embedding in live mode)."""

    def __init__(self, table):
        self.table, self.calls = table, []

    def embed_query(self, text):
        self.calls.append(text)
        return self.table[text]


TABLE = {BASE: [1.0, 0.0], "Close paraphrase": unit(0.97), "Just above": unit(0.9501), "Just below": unit(0.9499),
         "Far away": unit(0.80), "Opposite": [-1.0, 0.0]}


def answered() -> PartResult:
    return PartResult(answer="Use the self-service page.\n\nSources: IT-1 (Password)", citations=["IT-1"],
                      chunks=[ChunkRef(entry_id="IT-1", title="Password", score=0.8)], answerable=True)


@pytest.fixture
def make(new_backend):
    def build(embedder=None, version="v1", cutoff=CUTOFF, **kwargs):
        embedder = embedder or FakeEmbedder(TABLE)
        cache = AnswerCache(new_backend(50), kb_version=lambda: version, embed=embedder.embed_query, cutoff=cutoff,
                            **kwargs)
        cache.store(cache.lookup(BASE, "IT"), answered(), compute_ms=200.0)
        embedder.calls.clear()  # the set-up lookup's embedding is not under test
        return cache, embedder

    return build


# --- the cut-off -----------------------------------------------------------------------------------


def test_a_paraphrase_above_the_cut_off_is_a_semantic_hit(make):
    cache, _ = make()
    hit = cache.lookup("Close paraphrase", "IT")
    assert hit.kind == "semantic" and hit.part.answer == answered().answer
    assert hit.similarity == pytest.approx(0.97) and 0 < hit.saved_ms <= 200.0


@pytest.mark.parametrize("text, kind", [("Just above", "semantic"), ("Just below", "miss"), ("Far away", "miss"),
                                         ("Opposite", "miss")])
def test_the_cut_off_decides_between_a_hit_and_a_miss(make, text, kind):
    cache, _ = make()
    assert cache.lookup(text, "IT").kind == kind


def test_the_cut_off_is_a_parameter_not_a_constant(make):
    cache, _ = make(cutoff=0.75)
    assert cache.lookup("Far away", "IT").kind == "semantic"
    cache, _ = make(cutoff=0.99)
    assert cache.lookup("Close paraphrase", "IT").kind == "miss"


def test_a_semantic_hit_needs_the_same_domain_and_kb_version(new_backend):
    version = ["v1"]
    embedder = FakeEmbedder(TABLE)
    cache = AnswerCache(new_backend(50), kb_version=lambda: version[0], embed=embedder.embed_query, cutoff=CUTOFF)
    cache.store(cache.lookup(BASE, "IT"), answered(), compute_ms=10.0)
    assert cache.lookup("Close paraphrase", "HR").kind == "miss"
    assert cache.lookup("Close paraphrase", None).kind == "miss"
    version[0] = "v2"
    assert cache.lookup("Close paraphrase", "IT").kind == "miss"


def test_the_best_match_wins(new_backend):
    # BASE at 0 degrees, "Other stored" at 26 (cosine 0.90 to BASE, so it is stored), "Probe" at 16: cosine 0.961 to
    # BASE and 0.985 to "Other stored", both above the cut-off.
    embedder = FakeEmbedder(TABLE | {"Other stored": at(26), "Probe": at(16)})
    cache = AnswerCache(new_backend(50), kb_version=lambda: "v1", embed=embedder.embed_query, cutoff=CUTOFF)
    cache.store(cache.lookup(BASE, None), answered(), compute_ms=10.0)
    other = answered().model_copy(update={"answer": "Second answer.\n\nSources: IT-2 (X)"})
    probe = cache.lookup("Other stored", None)
    assert probe.kind == "miss"
    cache.store(probe, other, compute_ms=10.0)
    hit = cache.lookup("Probe", None)
    assert hit.part.answer == other.answer and hit.similarity == pytest.approx(0.985, abs=1e-3)


# --- one embedding per semantic lookup ---------------------------------------------------------------


def test_each_semantic_lookup_makes_exactly_one_embedding_call_hit_or_miss(make):
    cache, embedder = make()
    cache.lookup("Close paraphrase", "IT")
    cache.lookup("Far away", "IT")
    assert embedder.calls == ["Close paraphrase", "Far away"]


def test_an_exact_hit_makes_no_embedding_call(make):
    cache, embedder = make()
    assert cache.lookup(BASE, "IT").kind == "exact"
    assert embedder.calls == []


def test_storing_reuses_the_lookups_vector_and_makes_no_second_embedding_call(make):
    cache, embedder = make()
    probe = cache.lookup("Far away", "IT")
    cache.store(probe, answered(), compute_ms=10.0)
    assert embedder.calls == ["Far away"]
    assert cache.lookup("Far away", "IT").kind == "exact"  # and it is found again, with no new call
    assert embedder.calls == ["Far away"]


def test_a_bypassed_question_makes_no_embedding_call(make):
    cache, embedder = make()
    assert cache.lookup("Reset the password for [PHONE]", "IT").kind == "bypass" and embedder.calls == []


def test_a_failing_embedder_is_a_plain_miss_and_the_answer_is_still_stored_for_exact_matches(new_backend):
    class Broken:
        def embed_query(self, text):
            raise TimeoutError("down")

    cache = AnswerCache(new_backend(50), kb_version=lambda: "v1", embed=Broken().embed_query, cutoff=CUTOFF)
    probe = cache.lookup(BASE, "IT")
    assert probe.kind == "miss"
    cache.store(probe, answered(), compute_ms=10.0)
    assert cache.lookup(BASE, "IT").kind == "exact"


def test_the_log_records_the_semantic_hit_with_its_similarity(make, tmp_path):
    log = CacheLog(tmp_path / "cache.jsonl")
    cache, _ = make(log=log)
    cache.lookup("Close paraphrase", "IT")
    event = json.loads((tmp_path / "cache.jsonl").read_text().splitlines()[-1])
    assert event["event"] == "hit_semantic" and event["similarity"] == pytest.approx(0.97, abs=1e-3)


# --- the embedding cost is counted -------------------------------------------------------------------


def test_the_embedding_is_counted_by_the_cost_tracker_and_the_spend_ledger(settings, tmp_path, new_backend):
    ledger = tmp_path / "ledger.jsonl"
    sl.start_run(settings, "test", "script", ledger)
    inner = SimpleNamespace(embed_query=lambda t: TABLE[t], name="fake")
    tracker = CostTracker(settings, priced=True)
    embedder = CostedEmbeddings(sl.LedgeredEmbeddings(inner, settings.models.embedding.id), tracker, settings)
    cache = AnswerCache(new_backend(50), kb_version=lambda: "v1", embed=embedder.embed_query, cutoff=CUTOFF)
    ctx = RequestContext(session_id="s", request_id="r")
    with request_scope(ctx, Services(cost=tracker)):
        cache.store(cache.lookup(BASE, "IT"), answered(), compute_ms=10.0)
        assert cache.lookup("Close paraphrase", "IT").kind == "semantic"
    embeddings = [e for e in tracker.entries if e.step == "embedding"]
    assert len(embeddings) == 2 and all(e.usd > 0 and e.input_tokens > 0 for e in embeddings)  # one per lookup
    sl.end_run("test done")
    spent = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert sum(e["kind"] == "embedding" for e in spent if e["event"] == "spend") == 2


# --- in the app ----------------------------------------------------------------------------------------


def test_a_semantic_hit_in_the_mock_app_costs_one_embedding_and_skips_search_worker_and_judge(tmp_path):
    ctx = mock_context(tmp_path)
    first = ctx.service.handle("s1", BASE)
    paraphrase = ctx.service.handle("s2", "How do I reset my forgotten password")  # no question mark: not an exact match
    assert first.debug["cache"]["status"] == "miss"
    cache = paraphrase.debug["cache"]
    assert cache["status"] == "hit" and cache["kind"] == "semantic" and cache["similarity"] >= CUTOFF
    mine = [e for e in ctx.service.services.cost.entries if e.session_id == "s2"]
    assert [e.agent for e in mine if e.step == "embedding"] == ["retrieval"]  # exactly one
    assert not {e.agent for e in mine} & {"worker_answer", "grounding_judge"}
    assert paraphrase.text == first.text and "search_kb" not in paraphrase.debug["steps"]
    assert "cache_hit:semantic" in paraphrase.debug["steps"]


def test_a_different_question_in_the_mock_app_misses(tmp_path):
    ctx = mock_context(tmp_path)
    ctx.service.handle("s1", BASE)
    assert ctx.service.handle("s2", "How many sick days do I get each year?").debug["cache"]["status"] == "miss"


def test_the_app_takes_its_cut_off_from_config(settings, tmp_path):
    from smartdesk_ai.agent.factory import build_stack
    from smartdesk_ai.agent.graph import make_kb_responder

    loose = settings.model_copy(update={"thresholds": settings.thresholds.model_copy(update={"semantic_cache_cutoff": 0.01})})
    asked = ("How do I reset my forgotten password?", "How do I reset my forgotten VPN password?")  # cosine about 0.87

    def second_cache_debug(config, name):
        responder = make_kb_responder(build_stack(config, environ={}, dotenv_path=None, data_dir=tmp_path / name).graph)
        return [responder(f"{name}{i}", text).debug.get("cache", {}) for i, text in enumerate(asked)][1]

    strict = settings.model_copy(update={"thresholds": settings.thresholds.model_copy(update={"semantic_cache_cutoff": 0.95})})
    assert second_cache_debug(strict, "strict").get("status") == "miss"
    assert second_cache_debug(loose, "loose").get("kind") == "semantic"
