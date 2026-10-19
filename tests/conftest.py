"""Shared fixtures: a real ingest of the real KB, built once per session with mock embeddings."""
import os

import pytest

from smartdesk_ai.rag.ingest import ROOT

for _variable in ("SMARTDESK_LLM_MODE", "SMARTDESK_TICKETS_MODE"):  # a developer's exported override must not change the suite
    os.environ.pop(_variable, None)

DATA_DIR = ROOT / "data"  # holds the live ticket map (app.db), the live index and the embedding cache

from smartdesk_ai.config.settings import load_settings
from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.ingest import ingest
from tests.kb_helpers import KB_DIR


@pytest.fixture(autouse=True)
def ledger_never_touches_the_repo(tmp_path, monkeypatch):
    """Any ledger write in a test goes to a temp file, never to the repo's data/spend_ledger.jsonl."""
    from smartdesk_ai.ops import spend_ledger

    monkeypatch.setenv(spend_ledger.LEDGER_ENV, str(tmp_path / "test_ledger.jsonl"))
    spend_ledger.reset_for_tests()
    yield
    spend_ledger.reset_for_tests()


@pytest.fixture(scope="session")
def settings():
    return load_settings()


@pytest.fixture(scope="session")
def built_index_dir(tmp_path_factory, settings):
    index_dir = tmp_path_factory.mktemp("kb_index")
    ingest(KB_DIR, index_dir, HashingEmbeddings(), settings.chunking)
    return index_dir


@pytest.fixture
def kb_copy(tmp_path):
    """A private copy of the real kb/ directory that a test may edit."""
    import shutil

    target = tmp_path / "kb"
    shutil.copytree(KB_DIR, target)
    return target


@pytest.fixture
def fresh_index_dir(tmp_path):
    return tmp_path / "index"


def _snapshot(directory):
    if not directory.exists():
        return {}
    return {
        str(p.relative_to(directory)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(directory.rglob("*")) if p.is_file()
    }


@pytest.fixture(scope="session", autouse=True)
def data_dir_is_untouched():
    """No test may write to the repo's data/ folder. It holds the live ticket map (FR-C10)."""
    before = _snapshot(DATA_DIR)
    yield
    after = _snapshot(DATA_DIR)
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    assert not changed, f"the test suite changed data/: {changed}"


@pytest.fixture(params=["memory", "redis", "real-redis"])
def new_backend(request):
    """A factory for an empty answer-cache backend: the in-memory one, then the Redis one over a fake client, so the
    cache tests run both ways with no server (and, when SMARTDESK_TEST_REDIS_URL is set, over a real Redis too)."""
    from smartdesk_ai.ops.cache import MemoryBackend, RedisBackend
    from tests.fake_redis import FakeRedis

    counter = iter(range(10**6))
    made = []
    real_url = os.environ.get("SMARTDESK_TEST_REDIS_URL")
    if request.param == "real-redis" and not real_url:
        pytest.skip("set SMARTDESK_TEST_REDIS_URL to run the cache tests against a real Redis")

    def make(max_entries: int = 50):
        if request.param == "memory":
            return MemoryBackend(max_entries)
        if request.param == "redis":
            return RedisBackend(FakeRedis(), namespace=f"test{next(counter)}", max_entries=max_entries)
        import redis

        backend = RedisBackend(redis.Redis.from_url(real_url, decode_responses=True),
                               namespace=f"smartdesk-test:{os.getpid()}:{next(counter)}", max_entries=max_entries)
        made.append(backend)
        return backend

    yield make
    for backend in made:
        backend.clear()
