"""Shared fixtures: a real ingest of the real KB, built once per session with mock embeddings."""
import pytest

from smartdesk_ai.rag.ingest import ROOT

DATA_DIR = ROOT / "data"  # holds the live ticket map (app.db), the live index and the embedding cache

from smartdesk_ai.config.settings import load_settings
from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.ingest import ingest
from tests.kb_helpers import KB_DIR


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
