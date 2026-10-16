"""Shared fixtures: a real ingest of the real KB, built once per session with mock embeddings."""
import pytest

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
