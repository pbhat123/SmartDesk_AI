"""Test isolation: the suite must never write to the repo's data/ folder, which holds the live ticket
map (FR-C10). The session fixture in conftest.py enforces it; these tests check the check."""
from pathlib import Path

from smartdesk_ai.agent.factory import DATA_DIR_ENV, DEFAULT_DATA_DIR, resolve_data_dir
from tests.conftest import DATA_DIR, _snapshot


def test_the_guard_watches_the_repos_data_folder():
    assert DATA_DIR == DEFAULT_DATA_DIR and DATA_DIR.name == "data"


def test_a_new_file_a_change_and_a_deletion_are_all_detected(tmp_path):
    (tmp_path / "keep.db").write_bytes(b"1")
    (tmp_path / "gone.db").write_bytes(b"1")
    before = _snapshot(tmp_path)
    (tmp_path / "new.db").write_bytes(b"1")
    assert _snapshot(tmp_path) != before
    (tmp_path / "new.db").unlink()
    assert _snapshot(tmp_path) == before
    (tmp_path / "keep.db").write_bytes(b"22")
    assert _snapshot(tmp_path) != before
    (tmp_path / "keep.db").write_bytes(b"1")
    (tmp_path / "gone.db").unlink()
    assert _snapshot(tmp_path) != before


def test_a_missing_folder_is_an_empty_snapshot(tmp_path):
    assert _snapshot(tmp_path / "nope") == {}


def test_the_data_dir_can_be_moved_with_an_environment_variable(tmp_path):
    assert resolve_data_dir(None, {DATA_DIR_ENV: str(tmp_path)}) == tmp_path
    assert resolve_data_dir(None, {}) == DEFAULT_DATA_DIR
    assert resolve_data_dir(Path("/x"), {DATA_DIR_ENV: str(tmp_path)}) == Path("/x")  # explicit wins


def test_mock_tickets_never_touch_the_live_ticket_map(tmp_path):
    from smartdesk_ai.agent.__main__ import main
    from smartdesk_ai.ops.health import HealthRegistry

    feed = iter(["My monitor keeps flickering", "priya.k@smartdesk.example", "yes", "yes", "quit"])
    main([], input_fn=lambda _: next(feed), output=lambda _: None, environ={}, dotenv_path=Path("/none"),
         data_dir=tmp_path, registry=HealthRegistry())
    assert not (tmp_path / "app.db").exists(), "mock tickets must stay in memory"
