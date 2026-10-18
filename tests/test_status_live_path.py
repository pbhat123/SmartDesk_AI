"""Live status lookups read the same ticket map file as the seed script (data/app.db), and mock ones never
touch it (FR-C8, FR-C10). Nothing here reaches GitHub: building the live client makes no request."""
from pathlib import Path

from smartdesk_ai.agent.__main__ import main
from smartdesk_ai.agent.factory import DEFAULT_DATA_DIR, TICKET_MAP_DB, build_responder
from smartdesk_ai.ops.health import HealthRegistry
from tests.seed_helper import seed_module

NO_FILE = Path("/nonexistent/.env")


def test_the_live_map_is_the_seed_scripts_map():
    assert seed_module.DEFAULT_DB == DEFAULT_DATA_DIR / TICKET_MAP_DB


def test_live_tickets_open_the_map_file_in_the_data_dir(settings, tmp_path):
    live = settings.model_copy(update={"modes": settings.modes.model_copy(update={"tickets": "live"})})
    build_responder(live, environ={"GITHUB_TOKEN": "ghp_not_real", "TICKETS_REPO": "octo/tickets"},
                    dotenv_path=NO_FILE, data_dir=tmp_path, registry=HealthRegistry())
    assert (tmp_path / TICKET_MAP_DB).exists()


def test_mock_tickets_leave_no_map_file(settings, tmp_path):
    build_responder(settings, environ={}, dotenv_path=NO_FILE, data_dir=tmp_path, registry=HealthRegistry())
    assert not (tmp_path / TICKET_MAP_DB).exists()


def test_the_cli_shows_a_ticket_created_in_the_same_session(tmp_path):
    feed = iter(["My monitor keeps flickering", "priya.k@smartdesk.example", "yes please",
                 "What is the status of my ticket?", "quit"])
    out: list[str] = []
    main([], input_fn=lambda _: next(feed), output=out.append, environ={}, dotenv_path=NO_FILE,
         data_dir=tmp_path, registry=HealthRegistry())
    text = "\n".join(out)
    assert "Your ticket has been created. Ticket #5" in text  # after the four preloaded demo tickets
    assert "Ticket #5 (mock): My monitor keeps flickering" in text and "Status: Open" in text
    assert "No updates from the support team yet." in text
