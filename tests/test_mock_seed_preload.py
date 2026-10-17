"""T4.26 follow-up: the mock CLI preloads the four seed tickets (same emails, titles, statuses and team
comments as scripts/seed_tickets.py) into its in-memory client and map, so the status flow can be shown
with no keys. They are never written to data/app.db (FR-C8)."""
from pathlib import Path

import pytest

from smartdesk_ai.agent.__main__ import main
from smartdesk_ai.agent.factory import TICKET_MAP_DB, build_responder
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.tickets import MockTicketClient, find_tickets
from smartdesk_ai.tickets.seed import CLOSED_EMAIL, EMAILS, MANY_EMAIL, ONE_EMAIL, SEEDS, ZERO_EMAIL, seed
from smartdesk_ai.tickets.sqlite_map import TicketStore
from tests.seed_helper import seed_module

NO_FILE = Path("/nonexistent/.env")


def chat(lines, tmp_path):
    feed = iter([*lines, "quit"])
    out: list[str] = []
    main([], input_fn=lambda _: next(feed), output=out.append, environ={}, dotenv_path=NO_FILE,
         data_dir=tmp_path, registry=HealthRegistry())
    return "\n".join(out)


def test_the_script_and_the_package_share_one_seed_definition():
    assert seed_module.SEEDS is SEEDS and seed_module.seed is seed and seed_module.EMAILS == EMAILS


def test_seeding_the_package_way_matches_the_script_way():
    client, store = MockTicketClient(), TicketStore(":memory:")
    seed(client, store)
    assert [t.title for t in client._issues.values()] == [s.title for s in SEEDS]


def test_one_ticket_email_shows_the_seeded_ticket_and_its_team_comment(tmp_path):
    out = chat([f"Any updates on my monitor ticket?", ONE_EMAIL], tmp_path)
    item = next(s for s in SEEDS if s.email == ONE_EMAIL)
    assert f"Ticket #1: {item.title}" in out and "Status: In Progress" in out and f"> {item.comment}" in out


def test_two_tickets_email_gets_the_list_and_can_pick(tmp_path):
    out = chat(["What is the status of my ticket?", MANY_EMAIL, "the second one"], tmp_path)
    vpn = next(s for s in SEEDS if s.key == "vpn-drops")
    assert "I found 2 tickets" in out and "Status: Resolved" in out and f"> {vpn.comment}" in out


def test_the_zero_and_closed_emails(tmp_path):
    assert "no tickets on file" in chat(["What is the status of my ticket?", ZERO_EMAIL], tmp_path)
    closed = chat(["What is the status of my ticket?", CLOSED_EMAIL], tmp_path)
    assert "Status: Closed" in closed and "facilities cabinet" in closed


def test_every_seeded_title_status_and_comment_is_shown_somewhere(tmp_path):
    out = "\n".join(chat(["What is the status of my ticket?", email], tmp_path) for email in (ONE_EMAIL, CLOSED_EMAIL))
    for item in (s for s in SEEDS if s.email in (ONE_EMAIL, CLOSED_EMAIL)):
        assert item.title in out and item.status.value in out and item.comment in out


def test_the_preload_is_in_memory_and_never_written_to_the_map_file(tmp_path):
    chat(["What is the status of my ticket?", ONE_EMAIL], tmp_path)
    assert not (tmp_path / TICKET_MAP_DB).exists()


def test_a_new_ticket_gets_the_next_number_after_the_seeded_ones(tmp_path):
    out = chat(["My monitor keeps flickering", "priya.k@smartdesk.example", "yes please"], tmp_path)
    assert f"Ticket #{len(SEEDS) + 1}" in out


def test_the_banner_names_the_demo_emails(tmp_path):
    out = chat([], tmp_path)
    assert "[demo]" in out and all(email in out for email in (ONE_EMAIL, MANY_EMAIL, ZERO_EMAIL, CLOSED_EMAIL))


def test_live_tickets_are_never_preloaded(settings, tmp_path, monkeypatch):
    import smartdesk_ai.agent.factory as factory

    monkeypatch.setattr(factory, "seed", lambda *a, **k: pytest.fail("seed must not run for live tickets"))
    live = settings.model_copy(update={"modes": settings.modes.model_copy(update={"tickets": "live"})})
    build_responder(live, environ={"GITHUB_TOKEN": "ghp_not_real", "TICKETS_REPO": "octo/tickets"},
                    dotenv_path=NO_FILE, data_dir=tmp_path, registry=HealthRegistry())


def test_the_seed_emails_are_fake_addresses():
    assert all(e.endswith("@smartdesk.example") for e in EMAILS)
