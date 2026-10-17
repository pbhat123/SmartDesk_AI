"""T3.13: SQLite email-to-issue map (FR-C2, FR-C10, FR-B3)."""
import sqlite3

import pytest

from smartdesk_ai.tickets import MockTicketClient
from smartdesk_ai.tickets.sqlite_map import TicketStore
from smartdesk_ai.tools.executor import Outcome, ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry
from smartdesk_ai.tools.ticket_tools import register_ticket_tools

JANE, SAM = "jane.roe@example.com", "sam.lee@example.com"
TICKET = {"title": "VPN down", "description": "Cannot connect", "category": "IT", "email": JANE}


def test_link_stores_email_to_issue_number():
    store = TicketStore(":memory:")
    store.link_email(JANE, 12)
    assert store.numbers_for(JANE) == [12]


def test_lookup_returns_all_issue_numbers_for_an_email_in_order():
    store = TicketStore(":memory:")
    for number in (30, 4, 17):
        store.link_email(JANE, number)
    store.link_email(SAM, 5)
    assert store.numbers_for(JANE) == [4, 17, 30]
    assert store.numbers_for(SAM) == [5]


def test_unknown_email_has_no_numbers():
    assert TicketStore(":memory:").numbers_for("nobody@example.com") == []


def test_linking_twice_keeps_one_row():
    store = TicketStore(":memory:")
    store.link_email(JANE, 3)
    store.link_email(JANE, 3)
    assert store.numbers_for(JANE) == [3]


def test_email_match_ignores_case_and_surrounding_spaces():
    store = TicketStore(":memory:")
    store.link_email("  Jane.Roe@Example.com ", 3)
    assert store.numbers_for("jane.roe@example.COM") == [3]


def test_map_persists_across_reopen(tmp_path):
    path = tmp_path / "app.db"
    first = TicketStore(path)
    first.link_email(JANE, 8)
    first.link_email(JANE, 9)
    assert TicketStore(path).numbers_for(JANE) == [8, 9]


def test_request_log_still_holds_no_email(tmp_path):
    path = tmp_path / "app.db"
    store = TicketStore(path)
    store.begin("req-1")
    store.complete("req-1", 4)
    store.link_email(JANE, 4)
    columns = [r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(create_requests)")]
    assert columns == ["request_id", "status", "issue_number"]


# --- the executor writes the map on create ---------------------------------------

@pytest.fixture
def world(settings, tmp_path):
    store = TicketStore(tmp_path / "app.db")
    client = MockTicketClient()
    registry = ToolRegistry()
    register_ticket_tools(registry, client)
    return store, client, ToolExecutor(registry, settings, sleep=lambda s: None, request_store=store)


def create(executor, **over):
    return executor.execute("create_ticket", {**TICKET, **over}, agent="ticket", confirmed=True)


def test_create_stores_email_to_issue_number(world):
    store, _, executor = world
    result = create(executor)
    assert result.outcome is Outcome.SUCCESS
    assert store.numbers_for(JANE) == [result.data.number] == [1]


def test_two_creates_give_two_numbers_for_the_email(world):
    store, _, executor = world
    create(executor)
    create(executor)
    create(executor, email=SAM)
    assert store.numbers_for(JANE) == [1, 2] and store.numbers_for(SAM) == [3]


def test_recovered_create_after_a_lost_response_is_stored_once(world):
    store, client, executor = world
    client.lose_responses = 1
    create(executor)
    assert store.numbers_for(JANE) == [1]


def test_no_row_when_the_create_did_not_succeed(world):
    store, client, executor = world
    client.fail_before_create = 3
    assert create(executor).outcome is Outcome.UNAVAILABLE
    assert create(executor, email="not-an-email").outcome is Outcome.VALIDATION_ERROR
    assert store.numbers_for(JANE) == []


def test_unconfirmed_create_leaves_no_row(world):
    store, _, executor = world
    executor.execute("create_ticket", TICKET, agent="ticket", confirmed=False)
    assert store.numbers_for(JANE) == []
