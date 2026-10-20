"""T8.1: the long-term per-email store (FR-H10, FR-G11). It holds ticket numbers and the last category, keyed by a salted hash of
the email (MASK_SALT), and nothing else: no email, no title, no message text on disk. Mock tickets only, no network."""
import inspect
import logging
import re
import sqlite3

import pytest

from smartdesk_ai.memory.long_term import LongTermStore
from smartdesk_ai.ops.runtime import configure_mask_salt
from smartdesk_ai.tickets import MockTicketClient
from smartdesk_ai.tickets.sqlite_map import TicketStore
from smartdesk_ai.tools.executor import Outcome, ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry
from smartdesk_ai.tools.ticket_tools import register_ticket_tools

EMAIL = "priya.k@smartdesk.example"
OTHER = "arjun.m@smartdesk.example"
TITLE = "Flickering monitor on floor two"
TEXT = "My monitor flickers whenever the docking station wakes up"
TICKET = {"title": TITLE, "description": TEXT, "category": "IT", "email": EMAIL}


@pytest.fixture(autouse=True)
def reset_salt():
    configure_mask_salt("test-salt")
    yield
    configure_mask_salt(None)


def disk_bytes(path):
    """Everything a store left on disk: the database and its journal files, as bytes."""
    return b"".join(p.read_bytes() for p in sorted(path.parent.glob(path.name + "*")))


def all_rows(path):
    conn = sqlite3.connect(path)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    rows = {t: conn.execute(f"SELECT * FROM {t}").fetchall() for t in tables}
    columns = {t: [c[1] for c in conn.execute(f"PRAGMA table_info({t})")] for t in tables}
    conn.close()
    return rows, columns


# --- what it holds ------------------------------------------------------------------------------------------------------

def test_it_remembers_ticket_numbers_and_the_last_category_per_email(tmp_path):
    store = LongTermStore(tmp_path / "memory.db")
    store.remember(EMAIL, 7, "IT")
    store.remember(EMAIL, 12, "HR")
    store.remember(OTHER, 9, "MISC")
    assert store.ticket_numbers(EMAIL) == [7, 12] and store.last_category(EMAIL) == "HR"
    assert store.ticket_numbers(OTHER) == [9] and store.last_category(OTHER) == "MISC"
    assert store.ticket_numbers("nobody@smartdesk.example") == [] and store.last_category("nobody@smartdesk.example") is None


def test_it_is_idempotent_and_ignores_case_and_spaces_in_the_email(tmp_path):
    store = LongTermStore(tmp_path / "memory.db")
    store.remember(EMAIL, 7, "IT")
    store.remember(f"  {EMAIL.upper()} ", 7, "IT")
    assert store.ticket_numbers(EMAIL) == [7] and store.ticket_numbers(EMAIL.title()) == [7]


def test_it_survives_a_restart(tmp_path):
    LongTermStore(tmp_path / "memory.db").remember(EMAIL, 7, "HR")
    again = LongTermStore(tmp_path / "memory.db")
    assert again.ticket_numbers(EMAIL) == [7] and again.last_category(EMAIL) == "HR"


def test_a_category_outside_it_hr_misc_and_a_bad_number_are_refused(tmp_path):
    store = LongTermStore(tmp_path / "memory.db")
    for number, category in ((7, "FINANCE"), (0, "IT"), (-3, "IT")):
        with pytest.raises(ValueError):
            store.remember(EMAIL, number, category)
    assert store.ticket_numbers(EMAIL) == [] and store.last_category(EMAIL) is None


def test_the_api_has_no_place_for_a_title_status_or_message_text():
    assert list(inspect.signature(LongTermStore.remember).parameters) == ["self", "email", "number", "category"]
    assert {n for n in dir(LongTermStore) if not n.startswith("_")} == {"remember", "ticket_numbers", "last_category", "close"}


# --- what is on disk ----------------------------------------------------------------------------------------------------

def test_nothing_on_disk_is_an_email_a_title_or_message_text(settings, tmp_path):
    memory_path = tmp_path / "memory.db"
    memory = LongTermStore(memory_path)
    registry = ToolRegistry()
    register_ticket_tools(registry, MockTicketClient())
    executor = ToolExecutor(registry, settings, sleep=lambda s: None, request_store=TicketStore(tmp_path / "map.db"), memory=memory)
    assert executor.execute("create_ticket", TICKET, agent="ticket", confirmed=True).outcome is Outcome.SUCCESS
    memory.close()
    raw = disk_bytes(memory_path).lower()
    for secret in (EMAIL, "priya", "smartdesk.example", "flicker", "monitor", "docking", "floor two"):
        assert secret.encode() not in raw, secret
    rows, columns = all_rows(memory_path)
    assert {t: set(c) for t, c in columns.items()} == {
        "employee_tickets": {"user_key", "issue_number"}, "employee_categories": {"user_key", "last_category"}}
    for table, table_rows in rows.items():
        for row in table_rows:
            assert re.fullmatch(r"[0-9a-f]{64}", row[0]), "the key is a salted SHA-256, not an email"
            assert isinstance(row[1], int) or row[1] in {"IT", "HR", "MISC"}
    assert rows["employee_tickets"] and rows["employee_categories"][0][1] == "IT"


def test_the_key_depends_on_the_salt(tmp_path):
    store = LongTermStore(tmp_path / "memory.db")
    store.remember(EMAIL, 7, "IT")
    key_a = sqlite3.connect(tmp_path / "memory.db").execute("SELECT user_key FROM employee_tickets").fetchone()[0]
    configure_mask_salt("another-salt")
    assert store.ticket_numbers(EMAIL) == [], "with another salt the stored key no longer matches this email"
    store.remember(EMAIL, 8, "IT")
    keys = {r[0] for r in sqlite3.connect(tmp_path / "memory.db").execute("SELECT user_key FROM employee_tickets")}
    assert len(keys) == 2 and key_a in keys


def test_the_store_logs_nothing_with_an_email(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG):
        store = LongTermStore(tmp_path / "memory.db")
        store.remember(EMAIL, 7, "IT")
        store.ticket_numbers(EMAIL), store.last_category(EMAIL)
        with pytest.raises(ValueError) as error:
            store.remember(EMAIL, 8, "NOPE")
    assert EMAIL not in caplog.text and EMAIL not in str(error.value) and "priya" not in str(error.value).lower()


# --- when it is written -------------------------------------------------------------------------------------------------

def build(settings, tmp_path, client=None):
    client = client or MockTicketClient()
    memory = LongTermStore(tmp_path / "memory.db")
    registry = ToolRegistry()
    register_ticket_tools(registry, client)
    executor = ToolExecutor(registry, settings, sleep=lambda s: None, request_store=TicketStore(":memory:"), memory=memory)
    return client, memory, executor


def test_a_confirmed_create_records_the_number_and_category(settings, tmp_path):
    client, memory, executor = build(settings, tmp_path)
    result = executor.execute("create_ticket", {**TICKET, "category": "HR"}, agent="ticket", confirmed=True)
    assert result.outcome is Outcome.SUCCESS
    assert memory.ticket_numbers(EMAIL) == [result.data.number] and memory.last_category(EMAIL) == "HR"


def test_nothing_is_recorded_without_a_confirmed_successful_create(settings, tmp_path):
    client, memory, executor = build(settings, tmp_path)
    assert executor.execute("create_ticket", TICKET, agent="ticket", confirmed=False).outcome is not Outcome.SUCCESS
    client.fail_before_create = settings.tools.max_attempts
    assert executor.execute("create_ticket", TICKET, agent="ticket", confirmed=True).outcome is Outcome.UNAVAILABLE
    assert memory.ticket_numbers(EMAIL) == [] and memory.last_category(EMAIL) is None


def test_a_ticket_found_again_after_a_lost_response_is_recorded_once(settings, tmp_path):
    client, memory, executor = build(settings, tmp_path)
    first = executor.execute("create_ticket", TICKET, agent="ticket", confirmed=True)
    again = executor.execute("create_ticket", TICKET, agent="ticket", confirmed=True)
    assert first.outcome is again.outcome is Outcome.SUCCESS
    assert memory.ticket_numbers(EMAIL) == sorted({first.data.number, again.data.number})


def test_the_executor_works_without_a_memory_store(settings):
    registry = ToolRegistry()
    register_ticket_tools(registry, MockTicketClient())
    executor = ToolExecutor(registry, settings, sleep=lambda s: None, request_store=TicketStore(":memory:"))
    assert executor.execute("create_ticket", TICKET, agent="ticket", confirmed=True).outcome is Outcome.SUCCESS


# --- the app wiring -----------------------------------------------------------------------------------------------------

def test_the_app_writes_a_created_ticket_to_an_in_memory_store_in_mock_mode(settings, tmp_path):
    """Mock tickets are numbered again after a restart, so in mock mode no memory file is left on disk."""
    from smartdesk_ai.agent.factory import build_stack

    stack = build_stack(settings, environ={"MASK_SALT": "wired"}, dotenv_path=None, data_dir=tmp_path, client=MockTicketClient())
    result = stack.executor.execute("create_ticket", TICKET, agent="ticket", confirmed=True)
    assert result.outcome is Outcome.SUCCESS and stack.memory.ticket_numbers(EMAIL) == [result.data.number]
    assert not (tmp_path / "memory.db").exists()
