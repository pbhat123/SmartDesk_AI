"""Crash safety of ticket creation (FR-G5, FR-G6, FR-B3). `begin()` logs the intent before the GitHub call; if the call
or the process fails after that, a retry must not be blocked by the leftover row, and must not create a second ticket
when the first attempt actually reached GitHub. Mock tickets only."""
import sqlite3

import pytest

from smartdesk_ai.tickets import MockTicketClient, TransientTicketError
from smartdesk_ai.tickets.sqlite_map import TicketStore, request_fingerprint
from smartdesk_ai.tools.executor import Outcome, ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry
from smartdesk_ai.tools.ticket_tools import register_ticket_tools

EMAIL = "priya.k@smartdesk.example"
TICKET = {"title": "Flickering monitor", "description": "It flickers", "category": "IT", "email": EMAIL}


class DropsTheResponseThenGoesDown(MockTicketClient):
    """The first create reaches GitHub but its response is lost, and GitHub is unreachable right after."""

    lookup_fails = False  # only the lookup by request ID is down

    def create(self, draft):
        try:
            return super().create(draft)
        except TransientTicketError:
            self.unreachable = True
            raise

    def find_by_request_id(self, request_id):
        if self.lookup_fails:
            raise TransientTicketError("lookup down")
        return super().find_by_request_id(request_id)


def build(settings, tmp_path, client=None, db="map.db"):
    client = client or MockTicketClient()
    store = TicketStore(tmp_path / db)
    registry = ToolRegistry()
    register_ticket_tools(registry, client)
    return client, store, ToolExecutor(registry, settings, sleep=lambda s: None, request_store=store)


def create(executor, **override):
    return executor.execute("create_ticket", {**TICKET, **override}, agent="ticket", confirmed=True)


def statuses(tmp_path, db="map.db"):
    return [row[0] for row in sqlite3.connect(tmp_path / db).execute("SELECT status FROM create_requests ORDER BY rowid")]


# --- a leftover pending row never blocks a retry -------------------------------------------------------------------

def test_a_failure_after_begin_leaves_a_pending_row_and_the_retry_creates_exactly_one_ticket(settings, tmp_path):
    client, store, executor = build(settings, tmp_path)
    client.fail_before_create = settings.tools.max_attempts  # every attempt fails before anything exists
    assert create(executor).outcome is Outcome.UNAVAILABLE and statuses(tmp_path) == ["pending"]
    retry = create(executor)
    assert retry.outcome is Outcome.SUCCESS and len(client._issues) == 1
    assert statuses(tmp_path) == ["abandoned", "done"]  # the dead request was closed, the new one completed


def test_a_process_crash_after_begin_does_not_block_the_next_attempt(settings, tmp_path):
    client, store, executor = build(settings, tmp_path)
    store.begin("req-dead", request_fingerprint(EMAIL, TICKET["title"]))  # the process died right after this line
    assert create(executor).outcome is Outcome.SUCCESS and len(client._issues) == 1
    assert store.get("req-dead") == ("abandoned", None)


# --- the first attempt reached GitHub: the retry finds it instead of creating another ----------------------------------

def test_a_crash_after_the_issue_was_created_but_before_complete_is_recovered_not_duplicated(settings, tmp_path):
    client, store, executor = build(settings, tmp_path)
    first = create(executor)
    number = first.data.number
    # Simulate the crash between the GitHub call and complete(): the row is pending again and the map link is gone.
    con = sqlite3.connect(tmp_path / "map.db")
    con.execute("UPDATE create_requests SET status='pending', issue_number=NULL")
    con.execute("DELETE FROM email_tickets")
    con.commit()
    again = create(executor)
    assert again.outcome is Outcome.SUCCESS and again.data.number == number
    assert len(client._issues) == 1 and store.numbers_for(EMAIL) == [number]
    assert statuses(tmp_path) == ["done"]


def test_a_lost_response_followed_by_an_outage_does_not_duplicate_on_the_next_try(settings, tmp_path):
    client = DropsTheResponseThenGoesDown()
    client.lose_responses = 1
    client, store, executor = build(settings, tmp_path, client)
    down = create(executor)
    assert down.outcome is Outcome.UNAVAILABLE and len(client._issues) == 1  # it did reach GitHub
    client.unreachable = False  # GitHub is back; the employee says yes again
    again = create(executor)
    assert again.outcome is Outcome.SUCCESS and len(client._issues) == 1 and again.data.number == 1
    assert store.numbers_for(EMAIL) == [1]


def test_if_the_recovery_lookup_itself_fails_nothing_is_created(settings, tmp_path):
    client = DropsTheResponseThenGoesDown()
    client.lose_responses = 1
    client, store, executor = build(settings, tmp_path, client)
    create(executor)  # leaves a pending row and one issue on "GitHub"
    client.unreachable = False
    client.lookup_fails = True
    blocked = create(executor)
    assert blocked.outcome is Outcome.UNAVAILABLE and "try again later" in blocked.reply
    assert len(client._issues) == 1  # no second ticket while the first one cannot be ruled out
    client.lookup_fails = False
    assert create(executor).data.number == 1 and len(client._issues) == 1  # not blocked for good


# --- scope and privacy ------------------------------------------------------------------------------------------------

def test_another_employee_or_another_title_is_not_matched_to_the_pending_request(settings, tmp_path):
    client = DropsTheResponseThenGoesDown()
    client.lose_responses = 1
    client, store, executor = build(settings, tmp_path, client)
    create(executor)
    client.unreachable = False
    other_person = create(executor, email="someone.else@smartdesk.example")
    other_title = create(executor, title="Keyboard is broken")
    assert other_person.outcome is other_title.outcome is Outcome.SUCCESS
    assert len(client._issues) == 3 and {other_person.data.number, other_title.data.number} == {2, 3}


def test_the_request_log_holds_a_hash_not_the_email_or_the_title(settings, tmp_path):
    client, store, executor = build(settings, tmp_path)
    client.fail_before_create = settings.tools.max_attempts
    create(executor)
    raw = (tmp_path / "map.db").read_bytes()
    assert b"priya" not in raw.lower() and b"flickering" not in raw.lower()  # nothing was linked: the failure came first
    fp = request_fingerprint(EMAIL, TICKET["title"])
    assert fp == request_fingerprint("  Priya.K@smartdesk.example ", "flickering   MONITOR") and EMAIL not in fp


def test_a_map_from_before_this_change_opens_and_keeps_its_rows(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE create_requests (request_id TEXT PRIMARY KEY, status TEXT NOT NULL, issue_number INTEGER)")
    con.execute("INSERT INTO create_requests VALUES ('req-old', 'done', 5)")
    con.commit()
    con.close()
    store = TicketStore(path)
    assert store.get("req-old") == ("done", 5)
    store.begin("req-new", "fp")
    assert store.pending_for("fp") == ["req-new"]
