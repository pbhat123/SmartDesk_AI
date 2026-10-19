"""Only a ticket found by an earlier unfinished request is reported as already created; the in-call retry keeps
"has been created" (FR-G5, G6, B4).
Mock tickets only."""
from smartdesk_ai.tickets import MockTicketClient
from smartdesk_ai.tickets.sqlite_map import TicketStore
from smartdesk_ai.tools.executor import Outcome, ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry
from smartdesk_ai.tools.ticket_tools import register_ticket_tools
from tests.ui_helpers import mock_context

EMAIL = "priya.k@smartdesk.example"
ASK = "My laptop screen is cracked, please raise a ticket"


def new_ticket(ctx, thread):
    for text in (ASK, EMAIL, "yes"):
        reply = ctx.service.handle(thread, text)
    return reply.text


def test_a_ticket_found_by_the_in_call_retry_check_is_reported_as_created(tmp_path):
    client = MockTicketClient()
    ctx = mock_context(tmp_path, client=client)
    client.lose_responses = 1  # (after the seeding) the first attempt creates the issue, its response is lost
    reply = new_ticket(ctx, "a")
    assert reply.startswith("Your ticket has been created.") and "#5" in reply
    assert "already created" not in reply


def test_a_ticket_found_by_an_earlier_unfinished_request_is_reported_as_already_created(settings, tmp_path):
    client, store = MockTicketClient(), TicketStore(tmp_path / "map.db")
    registry = ToolRegistry()
    register_ticket_tools(registry, client)
    executor = ToolExecutor(registry, settings, sleep=lambda s: None, request_store=store)
    args = {"title": "Flickering monitor", "description": "It flickers", "category": "IT", "email": EMAIL}
    first = executor.execute("create_ticket", args, agent="ticket", confirmed=True)
    assert first.outcome is Outcome.SUCCESS and not first.recovered  # a normal create is not "recovered"
    store._conn.execute("UPDATE create_requests SET status='pending', issue_number=NULL")  # crashed before complete()
    again = executor.execute("create_ticket", args, agent="ticket", confirmed=True)
    assert again.outcome is Outcome.SUCCESS and again.recovered and again.data.number == first.data.number
    assert len(client._issues) == 1


def test_a_normal_create_still_says_created(tmp_path):
    assert new_ticket(mock_context(tmp_path), "a").startswith("Your ticket has been created.")
