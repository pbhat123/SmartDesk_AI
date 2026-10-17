"""T3.18: GitHub unreachable gives a polite "try again later" message after 3 attempts (FR-F1, FR-G4)."""
import httpx
import pytest
from pydantic import SecretStr

from smartdesk_ai.tickets import (
    ALL_LABELS,
    GitHubRestClient,
    MockTicketClient,
    TicketError,
    TransientTicketError,
    find_tickets,
)
from smartdesk_ai.tickets.sqlite_map import TicketStore
from smartdesk_ai.tools.executor import Outcome, ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry
from smartdesk_ai.tools.ticket_tools import register_ticket_tools

REPO = "octo/tickets"
EMAIL = "priya.k@smartdesk.example"
TICKET = {"title": "Flickering monitor", "description": "It flickers", "category": "IT", "email": EMAIL}
TECHNICAL = ("github", "http", "503", "connect", "exception", "traceback", "token", "timeout", EMAIL)


def rest(handler, settings):
    return GitHubRestClient(REPO, SecretStr("t"), settings.github, transport=httpx.MockTransport(handler))


def executor_for(client, settings, tmp_path):
    registry = ToolRegistry()
    register_ticket_tools(registry, client)
    store = TicketStore(tmp_path / "t.db")
    return store, ToolExecutor(registry, settings, sleep=lambda s: None, request_store=store)


def create(executor):
    return executor.execute("create_ticket", TICKET, agent="ticket", confirmed=True)


def assert_polite(reply):
    assert reply and "try again later" in reply
    assert not any(word.lower() in reply.lower() for word in TECHNICAL)


# --- network down, over the real REST client ------------------------------------------

def test_network_error_after_three_attempts_is_unavailable_with_a_retry_later_message(settings, tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ConnectError("refused")

    store, executor = executor_for(rest(handler, settings), settings, tmp_path)
    result = create(executor)
    assert result.outcome is Outcome.UNAVAILABLE and result.attempts == settings.tools.max_attempts == 3
    assert result.data is None
    assert_polite(result.reply)
    assert store.numbers_for(EMAIL) == []  # no map row for a ticket we cannot vouch for


def test_server_errors_count_as_unreachable_too(settings, tmp_path):
    _, executor = executor_for(rest(lambda r: httpx.Response(503), settings), settings, tmp_path)
    result = create(executor)
    assert result.outcome is Outcome.UNAVAILABLE and result.attempts == 3
    assert_polite(result.reply)


def test_the_reply_does_not_claim_the_ticket_exists(settings, tmp_path):
    def handler(request):
        raise httpx.ReadTimeout("slow")

    _, executor = executor_for(rest(handler, settings), settings, tmp_path)
    reply = create(executor).reply
    assert "created" in reply and "couldn't" in reply  # "couldn't confirm ... was created"
    assert "#" not in reply and "http" not in reply.lower()


def test_an_outage_that_clears_before_the_third_attempt_is_a_success_with_no_apology(settings, tmp_path):
    client = MockTicketClient()
    client.fail_before_create = 2
    _, executor = executor_for(client, settings, tmp_path)
    result = create(executor)
    assert result.outcome is Outcome.SUCCESS and result.attempts == 3 and result.reply is None


def test_a_permanent_error_is_not_dressed_up_as_an_outage(settings, tmp_path):
    _, executor = executor_for(rest(lambda r: httpx.Response(401, json={}), settings), settings, tmp_path)
    result = create(executor)
    assert result.outcome is Outcome.ERROR and result.attempts == 1 and result.reply is None


# --- the mock's "GitHub is down" switch -----------------------------------------------------

def test_mock_unreachable_makes_every_call_fail_transiently():
    client = MockTicketClient()
    made = client.add_issue("t", "b\n\nEmployee: a@b.cc\n")
    client.unreachable = True
    calls = [
        lambda: client.get(made.number), lambda: client.find_by_request_id("x"),
        lambda: client.ensure_labels(ALL_LABELS), lambda: client.latest_update(made.number),
        lambda: client.search_by_email("a@b.cc"), lambda: client.update_issue(made.number, state="closed"),
        lambda: client.add_comment(made.number, "hi"),
    ]
    for call in calls:
        with pytest.raises(TransientTicketError):
            call()
    client.unreachable = False
    assert client.get(made.number).title == "t"  # and it comes back when GitHub does


def test_mock_unreachable_on_create_leaves_nothing_behind(settings, tmp_path):
    client = MockTicketClient()
    client.unreachable = True
    _, executor = executor_for(client, settings, tmp_path)
    result = create(executor)
    assert result.outcome is Outcome.UNAVAILABLE and client._issues == {}
    assert_polite(result.reply)


def test_status_lookup_with_github_down_gets_its_own_polite_message(settings, tmp_path):
    client = MockTicketClient()
    client.unreachable = True
    store = TicketStore(":memory:")
    registry = ToolRegistry()
    registry.register("get_tickets", lambda args: find_tickets(args.email, client, store))
    executor = ToolExecutor(registry, settings, sleep=lambda s: None)
    result = executor.execute("get_tickets", {"email": EMAIL}, agent="ticket")
    assert result.outcome is Outcome.UNAVAILABLE and result.attempts == 3
    assert_polite(result.reply)
    assert "tickets" in result.reply


def test_an_unlisted_tool_gets_the_general_retry_later_message(settings):
    def always_down(args):
        raise TransientTicketError("down")

    registry = ToolRegistry()
    registry.register("search_kb", always_down)
    result = ToolExecutor(registry, settings, sleep=lambda s: None).execute("search_kb", {"query": "x"}, agent="a")
    assert result.outcome is Outcome.UNAVAILABLE
    assert_polite(result.reply)


def test_successes_and_rejections_carry_no_unavailable_message(settings, tmp_path):
    _, executor = executor_for(MockTicketClient(), settings, tmp_path)
    assert create(executor).reply is None
    rejected = executor.execute("create_ticket", TICKET, agent="ticket", confirmed=False)
    assert rejected.reply is None


def test_ticket_errors_stay_non_transient():
    assert not issubclass(TicketError, ConnectionError) and issubclass(TransientTicketError, ConnectionError)
