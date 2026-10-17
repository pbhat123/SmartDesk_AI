"""T3.12: ticket ID, link and status come only from GitHub's response; the email is never guessed (FR-B13).

The whole path runs: executor -> registry -> ticket client -> mocked HTTP layer.
"""
import json

import httpx
import pytest
from pydantic import SecretStr

from smartdesk_ai.tickets import GitHubRestClient, MockTicketClient
from smartdesk_ai.tickets.sqlite_map import TicketStore
from smartdesk_ai.tools.executor import ErrorCode, Outcome, ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry
from smartdesk_ai.tools.ticket_tools import register_ticket_tools

REPO = "octo/tickets"
TICKET = {"title": "VPN down", "description": "Cannot connect from home", "category": "IT",
          "priority": "high", "email": "jane.roe@example.com"}
GITHUB_ISSUE = {
    "number": 4242, "html_url": "https://github.com/octo/tickets/issues/4242",
    "title": "VPN down", "body": "x", "state": "open", "labels": [{"name": "IT"}, {"name": "high"}],
}


def executor_for(client, settings, tmp_path):
    registry = ToolRegistry()
    register_ticket_tools(registry, client)
    store = TicketStore(tmp_path / "t.db")
    return ToolExecutor(registry, settings, sleep=lambda s: None, request_store=store)


def rest_returning(response_json, settings, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(201, json=response_json)

    return GitHubRestClient(REPO, SecretStr("t"), settings.github, transport=httpx.MockTransport(handler))


def create(executor, args=None):
    return executor.execute("create_ticket", args or TICKET, agent="ticket", confirmed=True)


def test_reply_shows_exactly_the_id_link_and_status_from_the_response(settings, tmp_path):
    result = create(executor_for(rest_returning(GITHUB_ISSUE, settings), settings, tmp_path))
    assert result.outcome is Outcome.SUCCESS
    reply = result.data.receipt()
    assert "#4242" in reply
    assert "https://github.com/octo/tickets/issues/4242" in reply
    assert "open" in reply
    assert (result.data.number, result.data.url, result.data.state) == (
        4242, "https://github.com/octo/tickets/issues/4242", "open")


def test_values_differ_from_anything_the_agent_side_could_have_made_up(settings, tmp_path):
    """A response with unusual values is echoed as is: the numbering is GitHub's, not ours."""
    odd = {**GITHUB_ISSUE, "number": 90017, "html_url": "https://github.com/octo/tickets/issues/90017"}
    result = create(executor_for(rest_returning(odd, settings), settings, tmp_path))
    assert result.data.number == 90017 and result.data.url.endswith("/issues/90017")
    assert "4242" not in result.data.receipt()


@pytest.mark.parametrize("missing", ["number", "html_url", "state", "title"])
def test_a_response_missing_a_detail_is_an_error_never_a_filled_in_ticket(settings, tmp_path, missing):
    broken = {k: v for k, v in GITHUB_ISSUE.items() if k != missing}
    result = create(executor_for(rest_returning(broken, settings), settings, tmp_path))
    assert result.outcome is Outcome.ERROR and result.data is None


def test_an_unknown_state_in_the_response_is_an_error(settings, tmp_path):
    result = create(executor_for(rest_returning({**GITHUB_ISSUE, "state": "triaged"}, settings), settings, tmp_path))
    assert result.outcome is Outcome.ERROR


def test_the_ticket_sent_holds_only_what_the_employee_said_plus_the_two_metadata_lines(settings, tmp_path):
    seen = []
    create(executor_for(rest_returning(GITHUB_ISSUE, settings, seen), settings, tmp_path))
    sent = json.loads(seen[0].content)
    assert sent["title"] == TICKET["title"]
    extra = [ln for ln in sent["body"].splitlines() if ln and ln != TICKET["description"]]
    assert extra[0] == "Employee: jane.roe@example.com" and extra[1].startswith("Request-ID: req-")
    assert len(extra) == 2
    assert sent["labels"] == ["IT", "high"]


class CountingClient(MockTicketClient):
    def __init__(self):
        super().__init__()
        self.create_calls = 0

    def create(self, draft):
        self.create_calls += 1
        return super().create(draft)


@pytest.mark.parametrize("email", [None, "", "   ", "not-an-email"])
def test_no_usable_email_asks_for_one_and_creates_nothing(settings, tmp_path, email):
    client = CountingClient()
    args = {k: v for k, v in TICKET.items() if k != "email"}
    if email is not None:
        args["email"] = email
    result = create(executor_for(client, settings, tmp_path), args)
    assert result.outcome is Outcome.VALIDATION_ERROR
    assert result.code in (ErrorCode.EMAIL_REQUIRED, ErrorCode.EMAIL_INVALID)
    assert "Ask the employee" in result.error
    assert client.create_calls == 0


def test_a_lost_response_is_answered_with_the_existing_issue_not_a_new_number(settings, tmp_path):
    client = MockTicketClient()
    client.lose_responses = 1
    result = create(executor_for(client, settings, tmp_path))
    assert result.outcome is Outcome.SUCCESS and result.attempts == 2
    assert result.data.number == 1 and len(client._issues) == 1
    assert result.data == client.get(1)
