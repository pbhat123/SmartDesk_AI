"""T3.10: tickets client wrapper: interface, mock mode, REST over a mocked HTTP layer (FR-B12, FR-B10).

No test here touches the network: the REST client is driven through httpx.MockTransport.
"""
import ast
import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from smartdesk_ai.config.env import EnvSettings
from smartdesk_ai.tickets import (
    Category,
    GitHubRestClient,
    MockTicketClient,
    Priority,
    Ticket,
    TicketConfigError,
    TicketCreate,
    TicketError,
    TicketNotFound,
    TransientTicketError,
    create_ticket_client,
)
from smartdesk_ai.tools.definitions import Category as ToolCategory
from smartdesk_ai.tools.definitions import Priority as ToolPriority
from smartdesk_ai.tools.errors import TRANSIENT

ROOT = Path(__file__).resolve().parent.parent
REPO = "octo/tickets"
DRAFT = TicketCreate(
    title="VPN down", description="Cannot connect from home", category=Category.IT,
    priority=Priority.HIGH, email="jane.roe@example.com", request_id="req-abc",
)


def issue_json(number=7, **over):
    item = {
        "number": number, "html_url": f"https://github.com/{REPO}/issues/{number}",
        "title": "VPN down", "body": "text", "state": "open", "labels": [{"name": "IT"}],
    }
    return {**item, **over}


def rest(handler, settings, token="ghp_secret"):
    return GitHubRestClient(
        REPO, SecretStr(token), settings.github, transport=httpx.MockTransport(handler)
    )


# --- mock mode ---------------------------------------------------------------

def test_mock_create_returns_realistic_ticket():
    client = MockTicketClient()
    ticket = client.create(DRAFT)
    assert isinstance(ticket, Ticket)
    assert ticket.number == 1 and ticket.state == "open"
    assert ticket.url == "" and ticket.is_mock  # a mock ticket has no link, and none is invented
    assert ticket.title == DRAFT.title
    assert {"IT", "high"} <= set(ticket.labels)


def test_mock_numbers_increase_and_get_returns_the_same_ticket():
    client = MockTicketClient()
    first, second = client.create(DRAFT), client.create(DRAFT)
    assert (first.number, second.number) == (1, 2)
    assert client.get(2) == second


def test_mock_get_unknown_issue_is_not_found():
    with pytest.raises(TicketNotFound):
        MockTicketClient().get(99)


def test_mock_can_fail_before_creating():
    client = MockTicketClient()
    client.fail_before_create = 1
    with pytest.raises(TransientTicketError):
        client.create(DRAFT)
    assert client.create(DRAFT).number == 1  # nothing was created by the failed call


def test_mock_can_lose_a_response_after_creating():
    client = MockTicketClient()
    client.lose_responses = 1
    with pytest.raises(TransientTicketError):
        client.create(DRAFT)
    assert client.get(1).title == DRAFT.title  # the issue exists though the caller never saw it


def test_transient_ticket_error_is_retried_by_the_executor():
    assert issubclass(TransientTicketError, TRANSIENT)
    assert not issubclass(TicketNotFound, TRANSIENT)


# --- factory ---------------------------------------------------------------

def test_factory_returns_mock_in_mock_mode(settings):
    assert isinstance(create_ticket_client(settings, EnvSettings()), MockTicketClient)


def test_factory_live_mode_without_credentials_is_a_clear_error(settings):
    live = settings.model_copy(update={"modes": settings.modes.model_copy(update={"tickets": "live"})})
    with pytest.raises(TicketConfigError, match="GITHUB_TOKEN"):
        create_ticket_client(live, EnvSettings())


def test_factory_live_mode_builds_the_rest_client_without_any_call(settings):
    live = settings.model_copy(update={"modes": settings.modes.model_copy(update={"tickets": "live"})})
    env = EnvSettings(github_token="ghp_x", tickets_repo=REPO)
    assert isinstance(create_ticket_client(live, env), GitHubRestClient)


# --- REST over a mocked HTTP layer --------------------------------------------

def test_rest_create_posts_the_issue_and_returns_github_values(settings):
    seen = {}

    def handler(request):
        seen["method"], seen["path"] = request.method, request.url.path
        seen["auth"] = request.headers["authorization"]
        seen["version"] = request.headers["x-github-api-version"]
        seen["json"] = json.loads(request.content)
        return httpx.Response(201, json=issue_json(number=4242))

    ticket = rest(handler, settings).create(DRAFT)
    assert (seen["method"], seen["path"]) == ("POST", f"/repos/{REPO}/issues")
    assert seen["auth"] == "Bearer ghp_secret"
    assert seen["version"] == settings.github.api_version
    assert seen["json"]["title"] == DRAFT.title and "labels" in seen["json"]
    assert ticket.number == 4242 and ticket.url == f"https://github.com/{REPO}/issues/4242"


def test_rest_get_reads_state_and_labels(settings):
    def handler(request):
        assert request.method == "GET" and request.url.path == f"/repos/{REPO}/issues/7"
        return httpx.Response(200, json=issue_json(state="closed", labels=[{"name": "resolved"}, "IT"]))

    ticket = rest(handler, settings).get(7)
    assert ticket.state == "closed" and set(ticket.labels) == {"resolved", "IT"}


def test_rest_get_404_is_not_found_and_pull_requests_are_not_tickets(settings):
    with pytest.raises(TicketNotFound):
        rest(lambda r: httpx.Response(404, json={}), settings).get(7)
    pr = issue_json(pull_request={"url": "x"})
    with pytest.raises(TicketNotFound):
        rest(lambda r: httpx.Response(200, json=pr), settings).get(7)


@pytest.mark.parametrize("response", [
    httpx.Response(500), httpx.Response(502), httpx.Response(429),
    httpx.Response(403, headers={"x-ratelimit-remaining": "0"}),
    httpx.Response(403, headers={"retry-after": "30"}),
])
def test_rest_throttling_and_server_errors_are_transient(settings, response):
    with pytest.raises(TransientTicketError):
        rest(lambda r: response, settings).get(7)


@pytest.mark.parametrize("error", [
    httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), httpx.RemoteProtocolError("reset"),
])
def test_rest_network_errors_are_transient(settings, error):
    def handler(request):
        raise error

    with pytest.raises(TransientTicketError):
        rest(handler, settings).get(7)


@pytest.mark.parametrize("status", [400, 401, 403, 422])
def test_rest_client_errors_are_permanent(settings, status):
    with pytest.raises(TicketError) as info:
        rest(lambda r: httpx.Response(status, json={"message": "nope"}), settings).get(7)
    assert not isinstance(info.value, TRANSIENT)


def test_rest_errors_never_carry_the_token_or_response_text(settings):
    with pytest.raises(TicketError) as info:
        rest(lambda r: httpx.Response(401, json={"message": "jane.roe@example.com"}), settings,
             token="ghp_topsecret").get(7)
    assert "ghp_topsecret" not in str(info.value) and "jane.roe" not in str(info.value)


def test_token_stays_out_of_the_client_repr(settings):
    assert "ghp_secret" not in repr(rest(lambda r: httpx.Response(200), settings))


# --- boundaries ---------------------------------------------------------------

def test_agent_code_imports_only_the_wrapper():
    """Nothing outside tickets/ may import httpx or the REST module; they go through the package."""
    offenders = []
    for path in (ROOT / "smartdesk_ai").rglob("*.py"):
        if path.relative_to(ROOT / "smartdesk_ai").parts[0] == "tickets":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            if any(n == "httpx" or n.startswith("smartdesk_ai.tickets.github_client") for n in names):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_category_and_priority_values_match_the_tool_definitions():
    from typing import get_args

    assert [c.value for c in Category] == list(get_args(ToolCategory))
    assert [p.value for p in Priority] == list(get_args(ToolPriority))


def test_config_selects_the_tickets_mode(settings):
    assert settings.modes.tickets == "mock"  # tests and the default run never reach GitHub
