"""T3.15: the latest team update is the text of the most recent issue comment (FR-C9). Mocks only."""
import httpx
import pytest
from pydantic import SecretStr

from smartdesk_ai.tickets import (
    Category,
    GitHubRestClient,
    MockTicketClient,
    TicketCreate,
    TicketError,
    TicketNotFound,
    TransientTicketError,
)

REPO = "octo/tickets"


def rest(handler, settings):
    return GitHubRestClient(REPO, SecretStr("t"), settings.github, transport=httpx.MockTransport(handler))


def comment(body, created_at, login="support-bot"):
    return {"body": body, "created_at": created_at, "user": {"login": login}}


def mock_with_ticket():
    client = MockTicketClient()
    client.create(TicketCreate(title="t", description="d", category=Category.IT,
                               email="jane.roe@example.com", request_id="req-1"))
    return client


# --- mock mode -----------------------------------------------------------------

def test_mock_ticket_without_comments_has_no_update():
    assert mock_with_ticket().latest_update(1) is None


def test_mock_returns_the_newest_of_several_comments():
    client = mock_with_ticket()
    for text in ("Looking into it.", "Reset your token.", "Fixed on our side, please retry."):
        client.add_comment(1, text)
    assert client.latest_update(1) == "Fixed on our side, please retry."


def test_mock_unknown_issue_is_not_found():
    client = mock_with_ticket()
    with pytest.raises(TicketNotFound):
        client.latest_update(99)
    with pytest.raises(TicketNotFound):
        client.add_comment(99, "x")


# --- REST over a mocked HTTP layer ---------------------------------------------------

def test_rest_returns_the_newest_comment_whatever_the_list_order(settings):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=[
            comment("second", "2026-10-14T09:00:00Z"),
            comment("newest", "2026-10-15T09:30:00Z"),
            comment("first", "2026-10-13T08:00:00Z"),
        ])

    assert rest(handler, settings).latest_update(7) == "newest"
    assert seen[0].method == "GET" and seen[0].url.path == f"/repos/{REPO}/issues/7/comments"


def test_rest_with_no_comments_gives_no_update(settings):
    assert rest(lambda r: httpx.Response(200, json=[]), settings).latest_update(7) is None


def test_rest_finds_the_newest_on_a_later_page(settings):
    def handler(request):
        if request.url.params.get("page") == "2":
            return httpx.Response(200, json=[comment("on page two", "2026-10-16T10:00:00Z")])
        return httpx.Response(
            200, json=[comment("on page one", "2026-10-10T10:00:00Z")],
            headers={"link": f'<https://api.github.com/repos/{REPO}/issues/7/comments?page=2>; rel="next"'},
        )

    assert rest(handler, settings).latest_update(7) == "on page two"


def test_rest_trims_the_comment_text(settings):
    def handler(request):
        return httpx.Response(200, json=[comment("  Done.\r\n", "2026-10-14T09:00:00Z")])

    assert rest(handler, settings).latest_update(7) == "Done."


def test_rest_unknown_issue_is_not_found_and_outages_are_transient(settings):
    with pytest.raises(TicketNotFound):
        rest(lambda r: httpx.Response(404, json={}), settings).latest_update(7)
    with pytest.raises(TransientTicketError):
        rest(lambda r: httpx.Response(503), settings).latest_update(7)


def test_rest_comment_without_a_timestamp_is_an_error_not_a_guess(settings):
    with pytest.raises(TicketError):
        rest(lambda r: httpx.Response(200, json=[{"body": "x"}]), settings).latest_update(7)
