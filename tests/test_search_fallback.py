"""T3.16 and follow-up: map rows and a search of issue bodies are merged, new numbers backfilled (FR-C11, FR-C2)."""
import httpx
import pytest
from pydantic import SecretStr

from smartdesk_ai.tickets import (
    Category,
    GitHubRestClient,
    MockTicketClient,
    TicketCreate,
    TransientTicketError,
)
from smartdesk_ai.tickets.search_fallback import find_tickets
from smartdesk_ai.tickets.sqlite_map import TicketStore

REPO = "octo/tickets"
SAM = "sam.lee@example.com"
JANE = "jane.roe@example.com"


class SpyClient(MockTicketClient):
    def __init__(self):
        super().__init__()
        self.searches = []

    def search_by_email(self, email):
        self.searches.append(email)
        return super().search_by_email(email)


@pytest.fixture
def store():
    return TicketStore(":memory:")


def by_hand(client, who, title="Printer jammed"):
    """An issue typed straight into GitHub: no request ID, nothing in the map."""
    return client.add_issue(title, f"Third floor printer eats paper.\n\nEmployee: {who}\n")


# --- the fallback flow -----------------------------------------------------------

def test_hand_created_issue_not_in_map_is_found_by_email_then_appears_in_map(store):
    client = SpyClient()
    made = by_hand(client, SAM)
    assert store.numbers_for(SAM) == []
    found = find_tickets(SAM, client, store)
    assert [t.number for t in found] == [made.number]
    assert store.numbers_for(SAM) == [made.number]  # backfilled


def test_second_lookup_searches_again_but_does_not_duplicate(store):
    client = SpyClient()
    made = by_hand(client, SAM)
    find_tickets(SAM, client, store)
    again = find_tickets(SAM, client, store)
    assert client.searches == [SAM, SAM]
    assert [t.number for t in again] == [made.number]
    assert store.numbers_for(SAM) == [made.number]


def test_email_with_map_rows_and_a_hand_made_issue_returns_both(store):
    client = SpyClient()
    mine = client.create(TicketCreate(title="VPN", description="d", category=Category.IT,
                                      email=SAM, request_id="req-1"))
    store.link_email(SAM, mine.number)
    hand = by_hand(client, SAM)
    found = find_tickets(SAM, client, store)
    assert [t.number for t in found] == [mine.number, hand.number]
    assert store.numbers_for(SAM) == [mine.number, hand.number]  # the new number was backfilled


def test_a_ticket_in_both_the_map_and_the_search_appears_once(store):
    client = SpyClient()
    mine = client.create(TicketCreate(title="VPN", description="d", category=Category.IT,
                                      email=SAM, request_id="req-1"))
    store.link_email(SAM, mine.number)
    assert [t.number for t in find_tickets(SAM, client, store)] == [mine.number]
    assert client.searches == [SAM]


def test_every_hand_created_issue_is_backfilled(store):
    client = SpyClient()
    first, second = by_hand(client, SAM), by_hand(client, SAM, "Monitor")
    by_hand(client, JANE)
    found = find_tickets(SAM, client, store)
    assert [t.number for t in found] == [first.number, second.number]
    assert store.numbers_for(SAM) == [first.number, second.number]
    assert store.numbers_for(JANE) == []


def test_nothing_found_returns_empty_and_writes_nothing(store):
    client = SpyClient()
    by_hand(client, JANE)
    assert find_tickets(SAM, client, store) == []
    assert store.numbers_for(SAM) == []


def test_match_ignores_email_case(store):
    client = SpyClient()
    made = by_hand(client, "SAM.Lee@Example.com")
    assert [t.number for t in find_tickets(SAM, client, store)] == [made.number]


def test_an_issue_that_only_mentions_the_email_is_not_theirs(store):
    client = SpyClient()
    client.add_issue("Cover for Sam", f"Please also tell {SAM} about this.\n\nEmployee: {JANE}\n")
    assert find_tickets(SAM, client, store) == []


def test_a_map_row_for_a_deleted_issue_is_skipped(store):
    client = SpyClient()
    kept = by_hand(client, SAM)
    store.link_email(SAM, 999)  # no such issue any more
    store.link_email(SAM, kept.number)
    assert [t.number for t in find_tickets(SAM, client, store)] == [kept.number]


def test_search_failure_with_an_empty_map_surfaces_as_a_transient_error(store):
    client = SpyClient()
    client.fail_searches = 1
    with pytest.raises(TransientTicketError):
        find_tickets(SAM, client, store)
    assert store.numbers_for(SAM) == []


def test_search_failure_does_not_hide_tickets_the_map_already_knows(store):
    client = SpyClient()
    mine = client.create(TicketCreate(title="VPN", description="d", category=Category.IT,
                                      email=SAM, request_id="req-1"))
    store.link_email(SAM, mine.number)
    client.fail_searches = 1
    assert [t.number for t in find_tickets(SAM, client, store)] == [mine.number]


# --- REST search over a mocked HTTP layer -----------------------------------------------

def rest(handler, settings):
    return GitHubRestClient(REPO, SecretStr("t"), settings.github, transport=httpx.MockTransport(handler))


def item(number, body, **over):
    return {"number": number, "html_url": f"https://github.com/{REPO}/issues/{number}", "title": "t",
            "body": body, "state": "open", "labels": [], **over}


def test_rest_search_queries_the_repo_for_the_employee_line(settings):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"total_count": 1, "items": [item(4, f"x\n\nEmployee: {SAM}")]})

    found = rest(handler, settings).search_by_email(SAM)
    assert [t.number for t in found] == [4]
    request = seen[0]
    assert request.method == "GET" and request.url.path == "/search/issues"
    q = request.url.params["q"]
    assert f"repo:{REPO}" in q and "is:issue" in q and "in:body" in q and f'"Employee: {SAM}"' in q


def test_rest_search_drops_fuzzy_hits_and_pull_requests(settings):
    items = [
        item(1, f"mentions {SAM}\n\nEmployee: {JANE}"),         # fuzzy match, someone else's
        item(2, f"x\n\nEmployee: {SAM}", pull_request={}),      # a PR
        item(3, f"x\n\nEmployee: {SAM.upper()}"),               # theirs, other case
    ]
    found = rest(lambda r: httpx.Response(200, json={"items": items}), settings).search_by_email(SAM)
    assert [t.number for t in found] == [3]


def test_rest_search_follows_pages(settings):
    def handler(request):
        if request.url.params.get("page") == "2":
            return httpx.Response(200, json={"items": [item(9, f"Employee: {SAM}")]})
        return httpx.Response(
            200, json={"items": [item(2, f"Employee: {SAM}")]},
            headers={"link": '<https://api.github.com/search/issues?page=2>; rel="next"'},
        )

    assert [t.number for t in rest(handler, settings).search_by_email(SAM)] == [2, 9]


def test_rest_search_rate_limit_is_transient(settings):
    response = httpx.Response(403, headers={"x-ratelimit-remaining": "0"})
    with pytest.raises(TransientTicketError):
        rest(lambda r: response, settings).search_by_email(SAM)
