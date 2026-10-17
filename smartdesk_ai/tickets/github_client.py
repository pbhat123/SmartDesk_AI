"""The ticket client: one interface, an in-memory mock, and the GitHub REST implementation (FR-B12).

The agent graph and the tools see `TicketClient` only. Whether tickets live in memory (mock mode,
FR-B10), behind REST, or one day behind an MCP server is chosen in `create_ticket_client`.
Error messages carry a status code and a path template, never response text or the token (FR-G11).
"""
from collections.abc import Iterator
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, SecretStr, ValidationError

from smartdesk_ai.config.env import EnvSettings
from smartdesk_ai.config.settings import GitHub, Settings
from smartdesk_ai.tickets.schemas import (
    Ticket,
    TicketConfigError,
    TicketCreate,
    TicketError,
    TicketNotFound,
    TransientTicketError,
)


class IssuePayload(BaseModel):
    """The body of a create-issue call."""

    title: str
    body: str
    labels: list[str]


def build_issue(draft: TicketCreate) -> IssuePayload:
    """Turn a ticket draft into the issue to create. Mock and REST share this, so they agree."""
    labels = [draft.category.value]
    if draft.priority is not None:
        labels.append(draft.priority.value)
    return IssuePayload(title=draft.title, body=draft.description, labels=labels)


class TicketClient(Protocol):
    def create(self, draft: TicketCreate) -> Ticket: ...

    def get(self, number: int) -> Ticket: ...


# --- mock mode ---------------------------------------------------------------

class MockTicketClient:
    """In-memory tickets for tests and for running without GitHub (FR-B10).

    `fail_before_create` and `lose_responses` count down the next creates: the first fails before
    anything exists, the second creates the issue and then raises, as a lost response would.
    """

    def __init__(self, repo: str = "mock-org/mock-tickets") -> None:
        self.repo = repo
        self.fail_before_create = 0
        self.lose_responses = 0
        self._issues: dict[int, Ticket] = {}

    def create(self, draft: TicketCreate) -> Ticket:
        if self.fail_before_create:
            self.fail_before_create -= 1
            raise TransientTicketError("mock: service unavailable")
        payload = build_issue(draft)
        number = len(self._issues) + 1
        ticket = Ticket(
            number=number, url=f"https://github.com/{self.repo}/issues/{number}",
            title=payload.title, body=payload.body, state="open", labels=tuple(payload.labels),
        )
        self._issues[number] = ticket
        if self.lose_responses:
            self.lose_responses -= 1
            raise TransientTicketError("mock: response lost")
        return ticket

    def get(self, number: int) -> Ticket:
        try:
            return self._issues[number]
        except KeyError:
            raise TicketNotFound(f"issue {number}") from None


# --- REST --------------------------------------------------------------------

def _to_ticket(item: Any) -> Ticket:
    """Read a GitHub issue object. Anything missing is an error, never filled in (FR-B13)."""
    if not isinstance(item, dict) or "pull_request" in item:
        raise TicketNotFound("not an issue")
    labels = [x["name"] if isinstance(x, dict) else x for x in item.get("labels") or []]
    try:
        return Ticket(
            number=item["number"], url=item["html_url"], title=item["title"],
            body=item.get("body") or "", state=item["state"], labels=tuple(labels),
        )
    except (KeyError, ValidationError, TypeError):
        raise TicketError("malformed issue in the GitHub response") from None


class GitHubRestClient:
    def __init__(
        self, repo: str, token: SecretStr, cfg: GitHub, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._repo = repo
        self._per_page = cfg.per_page
        self._http = httpx.Client(
            base_url=cfg.api_base_url,
            timeout=cfg.timeout_seconds,
            transport=transport,
            headers={
                "Authorization": f"Bearer {token.get_secret_value()}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": cfg.api_version,
            },
        )

    def __repr__(self) -> str:
        return f"GitHubRestClient(repo={self._repo!r})"

    def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, url, **kwargs)
        except httpx.TransportError:  # connect, timeout, protocol: all worth another try
            raise TransientTicketError("could not reach GitHub") from None
        code = response.status_code
        throttled = code == 429 or (
            code == 403 and ("retry-after" in response.headers
                             or response.headers.get("x-ratelimit-remaining") == "0")
        )
        if throttled or code >= 500:
            raise TransientTicketError(f"GitHub answered {code}")
        if code == 404:
            raise TicketNotFound("not found")
        if code >= 400:
            raise TicketError(f"GitHub answered {code}")
        return response

    def _json(self, response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            raise TicketError("GitHub sent a body that is not JSON") from None

    def _pages(self, url: str, params: dict[str, Any]) -> Iterator[list[Any]]:
        query: dict[str, Any] | None = {**params, "per_page": self._per_page}
        next_url: str | None = url
        while next_url:
            response = self._send("GET", next_url, params=query)
            body = self._json(response)
            items = body.get("items") if isinstance(body, dict) else body  # search wraps its list
            if not isinstance(items, list):
                raise TicketError("unexpected list shape in the GitHub response")
            yield items
            next_url = response.links.get("next", {}).get("url")
            query = None  # the next link already carries the query

    def create(self, draft: TicketCreate) -> Ticket:
        response = self._send("POST", f"/repos/{self._repo}/issues", json=build_issue(draft).model_dump())
        return _to_ticket(self._json(response))

    def get(self, number: int) -> Ticket:
        return _to_ticket(self._json(self._send("GET", f"/repos/{self._repo}/issues/{number}")))


# --- factory -------------------------------------------------------------------

def create_ticket_client(
    settings: Settings, env: EnvSettings, *, transport: httpx.BaseTransport | None = None
) -> TicketClient:
    """Pick the client from `modes.tickets`. Live mode needs the token and repo, or says so plainly."""
    if settings.modes.tickets == "mock":
        return MockTicketClient()
    if env.github_token is None or env.tickets_repo is None:
        missing = [k for k, v in (("GITHUB_TOKEN", env.github_token), ("TICKETS_REPO", env.tickets_repo)) if v is None]
        raise TicketConfigError(f"modes.tickets is live but {', '.join(missing)} is not set")
    return GitHubRestClient(env.tickets_repo, env.github_token, settings.github, transport=transport)
