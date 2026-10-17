"""The ticket client: one interface, an in-memory mock, and the GitHub REST implementation (FR-B12).

The agent graph and the tools see `TicketClient` only. Whether tickets live in memory (mock mode,
FR-B10), behind REST, or one day behind an MCP server is chosen in `create_ticket_client`.
Error messages carry a status code and a path template, never response text or the token (FR-G11).
"""
from collections.abc import Iterator, Sequence
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, SecretStr, ValidationError

from smartdesk_ai.config.env import EnvSettings
from smartdesk_ai.config.settings import GitHub, Settings
from smartdesk_ai.tickets.schemas import (
    TEAM_ASSOCIATIONS,
    LabelSpec,
    TeamUpdate,
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


EMPLOYEE_PREFIX = "Employee: "
REQUEST_ID_PREFIX = "Request-ID: "


def build_issue(draft: TicketCreate) -> IssuePayload:
    """Turn a ticket draft into the issue to create (FR-B11). Mock and REST share this, so they agree.

    The body is the description, then the `Employee:` and `Request-ID:` lines last, so a lookalike
    line typed inside the description is never the one that is read back.
    """
    body = f"{draft.description.rstrip()}\n\n{EMPLOYEE_PREFIX}{draft.email}\n{REQUEST_ID_PREFIX}{draft.request_id}\n"
    labels = [draft.category.value]
    if draft.priority is not None:
        labels.append(draft.priority.value)
    return IssuePayload(title=draft.title, body=body, labels=labels)


def _same_email(found: str | None, wanted: str) -> bool:
    return found is not None and found.lower() == wanted.strip().lower()


def _last_value(body: str, prefix: str) -> str | None:
    for line in reversed(body.splitlines()):
        if line.startswith(prefix):
            return line[len(prefix):].strip() or None
    return None


def employee_email_of(body: str) -> str | None:
    """The email on the last `Employee:` line of an issue body (FR-B11, FR-C11)."""
    return _last_value(body, EMPLOYEE_PREFIX)


def request_id_of(body: str) -> str | None:
    return _last_value(body, REQUEST_ID_PREFIX)


class TicketClient(Protocol):
    def create(self, draft: TicketCreate) -> Ticket: ...

    def get(self, number: int) -> Ticket: ...

    def find_by_request_id(self, request_id: str) -> Ticket | None:
        """Look through the repo's issues (a list call, not search) for this request ID (FR-G6)."""
        ...

    def ensure_labels(self, labels: Sequence[LabelSpec]) -> list[str]:
        """Create the labels that do not exist yet; return the names created (FR-B11)."""
        ...

    def latest_update(self, number: int) -> TeamUpdate | None:
        """The newest comment by the repo's owner, a member or a collaborator, or None (FR-C9).

        Comments from anyone else are ignored. The text is untrusted data: see `TeamUpdate`.
        """
        ...

    def update_issue(
        self, number: int, *, state: str | None = None, labels: Sequence[str] | None = None
    ) -> Ticket:
        """Set the state (open/closed) and/or replace the whole label list. Used by the seed script."""
        ...

    def add_comment(self, number: int, body: str) -> None:
        """Add a comment to an issue. Used by the seed script."""
        ...

    def search_by_email(self, email: str) -> list[Ticket]:
        """Issues whose last `Employee:` line is this email, by number (FR-C11).

        Search can lag a minute behind a new issue, so this is the fallback, not the main path.
        """
        ...


# --- mock mode ---------------------------------------------------------------

class MockTicketClient:
    """In-memory tickets for tests and for running without GitHub (FR-B10).

    `fail_before_create` and `lose_responses` count down the next creates: the first fails before
    anything exists, the second creates the issue and then raises, as a lost response would.
    `unreachable` makes every call fail transiently, as if GitHub were down (FR-F1).
    """

    def __init__(self, repo: str = "mock-org/mock-tickets") -> None:
        self.repo = repo
        self.fail_before_create = 0
        self.lose_responses = 0
        self._issues: dict[int, Ticket] = {}
        self._labels: set[str] = set()
        self._comments: dict[int, list[tuple[str, str]]] = {}
        self.fail_searches = 0
        self.unreachable = False

    def _reach(self) -> None:
        if self.unreachable:
            raise TransientTicketError("mock: GitHub is unreachable")

    def create(self, draft: TicketCreate) -> Ticket:
        self._reach()
        if self.fail_before_create:
            self.fail_before_create -= 1
            raise TransientTicketError("mock: service unavailable")
        payload = build_issue(draft)
        ticket = self.add_issue(payload.title, payload.body, labels=payload.labels)
        if self.lose_responses:
            self.lose_responses -= 1
            raise TransientTicketError("mock: response lost")
        return ticket

    def get(self, number: int) -> Ticket:
        self._reach()
        try:
            return self._issues[number]
        except KeyError:
            raise TicketNotFound(f"issue {number}") from None

    def find_by_request_id(self, request_id: str) -> Ticket | None:
        self._reach()
        return next((t for t in reversed(self._issues.values()) if request_id_of(t.body) == request_id), None)

    def ensure_labels(self, labels: Sequence[LabelSpec]) -> list[str]:
        self._reach()
        created = [spec.name for spec in labels if spec.name.lower() not in self._labels]
        self._labels.update(name.lower() for name in created)
        return created

    def add_issue(self, title: str, body: str, *, state: str = "open", labels: Sequence[str] = ()) -> Ticket:
        """An issue made by hand in GitHub: no request ID and no map row. Mock-only helper."""
        number = len(self._issues) + 1
        ticket = Ticket(number=number, url=f"https://github.com/{self.repo}/issues/{number}", title=title,
                        body=body, state=state, labels=tuple(labels))
        self._issues[number] = ticket
        return ticket

    def search_by_email(self, email: str) -> list[Ticket]:
        self._reach()
        if self.fail_searches:
            self.fail_searches -= 1
            raise TransientTicketError("mock: search unavailable")
        return [t for t in self._issues.values() if _same_email(employee_email_of(t.body), email)]

    def update_issue(
        self, number: int, *, state: str | None = None, labels: Sequence[str] | None = None
    ) -> Ticket:
        self._reach()
        changes: dict = {}
        if state is not None:
            changes["state"] = state
        if labels is not None:
            changes["labels"] = tuple(labels)
        updated = Ticket.model_validate({**self.get(number).model_dump(), **changes})
        self._issues[number] = updated
        return updated

    def add_comment(self, number: int, body: str, *, author_association: str = "MEMBER") -> None:
        self._reach()
        """A comment, as someone would add in GitHub (a team member by default). Mock-only helper."""
        self.get(number)
        self._comments.setdefault(number, []).append((author_association, body))

    def latest_update(self, number: int) -> TeamUpdate | None:
        self._reach()
        self.get(number)
        team = [body for assoc, body in self._comments.get(number, []) if assoc in TEAM_ASSOCIATIONS]
        return TeamUpdate(text=team[-1].strip()) if team else None


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
            raise TicketError(f"GitHub answered {code}", status=code)
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

    def update_issue(
        self, number: int, *, state: str | None = None, labels: Sequence[str] | None = None
    ) -> Ticket:
        changes = {k: v for k, v in (("state", state), ("labels", list(labels) if labels is not None else None))
                   if v is not None}
        response = self._send("PATCH", f"/repos/{self._repo}/issues/{number}", json=changes)
        return _to_ticket(self._json(response))

    def add_comment(self, number: int, body: str) -> None:
        self._send("POST", f"/repos/{self._repo}/issues/{number}/comments", json={"body": body})

    def find_by_request_id(self, request_id: str) -> Ticket | None:
        # Newest first, so a ticket created a moment ago is on page one. GitHub search can lag.
        params = {"state": "all", "sort": "created", "direction": "desc"}
        for page in self._pages(f"/repos/{self._repo}/issues", params):
            for item in page:
                try:
                    ticket = _to_ticket(item)
                except TicketNotFound:  # pull requests share this endpoint
                    continue
                if request_id_of(ticket.body) == request_id:
                    return ticket
        return None

    def latest_update(self, number: int) -> TeamUpdate | None:
        newest: tuple[str, str] | None = None
        for page in self._pages(f"/repos/{self._repo}/issues/{number}/comments", {}):
            for item in page:
                try:
                    stamp, body = item["created_at"], item["body"]
                except (KeyError, TypeError):
                    raise TicketError("malformed comment in the GitHub response") from None
                if not isinstance(stamp, str) or not isinstance(body, str):
                    raise TicketError("malformed comment in the GitHub response")
                if item.get("author_association") not in TEAM_ASSOCIATIONS:
                    continue  # anyone can comment on a public issue; only the team's words count
                if newest is None or stamp > newest[0]:  # ISO 8601 UTC stamps sort as text
                    newest = (stamp, body)
        return TeamUpdate(text=newest[1].strip()) if newest else None

    def search_by_email(self, email: str) -> list[Ticket]:
        # Search matches loosely, so every hit is checked against the real Employee line.
        query = f'repo:{self._repo} is:issue in:body "{EMPLOYEE_PREFIX}{email.strip()}"'
        found = []
        for page in self._pages("/search/issues", {"q": query}):
            for item in page:
                try:
                    ticket = _to_ticket(item)
                except TicketNotFound:
                    continue
                if _same_email(employee_email_of(ticket.body), email):
                    found.append(ticket)
        return sorted(found, key=lambda t: t.number)

    def ensure_labels(self, labels: Sequence[LabelSpec]) -> list[str]:
        have = {item["name"].lower() for page in self._pages(f"/repos/{self._repo}/labels", {}) for item in page}
        created = []
        for spec in labels:
            if spec.name.lower() in have:
                continue
            try:
                self._send("POST", f"/repos/{self._repo}/labels", json=spec.model_dump())
            except TicketError as exc:
                if exc.status != 422:  # 422: another run created it first
                    raise
                continue
            created.append(spec.name)
        return created


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
