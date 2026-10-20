"""Types the ticket client speaks (FR-B11, FR-B12). Nothing here knows about REST."""
import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from smartdesk_ai.tickets.status import STATUS_IN_PROGRESS, STATUS_RESOLVED, TicketStatus, derive_status

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")  # no whitespace: it goes into a body line


class Category(StrEnum):
    IT = "IT"
    HR = "HR"
    MISC = "MISC"


class Priority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketCreate(BaseModel):
    """What the executor hands the client; the client turns it into an issue (FR-B11)."""

    model_config = ConfigDict(frozen=True)

    title: str
    description: str
    category: Category
    priority: Priority | None = None
    email: Annotated[str, Field(pattern=EMAIL_PATTERN.pattern)]
    request_id: Annotated[str, Field(pattern=r"^\S+$")]


class Ticket(BaseModel):
    """A ticket as GitHub reports it. Number, link, state and labels are copied, never composed (FR-B13)."""

    model_config = ConfigDict(frozen=True)

    number: int
    url: str  # GitHub's link; empty for a mock ticket, which has none and never gets an invented one
    title: str
    body: str = ""
    state: Literal["open", "closed"]
    labels: tuple[str, ...] = ()

    @property
    def status(self) -> TicketStatus:
        return derive_status(self.state, self.labels)

    @property
    def is_mock(self) -> bool:
        return not self.url

    def receipt(self) -> str:
        """The facts for a "ticket created" reply, only from the GitHub response (FR-B4, FR-B13).
        A mock ticket is marked as one and has no link."""
        if self.is_mock:
            return f"Ticket #{self.number} (mock): {self.status.value}"
        return f"Ticket #{self.number} ({self.status.value}): {self.url}"


class LabelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    color: str  # hex without '#', as GitHub wants it
    description: str


ALL_LABELS: tuple[LabelSpec, ...] = (
    LabelSpec(name="IT", color="1d76db", description="IT support request"),
    LabelSpec(name="HR", color="d93f0b", description="HR or payroll request"),
    LabelSpec(name="MISC", color="808080", description="Neither IT nor HR, e.g. facilities"),
    LabelSpec(name="low", color="c2e0c6", description="Low priority"),
    LabelSpec(name="medium", color="fbca04", description="Medium priority"),
    LabelSpec(name="high", color="f9a03f", description="High priority"),
    LabelSpec(name="critical", color="b60205", description="Critical priority"),
    LabelSpec(name=STATUS_IN_PROGRESS, color="0e8a16", description="The support team is working on it"),
    LabelSpec(name=STATUS_RESOLVED, color="5319e7", description="Resolved; closing the issue completes it"),
)


# GitHub's author_association values that mark the support team. Anyone else can comment on a
# public issue, so their text is never a team update (FR-C9).
TEAM_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})


class TeamUpdate(BaseModel):
    """The latest team comment. UNTRUSTED DATA: it is text from a public repo.

    Show it to the model only through `quoted()`, as quoted material to relay, never as part of the
    instructions (T4.x status flow; the T5.4 injection check should treat it as untrusted input).
    """

    model_config = ConfigDict(frozen=True)

    text: str

    def quoted(self) -> str:
        return "\n".join(f"> {line}" for line in self.text.splitlines() or [""])


class TicketError(Exception):
    """A permanent failure: retrying the same call will not help."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class TicketNotFound(TicketError):
    """No such issue, or the number belongs to a pull request."""


class TicketConfigError(TicketError):
    """Live mode was asked for without the settings it needs."""


class TransientTicketError(ConnectionError):
    """GitHub is unreachable, throttling, or failing (network, timeout, 429, 5xx).

    A ConnectionError on purpose: the tool executor already retries those, and `tickets/` must not
    import from `tools/`.
    """
