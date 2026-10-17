"""Types the ticket client speaks (FR-B11, FR-B12). Nothing here knows about REST or MCP."""
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict


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
    email: str
    request_id: str


class Ticket(BaseModel):
    """A ticket as GitHub reports it. Number, link, state and labels are copied, never composed (FR-B13)."""

    model_config = ConfigDict(frozen=True)

    number: int
    url: str
    title: str
    body: str = ""
    state: Literal["open", "closed"]
    labels: tuple[str, ...] = ()


class TicketError(Exception):
    """A permanent failure: retrying the same call will not help."""


class TicketNotFound(TicketError):
    """No such issue, or the number belongs to a pull request."""


class TicketConfigError(TicketError):
    """Live mode was asked for without the settings it needs."""


class TransientTicketError(ConnectionError):
    """GitHub is unreachable, throttling, or failing (network, timeout, 429, 5xx).

    A ConnectionError on purpose: the tool executor already retries those, and `tickets/` must not
    import from `tools/`.
    """
