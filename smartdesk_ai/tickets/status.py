"""Ticket status from the GitHub issue state and labels (FR-C6, FR-C9).

A leaf module: it knows the two status labels and nothing else about tickets.
"""
from collections.abc import Iterable
from enum import StrEnum

STATUS_IN_PROGRESS = "in-progress"
STATUS_RESOLVED = "resolved"


class TicketStatus(StrEnum):
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    RESOLVED = "Resolved"
    CLOSED = "Closed"


def derive_status(state: str, labels: Iterable[str]) -> TicketStatus:
    """Open with no status label is Open; `in-progress` makes an open issue In Progress;
    a closed issue is Resolved with the `resolved` label and Closed without it. A status label
    that does not fit the state is ignored (open + `resolved` only is Open)."""
    names = {label.lower() for label in labels}
    if state == "open":
        return TicketStatus.IN_PROGRESS if STATUS_IN_PROGRESS in names else TicketStatus.OPEN
    if state == "closed":
        return TicketStatus.RESOLVED if STATUS_RESOLVED in names else TicketStatus.CLOSED
    raise ValueError(f"unknown issue state {state!r}")
