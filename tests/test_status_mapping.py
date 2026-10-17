"""T3.14: ticket status from issue state and labels (FR-C6, FR-C9)."""
import pytest

from smartdesk_ai.tickets import Ticket
from smartdesk_ai.tickets.status import TicketStatus, derive_status

IP, RES = "in-progress", "resolved"


@pytest.mark.parametrize("state, labels, expected", [
    ("open", [], TicketStatus.OPEN),
    ("open", [IP], TicketStatus.IN_PROGRESS),
    ("closed", [RES], TicketStatus.RESOLVED),
    ("closed", [], TicketStatus.CLOSED),
    ("open", [IP, RES], TicketStatus.IN_PROGRESS),     # both labels on an open issue
    ("open", [RES], TicketStatus.OPEN),                # resolved alone does not count while open
    ("closed", [IP], TicketStatus.CLOSED),             # in-progress alone does not count once closed
    ("closed", [IP, RES], TicketStatus.RESOLVED),
    ("open", ["IT", "high"], TicketStatus.OPEN),       # category and priority labels are not status
    ("closed", ["HR", "low", RES], TicketStatus.RESOLVED),
    ("open", ["In-Progress"], TicketStatus.IN_PROGRESS),  # GitHub label names are case-insensitive
    ("closed", ["RESOLVED"], TicketStatus.RESOLVED),
])
def test_status_table(state, labels, expected):
    assert derive_status(state, labels) is expected


def test_status_words_are_the_four_reported_to_the_employee():
    assert [s.value for s in TicketStatus] == ["Open", "In Progress", "Resolved", "Closed"]


def test_unknown_state_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        derive_status("merged", [])


def test_ticket_exposes_its_status():
    ticket = Ticket(number=1, url="u", title="t", state="closed", labels=("IT", "resolved"))
    assert ticket.status is TicketStatus.RESOLVED


def test_receipt_reports_the_mapped_status():
    open_ticket = Ticket(number=5, url="https://x/5", title="t", state="open", labels=("IT",))
    done = Ticket(number=6, url="https://x/6", title="t", state="closed", labels=("resolved",))
    assert open_ticket.receipt() == "Ticket #5 (Open): https://x/5"
    assert done.receipt() == "Ticket #6 (Resolved): https://x/6"
