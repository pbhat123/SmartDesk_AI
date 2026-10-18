"""UI review fix 2: a mock ticket is shown as "Ticket #N (mock)" with no link. A link appears only for a ticket GitHub
reported (live mode), and nothing in the app builds a github.com URL of its own (FR-B4, FR-B13)."""
import re

from smartdesk_ai.tickets import MockTicketClient
from smartdesk_ai.tickets.schemas import Ticket
from tests.ui_helpers import mock_context

EMAIL = "priya.k@smartdesk.example"
LINK = re.compile(r"https?://|github\.com", re.I)


def _create(ctx, session="s"):
    replies = [ctx.service.handle(session, t).text for t in ("My monitor keeps flickering", EMAIL, "yes")]
    return replies[-1]


def test_the_mock_client_never_makes_a_url(tmp_path):
    client = MockTicketClient()
    ticket = client.add_issue("t", "b")
    assert ticket.url == "" and ticket.is_mock


def test_a_created_mock_ticket_has_a_number_the_mock_marker_and_no_link(tmp_path):
    reply = _create(mock_context(tmp_path))
    assert re.search(r"Ticket #\d+ \(mock\)", reply) and not LINK.search(reply)
    assert "Open" in reply  # the status is still copied from the response


def test_the_status_of_a_mock_ticket_has_no_link(tmp_path):
    ctx = mock_context(tmp_path)
    _create(ctx)
    reply = ctx.service.handle("s", "any update on my ticket?").text
    assert "(mock)" in reply and "Status: Open" in reply and not LINK.search(reply)


def test_the_duplicate_question_for_a_mock_ticket_has_no_link(tmp_path):
    ctx = mock_context(tmp_path)
    _create(ctx)
    texts = [ctx.service.handle("t", t).text for t in ("My monitor keeps flickering", EMAIL)]
    assert "already have an open ticket" in texts[-1] and "(mock)" in texts[-1] and not LINK.search(texts[-1])


def test_a_ticket_github_reported_keeps_its_link():
    ticket = Ticket(number=7, url="https://github.com/acme/desk-tickets/issues/7", title="t", state="open")
    assert not ticket.is_mock
    assert ticket.receipt() == "Ticket #7 (Open): https://github.com/acme/desk-tickets/issues/7"
