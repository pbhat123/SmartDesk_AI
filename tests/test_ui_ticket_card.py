"""The ticket card is a view of the agent's own approval summary and receipt (FR-B8, FR-B4): it must read back the
same fields, and anything else stays ordinary text."""
from smartdesk_ai.agent.ticket_agent import confirmation_summary, created_reply
from smartdesk_ai.agent.state import TicketDraft
from smartdesk_ai.ui.ticket_card import parse_draft, parse_receipt


def _draft(**extra):
    return TicketDraft(title="Pet policy question", description="Are dogs allowed?\nAsked twice.", category="HR", **extra)


def test_the_card_reads_back_every_field_of_the_summary():
    view = parse_draft(confirmation_summary(_draft(priority="high"), "a.b@x.example"))
    assert view.fields == {"Title": "Pet policy question", "Description": "Are dogs allowed?\nAsked twice.",
                           "Category": "HR", "Priority": "High", "Email": "a.b@x.example"}


def test_other_replies_are_not_drafts():
    assert parse_draft("Sure, here is how to reset it.") is None
    assert parse_draft("Title: x\n\nShall I go ahead and create this?") is None


class _Ticket:
    def __init__(self, url):
        self.number, self.url = 5, url

    def receipt(self):
        return f"Ticket #5 (mock): Open" if not self.url else f"Ticket #5 (Open): {self.url}"


def test_a_mock_receipt_has_no_link_and_a_live_one_keeps_it():
    mock = parse_receipt(created_reply(_Ticket("")))
    assert (mock.number, mock.status, mock.mock, mock.url) == (5, "Open", True, "")
    assert "updates" in mock.note
    live = parse_receipt(created_reply(_Ticket("https://github.com/o/r/issues/5")))
    assert (live.number, live.status, live.mock, live.url) == (5, "Open", False, "https://github.com/o/r/issues/5")
    assert parse_receipt("Ticket #5 (mock): Open\n\nStatus: Open") is None  # a status lookup is not a creation
