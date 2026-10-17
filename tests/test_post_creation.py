"""T4.18: the reply after creation gives the ticket ID and link exactly as GitHub returned them, and
says the employee can ask for updates anytime (FR-B4, FR-B9, FR-B13). Mock tickets client."""
import re

import pytest

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.agent.ticket_agent import UPDATES_NOTE, created_reply
from smartdesk_ai.llm.mock import MockDraftModel
from smartdesk_ai.tickets import MockTicketClient, Ticket, TicketCreate
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM

EMAIL = "priya.k@smartdesk.example"


class FixedClient(MockTicketClient):
    """Returns a ticket whose number, link and labels are unlike anything the agent could guess."""

    def __init__(self, ticket):
        super().__init__()
        self.ticket = ticket

    def create(self, draft: TicketCreate) -> Ticket:
        return self.ticket


def create_through_the_graph(built_index_dir, settings, tmp_path, ticket):
    s = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"),
                 drafter_llm=MockDraftModel(), client=FixedClient(ticket))
    tid = new_thread_id()
    run_turn(s.graph, tid, "My monitor keeps flickering", email=EMAIL)
    run_turn(s.graph, tid, "yes")
    return run_turn(s.graph, tid, "yes")


TICKET = Ticket(number=4242, url="https://github.com/acme/desk-tickets/issues/4242", title="Monitor", state="open",
                labels=("IT",))


def test_the_reply_contains_exactly_the_returned_id_and_link(built_index_dir, settings, tmp_path):
    out = create_through_the_graph(built_index_dir, settings, tmp_path, TICKET)
    assert re.findall(r"#\d+", out.reply) == ["#4242"]
    assert re.findall(r"https?://\S+", out.reply) == [TICKET.url]


def test_the_reply_tells_the_employee_they_can_ask_for_updates_anytime(built_index_dir, settings, tmp_path):
    out = create_through_the_graph(built_index_dir, settings, tmp_path, TICKET)
    assert UPDATES_NOTE in out.reply and "updates" in out.reply and "anytime" in out.reply


def test_the_status_comes_only_from_the_response(built_index_dir, settings, tmp_path):
    in_progress = Ticket(number=7, url="https://github.com/acme/desk-tickets/issues/7", title="x", state="open",
                         labels=("IT", "in-progress"))
    assert "(In Progress)" in create_through_the_graph(built_index_dir, settings, tmp_path, in_progress).reply


def test_the_reply_confirms_the_creation_and_names_no_email(built_index_dir, settings, tmp_path):
    out = create_through_the_graph(built_index_dir, settings, tmp_path, TICKET)
    assert out.reply.lower().startswith("your ticket has been created") and EMAIL not in out.reply


@pytest.mark.parametrize("number, url", [(1, "https://github.com/o/r/issues/1"), (99999, "https://example.test/i/99999")])
def test_whatever_the_response_says_is_what_the_reply_says(number, url):
    reply = created_reply(Ticket(number=number, url=url, title="t", state="open"))
    assert f"#{number}" in reply and url in reply
    assert re.findall(r"#\d+", reply) == [f"#{number}"]


def test_nothing_is_invented_when_creation_fails(built_index_dir, settings, tmp_path):
    client = MockTicketClient()
    client.unreachable = True
    s = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"),
                 drafter_llm=MockDraftModel(), client=client)
    tid = new_thread_id()
    run_turn(s.graph, tid, "My monitor keeps flickering", email=EMAIL)
    run_turn(s.graph, tid, "yes")
    out = run_turn(s.graph, tid, "yes")
    assert "try again later" in out.reply.lower() and "#" not in out.reply and "http" not in out.reply
    assert "create_failed" in out.steps and out.ticket is None
