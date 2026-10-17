"""Find an employee's tickets: the SQLite map first, GitHub search only when the map is empty (FR-C2, FR-C11).

The fallback covers tickets created by hand in GitHub. Whatever search finds is written back to
the map, so the next lookup is served locally.
"""
from smartdesk_ai.tickets.github_client import TicketClient
from smartdesk_ai.tickets.schemas import Ticket, TicketNotFound
from smartdesk_ai.tickets.sqlite_map import TicketStore


def find_tickets(email: str, client: TicketClient, store: TicketStore) -> list[Ticket]:
    """The employee's tickets, read live from GitHub, in issue-number order."""
    numbers = store.numbers_for(email)
    if numbers:
        return _fetch(numbers, client)
    found = client.search_by_email(email)
    for ticket in found:
        store.link_email(email, ticket.number)
    return found


def _fetch(numbers: list[int], client: TicketClient) -> list[Ticket]:
    tickets = []
    for number in numbers:
        try:
            tickets.append(client.get(number))
        except TicketNotFound:  # a map row whose issue is gone; the rest still count
            continue
    return tickets
