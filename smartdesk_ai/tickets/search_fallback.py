"""Find an employee's tickets: the SQLite map plus a search of the tickets repo, merged (FR-C2, FR-C11).

The search covers tickets created by hand in GitHub. Every lookup runs it, merges its hits with the
map's tickets (one entry per issue number) and writes any new numbers back to the map. The map
stays the main path: if the search fails, the tickets the map knows are still returned.
"""
from smartdesk_ai.tickets.github_client import TicketClient
from smartdesk_ai.tickets.schemas import Ticket, TicketNotFound, TransientTicketError
from smartdesk_ai.tickets.sqlite_map import TicketStore


def find_tickets(email: str, client: TicketClient, store: TicketStore) -> list[Ticket]:
    """The employee's tickets, read live from GitHub, in issue-number order."""
    mapped = store.numbers_for(email)
    try:
        searched = client.search_by_email(email)
    except TransientTicketError:
        if not mapped:  # nothing to show and no way to look: let the caller report it
            raise
        searched = []
    for ticket in searched:
        store.link_email(email, ticket.number)  # a no-op for numbers the map already holds
    found = {ticket.number: ticket for ticket in searched}
    for number in mapped:
        if number not in found:
            try:
                found[number] = client.get(number)
            except TicketNotFound:  # a map row whose issue is gone; the rest still count
                continue
    return [found[number] for number in sorted(found)]
