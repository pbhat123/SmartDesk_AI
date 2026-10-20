"""Find an employee's tickets: the SQLite map, the long-term memory and a search of the tickets repo, merged (FR-C2, FR-C11, FR-H10).

The search covers tickets created by hand in GitHub. Every lookup runs it, merges its hits with the
map's tickets (one entry per issue number) and writes any new numbers back to the map. The map
stays the main path: if the search fails, the tickets the map knows are still returned.

`remembered` are numbers from the long-term memory of an earlier session. They are read live like the others, but a
remembered number is kept for an email only as a hash and can go stale (a mock restart renumbers), so a ticket found
only that way is shown only if its own `Employee:` line is this email.
"""
from collections.abc import Iterable

from smartdesk_ai.tickets.github_client import TicketClient, _same_email, employee_email_of
from smartdesk_ai.tickets.schemas import Ticket, TicketNotFound, TransientTicketError
from smartdesk_ai.tickets.sqlite_map import TicketStore


def find_tickets(email: str, client: TicketClient, store: TicketStore, remembered: Iterable[int] = ()) -> list[Ticket]:
    """The employee's tickets, read live from GitHub, in issue-number order."""
    mapped = store.numbers_for(email)
    remembered = list(remembered)
    try:
        searched = client.search_by_email(email)
    except TransientTicketError:
        if not mapped and not remembered:  # nothing to show and no way to look: let the caller report it
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
    for number in remembered:
        if number not in found:
            try:
                ticket = client.get(number)
            except TicketNotFound:
                continue
            if _same_email(employee_email_of(ticket.body), email):
                found[number] = ticket
    return [found[number] for number in sorted(found)]
