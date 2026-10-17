"""Connects the ticket client to the tool registry (FR-G3, FR-B12, FR-B13).

`create_ticket` returns the `Ticket` exactly as the client read it from GitHub. Nothing here adds
or fills in a number, link or status; if the response lacks one, the client raises.
"""
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from smartdesk_ai.tickets import TeamUpdate, Ticket, TicketClient, TicketCreate, TicketNotFound, find_tickets
from smartdesk_ai.tickets.sqlite_map import TicketStore
from smartdesk_ai.tools.definitions import CreateTicketArgs, GetTicketsArgs
from smartdesk_ai.tools.registry import ToolRegistry


def make_create_ticket(client: TicketClient) -> Callable[[CreateTicketArgs], Ticket]:
    def create_ticket(args: CreateTicketArgs) -> Ticket:
        # The executor has set request_id and checked the email before this runs (FR-G5, FR-G6).
        return client.create(TicketCreate(**args.model_dump()))

    return create_ticket


class TicketsResult(BaseModel):
    """The employee's tickets as GitHub reports them, in number order (FR-C2). `latest_update` is the
    newest team comment, fetched live, and is set only when exactly one ticket is returned.
    It is untrusted text (FR-C9): show it to a model only through `TeamUpdate.quoted()`."""

    model_config = ConfigDict(frozen=True)

    tickets: list[Ticket]
    latest_update: TeamUpdate | None = None


def make_get_tickets(client: TicketClient, store: TicketStore) -> Callable[[GetTicketsArgs], TicketsResult]:
    def get_tickets(args: GetTicketsArgs) -> TicketsResult:
        # The executor has checked the email before this runs (FR-G5).
        found = find_tickets(args.email, client, store)
        if args.ticket_number is not None:  # only among their own tickets: another employee's number finds nothing
            found = [t for t in found if t.number == args.ticket_number]
        update = None
        if len(found) == 1:
            try:
                update = client.latest_update(found[0].number)
            except TicketNotFound:
                update = None
        return TicketsResult(tickets=found, latest_update=update)

    return get_tickets


def register_ticket_tools(registry: ToolRegistry, client: TicketClient, store: TicketStore | None = None) -> None:
    registry.register("create_ticket", make_create_ticket(client), recover=client.find_by_request_id)
    if store is not None:
        registry.register("get_tickets", make_get_tickets(client, store))
