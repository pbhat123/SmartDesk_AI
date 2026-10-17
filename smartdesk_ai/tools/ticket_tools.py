"""Connects the ticket client to the tool registry (FR-G3, FR-B12, FR-B13).

`create_ticket` returns the `Ticket` exactly as the client read it from GitHub. Nothing here adds
or fills in a number, link or status; if the response lacks one, the client raises.
"""
from collections.abc import Callable

from smartdesk_ai.tickets import Ticket, TicketClient, TicketCreate, find_tickets
from smartdesk_ai.tickets.sqlite_map import TicketStore
from smartdesk_ai.tools.definitions import CreateTicketArgs, GetTicketsArgs
from smartdesk_ai.tools.registry import ToolRegistry


def make_create_ticket(client: TicketClient) -> Callable[[CreateTicketArgs], Ticket]:
    def create_ticket(args: CreateTicketArgs) -> Ticket:
        # The executor has set request_id and checked the email before this runs (FR-G5, FR-G6).
        return client.create(TicketCreate(**args.model_dump()))

    return create_ticket


def make_get_tickets(client: TicketClient, store: TicketStore) -> Callable[[GetTicketsArgs], list[Ticket]]:
    def get_tickets(args: GetTicketsArgs) -> list[Ticket]:
        # The executor has checked the email before this runs (FR-G5).
        return find_tickets(args.email, client, store)

    return get_tickets


def register_ticket_tools(registry: ToolRegistry, client: TicketClient, store: TicketStore | None = None) -> None:
    registry.register("create_ticket", make_create_ticket(client), recover=client.find_by_request_id)
    if store is not None:
        registry.register("get_tickets", make_get_tickets(client, store))
