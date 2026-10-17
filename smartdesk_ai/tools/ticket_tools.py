"""Connects the ticket client to the tool registry (FR-G3, FR-B12, FR-B13).

`create_ticket` returns the `Ticket` exactly as the client read it from GitHub. Nothing here adds
or fills in a number, link or status; if the response lacks one, the client raises.
"""
from collections.abc import Callable

from smartdesk_ai.tickets import Ticket, TicketClient, TicketCreate
from smartdesk_ai.tools.definitions import CreateTicketArgs
from smartdesk_ai.tools.registry import ToolRegistry


def make_create_ticket(client: TicketClient) -> Callable[[CreateTicketArgs], Ticket]:
    def create_ticket(args: CreateTicketArgs) -> Ticket:
        # The executor has set request_id and checked the email before this runs (FR-G5, FR-G6).
        return client.create(TicketCreate(**args.model_dump()))

    return create_ticket


def register_ticket_tools(registry: ToolRegistry, client: TicketClient) -> None:
    registry.register("create_ticket", make_create_ticket(client), recover=client.find_by_request_id)
