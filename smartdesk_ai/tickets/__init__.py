"""Ticket access for the rest of the app (FR-B12). Import from here, not from `github_client`."""
from smartdesk_ai.tickets.github_client import (
    GitHubRestClient,
    MockTicketClient,
    TicketClient,
    create_ticket_client,
)
from smartdesk_ai.tickets.schemas import (
    Category,
    Priority,
    Ticket,
    TicketConfigError,
    TicketCreate,
    TicketError,
    TicketNotFound,
    TransientTicketError,
)

__all__ = [
    "Category", "GitHubRestClient", "MockTicketClient", "Priority", "Ticket", "TicketClient",
    "TicketConfigError", "TicketCreate", "TicketError", "TicketNotFound", "TransientTicketError",
    "create_ticket_client",
]
