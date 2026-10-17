"""Ticket access for the rest of the app (FR-B12). Import from here, not from `github_client`."""
from smartdesk_ai.tickets.github_client import (
    GitHubRestClient,
    MockTicketClient,
    TicketClient,
    build_issue,
    create_ticket_client,
    employee_email_of,
    request_id_of,
)
from smartdesk_ai.tickets.schemas import (
    ALL_LABELS,
    Category,
    Priority,
    Ticket,
    TicketConfigError,
    TicketCreate,
    TicketError,
    TicketNotFound,
    TransientTicketError,
)
from smartdesk_ai.tickets.search_fallback import find_tickets
from smartdesk_ai.tickets.status import TicketStatus, derive_status

__all__ = [
    "ALL_LABELS", "Category", "GitHubRestClient", "MockTicketClient", "Priority", "Ticket", "TicketClient",
    "TicketConfigError", "TicketCreate", "TicketError", "TicketNotFound", "TicketStatus", "TransientTicketError",
    "build_issue", "create_ticket_client", "derive_status", "employee_email_of", "find_tickets", "request_id_of",
]
