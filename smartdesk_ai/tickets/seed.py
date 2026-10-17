"""The demo tickets for the status flow (FR-C8, FAQ Q17) and the idempotent `seed` that creates them.

Lives in the package so both `scripts/seed_tickets.py` and the mock CLI use one definition. It goes through
the same `tickets/` client the agent uses and writes the matching map rows. Each seed has a fixed request
ID, so a second run finds its issue instead of creating another, and only repairs what drifted (state,
labels, a missing team comment, a missing map row).
"""
from pydantic import BaseModel, ConfigDict

from smartdesk_ai.tickets.github_client import TicketClient, build_issue
from smartdesk_ai.tickets.schemas import ALL_LABELS, Category, Priority, TicketCreate
from smartdesk_ai.tickets.sqlite_map import TicketStore
from smartdesk_ai.tickets.status import TicketStatus

ZERO_EMAIL = "zero.tickets@smartdesk.example"      # has no tickets
ONE_EMAIL = "one.ticket@smartdesk.example"         # exactly one
MANY_EMAIL = "two.tickets@smartdesk.example"       # two
CLOSED_EMAIL = "closed.ticket@smartdesk.example"   # one ticket that was closed without a fix
EMAILS = (ZERO_EMAIL, ONE_EMAIL, MANY_EMAIL, CLOSED_EMAIL)
HAND_MADE_EMAIL = "hand.made@smartdesk.example"    # for the by-hand issue that tests the search fallback


class SeedTicket(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    email: str
    title: str
    description: str
    category: Category
    priority: Priority
    status: TicketStatus
    comment: str | None = None

    @property
    def request_id(self) -> str:
        return f"seed-{self.key}"

    @property
    def state(self) -> str:
        return "open" if self.status in (TicketStatus.OPEN, TicketStatus.IN_PROGRESS) else "closed"

    def labels(self) -> list[str]:
        draft = TicketCreate(title=self.title, description=self.description, category=self.category,
                             priority=self.priority, email=self.email, request_id=self.request_id)
        extra = {TicketStatus.IN_PROGRESS: ["in-progress"], TicketStatus.RESOLVED: ["resolved"]}
        return [*build_issue(draft).labels, *extra.get(self.status, [])]


SEEDS: tuple[SeedTicket, ...] = (
    SeedTicket(
        key="monitor-flicker", email=ONE_EMAIL, title="Monitor flickers after docking",
        description="My external monitor flickers every few seconds once the laptop is on the dock.",
        category=Category.IT, priority=Priority.MEDIUM, status=TicketStatus.IN_PROGRESS,
        comment="We are testing a replacement docking cable and will update you tomorrow.",
    ),
    SeedTicket(
        key="vpn-drops", email=MANY_EMAIL, title="VPN drops every hour from home",
        description="The VPN disconnects roughly once an hour while working from home.",
        category=Category.IT, priority=Priority.HIGH, status=TicketStatus.RESOLVED,
        comment="Fixed: the VPN profile was updated. Please reconnect and let us know if it persists.",
    ),
    SeedTicket(
        key="payslip-question", email=MANY_EMAIL, title="Question about a payslip deduction",
        description="There is a deduction on my last payslip that I do not recognise.",
        category=Category.HR, priority=Priority.LOW, status=TicketStatus.OPEN,
    ),
    SeedTicket(
        key="monitor-stand", email=CLOSED_EMAIL, title="Request for a second monitor stand",
        description="I would like a second monitor stand for my desk.",
        category=Category.MISC, priority=Priority.LOW, status=TicketStatus.CLOSED,
        comment="Closing this: monitor stands are available from the facilities cabinet.",
    ),
)


class SeedReport(BaseModel):
    created: int = 0
    updated: int = 0
    commented: int = 0


def seed(client: TicketClient, store: TicketStore) -> SeedReport:
    """Bring the repo to the seeded state. Safe to repeat."""
    client.ensure_labels(ALL_LABELS)
    report = SeedReport()
    for item in SEEDS:
        ticket = client.find_by_request_id(item.request_id)
        if ticket is None:
            ticket = client.create(TicketCreate(
                title=item.title, description=item.description, category=item.category,
                priority=item.priority, email=item.email, request_id=item.request_id,
            ))
            report.created += 1
        store.link_email(item.email, ticket.number)
        wanted = item.labels()
        if ticket.state != item.state or {x.lower() for x in ticket.labels} != {x.lower() for x in wanted}:
            client.update_issue(ticket.number, state=item.state, labels=wanted)
            report.updated += 1
        if item.comment and client.latest_update(ticket.number) is None:
            client.add_comment(ticket.number, item.comment)
            report.commented += 1
    return report
