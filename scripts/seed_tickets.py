"""Seed the tickets repo with demo tickets for the status flow (FR-C8, FAQ Q17).

Goes through the same `tickets/` client the agent uses and writes the matching SQLite map rows.
Idempotent: each seed has a fixed request ID, so a second run finds its issue instead of creating
another, and only repairs what drifted (state, labels, a missing team comment, a missing map row).

Mock mode (the config default) keeps everything in memory. Live mode creates issues in the public
tickets repo, so it needs `modes.tickets: live`, GITHUB_TOKEN, TICKETS_REPO and `--confirm-live`.
`--dry-run` prints exactly what a run would create and touches nothing.
"""
import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartdesk_ai.config.env import load_env  # noqa: E402
from smartdesk_ai.config.settings import Settings, load_settings  # noqa: E402
from smartdesk_ai.tickets import (  # noqa: E402
    ALL_LABELS,
    Category,
    Priority,
    TicketClient,
    TicketCreate,
    TicketError,
    TicketStatus,
    build_issue,
    create_ticket_client,
)
from smartdesk_ai.tickets.sqlite_map import TicketStore  # noqa: E402

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "app.db"

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


def open_store(mode: str, db: str | None) -> TicketStore:
    """Live runs use the real map file. Mock runs use a throwaway one, so mock issue numbers never
    land in the map the live app reads."""
    if db:
        path = Path(db)
        path.parent.mkdir(parents=True, exist_ok=True)
        return TicketStore(path)
    if mode == "mock":
        return TicketStore(":memory:")
    DEFAULT_DB.parent.mkdir(parents=True, exist_ok=True)
    return TicketStore(DEFAULT_DB)


def print_plan() -> None:
    print(f"Labels ({len(ALL_LABELS)}): " + ", ".join(label.name for label in ALL_LABELS))
    print(f"Issues ({len(SEEDS)}), each with the line 'Employee: <email>' and a seed Request-ID:")
    for item in SEEDS:
        print(f"  {item.request_id}: {item.email} | {item.title} | {item.category.value}, "
              f"{item.priority.value} | final status {item.status.value} ({', '.join(item.labels())})"
              f" | team comment: {'yes' if item.comment else 'no'}")
    print(f"No tickets for {ZERO_EMAIL}. Create one issue by hand with the body line "
          f"'Employee: {HAND_MADE_EMAIL}' to test the search fallback.")


def main(
    argv: Sequence[str] | None = None, *, settings: Settings | None = None,
    client: TicketClient | None = None, store: TicketStore | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print what would be created and stop")
    parser.add_argument("--confirm-live", action="store_true", help="required when modes.tickets is live")
    parser.add_argument("--db", help="SQLite file for the email map (default data/app.db; mock: in memory)")
    args = parser.parse_args(argv)
    settings = settings or load_settings()
    mode = settings.modes.tickets
    if args.dry_run:
        print_plan()
        return 0
    if mode == "live" and not args.confirm_live:
        print("modes.tickets is live: this creates issues in the public tickets repo. "
              "Run with --dry-run to see them, then add --confirm-live.")
        return 2
    try:
        report = seed(client or create_ticket_client(settings, load_env()), store or open_store(mode, args.db))
    except TicketError as exc:
        print(f"Seeding stopped ({mode} mode): {exc}")
        return 1
    print(f"{mode} mode: {report.created} issues created, {report.updated} repaired, "
          f"{report.commented} comments added; the rest already matched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
