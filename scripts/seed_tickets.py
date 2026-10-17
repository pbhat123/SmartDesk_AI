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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartdesk_ai.config.env import load_env  # noqa: E402
from smartdesk_ai.config.settings import Settings, load_settings  # noqa: E402
from smartdesk_ai.tickets import ALL_LABELS, TicketClient, TicketError, create_ticket_client  # noqa: E402
from smartdesk_ai.tickets.seed import (  # noqa: E402,F401  (re-exported: tests and callers use these names)
    CLOSED_EMAIL,
    EMAILS,
    HAND_MADE_EMAIL,
    MANY_EMAIL,
    ONE_EMAIL,
    SEEDS,
    ZERO_EMAIL,
    SeedReport,
    SeedTicket,
    seed,
)
from smartdesk_ai.tickets.sqlite_map import TicketStore  # noqa: E402

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "app.db"


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
