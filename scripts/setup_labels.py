"""Create the ticket labels in the tickets repo, once (FR-B11).

Uses `modes.tickets` from config.yaml: in mock mode nothing leaves this process. Going live means
setting `modes.tickets: live` plus GITHUB_TOKEN and TICKETS_REPO, and it creates labels in a
public repo, so confirm before you do. Running it again creates nothing new.
"""
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartdesk_ai.config.env import load_env  # noqa: E402
from smartdesk_ai.config.settings import Settings, load_settings  # noqa: E402
from smartdesk_ai.tickets import ALL_LABELS, TicketError, create_ticket_client  # noqa: E402


def main(argv: Sequence[str] | None = None, *, settings: Settings | None = None) -> int:
    settings = settings or load_settings()
    mode = settings.modes.tickets
    try:
        created = create_ticket_client(settings, load_env()).ensure_labels(ALL_LABELS)
    except TicketError as exc:
        print(f"Could not set up labels ({mode} mode): {exc}")
        return 1
    print(f"{mode} mode: {len(created)} of {len(ALL_LABELS)} labels created"
          + (f" ({', '.join(created)})" if created else ", the rest already existed"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
