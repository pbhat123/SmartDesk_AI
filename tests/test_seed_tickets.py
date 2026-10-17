"""T3.17: seed script (FR-C8, FR-C11, FAQ Q17). Mock mode and a fake GitHub behind MockTransport only.

Nothing here reaches GitHub; the live run is a separate, confirmed step.
"""
import importlib.util
import json
import re
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from smartdesk_ai.tickets import (
    ALL_LABELS,
    Category,
    GitHubRestClient,
    MockTicketClient,
    TicketCreate,
    TicketStatus,
    find_tickets,
)
from smartdesk_ai.tickets.sqlite_map import TicketStore

ROOT = Path(__file__).resolve().parent.parent
REPO = "octo/tickets"


def load_script():
    spec = importlib.util.spec_from_file_location("seed_tickets", ROOT / "scripts" / "seed_tickets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seed_script = load_script()


@pytest.fixture
def store():
    return TicketStore(":memory:")


# --- mock mode ---------------------------------------------------------------

def test_running_twice_gives_the_same_issue_count(store):
    client = MockTicketClient()
    first = seed_script.seed(client, store)
    count = len(client._issues)
    second = seed_script.seed(client, store)
    assert len(client._issues) == count == len(seed_script.SEEDS)
    assert first.created == count and second.created == 0


def test_second_run_adds_no_comments_and_changes_nothing(store):
    client = MockTicketClient()
    seed_script.seed(client, store)
    comments = {k: list(v) for k, v in client._comments.items()}
    again = seed_script.seed(client, store)
    assert client._comments == comments
    assert (again.updated, again.commented) == (0, 0)


def test_each_seeded_email_finds_its_tickets(store):
    client = MockTicketClient()
    seed_script.seed(client, store)
    counts = {email: len(find_tickets(email, client, store)) for email in seed_script.EMAILS}
    assert counts[seed_script.ZERO_EMAIL] == 0
    assert counts[seed_script.ONE_EMAIL] == 1
    assert counts[seed_script.MANY_EMAIL] == 2
    assert counts[seed_script.CLOSED_EMAIL] == 1


def test_map_rows_are_written_for_every_seeded_ticket(store):
    client = MockTicketClient()
    seed_script.seed(client, store)
    assert len(store.numbers_for(seed_script.MANY_EMAIL)) == 2
    assert store.numbers_for(seed_script.ZERO_EMAIL) == []


def test_every_status_is_present(store):
    client = MockTicketClient()
    seed_script.seed(client, store)
    statuses = {t.status for t in client._issues.values()}
    assert statuses == set(TicketStatus)


def test_the_status_matches_what_each_seed_asked_for(store):
    client = MockTicketClient()
    seed_script.seed(client, store)
    for seed in seed_script.SEEDS:
        (ticket,) = [t for t in find_tickets(seed.email, client, store) if t.title == seed.title]
        assert ticket.status is seed.status


def test_team_comments_are_in_place_and_come_from_the_team(store):
    client = MockTicketClient()
    seed_script.seed(client, store)
    for seed in seed_script.SEEDS:
        (ticket,) = [t for t in find_tickets(seed.email, client, store) if t.title == seed.title]
        update = client.latest_update(ticket.number)
        assert (update.text if update else None) == seed.comment


def test_a_drifted_issue_is_repaired_without_a_duplicate(store):
    client = MockTicketClient()
    seed_script.seed(client, store)
    resolved = next(t for t in client._issues.values() if t.status is TicketStatus.RESOLVED)
    client.update_issue(resolved.number, state="open", labels=["IT"])
    again = seed_script.seed(client, store)
    assert client.get(resolved.number).status is TicketStatus.RESOLVED
    assert again.updated == 1 and len(client._issues) == len(seed_script.SEEDS)


def test_a_missing_map_row_is_rewritten(store):
    client = MockTicketClient()
    seed_script.seed(client, store)
    fresh = TicketStore(":memory:")
    seed_script.seed(client, fresh)
    assert len(fresh.numbers_for(seed_script.MANY_EMAIL)) == 2 and len(client._issues) == len(seed_script.SEEDS)


def test_labels_are_created_first(store):
    client = MockTicketClient()
    seed_script.seed(client, store)
    assert client.ensure_labels(ALL_LABELS) == []


def test_only_fake_emails_are_used():
    assert all(e.endswith("@smartdesk.example") for e in seed_script.EMAILS)
    assert all(s.email in seed_script.EMAILS for s in seed_script.SEEDS)


def test_seeds_cover_zero_one_and_two_or_more_tickets():
    per_email = {e: sum(s.email == e for s in seed_script.SEEDS) for e in seed_script.EMAILS}
    assert per_email[seed_script.ZERO_EMAIL] == 0 and per_email[seed_script.ONE_EMAIL] == 1
    assert per_email[seed_script.MANY_EMAIL] >= 2


def test_seed_request_ids_are_unique_and_stable():
    ids = [s.request_id for s in seed_script.SEEDS]
    assert len(set(ids)) == len(ids) and all(i.startswith("seed-") and " " not in i for i in ids)


# --- the same flow over REST, against a small fake GitHub -------------------------------

class FakeGitHub:
    """Just enough of the issues, comments and labels endpoints, with state."""

    def __init__(self):
        self.issues, self.comments, self.labels, self.log = {}, {}, set(), []

    def issue_json(self, n):
        i = self.issues[n]
        return {"number": n, "html_url": f"https://github.com/{REPO}/issues/{n}", "title": i["title"],
                "body": i["body"], "state": i["state"], "labels": [{"name": x} for x in i["labels"]]}

    def __call__(self, request):
        method, path = request.method, request.url.path
        self.log.append((method, path))
        body = json.loads(request.content) if request.content else {}
        if (method, path) == ("POST", f"/repos/{REPO}/issues"):
            n = len(self.issues) + 1
            self.issues[n] = {"title": body["title"], "body": body["body"], "state": "open",
                              "labels": list(body["labels"])}
            return httpx.Response(201, json=self.issue_json(n))
        if (method, path) == ("GET", f"/repos/{REPO}/issues"):
            ordered = sorted(self.issues, reverse=True)
            return httpx.Response(200, json=[self.issue_json(n) for n in ordered])
        if m := re.fullmatch(rf"/repos/{REPO}/issues/(\d+)", path):
            n = int(m[1])
            if method == "PATCH":
                self.issues[n].update({k: v for k, v in body.items() if k in ("state", "labels")})
            return httpx.Response(200, json=self.issue_json(n))
        if m := re.fullmatch(rf"/repos/{REPO}/issues/(\d+)/comments", path):
            n = int(m[1])
            if method == "POST":
                self.comments.setdefault(n, []).append(body["body"])
                return httpx.Response(201, json={})
            return httpx.Response(200, json=[
                {"body": b, "created_at": f"2026-10-14T09:00:{i:02d}Z", "author_association": "OWNER"}
                for i, b in enumerate(self.comments.get(n, []))])
        if path == f"/repos/{REPO}/labels":
            if method == "POST":
                self.labels.add(body["name"])
                return httpx.Response(201, json={})
            return httpx.Response(200, json=[{"name": n} for n in sorted(self.labels)])
        if path == "/search/issues":
            q = request.url.params["q"]
            hits = [n for n, i in self.issues.items() if q.split('"')[1].split("Employee: ")[1] in i["body"]]
            return httpx.Response(200, json={"items": [self.issue_json(n) for n in hits]})
        return httpx.Response(404, json={})


def rest_world(settings):
    fake = FakeGitHub()
    client = GitHubRestClient(REPO, SecretStr("t"), settings.github, transport=httpx.MockTransport(fake))
    return fake, client


def test_rest_seed_twice_gives_the_same_issues_and_no_second_create(settings, store):
    fake, client = rest_world(settings)
    seed_script.seed(client, store)
    assert len(fake.issues) == len(seed_script.SEEDS)
    fake.log.clear()
    again = seed_script.seed(client, store)
    assert len(fake.issues) == len(seed_script.SEEDS) and again.created == 0
    assert ("POST", f"/repos/{REPO}/issues") not in fake.log
    assert not any(m == "POST" and p.endswith("/comments") for m, p in fake.log)


def test_rest_closed_issues_carry_their_status_labels(settings, store):
    fake, client = rest_world(settings)
    seed_script.seed(client, store)
    got = {i["title"]: (i["state"], set(i["labels"])) for i in fake.issues.values()}
    resolved = next(s for s in seed_script.SEEDS if s.status is TicketStatus.RESOLVED)
    assert got[resolved.title][0] == "closed" and "resolved" in got[resolved.title][1]
    in_progress = next(s for s in seed_script.SEEDS if s.status is TicketStatus.IN_PROGRESS)
    assert got[in_progress.title] == ("open", {in_progress.category.value, in_progress.priority.value, "in-progress"})


def test_rest_each_email_finds_its_tickets_after_seeding(settings, store):
    _, client = rest_world(settings)
    seed_script.seed(client, store)
    assert len(find_tickets(seed_script.MANY_EMAIL, client, store)) == 2
    assert find_tickets(seed_script.ZERO_EMAIL, client, store) == []


# --- main ------------------------------------------------------------------------

def test_dry_run_prints_the_plan_and_calls_nothing(settings, capsys):
    class Exploding:
        def __getattr__(self, name):
            raise AssertionError(f"dry run touched the client: {name}")

    assert seed_script.main(["--dry-run"], settings=settings, client=Exploding()) == 0
    out = capsys.readouterr().out
    assert all(label.name in out for label in ALL_LABELS)
    assert all(s.email in out and s.title in out for s in seed_script.SEEDS)


def test_mock_mode_main_runs_and_keeps_nothing_on_disk(settings, capsys):
    assert seed_script.open_store("mock", None).numbers_for("x@y.zz") == []
    assert seed_script.main([], settings=settings) == 0
    assert "mock" in capsys.readouterr().out


def test_mock_mode_uses_a_throwaway_store_so_mock_numbers_never_reach_the_real_map():
    store = seed_script.open_store("mock", None)
    store.link_email("a@b.cc", 1)
    assert seed_script.open_store("mock", None).numbers_for("a@b.cc") == []


def test_live_mode_needs_an_explicit_flag(settings, capsys):
    live = settings.model_copy(update={"modes": settings.modes.model_copy(update={"tickets": "live"})})

    class Exploding:
        def __getattr__(self, name):
            raise AssertionError("live run started without the flag")

    assert seed_script.main([], settings=live, client=Exploding()) == 2
    assert "--confirm-live" in capsys.readouterr().out
