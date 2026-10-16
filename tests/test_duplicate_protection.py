"""T3.7: duplicate protection for create_ticket (FR-G6). A lost create response never makes a second issue."""
import pytest

from smartdesk_ai.tickets.sqlite_map import TicketStore
from smartdesk_ai.tools.errors import TransientToolError
from smartdesk_ai.tools.executor import Outcome, ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry

TICKET = {"title": "VPN down", "description": "Cannot connect", "category": "IT",
          "email": "jane.roe@example.com"}


class FakeGitHub:
    """Records issues. `lose_responses` creates return after the issue exists but raise to the caller."""

    def __init__(self, lose_responses=0, fail_before_create=0):
        self.issues = []
        self.lose, self.fail_first = lose_responses, fail_before_create
        self.list_calls = self.search_calls = 0
        self.pending_seen = []
        self.store = None

    def create(self, args):
        self.pending_seen.append(self.store.get(args.request_id))
        if self.fail_first:
            self.fail_first -= 1
            raise TransientToolError("503 before anything was created")
        issue = {"number": len(self.issues) + 1, "body": f"{args.description}\nRequest ID: {args.request_id}"}
        self.issues.append(issue)
        if self.lose:
            self.lose -= 1
            raise TransientToolError("response lost")
        return issue

    def list_issues(self):
        self.list_calls += 1
        return list(self.issues)

    def search_issues(self, *_):
        self.search_calls += 1
        raise AssertionError("search can lag; FR-G6 requires list-issues")

    def recover(self, request_id):
        return next((i for i in self.list_issues() if f"Request ID: {request_id}" in i["body"]), None)


@pytest.fixture
def world(settings, tmp_path):
    store = TicketStore(tmp_path / "tickets.db")

    def make(recover=None, **kw):
        gh = FakeGitHub(**kw)
        gh.store = store
        registry = ToolRegistry()
        registry.register("create_ticket", gh.create, recover=recover or gh.recover)
        return gh, ToolExecutor(registry, settings, sleep=lambda s: None, request_store=store)

    return store, make


def create(executor):
    return executor.execute("create_ticket", TICKET, agent="ticket", confirmed=True)


def test_lost_response_then_retry_leaves_exactly_one_issue(world):
    store, make = world
    gh, executor = make(lose_responses=1)
    result = create(executor)
    assert len(gh.issues) == 1
    assert result.outcome is Outcome.SUCCESS and result.attempts == 2
    assert result.data["number"] == 1


def test_recovery_check_uses_list_issues_not_search(world):
    _, make = world
    gh, executor = make(lose_responses=1)
    create(executor)
    assert gh.list_calls == 1 and gh.search_calls == 0


def test_request_id_is_written_to_sqlite_before_the_call(world):
    store, make = world
    gh, executor = make()
    create(executor)
    assert gh.pending_seen == [("pending", None)]
    rid = gh.issues[0]["body"].split("Request ID: ")[1]
    assert store.get(rid) == ("done", 1)


def test_failure_before_creation_retries_and_creates_once(world):
    _, make = world
    gh, executor = make(fail_before_create=1)
    result = create(executor)
    assert len(gh.issues) == 1 and result.attempts == 2 and result.outcome is Outcome.SUCCESS


def test_every_attempt_lost_still_one_issue(world):
    _, make = world
    gh, executor = make(lose_responses=5)
    result = create(executor)
    assert len(gh.issues) == 1 and result.outcome is Outcome.SUCCESS


def test_failing_recovery_check_never_triggers_a_blind_second_create(world):
    _, make = world

    def broken(request_id):
        raise TransientToolError("list failed")

    gh, executor = make(lose_responses=1, recover=broken)
    result = create(executor)
    assert len(gh.issues) == 1 and result.outcome is Outcome.UNAVAILABLE


def test_model_supplied_request_id_is_ignored_and_ids_are_unique(world):
    _, make = world
    gh, executor = make()
    executor.execute("create_ticket", {**TICKET, "request_id": "evil"}, agent="t", confirmed=True)
    create(executor)
    ids = [i["body"].split("Request ID: ")[1] for i in gh.issues]
    assert "evil" not in ids and len(set(ids)) == 2


def test_other_tools_get_no_request_id(world, settings):
    store, _ = world
    registry = ToolRegistry()
    seen = []
    registry.register("search_kb", lambda a: seen.append(a))
    ToolExecutor(registry, settings, request_store=store).execute("search_kb", {"query": "x"}, agent="a")
    assert not hasattr(seen[0], "request_id")


def test_store_survives_reopen(tmp_path):
    path = tmp_path / "t.db"
    s = TicketStore(path)
    s.begin("req-1")
    s.complete("req-1", 7)
    assert TicketStore(path).get("req-1") == ("done", 7)
    assert TicketStore(path).get("missing") is None
