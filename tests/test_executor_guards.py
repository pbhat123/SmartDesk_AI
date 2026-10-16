"""T3.6: executor guards: schema validation, confirmation before create_ticket, email check (FR-G5, FR-B2)."""
import pytest

from smartdesk_ai.tools.executor import ErrorCode, Outcome, ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry

TICKET = {"title": "VPN down", "description": "Cannot connect from home", "category": "IT",
          "email": "jane.roe@example.com"}


class Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, args):
        self.calls.append(args)
        return "ran"


@pytest.fixture
def spies(settings):
    registry = ToolRegistry()
    spy = {n: Spy() for n in ("search_kb", "create_ticket", "get_tickets")}
    for name, fn in spy.items():
        registry.register(name, fn)
    return ToolExecutor(registry, settings, sleep=lambda s: None), spy


def create(executor, args=None, **kw):
    return executor.execute("create_ticket", args or TICKET, agent="ticket", **kw)


def test_create_ticket_without_confirmation_does_not_run(spies):
    executor, spy = spies
    result = create(executor)
    assert result.outcome is Outcome.VALIDATION_ERROR
    assert result.code is ErrorCode.CONFIRMATION_REQUIRED
    assert spy["create_ticket"].calls == []


def test_create_ticket_with_confirmation_runs_once(spies):
    executor, spy = spies
    result = create(executor, confirmed=True)
    assert result.outcome is Outcome.SUCCESS
    assert len(spy["create_ticket"].calls) == 1


@pytest.mark.parametrize("email", ["not-an-email", "a@b", "two words@x.com", "a@@b.com"])
def test_invalid_email_returns_ask_for_email_error(spies, email):
    executor, spy = spies
    result = create(executor, {**TICKET, "email": email}, confirmed=True)
    assert result.outcome is Outcome.VALIDATION_ERROR
    assert result.code is ErrorCode.EMAIL_INVALID
    assert "ask" in result.error.lower() and "email" in result.error.lower()
    assert spy["create_ticket"].calls == []


def test_missing_email_returns_ask_for_email_error(spies):
    executor, spy = spies
    args = {k: v for k, v in TICKET.items() if k != "email"}
    result = create(executor, args, confirmed=True)
    assert result.code is ErrorCode.EMAIL_REQUIRED
    assert "ask" in result.error.lower()
    assert spy["create_ticket"].calls == []


def test_blank_email_counts_as_missing(spies):
    executor, _ = spies
    assert create(executor, {**TICKET, "email": ""}, confirmed=True).code is ErrorCode.EMAIL_REQUIRED


def test_get_tickets_also_needs_an_email(spies):
    executor, spy = spies
    result = executor.execute("get_tickets", {}, agent="ticket")
    assert result.code is ErrorCode.EMAIL_REQUIRED and spy["get_tickets"].calls == []
    ok = executor.execute("get_tickets", {"email": "jane.roe@example.com"}, agent="ticket")
    assert ok.outcome is Outcome.SUCCESS


def test_schema_errors_return_validation_error_without_running(spies):
    executor, spy = spies
    result = create(executor, {**TICKET, "category": "FINANCE"}, confirmed=True)
    assert result.outcome is Outcome.VALIDATION_ERROR and result.code is ErrorCode.SCHEMA
    assert "category" in result.error
    assert spy["create_ticket"].calls == []


def test_error_messages_never_echo_the_email(spies):
    executor, _ = spies
    result = create(executor, {**TICKET, "email": "bad@@x.com", "category": "FINANCE"}, confirmed=True)
    assert "bad@@x.com" not in (result.error or "")


def test_confirmed_flag_is_not_a_model_argument(spies):
    executor, spy = spies
    result = create(executor, {**TICKET, "confirmed": True})
    assert result.code is ErrorCode.SCHEMA and spy["create_ticket"].calls == []


def test_search_kb_needs_no_confirmation_or_email(spies):
    executor, _ = spies
    assert executor.execute("search_kb", {"query": "vpn"}, agent="it").outcome is Outcome.SUCCESS
