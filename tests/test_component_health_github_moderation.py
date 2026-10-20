"""GitHub tickets and Moderation in the health registry and on the Monitoring page (FR-F7, FR-G10)."""
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from smartdesk_ai.guardrails.input_guard import InputGuard
from smartdesk_ai.guardrails.moderation import KeywordModerator, ModerationResult
from smartdesk_ai.guardrails.names import NameReducer
from smartdesk_ai.guardrails.output_guard import OutputGuard
from smartdesk_ai.guardrails.pii import PiiRedactor
from smartdesk_ai.ops.health import Component, HealthRegistry, Status
from smartdesk_ai.tickets import GitHubRestClient, MockTicketClient, TicketError, TransientTicketError
from smartdesk_ai.tickets.health_client import HealthTrackedClient
from smartdesk_ai.tickets.schemas import TicketCreate
from smartdesk_ai.ui.monitoring import load_monitoring
from tests.ui_helpers import element_text, mock_context, monitoring_app

REPO = "octo/tickets"
DRAFT = TicketCreate(title="VPN down", description="Cannot connect", category="IT",
                     email="jane.roe@example.com", request_id="req-1")


@pytest.fixture()
def registry():
    return HealthRegistry(clock=lambda: datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc))


def tracked_rest(handler, settings, registry):
    client = GitHubRestClient(REPO, SecretStr("ghp_secret"), settings.github, transport=httpx.MockTransport(handler))
    return HealthTrackedClient(client, registry)


# --- GitHub ----------------------------------------------------------------------------------------------------

def test_an_unreachable_github_degrades_with_a_reason_and_recovers(settings, registry):
    state = {"up": False}

    def handler(request):
        if not state["up"]:
            raise httpx.ConnectError("no route", request=request)
        return httpx.Response(200, json={"number": 7, "html_url": "https://github.com/octo/tickets/issues/7",
                                         "title": "t", "body": "b", "state": "open", "labels": []})

    client = tracked_rest(handler, settings, registry)
    with pytest.raises(TransientTicketError):
        client.get(7)
    health = registry.get(Component.GITHUB)
    assert health.status is Status.DEGRADED and health.reason == "could not reach GitHub"
    assert [h.component for h in registry.degraded_modes()] == [Component.GITHUB]
    assert "ghp_secret" not in repr(health)
    state["up"] = True
    assert client.get(7).number == 7
    assert registry.get(Component.GITHUB).status is Status.OK


def test_a_server_error_or_a_refused_token_degrades_but_not_found_does_not(settings, registry):
    for code, degraded in ((503, True), (401, True), (404, False)):
        reg = HealthRegistry()
        client = tracked_rest(lambda request, code=code: httpx.Response(code, json={}), settings, reg)
        with pytest.raises((TicketError, TransientTicketError)):
            client.get(1)
        assert (reg.get(Component.GITHUB).status is Status.DEGRADED) is degraded, code


def test_the_mock_client_can_simulate_an_outage_through_the_tracker(registry):
    mock = MockTicketClient()
    client = HealthTrackedClient(mock, registry)
    assert client.create(DRAFT).number >= 1 and registry.get(Component.GITHUB).status is Status.OK
    mock.unreachable = True
    with pytest.raises(TransientTicketError):
        client.get(1)
    assert registry.get(Component.GITHUB).status is Status.DEGRADED
    assert client.add_issue  # attributes the wrapped client has are still reachable


def test_mock_tickets_are_listed_as_mock_mode_not_degraded(tmp_path):
    ctx = mock_context(tmp_path)
    health = ctx.registry.get(Component.GITHUB)
    assert health.status is Status.MOCK and Component.GITHUB not in [h.component for h in ctx.registry.degraded_modes()]


# --- Moderation ------------------------------------------------------------------------------------------------

class Down:
    def check(self, text):
        raise ConnectionError("moderation API down")


class Flaky:
    def __init__(self):
        self.up = False

    def check(self, text):
        if not self.up:
            raise ConnectionError("down")
        return ModerationResult(flagged=False)


def test_an_input_moderation_outage_degrades_and_a_good_check_recovers(registry):
    moderator = Flaky()
    guard = InputGuard(pii=PiiRedactor(None, 0.5), names=NameReducer(None, 0.5), moderator=moderator, registry=registry)
    assert not guard("hello").blocked  # the outage never stops the service
    health = registry.get(Component.MODERATION)
    assert health.status is Status.DEGRADED and health.reason == "ConnectionError"
    moderator.up = True
    guard("hello again")
    assert registry.get(Component.MODERATION).status is Status.OK


def test_an_output_moderation_outage_degrades_too(registry):
    guard = OutputGuard(PiiRedactor(None, 0.5), Down(), registry=registry)
    assert guard.check("A reply.").text == "A reply."
    assert registry.get(Component.MODERATION).status is Status.DEGRADED
    moderator = Flaky()
    moderator.up = True
    OutputGuard(PiiRedactor(None, 0.5), moderator, registry=registry).check("A reply.")
    assert registry.get(Component.MODERATION).status is Status.OK


def test_the_keyword_stand_in_is_mock_mode_not_ok(registry):
    OutputGuard(PiiRedactor(None, 0.5), KeywordModerator(), registry=registry).check("A reply.")
    state = registry.get(Component.MODERATION)
    assert state.status is Status.MOCK and Component.MODERATION not in [h.component for h in registry.degraded_modes()]


# --- the page --------------------------------------------------------------------------------------------------

def test_the_components_list_names_github_and_moderation_with_their_badges(tmp_path):
    ctx = mock_context(tmp_path)
    ctx.service.handle("t", "How do I reset my forgotten password?")  # one guarded turn: moderation was checked
    rows = {c.component: c for c in load_monitoring(ctx.settings, tmp_path, ctx.kb_dir, ctx.registry,
                                                    results_dir=ctx.results_dir).components}
    assert rows["github"].status == "mock" and rows["moderation"].status == "mock"
    page = monitoring_app(ctx)
    shown = " ".join(m.value for m in page.main.markdown)
    assert "**GitHub tickets** :blue-badge[mock mode]" in shown and "**Moderation** :blue-badge[mock mode (keyword stand-in)]" in shown
    assert not page.exception


def test_a_degraded_github_shows_orange_with_its_reason(tmp_path):
    ctx = mock_context(tmp_path)
    ctx.registry.mark_degraded(Component.GITHUB, "could not reach GitHub")
    page = monitoring_app(ctx)
    shown = " ".join(m.value for m in page.main.markdown) + element_text(page.main)
    assert "**GitHub tickets** :orange-badge[degraded]" in shown and "could not reach GitHub" in shown


# --- the live-mode factory and the page captions -----------------------------------------------------------------

def test_the_live_mode_factory_wraps_the_ticket_client_with_the_health_wrapper(settings, tmp_path, monkeypatch):
    from smartdesk_ai.agent import factory

    wrapped = []
    real = factory.HealthTrackedClient
    monkeypatch.setattr(factory, "HealthTrackedClient", lambda inner, reg: wrapped.append((inner, reg)) or real(inner, reg))
    client = MockTicketClient()  # stands in for the REST client: only the wrapping is under test
    live_tickets = settings.model_copy(update={"modes": settings.modes.model_copy(update={"tickets": "live"})})
    registry = HealthRegistry()
    factory.build_stack(live_tickets, environ={}, dotenv_path=None, data_dir=tmp_path, registry=registry, client=client)
    assert wrapped == [(client, registry)]
    assert registry.get(Component.GITHUB).status is Status.UNKNOWN  # not mock: it is checked on the first call


def test_mock_tickets_are_not_wrapped(settings, tmp_path, monkeypatch):
    from smartdesk_ai.agent import factory

    monkeypatch.setattr(factory, "HealthTrackedClient", lambda *a: pytest.fail("mock tickets must not be wrapped"))
    factory.build_stack(settings, environ={}, dotenv_path=None, data_dir=tmp_path, registry=HealthRegistry())


def test_cost_by_agent_says_it_counts_app_sessions_only(tmp_path):
    ctx = mock_context(tmp_path)
    ctx.service.handle("t", "How do I reset my forgotten password?")
    shown = " ".join(c.value for c in monitoring_app(ctx).main.caption)
    assert "App sessions only; evals are in Spend." in shown


# --- the live moderator gets the plain API key (the cause of the "Degraded: moderation" notice) -------------------

@pytest.mark.parametrize("module", ["input_guard", "output_guard"])
def test_the_live_guards_give_the_openai_client_the_plain_key_not_a_masked_secret(settings, monkeypatch, module):
    """A SecretStr handed to the SDK is sent as "**********": every moderation call got a 401 and the registry (rightly)
    showed Moderation degraded. Both live guards must unwrap the key."""
    import importlib

    import openai

    seen = {}

    class Recorder:
        def __init__(self, api_key=None, **kwargs):
            seen["key"] = api_key

    monkeypatch.setattr(openai, "OpenAI", Recorder)
    guard_module = importlib.import_module(f"smartdesk_ai.guardrails.{module}")
    monkeypatch.setattr(guard_module, "make_presidio_engine", lambda registry=None: None)
    live = settings.model_copy(update={"modes": settings.modes.model_copy(update={"llm": "live"})})
    build = getattr(guard_module, f"build_{module}")
    args = (live, Path(".")) if module == "output_guard" else (live,)
    build(*args, environ={"OPENAI_API_KEY": "sk-test-plain-key"}, dotenv_path=None, registry=HealthRegistry())
    assert seen["key"] == "sk-test-plain-key" and isinstance(seen["key"], str)
