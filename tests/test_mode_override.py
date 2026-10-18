"""UI review fix 7: SMARTDESK_LLM_MODE and SMARTDESK_TICKETS_MODE override `modes.llm` and `modes.tickets` for one run,
so the app can run a live LLM with mock tickets without editing config.yaml (FR-B10, FR-H16)."""
import pytest

from smartdesk_ai.agent.chat_service import build_service
from smartdesk_ai.config.settings import ConfigError, load_settings
from smartdesk_ai.ui.context import build_ui_context


def test_without_the_variables_config_yaml_decides():
    settings = load_settings(environ={})
    assert (settings.modes.llm, settings.modes.tickets) == ("mock", "mock")  # what config.yaml says today


def test_the_variables_override_each_mode_independently():
    assert load_settings(environ={"SMARTDESK_LLM_MODE": "live"}).modes.model_dump() == {"tickets": "mock", "llm": "live"}
    assert load_settings(environ={"SMARTDESK_TICKETS_MODE": "live"}).modes.model_dump() == {"tickets": "live", "llm": "mock"}
    both = load_settings(environ={"SMARTDESK_LLM_MODE": "live", "SMARTDESK_TICKETS_MODE": "mock"})
    assert (both.modes.llm, both.modes.tickets) == ("live", "mock")


def test_the_value_is_case_insensitive_and_an_empty_one_is_ignored():
    assert load_settings(environ={"SMARTDESK_LLM_MODE": " LIVE "}).modes.llm == "live"
    assert load_settings(environ={"SMARTDESK_LLM_MODE": ""}).modes.llm == "mock"


def test_a_bad_value_fails_loudly_naming_the_variable():
    with pytest.raises(ConfigError, match="SMARTDESK_LLM_MODE.*'maybe'.*mock, live"):
        load_settings(environ={"SMARTDESK_LLM_MODE": "maybe"})


def test_only_the_modes_can_be_overridden(tmp_path):
    base = load_settings(environ={})
    over = load_settings(environ={"SMARTDESK_LLM_MODE": "live", "SMARTDESK_TICKETS_MODE": "live"})
    assert over.model_copy(update={"modes": base.modes}) == base  # nothing else changed


def test_the_process_environment_is_used_by_default(monkeypatch):
    monkeypatch.setenv("SMARTDESK_LLM_MODE", "live")
    assert load_settings().modes.llm == "live"


def test_the_service_and_the_ui_context_honour_the_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SMARTDESK_TICKETS_MODE", "live")
    # live tickets with no token must be refused, never quietly run as mock (the override reached the stack)
    from smartdesk_ai.agent.factory import LiveModeUnavailable

    with pytest.raises(LiveModeUnavailable):
        build_ui_context(environ=None, dotenv_path=None, data_dir=tmp_path)
    monkeypatch.delenv("SMARTDESK_TICKETS_MODE")
    assert build_service(environ={}, dotenv_path=None).settings.modes.tickets == "mock"


def test_the_test_suite_ignores_a_developers_exported_override(settings):
    assert (settings.modes.llm, settings.modes.tickets) == ("mock", "mock")
