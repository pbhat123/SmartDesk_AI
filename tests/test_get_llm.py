"""T3.1: task-to-tier map and get_llm(task) factory (FR-H17, FR-H16). Mock builder, no network."""
import pytest

from smartdesk_ai.config.settings import REQUIRED_TASKS
from smartdesk_ai.llm.factory import get_llm
from smartdesk_ai.llm.tiers import resolve_tier


class RecordingBuilder:
    """Stands in for the real chat-model constructor; records which model ID was asked for."""

    def __init__(self):
        self.built = []

    def __call__(self, model_id, model_cfg):
        self.built.append(model_id)
        return object()


@pytest.mark.parametrize("task", REQUIRED_TASKS)
def test_each_task_gets_its_configured_tiers_model(settings, task):
    builder = RecordingBuilder()
    handle = get_llm(task, settings, builder=builder)
    tier = settings.tiers[task]
    assert handle.tier == tier
    assert handle.model_id == getattr(settings.models, tier).id
    assert builder.built == [handle.model_id]


def test_confirm_parse_and_chitchat_are_on_light(settings):
    for task in ("CONFIRM_PARSE", "CHITCHAT"):
        assert get_llm(task, settings, builder=RecordingBuilder()).tier == "light"


def test_changing_one_task_changes_only_that_task(settings):
    changed = settings.model_copy(update={"tiers": {**settings.tiers, "CHITCHAT": "main"}})
    for task in REQUIRED_TASKS:
        expected = "main" if task == "CHITCHAT" else settings.tiers[task]
        assert resolve_tier(task, changed) == expected


def test_unknown_task_is_an_error(settings):
    with pytest.raises(ValueError, match="NOPE"):
        get_llm("NOPE", settings, builder=RecordingBuilder())


def test_budget_level_one_moves_every_task_to_light(settings):
    for task in REQUIRED_TASKS:
        assert resolve_tier(task, settings, budget_level=1) == "light"
        assert resolve_tier(task, settings, budget_level=2) == "light"
    assert resolve_tier("ORCHESTRATOR", settings, budget_level=0) == "main"


def test_handle_has_no_hardcoded_ids(settings):
    swapped = settings.model_copy(
        update={"models": settings.models.model_copy(
            update={"light": settings.models.light.model_copy(update={"id": "other-light"})})}
    )
    assert get_llm("CHITCHAT", swapped, builder=RecordingBuilder()).model_id == "other-light"


def test_openai_builder_passes_reasoning_effort_for_main_only(settings, monkeypatch):
    import langchain_openai

    from smartdesk_ai.config.env import EnvSettings
    from smartdesk_ai.llm import factory

    seen = []
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **kw: seen.append(kw) or object())
    monkeypatch.setattr(factory, "load_env", lambda: EnvSettings(openai_api_key="sk-test"))
    factory.get_llm_for_tier("ORCHESTRATOR", "main", settings)
    factory.get_llm_for_tier("CHITCHAT", "light", settings)
    main_kw, light_kw = seen
    assert main_kw["model"] == settings.models.main.id
    assert main_kw["reasoning_effort"] == settings.models.main.reasoning_effort
    assert "reasoning_effort" not in light_kw
