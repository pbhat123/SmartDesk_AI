"""Per-task reasoning effort (FR-H16, FR-H17): config.yaml can set it for one task on the Main tier; every other
task keeps the tier default."""
import pytest

from smartdesk_ai.agent.grounding import make_grounding_judge
from smartdesk_ai.config.settings import Settings
from smartdesk_ai.llm.factory import get_llm, get_llm_for_tier


def seen_efforts(settings, task, **kwargs):
    seen = []

    def builder(model_id, cfg):
        seen.append((model_id, getattr(cfg, "reasoning_effort", None)))
        return object()

    get_llm(task, settings, builder=builder, **kwargs)
    return seen


def test_the_judge_has_its_own_effort_in_config(settings):
    assert settings.reasoning_effort == {"GROUNDING_JUDGE": "low"}
    assert settings.models.main.reasoning_effort == "none"  # the tier default is unchanged


def test_the_judge_builds_with_the_override_and_other_main_tasks_keep_the_default(settings):
    assert seen_efforts(settings, "GROUNDING_JUDGE") == [(settings.models.main.id, "low")]
    for task in ("WORKER_ANSWER", "ORCHESTRATOR", "SYNTHESIZER"):
        assert seen_efforts(settings, task) == [(settings.models.main.id, "none")], task


def test_light_tasks_have_no_reasoning_effort(settings):
    assert seen_efforts(settings, "CONFIRM_PARSE") == [(settings.models.light.id, None)]


def test_the_override_does_not_follow_the_judge_onto_the_light_tier(settings):
    seen = []
    get_llm_for_tier("GROUNDING_JUDGE", "light", settings, builder=lambda i, c: seen.append(getattr(c, "reasoning_effort", None)))
    assert seen == [None]


def test_the_judge_factory_uses_the_override(settings):
    seen = []
    judge = make_grounding_judge(settings, builder=lambda i, c: seen.append(c.reasoning_effort) or object())
    assert callable(judge)  # the model is built lazily; invoking would need a reply
    from smartdesk_ai.llm.fallback import FallbackChat

    chat = FallbackChat("GROUNDING_JUDGE", settings, builder=lambda i, c: seen.append(c.reasoning_effort) or object())
    try:
        chat.invoke([])
    except Exception:
        pass
    assert seen and set(seen) == {"low"}


def test_an_unknown_task_or_effort_is_rejected(settings):
    data = settings.model_dump()
    with pytest.raises(ValueError, match="reasoning_effort"):
        Settings(**{**data, "reasoning_effort": {"NOT_A_TASK": "low"}})
    with pytest.raises(ValueError):
        Settings(**{**data, "reasoning_effort": {"GROUNDING_JUDGE": "turbo"}})
