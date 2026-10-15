"""T0.5: component health registry with degraded-mode reporting (FR-F7)."""
import logging
from datetime import datetime, timezone

import pytest

from smartdesk_ai.config.env import load_env
from smartdesk_ai.ops.health import (
    SUBSTITUTES,
    Component,
    HealthRegistry,
    Status,
    get_registry,
    reset_registry,
)

NO_FILE = "/nonexistent/.env"


def boom():
    raise RuntimeError("model weights missing")


@pytest.fixture()
def reg():
    return HealthRegistry(clock=lambda: datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc))


def test_all_components_listed_and_unchecked_at_start(reg):
    assert {h.component for h in reg.all()} == set(Component)
    assert all(h.status is Status.UNKNOWN for h in reg.all())
    assert reg.degraded_modes() == []


def test_covers_the_fr_f7_components():
    assert {c.value for c in Component} == {"reranker", "presidio", "redis", "langsmith", "langfuse"}
    assert set(SUBSTITUTES) == set(Component)


def test_failing_loader_records_degraded_mode_and_warning_and_raises_nothing(reg, caplog):
    with caplog.at_level(logging.WARNING, logger="smartdesk_ai.ops.health"):
        result = reg.try_load(Component.RERANKER, boom)
    assert result is None
    health = reg.get(Component.RERANKER)
    assert health.status is Status.DEGRADED
    assert "RuntimeError" in health.reason and "model weights missing" in health.reason
    assert health.mode == SUBSTITUTES[Component.RERANKER]
    assert health.since == datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "reranker" in warnings[0].getMessage()


def test_successful_loader_returns_value_and_marks_ok(reg):
    assert reg.try_load(Component.REDIS, lambda: "client") == "client"
    assert reg.get(Component.REDIS).status is Status.OK
    assert reg.get(Component.REDIS).mode is None


def test_falsy_result_still_counts_as_loaded(reg):
    assert reg.try_load(Component.REDIS, lambda: 0) == 0
    assert reg.get(Component.REDIS).status is Status.OK


def test_registry_lists_all_degraded_modes(reg):
    reg.try_load(Component.RERANKER, boom)
    reg.try_load(Component.PRESIDIO, boom)
    reg.try_load(Component.REDIS, lambda: object())
    modes = reg.degraded_modes()
    assert {h.component for h in modes} == {Component.RERANKER, Component.PRESIDIO}
    assert {h.mode for h in modes} == {
        SUBSTITUTES[Component.RERANKER],
        SUBSTITUTES[Component.PRESIDIO],
    }


def test_repeated_failure_warns_once_and_keeps_original_since(caplog):
    times = iter([datetime(2026, 10, 14, 9, 0), datetime(2026, 10, 14, 10, 0)])
    reg = HealthRegistry(clock=lambda: next(times))
    with caplog.at_level(logging.WARNING, logger="smartdesk_ai.ops.health"):
        reg.try_load(Component.PRESIDIO, boom)
        reg.try_load(Component.PRESIDIO, boom)
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
    assert reg.get(Component.PRESIDIO).since == datetime(2026, 10, 14, 9, 0)


def test_recovery_clears_degraded_mode(reg):
    reg.try_load(Component.REDIS, boom)
    assert reg.degraded_modes()
    reg.try_load(Component.REDIS, lambda: "ok")
    assert reg.get(Component.REDIS).status is Status.OK
    assert reg.degraded_modes() == []


def test_only_exceptions_are_swallowed_not_interrupts(reg):
    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        reg.try_load(Component.REDIS, interrupted)


def test_long_reason_is_truncated(reg):
    def noisy():
        raise RuntimeError("x" * 5000)

    reg.try_load(Component.RERANKER, noisy)
    assert len(reg.get(Component.RERANKER).reason) <= 300


def test_mark_degraded_directly(reg):
    reg.mark_degraded(Component.LANGFUSE, "keys not set")
    assert reg.get(Component.LANGFUSE).status is Status.DEGRADED


def test_missing_env_keys_register_expected_degraded_modes(reg):
    reg.register_env(load_env(environ={}, dotenv_path=NO_FILE))
    degraded = {h.component for h in reg.degraded_modes()}
    assert degraded == {Component.LANGSMITH, Component.LANGFUSE, Component.REDIS}
    assert reg.get(Component.LANGSMITH).mode == "local log only"
    assert reg.get(Component.REDIS).mode == "in-memory cache"


def test_present_env_keys_leave_components_unchecked_not_ok(reg):
    """Configured is not the same as working: the real loader decides OK or degraded."""
    env = load_env(
        environ={
            "LANGSMITH_API_KEY": "a",
            "LANGFUSE_PUBLIC_KEY": "b",
            "LANGFUSE_SECRET_KEY": "c",
            "REDIS_URL": "redis://localhost:6379/0",
        },
        dotenv_path=NO_FILE,
    )
    reg.register_env(env)
    assert reg.degraded_modes() == []
    assert all(h.status is Status.UNKNOWN for h in reg.all())


def test_env_reasons_never_contain_secret_values(reg):
    env = load_env(environ={"LANGFUSE_PUBLIC_KEY": "pk-secret-123"}, dotenv_path=NO_FILE)
    reg.register_env(env)
    assert "pk-secret-123" not in repr(reg.all())


def test_shared_registry_is_a_singleton_and_resettable():
    reset_registry()
    first = get_registry()
    assert get_registry() is first
    first.mark_degraded(Component.REDIS, "down")
    reset_registry()
    assert get_registry().degraded_modes() == []
