"""Component health registry (FR-F7).

Optional components (reranker, Presidio, Redis, tracing) must never crash
startup. Each one is loaded through `try_load`, which turns any failure into a
recorded degraded mode plus one warning, and returns None so the caller can
switch to the substitute. The Monitoring tab reads `degraded_modes()`.
"""
import logging
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from enum import StrEnum
from typing import TypeVar

from pydantic import BaseModel, ConfigDict

from smartdesk_ai.config.env import EnvSettings

logger = logging.getLogger(__name__)

T = TypeVar("T")
_MAX_REASON = 300


class Component(StrEnum):
    RERANKER = "reranker"
    PRESIDIO = "presidio"
    REDIS = "redis"
    LANGSMITH = "langsmith"
    LANGFUSE = "langfuse"
    GITHUB = "github"  # the ticket tracker (shown as "GitHub tickets")
    MODERATION = "moderation"


class Status(StrEnum):
    UNKNOWN = "unknown"  # not checked yet
    OK = "ok"
    DEGRADED = "degraded"
    SKIPPED = "skipped"  # left out on purpose (mock mode, or a strategy that does not use it); not a fault
    MOCK = "mock"  # an in-memory stand-in is the real thing in this run (mock tickets); not a fault


# What runs instead when a component is unavailable (FR-F7).
SUBSTITUTES: dict[Component, str] = {
    Component.RERANKER: "RRF order (the escalation threshold is unaffected)",
    Component.PRESIDIO: "regex PII checks; name reduction skipped",
    Component.REDIS: "in-memory cache",
    Component.LANGSMITH: "local log only",
    Component.LANGFUSE: "local log only",
    Component.GITHUB: "ticket requests get a try-again-later reply",
    Component.MODERATION: "messages and replies pass unchecked (the outage is logged)",
}


class ComponentHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    component: Component
    status: Status
    mode: str | None = None  # substitute in use, set only while degraded
    reason: str | None = None
    since: datetime | None = None  # when the current status began
    load_seconds: float | None = None  # how long the last load attempt took (try_load)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class HealthRegistry:
    def __init__(self, clock: Callable[[], datetime] = _utcnow) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._items = {c: ComponentHealth(component=c, status=Status.UNKNOWN) for c in Component}

    def get(self, component: Component) -> ComponentHealth:
        with self._lock:
            return self._items[component]

    def all(self) -> list[ComponentHealth]:
        with self._lock:
            return [self._items[c] for c in Component]

    def degraded_modes(self) -> list[ComponentHealth]:
        return [h for h in self.all() if h.status is Status.DEGRADED]

    def mark_skipped(self, component: Component, reason: str) -> None:
        """The component is not used in this run, by design. It is listed, but it is not a degraded mode."""
        with self._lock:
            self._items[component] = ComponentHealth(
                component=component, status=Status.SKIPPED, reason=reason[:_MAX_REASON], since=self._clock())

    def mark_mock(self, component: Component, reason: str) -> None:
        """An in-memory stand-in is used by design (mock mode). Listed as "mock", not a degraded mode."""
        with self._lock:
            self._items[component] = ComponentHealth(
                component=component, status=Status.MOCK, reason=reason[:_MAX_REASON], since=self._clock())

    def mark_ok(self, component: Component, load_seconds: float | None = None) -> None:
        with self._lock:
            current = self._items[component]
            if current.status is Status.OK:
                if load_seconds is not None:
                    self._items[component] = current.model_copy(update={"load_seconds": load_seconds})
                return
            self._items[component] = ComponentHealth(
                component=component, status=Status.OK, since=self._clock(), load_seconds=load_seconds
            )
        if current.status is Status.DEGRADED:
            logger.info("component %s recovered", component.value)

    def mark_degraded(self, component: Component, reason: str, load_seconds: float | None = None) -> None:
        reason = reason[:_MAX_REASON]
        mode = SUBSTITUTES[component]
        with self._lock:
            current = self._items[component]
            if current.status is Status.DEGRADED:
                # Already degraded: keep the original start time, do not warn again.
                update = {"reason": reason} | ({"load_seconds": load_seconds} if load_seconds is not None else {})
                self._items[component] = current.model_copy(update=update)
                return
            self._items[component] = ComponentHealth(
                component=component,
                status=Status.DEGRADED,
                mode=mode,
                reason=reason,
                since=self._clock(),
                load_seconds=load_seconds,
            )
        logger.warning("component %s degraded (%s); using: %s", component.value, reason, mode)

    def try_load(self, component: Component, loader: Callable[[], T]) -> T | None:
        """Run `loader`. On success mark the component OK and return its value.

        On any Exception record the degraded mode, log one warning and return None.
        KeyboardInterrupt and SystemExit still propagate.
        """
        started = time.perf_counter()
        try:
            value = loader()
        except Exception as exc:
            self.mark_degraded(component, f"{type(exc).__name__}: {exc}", time.perf_counter() - started)
            return None
        self.mark_ok(component, time.perf_counter() - started)
        return value

    def register_env(self, env: EnvSettings) -> None:
        """Mark optional components degraded when their keys are absent.

        Present keys only mean "configured": the component stays unchecked until
        its real loader runs.
        """
        if not env.langsmith_enabled:
            self.mark_degraded(Component.LANGSMITH, "LANGSMITH_API_KEY not set")
        if not env.langfuse_enabled:
            self.mark_degraded(Component.LANGFUSE, "LANGFUSE keys not set")
        if env.redis_url is None:
            self.mark_degraded(Component.REDIS, "REDIS_URL not set")


_registry: HealthRegistry | None = None
_registry_lock = threading.Lock()


def get_registry() -> HealthRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = HealthRegistry()
        return _registry


def reset_registry() -> None:
    """Drop the shared registry (tests)."""
    global _registry
    with _registry_lock:
        _registry = None
