"""A ticket client that reports whether GitHub can be reached, for the Monitoring tab (FR-F7, FR-G10).

It wraps a client and passes every call through. GitHub counts as unreachable (degraded) after a transient failure
(network, timeout, 429, 5xx) or a refused token (401, 403); any other answer, even "not found", shows it is reachable.
The reason is the error's own message, which holds a status code and never response text or the token (FR-G11).
"""
from collections.abc import Callable
from typing import Any

from smartdesk_ai.ops.health import Component, HealthRegistry
from smartdesk_ai.tickets.schemas import TicketError, TransientTicketError

_REFUSED = (401, 403)


class HealthTrackedClient:
    def __init__(self, inner: Any, registry: HealthRegistry) -> None:
        self._inner = inner
        self._registry = registry

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._inner, name)
        return self._tracked(attribute) if callable(attribute) else attribute

    def __repr__(self) -> str:
        return f"HealthTrackedClient({self._inner!r})"

    def _tracked(self, call: Callable[..., Any]) -> Callable[..., Any]:
        def run(*args: Any, **kwargs: Any) -> Any:
            try:
                result = call(*args, **kwargs)
            except TransientTicketError as error:
                self._registry.mark_degraded(Component.GITHUB, str(error))
                raise
            except TicketError as error:
                if error.status in _REFUSED:
                    self._registry.mark_degraded(Component.GITHUB, str(error))
                else:
                    self._registry.mark_ok(Component.GITHUB)
                raise
            self._registry.mark_ok(Component.GITHUB)
            return result

        return run
