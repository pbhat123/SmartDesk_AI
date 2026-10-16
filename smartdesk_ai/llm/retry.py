"""Retry with exponential backoff for LLM calls (FR-F2)."""
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int,
    backoff: float,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn` up to `attempts` times, waiting `backoff`, 2*backoff, ... between tries.
    The last error is re-raised once attempts run out."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:
            if attempt == attempts - 1:
                raise
            sleep(backoff * 2**attempt)
    raise AssertionError("unreachable: attempts must be >= 1")
