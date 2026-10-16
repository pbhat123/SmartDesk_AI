"""Retry with exponential backoff for LLM calls (FR-F2). Only transient errors are retried."""
import time
from collections.abc import Callable
from typing import TypeVar

from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

T = TypeVar("T")

# APITimeoutError is a subclass of APIConnectionError; both listed for clarity.
TRANSIENT_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


def is_transient(error: Exception) -> bool:
    """Rate limit, timeout, connection and 5xx errors may succeed on a retry. Everything else
    (auth, permission, bad request, unknown) is treated as permanent."""
    return isinstance(error, TRANSIENT_ERRORS)


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int,
    backoff: float,
    sleep: Callable[[float], None] = time.sleep,
    retryable: Callable[[Exception], bool] = is_transient,
) -> T:
    """Call `fn` up to `attempts` times, waiting `backoff`, 2*backoff, ... between tries.
    A permanent error is raised at once; so is the last error when attempts run out."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as error:
            if attempt == attempts - 1 or not retryable(error):
                raise
            sleep(backoff * 2**attempt)
    raise AssertionError("unreachable: attempts must be >= 1")
