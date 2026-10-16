"""Tool failure types. Only transient failures are retried (FR-G4)."""


class TransientToolError(Exception):
    """A network error, timeout, rate limit or 5xx: the same call may succeed if retried."""


TRANSIENT = (TransientToolError, ConnectionError, TimeoutError)
