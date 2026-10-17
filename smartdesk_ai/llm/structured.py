"""Validate a model's JSON reply against a Pydantic model (FR-H2)."""
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


class ReplyError(ValueError):
    """The reply is not valid JSON for the expected model."""


def parse_model_reply(model: type[T], content: Any) -> T:
    """`content` is a reply string (a code fence is tolerated) or an instance of `model`
    already produced by the client's structured-output mode."""
    if isinstance(content, model):
        return content
    raw = (content if isinstance(content, str) else str(content)).strip()
    fenced = _FENCE.match(raw)
    if fenced:
        raw = fenced.group(1)
    try:
        return model.model_validate_json(raw)
    except ValidationError as error:
        raise ReplyError(f"reply is not a valid {model.__name__}") from error
