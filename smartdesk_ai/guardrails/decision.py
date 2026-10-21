"""What an input guard returns to `ChatService`: the text to pass on, or a refusal."""
from pydantic import BaseModel, ConfigDict


class GuardDecision(BaseModel):
    """Result of the input guard: the (possibly redacted) text, or a refusal."""

    model_config = ConfigDict(frozen=True)

    text: str
    blocked: bool = False
    refusal: str | None = None
    reason: str | None = None
    calm: bool = False  # allowed through although Moderation flagged plain harassment: the reply must stay calm (FR-H1)
