"""Pydantic row schemas for the eval data sets (SPEC section 7)."""
from typing import Literal

from pydantic import BaseModel, Field

Split = Literal["dev", "test"]
Domain = Literal["IT", "HR"]
Category = Literal["IT", "HR", "MISC"]
QuestionType = Literal["simple", "how-to", "policy interpretation"]


class KBQuestion(BaseModel):
    id: str
    question: str = Field(min_length=10)
    expected_entry_id: str
    expected_domain: Domain
    type: QuestionType
    reference_answer: str = Field(min_length=20)
    split: Split
