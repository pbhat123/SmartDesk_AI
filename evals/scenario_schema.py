"""Schema for scripted conversation scenarios (SPEC section 7, FR-F7, FR-G12)."""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from evals.schemas import Awaiting, Intent, Tool

Step = Literal[
    "kb_answer", "thanks", "goodbye",
    "ticket_offer", "ticket_offer_with_email_request", "email_requested",
    "ticket_summary", "ticket_created", "ticket_cancelled", "ticket_unavailable",
    "tickets_none", "ticket_list", "ticket_status_single",
]
Component = Literal["reranker", "presidio", "redis", "tracing"]


class Setup(BaseModel):
    email_known: str | None = None  # fake address already given in the session
    github: Literal["up", "down"] = "up"
    tickets_for_email: int = Field(default=0, ge=0)  # seeded tickets for the email used (FR-C8)
    degraded: list[Component] = []  # components that fail to load (FR-F7)


class Expect(BaseModel):
    step: Step
    intent: Intent | None = None
    tool: Tool | None = None
    cites: list[str] = []  # KB entry ids the reply must cite (FR-A13)
    reply_contains: list[str] = []
    summary_fields: list[Literal["title", "description", "category", "email", "priority"]] = []
    ticket_created: bool | None = None
    ticket_id_in_reply: bool = False
    ticket_link_in_reply: bool = False
    status_in_reply: bool = False
    team_update_in_reply: bool = False
    awaiting_after: Awaiting | None = None


class Turn(BaseModel):
    user: str = Field(min_length=1)
    expect: Expect


class Scenario(BaseModel):
    id: str
    title: str
    kind: Literal["scenario", "flow"]
    fr: list[str] = Field(min_length=1)
    setup: Setup = Setup()
    turns: list[Turn] = Field(min_length=1)


def load_scenario(path: Path) -> Scenario:
    return Scenario(**yaml.safe_load(path.read_text()))
