"""The three tools the model can see, with typed arguments (FR-G1).

`email` is optional in the schema on purpose: a missing or invalid email is a guard error the
executor reports with an ask-for-email message (FR-G5), not a schema error.
`request_id` is set by the executor (FR-G6) and is hidden from the model.
"""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Domain = Literal["IT", "HR"]
Category = Literal["IT", "HR", "MISC"]
Priority = Literal["low", "medium", "high", "critical"]

_INTERNAL = {"internal": True}


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchKBArgs(_Args):
    query: str = Field(min_length=1, description="The question to look up in the knowledge base.")
    domain: Domain | None = Field(default=None, description="IT or HR to search one area; omit for the whole KB.")


class CreateTicketArgs(_Args):
    title: str = Field(min_length=1, description="One-line summary of the issue.")
    description: str = Field(min_length=1, description="What the employee reported; nothing invented.")
    category: Category = Field(description="IT, HR or MISC.")
    priority: Priority | None = Field(default=None, description="Only if the employee gave one.")
    email: str | None = Field(default=None, description="The employee's email; ask if unknown.")
    request_id: str | None = Field(default=None, json_schema_extra=_INTERNAL)


class GetTicketsArgs(_Args):
    email: str | None = Field(default=None, description="The employee's email; ask if unknown.")


class ToolDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    args_model: type[BaseModel]

    def parameters_schema(self) -> dict[str, Any]:
        """JSON schema for the model, without executor-internal fields."""
        schema = self.args_model.model_json_schema()
        schema["properties"] = {
            k: v for k, v in schema["properties"].items() if not v.get("internal")
        }
        return schema


TOOLS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        name="search_kb",
        description="Search the IT and HR knowledge base. Returns matching entries with scores.",
        args_model=SearchKBArgs,
    ),
    ToolDefinition(
        name="create_ticket",
        description="Create a help desk ticket after the employee confirmed the summary.",
        args_model=CreateTicketArgs,
    ),
    ToolDefinition(
        name="get_tickets",
        description="List the employee's tickets with live status and the latest team update.",
        args_model=GetTicketsArgs,
    ),
)


def get_definition(name: str) -> ToolDefinition:
    for tool in TOOLS:
        if tool.name == name:
            return tool
    raise KeyError(name)


def openai_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters_schema()},
        }
        for t in TOOLS
    ]
