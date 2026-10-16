"""T3.3: tool definitions with typed args (FR-G1)."""
import pytest
from pydantic import BaseModel, ValidationError

from smartdesk_ai.tools.definitions import (
    TOOLS,
    CreateTicketArgs,
    GetTicketsArgs,
    SearchKBArgs,
    get_definition,
    openai_tool_schemas,
)


def test_exactly_the_three_tools_exist():
    assert [t.name for t in TOOLS] == ["search_kb", "create_ticket", "get_tickets"]


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t.name)
def test_each_tool_has_name_description_and_pydantic_args(tool):
    assert tool.name and tool.description.strip()
    assert issubclass(tool.args_model, BaseModel)


@pytest.mark.parametrize("domain", ["IT", "HR", None])
def test_search_kb_accepts_it_hr_or_none(domain):
    assert SearchKBArgs(query="vpn", domain=domain).domain == domain


def test_search_kb_domain_defaults_to_whole_kb():
    assert SearchKBArgs(query="vpn").domain is None


@pytest.mark.parametrize("bad", ["MISC", "it", "Finance", ""])
def test_search_kb_rejects_other_domains(bad):
    with pytest.raises(ValidationError):
        SearchKBArgs(query="vpn", domain=bad)


def test_search_kb_needs_a_query():
    with pytest.raises(ValidationError):
        SearchKBArgs(query="")


def test_create_ticket_category_and_priority_are_closed_sets():
    ok = CreateTicketArgs(title="VPN down", description="Cannot connect", category="IT", priority="high")
    assert ok.priority == "high"
    with pytest.raises(ValidationError):
        CreateTicketArgs(title="t", description="d", category="FINANCE")
    with pytest.raises(ValidationError):
        CreateTicketArgs(title="t", description="d", category="IT", priority="urgent")


def test_create_ticket_priority_is_optional():
    assert CreateTicketArgs(title="t", description="d", category="HR").priority is None


def test_get_tickets_takes_an_email():
    assert GetTicketsArgs(email="a@b.com").email == "a@b.com"


def test_llm_schemas_hide_internal_fields_and_list_only_the_three_tools():
    schemas = openai_tool_schemas()
    assert [s["function"]["name"] for s in schemas] == ["search_kb", "create_ticket", "get_tickets"]
    create = next(s for s in schemas if s["function"]["name"] == "create_ticket")
    assert "request_id" not in create["function"]["parameters"]["properties"]


def test_get_definition_unknown_name_errors():
    with pytest.raises(KeyError):
        get_definition("drop_tables")
