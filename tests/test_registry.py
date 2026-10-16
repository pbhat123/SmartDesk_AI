"""T3.4: tool registry, name to function (FR-G3)."""
import pytest

from smartdesk_ai.tools.definitions import TOOLS, openai_tool_schemas
from smartdesk_ai.tools.registry import ToolRegistry, UnknownToolError


def test_lookup_returns_the_registered_function():
    registry = ToolRegistry()

    def fn(args):
        return "ok"

    registry.register("search_kb", fn)
    assert registry.get("search_kb").fn is fn


def test_unknown_name_errors():
    with pytest.raises(UnknownToolError, match="nope"):
        ToolRegistry().get("nope")


def test_duplicate_registration_errors():
    registry = ToolRegistry()
    registry.register("search_kb", lambda a: 1)
    with pytest.raises(ValueError, match="already"):
        registry.register("search_kb", lambda a: 2)


def test_only_defined_tools_can_be_registered():
    with pytest.raises(UnknownToolError):
        ToolRegistry().register("drop_tables", lambda a: 1)


def test_registry_is_not_a_tool_the_llm_sees():
    visible = {s["function"]["name"] for s in openai_tool_schemas()}
    assert visible == {t.name for t in TOOLS}
    assert not {"registry", "tool_registry", "execute", "executor"} & visible
