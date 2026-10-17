"""Ticket status flow (flow C, FR-C1 to FR-C7).

T4.22 routes the status trigger phrases here; the lookup, the list and the single-ticket display
follow in T4.23 to T4.26.
"""
from typing import Any

from langchain_core.messages import AIMessage

from smartdesk_ai.agent.state import AgentState

# Placeholder until the lookup arrives (T4.23).
STATUS_PLACEHOLDER = "Ticket status checks are not available yet."


def status_stub_node(state: AgentState) -> dict[str, Any]:
    return {"steps": state.steps + ["status"], "reply": STATUS_PLACEHOLDER,
            "messages": [AIMessage(STATUS_PLACEHOLDER)]}
