"""Smoke test for the repo scaffold (T0.1): Python version and package layout."""
import importlib
import sys

import pytest

PACKAGES = [
    "smartdesk_ai",
    "smartdesk_ai.agent",
    "smartdesk_ai.rag",
    "smartdesk_ai.tickets",
    "smartdesk_ai.llm",
    "smartdesk_ai.tools",
    "smartdesk_ai.guardrails",
    "smartdesk_ai.ops",
    "smartdesk_ai.memory",
    "smartdesk_ai.ui",
]


def test_python_version_is_3_12():
    assert sys.version_info[:2] == (3, 12)


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name):
    assert importlib.import_module(name)
