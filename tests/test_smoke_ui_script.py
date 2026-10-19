"""scripts/smoke_ui.py needs a browser and a server, so the commit gate does not run it; these checks only keep it
importable, documented and wired into the project's definition of done."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("smoke_ui", ROOT / "scripts" / "smoke_ui.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_imports_without_starting_anything_and_has_help(capsys):
    module = load()
    with pytest.raises(SystemExit) as stop:
        module.main(["--help"])
    assert stop.value.code == 0 and "--headed" in capsys.readouterr().out


def test_the_script_never_points_at_a_live_service():
    text = (ROOT / "scripts" / "smoke_ui.py").read_text()
    assert '"SMARTDESK_LLM_MODE": "mock"' in text and '"SMARTDESK_TICKETS_MODE": "mock"' in text
    for secret in ("OPENAI_API_KEY", "GITHUB_TOKEN"):
        assert f'"{secret}"' in text  # removed from the server's environment, never passed to it


def test_claude_md_makes_the_smoke_test_part_of_done():
    text = (ROOT / "CLAUDE.md").read_text()
    assert "Definition of done: unit tests pass AND `scripts/smoke_ui.py` passes" in text
    assert "Run it before reporting any UI or agent-flow task as done" in text
