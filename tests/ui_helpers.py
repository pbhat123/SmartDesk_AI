"""Builds the UI's context in mock mode inside a temp folder, and an AppTest around the app script."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.ui.context import UiContext, build_ui_context

APP = Path(__file__).resolve().parents[1] / "smartdesk_ai" / "ui" / "app.py"


def mock_context(tmp_path: Path) -> UiContext:
    """Mock models, mock tickets, no .env, every runtime file under tmp_path (never the repo's data/)."""
    return build_ui_context(environ={}, dotenv_path=None, data_dir=tmp_path, registry=HealthRegistry())


def app_test(ctx: UiContext, **session: object) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["ctx"] = ctx
    for key, value in session.items():
        at.session_state[key] = value
    return at.run()
