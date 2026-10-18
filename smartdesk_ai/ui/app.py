"""SmartDesk AI in the browser: `streamlit run smartdesk_ai/ui/app.py`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # `streamlit run` puts this folder on the path, not the repo root
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from smartdesk_ai.agent.factory import LiveModeUnavailable  # noqa: E402
from smartdesk_ai.ui.chat import render_chat  # noqa: E402
from smartdesk_ai.ui.context import UiContext, build_ui_context  # noqa: E402
from smartdesk_ai.ui.monitoring import render_monitoring  # noqa: E402


@st.cache_resource(show_spinner="Starting SmartDesk AI")
def _default_context() -> UiContext:
    return build_ui_context()


def get_context() -> UiContext:
    """The context a test put in the session, else the one shared by every browser session of this process."""
    ctx = st.session_state.get("ctx")
    return ctx if ctx is not None else _default_context()


def main() -> None:
    st.set_page_config(page_title="SmartDesk AI", page_icon=":speech_balloon:")
    st.title("SmartDesk AI")
    try:
        ctx = get_context()
    except LiveModeUnavailable as error:  # never quietly run as mock when live was asked for
        st.error(f"Cannot start in live mode: {error}")
        st.stop()
    chat_tab, monitoring_tab = st.tabs(["Chat", "Monitoring"])
    with chat_tab:
        render_chat(ctx)
    with monitoring_tab:
        render_monitoring(ctx)


main()
