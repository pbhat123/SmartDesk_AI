"""SmartDesk AI in the browser: `streamlit run smartdesk_ai/ui/app.py`.

Two pages, Chat and Monitoring, with `st.navigation`: the chat input is then a top-level element, which Streamlit
pins to the bottom of the window (inside `st.tabs` it would sit at the end of the tab).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # `streamlit run` puts this folder on the path, not the repo root
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from smartdesk_ai.agent.factory import LiveModeUnavailable  # noqa: E402
from smartdesk_ai.ui.session import get_context  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="SmartDesk AI", page_icon=":speech_balloon:")
    try:
        get_context()
    except LiveModeUnavailable as error:  # never quietly run as mock when live was asked for
        st.title("SmartDesk AI")
        st.error(f"Cannot start in live mode: {error}")
        st.stop()
    st.title("SmartDesk AI")
    page = st.navigation([
        st.Page("views/chat_page.py", title="Chat", icon=":material/chat:", default=True),
        st.Page("views/monitoring_page.py", title="Monitoring", icon=":material/monitoring:"),
    ])
    page.run()


main()
