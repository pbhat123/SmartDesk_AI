"""The app context for the current Streamlit session."""
import streamlit as st

from smartdesk_ai.ui.context import UiContext, build_ui_context


@st.cache_resource(show_spinner="Starting SmartDesk AI")
def _default_context() -> UiContext:
    """Built once per process (the models, the reranker and the index are loaded here, not on every rerun)."""
    return build_ui_context()


def get_context() -> UiContext:
    """The context a test put in the session, else the one shared by every browser session of this process."""
    ctx = st.session_state.get("ctx")
    return ctx if ctx is not None else _default_context()
