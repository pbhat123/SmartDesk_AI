"""The chat tab (SPEC section 4 item 3, FR-A12): messages, citations as the agent wrote them, and under each
reply a panel with the retrieved chunks and their scores. Only the entry id, title and score are shown, never
chunk text. History lives in the browser session; the thread ID is the only identifier sent to the graph (FR-G11)."""
import uuid
from typing import Any

import pandas as pd
import streamlit as st

from smartdesk_ai.ui.context import UiContext

DEBUG_LABEL = "Show retrieval debug"
PANEL_LABEL = "Retrieval debug"
GREETING = "Ask an IT or HR question, report a problem, or ask about one of your tickets."


def chunk_table(chunks: list[dict[str, Any]]) -> pd.DataFrame:
    """Retrieved chunks in rank order, as the debug panel shows them."""
    return pd.DataFrame(
        [{"rank": rank, "entry_id": c["entry_id"], "title": c["title"], "score": round(float(c["score"]), 4)}
         for rank, c in enumerate(chunks, 1)],
        columns=["rank", "entry_id", "title", "score"])


def _new_conversation() -> None:
    st.session_state["thread_id"] = uuid.uuid4().hex
    st.session_state["messages"] = []


def _render_panel(message: dict[str, Any]) -> None:
    chunks = message.get("chunks")
    if chunks is None:  # the turn did no retrieval (small talk, a ticket step)
        return
    with st.expander(PANEL_LABEL, expanded=False):
        if chunks:
            st.dataframe(chunk_table(chunks), hide_index=True, width="stretch")
        else:
            st.caption("No chunks were retrieved.")
        if message.get("steps"):
            st.caption("Steps: " + " → ".join(message["steps"]))


def render_chat(ctx: UiContext) -> None:
    if "thread_id" not in st.session_state:
        _new_conversation()
    with st.sidebar:
        show_debug = st.toggle(DEBUG_LABEL, value=True)
        if st.button("New conversation"):
            _new_conversation()
            st.rerun()
        st.caption(f"Mode: LLM {ctx.settings.modes.llm}, tickets {ctx.settings.modes.tickets}")
        for health in ctx.registry.degraded_modes():
            st.caption(f"Degraded: {health.component.value} → {health.mode}")
    messages: list[dict[str, Any]] = st.session_state["messages"]
    if not messages:
        st.caption(GREETING)
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and show_debug:
                _render_panel(message)
    prompt = st.chat_input("Type your message")
    if prompt:
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking"):
                reply = ctx.service.handle(st.session_state["thread_id"], prompt)
            st.markdown(reply.text)
            entry = {"role": "assistant", "content": reply.text, "chunks": reply.debug.get("chunks"),
                     "steps": reply.debug.get("steps")}
            messages.append(entry)
            if show_debug:
                _render_panel(entry)
