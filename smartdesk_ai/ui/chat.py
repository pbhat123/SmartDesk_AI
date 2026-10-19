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
    """Retrieved chunks in rank order with each stage's score. An empty cell means the stage did not run for it
    (no reranker, or the piece was outside the dense list)."""
    return pd.DataFrame(
        [{"rank": rank, "entry_id": c["entry_id"], "title": c["title"],
          "dense cosine": _score(c.get("dense")), "RRF score": _score(c.get("rrf")),
          "rerank score": _score(c.get("rerank"))}
         for rank, c in enumerate(chunks, 1)],
        columns=["rank", "entry_id", "title", "dense cosine", "RRF score", "rerank score"])


def _score(value: float | None) -> float:
    return float("nan") if value is None else round(float(value), 4)


def escalation_line(retrieval: dict[str, Any]) -> str:
    cosine, threshold = retrieval.get("top_cosine"), retrieval.get("threshold")
    parts = []
    if threshold is not None:
        parts.append(f"Escalation threshold: {threshold:.2f}")
    if cosine is not None:
        parts.append(f"top dense cosine: {cosine:.3f}")
    parts.append(f"Escalated this turn: {'yes' if retrieval.get('escalated') else 'no'}")
    return " · ".join(parts)


def cache_line(cache: dict[str, Any]) -> str:
    """Whether the answer cache served this turn's KB question, and what a hit saved."""
    if cache.get("status") != "hit":
        return "Cache: miss"
    kind = cache.get("kind") or "exact"
    if cache.get("similarity") is not None:
        kind += f", similarity {cache['similarity']:.3f}"
    saved = cache.get("saved_ms")
    return f"Cache: hit ({kind})" + (f" · saved about {saved:.0f} ms" if saved is not None else "")


def _new_conversation() -> None:
    thread_id = uuid.uuid4().hex
    st.session_state["thread_id"] = thread_id
    st.session_state["thread_ids"] = [*st.session_state.get("thread_ids", []), thread_id]  # for the session's cost
    st.session_state["messages"] = []


MOCK_TICKETS_CAPTION = "Mock tickets live in memory and reset on restart (numbering starts again at #5)."
MOCK_CAPTION = ("Mock mode uses fake embeddings (hashing) and no reranker, so mock retrieval is not meaningful: "
                "the scores and the order only show that the plumbing works.")


def _render_panel(message: dict[str, Any], ctx: UiContext) -> None:
    chunks = message.get("chunks")
    if chunks is None:  # the turn did no retrieval (small talk, a ticket step)
        return
    with st.expander(PANEL_LABEL, expanded=False):
        if chunks:
            st.dataframe(chunk_table(chunks), hide_index=True, width="stretch")
        else:
            st.caption("No chunks were retrieved.")
        if message.get("retrieval"):
            st.caption(escalation_line(message["retrieval"]))
        if ctx.settings.modes.llm == "mock":
            st.caption(MOCK_CAPTION)
        if message.get("cache"):
            st.caption(cache_line(message["cache"]))
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
                _render_panel(message, ctx)
    exchange = st.container()  # the new messages go here, above the input
    if ctx.settings.modes.tickets == "mock":
        st.caption(MOCK_TICKETS_CAPTION)
    prompt = st.chat_input("Type your message")  # a top-level element: pinned to the bottom of the window
    if prompt:
        messages.append({"role": "user", "content": prompt})
        with exchange:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking"):
                    reply = ctx.service.handle(st.session_state["thread_id"], prompt)
                st.markdown(reply.text)
                entry = {"role": "assistant", "content": reply.text, "chunks": reply.debug.get("chunks"),
                         "steps": reply.debug.get("steps"), "retrieval": reply.debug.get("retrieval"),
                         "cache": reply.debug.get("cache")}
                messages.append(entry)
                if show_debug:
                    _render_panel(entry, ctx)
