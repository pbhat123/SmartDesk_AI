"""The chat tab (SPEC section 4 item 3, FR-A12): messages, citations as the agent wrote them, and under each
reply a panel with the retrieved chunks and their scores. Only the entry id, title and score are shown, never
chunk text. History lives in the browser session; the thread ID is the only identifier sent to the graph (FR-G11)."""
import uuid
from typing import Any

import pandas as pd
import streamlit as st

from smartdesk_ai.ui.context import UiContext
from smartdesk_ai.ui.formats import usd
from smartdesk_ai.ui.monitoring import session_cost
from smartdesk_ai.ui.theme import MARK

DEBUG_LABEL = "Show retrieval debug"
PANEL_LABEL = "Retrieval debug"
SOURCES_PREFIX = "Sources: "
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


MOCK_TICKETS_NOTE = "Mock tickets live in memory; a restart clears them and numbering restarts at #5."  # under the mode pills
MOCK_CAPTION = ("Mock mode uses fake embeddings (hashing) and no reranker, so mock retrieval is not meaningful: "
                "the scores and the order only show that the plumbing works.")


def _render_part(part: dict[str, Any]) -> None:
    """One part of a compound question: its sub-question, the worker, its chunks and its escalation line."""
    st.markdown(f"**Part {part['part_id'].lstrip('p')} ({part['agent']}):** {part['sub_question']}")
    if part["chunks"]:
        st.dataframe(chunk_table(part["chunks"]), hide_index=True, width="stretch")
    else:
        st.caption("No chunks were retrieved.")
    st.caption(escalation_line(part["retrieval"]) + (" · retried against the whole KB" if part["retried"] else ""))
    if part.get("cache"):
        st.caption(cache_line(part["cache"]))


def _render_panel(message: dict[str, Any], ctx: UiContext) -> None:
    parts = message.get("parts")
    chunks = message.get("chunks")
    if chunks is None and not parts:  # the turn did no retrieval (small talk, a ticket step)
        return
    with st.expander(PANEL_LABEL, expanded=False):
        for part in parts or []:
            _render_part(part)
        if parts:
            if ctx.settings.modes.llm == "mock":
                st.caption(MOCK_CAPTION)
            if message.get("steps"):
                st.caption("Steps: " + " → ".join(message["steps"]))
            return
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


def _show_cost(slot: Any, ctx: UiContext) -> None:
    cost = session_cost(ctx.service.services.cost, st.session_state.get("thread_ids", []))
    slot.markdown(f"Session cost **{usd(cost.usd)}**")


def _mode_pills(ctx: UiContext) -> None:
    """LLM and ticket mode as two badges: blue for live, gray for mock."""
    def pill(name: str, mode: str) -> str:
        return f":{'blue' if mode == 'live' else 'gray'}-badge[{name} {mode}]"

    modes = ctx.settings.modes
    st.markdown(f"{pill('LLM', modes.llm)} {pill('Tickets', modes.tickets)}")


def render_chat(ctx: UiContext) -> None:
    if "thread_id" not in st.session_state:
        _new_conversation()
    with st.sidebar:
        if st.button("New conversation", icon=":material/add:", width="stretch"):
            _new_conversation()
            st.rerun()
        _mode_pills(ctx)
        if ctx.settings.modes.tickets == "mock":
            st.caption(MOCK_TICKETS_NOTE)
        for health in ctx.registry.degraded_modes():
            st.caption(f"Degraded: {health.component.value} → {health.mode}")
        cost_slot = st.empty()
        _show_cost(cost_slot, ctx)
        show_debug = st.toggle(DEBUG_LABEL, value=True)
    st.title("Help desk")
    st.caption(GREETING)
    messages: list[dict[str, Any]] = st.session_state["messages"]
    for message in messages:
        _render_message(message, ctx, show_debug)
    exchange = st.container()  # the new messages go here, above the input
    prompt = st.chat_input("Type your message")  # a top-level element: pinned to the bottom of the window
    if prompt:
        messages.append({"role": "user", "content": prompt})
        with exchange:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar=str(MARK)):
                with st.spinner("Thinking"):
                    reply = ctx.service.handle(st.session_state["thread_id"], prompt)
                entry = {"role": "assistant", "content": reply.text, "chunks": reply.debug.get("chunks"),
                         "steps": reply.debug.get("steps"), "retrieval": reply.debug.get("retrieval"),
                         "cache": reply.debug.get("cache"), "parts": reply.debug.get("parts")}
                messages.append(entry)
                _render_reply(entry, ctx, show_debug)
        _show_cost(cost_slot, ctx)  # the turn just paid for is counted now, not on the next rerun


def _render_message(message: dict[str, Any], ctx: UiContext, show_debug: bool) -> None:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
        return
    with st.chat_message("assistant", avatar=str(MARK)):
        _render_reply(message, ctx, show_debug)


def split_sources(text: str) -> list[tuple[str, str | list[str]]]:
    """A reply as ("text", markdown) and ("sources", entry ids) segments in order. A "Sources: ID (Title); ..." line
    can close the reply or sit before a later part (a partial answer names the unanswered part after it)."""
    segments: list[tuple[str, str | list[str]]] = []
    buffer: list[str] = []
    for line in text.split("\n"):
        if not line.startswith(SOURCES_PREFIX):
            buffer.append(line)
            continue
        if "\n".join(buffer).strip():
            segments.append(("text", "\n".join(buffer).strip()))
        buffer = []
        ids = [item.split(" (", 1)[0].strip() for item in line[len(SOURCES_PREFIX):].split("; ")]
        segments.append(("sources", [i for i in ids if i]))
    if "\n".join(buffer).strip():
        segments.append(("text", "\n".join(buffer).strip()))
    return segments


def _domain(agent: str) -> str:
    return agent.removesuffix("_worker").upper()


def parts_chip(message: dict[str, Any]) -> str | None:
    """"2 parts · IT + HR · cache miss" for a compound question, else just the cache state when there is one."""
    parts = message.get("parts") or []
    if parts:
        states = [(p.get("cache") or {}).get("status") for p in parts]
        hits = sum(state == "hit" for state in states)
        cache = "hit" if hits == len(parts) else "miss" if not hits else f"{hits} of {len(parts)} hit"
        domains = " + ".join(dict.fromkeys(_domain(p["agent"]) for p in parts))
        return f"{len(parts)} parts · {domains} · cache {cache}"
    if message.get("cache"):
        return f"cache {'hit' if message['cache'].get('status') == 'hit' else 'miss'}"
    return None


def _render_reply(message: dict[str, Any], ctx: UiContext, show_debug: bool) -> None:
    chip = parts_chip(message)
    segments = split_sources(message["content"])
    for kind, value in segments:
        if kind == "text":
            st.markdown(value)
        else:
            st.markdown(":gray[**Sources**] " + " ".join(f":blue-badge[{source}]" for source in value))
    if chip:
        st.markdown(f":gray-badge[{chip}]")
    if show_debug:
        _render_panel(message, ctx)
