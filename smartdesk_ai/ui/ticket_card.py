"""The ticket draft and the created ticket as cards (FR-B8, FR-B4). The agent's reply is plain text; the card is only
a nicer view of the same words. A reply that does not look like a draft or a receipt is shown as ordinary text."""
import re
from dataclasses import dataclass, field

import streamlit as st

from smartdesk_ai.agent.ticket_agent import CONFIRM_QUESTION

DRAFT_INTRO = "Here is the ticket I'm ready to create:"
DRAFT_FIELDS = ("Title", "Category", "Description", "Priority", "Email")  # the order the card shows them in
WAITING_BADGE = "Waiting for your OK"
DRAFT_HINT = "Reply **yes** to create it, **no** to cancel, or tell me what to change."
_FIELD_LINE = re.compile(rf"^({'|'.join(DRAFT_FIELDS)}): ?(.*)$")
_RECEIPT = re.compile(r"^(Your ticket has been created\.|Your ticket was already created:) "
                      r"Ticket #(\d+) \((mock|[^)]+)\): ?(\S*)")  # mock: "(mock): Open"; live: "(Open): <url>"


@dataclass
class DraftView:
    fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ReceiptView:
    number: int
    status: str
    mock: bool
    url: str
    note: str  # what follows the receipt line (the updates note)


def parse_draft(text: str) -> DraftView | None:
    """The fields of an approval-gate summary, or None when the text is not one."""
    lines = text.split("\n")
    if not lines or lines[0] != DRAFT_INTRO or CONFIRM_QUESTION not in text:
        return None
    view, current = DraftView(), None
    for line in lines[1:]:
        if line == CONFIRM_QUESTION:
            break
        found = _FIELD_LINE.match(line)
        if found:
            current = found.group(1)
            view.fields[current] = found.group(2)
        elif current and line.strip():
            view.fields[current] += "\n" + line  # a description that runs over several lines
    return view if "Title" in view.fields else None


def parse_receipt(text: str) -> ReceiptView | None:
    found = _RECEIPT.match(text)
    if not found:
        return None
    mock = found.group(3) == "mock"
    rest = text.partition("\n")[2].strip()
    return ReceiptView(number=int(found.group(2)), status=found.group(4) if mock else found.group(3), mock=mock,
                       url="" if mock else found.group(4), note=rest)


def render_draft(view: DraftView) -> None:
    st.markdown(DRAFT_INTRO)
    with st.container(border=True):
        left, right = st.columns([3, 2], vertical_alignment="center")
        left.markdown("**Ticket draft**")
        right.markdown(f":orange-badge[{WAITING_BADGE}]")
        for name in DRAFT_FIELDS:
            if name in view.fields:
                st.markdown(f":gray[{name}]  \n{view.fields[name]}")
        st.caption(DRAFT_HINT)


def render_receipt(view: ReceiptView) -> None:
    label = f"Ticket #{view.number}{' (mock)' if view.mock else ''} created"
    st.success(f"**{label}** · {view.status}", icon=":material/check_circle:")
    if view.url:
        st.markdown(f"[{view.url}]({view.url})")
    if view.note:
        st.markdown(view.note)
