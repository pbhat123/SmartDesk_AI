"""The app's look beyond what `.streamlit/config.toml` can set: the logo, and a little scoped CSS for the card style
of docs/ui_mockups/ (white card, 1px border, 12px radius). The colours and the font stack live in the config; no web
font or other network resource is loaded, so the page looks the same offline."""
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
LOGO = ASSETS / "sd_logo.svg"  # the mark plus the name and tagline, for the top of the sidebar
MARK = ASSETS / "sd_mark.svg"  # the mark alone: page icon, collapsed sidebar, assistant avatar
PAGE_TITLE = "SmartDesk AI"

CARD_CSS = """
<style>
[data-testid="stFooter"], footer, #MainMenu, [data-testid="stDecoration"] { display: none !important; }
[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; }
/* chat: the employee's messages are accent bubbles on the right, the assistant's are white cards with the SD avatar */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  flex-direction: row-reverse; margin-left: auto; width: fit-content; max-width: 70%;
  background: #2563EB; color: #FFFFFF; border-radius: 14px 14px 4px 14px; padding: 0.6rem 1rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] { display: none; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) a { color: #FFFFFF; }
[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) [data-testid="stChatMessageContent"] {
  background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 4px 14px 14px 14px; padding: 0.9rem 1.1rem;
}
[data-testid="stChatMessage"]:not(:has([data-testid="stChatMessageAvatarUser"])) { background: transparent; }
[data-testid="stMetricLabel"] { color: #64748B; }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] a { border-radius: 8px; }
</style>
"""


def apply_chrome() -> None:
    """Page title and icon, the logo and the card CSS. Called once per run, before anything else is drawn."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=str(MARK), layout="wide")
    st.logo(str(LOGO), icon_image=str(MARK), size="large")
    st.markdown(CARD_CSS, unsafe_allow_html=True)
