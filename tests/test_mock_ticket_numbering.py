"""Mock tickets live in memory and reset on restart (numbering starts again at #5, after the four seeded demo tickets):
the chat says so, and no mock ticket or map row ever reaches data/app.db. Mock tickets only (FR-B10, C10)."""
import sqlite3

from tests.ui_helpers import app_test, element_text, mock_context

EMAIL = "priya.k@smartdesk.example"
ASK = "My laptop screen is cracked, please raise a ticket"
CAPTION = "Mock tickets live in memory and reset on restart (numbering starts again at #5)."


def new_ticket(ctx, thread):
    for text in (ASK, EMAIL, "yes"):
        reply = ctx.service.handle(thread, text)
    return reply.text


# --- the caption ---------------------------------------------------------------------------------------------------

def test_the_chat_says_mock_tickets_reset_on_restart(tmp_path):
    page = app_test(mock_context(tmp_path))
    assert not page.exception and CAPTION in element_text(page.main)


# --- numbering and the live map -----------------------------------------------------------------------------------

def test_a_fresh_stack_numbers_its_first_ticket_5_and_a_restart_starts_again(tmp_path):
    assert "Ticket #5 (mock)" in new_ticket(mock_context(tmp_path), "a")
    restarted = mock_context(tmp_path)  # same data folder, new process: the first ticket is gone, the seeds are back
    assert "Ticket #5 (mock)" in new_ticket(restarted, "b")


def test_numbers_keep_counting_up_inside_one_stack(tmp_path):
    ctx = mock_context(tmp_path)
    assert "Ticket #5 (mock)" in new_ticket(ctx, "a")
    ctx.service.handle("b", "The printer on floor 3 jams constantly, please raise a ticket")
    ctx.service.handle("b", EMAIL)
    assert "Ticket #6 (mock)" in ctx.service.handle("b", "yes").text


def test_no_mock_ticket_or_map_row_reaches_app_db(tmp_path):
    ctx = mock_context(tmp_path)
    new_ticket(ctx, "a")
    ctx.service.handle("a", "any update on my ticket?")
    new_ticket(mock_context(tmp_path), "b")
    db = tmp_path / "app.db"
    if db.exists():  # the live map: it may exist, but must hold nothing from mock mode (seed #1-#4 or a new one)
        con = sqlite3.connect(db)
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in tables & {"email_tickets", "create_requests"}:
            assert con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table
