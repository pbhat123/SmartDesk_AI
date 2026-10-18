"""T6.5: the Streamlit chat with the retrieval-debug panel (SPEC section 4 item 3, FR-A12, FR-G11). AppTest, mock mode."""
from pathlib import Path

from tests.pii_helpers import has_phone
from tests.ui_helpers import app_test, mock_context

ASK = "How do I reset my forgotten password?"


def _chat_texts(at) -> list[str]:
    return [" ".join(m.markdown[i].value for i in range(len(m.markdown))) for m in at.chat_message]


def _ask(at, text: str):
    at.chat_input[0].set_value(text).run()
    return at


def test_a_question_gets_a_cited_reply(tmp_path):
    at = _ask(app_test(mock_context(tmp_path)), ASK)
    assert not at.exception
    texts = _chat_texts(at)
    assert len(texts) == 2
    assert ASK in texts[0]
    assert "reset a forgotten or expired password" in texts[1]
    assert "Sources: IT-PWD-01" in texts[1]  # the citation line of the reply (FR-A9)


def test_the_debug_panel_lists_chunks_with_scores_in_rank_order(tmp_path):
    at = _ask(app_test(mock_context(tmp_path)), ASK)
    panel = [e for e in at.expander if "Retrieval debug" in e.label]
    assert len(panel) == 1
    rows = panel[0].dataframe[0].value
    assert list(rows.columns) == ["rank", "entry_id", "title", "score"]
    assert rows["rank"].tolist() == list(range(1, len(rows) + 1))
    assert rows["entry_id"].iloc[0] == "IT-PWD-01"
    assert (rows["score"] > 0).all()
    assert "kb_responder" not in str(panel[0].markdown)  # the panel shows retrieval, not internals


def test_the_debug_panel_can_be_switched_off(tmp_path):
    at = app_test(mock_context(tmp_path))
    toggle = next(t for t in at.sidebar.toggle if "debug" in t.label.lower())
    toggle.set_value(False).run()
    _ask(at, ASK)
    assert [e for e in at.expander if "Retrieval debug" in e.label] == []


def test_a_turn_without_retrieval_shows_no_panel_and_no_error(tmp_path):
    at = _ask(app_test(mock_context(tmp_path)), "hello there")
    assert not at.exception
    assert len(_chat_texts(at)) == 2
    assert [e for e in at.expander if "Retrieval debug" in e.label] == []


def test_each_reply_keeps_its_own_panel_and_the_history_survives_reruns(tmp_path):
    at = _ask(app_test(mock_context(tmp_path)), ASK)
    _ask(at, "What is the Wi-Fi network I should use in the office?")
    texts = _chat_texts(at)
    assert len(texts) == 4 and ASK in texts[0]
    assert len([e for e in at.expander if "Retrieval debug" in e.label]) == 2


def test_the_same_browser_session_keeps_one_thread(tmp_path):
    at = app_test(mock_context(tmp_path))
    first = at.session_state["thread_id"]
    _ask(at, ASK)
    assert at.session_state["thread_id"] == first
    at.sidebar.button[0].click().run()  # "New conversation"
    assert at.session_state["thread_id"] != first and _chat_texts(at) == []


def test_a_typed_email_and_phone_never_reach_the_trace_log(tmp_path):
    ctx = mock_context(tmp_path)
    at = app_test(ctx)
    _ask(at, "I am priya.k@smartdesk.example, phone 98765 43210. " + ASK)
    blob = (Path(tmp_path) / "trace.jsonl").read_text()
    assert "priya.k@smartdesk.example" not in blob and not has_phone(blob)
