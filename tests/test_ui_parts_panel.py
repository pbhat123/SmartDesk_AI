"""T7.4/T7.5: a compound answer's retrieval-debug panel lists each part with its sub-question, the worker that
answered it, its chunks and whether it escalated (FR-A12, FR-I9). A single question's panel is unchanged."""
from tests.ui_helpers import app_test, element_text, mock_context

COMPOUND = "How do I reset my password, and how many casual leave days do I get?"
SINGLE = "How do I reset my forgotten password?"


def _panel(at):
    (panel,) = [e for e in at.expander if "Retrieval debug" in e.label]
    return panel


def _ask(ctx, text):
    at = app_test(ctx)
    at.chat_input[0].set_value(text).run()
    assert not at.exception
    return at


def test_a_compound_reply_has_a_panel_with_one_table_per_part(tmp_path):
    panel = _panel(_ask(mock_context(tmp_path), COMPOUND))
    shown = element_text(panel)
    assert "**Part 1 (it_worker):** How do I reset my password?" in shown
    assert "**Part 2 (hr_worker):** How many casual leave days do I get?" in shown
    assert len(panel.dataframe) == 2
    assert "fan_out:2" in shown and "synthesize" in shown


def test_a_compound_reply_is_one_merged_answer_with_sources_from_both_domains(tmp_path):
    at = _ask(mock_context(tmp_path), COMPOUND)
    reply = " ".join(m.value for m in at.chat_message[-1].markdown)  # the answer, then the source chips
    assert "Sources" in reply and "IT-" in reply and "HR-" in reply


def test_a_single_question_panel_has_no_part_headings(tmp_path):
    panel = _panel(_ask(mock_context(tmp_path), SINGLE))
    assert "Part 1" not in element_text(panel) and len(panel.dataframe) == 1
