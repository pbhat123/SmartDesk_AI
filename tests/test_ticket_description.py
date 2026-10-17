"""T4.13: the ticket description summarises the conversation so far, contains only what the employee
said, and picks up details the employee adds (FR-B6, FR-B13). Mock LLM, no network."""
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from smartdesk_ai.agent.state import AgentState, TicketDraft
from smartdesk_ai.agent.ticket_agent import (
    DraftReply,
    DraftUnavailable,
    add_details,
    draft_from_statements,
    fallback_draft,
    is_grounded,
    make_drafter,
)
from smartdesk_ai.llm.fallback import LLMUnavailableError

STATEMENTS = ["My monitor keeps flickering", "It started after the Windows update on Monday"]


class FakeDrafter:
    def __init__(self, reply=None, raw=None):
        self.reply, self.raw, self.calls = reply, raw, []

    def invoke(self, messages):
        self.calls.append(messages)
        content = self.raw if self.raw is not None else json.dumps(self.reply)
        return SimpleNamespace(content=content)


def reply(**over):
    base = {"title": "Monitor flickering", "description": "The monitor keeps flickering. It started after the "
            "Windows update on Monday.", "category": "IT", "priority": None}
    return {**base, **over}


def draft(llm, statements=STATEMENTS):
    return draft_from_statements(llm, "prompt", statements)


def test_the_description_summarises_the_employees_statements():
    out = draft(FakeDrafter(reply()))
    assert out.title == "Monitor flickering" and "flickering" in out.description and "Monday" in out.description
    assert isinstance(out, TicketDraft)


def test_every_statement_is_sent_to_the_model_and_nothing_else():
    llm = FakeDrafter(reply())
    draft(llm)
    system, human = llm.calls[0]
    for statement in STATEMENTS:
        assert statement in human.content
    assert "email" not in human.content.lower()


@pytest.mark.parametrize(
    "invented",
    [
        "The monitor keeps flickering. The employee already tried rebooting twice.",  # a step they never mentioned
        "The monitor keeps flickering. Error code 0x80070005 is shown.",  # an invented code
        "The monitor keeps flickering since 14 March.",  # an invented date
        "Priya from Finance reports the monitor keeps flickering.",  # an invented name
    ],
)
def test_an_invented_detail_falls_back_to_the_employees_own_words(invented):
    out = draft(FakeDrafter(reply(description=invented)))
    assert out.description == "\n".join(STATEMENTS)


def test_an_invented_title_detail_falls_back_to_a_title_from_their_words():
    out = draft(FakeDrafter(reply(title="Replace broken Dell U2720Q monitor")))
    assert out.title == "My monitor keeps flickering"


def test_a_faithful_paraphrase_is_kept():
    out = draft(FakeDrafter(reply(description="The employee reports that the monitor keeps flickering, "
                                              "which started after the Windows update on Monday.")))
    assert out.description.startswith("The employee reports")


def test_is_grounded_checks_words_numbers_and_urls():
    said = ["My VPN drops every 10 minutes at home"]
    assert is_grounded("The VPN drops every 10 minutes when the employee is at home.", said)
    assert not is_grounded("The VPN drops every 15 minutes.", said)
    assert not is_grounded("See https://vpn.example.com for details", said)
    assert not is_grounded("The router firmware is outdated", said)


def test_a_malformed_reply_falls_back_to_a_draft_from_their_words():
    out = draft(FakeDrafter(raw="here is your ticket!"))
    assert out == fallback_draft(STATEMENTS)
    assert out.category == "MISC" and out.priority is None


def test_the_fallback_title_is_one_short_line():
    long = "word " * 40
    out = fallback_draft([long + "\nsecond line"])
    assert "\n" not in out.title and len(out.title) <= 80


def test_an_email_the_model_adds_is_not_accepted():
    out = draft(FakeDrafter(reply(description="Monitor flickers. Contact priya.k@smartdesk.example.")))
    assert "@" not in out.description


def test_an_outage_raises_draft_unavailable_with_the_graceful_message():
    class Down:
        def invoke(self, messages):
            raise LLMUnavailableError("try again later")

    with pytest.raises(DraftUnavailable) as caught:
        draft(Down())
    assert caught.value.message == "try again later"


# --- added details ------------------------------------------------------------------------------


def test_added_details_appear_in_the_next_summary_and_not_before():
    first = draft(FakeDrafter(reply()))
    assert "second floor" not in first.description
    second = add_details(first, "I sit on the second floor")
    assert "second floor" in second.description and second.description.startswith(first.description)
    assert second.title == first.title


def test_added_details_are_the_employees_words_and_carry_no_email():
    out = add_details(fallback_draft(STATEMENTS), "Reach me at priya.k@smartdesk.example, desk 4B")
    assert "desk 4B" in out.description and "@" not in out.description


def test_blank_added_details_change_nothing():
    base = fallback_draft(STATEMENTS)
    assert add_details(base, "   ") == base


def test_a_redraft_with_the_extra_statement_includes_it():
    llm = FakeDrafter(reply(description="The monitor flickers after the Windows update. I sit on the second floor."))
    out = draft(llm, STATEMENTS + ["I sit on the second floor"])
    assert "second floor" in out.description


# --- the drafter on the TICKET_DRAFT task ---------------------------------------------------------


def test_make_drafter_uses_the_ticket_draft_task_and_the_active_prompt(settings):
    built = []

    def builder(model_id, cfg):
        built.append(model_id)
        return FakeDrafter(reply())

    drafter = make_drafter(settings, builder=builder)
    assert drafter(STATEMENTS).title == "Monitor flickering"
    assert built == [getattr(settings.models, settings.tiers["TICKET_DRAFT"]).id]


def test_the_draft_reply_model_is_strict():
    DraftReply(title="t", description="d", category="IT", priority=None)
    with pytest.raises(ValueError):
        DraftReply(title="t", description="d", category="IT", priority="urgent")
    with pytest.raises(ValueError):
        DraftReply(title="two\nlines", description="d", category="IT", priority=None)
    with pytest.raises(ValueError):
        DraftReply(title="t", description="d", category="IT", priority=None, extra=1)


def test_the_prompt_says_to_use_only_what_the_employee_said():
    from smartdesk_ai.agent.prompt_loader import load_prompt
    from smartdesk_ai.config.settings import load_settings

    text = load_prompt("ticket_summary", load_settings()).text
    assert "only what the employee said" in text and "priority" in text


def test_state_is_untouched_by_drafting():
    state = AgentState(messages=[HumanMessage("x")])
    draft(FakeDrafter(reply()))
    assert state.ticket is None


# --- through the graph ---------------------------------------------------------------------------


def test_the_graph_drafts_from_the_unanswered_question_only(built_index_dir, settings, tmp_path):
    from smartdesk_ai.agent.graph import run_turn
    from smartdesk_ai.agent.state import new_thread_id
    from tests.agent_helpers import build_test_graph
    from tests.llm_fakes import FakeAnswerLLM

    drafter = FakeDrafter(reply(title="Monitor flickering", description="My monitor keeps flickering"))
    graph, _ = build_test_graph(
        built_index_dir, settings, tmp_path, llm=FakeAnswerLLM(["cite_top", "insufficient"]), drafter_llm=drafter
    )
    tid = new_thread_id()
    run_turn(graph, tid, "How do I reset my forgotten password?", email="priya.k@smartdesk.example")
    run_turn(graph, tid, "My monitor keeps flickering")
    out = run_turn(graph, tid, "yes")
    assert out.ticket is not None and out.ticket.title == "Monitor flickering"
    sent = drafter.calls[0][1].content
    assert "monitor keeps flickering" in sent and "password" not in sent
    assert "t_draft" in out.steps


def test_a_drafting_outage_creates_nothing_and_says_so(built_index_dir, settings, tmp_path):
    from smartdesk_ai.agent.graph import run_turn
    from smartdesk_ai.agent.state import new_thread_id
    from tests.agent_helpers import build_test_graph
    from tests.llm_fakes import FakeAnswerLLM

    class Down:
        def invoke(self, messages):
            raise ConnectionError("down")

    graph, _ = build_test_graph(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("insufficient"), drafter_llm=Down())
    tid = new_thread_id()
    run_turn(graph, tid, "My monitor keeps flickering", email="priya.k@smartdesk.example")
    out = run_turn(graph, tid, "yes")
    assert out.ticket is None and "try again" in out.reply.lower() and out.pending_offer is None
