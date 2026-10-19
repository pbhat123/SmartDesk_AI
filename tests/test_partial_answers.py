"""T7.7: a compound question with an unanswered part (FR-I7, FR-B6, FR-B13, FR-A16): the answered parts come first with
their sources, the part the KB cannot answer is named in fixed wording, and the ticket offered covers that part only.
Scripted and mock models, no network."""
import json
import re
from types import SimpleNamespace

import pytest

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.agent.ticket_agent import (
    PARTIAL_OFFER_WITH_EMAIL_REQUEST,
    PARTIAL_OFFER_YES_NO,
    partial_reply,
)
from smartdesk_ai.llm.fallback import LLMUnavailableError
from smartdesk_ai.agent.confirm import make_confirm_parser
from smartdesk_ai.llm.mock import MockConfirmModel, MockDraftModel
from tests.agent_helpers import assemble
from tests.test_fanout import HR_BODY, IT_BODY, Synth
from tests.test_misroute_retry import mock_settings

Q_IT = "How do I reset my password?"
Q_HR = "How many casual leave days do I get?"
Q_GAP = "Where can I park my car at the Bengaluru office?"
Q_GAP2 = "Can I book a desk for Tuesday?"
EMAIL = "priya.k@smartdesk.example"


class Answers:
    """Answers the password and casual leave questions; has nothing for anything else."""

    def invoke(self, messages):
        human = messages[-1].content
        question = human.split("\n\nContext:")[0]
        ids = re.findall(r"^\[([A-Z]+-[A-Z0-9-]+)\]", human, re.M)
        if "password" in question:
            body = {"answer": IT_BODY, "insufficient": False, "citations": ids[:1]}
        elif "casual leave" in question:
            body = {"answer": HR_BODY, "insufficient": False, "citations": ids[:1]}
        else:
            body = {"answer": "", "insufficient": True, "citations": []}
        return SimpleNamespace(content=json.dumps(body))


def scripted(*parts):
    reply = {"intent": "kb_query", "confidence": 0.95, "domain": parts[0][1],
             "parts": [{"text": t, "domain": d, "follow_up": False} for t, d in parts]}
    return SimpleNamespace(invoke=lambda messages: SimpleNamespace(content=json.dumps(reply)))


def stack_for(built_index_dir, settings, tmp_path, *parts, synth=None, **kw):
    return assemble(built_index_dir, mock_settings(settings), tmp_path, llm=Answers(), routed=True,
                    route_model=scripted(*parts), synth_model=synth if synth is not None else Synth(),
                    drafter_llm=MockDraftModel(),
                    parse_confirm=make_confirm_parser(settings, builder=lambda m, c: MockConfirmModel()), **kw)


def ask(stack, text="a compound question", tid=None, **kw):
    return run_turn(stack.graph, tid or new_thread_id(), text, **kw)


IT, HR, GAP, GAP2 = (Q_IT, "IT"), (Q_HR, "HR"), (Q_GAP, "MISC"), (Q_GAP2, "MISC")


# --- the reply ------------------------------------------------------------------------------------------------------

def test_one_answered_part_and_one_gap_answers_the_first_names_the_gap_and_offers_a_ticket(built_index_dir, settings, tmp_path):
    synth = Synth()
    state = ask(stack_for(built_index_dir, settings, tmp_path, HR, GAP, synth=synth))
    answered, sources, gap = state.reply.split("\n\n")
    assert answered == HR_BODY and sources.startswith("Sources: HR-")
    assert gap == partial_reply([Q_GAP], email_known=False)
    assert f'"{Q_GAP}"' in gap and gap.endswith(PARTIAL_OFFER_WITH_EMAIL_REQUEST)
    assert synth.calls == [], "one answered part: nothing to merge, so no synthesizer call"
    assert state.awaiting == "email"


def test_the_part_that_cannot_be_answered_is_named_and_the_answered_one_is_not(built_index_dir, settings, tmp_path):
    state = ask(stack_for(built_index_dir, settings, tmp_path, IT, GAP))
    assert IT_BODY in state.reply and Q_GAP in state.reply and Q_IT not in state.reply
    assert "I don't know the answer to that" not in state.reply, "the generic gap wording is for a whole-question gap"


def test_with_a_known_email_the_offer_is_a_plain_yes_or_no(built_index_dir, settings, tmp_path):
    state = ask(stack_for(built_index_dir, settings, tmp_path, HR, GAP), email=EMAIL)
    assert state.reply.endswith(PARTIAL_OFFER_YES_NO) and state.awaiting == "offer_reply"


def test_two_answered_parts_are_merged_once_and_the_gap_is_added_after(built_index_dir, settings, tmp_path):
    synth = Synth()
    state = ask(stack_for(built_index_dir, settings, tmp_path, IT, HR, GAP, synth=synth))
    assert len(synth.calls) == 1
    seen = synth.calls[0][1].content
    assert Q_GAP not in seen and "Question 3" not in seen, "the synthesizer is given the answered parts only"
    merged, sources, gap = state.reply.split("\n\n")[:3]
    assert merged == f"{IT_BODY} {HR_BODY}" and sources.startswith("Sources: ") and "IT-" in sources and "HR-" in sources
    assert f'"{Q_GAP}"' in gap
    assert state.steps[-2:] == ["synthesize", "offer"]


def test_a_failed_merge_still_gives_the_answered_parts_then_the_gap(built_index_dir, settings, tmp_path):
    state = ask(stack_for(built_index_dir, settings, tmp_path, IT, HR, GAP, synth=Synth(error=LLMUnavailableError())))
    assert state.reply.startswith(f"{IT_BODY}\n\n{HR_BODY}") and f'"{Q_GAP}"' in state.reply
    assert "synthesize_fallback" in state.steps


def test_two_gaps_are_listed_and_the_offer_is_for_both(built_index_dir, settings, tmp_path):
    state = ask(stack_for(built_index_dir, settings, tmp_path, HR, GAP, GAP2))
    assert f'- "{Q_GAP}"' in state.reply and f'- "{Q_GAP2}"' in state.reply and "these parts" in state.reply
    assert state.pending_offer.unanswered_parts == [Q_GAP, Q_GAP2]


def test_no_part_answered_names_every_part_without_a_sources_line(built_index_dir, settings, tmp_path):
    state = ask(stack_for(built_index_dir, settings, tmp_path, GAP, GAP2))
    assert "Sources:" not in state.reply and Q_GAP in state.reply and Q_GAP2 in state.reply
    assert state.pending_offer.unanswered_parts == [Q_GAP, Q_GAP2] and state.part_result is None


def test_a_single_part_gap_keeps_the_original_wording(built_index_dir, settings, tmp_path):
    from smartdesk_ai.agent.ticket_agent import escalation_offer

    state = ask(stack_for(built_index_dir, settings, tmp_path, GAP))
    assert state.reply == escalation_offer(email_known=False) and state.pending_offer.unanswered_parts == []


def test_the_answered_parts_are_cached_the_merged_reply_is_not(built_index_dir, settings, tmp_path):
    from smartdesk_ai.ops.cache import AnswerCache, MemoryBackend

    stack = stack_for(built_index_dir, settings, tmp_path, HR, GAP, cache=AnswerCache(MemoryBackend(10), kb_version=lambda: "v"))
    ask(stack)
    again = ask(stack)
    assert again.steps.count("cache_hit:exact") == 1 and f'"{Q_GAP}"' in again.reply


# --- the ticket covers only the unanswered part -----------------------------------------------------------------------

def draft_after_offer(stack):
    """Ask the compound question, then give the email: the draft is shown at the gate."""
    tid = new_thread_id()
    ask(stack, tid=tid)
    return tid, run_turn(stack.graph, tid, EMAIL)


def test_the_draft_after_a_partial_reply_covers_only_the_unanswered_part(built_index_dir, settings, tmp_path):
    stack = stack_for(built_index_dir, settings, tmp_path, HR, GAP)
    tid, state = draft_after_offer(stack)
    ticket = state.ticket
    assert ticket is not None and Q_GAP.rstrip("?") in ticket.title
    assert Q_GAP in ticket.description and "casual leave" not in ticket.description.lower() and "casual leave" not in ticket.title.lower()
    assert ticket.category == "MISC"
    assert "Shall I go ahead" in state.reply and "casual leave" not in state.reply.lower()


def test_what_the_employee_adds_after_the_offer_joins_the_unanswered_part(built_index_dir, settings, tmp_path):
    stack = stack_for(built_index_dir, settings, tmp_path, HR, GAP)
    tid, _ = draft_after_offer(stack)
    edited = run_turn(stack.graph, tid, "also: it is for the Bengaluru office annex")
    assert Q_GAP in edited.ticket.description and "annex" in edited.ticket.description
    assert "casual leave" not in edited.ticket.description.lower()


def test_yes_creates_one_ticket_for_the_gap_and_it_names_no_answered_part(built_index_dir, settings, tmp_path):
    stack = stack_for(built_index_dir, settings, tmp_path, IT, HR, GAP)
    tid, _ = draft_after_offer(stack)
    before = len(stack.client._issues)
    done = run_turn(stack.graph, tid, "yes")
    assert len(stack.client._issues) == before + 1 and "has been created" in done.reply
    issue = list(stack.client._issues.values())[-1]
    assert Q_GAP in issue.body and "casual leave" not in issue.body.lower() and "password" not in issue.body.lower()


def test_declining_the_offer_creates_nothing_and_clears_it(built_index_dir, settings, tmp_path):
    stack = stack_for(built_index_dir, settings, tmp_path, HR, GAP)
    tid = new_thread_id()
    ask(stack, tid=tid)
    before = len(stack.client._issues)
    state = run_turn(stack.graph, tid, "no thanks")
    assert len(stack.client._issues) == before and state.pending_offer is None and state.awaiting == "none"


def test_the_draft_invents_nothing_the_employee_did_not_say(built_index_dir, settings, tmp_path):
    """FR-B13: the sub-question is the orchestrator's wording of what the employee asked; a draft that adds a number or
    a name that is in no message of the conversation falls back to the sub-question itself."""
    class Inventing:
        def invoke(self, messages):
            return SimpleNamespace(content=json.dumps({"title": "Parking at Gate 7", "category": "MISC", "priority": None,
                                                       "description": "Parking at Gate 7 for Rahul"}))

    stack = assemble(built_index_dir, mock_settings(settings), tmp_path, llm=Answers(), routed=True,
                     route_model=scripted(HR, GAP), synth_model=Synth(), drafter_llm=Inventing())
    tid = new_thread_id()
    ask(stack, "How many casual leave days do I get, and where can I park my car at the Bengaluru office?", tid=tid)
    state = run_turn(stack.graph, tid, EMAIL)
    assert "Gate 7" not in state.ticket.title and "Rahul" not in state.ticket.description
    assert Q_GAP in state.ticket.description


def test_a_name_from_the_employees_own_message_is_allowed_in_the_draft(built_index_dir, settings, tmp_path):
    class Naming:
        def invoke(self, messages):
            return SimpleNamespace(content=json.dumps({"title": "Parking at the Bengaluru office", "category": "MISC",
                                                       "priority": None, "description": "Where can I park my car at the Bengaluru office?"}))

    stack = assemble(built_index_dir, mock_settings(settings), tmp_path, llm=Answers(), routed=True,
                     route_model=scripted(HR, GAP), synth_model=Synth(), drafter_llm=Naming())
    tid = new_thread_id()
    ask(stack, "How many casual leave days do I get, and where can I park my car at the Bengaluru office?", tid=tid)
    state = run_turn(stack.graph, tid, EMAIL)
    assert state.ticket.title == "Parking at the Bengaluru office"


def test_the_drafter_sees_only_the_unanswered_part_and_what_came_after(built_index_dir, settings, tmp_path):
    class Spy:
        def __init__(self):
            self.inputs = []

        def invoke(self, messages):
            self.inputs.append(messages[-1].content)
            return MockDraftModel().invoke(messages)

    spy = Spy()
    stack = assemble(built_index_dir, mock_settings(settings), tmp_path, llm=Answers(), routed=True,
                     route_model=scripted(IT, GAP), synth_model=Synth(), drafter_llm=spy)
    tid = new_thread_id()
    ask(stack, "How do I reset my password, and where can I park my car at the Bengaluru office?", tid=tid)
    run_turn(stack.graph, tid, EMAIL)
    assert spy.inputs == [f"Employee statements:\n- {Q_GAP}"]


def test_the_wording_of_the_offers_names_the_part(built_index_dir, settings, tmp_path):
    assert "that part" in PARTIAL_OFFER_YES_NO and "that part" in PARTIAL_OFFER_WITH_EMAIL_REQUEST
    one = partial_reply([Q_GAP], email_known=True)
    assert one.startswith("I'm sorry") and one.count(Q_GAP) == 1
    two = partial_reply([Q_GAP, Q_GAP2], email_known=False)
    assert "those parts" in two and two.count("\n- ") == 2
