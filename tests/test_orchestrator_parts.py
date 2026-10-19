"""T7.3: the orchestrator splits a message into standalone, domain-tagged sub-questions and sees the whole
(masked, compacted) conversation to do it (FR-I3, FR-I4, FR-D6). Scripted and mock models, no network."""
import json
from types import SimpleNamespace

import pytest

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.intents import ClassifierError, classify_intent
from smartdesk_ai.agent.orchestrator import plan_parts, transcript_for
from smartdesk_ai.agent.state import IntentResult, SubQuestion, new_thread_id
from langchain_core.messages import AIMessage, HumanMessage
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM

PROMPT = "classify"


class Scripted:
    """An orchestrator model that records its input and returns the next scripted reply."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def invoke(self, messages):
        self.calls.append(messages)
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return SimpleNamespace(content=json.dumps(reply) if isinstance(reply, dict) else reply)


def kb(parts, domain="HR", confidence=0.95):
    return {"intent": "kb_query", "confidence": confidence, "domain": domain, "parts": parts}


def part(text, domain="HR", follow_up=False):
    return {"text": text, "domain": domain, "follow_up": follow_up}


# --- the reply shape ---------------------------------------------------------------------------------

def test_a_reply_with_parts_gives_domain_tagged_sub_questions_with_ids():
    model = Scripted(kb([part("How do I reset my password?", "IT"), part("How many casual leave days do I get?")]))
    result = classify_intent(model, PROMPT, "two questions")
    assert [(p.part_id, p.text, p.domain) for p in result.parts] == [
        ("p1", "How do I reset my password?", "IT"), ("p2", "How many casual leave days do I get?", "HR")]


def test_a_reply_without_parts_still_parses_as_before():
    result = classify_intent(Scripted({"intent": "kb_query", "confidence": 0.9, "domain": "IT"}), PROMPT, "q")
    assert result.parts == [] and result.domain == "IT"


@pytest.mark.parametrize("bad", [{"text": "x", "domain": "FINANCE"}, {"domain": "IT"}, {"text": "x", "domain": "IT", "extra": 1}])
def test_an_invalid_part_is_an_invalid_classification(bad):
    with pytest.raises(ClassifierError):
        classify_intent(Scripted(kb([bad])), PROMPT, "q")


def test_parts_are_ignored_for_any_intent_but_kb_query():
    reply = {"intent": "smalltalk", "confidence": 0.9, "domain": None, "parts": [part("x question here", "IT")]}
    assert classify_intent(Scripted(reply), PROMPT, "hi").parts == []


# --- planning: the guards that keep a plain question exactly as asked -----------------------------------

RESULT = lambda *parts, domain=None: IntentResult(intent="kb_query", confidence=0.9, domain=domain, parts=list(parts))  # noqa: E731


def sub(text, domain="HR", pid="p1", rewritten=False):
    return SubQuestion(part_id=pid, text=text, domain=domain, rewritten=rewritten)


def test_a_single_standalone_question_is_searched_with_the_employees_own_words():
    plan = plan_parts("How many sick days do I get?", None, RESULT(sub("How many sick leave days per year?")), max_parts=4)
    assert [(p.text, p.rewritten) for p in plan.parts] == [("How many sick days do I get?", False)]


def test_a_single_part_the_model_marks_as_a_follow_up_uses_the_model_rewrite():
    given = sub("How many casual leave days do contractors get?", rewritten=True)
    plan = plan_parts("what about contractors?", "How many casual leave days do I get?", RESULT(given), max_parts=4)
    assert [(p.text, p.rewritten, p.domain) for p in plan.parts] == [(given.text, True, "HR")]


def test_a_follow_up_phrase_without_a_model_rewrite_falls_back_to_the_topic_rule():
    plan = plan_parts("what about sick leave?", "casual leave", RESULT(sub("whatever")), max_parts=4)
    assert plan.parts[0].text == "what about sick leave? (following up on: casual leave)"


def test_no_parts_from_the_model_gives_one_part_from_the_classifier_domain():
    plan = plan_parts("How do I reset my password?", None, RESULT(domain="IT"), max_parts=4)
    assert [(p.part_id, p.text, p.domain) for p in plan.parts] == [("p1", "How do I reset my password?", "IT")]
    assert plan_parts("x?", None, RESULT(domain=None), max_parts=4).parts[0].domain == "MISC"


def test_several_parts_keep_the_models_texts_in_order_with_ids():
    plan = plan_parts("a and b", None, RESULT(sub("How do I reset my password?", "IT", "p1"),
                                              sub("How long is probation?", "HR", "p2")), max_parts=4)
    assert [(p.part_id, p.text, p.domain, p.rewritten) for p in plan.parts] == [
        ("p1", "How do I reset my password?", "IT", True), ("p2", "How long is probation?", "HR", True)]


def test_emails_are_removed_from_sub_questions_and_empty_parts_dropped():
    plan = plan_parts("m", None, RESULT(sub("Send jane.roe@example.com the VPN steps?", "IT", "p1"),
                                        sub("   ", "HR", "p2"), sub("How long is probation?", "HR", "p3")), max_parts=4)
    assert [p.text for p in plan.parts] == ["Send the VPN steps?", "How long is probation?"]
    assert [p.part_id for p in plan.parts] == ["p1", "p2"], "ids are renumbered after a drop"


def test_parts_beyond_the_limit_are_counted_not_answered():
    many = [sub(f"Question number {i} about leave?", "HR", f"p{i}") for i in range(1, 7)]
    plan = plan_parts("m", None, RESULT(*many), max_parts=4)
    assert len(plan.parts) == 4 and plan.dropped == 2


# --- the model sees the whole conversation, masked ------------------------------------------------------

def test_the_transcript_is_every_earlier_turn_with_emails_masked():
    messages = [HumanMessage("my email is jane.roe@example.com, how many casual leave days do I get?"),
                AIMessage("You get 12 days.\n\nSources: HR-LEAVE-01"), HumanMessage("what about contractors?")]
    text = transcript_for(messages)
    assert text == ("Employee: my email is [email], how many casual leave days do I get?\n"
                    "Assistant: You get 12 days.\n\nSources: HR-LEAVE-01")
    assert "jane.roe" not in text and transcript_for(messages[-1:]) == ""


def test_a_ticket_status_reply_is_not_shown_to_the_orchestrator():
    """It quotes the support team's comment, which is untrusted text no model may see (FR-C9)."""
    status = AIMessage("Ticket #4 ...\n> SECRET-TEAM-TEXT", name="ticket_status")
    text = transcript_for([HumanMessage("any update?"), status, HumanMessage("thanks")])
    assert "SECRET" not in text and text == "Employee: any update?\nAssistant: (ticket details shown to the employee)"


def routed(built_index_dir, settings, tmp_path, model, **kw):
    return assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM(), routed=True, route_model=model, **kw)


def test_a_follow_up_is_rewritten_into_a_standalone_sub_question_using_earlier_turns(built_index_dir, settings, tmp_path):
    model = Scripted(
        kb([part("How many casual leave days do I get?")]),
        kb([part("How many casual leave days do contractors get?", follow_up=True)]))
    stack = routed(built_index_dir, settings, tmp_path, model)
    tid = new_thread_id()
    run_turn(stack.graph, tid, "How many casual leave days do I get?", email="jane.roe@example.com")
    state = run_turn(stack.graph, tid, "what about contractors?")
    seen = str(model.calls[1][-1].content)
    assert "How many casual leave days do I get?" in seen and "Employee:" in seen and "Assistant:" in seen
    assert "Message: what about contractors?" in seen and "jane.roe" not in seen
    assert [(p.text, p.domain, p.rewritten) for p in state.parts] == [
        ("How many casual leave days do contractors get?", "HR", True)]
    assert stack.spy.calls[-1][0] == "How many casual leave days do contractors get?"
    assert state.last_topic == "How many casual leave days do contractors get?"


def test_the_first_turn_has_no_conversation_block(built_index_dir, settings, tmp_path):
    model = Scripted(kb([part("How many casual leave days do I get?")]))
    stack = routed(built_index_dir, settings, tmp_path, model)
    run_turn(stack.graph, new_thread_id(), "How many casual leave days do I get?")
    assert "Conversation so far" not in str(model.calls[0][-1].content)


def test_the_mock_orchestrator_splits_a_compound_question_into_domain_tagged_parts(built_index_dir, settings, tmp_path):
    stack = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM(), routed=True)
    state = run_turn(stack.graph, new_thread_id(), "How do I reset my password, and how many casual leave days do I get?")
    assert [(p.text, p.domain) for p in state.parts] == [
        ("How do I reset my password?", "IT"), ("How many casual leave days do I get?", "HR")]


def test_a_compaction_summary_is_passed_on_as_it_is():
    from langchain_core.messages import SystemMessage

    messages = [SystemMessage("Summary of the earlier conversation: the employee asked about leave."),
                HumanMessage("what about contractors?")]
    assert transcript_for(messages) == "Summary of the earlier conversation: the employee asked about leave."


# --- clauses about one topic stay one part (orchestrator_v5; the mock follows the same rule) --------------------

TWO_CLAUSES_ONE_TOPIC = [
    "What are the possible performance scores, and when are promotions decided?",
    "How many sick days do I get, and do I need a certificate?",
    "Is there a spot for my motorbike at the Pune site, and does it cost anything?",
]


@pytest.mark.parametrize("question", TWO_CLAUSES_ONE_TOPIC)
def test_a_two_clause_question_about_one_topic_stays_one_part(built_index_dir, settings, tmp_path, question):
    stack = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM(), routed=True)
    state = run_turn(stack.graph, new_thread_id(), question)
    assert [(p.text, p.rewritten) for p in state.parts] == [(question, False)]
    assert not any(s.startswith("fan_out") for s in state.steps)


def test_two_questions_that_need_different_domains_are_still_split(built_index_dir, settings, tmp_path):
    stack = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM(), routed=True)
    state = run_turn(stack.graph, new_thread_id(), "How do I set up MFA, and how many sick days do I get?")
    assert [p.domain for p in state.parts] == ["IT", "HR"]


# --- a fragment is merged back into the part before it (orchestrator_v8's code guard) ------------------------------

from smartdesk_ai.agent.orchestrator import MIN_PART_WORDS, merge_fragments, plan_parts  # noqa: E402
from smartdesk_ai.agent.state import IntentResult, SubQuestion  # noqa: E402

KB_023 = "Who is the person assigned to help new starters settle in, and for how long?"
KB_045 = "What are the possible performance scores, and when are promotions decided?"
KB_053 = "How many people work at SmartDesk AI and in which cities?"


def planned(raw, *parts, last_topic=None):
    """`plan_parts` for a scripted classification: each part is (text, domain) or (text, domain, follow_up)."""
    subs = [SubQuestion(part_id=f"p{i}", text=t, domain=d, rewritten=bool(rest and rest[0])) for i, (t, d, *rest) in enumerate(parts, 1)]
    result = IntentResult(intent="kb_query", confidence=0.95, domain=parts[0][1], parts=subs)
    return plan_parts(raw, last_topic, result, max_parts=4)


@pytest.mark.parametrize("raw, first, fragment, domain", [
    (KB_023, "Who is the person assigned to help new starters settle in?", "for how long?", "HR"),
    (KB_045, "What are the possible performance scores?", "when are promotions decided?", "HR"),
    (KB_053, "How many people work at SmartDesk AI?", "in which cities?", "MISC"),
])
def test_a_fragment_is_merged_back_and_the_employees_own_words_are_searched(raw, first, fragment, domain):
    plan = planned(raw, (first, domain), (fragment, domain))
    assert [(p.text, p.rewritten) for p in plan.parts] == [(raw, False)]  # one part: today's single-question path, golden parity


def test_two_real_questions_in_the_same_domain_stay_split_cmp_013():
    plan = planned("What is the probation period, and does SmartDesk AI offer a sabbatical?",
                   ("What is the probation period?", "HR"), ("Does SmartDesk AI offer a sabbatical?", "HR"))
    assert [p.text for p in plan.parts] == ["What is the probation period?", "Does SmartDesk AI offer a sabbatical?"]


def test_a_short_part_for_another_domain_is_a_real_part_and_is_not_merged_cmp_011():
    plan = planned("q", ("How do I reset my password?", "IT"), ("How long is probation?", "HR"), ("Can I book a desk for Tuesday?", "MISC"))
    assert [(p.text, p.domain) for p in plan.parts] == [
        ("How do I reset my password?", "IT"), ("How long is probation?", "HR"), ("Can I book a desk for Tuesday?", "MISC")]


def test_a_fragment_inside_a_longer_split_joins_only_its_own_neighbour():
    plan = planned("q", ("How do I set up MFA on a new phone?", "IT"), ("Who is my buddy?", "HR"), ("and for how long?", "HR"))
    assert [(p.text, p.domain) for p in plan.parts] == [
        ("How do I set up MFA on a new phone?", "IT"), ("Who is my buddy, and for how long?", "HR")]


def test_a_first_part_is_never_merged_into_anything_and_the_guard_never_splits():
    parts = merge_fragments([SubQuestion(part_id="p1", text="Hi?", domain="HR"), SubQuestion(part_id="p2", text="ok?", domain="HR")])
    assert len(parts) == 1 and parts[0].text == "Hi, ok?"  # the short first part keeps its place; the second joins it
    many = [SubQuestion(part_id=f"p{i}", text=f"A complete question number {i}?", domain="IT") for i in range(3)]
    assert merge_fragments(many) == many and MIN_PART_WORDS == 5


def test_a_follow_up_keeps_its_rewritten_text_when_a_fragment_is_merged():
    plan = planned("and for contractors?", ("How many casual leave days do contractors get?", "HR", True), ("per year?", "HR"))
    assert [(p.text, p.rewritten) for p in plan.parts] == [("How many casual leave days do contractors get, per year?", True)]


def test_through_the_graph_a_fragmented_question_is_one_part_with_the_raw_text(built_index_dir, settings, tmp_path):
    model = Scripted(kb([part("Who is the person assigned to help new starters settle in?"), part("for how long?")]))
    state = run_turn(routed(built_index_dir, settings, tmp_path, model).graph, new_thread_id(), KB_023)
    assert [(p.text, p.rewritten) for p in state.parts] == [(KB_023, False)] and not any(s.startswith("fan_out") for s in state.steps)


def test_through_the_graph_two_real_questions_still_fan_out(built_index_dir, settings, tmp_path):
    model = Scripted(kb([part("What is the probation period?"), part("Does SmartDesk AI offer a sabbatical?")]))
    state = run_turn(routed(built_index_dir, settings, tmp_path, model).graph, new_thread_id(),
                     "What is the probation period, and does SmartDesk AI offer a sabbatical?")
    assert len(state.parts) == 2 and any(s.startswith("fan_out") for s in state.steps)
