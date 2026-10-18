"""T4.3: intent classifier with confidence and the `awaiting` dialog state (FR-D1, FR-D2).

The mock LLM answers from the eval set (`evals/intents.jsonl`), keyed on the utterance AND
the dialog state it finds in the prompt, so a classifier that forgets to pass `awaiting`
fails here.
"""
import json
import re
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from evals.common import EVALS_DIR, load_jsonl
from evals.schemas import IntentRow
from smartdesk_ai.agent.intents import (
    ClassifierError,
    ClassifierReply,
    classify_intent,
    dialog_update,
    make_classifier,
    route_for,
)
from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.agent.state import AgentState, IntentResult, PendingOffer, TicketDraft

ROWS = [IntentRow(**r) for r in load_jsonl(EVALS_DIR / "intents.jsonl")]

EXTRA = {  # cases the eval set does not hold: bare confirmations
    ("yes", "offer_reply"): {"intent": "create_ticket", "confidence": 0.95, "domain": None},
    ("yes", "none"): {"intent": "ambiguous", "confidence": 0.9, "domain": None},
}


def _domain(route: str) -> str | None:
    return {"it_worker": "IT", "hr_worker": "HR", "misc": "MISC"}.get(route)


TABLE = {
    (r.utterance, r.awaiting): {
        "intent": r.expected_intent,
        "confidence": 0.9,
        "domain": _domain(r.expected_route),
    }
    for r in ROWS
} | EXTRA


class FakeLLM:
    """Reads the message and dialog state out of the prompt, as a real model would."""

    def __init__(self, table=TABLE, raw=None):
        self.table, self.raw, self.calls = table, raw, []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.raw is not None:
            return SimpleNamespace(content=self.raw)
        text = messages[-1].content
        message = re.search(r"^Message: (.*)$", text, re.M).group(1)
        awaiting = re.search(r"^Awaiting: (\w+)$", text, re.M).group(1)
        return SimpleNamespace(content=json.dumps(self.table[(message, awaiting)]))


@pytest.fixture(scope="module")
def prompt(settings):
    return load_prompt("orchestrator", settings).text


@pytest.mark.parametrize("row", ROWS, ids=[r.id for r in ROWS])
def test_route_matches_the_eval_set(row, prompt):
    result = classify_intent(FakeLLM(), prompt, row.utterance, row.awaiting)
    assert result.intent == row.expected_intent
    assert route_for(result) == row.expected_route


def test_yes_routes_by_the_awaiting_state(prompt):
    llm = FakeLLM()
    assert classify_intent(llm, prompt, "yes", "offer_reply").intent == "create_ticket"
    assert classify_intent(llm, prompt, "yes", "none").intent == "ambiguous"


def test_awaiting_and_last_topic_are_sent_to_the_model(prompt):
    llm = FakeLLM()
    classify_intent(llm, prompt, "yes", "offer_reply", last_topic="vpn")
    system, human = llm.calls[0]
    assert "kb_query" in system.content and "ambiguous" in system.content
    assert "Awaiting: offer_reply" in human.content and "Last topic: vpn" in human.content


def test_result_carries_confidence_and_classifier_source(prompt):
    result = classify_intent(FakeLLM(), prompt, "Hi there", "none")
    assert result.confidence == 0.9 and result.route_source == "classifier"


def test_the_prompt_names_all_six_intents(prompt):
    for name in ("kb_query", "create_ticket", "ticket_status", "smalltalk", "out_of_scope", "ambiguous"):
        assert name in prompt


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        '{"intent": "chat", "confidence": 0.9}',
        '{"intent": "kb_query"}',
        '{"intent": "kb_query", "confidence": 1.7}',
        '{"intent": "kb_query", "confidence": "high"}',
    ],
)
def test_invalid_output_raises_classifier_error(prompt, raw):
    with pytest.raises(ClassifierError):
        classify_intent(FakeLLM(raw=raw), prompt, "hello", "none")


def test_extra_or_misspelled_fields_are_rejected_by_the_reply_model(prompt):
    raw = '{"intent": "kb_query", "confidence": 0.9, "domian": "IT"}'
    with pytest.raises(ClassifierError):
        classify_intent(FakeLLM(raw=raw), prompt, "hello", "none")


def test_reply_model_validates_the_json(prompt):
    reply = ClassifierReply.model_validate_json('{"intent": "kb_query", "confidence": 0.8, "domain": "HR"}')
    assert (reply.intent, reply.confidence, reply.domain) == ("kb_query", 0.8, "HR")
    with pytest.raises(ValidationError):
        ClassifierReply.model_validate_json('{"intent": "kb_query", "confidence": 2}')


class StructuredLLM:
    """Has `with_structured_output`, like a live chat model. The schema is bound on each call, with the raw message
    kept (`include_raw`), because the call goes through the shared fallback wrapper."""

    def __init__(self, reply):
        self.reply, self.schema, self.include_raw = reply, None, None

    def with_structured_output(self, schema, include_raw=False):
        self.schema, self.include_raw = schema, include_raw
        return SimpleNamespace(invoke=lambda messages: {"raw": SimpleNamespace(content=""), "parsed": self.reply})


def test_live_client_uses_structured_output_mode(settings):
    llm = StructuredLLM(ClassifierReply(intent="smalltalk", confidence=0.9))
    classify = make_classifier(settings, builder=lambda model_id, cfg: llm, structured=True)
    assert classify("hello").intent == "smalltalk"
    assert llm.schema is ClassifierReply and llm.include_raw is True


def test_structured_output_failure_is_a_classifier_error(settings):
    def boom(messages):
        raise ValueError("schema violation")

    llm = StructuredLLM(None)
    llm.with_structured_output = lambda schema, include_raw=False: SimpleNamespace(invoke=boom)
    classify = make_classifier(settings, builder=lambda model_id, cfg: llm, structured=True)
    with pytest.raises(ClassifierError):
        classify("hello")


def test_mock_builder_path_stays_plain_json(settings):
    llm = FakeLLM()  # no with_structured_output
    assert make_classifier(settings, builder=lambda model_id, cfg: llm)("Hi there").intent == "smalltalk"


def test_json_in_a_code_fence_is_accepted(prompt):
    raw = '```json\n{"intent": "smalltalk", "confidence": 0.8, "domain": null}\n```'
    assert classify_intent(FakeLLM(raw=raw), prompt, "hello", "none").intent == "smalltalk"


def test_a_failing_llm_call_becomes_classifier_error(prompt):
    class Boom:
        def invoke(self, messages):
            raise TimeoutError("down")

    with pytest.raises(ClassifierError):
        classify_intent(Boom(), prompt, "hello", "none")


def test_domain_is_kept_only_for_kb_query(prompt):
    raw = '{"intent": "smalltalk", "confidence": 0.9, "domain": "IT"}'
    assert classify_intent(FakeLLM(raw=raw), prompt, "hi", "none").domain is None


def test_kb_query_without_a_domain_goes_to_the_whole_kb(prompt):
    raw = '{"intent": "kb_query", "confidence": 0.9, "domain": null}'
    assert route_for(classify_intent(FakeLLM(raw=raw), prompt, "q", "none")) == "misc"


def test_make_classifier_uses_the_orchestrator_task_and_prompt(settings):
    built = []

    def builder(model_id, cfg):
        built.append(model_id)
        return FakeLLM()

    classify = make_classifier(settings, builder=builder)
    assert classify("How many sick days do I get?").intent == "kb_query"
    assert built == [settings.models.main.id]  # the model is built on the call, as for every task behind the fallback


# --- topic change clears a pending offer (FR-D2) ---------------------------------------


def _state(awaiting):
    return AgentState(awaiting=awaiting, pending_offer=PendingOffer(unanswered_text="x") if awaiting == "offer_reply" else None)


def _result(intent):
    return IntentResult(intent=intent, confidence=0.9)


@pytest.mark.parametrize("intent", ["kb_query", "ticket_status", "out_of_scope", "smalltalk"])
def test_new_intent_mid_offer_clears_the_pending_offer(intent):
    update = dialog_update(_state("offer_reply"), _result(intent))
    assert update == {"awaiting": "none", "pending_offer": None, "ticket": None}


def test_topic_change_also_clears_a_partly_built_ticket_but_keeps_the_email():
    state = AgentState(
        awaiting="email",
        email="priya.k@smartdesk.example",
        ticket=TicketDraft(title="VPN", description="down", category="IT"),
    )
    update = dialog_update(state, _result("kb_query"))
    assert update == {"awaiting": "none", "pending_offer": None, "ticket": None}
    assert "email" not in update


def test_continuing_the_flow_keeps_the_ticket_draft():
    state = AgentState(awaiting="email", ticket=TicketDraft(title="VPN", description="down", category="IT"))
    assert dialog_update(state, _result("create_ticket")) == {}


def test_agreeing_keeps_the_pending_offer():
    assert dialog_update(_state("offer_reply"), _result("create_ticket")) == {}


def test_eval_topic_change_rows_clear_the_offer(prompt):
    rows = [r for r in ROWS if r.awaiting in ("offer_reply", "email") and r.expected_intent == "kb_query"]
    assert rows
    for r in rows:
        result = classify_intent(FakeLLM(), prompt, r.utterance, r.awaiting)
        assert dialog_update(_state(r.awaiting), result)["awaiting"] == "none"


@pytest.mark.parametrize(
    "awaiting, keep",
    [("email", "create_ticket"), ("ticket_pick", "ticket_status"), ("offer_reply", "create_ticket")],
)
def test_the_flow_continuation_intent_keeps_its_state(awaiting, keep):
    assert dialog_update(_state(awaiting), _result(keep)) == {}


def test_ambiguous_reply_keeps_the_state():
    assert dialog_update(_state("offer_reply"), _result("ambiguous")) == {}


def test_nothing_pending_means_nothing_to_clear():
    assert dialog_update(_state("none"), _result("kb_query")) == {}
