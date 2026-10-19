"""orchestrator.history_max_tokens (FR-I3, FR-H5, FR-H7): the orchestrator gets the compacted summary plus the most recent
turns that fit the cap, newest first, masked. Compaction at 75% of the window is far too late for a Main call on every
turn, so this keeps the call's size flat however long the conversation gets. Mock models, tiktoken, no network."""
import json
import re
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.orchestrator import transcript_for
from smartdesk_ai.agent.state import STATUS_PLACEHOLDER, STATUS_REPLY, new_thread_id
from smartdesk_ai.ops.compaction import SUMMARY_PREFIX, encoding_for
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM

MODEL = "gpt-5.4-mini"


def tokens(text):
    return len(encoding_for(MODEL).encode(text))


def conversation(turns, words=60):
    """A synthetic conversation: each turn is an employee question and an assistant answer of about `words` words."""
    out = []
    for i in range(turns):
        filler = " ".join(f"t{i}w{n}" for n in range(words))
        out += [HumanMessage(f"question {i}: {filler}"), AIMessage(f"answer {i}: {filler}")]
    return out + [HumanMessage("the current question")]


def kept_turns(text):
    return [int(m.group(1)) for line in text.splitlines() if (m := re.match(r"Employee: question (\d+):", line))]


def test_the_config_default_is_4000_tokens(settings):
    assert settings.orchestrator.history_max_tokens == 4000 and settings.orchestrator.max_parts == 4


def test_a_long_conversation_is_cut_to_the_cap(settings):
    text = transcript_for(conversation(200), max_tokens=4000, model_id=MODEL)
    assert 3000 < tokens(text) <= 4000, "the cap is filled with whole turns, never exceeded"


def test_the_most_recent_turns_are_kept_in_order_and_the_oldest_dropped(settings):
    text = transcript_for(conversation(200), max_tokens=4000, model_id=MODEL)
    kept = kept_turns(text)
    assert kept == list(range(kept[0], 200)), "a contiguous run ending at the newest earlier turn, oldest first"
    assert kept[0] > 100 and 0 not in kept
    assert text.splitlines()[0].startswith("Employee: question"), "it starts at a question, not at an orphan answer"
    assert text.rstrip().splitlines()[-1].startswith("Assistant: answer 199")


def test_the_size_is_flat_however_long_the_conversation_gets(settings):
    sizes = [tokens(transcript_for(conversation(n), max_tokens=4000, model_id=MODEL)) for n in (60, 120, 400, 1000)]
    assert max(sizes) <= 4000 and min(sizes) > 3000


def test_a_conversation_under_the_cap_is_passed_whole(settings):
    messages = conversation(5)
    assert transcript_for(messages, max_tokens=4000, model_id=MODEL) == transcript_for(messages)
    assert kept_turns(transcript_for(messages, max_tokens=4000, model_id=MODEL)) == [0, 1, 2, 3, 4]


def test_the_summary_comes_first_and_is_always_kept(settings):
    summary = SystemMessage(SUMMARY_PREFIX + "the employee asked about leave and VPN.")
    text = transcript_for([summary, *conversation(200)], max_tokens=4000, model_id=MODEL)
    assert text.startswith(SUMMARY_PREFIX + "the employee asked about leave and VPN.\n")
    assert tokens(text) <= 4000 and kept_turns(text)[-1] == 199


def test_an_oversized_summary_takes_at_most_half_the_cap(settings):
    summary = SystemMessage(SUMMARY_PREFIX + " ".join(f"s{n}" for n in range(6000)))
    text = transcript_for([summary, *conversation(50)], max_tokens=1000, model_id=MODEL)
    assert tokens(text) <= 1000 and text.startswith(SUMMARY_PREFIX) and kept_turns(text)[-1] == 49
    head = text.split("\nEmployee:")[0]
    assert tokens(head) <= 500


def test_a_newest_turn_bigger_than_the_cap_is_cut_not_dropped(settings):
    messages = [HumanMessage("question 0: " + "x " * 3000), AIMessage("answer 0: " + "y " * 3000), HumanMessage("now")]
    text = transcript_for(messages, max_tokens=200, model_id=MODEL)
    assert text.startswith("Employee: question 0") and 0 < tokens(text) <= 200


def test_masking_still_applies_under_the_cap(settings):
    messages = [HumanMessage("my email is jane.roe@example.com, hello"),
                AIMessage("Ticket #4\n> SECRET-TEAM-TEXT", name=STATUS_REPLY), HumanMessage("now")]
    text = transcript_for(messages, max_tokens=4000, model_id=MODEL)
    assert "jane.roe" not in text and "[email]" in text and "SECRET" not in text and STATUS_PLACEHOLDER in text


def test_the_first_turn_still_has_no_history(settings):
    assert transcript_for([HumanMessage("hello")], max_tokens=4000, model_id=MODEL) == ""


# --- in the graph -------------------------------------------------------------------------------------------------

class Recording:
    def __init__(self):
        self.inputs = []

    def invoke(self, messages):
        self.inputs.append(str(messages[-1].content))
        reply = {"intent": "kb_query", "confidence": 0.9, "domain": "IT", "parts": []}
        return SimpleNamespace(content=json.dumps(reply))


def block(human):
    return human.split("Conversation so far:\n", 1)[1].rsplit("\nMessage: ", 1)[0] if "Conversation so far" in human else ""


def with_cap(settings, cap):
    return settings.model_copy(update={"orchestrator": settings.orchestrator.model_copy(update={"history_max_tokens": cap})})


def test_every_orchestrator_call_in_a_long_chat_stays_under_the_cap(built_index_dir, settings, tmp_path):
    cfg = with_cap(settings, 400)
    model = Recording()
    stack = assemble(built_index_dir, cfg, tmp_path, llm=FakeAnswerLLM(), routed=True, route_model=model)
    tid = new_thread_id()
    for i in range(25):
        run_turn(stack.graph, tid, f"How do I reset my forgotten password? (question {i})")
    blocks = [block(h) for h in model.inputs]
    assert blocks[0] == "" and all(tokens(b) <= 400 for b in blocks)
    assert tokens(blocks[-1]) > 300, "late in the chat the block is full, not empty"
    assert "question 23" in blocks[-1] and "question 0)" not in blocks[-1]


def test_a_follow_up_after_a_long_chat_still_sees_the_turn_before_it(built_index_dir, settings, tmp_path):
    cfg = with_cap(settings, 400)
    model = Recording()
    stack = assemble(built_index_dir, cfg, tmp_path, llm=FakeAnswerLLM(), routed=True, route_model=model)
    tid = new_thread_id()
    for i in range(25):
        run_turn(stack.graph, tid, f"How many casual leave days do I get? (variant {i})")
    run_turn(stack.graph, tid, "what about contractors?")
    seen = block(model.inputs[-1])
    assert "variant 24" in seen and "Assistant:" in seen
