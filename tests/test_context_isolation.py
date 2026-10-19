"""T7.3: a worker sees only its sub-question, its retrieved chunks and its domain prompt, never the conversation
(FR-I3, FR-G11). Mock models, no network."""
import json
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage

from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.agent.state import new_thread_id
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM

EARLIER = "How many casual leave days do I get?"


class Orchestrator:
    def __init__(self):
        self.n = 0

    def invoke(self, messages):
        self.n += 1
        if self.n == 1:
            reply = {"intent": "kb_query", "confidence": 0.95, "domain": "HR",
                     "parts": [{"text": EARLIER, "domain": "HR", "follow_up": False}]}
        else:
            reply = {"intent": "kb_query", "confidence": 0.95, "domain": "HR",
                     "parts": [{"text": "How many casual leave days do contractors get?", "domain": "HR", "follow_up": True}]}
        return SimpleNamespace(content=json.dumps(reply))


def test_the_worker_input_is_the_sub_question_and_chunks_only(built_index_dir, settings, tmp_path):
    answer = FakeAnswerLLM()
    stack = assemble(built_index_dir, settings, tmp_path, llm=answer, routed=True, route_model=Orchestrator())
    tid = new_thread_id()
    run_turn(stack.graph, tid, EARLIER, email="jane.roe@example.com")
    run_turn(stack.graph, tid, "what about contractors?")
    messages = answer.calls[-1]
    assert [type(m) for m in messages] == [SystemMessage, HumanMessage]
    assert messages[0].content == load_prompt("worker_hr", settings).text
    human = messages[1].content
    assert human.startswith("Question: How many casual leave days do contractors get?\n\nContext:\n[")
    assert "what about" not in human and EARLIER not in human
    assert "jane.roe" not in human and "Assistant" not in human and "Employee" not in human


def test_the_worker_input_has_nothing_from_a_long_conversation(built_index_dir, settings, tmp_path):
    answer = FakeAnswerLLM()
    stack = assemble(built_index_dir, settings, tmp_path, llm=answer, routed=True)
    tid = new_thread_id()
    for text in ("hello there", "Thanks a lot", "How do I reset my forgotten password?"):
        run_turn(stack.graph, tid, text)
    everything = " ".join(str(m.content) for m in answer.calls[-1])
    assert "hello there" not in everything and "Thanks a lot" not in everything
