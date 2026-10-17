"""T5.7: the runtime grounding judge runs inside the answering step; an unsupported draft answer becomes
the insufficient-context signal, so the whole-KB retry and ticket offer follow (FR-H2, FR-A15, FR-I6).
Mock LLMs only."""
import json
from types import SimpleNamespace

import pytest

from smartdesk_ai.agent.answer import INSUFFICIENT_REPLY, answer_from_context, make_answerer
from smartdesk_ai.agent.graph import run_turn
from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.agent.state import new_thread_id
from smartdesk_ai.llm.mock import MockAnswerModel, MockGroundingModel
from smartdesk_ai.agent.grounding import GroundingReply, make_grounding_judge
from smartdesk_ai.tools.search_kb import KBChunk
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM

CHUNKS = [KBChunk(entry_id="IT-PWD-01", domain="IT", title="Password reset", header="IT > Passwords",
                  text="Reset your password at the self-service portal.", score=0.8)]
Q = "How do I reset my forgotten password?"


class StubJudge:
    """Returns the scripted verdicts in order, the last one repeating."""

    def __init__(self, *verdicts):
        self.verdicts, self.calls = list(verdicts), []

    def __call__(self, question, answer, chunks):
        self.calls.append((question, answer, [c.entry_id for c in chunks]))
        return self.verdicts[min(len(self.calls) - 1, len(self.verdicts) - 1)]


@pytest.fixture(scope="module")
def prompt(settings):
    return load_prompt("answer", settings).text


def test_a_grounded_answer_is_kept_and_judged_once(prompt):
    judge = StubJudge(True)
    result = answer_from_context(FakeAnswerLLM(), prompt, Q, CHUNKS, judge=judge)
    assert not result.insufficient and result.citations == ["IT-PWD-01"]
    assert judge.calls == [(Q, "Here is what the policy says.", ["IT-PWD-01"])]


def test_an_unsupported_answer_becomes_the_insufficient_signal(prompt):
    result = answer_from_context(FakeAnswerLLM(), prompt, Q, CHUNKS, judge=StubJudge(False))
    assert result.insufficient and result.text == INSUFFICIENT_REPLY and result.citations == []


def test_no_judge_call_when_there_is_nothing_to_judge(prompt):
    judge = StubJudge(True)
    answer_from_context(FakeAnswerLLM("insufficient"), prompt, Q, CHUNKS, judge=judge)
    answer_from_context(FakeAnswerLLM(), prompt, Q, [], judge=judge)
    assert judge.calls == []


def test_a_judge_that_fails_counts_as_ungrounded(prompt):
    def broken(question, answer, chunks):
        raise ConnectionError("down")

    assert answer_from_context(FakeAnswerLLM(), prompt, Q, CHUNKS, judge=broken).insufficient


def test_unsupported_draft_goes_to_the_whole_kb_retry_then_the_ticket_offer(built_index_dir, settings, tmp_path):
    judge = StubJudge(False)
    stack = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("cite_top"), judge=judge)
    state = run_turn(stack.graph, new_thread_id(), Q, search_domain="IT")
    assert [d for _, d in stack.spy.calls] == ["IT", None]
    assert state.steps == ["start", "search_kb", "answer", "search_kb", "answer", "offer"]
    assert state.part_result.llm_insufficient and state.part_result.retried and state.pending_offer is not None
    assert "Here is what the policy says" not in state.reply and "ticket" in state.reply.lower()


def test_a_retry_that_is_grounded_avoids_the_offer(built_index_dir, settings, tmp_path):
    stack = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("cite_top"), judge=StubJudge(False, True))
    state = run_turn(stack.graph, new_thread_id(), Q, search_domain="IT")
    assert state.pending_offer is None and state.part_result.retried and "Sources:" in state.reply


# --- the real judge wrapper, with a fake model -------------------------------------------

class FakeJudgeLLM:
    def __init__(self, reply=None, raw=None, error=None):
        self.reply, self.raw, self.error, self.calls = reply, raw, error, []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.raw if self.raw is not None else json.dumps(self.reply))


def judge_with(settings, llm):
    return make_grounding_judge(settings, builder=lambda m, c: llm, sleep=lambda _: None)


def test_judge_sends_question_answer_and_context_and_reads_the_verdict(settings):
    llm = FakeJudgeLLM({"grounded": True, "unsupported": []})
    assert judge_with(settings, llm)(Q, "Use the portal.", CHUNKS) is True
    system, human = llm.calls[0]
    assert "grounded" in system.content.lower()
    for needle in (Q, "Use the portal.", "[IT-PWD-01]", "self-service portal"):
        assert needle in human.content


def test_judge_false_verdict(settings):
    assert judge_with(settings, FakeJudgeLLM({"grounded": False, "unsupported": ["a number"]}))(Q, "x", CHUNKS) is False


def test_malformed_judge_reply_raises_so_the_answer_step_treats_it_as_ungrounded(settings):
    with pytest.raises(Exception):
        judge_with(settings, FakeJudgeLLM(raw="looks fine"))(Q, "x", CHUNKS)


def test_judge_uses_the_main_model_task(settings):
    assert settings.tiers["GROUNDING_JUDGE"] == "main"


def test_mock_judge_accepts_the_mock_answers_and_rejects_invented_claims(settings):
    judge = judge_with(settings, MockGroundingModel())
    answer = json.loads(MockAnswerModel().invoke([None, SimpleNamespace(
        content="Question: reset forgotten password portal\n\nContext:\n[IT-PWD-01] Password reset\nReset your password at the "
                "self-service portal. Then sign in again.")]).content)["answer"]
    assert judge(Q, answer, CHUNKS) is True
    assert judge(Q, "Call 555-0199 and the CEO will reset it within 2 hours.", CHUNKS) is False


def test_make_answerer_passes_the_judge_through(settings):
    judge = StubJudge(False)
    answer = make_answerer(settings, builder=lambda m, c: FakeAnswerLLM(), judge=judge)
    assert answer(Q, CHUNKS).insufficient and len(judge.calls) == 1
