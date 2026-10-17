"""T4.6: answer from retrieved context only; "I don't know" when it is insufficient (FR-A4,
FR-A15 signal 3). Mock LLM, no network."""
import pytest

from smartdesk_ai.agent.answer import (
    INSUFFICIENT_REPLY,
    AnswerError,
    AnswerReply,
    answer_from_context,
    make_answerer,
    should_escalate,
)
from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.rag.answerable import Signals
from smartdesk_ai.tools.search_kb import KBChunk, SearchKBResult
from tests.llm_fakes import FakeAnswerLLM

CHUNKS = [
    KBChunk(entry_id="IT-PWD-01", domain="IT", title="Password reset", header="IT > Passwords",
            text="Reset your password at the self-service portal.", score=0.8),
    KBChunk(entry_id="HR-LEAVE-03", domain="HR", title="Sick leave", header="HR > Leave",
            text="Sick leave is twelve days a year.", score=0.5),
]


@pytest.fixture(scope="module")
def prompt(settings):
    return load_prompt("answer", settings).text


def search_result(answerable=True, chunks=CHUNKS):
    return SearchKBResult(
        chunks=list(chunks), ordered_by="dense", top_cosine=0.8 if answerable else 0.1,
        answerable=answerable, signals=Signals(no_chunks=not chunks, below_threshold=not answerable and bool(chunks)),
        threshold=0.45,
    )


def test_sufficient_context_gives_the_model_answer(prompt):
    llm = FakeAnswerLLM()
    result = answer_from_context(llm, prompt, "How do I reset my password?", CHUNKS)
    assert result.insufficient is False
    assert result.answer == "Here is what the policy says."
    assert len(llm.calls) == 1


def test_question_and_context_entries_are_sent_to_the_model(prompt):
    llm = FakeAnswerLLM()
    answer_from_context(llm, prompt, "How do I reset my password?", CHUNKS)
    system, human = llm.calls[0]
    assert "only the context" in system.content.lower()
    for needle in ("How do I reset my password?", "[IT-PWD-01]", "self-service portal", "[HR-LEAVE-03]"):
        assert needle in human.content


def test_insufficient_context_yields_an_i_dont_know_reply(prompt):
    result = answer_from_context(FakeAnswerLLM("insufficient"), prompt, "Is the cafeteria open?", CHUNKS)
    assert result.insufficient is True
    assert result.text == INSUFFICIENT_REPLY
    assert "don't know" in result.text.lower()


def test_no_chunks_means_i_dont_know_without_a_model_call(prompt):
    llm = FakeAnswerLLM()
    result = answer_from_context(llm, prompt, "anything", [])
    assert result.insufficient is True and result.text == INSUFFICIENT_REPLY
    assert llm.calls == []


def test_an_empty_answer_without_the_flag_counts_as_insufficient(prompt):
    llm = FakeAnswerLLM("raw", raw='{"answer": "  ", "insufficient": false, "citations": []}')
    assert answer_from_context(llm, prompt, "q", CHUNKS).insufficient is True


@pytest.mark.parametrize("raw", ["I think it is twelve days.", "{}", '{"answer": "x", "insufficient": "maybe"}', '{"answer": "x", "extra": 1}'])
def test_malformed_reply_raises_answer_error(prompt, raw):
    with pytest.raises(AnswerError):
        answer_from_context(FakeAnswerLLM("raw", raw=raw), prompt, "q", CHUNKS)


def test_a_failing_model_call_raises_answer_error(prompt):
    class Boom:
        def invoke(self, messages):
            raise TimeoutError("down")

    with pytest.raises(AnswerError):
        answer_from_context(Boom(), prompt, "q", CHUNKS)


def test_the_llm_insufficient_flag_escalates_even_when_scores_pass(prompt):
    search = search_result(answerable=True)
    answer = answer_from_context(FakeAnswerLLM("insufficient"), prompt, "q", CHUNKS)
    assert search.answerable is True
    assert should_escalate(search, answer) is True


@pytest.mark.parametrize(
    "answerable, insufficient, expected",
    [(True, False, False), (False, False, True), (True, True, True), (False, True, True)],
)
def test_any_one_signal_escalates(prompt, answerable, insufficient, expected):
    answer = answer_from_context(FakeAnswerLLM("insufficient" if insufficient else "cite_top"), prompt, "q", CHUNKS)
    assert should_escalate(search_result(answerable), answer) is expected


def test_failed_retrieval_signals_escalate_without_an_answer():
    assert should_escalate(search_result(answerable=False), None) is True
    assert should_escalate(search_result(answerable=True), None) is False


def test_the_reply_model_is_strict():
    assert AnswerReply(answer="a", insufficient=False, citations=["X-1"]).citations == ["X-1"]
    with pytest.raises(ValueError):
        AnswerReply(answer="a", surprise=True)


def test_make_answerer_uses_the_worker_answer_task_and_active_prompt(settings):
    built = []

    def builder(model_id, cfg):
        built.append(model_id)
        return FakeAnswerLLM()

    answer = make_answerer(settings, builder=builder)
    assert built == [getattr(settings.models, settings.tiers["WORKER_ANSWER"]).id]
    assert answer("q", CHUNKS).insufficient is False


def test_live_client_uses_structured_output(settings):
    seen = {}

    class Live(FakeAnswerLLM):
        def with_structured_output(self, schema):
            seen["schema"] = schema
            return self

    make_answerer(settings, builder=lambda m, c: Live(), structured=True)
    assert seen["schema"] is AnswerReply


def test_the_answer_prompt_forbids_outside_knowledge(prompt):
    assert "only the context" in prompt.lower() and "insufficient" in prompt
