"""T4.7: every KB answer names at least one retrieved entry (FR-A13). Mock LLM; retrieval runs
for real over the shipped KB with the hashing embedder."""
import pytest

from evals.common import EVALS_DIR, load_jsonl
from evals.schemas import KBQuestion
from smartdesk_ai.agent.answer import INSUFFICIENT_REPLY, answer_from_context
from smartdesk_ai.agent.prompt_loader import load_prompt
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.rag.embeddings import HashingEmbeddings
from smartdesk_ai.rag.index import open_index
from smartdesk_ai.tools.executor import ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry
from smartdesk_ai.tools.search_kb import KBChunk, make_search_kb
from tests.llm_fakes import FakeAnswerLLM

CHUNKS = [
    KBChunk(entry_id="IT-PWD-01", domain="IT", title="Password reset", header="h", text="Use the portal.", score=0.8),
    KBChunk(entry_id="HR-LEAVE-03", domain="HR", title="Sick leave", header="h", text="Twelve days.", score=0.5),
]


@pytest.fixture(scope="module")
def prompt(settings):
    return load_prompt("answer", settings).text


def raw_reply(citations):
    import json

    return FakeAnswerLLM("raw", raw=json.dumps({"answer": "Use the portal.", "insufficient": False, "citations": citations}))


def test_answer_text_names_the_cited_entries(prompt):
    result = answer_from_context(raw_reply(["IT-PWD-01"]), prompt, "q", CHUNKS)
    assert result.citations == ["IT-PWD-01"]
    assert "Sources: IT-PWD-01 (Password reset)" in result.text
    assert result.text.startswith("Use the portal.")


def test_several_citations_keep_order_and_drop_duplicates(prompt):
    result = answer_from_context(raw_reply(["HR-LEAVE-03", "IT-PWD-01", "HR-LEAVE-03"]), prompt, "q", CHUNKS)
    assert result.citations == ["HR-LEAVE-03", "IT-PWD-01"]


def test_an_id_that_was_not_retrieved_is_never_cited(prompt):
    result = answer_from_context(raw_reply(["IT-PWD-01", "HR-MADEUP-99"]), prompt, "q", CHUNKS)
    assert result.citations == ["IT-PWD-01"]
    assert "HR-MADEUP-99" not in result.text


@pytest.mark.parametrize("citations", [[], ["HR-MADEUP-99"]])
def test_an_answer_without_a_valid_citation_is_not_shown(prompt, citations):
    result = answer_from_context(raw_reply(citations), prompt, "q", CHUNKS)
    assert result.insufficient is True and result.citations == []
    assert result.text == INSUFFICIENT_REPLY


def test_an_i_dont_know_reply_cites_nothing(prompt):
    result = answer_from_context(FakeAnswerLLM("insufficient"), prompt, "q", CHUNKS)
    assert result.citations == [] and "Sources" not in result.text


QUESTIONS = [KBQuestion(**r) for r in load_jsonl(EVALS_DIR / "kb_questions.jsonl")]


@pytest.fixture(scope="module")
def run(built_index_dir, settings, prompt):
    index = open_index(built_index_dir, HashingEmbeddings())
    registry = ToolRegistry()
    registry.register("search_kb", make_search_kb(index, settings, reranker=None, health=HealthRegistry()))
    executor = ToolExecutor(registry, settings)

    def go(question):
        found = executor.execute("search_kb", {"query": question.question}, agent="eval").data
        return found, answer_from_context(FakeAnswerLLM(), prompt, question.question, found.chunks)

    return go


def test_every_kb_answer_in_the_mock_eval_run_cites_a_retrieved_entry(run):
    answered = 0
    for q in QUESTIONS:
        found, answer = run(q)
        if answer.insufficient:
            continue
        answered += 1
        retrieved = {c.entry_id for c in found.chunks}
        assert answer.citations, q.id
        assert set(answer.citations) <= retrieved, q.id
        assert all(entry_id in answer.text for entry_id in answer.citations), q.id
    assert answered >= len(QUESTIONS) // 2, "the mock run should answer most of the KB set"
