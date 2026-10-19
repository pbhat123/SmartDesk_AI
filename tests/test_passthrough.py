"""T7.5: a question with one part in one domain goes to one worker only and the synthesizer's LLM call is skipped; the
output guard still runs on the reply (FR-I5, FR-H2). Mock and scripted models, no network."""
import pytest

from smartdesk_ai.agent.chat_service import ChatService
from smartdesk_ai.agent.graph import make_kb_responder, run_turn
from smartdesk_ai.agent.state import new_thread_id
from tests.agent_helpers import assemble
from tests.llm_fakes import FakeAnswerLLM
from tests.test_fanout import COMPOUND, ByQuestion, Synth

SINGLES = {
    "IT": "How do I reset my forgotten password?",
    "HR": "How many casual leave days do I get?",
    "MISC": "Is there free parking at the Pune office?",
}


def stack_for(built_index_dir, settings, tmp_path, answer, synth):
    return assemble(built_index_dir, settings, tmp_path, llm=answer, routed=True, synth_model=synth)


@pytest.mark.parametrize("domain", sorted(SINGLES))
def test_a_single_part_question_calls_one_worker_and_no_synthesizer(built_index_dir, settings, tmp_path, domain):
    answer, synth = FakeAnswerLLM("cite_top"), Synth()
    stack = stack_for(built_index_dir, settings, tmp_path, answer, synth)
    state = run_turn(stack.graph, new_thread_id(), SINGLES[domain])
    assert len(state.parts) == 1 and state.parts[0].domain == domain
    assert synth.calls == [] and not any(s.startswith("synthesize") for s in state.steps)
    assert not any(s.startswith("fan_out") for s in state.steps)
    assert state.steps[-1] in ("reply", "offer")


def test_the_first_search_belongs_to_the_one_worker_of_the_domain(built_index_dir, settings, tmp_path):
    stack = stack_for(built_index_dir, settings, tmp_path, FakeAnswerLLM("cite_top"), Synth())
    run_turn(stack.graph, new_thread_id(), SINGLES["IT"])
    assert stack.spy.calls[0][1] == "IT" and all(d in ("IT", None) for _, d in stack.spy.calls)  # IT, then maybe the retry


def test_an_unanswered_single_part_offers_a_ticket_without_the_synthesizer(built_index_dir, settings, tmp_path):
    synth = Synth()
    stack = stack_for(built_index_dir, settings, tmp_path, FakeAnswerLLM("insufficient"), synth)
    state = run_turn(stack.graph, new_thread_id(), SINGLES["MISC"])
    assert state.pending_offer is not None and synth.calls == []


def test_a_cached_single_part_answer_skips_the_synthesizer_too(built_index_dir, settings, tmp_path):
    from smartdesk_ai.ops.cache import AnswerCache, MemoryBackend

    synth = Synth()
    stack = assemble(built_index_dir, settings, tmp_path, llm=FakeAnswerLLM("cite_top"), routed=True, synth_model=synth,
                     cache=AnswerCache(MemoryBackend(10), kb_version=lambda: "v"))
    run_turn(stack.graph, new_thread_id(), SINGLES["IT"])
    state = run_turn(stack.graph, new_thread_id(), SINGLES["IT"])
    assert "cache_hit:exact" in state.steps and synth.calls == []


# --- the output guard still runs (it sits in ChatService, after the graph) --------------------------------------------

def guarded(built_index_dir, settings, tmp_path, answer, synth):
    shown = []

    def output_guard(text):
        shown.append(text)
        return text

    stack = stack_for(built_index_dir, settings, tmp_path, answer, synth)
    return ChatService(settings, responder=make_kb_responder(stack.graph), output_guard=output_guard), shown


def test_the_output_guard_runs_on_a_pass_through_reply(built_index_dir, settings, tmp_path):
    synth = Synth()
    service, shown = guarded(built_index_dir, settings, tmp_path, FakeAnswerLLM("cite_top"), synth)
    reply = service.handle("s1", SINGLES["IT"])
    assert shown == [reply.text] and "Sources:" in reply.text and synth.calls == []


def test_the_output_guard_runs_on_a_synthesized_reply_as_well(built_index_dir, settings, tmp_path):
    synth = Synth()
    service, shown = guarded(built_index_dir, settings, tmp_path, ByQuestion(), synth)
    reply = service.handle("s1", COMPOUND)
    assert len(synth.calls) == 1 and shown == [reply.text]


def test_the_output_guard_can_still_change_a_pass_through_reply(built_index_dir, settings, tmp_path):
    stack = stack_for(built_index_dir, settings, tmp_path, FakeAnswerLLM("cite_top"), Synth())
    service = ChatService(settings, responder=make_kb_responder(stack.graph), output_guard=lambda t: "[checked] " + t)
    assert service.handle("s1", SINGLES["HR"]).text.startswith("[checked] ")
