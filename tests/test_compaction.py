"""T5.11: context compaction at 75% of the active model's window (FR-H5). Tests use small windows in
the config copy so the trigger can be reached; the summarizer is a stub, so there are no LLM calls."""
import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from smartdesk_ai.ops.compaction import SUMMARY_PREFIX, Compactor, count_tokens_in, make_summarizer, mask_summary

EMAIL = "jane.roe@example.com"


def small(settings, main_window=2000, light_window=700, light_out=100, keep=2, fraction=0.75):
    models = settings.models.model_copy(update={
        "main": settings.models.main.model_copy(update={"context_window": main_window}),
        "light": settings.models.light.model_copy(update={"context_window": light_window, "max_output": light_out}),
    })
    compaction = settings.compaction.model_copy(update={"keep_last_turns": keep, "trigger_fraction": fraction})
    return settings.model_copy(update={"models": models, "compaction": compaction})


class Stub:
    """A summarizer that records each call and answers 'S<n>'."""

    def __init__(self):
        self.calls = []

    def __call__(self, prior, transcript, tier):
        self.calls.append((prior, transcript, tier))
        return f"S{len(self.calls)}"


def turn(i, words=40):
    filler = " ".join(f"w{i}x{n}" for n in range(words))
    return [HumanMessage(f"question {i} {filler}", id=f"h{i}"), AIMessage(f"answer {i} {filler}", id=f"a{i}")]


def script(turns, words=40):
    out = []
    for i in range(turns):
        out += turn(i, words)
    return out


def test_below_75_percent_nothing_happens(settings):
    cfg = small(settings, main_window=4000)
    stub = Stub()
    messages = script(5)
    assert count_tokens_in(messages, cfg.models.main.id) < 0.75 * 4000
    assert Compactor(cfg, stub).compact(messages) is None and stub.calls == []


def test_a_long_script_triggers_at_75_percent_of_the_main_window(settings):
    cfg = small(settings, main_window=1000, light_window=2000, light_out=100)
    comp = Compactor(cfg, Stub())
    model_id = cfg.models.main.id
    first = next(n for n in range(1, 40) if comp.needs(script(n, 10), 0))
    assert count_tokens_in(script(first, 10), model_id) >= 750 > count_tokens_in(script(first - 1, 10), model_id)
    messages = script(first + 6, 10)
    result = Compactor(cfg, Stub()).compact(messages)
    assert result is not None and result.tier == "light" and result.slices == 1
    assert count_tokens_in(result.messages, model_id) < count_tokens_in(messages, model_id)


def test_the_last_turns_are_kept_as_they_are_and_the_summary_comes_first(settings):
    cfg = small(settings, main_window=1000, light_window=2000, keep=2)
    messages = script(12, 10)
    result = Compactor(cfg, Stub()).compact(messages)
    assert isinstance(result.messages[0], SystemMessage) and result.messages[0].content == SUMMARY_PREFIX + "S1"
    assert result.messages[1:] == messages[-4:]  # the last two employee turns with their replies
    assert result.cut == len(messages) - 4


def test_the_summary_replaces_the_first_message_id_so_the_graph_can_swap_it_in_place(settings):
    cfg = small(settings, main_window=1000, light_window=2000)
    messages = script(12, 10)
    assert Compactor(cfg, Stub()).compact(messages).messages[0].id == messages[0].id


def test_a_running_summary_is_fed_into_the_next_compaction(settings):
    cfg = small(settings, main_window=1000, light_window=2000, keep=2)
    stub = Stub()
    first = Compactor(cfg, stub).compact(script(12, 10))
    longer = first.messages + script(30, 10)[24:]  # more turns after the first summary
    second = Compactor(cfg, stub).compact(longer)
    prior, transcript, _ = stub.calls[1]
    assert prior == "S1" and SUMMARY_PREFIX not in transcript
    assert second.messages[0].content == SUMMARY_PREFIX + "S2"


def test_messages_from_protect_from_on_are_never_summarized(settings):
    cfg = small(settings, main_window=1000, light_window=2000, keep=1)
    messages = script(12, 10)
    result = Compactor(cfg, Stub()).compact(messages, protect_from=14)  # a ticket began at message 14
    assert result.cut <= 14 and result.messages[1:] == messages[result.cut:]
    assert Compactor(cfg, Stub()).compact(messages, protect_from=1) is None  # nothing before it to summarize


def test_after_a_budget_switch_the_trigger_uses_the_light_window(settings):
    cfg = small(settings, main_window=4000, light_window=700, light_out=100)
    messages = script(4)  # roughly 400 tokens: far below 75% of 4000, above 75% of 700
    comp = Compactor(cfg, Stub())
    assert comp.needs(messages, budget_level=0) is False
    assert comp.needs(messages, budget_level=1) is True
    assert comp.compact(messages, budget_level=0) is None
    assert comp.compact(messages, budget_level=1) is not None


def test_history_that_fits_light_is_summarized_by_light_in_one_call(settings):
    cfg = small(settings, main_window=1000, light_window=2000)
    stub = Stub()
    Compactor(cfg, stub).compact(script(12, 10))
    assert [t for *_, t in stub.calls] == ["light"]


def test_history_over_the_light_window_uses_main_at_level_zero(settings):
    cfg = small(settings, main_window=3000, light_window=500, light_out=100)
    stub = Stub()
    result = Compactor(cfg, stub).compact(script(16, 20))
    assert result.tier == "main" and result.slices == 1 and [t for *_, t in stub.calls] == ["main"]


def test_at_budget_level_one_the_history_is_summarized_in_light_sized_slices(settings):
    cfg = small(settings, main_window=3000, light_window=500, light_out=100)
    stub = Stub()
    comp = Compactor(cfg, stub, overhead_tokens=50)
    result = comp.compact(script(16, 20), budget_level=1)
    assert result.tier == "light" and result.slices == len(stub.calls) > 1
    limit = 500 - 100 - 50
    for prior, transcript, tier in stub.calls:
        assert tier == "light"
        assert count_tokens_in([HumanMessage(prior), HumanMessage(transcript)], cfg.models.light.id) <= limit
    assert [c[0] for c in stub.calls][1:] == [f"S{n}" for n in range(1, len(stub.calls))]  # each slice builds on the last
    assert result.messages[0].content == SUMMARY_PREFIX + f"S{len(stub.calls)}"


def test_a_single_oversized_message_is_cut_down_to_fit_a_slice(settings):
    cfg = small(settings, main_window=1000, light_window=500, light_out=100)
    stub = Stub()
    huge = [HumanMessage("big " * 2000, id="h0"), AIMessage("ok", id="a0")] + script(4)[0:8]
    Compactor(cfg, stub).compact(huge, budget_level=1)
    for prior, transcript, _ in stub.calls:
        assert count_tokens_in([HumanMessage(prior + transcript)], cfg.models.light.id) <= 400


def test_the_summary_never_holds_a_raw_email():
    assert EMAIL not in mask_summary(f"The employee {EMAIL} asked about VPN, call +91 98765 43210")
    assert "j***@example.com" in mask_summary(f"mail {EMAIL}") and "98765" not in mask_summary("+91 98765 43210")


def test_the_real_summarizer_sends_the_prompt_prior_summary_and_transcript(settings):
    from types import SimpleNamespace

    seen = []

    class Model:
        def invoke(self, messages):
            seen.append(messages)
            return SimpleNamespace(content=f"summary mentioning {EMAIL}")

    summarize = make_summarizer(settings, builder=lambda m, c: Model(), sleep=lambda _: None)
    out = summarize("old summary", "Employee: hi", "light")
    system, human = seen[0]
    assert "running summary" in system.content and "old summary" in human.content and "Employee: hi" in human.content
    assert EMAIL not in out


def test_a_summarizer_outage_leaves_the_messages_alone(settings, caplog):
    def down(prior, transcript, tier):
        raise ConnectionError("down")

    cfg = small(settings, main_window=1000, light_window=2000)
    with caplog.at_level(logging.WARNING):
        assert Compactor(cfg, down).compact(script(12)) is None
    assert "compaction skipped" in caplog.text


# --- in the graph --------------------------------------------------------------------

def test_the_graph_compacts_old_turns_and_keeps_the_email_and_the_ticket_flow(built_index_dir, settings, tmp_path):
    from smartdesk_ai.agent.confirm import make_confirm_parser
    from smartdesk_ai.agent.graph import run_turn
    from smartdesk_ai.agent.state import new_thread_id
    from smartdesk_ai.llm.mock import MockConfirmModel, MockDraftModel
    from tests.agent_helpers import assemble
    from tests.llm_fakes import FakeAnswerLLM

    cfg = small(settings, main_window=240, light_window=4000, keep=1)
    stub = Stub()
    stack = assemble(built_index_dir, cfg, tmp_path, llm=FakeAnswerLLM(["cite_top"] * 3 + ["insufficient"]),
                     routed=True, drafter_llm=MockDraftModel(),
                     parse_confirm=make_confirm_parser(cfg, builder=lambda m, c: MockConfirmModel()),
                     compactor=Compactor(cfg, stub))
    tid = new_thread_id()
    for question in ("How do I reset my forgotten password?", "How do I set up MFA on a new phone?",
                     "How do I connect to the VPN from home?"):
        state = run_turn(stack.graph, tid, question)
    used = count_tokens_in(state.messages, cfg.models.main.id)
    assert stub.calls == [] and 0.4 * 240 < used < 0.75 * 240  # still below the line
    run_turn(stack.graph, tid, "My monitor keeps flickering")       # gap: ticket offered, email requested
    state = run_turn(stack.graph, tid, "priya.k@smartdesk.example")  # summary shown at the gate
    assert stub.calls, "the history reached 75% of the window, so it was summarized"
    assert isinstance(state.messages[0], SystemMessage) and state.messages[0].content.startswith(SUMMARY_PREFIX)
    assert state.email == "priya.k@smartdesk.example" and state.ticket is not None
    assert "Shall I go ahead" in state.reply
    state = run_turn(stack.graph, tid, "Yes, create it")
    [issue] = stack.client._issues.values()
    assert issue.title.lower().startswith("my monitor") and "#" in state.reply
    assert state.ticket_context_start >= 0


def test_no_compaction_node_without_a_compactor(built_index_dir, settings, tmp_path):
    from tests.agent_helpers import assemble

    stack = assemble(built_index_dir, settings, tmp_path)
    assert "compact" not in stack.graph.get_graph().nodes


def test_the_mock_compaction_model_gives_a_deterministic_summary(settings):
    from smartdesk_ai.llm.mock import MockCompactionModel

    summarize = make_summarizer(settings, builder=lambda m, c: MockCompactionModel(), sleep=lambda _: None)
    first = summarize("", "Employee: how do I reset my password today please\nAssistant: use the portal", "light")
    assert "1 more question" in first and "how do I reset my password today please" in first
    assert first in summarize(first, "Employee: and the VPN?", "light")
