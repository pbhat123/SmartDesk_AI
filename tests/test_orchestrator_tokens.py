"""The orchestrator token count on the scenarios (FR-I3, FR-H5, FR-H7): local, mock models, no API call."""
from evals.orchestrator_tokens import measure, render


def test_the_scenarios_are_counted_with_tiktoken_and_no_api_call(settings, tmp_path):
    report = measure(settings, tmp_path, long_turns=25)
    assert report.turns and report.total_min <= report.total_median <= report.total_max
    for turn in report.turns:
        assert turn.total == turn.without_history + turn.history and turn.prompt < turn.total
        assert turn.history == 0 or turn.turn > 1, "the first turn of a conversation has no conversation block"
    assert max(t.history for t in report.turns) > 0, "a later turn carries the earlier conversation"
    assert report.first_compaction is None and report.projected_compaction_turn > 100
    assert report.trigger_tokens == int(settings.compaction.trigger_fraction * settings.models.main.context_window)
    assert "no API calls" in render(report)


def test_the_run_leaves_the_mock_model_as_it_found_it(settings, tmp_path):
    from smartdesk_ai.llm import mock

    before = mock.MockRoutingModel.invoke
    measure(settings, tmp_path, long_turns=3)
    assert mock.MockRoutingModel.invoke is before


def test_the_history_cap_keeps_a_long_conversation_flat(settings, tmp_path):
    long = measure(settings, tmp_path, long_turns=30).long_conversation
    prompt = 880  # the orchestrator prompt's size in tokens, with a little room
    assert long.history_max_tokens == settings.orchestrator.history_max_tokens == 4000
    assert long.capped_max <= prompt + long.history_max_tokens + 60, "the cap, the prompt and the message"
    assert long.uncapped_max > 1.5 * long.capped_max and long.first_capped_turn is not None
    assert long.usd_per_call_capped_max < long.usd_per_call_uncapped_max
