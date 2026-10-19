"""The orchestrator token count on the scenarios (FR-I3, FR-H5, FR-H7): local, mock models, no API call."""
from evals.orchestrator_tokens import measure, render


def test_the_scenarios_are_counted_with_tiktoken_and_no_api_call(settings, tmp_path):
    report = measure(settings, tmp_path)
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
    measure(settings, tmp_path)
    assert mock.MockRoutingModel.invoke is before
