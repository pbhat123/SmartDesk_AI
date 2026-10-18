"""Active prompt versions after the live runs (FR-H2, FR-A5): the judge lists only contradicted or missing claims,
the answer prompt allows simple inferences and partial answers. v1 files stay for comparison."""
import json

from smartdesk_ai.agent.prompt_loader import available_versions, load_prompt


def test_grounding_judge_v3_is_active_and_keeps_older_versions(settings):
    prompt = load_prompt("grounding_judge", settings)
    assert prompt.version == 3 and {1, 2} <= set(available_versions("grounding_judge"))
    assert "contradicts or does not contain" in prompt.text
    assert "direct answer to the question" in prompt.text and "neighbouring item" in prompt.text  # gap-006
    assert "different question" in prompt.text  # an off-topic answer must still fail (gap-004)
    assert '{"grounded"' in prompt.text and json.loads('{"grounded": true, "unsupported": []}')


def test_answer_v3_is_active_and_keeps_older_versions(settings):
    prompt = load_prompt("answer", settings)
    assert prompt.version == 3 and {1, 2} <= set(available_versions("answer"))
    text = prompt.text
    assert "specific item" in text and "parking fees" in text  # gap-006: a neighbouring rule is not the item asked about
    assert "several parts" in text and "A question about one item never gets a partial answer" in text
    assert 'Never begin with "Yes" or "No" unless the context states that fact' in text
    assert "different topic" in text
    assert '"insufficient"' in text and '"citations"' in text
