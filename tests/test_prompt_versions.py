"""Active prompt versions after the live runs (FR-H2, FR-A5): the judge lists only contradicted or missing claims,
the answer prompt allows simple inferences and partial answers. v1 files stay for comparison."""
import json

from smartdesk_ai.agent.prompt_loader import available_versions, load_prompt


def test_grounding_judge_v2_is_active_and_keeps_v1(settings):
    prompt = load_prompt("grounding_judge", settings)
    assert prompt.version == 2 and 1 in available_versions("grounding_judge")
    assert "contradicts or does not contain" in prompt.text
    assert "different question" in prompt.text  # an off-topic answer must still fail (gap-004)
    assert '{"grounded"' in prompt.text and json.loads('{"grounded": true, "unsupported": []}')


def test_answer_v2_is_active_and_keeps_v1(settings):
    prompt = load_prompt("answer", settings)
    assert prompt.version == 2 and 1 in available_versions("answer")
    assert "simple inference" in prompt.text and "partial answer" in prompt.text
    assert "core fact" in prompt.text and "different topic" in prompt.text  # the gap guard stays
    assert '"insufficient"' in prompt.text and '"citations"' in prompt.text
