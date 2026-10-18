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


def test_answer_v4_is_active_and_keeps_older_versions(settings):
    prompt = load_prompt("answer", settings)
    assert prompt.version == 4 and {1, 2, 3} <= set(available_versions("answer"))
    text = prompt.text
    assert '"insufficient"' in text and '"citations"' in text


def test_answer_v4_keeps_every_v3_rule_including_the_gap_006_guard():
    """New versions are new files: v4 must contain each v3 rule line unchanged, except the two it replaces."""
    v3 = [l for l in load_prompt_text(3).splitlines() if l.startswith("- ")]
    v4 = load_prompt_text(4)
    replaced = ('- Otherwise answer directly', '- "citations" lists')
    kept = [l for l in v3 if not l.startswith(replaced)]
    assert len(kept) == 6 and all(line in v4 for line in kept)
    assert "specific item" in v4 and "parking fees" in v4 and "A question about one item never gets a partial answer" in v4
    assert 'Never begin with "Yes" or "No" unless the context states that fact' in v4 and "different topic" in v4


def test_answer_v4_adds_the_three_new_rules():
    text = load_prompt_text(4)
    assert 'Never mention "the context", "the entry", "the knowledge base" or "the policy says"' in text
    assert "steps, deadlines, contacts and conditions from that same entry that bear on the employee's situation" in text
    assert "Leave out anything that is unrelated to the question" in text  # no padding with other material
    assert "the entries you took a fact from" in text and "do not cite an entry you did not use" in text
    assert "Once the entry states the item asked about" in text  # extras only after the asked item is found (gap-006)


def load_prompt_text(version: int) -> str:
    from smartdesk_ai.agent.prompt_loader import PROMPTS_DIR

    return (PROMPTS_DIR / f"answer_v{version}.md").read_text()
