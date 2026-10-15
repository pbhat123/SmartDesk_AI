"""T0.4: versioned prompt loader (FR-H14).

Prompts are files named <name>_v<N>.md; the active version per prompt comes from
config. Tests use temp prompt directories and a temp config.
"""
import copy
from pathlib import Path

import pytest
import yaml

from smartdesk_ai.agent.prompt_loader import (
    LoadedPrompt,
    PromptError,
    available_versions,
    load_prompt,
)
from smartdesk_ai.config.settings import load_settings

ROOT = Path(__file__).resolve().parent.parent
RAW = yaml.safe_load((ROOT / "config.yaml").read_text())


def settings_with(tmp_path: Path, **prompt_versions: int):
    data = copy.deepcopy(RAW)
    data["prompts"].update(prompt_versions)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_settings(path)


@pytest.fixture()
def prompts_dir(tmp_path) -> Path:
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "answer_v1.md").write_text("answer prompt, first version\n")
    (d / "answer_v2.md").write_text("answer prompt, second version\n")
    (d / "answer_v10.md").write_text("answer prompt, tenth version\n")
    (d / "chitchat_v1.md").write_text("chitchat prompt\n")
    return d


def test_returns_text_and_version_for_configured_version(tmp_path, prompts_dir):
    s = settings_with(tmp_path, answer=2)
    loaded = load_prompt("answer", s, prompts_dir)
    assert isinstance(loaded, LoadedPrompt)
    assert (loaded.name, loaded.version) == ("answer", 2)
    assert loaded.text == "answer prompt, second version\n"


def test_changing_config_version_changes_text_and_reported_version(tmp_path, prompts_dir):
    one = load_prompt("answer", settings_with(tmp_path, answer=1), prompts_dir)
    two = load_prompt("answer", settings_with(tmp_path, answer=2), prompts_dir)
    assert (one.version, two.version) == (1, 2)
    assert one.text != two.text


def test_version_ten_is_not_confused_with_version_one(tmp_path, prompts_dir):
    loaded = load_prompt("answer", settings_with(tmp_path, answer=10), prompts_dir)
    assert loaded.version == 10 and "tenth" in loaded.text


def test_other_prompts_are_unaffected_by_a_version_change(tmp_path, prompts_dir):
    loaded = load_prompt("chitchat", settings_with(tmp_path, answer=2), prompts_dir)
    assert (loaded.version, loaded.text) == (1, "chitchat prompt\n")


def test_unknown_version_raises_and_lists_what_exists(tmp_path, prompts_dir):
    with pytest.raises(PromptError, match=r"answer.*v3.*\[1, 2, 10\]"):
        load_prompt("answer", settings_with(tmp_path, answer=3), prompts_dir)


def test_prompt_not_in_config_raises(tmp_path, prompts_dir):
    with pytest.raises(PromptError, match="not configured"):
        load_prompt("mystery", settings_with(tmp_path), prompts_dir)


def test_configured_prompt_without_any_file_raises(tmp_path, prompts_dir):
    with pytest.raises(PromptError, match="no files"):
        load_prompt("synthesizer", settings_with(tmp_path), prompts_dir)


def test_empty_prompt_file_raises(tmp_path, prompts_dir):
    (prompts_dir / "answer_v4.md").write_text("  \n")
    with pytest.raises(PromptError, match="empty"):
        load_prompt("answer", settings_with(tmp_path, answer=4), prompts_dir)


@pytest.mark.parametrize("name", ["../secret", "a/b", "Answer", "", "answer v1"])
def test_unsafe_names_are_rejected_before_touching_the_disk(tmp_path, prompts_dir, name):
    with pytest.raises(PromptError, match="invalid prompt name"):
        load_prompt(name, settings_with(tmp_path), prompts_dir)


def test_available_versions_sorted_numerically(prompts_dir):
    assert available_versions("answer", prompts_dir) == [1, 2, 10]
    assert available_versions("chitchat", prompts_dir) == [1]
    assert available_versions("nothing", prompts_dir) == []


def test_available_versions_ignores_lookalike_files(prompts_dir):
    (prompts_dir / "answer_v3.txt").write_text("x")
    (prompts_dir / "answer_vX.md").write_text("x")
    (prompts_dir / "my_answer_v5.md").write_text("x")
    (prompts_dir / "answer_v2.md.bak").write_text("x")
    assert available_versions("answer", prompts_dir) == [1, 2, 10]


def test_default_prompts_dir_is_the_agent_prompts_package_dir():
    from smartdesk_ai.agent.prompt_loader import PROMPTS_DIR

    assert PROMPTS_DIR == ROOT / "smartdesk_ai" / "agent" / "prompts"
    assert PROMPTS_DIR.is_dir()
