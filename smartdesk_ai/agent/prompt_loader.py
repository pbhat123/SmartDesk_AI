"""Versioned prompt loader (FR-H14).

Prompts are Markdown files named `<name>_v<N>.md` in `agent/prompts/`. The active
version per prompt is set in config.yaml. Edit a prompt by adding a new version
file and bumping the config, never by rewriting an old one, so eval runs stay
comparable across versions.
"""
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from smartdesk_ai.config.settings import Settings

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class PromptError(LookupError):
    """A prompt is not configured, has no such version, or is unusable."""


class LoadedPrompt(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: int
    text: str


def _check_name(name: str) -> None:
    if not _NAME.match(name):
        raise PromptError(f"invalid prompt name {name!r}: use lowercase letters, digits, underscores")


def available_versions(name: str, prompts_dir: Path = PROMPTS_DIR) -> list[int]:
    """Version numbers found on disk for `name`, ascending."""
    _check_name(name)
    pattern = re.compile(rf"^{re.escape(name)}_v(\d+)\.md$")
    found = []
    for path in Path(prompts_dir).iterdir():
        match = pattern.match(path.name)
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


def load_prompt(name: str, settings: Settings, prompts_dir: Path = PROMPTS_DIR) -> LoadedPrompt:
    """Load the version of `name` that config.yaml marks active."""
    _check_name(name)
    if name not in settings.prompts:
        raise PromptError(f"prompt {name!r} is not configured in config.yaml (prompts section)")
    version = settings.prompts[name]
    prompts_dir = Path(prompts_dir)
    path = prompts_dir / f"{name}_v{version}.md"
    if not path.is_file():
        existing = available_versions(name, prompts_dir)
        if not existing:
            raise PromptError(f"prompt {name!r} v{version}: no files for this prompt in {prompts_dir}")
        raise PromptError(
            f"prompt {name!r} v{version} not found in {prompts_dir}; available versions: {existing}"
        )
    text = path.read_text()
    if not text.strip():
        raise PromptError(f"prompt file {path.name} is empty")
    return LoadedPrompt(name=name, version=version, text=text)
