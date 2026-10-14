"""Secrets via environment variables (FR-F3, FR-F5, FR-F6).

Values come from the process environment, with an optional `.env` file as
fallback. The environment wins over the file. Nothing is written back to
`os.environ`. Secrets are `SecretStr`, so they stay out of repr, str and JSON.
Blank values and `your-...-here` placeholders count as absent, so a copied
`.env.example` behaves like an empty setup. Missing keys never raise here;
callers decide what to do (FR-F7).
"""
import os
import re
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

REQUIRED_KEYS = ("OPENAI_API_KEY", "GITHUB_TOKEN", "TICKETS_REPO")
OPTIONAL_KEYS = (
    "LANGSMITH_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "REDIS_URL",
)
DEFAULT_DOTENV = Path(".env")

_PLACEHOLDER = re.compile(r"^your-.*-here$")
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class EnvSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    openai_api_key: SecretStr | None = None
    github_token: SecretStr | None = None
    tickets_repo: str | None = None
    langsmith_api_key: SecretStr | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    redis_url: str | None = None

    @field_validator("tickets_repo")
    @classmethod
    def _check_repo(cls, value: str | None) -> str | None:
        if value is not None and not _REPO.match(value):
            raise ValueError(f"TICKETS_REPO must look like 'owner/repo', got {value!r}")
        return value

    @property
    def langsmith_enabled(self) -> bool:
        return self.langsmith_api_key is not None

    @property
    def langfuse_enabled(self) -> bool:
        return self.langfuse_public_key is not None and self.langfuse_secret_key is not None

    def missing_required(self) -> list[str]:
        values = {
            "OPENAI_API_KEY": self.openai_api_key,
            "GITHUB_TOKEN": self.github_token,
            "TICKETS_REPO": self.tickets_repo,
        }
        return [key for key in REQUIRED_KEYS if values[key] is None]


def _clean(raw: Mapping[str, str | None]) -> dict[str, str]:
    cleaned = {}
    for key in (*REQUIRED_KEYS, *OPTIONAL_KEYS):
        value = (raw.get(key) or "").strip()
        if value and not _PLACEHOLDER.match(value):
            cleaned[key.lower()] = value
    return cleaned


def load_env(
    environ: Mapping[str, str] | None = None,
    dotenv_path: Path | None = DEFAULT_DOTENV,
) -> EnvSettings:
    """Build settings from `dotenv_path` (if it exists) overlaid by `environ`."""
    merged: dict[str, str | None] = {}
    if dotenv_path is not None and Path(dotenv_path).is_file():
        merged.update(dotenv_values(dotenv_path))
    merged.update(os.environ if environ is None else environ)
    return EnvSettings(**_clean(merged))
