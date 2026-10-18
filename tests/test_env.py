"""T0.2: secrets-via-env loader and .env.example (FR-F3, FR-F5, FR-F6).

Every test passes its own environ mapping and a temp dotenv path, so the real
.env in the repo is never read.
"""
from pathlib import Path

import pytest
from pydantic import ValidationError

from smartdesk_ai.config.env import REQUIRED_KEYS, OPTIONAL_KEYS, load_env

ROOT = Path(__file__).resolve().parent.parent
NO_FILE = Path("/nonexistent/.env")


def test_loads_with_all_keys_absent():
    env = load_env(environ={}, dotenv_path=NO_FILE)
    assert env.openai_api_key is None
    assert env.github_token is None
    assert env.tickets_repo is None
    assert env.redis_url is None
    assert not env.langsmith_enabled
    assert not env.langfuse_enabled
    assert env.missing_required() == ["OPENAI_API_KEY", "GITHUB_TOKEN", "TICKETS_REPO"]


def test_reads_values_from_environ():
    env = load_env(
        environ={
            "OPENAI_API_KEY": "k1",
            "GITHUB_TOKEN": "t1",
            "TICKETS_REPO": "acme/tickets",
            "REDIS_URL": "redis://localhost:6379/0",
            "LANGSMITH_API_KEY": "ls",
            "LANGFUSE_PUBLIC_KEY": "pk",
            "LANGFUSE_SECRET_KEY": "sk",
        },
        dotenv_path=NO_FILE,
    )
    assert env.openai_api_key.get_secret_value() == "k1"
    assert env.github_token.get_secret_value() == "t1"
    assert env.tickets_repo == "acme/tickets"
    assert env.redis_url == "redis://localhost:6379/0"
    assert env.langsmith_enabled
    assert env.langfuse_enabled
    assert env.missing_required() == []


def test_langfuse_needs_both_keys():
    env = load_env(environ={"LANGFUSE_PUBLIC_KEY": "pk"}, dotenv_path=NO_FILE)
    assert not env.langfuse_enabled


def test_secrets_do_not_appear_in_repr_or_str():
    env = load_env(
        environ={"OPENAI_API_KEY": "sk-supersecret", "GITHUB_TOKEN": "ghp_supersecret"},
        dotenv_path=NO_FILE,
    )
    text = repr(env) + str(env) + env.model_dump_json()
    assert "supersecret" not in text


def test_blank_and_placeholder_values_count_as_absent():
    env = load_env(
        environ={
            "OPENAI_API_KEY": "",
            "GITHUB_TOKEN": "your-github-token-here",
            "REDIS_URL": "   ",
        },
        dotenv_path=NO_FILE,
    )
    assert env.openai_api_key is None
    assert env.github_token is None
    assert env.redis_url is None


def test_dotenv_file_is_read_and_environ_wins(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("OPENAI_API_KEY=from-file\nTICKETS_REPO=acme/from-file\n")
    env = load_env(environ={"OPENAI_API_KEY": "from-environ"}, dotenv_path=dotenv)
    assert env.openai_api_key.get_secret_value() == "from-environ"
    assert env.tickets_repo == "acme/from-file"


def test_loading_does_not_mutate_process_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("TICKETS_REPO", raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text("TICKETS_REPO=acme/tickets\n")
    load_env(environ={}, dotenv_path=dotenv)
    import os

    assert "TICKETS_REPO" not in os.environ


@pytest.mark.parametrize("bad", ["notarepo", "a/b/c", "/repo", "owner/"])
def test_bad_tickets_repo_fails_with_clear_error(bad):
    with pytest.raises(ValidationError, match="TICKETS_REPO"):
        load_env(environ={"TICKETS_REPO": bad}, dotenv_path=NO_FILE)


def _parse_env_example():
    pairs = {}
    for line in (ROOT / ".env.example").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        pairs[key.strip()] = value.strip()
    return pairs


def test_env_example_lists_all_keys():
    pairs = _parse_env_example()
    for key in (*REQUIRED_KEYS, *OPTIONAL_KEYS):
        assert key in pairs, f"{key} missing from .env.example"
    for key in ("OPENAI_API_KEY", "GITHUB_TOKEN", "TICKETS_REPO"):
        assert key in REQUIRED_KEYS
    for key in ("LANGSMITH_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "REDIS_URL"):
        assert key in OPTIONAL_KEYS


def test_env_example_has_only_placeholders():
    for key, value in _parse_env_example().items():
        assert not value.startswith(("sk-", "ghp_", "github_pat_", "lsv2_", "pk-lf-", "sk-lf-")), key
        assert value == "" or (value.startswith("your-") and value.endswith("-here")), (
            f"{key} must be blank or a your-...-here placeholder"
        )


def test_env_example_loads_as_all_absent():
    env = load_env(environ={}, dotenv_path=ROOT / ".env.example")
    assert env.missing_required() == ["OPENAI_API_KEY", "GITHUB_TOKEN", "TICKETS_REPO"]
    assert not env.langsmith_enabled and not env.langfuse_enabled and env.redis_url is None


def test_dotenv_is_gitignored():
    lines = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in lines
    assert ".env.example" not in lines


def test_langfuse_host_is_optional_and_read_from_env():
    assert load_env(environ={}, dotenv_path=NO_FILE).langfuse_host is None
    env = load_env(environ={"LANGFUSE_HOST": " https://us.cloud.langfuse.com "}, dotenv_path=NO_FILE)
    assert env.langfuse_host == "https://us.cloud.langfuse.com"


def test_langfuse_host_must_be_an_http_url():
    with pytest.raises(ValueError, match="LANGFUSE_HOST"):
        load_env(environ={"LANGFUSE_HOST": "us.cloud.langfuse.com"}, dotenv_path=NO_FILE)
