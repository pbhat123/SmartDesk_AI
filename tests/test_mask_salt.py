"""Follow-up to T5.16: the masking salt comes from MASK_SALT (FR-G11). Unset, the code fallback is used and
one warning is logged. The rate limiter hashes emails with the same salt."""
import logging

import pytest

from smartdesk_ai.agent.factory import build_responder
from smartdesk_ai.config.env import load_env
from smartdesk_ai.ops.rate_limit import RateLimiter
from smartdesk_ai.ops.runtime import FALLBACK_SALT, configure_mask_salt, masked_user_id
from smartdesk_ai.tickets import MockTicketClient

EMAIL = "jane.roe@example.com"


@pytest.fixture(autouse=True)
def reset_salt():
    configure_mask_salt(None)
    yield
    configure_mask_salt(None)


def test_env_loads_the_salt_as_a_secret():
    env = load_env(environ={"MASK_SALT": "s3cret-salt"}, dotenv_path=None)
    assert env.mask_salt.get_secret_value() == "s3cret-salt" and "s3cret" not in repr(env)
    assert load_env(environ={}, dotenv_path=None).mask_salt is None
    assert load_env(environ={"MASK_SALT": "  "}, dotenv_path=None).mask_salt is None


def test_the_env_example_lists_the_salt_as_optional_and_blank():
    from pathlib import Path

    lines = (Path(__file__).resolve().parent.parent / ".env.example").read_text().splitlines()
    assert "MASK_SALT=" in lines


def test_a_configured_salt_changes_the_id_and_is_stable():
    fallback = masked_user_id(EMAIL)
    configure_mask_salt("one")
    first = masked_user_id(EMAIL)
    assert first != fallback and first == masked_user_id(EMAIL.upper()) and EMAIL not in first
    configure_mask_salt("two")
    assert masked_user_id(EMAIL) != first


def test_unset_uses_the_fallback_and_warns_once(caplog):
    with caplog.at_level(logging.WARNING):
        a, b = masked_user_id(EMAIL), masked_user_id("other@example.com")
    assert a != b and caplog.text.count("MASK_SALT is not set") == 1
    configure_mask_salt("")  # blank means unset again, and a later start warns again
    with caplog.at_level(logging.WARNING):
        masked_user_id(EMAIL)
    assert caplog.text.count("MASK_SALT is not set") == 2
    assert FALLBACK_SALT


def test_a_set_salt_logs_no_warning(caplog):
    configure_mask_salt("set")
    with caplog.at_level(logging.WARNING):
        masked_user_id(EMAIL)
    assert "MASK_SALT" not in caplog.text


def test_the_rate_limiter_uses_the_same_salt(settings, tmp_path):
    def stored_key(salt):
        configure_mask_salt(salt)
        path = tmp_path / f"{salt}.db"
        RateLimiter(settings.rate_limits, path).hit("ticket_lookups", "s1", EMAIL)
        import sqlite3

        return {k for (k,) in sqlite3.connect(path).execute("SELECT key FROM rate_events")}

    with_salt, without = stored_key("pepper"), stored_key(None)
    assert f"user:{masked_user_id(EMAIL)}" in without and with_salt != without
    configure_mask_salt("pepper")
    assert f"user:{masked_user_id(EMAIL)}" in with_salt


def test_the_app_reads_the_salt_from_the_environment(settings, tmp_path):
    build_responder(settings, environ={"MASK_SALT": "from-env"}, dotenv_path=None, data_dir=tmp_path,
                    client=MockTicketClient())
    configured = masked_user_id(EMAIL)
    configure_mask_salt("from-env")
    assert masked_user_id(EMAIL) == configured
