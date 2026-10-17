"""T4.11 follow-up: `python -m smartdesk_ai.agent` runs the real graph, selected by `modes.llm`
(mock by default), so --debug works end to end with no keys and no network (FR-A12)."""
from pathlib import Path

import pytest
import yaml

from smartdesk_ai.agent.__main__ import main
from smartdesk_ai.agent.factory import LiveModeUnavailable, build_responder
from smartdesk_ai.config.settings import load_settings
from smartdesk_ai.llm.mock import MockAnswerModel
from smartdesk_ai.ops.health import HealthRegistry

NO_FILE = Path("/nonexistent/.env")
ROOT = Path(__file__).resolve().parent.parent


def run_cli(argv, lines, tmp_path, config_path=None):
    feed = iter(lines)
    out: list[str] = []
    kwargs = {"config_path": config_path} if config_path else {}
    code = main(argv, input_fn=lambda _: next(feed), output=out.append, environ={}, dotenv_path=NO_FILE,
                data_dir=tmp_path, registry=HealthRegistry(), **kwargs)
    assert code == 0
    return out


def test_modes_llm_defaults_to_mock_in_config(settings):
    assert settings.modes.llm == "mock" and settings.modes.tickets == "mock"


def test_a_bad_modes_llm_value_is_rejected(tmp_path):
    data = yaml.safe_load((ROOT / "config.yaml").read_text())
    data["modes"]["llm"] = "maybe"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(Exception, match="llm"):
        load_settings(path)


def test_cli_answers_a_kb_question_end_to_end_in_mock_mode(tmp_path):
    out = run_cli([], ["How do I reset my forgotten password?", "quit"], tmp_path)
    assert any("Sources:" in line for line in out)
    assert not any("starting up" in line for line in out)


def test_debug_shows_chunks_scores_and_the_answer(tmp_path):
    out = run_cli(["--debug"], ["How do I reset my forgotten password?", "quit"], tmp_path)
    assert any(line.startswith("[debug] 1. ") and "score=" in line for line in out)
    assert any(line.startswith("[debug] answer:") for line in out)


def test_a_gap_question_gets_the_i_dont_know_and_a_ticket_offer(tmp_path):
    out = run_cli([], ["zzqx wvvk plorf", "quit"], tmp_path)
    assert any("don't know" in line and "ticket" in line for line in out)


def test_the_same_session_keeps_its_thread_across_turns(tmp_path):
    out = run_cli(["--debug"], ["How do I reset my forgotten password?", "zzqx wvvk plorf", "quit"], tmp_path)
    assert sum(1 for line in out if line.startswith("[debug] answer:")) == 2


def test_the_mock_index_is_separate_from_the_live_index(tmp_path):
    run_cli([], ["hello", "quit"], tmp_path)
    assert (tmp_path / "index_mock").is_dir() and not (tmp_path / "index").exists()


def live_config(tmp_path, tickets="mock"):
    data = yaml.safe_load((ROOT / "config.yaml").read_text())
    data["modes"]["llm"] = "live"
    data["modes"]["tickets"] = tickets
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def run_live(tmp_path, environ, lines=("How do I reset my forgotten password?", "quit"), tickets="mock"):
    feed = iter(lines)
    out: list[str] = []
    code = main([], input_fn=lambda _: next(feed), output=out.append, environ=environ, dotenv_path=NO_FILE,
                data_dir=tmp_path / "data", registry=HealthRegistry(), config_path=live_config(tmp_path, tickets))
    return code, "\n".join(out)


def test_live_mode_without_the_openai_key_exits_with_a_clear_error_and_does_not_run_as_mock(tmp_path):
    code, out = run_live(tmp_path, {})
    assert code == 2
    assert "OPENAI_API_KEY" in out and ".env" in out and "Cannot start in live mode" in out
    assert "Sources:" not in out and "SmartDesk AI. Type" not in out
    assert not (tmp_path / "data" / "index_mock").exists(), "mock mode must not have been started"


def test_live_mode_without_an_index_exits_naming_the_ingest_command(tmp_path):
    code, out = run_live(tmp_path, {"OPENAI_API_KEY": "sk-test-not-real"})
    assert code == 2
    assert "python -m smartdesk_ai.rag.ingest" in out and "index" in out
    assert "Sources:" not in out and not (tmp_path / "data" / "index_mock").exists()


def test_live_tickets_without_a_token_exits_naming_what_is_missing(tmp_path, built_index_dir):
    import shutil

    shutil.copytree(built_index_dir, tmp_path / "data" / "index")  # a (hashing) index is enough to get that far
    code, out = run_live(tmp_path, {"OPENAI_API_KEY": "sk-test-not-real"}, tickets="live")
    assert code == 2 and "GITHUB_TOKEN" in out and "TICKETS_REPO" in out and ".env" in out
    assert "Sources:" not in out


def test_the_error_is_printed_before_any_chat_prompt(tmp_path):
    feed = iter(["hello", "quit"])
    asked = []
    out: list[str] = []
    main([], input_fn=lambda p: asked.append(p) or next(feed), output=out.append, environ={}, dotenv_path=NO_FILE,
         data_dir=tmp_path / "data", registry=HealthRegistry(), config_path=live_config(tmp_path))
    assert asked == []


def test_build_responder_live_raises_a_clear_error_without_the_key(settings, tmp_path):
    live = settings.model_copy(update={"modes": settings.modes.model_copy(update={"llm": "live"})})
    with pytest.raises(LiveModeUnavailable, match="OPENAI_API_KEY"):
        build_responder(live, environ={}, dotenv_path=NO_FILE, data_dir=tmp_path, registry=HealthRegistry())


def test_mock_answer_model_cites_the_top_entry_and_uses_only_its_text():
    from langchain_core.messages import HumanMessage

    reply = MockAnswerModel().invoke([
        HumanMessage("Question: How do I reset my password with the portal?\n\nContext:\n[IT-PWD-01] Password reset\n"
                     "Use the portal. Then log in.\n\n[HR-X-1] Other\nNo.")
    ])
    assert '"IT-PWD-01"' in reply.content and "Use the portal." in reply.content


def test_mock_answer_model_says_insufficient_when_the_question_barely_overlaps():
    from langchain_core.messages import HumanMessage

    reply = MockAnswerModel().invoke([
        HumanMessage("Question: Where can I park my car?\n\nContext:\n[IT-PWD-01] Password reset\nUse the portal.")
    ])
    assert '"insufficient": true' in reply.content
