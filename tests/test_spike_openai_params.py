"""T0.9 spike: OpenAI parameter check (FR-H16, FR-H17). Mock client only: this test
never touches the network or spends money. The real run is a manual, approved step.
"""
import json
from pathlib import Path
from unittest import mock

import pytest
from openai.types import CompletionUsage
from openai.types.completion_usage import CompletionTokensDetails, PromptTokensDetails

from smartdesk_ai.config.settings import load_settings
from spikes import openai_param_check as spike

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def settings():
    return load_settings(ROOT / "config.yaml")


def chat_response(model, content="ok", finish="stop", prompt=20, completion=3, cached=0, reasoning=0):
    response = mock.MagicMock()
    response.model = model
    response.choices = [mock.MagicMock(finish_reason=finish)]
    response.choices[0].message.content = content
    response.usage = CompletionUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached),
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=reasoning),
    )
    return response


def embedding_response(model, prompt=5):
    response = mock.MagicMock()
    response.model = model
    response.data = [mock.MagicMock(embedding=[0.1] * 8)]
    response.usage = mock.MagicMock(prompt_tokens=prompt, total_tokens=prompt)
    return response


@pytest.fixture()
def client(settings):
    c = mock.MagicMock()
    c.chat.completions.create.side_effect = [
        chat_response("gpt-5.4-mini-2026-03-17"),
        chat_response("gpt-4o-mini-2024-07-18"),
    ]
    c.embeddings.create.return_value = embedding_response("text-embedding-3-small")
    return c


# --- what is sent -----------------------------------------------------------

def test_main_model_call_uses_configured_id_reasoning_effort_and_max_completion_tokens(client, settings):
    spike.run_checks(client, settings)
    kwargs = client.chat.completions.create.call_args_list[0].kwargs
    assert kwargs["model"] == settings.models.main.id
    assert kwargs["reasoning_effort"] == settings.models.main.reasoning_effort == "none"
    assert kwargs["max_completion_tokens"] == 16
    assert "max_tokens" not in kwargs
    assert kwargs["messages"] == [{"role": "user", "content": spike.PROMPT}]


def test_light_model_call_uses_max_tokens_and_no_reasoning_param(client, settings):
    spike.run_checks(client, settings)
    kwargs = client.chat.completions.create.call_args_list[1].kwargs
    assert kwargs["model"] == settings.models.light.id
    assert kwargs["max_tokens"] == 16
    assert "reasoning_effort" not in kwargs and "max_completion_tokens" not in kwargs


def test_embedding_call_uses_configured_id_and_same_string(client, settings):
    spike.run_checks(client, settings)
    kwargs = client.embeddings.create.call_args.kwargs
    assert kwargs == {"model": settings.models.embedding.id, "input": spike.PROMPT}


def test_exactly_one_call_each_and_prompt_is_the_fixed_constant(client, settings):
    spike.run_checks(client, settings)
    assert client.chat.completions.create.call_count == 2
    assert client.embeddings.create.call_count == 1
    assert spike.PROMPT == "Reply with the single word: ok"


def test_client_is_built_without_sdk_retries(monkeypatch):
    with mock.patch.object(spike, "OpenAI") as ctor:
        spike.make_client("sk-not-a-real-key")
    assert ctor.call_args.kwargs["max_retries"] == 0


# --- what is recorded -------------------------------------------------------

def test_usage_fields_including_cached_and_reasoning_tokens_are_recorded(settings):
    c = mock.MagicMock()
    c.chat.completions.create.side_effect = [
        chat_response("gpt-5.4-mini", prompt=100, completion=12, cached=40, reasoning=7),
        chat_response("gpt-4o-mini", prompt=20, completion=3),
    ]
    c.embeddings.create.return_value = embedding_response("text-embedding-3-small")
    main = spike.run_checks(c, settings)[0]
    assert main.ok and main.model_returned == "gpt-5.4-mini"
    assert main.usage["prompt_tokens"] == 100
    assert main.usage["prompt_tokens_details"]["cached_tokens"] == 40
    assert main.usage["completion_tokens_details"]["reasoning_tokens"] == 7


def test_cost_bills_cached_input_at_the_cached_rate_and_reasoning_as_output(settings):
    c = mock.MagicMock()
    c.chat.completions.create.side_effect = [
        chat_response("gpt-5.4-mini", prompt=100, completion=12, cached=40, reasoning=7),
        chat_response("gpt-4o-mini", prompt=20, completion=3),
    ]
    c.embeddings.create.return_value = embedding_response("text-embedding-3-small", prompt=5)
    main, light, emb = spike.run_checks(c, settings)
    p = settings.models.main.price
    assert main.cost_usd == pytest.approx((60 * p.input + 40 * p.cached_input + 12 * p.output) / 1e6)
    q = settings.models.light.price
    assert light.cost_usd == pytest.approx((20 * q.input + 3 * q.output) / 1e6)
    assert emb.cost_usd == pytest.approx(5 * settings.models.embedding.price.input / 1e6)


def test_empty_reply_from_the_cap_is_a_finding_not_a_failure(settings):
    c = mock.MagicMock()
    c.chat.completions.create.side_effect = [
        chat_response("gpt-5.4-mini", content="", finish="length", completion=16),
        chat_response("gpt-4o-mini"),
    ]
    c.embeddings.create.return_value = embedding_response("text-embedding-3-small")
    main = spike.run_checks(c, settings)[0]
    assert main.ok is True
    assert "empty_reply" in main.findings and main.finish_reason == "length"


def test_returned_model_differing_from_the_configured_id_is_noted(client, settings):
    results = spike.run_checks(client, settings)
    assert results[0].model_returned == "gpt-5.4-mini-2026-03-17"
    assert results[0].findings == []  # a dated snapshot of the same model is expected


def test_unexpected_model_name_is_flagged(settings):
    c = mock.MagicMock()
    c.chat.completions.create.side_effect = [chat_response("something-else"), chat_response("gpt-4o-mini")]
    c.embeddings.create.return_value = embedding_response("text-embedding-3-small")
    assert "model_mismatch" in spike.run_checks(c, settings)[0].findings


# --- failure handling -------------------------------------------------------

def test_a_failing_call_is_recorded_and_stops_further_calls(settings):
    c = mock.MagicMock()
    c.chat.completions.create.side_effect = RuntimeError("Unsupported parameter: reasoning_effort")
    results = spike.run_checks(c, settings)
    assert results[0].ok is False
    assert "Unsupported parameter" in results[0].error
    assert [r.status for r in results] == ["failed", "skipped", "skipped"]
    assert c.chat.completions.create.call_count == 1  # no retry, nothing after it
    assert c.embeddings.create.call_count == 0


# --- output and entry point -------------------------------------------------

def test_report_never_contains_the_api_key(client, settings):
    report = spike.render_report(spike.run_checks(client, settings))
    assert "sk-secret-key-123" not in report
    json.loads(report)  # valid JSON


def test_main_without_a_key_makes_no_call_and_exits_nonzero(capsys):
    with mock.patch.object(spike, "OpenAI") as ctor:
        code = spike.main(environ={}, dotenv_path=Path("/nonexistent/.env"))
    assert code != 0
    ctor.assert_not_called()
    assert "OPENAI_API_KEY" in capsys.readouterr().err
