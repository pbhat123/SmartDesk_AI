"""T0.9 spike: verify OpenAI model IDs, parameters and usage fields with three tiny calls.

Throwaway proof, not production code. Run by hand, after approval (it spends a
fraction of a cent):  python -m spikes.openai_param_check

Calls, one each, no retries (the SDK's own retries are turned off):
  1. main model with reasoning_effort and max_completion_tokens
  2. light model with max_tokens
  3. embedding model on the same string
It sends a fixed throwaway prompt only. The API key is never printed.
"""
import json
import re
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from smartdesk_ai.config.env import DEFAULT_DOTENV, load_env
from smartdesk_ai.config.settings import Settings, load_settings

PROMPT = "Reply with the single word: ok"
MAX_OUTPUT_TOKENS = 16
_KEY_LIKE = re.compile(r"sk-[A-Za-z0-9_*.\-]+")


class CallResult(BaseModel):
    name: str
    status: str  # ok | failed | skipped
    model_requested: str
    params: dict[str, Any] = {}
    model_returned: str | None = None
    finish_reason: str | None = None
    reply: str | None = None
    usage: dict[str, Any] = {}
    cost_usd: float | None = None
    latency_s: float | None = None
    error: str | None = None
    findings: list[str] = []

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def make_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, max_retries=0, timeout=30)


def _scrub(text: str, api_key: str | None = None) -> str:
    if api_key:
        text = text.replace(api_key, "[KEY]")
    return _KEY_LIKE.sub("[KEY]", text)


def _chat_cost(usage: Any, price: Any) -> float:
    cached = (usage.prompt_tokens_details.cached_tokens or 0) if usage.prompt_tokens_details else 0
    fresh = usage.prompt_tokens - cached
    # reasoning tokens are already inside completion_tokens and billed as output
    return (fresh * price.input + cached * price.cached_input + usage.completion_tokens * price.output) / 1e6


def _check_chat(client: Any, name: str, model_cfg: Any, params: dict[str, Any]) -> CallResult:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model_cfg.id, messages=[{"role": "user", "content": PROMPT}], **params
    )
    elapsed = time.perf_counter() - started
    choice = response.choices[0]
    reply = choice.message.content or ""
    findings = []
    if not response.model.startswith(model_cfg.id):
        findings.append("model_mismatch")
    if not reply.strip():
        findings.append("empty_reply")
    return CallResult(
        name=name,
        status="ok",
        model_requested=model_cfg.id,
        params=params,
        model_returned=response.model,
        finish_reason=choice.finish_reason,
        reply=reply,
        usage=response.usage.model_dump(),
        cost_usd=_chat_cost(response.usage, model_cfg.price),
        latency_s=round(elapsed, 3),
        findings=findings,
    )


def _check_embedding(client: Any, settings: Settings) -> CallResult:
    cfg = settings.models.embedding
    started = time.perf_counter()
    response = client.embeddings.create(model=cfg.id, input=PROMPT)
    elapsed = time.perf_counter() - started
    findings = [] if response.model.startswith(cfg.id) else ["model_mismatch"]
    return CallResult(
        name="embedding",
        status="ok",
        model_requested=cfg.id,
        params={"dimensions_returned": len(response.data[0].embedding)},
        model_returned=response.model,
        usage={"prompt_tokens": response.usage.prompt_tokens, "total_tokens": response.usage.total_tokens},
        cost_usd=response.usage.prompt_tokens * cfg.price.input / 1e6,
        latency_s=round(elapsed, 3),
        findings=findings,
    )


def run_checks(client: Any, settings: Settings, api_key: str | None = None) -> list[CallResult]:
    """Run the three calls in order. The first failure is recorded and the rest are skipped."""
    main, light = settings.models.main, settings.models.light
    plan: list[tuple[str, str, Callable[[], CallResult]]] = [
        (
            "main_chat",
            main.id,
            lambda: _check_chat(
                client,
                "main_chat",
                main,
                {"reasoning_effort": main.reasoning_effort, "max_completion_tokens": MAX_OUTPUT_TOKENS},
            ),
        ),
        (
            "light_chat",
            light.id,
            lambda: _check_chat(client, "light_chat", light, {"max_tokens": MAX_OUTPUT_TOKENS}),
        ),
        ("embedding", settings.models.embedding.id, lambda: _check_embedding(client, settings)),
    ]
    results: list[CallResult] = []
    failed = False
    for name, model_id, call in plan:
        if failed:
            results.append(CallResult(name=name, status="skipped", model_requested=model_id))
            continue
        try:
            results.append(call())
        except Exception as exc:
            failed = True
            results.append(
                CallResult(
                    name=name,
                    status="failed",
                    model_requested=model_id,
                    error=_scrub(f"{type(exc).__name__}: {exc}", api_key),
                )
            )
    return results


def render_report(results: list[CallResult]) -> str:
    total = sum(r.cost_usd or 0.0 for r in results)
    return json.dumps(
        {"calls": [r.model_dump() for r in results], "total_cost_usd": round(total, 8)}, indent=2
    )


def main(
    environ: Mapping[str, str] | None = None, dotenv_path: Path | None = DEFAULT_DOTENV
) -> int:
    env = load_env(environ=environ, dotenv_path=dotenv_path)
    if env.openai_api_key is None:
        print("OPENAI_API_KEY is not set; no call was made.", file=sys.stderr)
        return 2
    key = env.openai_api_key.get_secret_value()
    results = run_checks(make_client(key), load_settings(), api_key=key)
    print(render_report(results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
