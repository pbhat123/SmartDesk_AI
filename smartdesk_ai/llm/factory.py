"""`get_llm(task)`: the only way code obtains a chat model (FR-H17, FR-H16).

The model ID always comes from config.yaml. The builder is injectable so tests use a mock
and never touch the network; the default builder creates the real OpenAI chat model.
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from smartdesk_ai.config.env import load_env
from smartdesk_ai.config.settings import ChatModel, Settings, Tier
from smartdesk_ai.llm.tiers import resolve_tier

ModelBuilder = Callable[[str, ChatModel], Any]


@dataclass(frozen=True)
class LLMHandle:
    task: str
    tier: Tier
    model_id: str
    model: Any  # anything with .invoke(messages), e.g. a LangChain chat model


def openai_builder(model_id: str, model_cfg: ChatModel) -> Any:
    """The real chat model. Calling it later spends API budget; building it does not."""
    env = load_env()
    if env.openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not set (see .env.example)")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model_id, api_key=env.openai_api_key)


def get_llm_for_tier(
    task: str, tier: Tier, settings: Settings, *, builder: ModelBuilder = openai_builder
) -> LLMHandle:
    """A handle on an explicit tier; used by outage fallback (FR-F2), which overrides the map."""
    model_cfg = getattr(settings.models, tier)
    return LLMHandle(task, tier, model_cfg.id, builder(model_cfg.id, model_cfg))


def get_llm(
    task: str,
    settings: Settings,
    *,
    budget_level: int = 0,
    builder: ModelBuilder = openai_builder,
) -> LLMHandle:
    return get_llm_for_tier(task, resolve_tier(task, settings, budget_level), settings, builder=builder)
