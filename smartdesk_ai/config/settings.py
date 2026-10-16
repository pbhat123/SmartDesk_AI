"""Typed loader for config.yaml (FR-H16).

Every field is required and the models forbid unknown keys, so a missing or
misspelled key fails at startup with its dotted path. Nothing has a default in
code: model IDs, prices, thresholds and limits live only in config.yaml.
"""
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"

Tier = Literal["main", "light"]
Unit = Annotated[float, Field(gt=0.0, lt=1.0)]
NonNegative = Annotated[float, Field(ge=0.0)]

# Every LLM task that needs a tier (FR-H17, plus CONFIRM_PARSE and CHITCHAT).
REQUIRED_TASKS = (
    "ORCHESTRATOR",
    "WORKER_ANSWER",
    "SYNTHESIZER",
    "GROUNDING_JUDGE",
    "TOOL_SELECT",
    "TICKET_DRAFT",
    "INJECTION_CLASSIFIER",
    "COMPACTION",
    "CONFIRM_PARSE",
    "CHITCHAT",
)


class ConfigError(ValueError):
    """config.yaml is missing, unreadable, or fails validation."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChatPrice(_Strict):
    """USD per 1M tokens."""

    input: NonNegative
    cached_input: NonNegative
    output: NonNegative


class EmbeddingPrice(_Strict):
    input: NonNegative


class ChatModel(_Strict):
    id: str
    context_window: Annotated[int, Field(gt=0)]
    max_output: Annotated[int, Field(gt=0)]
    price: ChatPrice


class MainModel(ChatModel):
    reasoning_effort: str


class EmbeddingModel(_Strict):
    id: str
    price: EmbeddingPrice


class RerankerModel(_Strict):
    id: str  # Hugging Face model id, loaded locally by sentence-transformers


class Models(_Strict):
    main: MainModel
    light: ChatModel
    embedding: EmbeddingModel
    reranker: RerankerModel


class Budget(_Strict):
    timezone: str
    cutoff_1_usd: NonNegative
    cutoff_2_usd: NonNegative  # total daily spend, not the increment
    eval_bypass: bool

    @field_validator("timezone")
    @classmethod
    def _known_zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError(f"unknown IANA timezone {value!r}") from None
        return value

    @field_validator("cutoff_2_usd")
    @classmethod
    def _cutoff_order(cls, value: float, info) -> float:
        first = info.data.get("cutoff_1_usd")
        if first is not None and value <= first:
            raise ValueError("cutoff_2_usd must be greater than cutoff_1_usd")
        return value


class RateLimits(_Strict):
    window_seconds: Annotated[int, Field(gt=0)]
    questions: Annotated[int, Field(gt=0)]
    ticket_creations: Annotated[int, Field(gt=0)]
    ticket_lookups: Annotated[int, Field(gt=0)]
    bypass: bool


class Compaction(_Strict):
    trigger_fraction: Unit


class InputLimits(_Strict):
    max_chars: Annotated[int, Field(gt=0)]


class Thresholds(_Strict):
    escalation: Unit  # top dense cosine similarity; the one escalation threshold (FR-A15)
    routing_confidence: Unit  # FR-D8
    semantic_cache_cutoff: Unit
    injection_heuristic: Unit  # classifier runs only above this (FR-H1)


class Retrieval(_Strict):
    k_each: Annotated[int, Field(gt=0)]  # candidates from each retriever (FR-A6)
    rrf_k: Annotated[int, Field(gt=0)]
    top_n: Annotated[int, Field(gt=0)]  # entries kept after rerank and collapse

    @field_validator("top_n")
    @classmethod
    def _top_n_fits(cls, value: int, info) -> int:
        k_each = info.data.get("k_each")
        if k_each is not None and value > 2 * k_each:
            raise ValueError("top_n cannot exceed the candidates available (2 * k_each)")
        return value


class Chunking(_Strict):
    target_min_tokens: Annotated[int, Field(gt=0)]  # FR-A3
    target_max_tokens: Annotated[int, Field(gt=0)]
    split_above_tokens: Annotated[int, Field(gt=0)]
    overlap_pct: Annotated[int, Field(ge=0, lt=100)]

    @field_validator("target_max_tokens")
    @classmethod
    def _max_above_min(cls, value: int, info) -> int:
        low = info.data.get("target_min_tokens")
        if low is not None and value <= low:
            raise ValueError("target_max_tokens must be greater than target_min_tokens")
        return value

    @field_validator("split_above_tokens")
    @classmethod
    def _split_above_max(cls, value: int, info) -> int:
        high = info.data.get("target_max_tokens")
        if high is not None and value < high:
            raise ValueError("split_above_tokens must be at least target_max_tokens")
        return value


class Settings(_Strict):
    models: Models
    tiers: dict[str, Tier]
    budget: Budget
    rate_limits: RateLimits
    compaction: Compaction
    input: InputLimits
    thresholds: Thresholds
    retrieval: Retrieval
    chunking: Chunking
    prompts: dict[str, Annotated[int, Field(ge=1)]]

    @field_validator("tiers")
    @classmethod
    def _tier_map_complete(cls, value: dict[str, Tier]) -> dict[str, Tier]:
        missing = [t for t in REQUIRED_TASKS if t not in value]
        unknown = [t for t in value if t not in REQUIRED_TASKS]
        if missing or unknown:
            raise ValueError(f"tier map: missing tasks {missing}, unknown tasks {unknown}")
        return value


def _format(error: ValidationError) -> str:
    lines = []
    for item in error.errors():
        path = ".".join(str(part) for part in item["loc"])
        lines.append(f"  {path}: {item['msg']}")
    return "\n".join(lines)


def load_settings(path: Path | str = DEFAULT_CONFIG) -> Settings:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    try:
        return Settings(**data)
    except ValidationError as exc:
        raise ConfigError(f"invalid config in {path}:\n{_format(exc)}") from exc
