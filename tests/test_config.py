"""T0.3: typed config.yaml loader (FR-H16, FR-H17, FR-H8, FR-H13, FR-H5, FR-F4).

Values asserted here come from SPEC section 2 (model reference) and the FR-H17 table.
"""
import copy
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml
from pydantic import BaseModel

from smartdesk_ai.config.settings import (
    REQUIRED_TASKS,
    ConfigError,
    Settings,
    load_settings,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.yaml"


@pytest.fixture(scope="module")
def raw() -> dict:
    return yaml.safe_load(CONFIG.read_text())


@pytest.fixture()
def settings() -> Settings:
    return load_settings(CONFIG)


def write(tmp_path: Path, data) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data) if not isinstance(data, str) else data)
    return path


# --- values from SPEC -------------------------------------------------------

def test_model_ids_windows_and_prices_match_spec(settings):
    main, light, emb = settings.models.main, settings.models.light, settings.models.embedding
    assert (main.id, main.context_window, main.max_output) == ("gpt-5.4-mini", 400_000, 128_000)
    assert (main.price.input, main.price.cached_input, main.price.output) == (0.75, 0.075, 4.50)
    assert (light.id, light.context_window, light.max_output) == ("gpt-4o-mini", 128_000, 16_384)
    assert (light.price.input, light.price.cached_input, light.price.output) == (0.15, 0.075, 0.60)
    assert (emb.id, emb.price.input) == ("text-embedding-3-small", 0.02)
    assert main.reasoning_effort == "none"
    assert main.id != light.id  # right-sizing comparison needs two different models


def test_tier_map_matches_fr_h17(settings):
    tiers = {task: tier for task, tier in settings.tiers.items()}
    main = {"ORCHESTRATOR", "WORKER_ANSWER", "SYNTHESIZER", "GROUNDING_JUDGE"}
    light = {"TOOL_SELECT", "TICKET_DRAFT", "INJECTION_CLASSIFIER", "COMPACTION",
             "CONFIRM_PARSE", "CHITCHAT"}
    assert set(tiers) == main | light == set(REQUIRED_TASKS)
    assert all(tiers[t] == "main" for t in main)
    assert all(tiers[t] == "light" for t in light)


def test_budget_rate_limit_compaction_and_input(settings):
    assert settings.budget.timezone == "America/New_York"
    ZoneInfo(settings.budget.timezone)
    assert (settings.budget.cutoff_1_usd, settings.budget.cutoff_2_usd) == (5.0, 10.0)
    assert settings.budget.eval_bypass is False
    rl = settings.rate_limits
    assert (rl.window_seconds, rl.questions, rl.ticket_creations, rl.ticket_lookups) == (3600, 100, 40, 40)
    assert rl.bypass is False
    assert settings.compaction.trigger_fraction == 0.75
    assert settings.input.max_chars == 4000


def test_thresholds_present_and_in_unit_range(settings):
    t = settings.thresholds
    for value in (t.escalation, t.routing_confidence,
                  t.semantic_cache_cutoff, t.injection_heuristic):
        assert 0.0 < value < 1.0
    assert t.semantic_cache_cutoff >= 0.8  # SPEC section 4: set from evidence, 0.10 above the highest different-answer pair (cache_cutoff.json)


def test_retrieval_and_chunking_match_spec(settings):
    r, c = settings.retrieval, settings.chunking
    assert (r.k_each, r.rrf_k, r.top_n) == (20, 60, 5)
    assert (c.target_min_tokens, c.target_max_tokens) == (200, 500)
    assert (c.split_above_tokens, c.overlap_pct) == (600, 15)


def test_prompt_versions_are_positive_ints(settings):
    assert settings.prompts
    assert all(isinstance(v, int) and v >= 1 for v in settings.prompts.values())


def test_provisional_thresholds_are_marked_in_the_file():
    lines = CONFIG.read_text().splitlines()
    for key in ("routing_confidence", "injection_heuristic"):
        (line,) = [l for l in lines if re.match(rf"\s+{key}:", l)]
        assert "provisional" in line, f"{key} should be marked provisional until its task tunes it"
    (line,) = [l for l in lines if re.match(r"\s+escalation:", l)]  # tuned end to end by the live sweep
    assert "final" in line and "gap_sweep.json" in line
    (line,) = [l for l in lines if re.match(r"\s+semantic_cache_cutoff:", l)]  # set from the cached-embedding evidence
    assert "final" in line and "cache_cutoff.json" in line
    assert not any(re.match(r"\s+fallback:", l) for l in lines)  # SPEC v3.1: one escalation threshold


# --- failure modes ----------------------------------------------------------

def test_missing_key_fails_naming_the_key(tmp_path, raw):
    bad = copy.deepcopy(raw)
    del bad["thresholds"]["escalation"]
    with pytest.raises(ConfigError, match=r"thresholds\.escalation"):
        load_settings(write(tmp_path, bad))


def test_mistyped_value_fails_naming_the_key(tmp_path, raw):
    bad = copy.deepcopy(raw)
    bad["rate_limits"]["questions"] = "lots"
    with pytest.raises(ConfigError, match=r"rate_limits\.questions"):
        load_settings(write(tmp_path, bad))


def test_unknown_key_is_rejected_so_typos_surface(tmp_path, raw):
    bad = copy.deepcopy(raw)
    bad["thresholds"]["escalaton"] = 0.5
    with pytest.raises(ConfigError, match="escalaton"):
        load_settings(write(tmp_path, bad))


def test_tier_map_missing_task_fails(tmp_path, raw):
    bad = copy.deepcopy(raw)
    del bad["tiers"]["CHITCHAT"]
    with pytest.raises(ConfigError, match="CHITCHAT"):
        load_settings(write(tmp_path, bad))


def test_tier_map_unknown_task_or_tier_fails(tmp_path, raw):
    bad = copy.deepcopy(raw)
    bad["tiers"]["MYSTERY"] = "light"
    with pytest.raises(ConfigError, match="MYSTERY"):
        load_settings(write(tmp_path, bad))
    bad = copy.deepcopy(raw)
    bad["tiers"]["CHITCHAT"] = "huge"
    with pytest.raises(ConfigError, match=r"tiers\.CHITCHAT"):
        load_settings(write(tmp_path, bad))


def test_retrieval_values_must_be_positive(tmp_path, raw):
    bad = copy.deepcopy(raw)
    bad["retrieval"]["k_each"] = 0
    with pytest.raises(ConfigError, match=r"retrieval\.k_each"):
        load_settings(write(tmp_path, bad))


def test_top_n_cannot_exceed_candidates(tmp_path, raw):
    bad = copy.deepcopy(raw)
    bad["retrieval"]["top_n"] = bad["retrieval"]["k_each"] * 2 + 1
    with pytest.raises(ConfigError, match=r"retrieval\.top_n"):
        load_settings(write(tmp_path, bad))


def test_chunk_sizes_must_be_ordered(tmp_path, raw):
    bad = copy.deepcopy(raw)
    bad["chunking"]["target_max_tokens"] = bad["chunking"]["target_min_tokens"]
    with pytest.raises(ConfigError, match=r"chunking\.target_max_tokens"):
        load_settings(write(tmp_path, bad))
    bad = copy.deepcopy(raw)
    bad["chunking"]["split_above_tokens"] = bad["chunking"]["target_max_tokens"] - 1
    with pytest.raises(ConfigError, match=r"chunking\.split_above_tokens"):
        load_settings(write(tmp_path, bad))


@pytest.mark.parametrize("pct", [-1, 100])
def test_overlap_pct_must_be_in_range(tmp_path, raw, pct):
    bad = copy.deepcopy(raw)
    bad["chunking"]["overlap_pct"] = pct
    with pytest.raises(ConfigError, match=r"chunking\.overlap_pct"):
        load_settings(write(tmp_path, bad))


def test_cutoff_2_must_exceed_cutoff_1(tmp_path, raw):
    bad = copy.deepcopy(raw)
    bad["budget"]["cutoff_2_usd"] = bad["budget"]["cutoff_1_usd"]
    with pytest.raises(ConfigError, match="cutoff_2_usd"):
        load_settings(write(tmp_path, bad))


def test_bad_timezone_fails(tmp_path, raw):
    bad = copy.deepcopy(raw)
    bad["budget"]["timezone"] = "Mars/Olympus"
    with pytest.raises(ConfigError, match=r"budget\.timezone"):
        load_settings(write(tmp_path, bad))


def test_threshold_out_of_range_fails(tmp_path, raw):
    bad = copy.deepcopy(raw)
    bad["thresholds"]["semantic_cache_cutoff"] = 1.5
    with pytest.raises(ConfigError, match=r"thresholds\.semantic_cache_cutoff"):
        load_settings(write(tmp_path, bad))


def test_missing_empty_or_non_mapping_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.yaml")
    with pytest.raises(ConfigError, match="mapping"):
        load_settings(write(tmp_path, ""))
    with pytest.raises(ConfigError, match="mapping"):
        load_settings(write(tmp_path, "- a\n- b\n"))


# --- no hardcoding ----------------------------------------------------------

def _all_models(model_cls, seen=None):
    seen = seen or set()
    if model_cls in seen:
        return
    seen.add(model_cls)
    yield model_cls
    for field in model_cls.model_fields.values():
        for arg in [field.annotation, *getattr(field.annotation, "__args__", ())]:
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                yield from _all_models(arg, seen)


def test_no_field_has_a_code_default():
    """Every value must come from config.yaml, so nothing is silently hardcoded."""
    for model in _all_models(Settings):
        for name, field in model.model_fields.items():
            assert field.is_required(), f"{model.__name__}.{name} has a code default"


def test_no_model_id_hardcoded_outside_config():
    ids = ["gpt-5.4-mini", "gpt-4o-mini", "text-embedding-3-small"]
    offenders = []
    for path in (ROOT / "smartdesk_ai").rglob("*.py"):
        text = path.read_text()
        offenders += [f"{path.relative_to(ROOT)}: {i}" for i in ids if i in text]
    assert offenders == []
