"""Task-to-tier resolution (FR-H17). The map lives in config.yaml; budget level overrides it (FR-H8)."""
from smartdesk_ai.config.settings import Settings, Tier


def resolve_tier(task: str, settings: Settings, budget_level: int = 0) -> Tier:
    """The tier a task runs on. At budget level 1 or higher every task moves to Light-Model."""
    if task not in settings.tiers:
        raise ValueError(f"unknown LLM task {task!r}; known: {sorted(settings.tiers)}")
    return "light" if budget_level >= 1 else settings.tiers[task]
