"""The record stored with every eval run (FR-H11, FR-H14): which prompt versions, models and settings
produced the numbers, so results compare across versions. Written as `run_record.json` next to the
results, with a copy under `runs/` so history is kept, and its `run_id` is added to each result file."""
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from smartdesk_ai.agent.prompt_loader import PROMPTS_DIR
from smartdesk_ai.config.settings import Settings

ROOT = Path(__file__).resolve().parent.parent


class PromptUsed(BaseModel):
    version: int
    file: str | None  # None while the prompt's file is not written yet


class RunRecord(BaseModel):
    run_id: str
    started_at: str
    mock: bool
    steps: list[str]
    models: dict[str, str]
    tier_map: dict[str, str]
    prompts: dict[str, PromptUsed]
    thresholds: dict[str, float]
    retrieval: dict[str, object]
    chunking: dict[str, object]
    git_commit: str | None = None
    python: str = Field(default_factory=platform.python_version)
    result_files: list[str] = Field(default_factory=list)


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def build_record(settings: Settings, *, mock: bool, steps: list[str]) -> RunRecord:
    now = datetime.now(timezone.utc)
    prompts = {}
    for name, version in settings.prompts.items():
        file = f"{name}_v{version}.md"
        prompts[name] = PromptUsed(version=version, file=file if (PROMPTS_DIR / file).exists() else None)
    models = settings.models
    return RunRecord(
        run_id=f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}", started_at=now.isoformat(), mock=mock, steps=steps,
        models={"main": models.main.id, "light": models.light.id, "embedding": models.embedding.id,
                "reranker": models.reranker.id},
        tier_map=dict(settings.tiers), prompts=prompts, thresholds=settings.thresholds.model_dump(),
        retrieval=settings.retrieval.model_dump(), chunking=settings.chunking.model_dump(), git_commit=_git_commit(),
    )


def write_record(record: RunRecord, out_dir: Path, result_files: list[str]) -> Path:
    """Write the record, tag each result file with the run ID, and keep a copy under `runs/`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    record = record.model_copy(update={"result_files": sorted(result_files)})
    for name in record.result_files:
        path = out_dir / name
        payload = json.loads(path.read_text())
        payload["run_id"] = record.run_id
        path.write_text(json.dumps(payload, indent=2, default=str))
    text = record.model_dump_json(indent=2)
    (out_dir / "runs").mkdir(exist_ok=True)
    (out_dir / "runs" / f"{record.run_id}.json").write_text(text)
    path = out_dir / "run_record.json"
    path.write_text(text)
    return path
