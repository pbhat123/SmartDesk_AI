"""T6.1: `python -m evals.run` writes a run record with the prompt versions and model IDs (FR-H11, FR-H14),
so results compare across versions. Mock mode only."""
import json
import re

import pytest

from evals.run import main
from evals.run_record import build_record, write_record
from smartdesk_ai.agent.prompt_loader import PROMPTS_DIR


def test_the_record_holds_prompt_versions_and_model_ids(settings):
    record = build_record(settings, mock=True, steps=["retrieval"])
    assert record.models == {
        "main": settings.models.main.id, "light": settings.models.light.id,
        "embedding": settings.models.embedding.id, "reranker": settings.models.reranker.id,
    }
    assert {name: p.version for name, p in record.prompts.items()} == settings.prompts
    assert record.tier_map == settings.tiers
    assert record.mock is True and record.steps == ["retrieval"]


def test_each_prompt_names_its_file_or_none_when_the_file_is_not_written_yet(settings):
    record = build_record(settings, mock=True, steps=[])
    for name, prompt in record.prompts.items():
        expected = f"{name}_v{prompt.version}.md"
        assert prompt.file == (expected if (PROMPTS_DIR / expected).exists() else None)
    assert record.prompts["answer"].file == f"answer_v{settings.prompts['answer']}.md"


def test_the_record_carries_the_knobs_that_change_results(settings):
    record = build_record(settings, mock=False, steps=[])
    assert record.thresholds == settings.thresholds.model_dump()
    assert record.retrieval["strategy"] == settings.retrieval.strategy
    assert record.chunking["split_above_tokens"] == settings.chunking.split_above_tokens


def test_run_ids_are_unique_and_sortable(settings):
    a, b = build_record(settings, mock=True, steps=[]), build_record(settings, mock=True, steps=[])
    assert a.run_id != b.run_id and re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{6}", a.run_id)


def test_a_changed_prompt_version_changes_the_record(settings):
    bumped = settings.model_copy(update={"prompts": {**settings.prompts, "answer": settings.prompts["answer"] + 1}})
    assert build_record(bumped, mock=True, steps=[]).prompts["answer"].version == settings.prompts["answer"] + 1


def test_a_mock_run_writes_the_record_next_to_its_results(tmp_path):
    assert main(["--mock", "--only", "retrieval", "--out-dir", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "run_record.json").read_text())
    assert record["mock"] is True and record["steps"] == ["retrieval"]
    assert record["models"]["main"] and record["prompts"]["answer"]["version"] >= 1
    assert record["result_files"] == ["retrieval.json"]
    assert json.loads((tmp_path / "retrieval.json").read_text())["run_id"] == record["run_id"]


def test_a_second_run_keeps_the_history(tmp_path):
    main(["--mock", "--only", "retrieval", "--out-dir", str(tmp_path)])
    main(["--mock", "--only", "retrieval", "--out-dir", str(tmp_path)])
    history = sorted((tmp_path / "runs").glob("*.json"))
    assert len(history) == 2 and (tmp_path / "run_record.json").exists()


def test_without_an_out_dir_nothing_is_written_to_the_repo(capsys):
    from evals.harness import RESULTS_DIR

    before = sorted(p.name for p in RESULTS_DIR.glob("*"))
    assert main(["--mock", "--only", "retrieval"]) == 0
    assert sorted(p.name for p in RESULTS_DIR.glob("*")) == before
    assert "run_id" in capsys.readouterr().out


def test_write_record_creates_the_folder(settings, tmp_path):
    path = write_record(build_record(settings, mock=True, steps=[]), tmp_path / "new" / "dir", [])
    assert path.exists() and path.name == "run_record.json"
