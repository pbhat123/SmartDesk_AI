"""T9.5: the mock-mode load test runs the whole pipeline, reports requests per second and p50/p95, and touches nothing
outside its temp folder (FR-H11). No network, no spend."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "load_test.py"


@pytest.fixture(scope="module")
def load_test():
    spec = importlib.util.spec_from_file_location("load_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_percentile_is_nearest_rank(load_test):
    values = [float(n) for n in range(1, 101)]
    assert load_test.percentile(values, 0.50) == 50.0
    assert load_test.percentile(values, 0.95) == 95.0
    assert load_test.percentile([7.0], 0.95) == 7.0


def test_a_small_run_reports_throughput_and_latency(load_test, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SMARTDESK_LEDGER_PATH", str(tmp_path / "ledger.jsonl"))
    out = tmp_path / "result.json"
    assert load_test.main(["--requests", "12", "--workers", "3", "--out", str(out)]) == 0
    summary = json.loads(out.read_text())
    assert summary["mode"] == "mock" and summary["requests"] == 12 and summary["failed"] == 0
    assert summary["requests_per_second"] > 0
    assert 0 < summary["latency_p50_seconds"] <= summary["latency_p95_seconds"]
    assert "requests/s" in capsys.readouterr().out
