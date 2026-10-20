"""The commit gate must not wait forever: scripts/run_with_limit.py runs a command and kills it after a time limit (macOS has no
`timeout`). Also pins the pytest per-test timeout and the `slow` marker the gate relies on. No network, no model."""
import configparser
import sys
import time
from pathlib import Path

from scripts.run_with_limit import TIMED_OUT, run_with_limit

ROOT = Path(__file__).resolve().parent.parent


def test_a_command_that_finishes_returns_its_own_exit_code():
    assert run_with_limit(5, [sys.executable, "-c", "raise SystemExit(0)"]) == 0
    assert run_with_limit(5, [sys.executable, "-c", "raise SystemExit(3)"]) == 3


def test_a_command_over_the_limit_is_killed_and_reported_as_timed_out():
    started = time.monotonic()
    code = run_with_limit(1, [sys.executable, "-c", "import time; time.sleep(60)"])
    assert code == TIMED_OUT != 0 and time.monotonic() - started < 15


def test_children_of_the_command_are_killed_too(tmp_path):
    """A hung pytest or smoke run may have started helper processes; the whole group goes."""
    marker = tmp_path / "never_written.flag"  # the grandchild sleeps past the limit, so it must never get to write this
    grandchild = tmp_path / "grandchild.py"
    grandchild.write_text(f"import time\ntime.sleep(3)\nopen({str(marker)!r}, 'w').close()\n")
    child = tmp_path / "child.py"
    child.write_text(f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, {str(grandchild)!r}])\ntime.sleep(60)\n")
    assert run_with_limit(1, [sys.executable, str(child)]) == TIMED_OUT
    time.sleep(4)
    assert not marker.exists()


def test_pytest_has_a_per_test_timeout_and_a_slow_marker():
    config = configparser.ConfigParser()
    config.read(ROOT / "pytest.ini")
    section = config["pytest"]
    assert int(section["timeout"]) == 60
    assert "slow" in section["markers"]


def test_the_commit_hook_runs_pytest_and_smoke_under_the_limit():
    hook = (ROOT / "scripts" / "git-hooks" / "pre-commit").read_text()
    assert hook.count("$RUN ") >= 2 and "scripts/run_with_limit.py" in hook
