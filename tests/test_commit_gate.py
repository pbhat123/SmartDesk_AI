"""The commit gate is mechanical (CLAUDE.md workflow rules): a tracked pre-commit hook runs pytest and blocks on failure."""
import os
import subprocess

from smartdesk_ai.rag.ingest import ROOT

HOOK = ROOT / "scripts" / "git-hooks" / "pre-commit"


def test_the_pre_commit_hook_exists_is_executable_and_runs_pytest_blocking_on_failure():
    text = HOOK.read_text()
    assert os.access(HOOK, os.X_OK) and text.startswith("#!/bin/bash")
    assert "-m pytest" in text and "exit 1" in text and "commit blocked" in text


def test_the_install_script_points_git_at_the_hooks_folder():
    script = (ROOT / "scripts" / "install_hooks.sh").read_text()
    assert "core.hooksPath scripts/git-hooks" in script and os.access(ROOT / "scripts" / "install_hooks.sh", os.X_OK)


def test_this_clone_uses_the_tracked_hooks():
    configured = subprocess.run(["git", "config", "core.hooksPath"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    assert configured == "scripts/git-hooks", "run scripts/install_hooks.sh once per clone"
