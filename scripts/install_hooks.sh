#!/bin/bash
# Point git at the tracked hooks folder, so the commit gate (pytest, plus smoke_ui.py for app code) applies to every commit.
cd "$(git rev-parse --show-toplevel)" || exit 1
chmod +x scripts/git-hooks/*
git config core.hooksPath scripts/git-hooks
echo "git hooks installed: core.hooksPath = $(git config core.hooksPath)"
