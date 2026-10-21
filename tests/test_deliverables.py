"""T9.6: the deliverables check (SPEC section 8, rubric F3, F5): nothing secret is tracked, `.env.example` lists every
variable the code reads, the KB and config files are there, the LICENSE is MIT, and no `.DS_Store` is in git."""
import re
import subprocess
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

SECRET_PATTERNS = {
    "OpenAI key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}"),
    "Langfuse key": re.compile(r"\b[ps]k-lf-[0-9a-f-]{20,}"),
    "LangSmith key": re.compile(r"\blsv2_(?:pt|sk)_[A-Za-z0-9_]{20,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}

ENV_VARIABLES = (
    "OPENAI_API_KEY", "GITHUB_TOKEN", "TICKETS_REPO", "LANGSMITH_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST", "REDIS_URL", "MASK_SALT", "SMARTDESK_LLM_MODE", "SMARTDESK_TICKETS_MODE", "SMARTDESK_DATA_DIR",
    "SMARTDESK_LEDGER_PATH",
)


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [name for name in out.split("\0") if name and (ROOT / name).is_file()]


def test_no_secret_looking_value_is_in_a_tracked_file():
    hits = []
    for name in tracked_files():
        path = ROOT / name
        if path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        hits += [f"{name}: {label}" for label, pattern in SECRET_PATTERNS.items() if pattern.search(text)]
    assert not hits, hits


def test_no_env_file_or_runtime_data_is_tracked():
    bad = [n for n in tracked_files() if n == ".env" or (n.startswith(".env.") and n != ".env.example") or n.startswith("data/")]
    assert bad == []
    ignore = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in ignore and "data/" in ignore


def test_env_example_lists_every_variable_the_code_reads():
    text = (ROOT / ".env.example").read_text()
    missing = [name for name in ENV_VARIABLES if not re.search(rf"^#?\s*{name}=", text, re.M)]
    assert missing == []
    for line in text.splitlines():  # the example ships placeholders only
        if re.match(r"^(OPENAI_API_KEY|GITHUB_TOKEN)=", line):
            assert "your-" in line and "-here" in line, line


def test_the_env_variables_the_code_reads_are_all_known_to_this_test():
    read = set()
    for path in (ROOT / "smartdesk_ai").rglob("*.py"):
        read |= set(re.findall(r"""["']((?:SMARTDESK|LANGFUSE|LANGSMITH)_[A-Z_]+|OPENAI_API_KEY|GITHUB_TOKEN|TICKETS_REPO|REDIS_URL|MASK_SALT)["']""", path.read_text()))
    assert read - set(ENV_VARIABLES) - {"LANGSMITH_TRACING"} == set()  # LANGSMITH_TRACING is only ever checked, never enabled


@pytest.mark.parametrize("name", [
    "config.yaml", "requirements.txt", ".env.example", "README.md", "SPEC.md", "docs/architecture.md",
    "docs/architecture.svg", "docs/architecture.png", "scripts/seed_tickets.py", "scripts/load_test.py", "scripts/smoke_ui.py",
    "kb/it_support.md", "kb/hr_policies.md", "kb/onboarding.md", "kb/payroll_comp.md", "kb/company.md", "kb/qa.json",
])
def test_required_files_exist_and_are_tracked(name):
    assert (ROOT / name).is_file(), name
    assert name in tracked_files(), f"{name} exists but is not tracked by git"


def test_the_license_is_mit():
    text = (ROOT / "LICENSE").read_text()
    assert text.startswith("MIT License") and "Permission is hereby granted, free of charge" in text
    assert re.search(r"^Copyright \(c\) 20\d\d \S", text, re.M)
    if "[YOUR NAME]" in text:
        warnings.warn("LICENSE still has the [YOUR NAME] placeholder: replace it before the first push", stacklevel=1)


def test_ds_store_is_ignored_and_not_tracked():
    assert not [n for n in tracked_files() if Path(n).name == ".DS_Store"]
    assert ".DS_Store" in (ROOT / ".gitignore").read_text().splitlines()
