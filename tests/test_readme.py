"""T9.1, T9.2, T9.4: the README has the template sections, the ablation tables, the disclosures, and every repo path it
mentions exists (B10 note, FR-E). Reads files only; no network."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()

SECTIONS = (
    "1️⃣ Title and value proposition", "2️⃣ Background", "3️⃣ Target user", "4️⃣ Why agentic", "5️⃣ Role, scope and autonomy",
    "6️⃣ Architecture", "7️⃣ End-to-end workflow", "8️⃣ Tools, models and stack", "9️⃣ Evaluation and metrics",
    "🔟 Guardrails", "1️⃣1️⃣ Failure modes", "1️⃣2️⃣ Results and learnings", "1️⃣3️⃣ Future improvements",
    "1️⃣4️⃣ Demo and artifacts", "1️⃣5️⃣ Role-based signal",
)


def test_the_title_quick_start_and_all_fifteen_sections_are_present():
    assert README.startswith("# 🧠 ")
    assert "## 🚀 Quick Start" in README
    headings = re.findall(r"^## (.+)$", README, re.M)
    for section in SECTIONS:
        assert any(h.startswith(section) for h in headings), section
    assert any(h.startswith("Appendix") for h in headings)


def test_sections_come_in_order():
    positions = [README.index(f"## {section}") for section in SECTIONS]
    assert positions == sorted(positions)
    assert README.index("## 🚀 Quick Start") < positions[0] < README.index("## Appendix")


def test_the_ablation_tables_and_the_final_results_are_there():
    assert "### Retrieval ablation" in README and "| hybrid + rerank (production) |" in README
    assert "### Chunking ablation" in README and "### Ragas" in README
    assert "v9 run 1" in README and "stopped by a stop rule" in README  # every run is shown, not only the best
    assert "### Throughput" in README and "Requests per second" in README


def test_the_disclosures_are_stated():
    assert "MCP: not used; direct GitHub REST API (allowed per FAQ Q12)" in README
    assert "default run uses **mock tickets**" in README
    assert "SMARTDESK_TICKETS_MODE=mock" in README and "fine-grained personal access token" in README
    assert "[REPO_URL]" in README  # replaced when the public URL is known
    assert "![SmartDesk AI architecture](docs/architecture.svg)" in README


def test_the_rubric_map_names_every_row_and_bonus_item():
    for row in ("**A. KB and RAG", "**B. Ticket creation", "**C. Ticket status", "**D. Orchestration", "**E. Code and docs",
                "**F. Robustness") + tuple(f"**Bonus {n}:" for n in range(1, 7)):
        assert row in README, row


def test_every_repo_path_in_the_readme_exists():
    missing = []
    for path in set(re.findall(r"`((?:smartdesk_ai|tests|evals|scripts|docs|kb)/[\w./-]+)`", README)):
        if "*" in path or path.endswith("/"):
            path = path.rstrip("/")
        if not (ROOT / path).exists():
            missing.append(path)
    assert not missing, missing


def test_test_files_named_by_bare_name_exist():
    named = re.findall(r"`(test_\w+\.py)`", README)
    assert named
    assert [name for name in named if not (ROOT / "tests" / name).exists()] == []


def test_only_fake_emails_appear():
    emails = set(re.findall(r"[\w.*]+@[\w.-]+\.\w+", README))
    assert {e for e in emails if not e.endswith("@smartdesk.example")} == set()
