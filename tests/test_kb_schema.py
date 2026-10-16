"""T1.1: KB entry schema and validator (FR-A1)."""
import pytest

from smartdesk_ai.rag.kb_schema import KBValidationError, parse_entries, validate_entries


def section(id_="HR-LEAVE-03", domain="HR", heading="Annual leave", **overrides):
    fields = {
        "id": id_,
        "domain": domain,
        "topic": "leave",
        "title": "Annual leave",
        "owner": "People Team",
        "last_reviewed": "2026-05-01",
    }
    fields.update(overrides)
    lines = [f"{k}: {v}" for k, v in fields.items() if v is not None]
    return f"## {heading}\n\n```yaml\n" + "\n".join(lines) + "\n```\n\nEmployees get 25 days.\n"


def test_accepts_good_entry():
    (entry,) = parse_entries("# Leave policy\n\n" + section(), source="hr.md")
    assert entry.id == "HR-LEAVE-03"
    assert entry.domain == "HR"
    assert entry.title == "Annual leave"
    assert "25 days" in entry.body
    assert entry.source == "hr.md"
    assert entry.document == "Leave policy"


@pytest.mark.parametrize("field", ["id", "domain", "topic", "title", "owner", "last_reviewed"])
def test_rejects_missing_field(field):
    with pytest.raises(KBValidationError, match=field):
        parse_entries(section(**{field: None}), source="x.md")


@pytest.mark.parametrize("domain", ["MISC", "it", "Finance"])
def test_rejects_bad_domain(domain):
    with pytest.raises(KBValidationError, match="domain"):
        parse_entries(section(domain=domain), source="x.md")


def test_rejects_bad_id_format():
    with pytest.raises(KBValidationError, match="id"):
        parse_entries(section(id_="leave3"), source="x.md")


def test_rejects_section_without_metadata_block():
    with pytest.raises(KBValidationError, match="metadata"):
        parse_entries("## Parking\n\nNo block here.\n", source="x.md")


def test_domain_is_per_entry_not_per_file():
    text = section("HR-ONB-01", "HR", "Day one") + "\n" + section("IT-EQUIP-01", "IT", "Laptop allocation")
    entries = parse_entries(text, source="onboarding.md")
    assert [e.domain for e in entries] == ["HR", "IT"]


def test_duplicate_id_rejected_across_files():
    a = parse_entries(section(), source="a.md")
    b = parse_entries(section(), source="b.md")
    with pytest.raises(KBValidationError, match="HR-LEAVE-03"):
        validate_entries(a + b)


def test_unique_ids_pass():
    a = parse_entries(section("HR-LEAVE-01"), source="a.md")
    b = parse_entries(section("HR-LEAVE-02"), source="b.md")
    assert len(validate_entries(a + b)) == 2


def test_error_names_source():
    with pytest.raises(KBValidationError, match="bad.md"):
        parse_entries(section(domain="XX"), source="bad.md")
