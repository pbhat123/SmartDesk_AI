"""T1.2: KB content quality (FR-A1, FR-A9)."""
import re

from smartdesk_ai.rag.kb_schema import validate_entries
from tests.kb_helpers import load_all_entries

ENTRIES = load_all_entries()
DETAIL = re.compile(r"https?://|extension \d{4}|₹|\d")
NUMBERED_STEP = re.compile(r"^\d+\. ", re.MULTILINE)
POLICY_MARKER = re.compile(r"Interpretation|\*\*[^*]+\?\*\*")


def test_at_least_30_valid_unique_entries():
    assert len(ENTRIES) >= 30
    validate_entries(ENTRIES)


def test_both_domains_present_and_domain_is_per_entry():
    assert {e.domain for e in ENTRIES} == {"IT", "HR"}
    by_file = {}
    for e in ENTRIES:
        by_file.setdefault(e.source, set()).add(e.domain)
    assert by_file["onboarding.md"] == {"IT", "HR"}


def test_every_entry_has_concrete_detail():
    bare = [e.id for e in ENTRIES if not DETAIL.search(e.body)]
    assert not bare, f"entries without a number, URL or extension: {bare}"


def test_entry_length_in_chunk_target():
    # FR-A3 targets 200 to 500 tokens; roughly 150 to 375 words.
    off = {e.id: len(e.body.split()) for e in ENTRIES if not 150 <= len(e.body.split()) <= 375}
    assert not off, f"entries outside 150-375 words: {off}"


def test_mix_of_simple_howto_and_policy_entries():
    how_to = [e for e in ENTRIES if len(NUMBERED_STEP.findall(e.body)) >= 3]
    policy = [e for e in ENTRIES if POLICY_MARKER.search(e.body)]
    simple = [e for e in ENTRIES if e not in how_to and e not in policy]
    assert len(how_to) >= 10
    assert len(policy) >= 5
    assert len(simple) >= 8


def test_fictional_company_name_is_smartdesk_ai():
    text = "\n".join(e.body for e in ENTRIES)
    assert "SmartDesk AI" in text
    assert "Lumora" not in text
    assert "Kreeda" not in text


def test_main_switchboard_number_is_only_ever_the_switchboard():
    number = "+91-80-5550-0100"
    for e in ENTRIES:
        for m in re.finditer(re.escape(number), e.body):
            before = e.body[max(0, m.start() - 60):m.start()].lower()
            assert "switchboard" in before, f"{e.id}: {number} used for something else"


def test_company_profile_entry_has_industry_size_offices_and_contacts():
    about = next(e for e in ENTRIES if e.id == "HR-ABOUT-01")
    assert about.source == "company.md" and about.domain == "HR"
    for needle in ("supply-chain", "450", "Bengaluru", "Pune", "extension 4100", "extension 4200"):
        assert needle in about.body


def test_no_real_looking_personal_emails():
    text = "\n".join(e.body for e in ENTRIES)
    emails = set(re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", text))
    assert all(m.endswith("smartdesk.example") for m in emails), emails
