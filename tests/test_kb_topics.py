"""T1.3: KB covers every FR-A8 topic, with domains as specified (FR-A8)."""
import re

import pytest

from tests.kb_helpers import load_all_entries

ENTRIES = load_all_entries()

# area -> {required topic: regex matched against "topic | title"}
REQUIRED = {
    "IT Support": {
        "password reset": r"password.*reset|reset.*password",
        "VPN configuration": r"VPN.*(install|configure)",
        "software installation requests": r"request new software",
        "MFA setup": r"set up multi-factor",
        "guest Wi-Fi": r"guest Wi-Fi",
        "email on mobile": r"email on your phone",
    },
    "HR Policies": {
        "casual leave": r"casual leave",
        "sick leave": r"sick leave",
        "earned leave": r"earned leave",
        "WFH guidelines": r"hybrid schedule",
        "reimbursement": r"expense claim",
        "code of conduct": r"code of conduct",
        "anti-harassment": r"anti-harassment policy",
        "performance review cycle": r"review cycle",
    },
    "Onboarding": {
        "day-one checklist": r"day-one checklist",
        "IT equipment allocation": r"what equipment a new joiner",
        "benefits enrollment": r"enrol in benefits",
        "buddy programme": r"buddy programme",
        "probation": r"probation: length",
    },
    "Payroll and Comp": {
        "pay-day schedule": r"pay-day schedule",
        "tax declaration": r"tax declaration",
        "salary structure": r"salary structure",
        "stock-option vesting": r"stock-option vesting",
    },
}


def matches(entry, pattern):
    return re.search(pattern, f"{entry.topic} | {entry.title}", re.IGNORECASE)


@pytest.mark.parametrize(
    "area,topic,pattern",
    [(a, t, p) for a, topics in REQUIRED.items() for t, p in topics.items()],
)
def test_required_topic_has_an_entry(area, topic, pattern):
    assert [e.id for e in ENTRIES if matches(e, pattern)], f"{area}: no entry for '{topic}'"


def test_it_support_and_equipment_entries_are_it():
    it_file = [e for e in ENTRIES if e.source == "it_support.md"]
    equipment = [e for e in ENTRIES if e.topic == "equipment"]
    assert it_file and equipment
    assert all(e.domain == "IT" for e in it_file + equipment)


def test_rest_of_onboarding_and_all_hr_files_are_hr():
    rest = [
        e for e in ENTRIES
        if e.source in {"company.md", "hr_policies.md", "payroll_comp.md"}
        or (e.source == "onboarding.md" and e.topic != "equipment")
    ]
    assert rest
    assert all(e.domain == "HR" for e in rest)


def test_equipment_allocation_sits_inside_the_hr_onboarding_file():
    eq = next(e for e in ENTRIES if e.id == "IT-EQUIP-01")
    assert eq.source == "onboarding.md" and eq.domain == "IT"
