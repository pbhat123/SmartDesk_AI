"""T1.12: scenario scripts parse against a schema; flows A, B, C match SPEC section 7."""
import re

import pytest

from evals.common import EVALS_DIR
from evals.scenario_schema import load_scenario
from tests.kb_helpers import load_all_entries

DIR = EVALS_DIR / "scenarios"
REQUIRED = {
    "kb_answer", "ticket_create_yes", "ticket_create_no", "ticket_create_edit",
    "status_0", "status_1", "status_2plus", "api_down", "degraded_mode",
    "flow_a", "flow_b", "flow_c",
}
FLOW_STEPS = {  # SPEC section 7 (document section 8), as agent-reply steps
    "flow_a": ["kb_answer", "thanks", "goodbye"],
    "flow_b": ["ticket_offer_with_email_request", "ticket_summary", "ticket_created"],
    "flow_c": ["email_requested", "ticket_status_single"],
}
ENTRY_IDS = {e.id for e in load_all_entries()}
SCENARIOS = {p.stem: load_scenario(p) for p in sorted(DIR.glob("*.yaml"))}


def steps(name):
    return [t.expect.step for t in SCENARIOS[name].turns]


def test_all_required_scripts_exist_and_ids_match_filenames():
    assert REQUIRED <= set(SCENARIOS)
    for name, s in SCENARIOS.items():
        assert s.id == name


def test_fr_tags_cited_and_emails_are_fake():
    for s in SCENARIOS.values():
        assert all(re.fullmatch(r"FR-[A-I]\d+", fr) for fr in s.fr), s.id
        emails = [s.setup.email_known] if s.setup.email_known else []
        emails += [t.user for t in s.turns if "@" in t.user]
        assert all(e.endswith("@smartdesk.example") for e in emails), s.id


def test_cited_entry_ids_exist():
    for s in SCENARIOS.values():
        for t in s.turns:
            assert set(t.expect.cites) <= ENTRY_IDS, s.id


@pytest.mark.parametrize("name,expected", FLOW_STEPS.items())
def test_flows_match_spec_steps(name, expected):
    assert SCENARIOS[name].kind == "flow"
    assert steps(name) == expected


def test_flow_a_is_password_reset_with_no_ticket():
    s = SCENARIOS["flow_a"]
    assert "password" in s.turns[0].user.lower()
    assert s.turns[0].expect.cites == ["IT-PWD-01"]
    assert not any(t.expect.ticket_created for t in s.turns)


def test_flow_b_is_monitor_flicker_with_full_summary_and_ticket_id():
    s = SCENARIOS["flow_b"]
    assert "monitor" in s.turns[0].user.lower() and "flicker" in s.turns[0].user.lower()
    assert s.setup.email_known is None and "@" in s.turns[1].user
    assert s.turns[1].expect.summary_fields == ["title", "description", "category", "email"]
    assert s.turns[2].user.lower().startswith("yes")
    last = s.turns[2].expect
    assert last.ticket_created and last.ticket_id_in_reply and last.ticket_link_in_reply


def test_flow_c_shows_id_title_status_and_team_update_for_one_ticket():
    s = SCENARIOS["flow_c"]
    assert s.setup.tickets_for_email == 1 and s.setup.email_known is None
    assert s.turns[0].expect.awaiting_after == "email"
    last = s.turns[1].expect
    assert last.ticket_id_in_reply and last.status_in_reply and last.team_update_in_reply


def test_ticket_create_yes_no_edit_variants():
    yes, no, edit = (SCENARIOS[n] for n in ("ticket_create_yes", "ticket_create_no", "ticket_create_edit"))
    assert yes.turns[-1].expect.ticket_created is True
    assert no.turns[-1].expect.step == "ticket_cancelled" and no.turns[-1].expect.ticket_created is False
    assert steps("ticket_create_edit") == ["ticket_summary", "ticket_summary", "ticket_created"]
    assert "priority" in edit.turns[1].expect.summary_fields


def test_never_creates_without_a_confirmation_turn():
    for s in SCENARIOS.values():
        for i, t in enumerate(s.turns):
            if t.expect.step == "ticket_created":
                assert s.turns[i - 1].expect.step == "ticket_summary", s.id


def test_status_variants_cover_zero_one_and_many():
    assert SCENARIOS["status_0"].setup.tickets_for_email == 0
    assert steps("status_0") == ["tickets_none"]
    assert SCENARIOS["status_1"].setup.tickets_for_email == 1
    many = SCENARIOS["status_2plus"]
    assert many.setup.tickets_for_email >= 2
    assert steps("status_2plus") == ["ticket_list", "ticket_status_single"]
    assert many.turns[0].expect.awaiting_after == "ticket_pick"


def test_api_down_and_degraded_mode_setups():
    down = SCENARIOS["api_down"]
    assert down.setup.github == "down"
    assert down.turns[-1].expect.step == "ticket_unavailable" and down.turns[-1].expect.ticket_created is False
    assert "try again" in " ".join(down.turns[-1].expect.reply_contains).lower()
    deg = SCENARIOS["degraded_mode"]
    assert "reranker" in deg.setup.degraded
    assert steps("degraded_mode") == ["kb_answer", "ticket_offer"]
