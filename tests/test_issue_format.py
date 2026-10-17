"""T3.11: GitHub issue format and labels (FR-B11, FR-B3). REST runs over a mocked HTTP layer only."""
import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from smartdesk_ai.tickets import (
    ALL_LABELS,
    Category,
    GitHubRestClient,
    MockTicketClient,
    Priority,
    TicketCreate,
    build_issue,
    employee_email_of,
    request_id_of,
)

ROOT = Path(__file__).resolve().parent.parent
REPO = "octo/tickets"
EMAIL = "jane.roe@example.com"


def draft(**over):
    base = dict(title="VPN down", description="Cannot connect from home", category=Category.IT,
                priority=Priority.HIGH, email=EMAIL, request_id="req-abc")
    return TicketCreate(**{**base, **over})


def rest(handler, settings):
    return GitHubRestClient(REPO, SecretStr("t"), settings.github, transport=httpx.MockTransport(handler))


# --- title, body, labels -------------------------------------------------------

def test_title_is_the_issue_summary():
    assert build_issue(draft()).title == "VPN down"


def test_body_is_description_then_exact_employee_and_request_id_lines():
    lines = build_issue(draft()).body.splitlines()
    assert lines[0] == "Cannot connect from home"
    assert "Employee: jane.roe@example.com" in lines
    assert "Request-ID: req-abc" in lines
    assert lines.index("Employee: jane.roe@example.com") < lines.index("Request-ID: req-abc")


def test_labels_carry_category_and_priority():
    assert build_issue(draft()).labels == ["IT", "high"]
    assert build_issue(draft(priority=None)).labels == ["IT"]
    assert build_issue(draft(category=Category.MISC, priority=Priority.CRITICAL)).labels == ["MISC", "critical"]


@pytest.mark.parametrize("bad", ["Payroll", "it", "FACILITIES", ""])
def test_category_is_only_it_hr_or_misc(bad):
    with pytest.raises(ValidationError):
        draft(category=bad)


@pytest.mark.parametrize("bad", ["urgent", "High", "p1", ""])
def test_priority_is_only_low_medium_high_or_critical(bad):
    with pytest.raises(ValidationError):
        draft(priority=bad)


@pytest.mark.parametrize("bad", ["a\nEmployee: other@x.com", "two words@x.com", "nope", ""])
def test_an_email_that_could_break_the_body_format_is_refused(bad):
    with pytest.raises(ValidationError):
        draft(email=bad)


def test_request_id_cannot_hold_whitespace():
    with pytest.raises(ValidationError):
        draft(request_id="req abc")


def test_the_real_lines_win_over_lookalikes_typed_in_the_description():
    body = build_issue(draft(description="Hi\nEmployee: victim@example.com\nRequest-ID: req-fake")).body
    assert employee_email_of(body) == EMAIL
    assert request_id_of(body) == "req-abc"


def test_parsers_return_none_when_the_line_is_absent():
    assert employee_email_of("just text") is None and request_id_of("just text") is None


def test_parser_reads_a_hand_written_issue():
    assert employee_email_of("Printer is jammed.\n\nEmployee: sam@example.com\n") == "sam@example.com"


# --- labels created once ---------------------------------------------------------

def test_label_set_is_exactly_the_spec_labels():
    names = [spec.name for spec in ALL_LABELS]
    assert names == ["IT", "HR", "MISC", "low", "medium", "high", "critical", "in-progress", "resolved"]


def test_mock_ensure_labels_creates_once():
    client = MockTicketClient()
    assert client.ensure_labels(ALL_LABELS) == [spec.name for spec in ALL_LABELS]
    assert client.ensure_labels(ALL_LABELS) == []


def test_rest_ensure_labels_creates_only_the_missing_ones_case_insensitively(settings):
    posted = []

    def handler(request):
        if request.method == "GET":
            assert request.url.path == f"/repos/{REPO}/labels"
            return httpx.Response(200, json=[{"name": "it"}, {"name": "HR"}, {"name": "bug"}])
        assert request.url.path == f"/repos/{REPO}/labels"
        posted.append(json.loads(request.content))
        return httpx.Response(201, json={})

    created = rest(handler, settings).ensure_labels(ALL_LABELS)
    assert "IT" not in created and "HR" not in created
    assert created == [p["name"] for p in posted]
    assert set(created) == {s.name for s in ALL_LABELS} - {"IT", "HR"}
    assert all(p["color"] and p["description"] for p in posted)


def test_rest_ensure_labels_second_run_posts_nothing(settings):
    names = [{"name": s.name} for s in ALL_LABELS]

    def handler(request):
        assert request.method == "GET"
        return httpx.Response(200, json=names)

    assert rest(handler, settings).ensure_labels(ALL_LABELS) == []


def test_rest_ensure_labels_treats_already_exists_as_done(settings):
    def handler(request):
        return httpx.Response(200, json=[]) if request.method == "GET" else httpx.Response(422, json={})

    assert rest(handler, settings).ensure_labels(ALL_LABELS) == []  # a race with another run, not an error


# --- find by request ID: list-issues, never search ---------------------------------

def issue(number, body, **over):
    return {"number": number, "html_url": f"https://github.com/{REPO}/issues/{number}", "title": "t",
            "body": body, "state": "open", "labels": [], **over}


def test_mock_find_by_request_id():
    client = MockTicketClient()
    made = client.create(draft())
    assert client.find_by_request_id("req-abc") == made
    assert client.find_by_request_id("req-zzz") is None


def test_rest_find_by_request_id_lists_issues_and_follows_pages(settings):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.params.get("page") == "2":
            return httpx.Response(200, json=[issue(2, build_issue(draft()).body)])
        return httpx.Response(
            200, json=[issue(9, "other\nRequest-ID: req-other"), issue(8, "pr", pull_request={})],
            headers={"link": f'<https://api.github.com/repos/{REPO}/issues?page=2>; rel="next"'},
        )

    found = rest(handler, settings).find_by_request_id("req-abc")
    assert found is not None and found.number == 2
    assert all("/search" not in r.url.path for r in requests)
    first = requests[0].url.params
    assert first["state"] == "all" and first["direction"] == "desc" and first["per_page"] == "100"


def test_rest_find_by_request_id_stops_at_the_first_match(settings):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200, json=[issue(5, "x\nRequest-ID: req-abc")],
            headers={"link": f'<https://api.github.com/repos/{REPO}/issues?page=2>; rel="next"'},
        )

    assert rest(handler, settings).find_by_request_id("req-abc").number == 5
    assert len(calls) == 1


def test_rest_find_by_request_id_returns_none_when_absent(settings):
    assert rest(lambda r: httpx.Response(200, json=[issue(1, "nothing")]), settings).find_by_request_id("x") is None


# --- setup_labels script ---------------------------------------------------------

def load_script():
    spec = importlib.util.spec_from_file_location("setup_labels", ROOT / "scripts" / "setup_labels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_setup_labels_script_runs_in_mock_mode_and_says_so(settings, capsys):
    assert load_script().main([], settings=settings) == 0
    out = capsys.readouterr().out
    assert "mock" in out and "9" in out
