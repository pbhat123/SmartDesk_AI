"""Checks one scripted turn against its `expect` block and returns what did not match (empty = matched).

Shared by the pytest scenario runner (which asserts there are no problems) and the agent-metrics run
(which counts them), so both judge a turn the same way.
"""
import re
from typing import Any

from evals.scenario_schema import Expect

FIELD_LABELS = {"title": "Title:", "description": "Description:", "category": "Category:", "email": "Email:",
                "priority": "Priority:"}


def _step_problems(expect: Expect, state: Any, issues_before: int, issues_after: int) -> list[str]:
    step, reply = expect.step, state.reply
    checks: dict[str, tuple[bool, str]] = {
        "kb_answer": (state.part_result is not None and state.part_result.answerable and state.pending_offer is None,
                      "expected a KB answer"),
        "thanks": (bool(re.search(r"welcome|glad|happy", reply, re.I)) and state.route_kind == "smalltalk",
                   "expected a thanks reply"),
        "goodbye": (bool(re.search(r"bye|take care", reply, re.I)) and state.route_kind == "smalltalk",
                    "expected a goodbye reply"),
        "ticket_offer": (state.pending_offer is not None and "yes or no" in reply.lower(), "expected a ticket offer"),
        "ticket_offer_with_email_request": (
            state.pending_offer is not None and "email" in reply.lower() and "don't know" in reply.lower(),
            "expected the offer with an email request"),
        "email_requested": (state.awaiting == "email" and "email" in reply.lower(), "expected an email request"),
        "ticket_summary": ("Shall I go ahead and create this?" in reply, "expected the ticket summary"),
        "ticket_created": (issues_after == issues_before + 1, "exactly one ticket must be created"),
        "ticket_cancelled": ("won't create" in reply and issues_after == issues_before, "expected a cancellation"),
        "tickets_none": ("no tickets" in reply.lower() and not re.search(r"#\d+|https?://", reply),
                         "expected 'no tickets'"),
        "ticket_list": (state.awaiting == "ticket_pick" and len(re.findall(r"^\d+\. #\d+ ", reply, re.M)) >= 2,
                        "expected a numbered ticket list"),
        "ticket_status_single": (len(re.findall(r"#\d+", reply)) == 1, "expected one ticket"),
        "ticket_unavailable": ("try again later" in reply.lower(), "expected a try-again-later reply"),
    }
    ok, message = checks[step]
    return [] if ok else [f"{step}: {message}"]


def check_turn(
    expect: Expect, state: Any, *, client: Any, issues_before: int, issues_after: int, searched: bool
) -> list[str]:
    problems = _step_problems(expect, state, issues_before, issues_after)
    for entry_id in expect.cites:
        if state.part_result is None or entry_id not in state.part_result.citations:
            problems.append(f"missing citation {entry_id}")
    problems += [f"reply lacks {text!r}" for text in expect.reply_contains if text not in state.reply]
    problems += [f"summary lacks {field}" for field in expect.summary_fields if FIELD_LABELS[field] not in state.reply]
    created = issues_after > issues_before
    if expect.ticket_created is not None and created != expect.ticket_created:
        problems.append(f"ticket_created should be {expect.ticket_created}")
    if expect.intent is not None and (state.intent is None or state.intent.intent != expect.intent):
        problems.append(f"intent should be {expect.intent}")
    if expect.tool == "search_kb" and not searched:
        problems.append("expected a KB search")
    if expect.tool == "create_ticket" and issues_after != issues_before + 1:
        problems.append("expected one create_ticket call to succeed")
    if expect.tool == "none" and (searched or created):
        problems.append("expected no tool call")
    if expect.awaiting_after is not None and state.awaiting != expect.awaiting_after:
        problems.append(f"awaiting should be {expect.awaiting_after}")
    if expect.ticket_id_in_reply or expect.ticket_link_in_reply or expect.status_in_reply or expect.team_update_in_reply:
        numbers = re.findall(r"#(\d+)", state.reply)
        if len(numbers) != 1:
            problems.append("expected exactly one ticket in the reply")
        else:
            ticket = client._issues[int(numbers[0])]
            if expect.ticket_link_in_reply and ticket.url not in state.reply:
                problems.append("ticket link missing")
            if expect.status_in_reply and ticket.status.value not in state.reply:
                problems.append("ticket status missing")
            if expect.team_update_in_reply:
                update = client.latest_update(ticket.number)
                if update is None or update.text not in state.reply:
                    problems.append("team update missing")
    return problems
