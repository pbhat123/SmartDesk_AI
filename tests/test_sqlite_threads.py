"""Streamlit runs each rerun on a different thread and the app context is shared through st.cache_resource, so every
SQLite store must work from any thread (live UI test, 2026-10-18: ProgrammingError "SQLite objects created in a thread
can only be used in that same thread" at the ticket gate). Mock mode only."""
import re
import threading
from pathlib import Path

import pytest

from smartdesk_ai.agent.factory import build_stack
from smartdesk_ai.ops.budget import SpendStore
from smartdesk_ai.ops.health import HealthRegistry
from smartdesk_ai.ops.rate_limit import RateLimiter
from smartdesk_ai.ops.hitl_counters import load_counts
from smartdesk_ai.tickets.sqlite_map import TicketStore
from tests.ui_helpers import mock_context

EMAIL = "priya.k@smartdesk.example"
ROOT = Path(__file__).resolve().parents[1]


def in_new_thread(fn, *args):
    """Run fn(*args) on a thread that did not create anything, and return its result or re-raise its error."""
    box: dict = {}

    def target():
        try:
            box["value"] = fn(*args)
        except BaseException as error:  # noqa: BLE001 - re-raised below
            box["error"] = error

    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


# --- every store, used from another thread --------------------------------------------------------------------

@pytest.mark.parametrize("where", [":memory:", "file"])
def test_the_ticket_map_works_from_another_thread(tmp_path, where):
    store = TicketStore(":memory:" if where == ":memory:" else tmp_path / "t.db")
    in_new_thread(store.begin, "req-1")
    in_new_thread(store.complete, "req-1", 7)
    in_new_thread(store.link_email, EMAIL, 7)
    assert in_new_thread(store.get, "req-1") == ("done", 7)
    assert in_new_thread(store.numbers_for, EMAIL) == [7]


def test_the_budget_store_and_the_rate_limiter_work_from_another_thread(tmp_path, settings):
    spend = SpendStore(tmp_path / "ops.db")
    assert in_new_thread(spend.add, "2026-10-18", 1.5) == 1.5 and in_new_thread(spend.get, "2026-10-18") == 1.5
    limiter = RateLimiter(settings.rate_limits, tmp_path / "ops.db")
    assert in_new_thread(limiter.hit, "questions", "s1").allowed


def test_the_checkpointer_works_from_another_thread(settings, tmp_path):
    stack = build_stack(settings, environ={}, dotenv_path=None, data_dir=tmp_path, registry=HealthRegistry())
    config = {"configurable": {"thread_id": "t"}}
    assert in_new_thread(lambda: stack.graph.get_state(config).values) == {}


def test_no_store_in_the_code_base_opens_a_connection_that_is_tied_to_its_thread():
    offenders = []
    for path in (ROOT / "smartdesk_ai").rglob("*.py"):
        for call in re.finditer(r"sqlite3\.connect\((?:[^()]|\([^()]*\))*\)", path.read_text()):
            if "check_same_thread=False" not in call.group(0):
                offenders.append(f"{path.relative_to(ROOT)}: {call.group(0)}")
    assert offenders == []


# --- the whole ticket flow, every turn on a different thread -------------------------------------------------------

def test_a_full_ticket_flow_works_when_every_turn_runs_on_its_own_thread(tmp_path):
    ctx = mock_context(tmp_path)
    turns = ["My laptop screen is cracked, please raise a ticket", EMAIL,
             "change the title to Cracked laptop screen", "yes"]
    replies = [in_new_thread(ctx.service.handle, "s1", text).text for text in turns]
    assert "To open a ticket I need your work email" in replies[0]
    assert "Title: Cracked laptop screen" in replies[2]  # draft -> edit
    assert re.search(r"Ticket #\d+ \(mock\)", replies[3])  # -> yes
    counts = load_counts(tmp_path / "hitl.jsonl")
    assert (counts.approvals, counts.edits) == (1, 1)


def test_the_duplicate_check_and_status_lookup_also_work_across_threads(tmp_path):
    ctx = mock_context(tmp_path)
    for text in ("My monitor keeps flickering", EMAIL, "yes"):
        in_new_thread(ctx.service.handle, "a", text)
    in_new_thread(ctx.service.handle, "b", "any update on my ticket?")
    status = in_new_thread(ctx.service.handle, "b", EMAIL).text
    assert "(mock)" in status and "Status:" in status
    asked = [in_new_thread(ctx.service.handle, "c", t).text for t in ("My monitor keeps flickering", EMAIL)]
    assert "already have an open ticket" in asked[-1]  # the duplicate check could read the map from another thread


def test_parallel_conversations_each_get_their_own_ticket(tmp_path):
    ctx = mock_context(tmp_path)
    results: dict[int, str] = {}
    errors: list[BaseException] = []

    def converse(n: int):
        try:
            email = f"user{n}@smartdesk.example"
            for text in (f"My monitor {n} keeps flickering", email, "yes"):
                results[n] = ctx.service.handle(f"s{n}", text).text
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=converse, args=(n,)) for n in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == []
    numbers = [int(re.search(r"Ticket #(\d+)", text).group(1)) for text in results.values()]
    assert len(numbers) == 6 and len(set(numbers)) == 6
