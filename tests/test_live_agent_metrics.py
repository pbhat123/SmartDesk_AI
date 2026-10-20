"""Agent metrics from the live trace (FR-H11, FR-G10): a task is one session, a turn is one request. Local files only."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from smartdesk_ai.ops.live_agent_metrics import FAILURE_PHRASES, compute_live_metrics

T0 = datetime(2026, 10, 20, 9, 0, 0, tzinfo=timezone.utc)


def at(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat()


def turn(session, rid, start, took, *, steps=3, reply="Here is the answer.", blocked=False, no_reply=False, tool_error=False):
    events = [{"ts": at(start), "kind": "request", "text": "q", "blocked": blocked, "reason": None,
               "session_id": session, "request_id": rid}]
    events += [{"ts": at(start + 0.01 * i), "kind": "agent_step", "step": "start" if i == 0 else f"s{i}", "index": i,
                "session_id": session, "request_id": rid} for i in range(steps + 1)]
    if tool_error:
        events.append({"ts": at(start + 0.1), "kind": "tool_call", "tool": "create_ticket", "outcome": "error",
                       "session_id": session, "request_id": rid})
    if not no_reply:
        events.append({"ts": at(start + took), "kind": "reply", "text": reply, "blocked": blocked,
                       "session_id": session, "request_id": rid})
    return events


def cost(session, rid, usd):
    return {"ts": at(0), "request_id": rid, "session_id": session, "task": "kb", "agent": "it_worker", "step": "llm", "usd": usd}


def test_no_sessions_gives_none():
    assert compute_live_metrics([], [], since=None) is None


def test_rates_steps_latency_percentiles_and_cost_per_task():
    events = (turn("a", "a1", 0, 1.0, steps=4) + turn("a", "a2", 10, 3.0, steps=2)  # task a: 2 turns, completed
              + turn("b", "b1", 20, 2.0, steps=5, reply="Sorry, please try again later.")  # task b: failed
              + turn("c", "c1", 30, 4.0, steps=3))  # task c: completed
    costs = [cost("a", "a1", 0.002), cost("a", "a2", 0.004), cost("b", "b1", 0.001), cost("c", "c1", 0.003)]
    m = compute_live_metrics(events, costs, since=None)
    assert (m.n_tasks, m.n_turns) == (3, 4)
    assert m.completion_rate == pytest.approx(2 / 3) and m.error_rate == pytest.approx(1 / 4)
    assert m.steps_to_completion == pytest.approx((6 + 3) / 2)  # completed tasks only: (4 + 2) and 3; "start" is not a step
    assert m.turn_p50_ms == pytest.approx(2000.0) and m.turn_p95_ms == pytest.approx(4000.0)
    assert m.usd_per_task == pytest.approx((0.006 + 0.001 + 0.003) / 3)


def test_a_turn_without_a_reply_or_with_a_failed_tool_call_is_an_error_but_a_blocked_reply_is_not():
    events = (turn("a", "a1", 0, 1.0, no_reply=True) + turn("b", "b1", 5, 1.0, tool_error=True)
              + turn("c", "c1", 10, 1.0, reply="I can't help with that.", blocked=True) + turn("d", "d1", 15, 1.0))
    m = compute_live_metrics(events, [], since=None)
    assert m.n_turns == 4 and m.error_rate == pytest.approx(2 / 4) and m.completion_rate == pytest.approx(2 / 4)
    assert m.turn_p50_ms is not None  # the turn with no reply has no latency, the other three do
    assert "try again later" in FAILURE_PHRASES


def test_the_window_keeps_only_turns_that_started_inside_it():
    events = turn("old", "o1", 0, 1.0) + turn("new", "n1", 100, 2.0) + turn("new", "n2", 110, 2.0)
    since = T0 + timedelta(seconds=50)
    m = compute_live_metrics(events, [cost("old", "o1", 9.0), cost("new", "n1", 0.01)], since=since)
    assert (m.n_tasks, m.n_turns) == (1, 2) and m.usd_per_task == pytest.approx(0.01)
    assert compute_live_metrics(events, [], since=T0 + timedelta(days=1)) is None


def test_damaged_lines_are_skipped_and_no_text_is_kept():
    from smartdesk_ai.ops.live_agent_metrics import load_events

    events = turn("a", "a1", 0, 1.0)
    path_text = "\n".join([json.dumps(e) for e in events] + ["{not json", ""])
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "trace.jsonl"
        p.write_text(path_text)
        loaded = load_events(p)
        assert len(loaded) == len(events) and load_events(pathlib.Path(d) / "missing.jsonl") == []
    assert "text" not in compute_live_metrics(events, [], since=None).model_dump()
