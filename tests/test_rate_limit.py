"""T5.14: rolling 1-hour rate limits (FR-H13): 100 questions, 40 ticket creations, 40 ticket lookups, three
independent counters in SQLite, keyed by session (lookups also by email). Fake clock, no real waiting."""
import pytest

from smartdesk_ai.agent.chat_service import ChatService
from smartdesk_ai.ops.rate_limit import RateLimiter, usage_limit_message
from smartdesk_ai.ops.runtime import RequestContext, Services, request_scope
from smartdesk_ai.tools.executor import Outcome, ToolExecutor
from smartdesk_ai.tools.registry import ToolRegistry

EMAIL = "jane.roe@example.com"
WINDOW = 3600


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def limiter(settings, path=":memory:", clock=None, bypass=False):
    cfg = settings.rate_limits.model_copy(update={"bypass": bypass})
    return RateLimiter(cfg, path, clock=clock or Clock())


def use(lim, counter, times, session="s1", email=None):
    for _ in range(times):
        assert lim.hit(counter, session, email).allowed


def test_the_101st_question_is_refused_with_the_real_wait(settings):
    clock = Clock()
    lim = limiter(settings, clock=clock)
    use(lim, "questions", 100)
    clock.now += 600  # ten minutes after the first 100
    decision = lim.hit("questions", "s1")
    assert not decision.allowed and decision.wait_seconds == WINDOW - 600
    assert decision.message == "Usage limit reached. Please try again in 50 minutes."


def test_the_41st_creation_and_the_41st_lookup_are_refused(settings):
    lim = limiter(settings)
    use(lim, "ticket_creations", 40)
    assert not lim.hit("ticket_creations", "s1").allowed
    use(lim, "ticket_lookups", 40, email=EMAIL)
    assert not lim.hit("ticket_lookups", "s1", EMAIL).allowed


def test_the_three_counters_are_independent(settings):
    lim = limiter(settings)
    use(lim, "questions", 100)
    assert not lim.hit("questions", "s1").allowed
    assert lim.hit("ticket_creations", "s1").allowed and lim.hit("ticket_lookups", "s1", EMAIL).allowed
    use(lim, "ticket_creations", 39)
    assert not lim.hit("ticket_creations", "s1").allowed
    assert lim.hit("ticket_lookups", "s1", EMAIL).allowed


def test_a_new_session_has_its_own_question_counter(settings):
    lim = limiter(settings)
    use(lim, "questions", 100, session="a")
    assert not lim.hit("questions", "a").allowed and lim.hit("questions", "b").allowed


def test_lookups_are_limited_per_session_and_per_email_and_both_must_be_within_limit(settings):
    lim = limiter(settings)
    use(lim, "ticket_lookups", 40, session="a", email=EMAIL)
    assert not lim.hit("ticket_lookups", "a", "other@example.com").allowed  # the session is full
    assert not lim.hit("ticket_lookups", "b", EMAIL).allowed  # the email is full, across sessions
    assert lim.hit("ticket_lookups", "b", "other@example.com").allowed  # a fresh session and email


def test_the_window_rolls(settings):
    clock = Clock()
    lim = limiter(settings, clock=clock)
    use(lim, "questions", 50)
    clock.now += 1800
    use(lim, "questions", 50)
    assert not lim.hit("questions", "s1").allowed  # 100 inside the last hour
    clock.now += 1801  # the first 50 are now more than an hour old
    assert lim.hit("questions", "s1").allowed
    assert lim.count("questions", "s1") == 51


def test_the_wait_is_the_time_until_the_oldest_use_leaves_the_window(settings):
    clock = Clock()
    lim = limiter(settings, clock=clock)
    lim.hit("ticket_creations", "s1")
    clock.now += 100
    use(lim, "ticket_creations", 39)
    clock.now += 200
    decision = lim.hit("ticket_creations", "s1")
    assert decision.wait_seconds == WINDOW - 300 and "55 minutes" in decision.message


def test_a_short_wait_reads_one_minute():
    assert usage_limit_message(5) == "Usage limit reached. Please try again in 1 minute."
    assert usage_limit_message(61) == "Usage limit reached. Please try again in 2 minutes."


def test_counters_survive_a_restart(settings, tmp_path):
    clock = Clock()
    use(limiter(settings, tmp_path / "ops.db", clock), "questions", 100)
    again = limiter(settings, tmp_path / "ops.db", clock)
    assert not again.hit("questions", "s1").allowed


def test_the_bypass_flag_turns_the_limits_off(settings):
    lim = limiter(settings, bypass=True)
    for _ in range(150):
        assert lim.hit("questions", "s1").allowed
    assert lim.count("questions", "s1") == 0


def test_no_raw_email_is_stored(settings, tmp_path):
    lim = limiter(settings, tmp_path / "ops.db")
    lim.hit("ticket_lookups", "s1", EMAIL)
    assert EMAIL.encode() not in (tmp_path / "ops.db").read_bytes()


# --- wiring -----------------------------------------------------------------------------

def test_the_101st_message_gets_the_usage_limit_reply_and_never_reaches_the_graph(settings):
    seen = []
    service = ChatService(settings, responder=lambda sid, text: seen.append(text) or "ok",
                          services=Services(limiter=limiter(settings)))
    for i in range(100):
        assert service.handle("s1", "hi" if i % 2 else "hlep me").text == "ok"  # small talk and typos count too
    reply = service.handle("s1", "one more")
    assert len(seen) == 100 and reply.text.startswith("Usage limit reached. Please try again in ")
    assert reply.debug["edge"] == "rate_limited"
    assert service.handle("s2", "fresh session").text == "ok"


def tool_executor(settings, lim):
    calls = []
    registry = ToolRegistry()
    registry.register("create_ticket", lambda args: calls.append("create") or type("T", (), {"number": len(calls)})())
    registry.register("get_tickets", lambda args: calls.append("get") or [])
    return ToolExecutor(registry, settings), calls, Services(limiter=lim)


CREATE = {"title": "Dock", "description": "Dock broken", "category": "IT", "email": EMAIL}


def test_the_41st_creation_is_refused_before_the_tool_runs(settings):
    executor, calls, services = tool_executor(settings, limiter(settings))
    with request_scope(RequestContext(session_id="s1", request_id="r"), services):
        for _ in range(40):
            assert executor.execute("create_ticket", CREATE, agent="ticket_agent", confirmed=True).outcome is Outcome.SUCCESS
        result = executor.execute("create_ticket", CREATE, agent="ticket_agent", confirmed=True)
    assert result.outcome is Outcome.RATE_LIMITED and result.reply.startswith("Usage limit reached.")
    assert calls.count("create") == 40


def test_only_successful_creations_count(settings):
    lim = limiter(settings)
    registry = ToolRegistry()
    registry.register("create_ticket", lambda args: (_ for _ in ()).throw(ValueError("boom")))
    executor = ToolExecutor(registry, settings)
    with request_scope(RequestContext(session_id="s1", request_id="r"), Services(limiter=lim)):
        for _ in range(45):
            assert executor.execute("create_ticket", CREATE, agent="ticket_agent", confirmed=True).outcome is Outcome.ERROR
    assert lim.count("ticket_creations", "s1") == 0


def test_the_41st_status_lookup_is_refused_but_the_duplicate_check_is_not_a_lookup(settings):
    executor, calls, services = tool_executor(settings, limiter(settings))
    with request_scope(RequestContext(session_id="s1", request_id="r"), services):
        for _ in range(100):  # the ticket agent's own duplicate check never counts
            assert executor.execute("get_tickets", {"email": EMAIL}, agent="ticket_agent").outcome is Outcome.SUCCESS
        for _ in range(40):
            assert executor.execute("get_tickets", {"email": EMAIL}, agent="status_flow").outcome is Outcome.SUCCESS
        result = executor.execute("get_tickets", {"email": EMAIL}, agent="status_flow")
    assert result.outcome is Outcome.RATE_LIMITED and "Usage limit reached" in result.reply


def test_outside_a_request_scope_the_executor_is_not_limited(settings):
    executor, calls, _ = tool_executor(settings, limiter(settings))
    for _ in range(50):
        executor.execute("create_ticket", CREATE, agent="ticket_agent", confirmed=True)
    assert calls.count("create") == 50
