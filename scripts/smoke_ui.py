#!/usr/bin/env python
"""End-to-end smoke test of the real Streamlit app in a real browser (mock models, mock tickets, no network).

    python scripts/smoke_ui.py [--headed] [--timeout SECONDS] [--keep]

It starts `streamlit run smartdesk_ai/ui/app.py` headless on a free port with a temp data folder, a temp spend ledger,
a temp working directory (so no `.env` is found) and both modes forced to mock, then drives it with Playwright:

  1. a KB question gets an answer with its sources line and a retrieval-debug panel
  2. a question the KB cannot answer leads to a ticket offer, an email, a draft, an edit at the approval gate, "yes",
     and a reply "Ticket #N (mock)" with no link
  3. a status check shows that ticket
  4. the Monitoring page loads with no exception, and the HITL counters read 1 edit, 1 approval, 0 rejections

and finally checks that the server log holds no traceback. It exists because unit tests missed a bug that only a real
server shows: Streamlit runs every rerun on its own thread (SQLite "objects created in a thread" error at the gate).

Why a second browser session runs alongside: the operating system reuses the ids of finished threads, so with one
session a rerun often lands on the very id that built the app context and the bug stays hidden (it showed in about half
of the runs against the buggy code). A second session that fires a status lookup of a demo ticket at the same moment as
each step of the main flow guarantees two live script threads at once, and two live threads cannot share an id, so at
least one of them is foreign to the thread that built the context.
Exit code 0 only if every step passes. Never calls OpenAI, GitHub or any other paid or public service.
"""
import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "smartdesk_ai" / "ui" / "app.py"

KB_QUESTION = "How do I reset my forgotten password?"
GAP_QUESTION = "How do I register my tractor for the office parking lot?"  # not in the KB: the mock escalates it
EMAIL = "priya.k@smartdesk.example"
DEMO_EMAIL = "one.ticket@smartdesk.example"  # a preloaded mock employee with exactly one ticket (tickets/seed.py)
NEW_TITLE = "Tractor parking permit"
PARTIAL_QUESTION = "How do I reset my password, and where can I park my car at the Bengaluru office?"  # one answer, one gap
COMPOUND_QUESTION = "How do I reset my password, and how many casual leave days do I get?"  # one IT part, one HR part

CHAT_INPUT = '[data-testid="stChatInputTextArea"]'
MESSAGE = '[data-testid="stChatMessage"]'
EXCEPTION = '[data-testid="stException"]'


class StepFailed(AssertionError):
    pass


def check(condition: object, message: str) -> None:
    if not condition:
        raise StepFailed(message)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Server:
    """The real app in a subprocess, isolated from the repo's data/, ledger and .env."""

    def __init__(self, root: Path, port: int, work: Path) -> None:
        self.root, self.port, self.work = root, port, work
        self.log_path = work / "server.log"
        self.proc: subprocess.Popen | None = None

    def start(self, timeout: float) -> None:
        env = {**os.environ,
               "SMARTDESK_LLM_MODE": "mock", "SMARTDESK_TICKETS_MODE": "mock",
               "SMARTDESK_DATA_DIR": str(self.work / "data"),
               "SMARTDESK_LEDGER_PATH": str(self.work / "ledger.jsonl"),
               "PYTHONDONTWRITEBYTECODE": "1"}
        for secret in ("OPENAI_API_KEY", "GITHUB_TOKEN", "TICKETS_REPO", "LANGSMITH_API_KEY", "LANGFUSE_SECRET_KEY",
                       "LANGFUSE_PUBLIC_KEY", "REDIS_URL"):
            env.pop(secret, None)
        cmd = [sys.executable, "-m", "streamlit", "run", str(self.root / "smartdesk_ai" / "ui" / "app.py"),
               "--server.headless", "true", "--server.port", str(self.port), "--server.address", "127.0.0.1",
               "--browser.gatherUsageStats", "false", "--server.fileWatcherType", "none"]
        shutil.copytree(self.root / ".streamlit", self.work / ".streamlit", dirs_exist_ok=True)  # the theme (cwd-relative)
        log = self.log_path.open("w")
        # cwd is the temp folder: the app looks for `.env` in the working directory and must not find the real one.
        self.proc = subprocess.Popen(cmd, cwd=self.work, env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise StepFailed(f"the server exited early (code {self.proc.returncode}); see {self.log_path}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/_stcore/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(0.3)
        raise StepFailed(f"the server did not become healthy within {timeout:.0f}s; see {self.log_path}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def log_text(self) -> str:
        return self.log_path.read_text(errors="replace") if self.log_path.exists() else ""


class Sidekick:
    """A second, independent session (its own websocket and thread) that looks up a demo ticket at the same moment as
    each main-flow step, and keeps every reply for the final check."""

    def __init__(self, page, timeout_ms: int) -> None:
        self.page, self.timeout = page, timeout_ms
        self.sent = 0  # status lookups fired after setup

    def _send(self, text: str, wait: bool) -> None:
        before = self.page.locator(MESSAGE).count()
        box = self.page.locator(CHAT_INPUT)
        box.fill(text)
        box.press("Enter")
        if wait:
            self.wait_for(before + 2)

    def wait_for(self, count: int) -> None:
        deadline = time.time() + self.timeout / 1000
        while time.time() < deadline:
            if self.page.locator(MESSAGE).count() >= count:
                texts = self.page.locator(MESSAGE).all_inner_texts()
                if all(t.strip() not in ("", "smart_toy") and "Thinking" not in t for t in texts):
                    return
            time.sleep(0.1)
        raise StepFailed("the second session did not answer in time")

    def setup(self) -> None:
        self._send("any update on my ticket?", wait=True)
        self._send(DEMO_EMAIL, wait=True)  # the session now knows who it is

    def poke(self) -> None:
        """Fire a status lookup without waiting; the previous one must have been answered first."""
        self.wait_for(2 * (2 + self.sent))
        self._send("any update on my ticket?", wait=False)
        self.sent += 1

    def verify(self) -> None:
        self.wait_for(2 * (2 + self.sent))
        replies = self.page.locator(MESSAGE).all_inner_texts()[5::2]  # the answers after the two setup exchanges
        check(len(replies) == self.sent and self.sent > 0, f"expected {self.sent} status replies, saw {len(replies)}")
        bad = [r for r in replies if "(mock)" not in r or "Status:" not in r]
        if bad:
            raise StepFailed(f"{len(bad)} of {len(replies)} status lookups in the second session failed, for example: {bad[0][:200]!r}")
        check(self.page.locator(EXCEPTION).count() == 0, "the second session shows an exception")


class Browser:
    def __init__(self, page, timeout_ms: int, sidekick: Sidekick | None = None) -> None:
        self.page, self.timeout, self.sidekick = page, timeout_ms, sidekick

    def no_exception(self, where: str) -> None:
        shown = self.page.locator(EXCEPTION)
        if shown.count():
            raise StepFailed(f"{where}: the page shows an exception: {shown.first.inner_text()[:300]!r}")

    def messages(self) -> int:
        return self.page.locator(MESSAGE).count()

    def settled_reply(self) -> str | None:
        """The last message once it is real text: not just the avatar's icon name, not the spinner, and unchanged
        for a moment (the reply is drawn after the model call returns)."""
        def read() -> str:
            return self.page.locator(MESSAGE).last.inner_text().strip()

        first = read()
        if not first or first == "smart_toy" or "Thinking" in first:
            return None
        time.sleep(0.4)
        second = read()
        return second if second == first else None

    def send(self, text: str) -> str:
        """Type a message, wait for the reply, return the reply text. Fails fast on a page exception."""
        before = self.messages()
        box = self.page.locator(CHAT_INPUT)
        box.fill(text)
        if self.sidekick is not None:
            self.sidekick.poke()  # the second session's lookup starts together with this step
        box.press("Enter")
        deadline = time.time() + self.timeout / 1000
        while time.time() < deadline:
            self.no_exception(f"after sending {text!r}")
            if self.messages() >= before + 2:
                reply = self.settled_reply()
                if reply:
                    return reply
            time.sleep(0.2)
        shown = [m.strip()[:900] for m in self.page.locator(MESSAGE).all_inner_texts()]
        raise StepFailed(f"no reply to {text!r} within {self.timeout / 1000:.0f}s (messages on the page: {shown})")


def step_kb_question(b: Browser) -> None:
    reply = b.send(KB_QUESTION)
    check("Sources" in reply and "IT-PWD-01" in reply, f"expected an answer with the source chip IT-PWD-01, got {reply[:200]!r}")
    check(b.page.get_by_text("Retrieval debug").count() >= 1, "the retrieval-debug panel is missing")
    sidebar = b.page.locator('[data-testid="stSidebar"]').inner_text()
    check("LLM" in sidebar and "Tickets" in sidebar, f"the sidebar lost the mode pills: {sidebar[:200]!r}")
    check("Degraded" not in sidebar and "Mock tickets live in memory" not in sidebar,
          f"the sidebar must not show degraded notices or the mock-tickets note: {sidebar[:300]!r}")


def step_repeated_kb_question(b: Browser, work: Path) -> None:
    """The same KB question again is served by the answer cache: same answer, and the panel says exact hit."""
    reply = b.send(KB_QUESTION)
    check("Sources" in reply and "IT-PWD-01" in reply, f"expected the same answer with sources, got {reply[:200]!r}")
    b.page.locator('[data-testid="stExpander"]').last.locator("summary").click()
    b.page.get_by_text("Cache: hit (exact)").first.wait_for(timeout=b.timeout)
    log = work / "data" / "cache.jsonl"
    events = [json.loads(line)["event"] for line in log.read_text().splitlines()] if log.exists() else []
    events = [e for e in events if e != "clear"]  # the start-up ingest of a fresh data folder clears the cache once
    check(events == ["miss", "hit_exact"], f"expected the cache log to show a miss then an exact hit, got {events}")


def step_ticket_flow(b: Browser) -> int:
    offer = b.send(GAP_QUESTION)
    check("ticket" in offer.lower(), f"expected a ticket offer for a question the KB cannot answer, got {offer[:200]!r}")
    asks = b.send("yes")
    check("email" in asks.lower(), f"expected a request for the work email, got {asks[:200]!r}")
    draft = b.send(EMAIL)
    check("ready to create" in draft and "Waiting for your OK" in draft, f"expected a ticket draft, got {draft[:200]!r}")
    edited = b.send(f"change the title to {NEW_TITLE}")
    check(re.search(rf"Title\s+{NEW_TITLE}", edited) and "Waiting for your OK" in edited,
          f"expected the draft with the new title, got {edited[:200]!r}")
    created = b.send("yes")
    found = re.search(r"Ticket #(\d+) \(mock\)", created)
    check(found, f"expected 'Ticket #N (mock)', got {created[:200]!r}")
    check("http" not in created and "github.com" not in created, f"a mock ticket must have no link: {created[:200]!r}")
    return int(found.group(1))


def step_status(b: Browser, number: int) -> None:
    reply = b.send("any update on my ticket?")
    if "email" in reply.lower():
        reply = b.send(EMAIL)
    check(f"Ticket #{number} (mock)" in reply and "Status:" in reply, f"expected the status of ticket #{number}, got {reply[:200]!r}")
    check("http" not in reply, f"a mock ticket must have no link: {reply[:200]!r}")


def step_monitoring(b: Browser, work: Path) -> None:
    b.page.get_by_role("link", name="Monitoring").first.click()
    b.page.get_by_text("Tool stats").first.wait_for(timeout=b.timeout)
    b.page.get_by_text("Cost by agent").first.wait_for(timeout=b.timeout)
    check({"orchestrator", "it_worker"} <= span_agents(work), "expected agent spans for the orchestrator and a worker by now")
    b.page.get_by_text("Human checks at the ticket approval gate").first.wait_for(timeout=b.timeout)
    b.page.locator('[data-testid="stMetricLabel"]', has_text="Approvals").first.wait_for(timeout=b.timeout)  # the headings draw before the data
    b.page.locator('[data-testid="stMetricLabel"]', has_text="Exact hits").first.wait_for(timeout=b.timeout)
    b.no_exception("on the Monitoring page")
    metrics = {}
    for tile in b.page.locator('[data-testid="stMetric"]').all():
        label = tile.locator('[data-testid="stMetricLabel"]').inner_text().strip()
        metrics[label] = tile.locator('[data-testid="stMetricValue"]').inner_text().strip()
    check(metrics.get("Exact hits") == "1" and metrics.get("Hit rate", "").endswith("%"),
          f"expected the cache tiles with one exact hit, got {metrics}")
    got = (metrics.get("Edits"), metrics.get("Approvals"), metrics.get("Rejections"))
    check(got == ("1", "1", "0"), f"expected HITL counters edits/approvals/rejections = 1/1/0, got {got} (metrics: {metrics})")
    # Streamlit draws tables on a canvas, so their cells are not in the DOM; read the file the table is built from.
    metrics_file = work / "data" / "tool_metrics.jsonl"
    rows = [json.loads(line) for line in metrics_file.read_text().splitlines()] if metrics_file.exists() else []
    created = [r for r in rows if r["tool"] == "create_ticket" and r["outcome"] == "success"]
    check(len(created) == 1, f"expected one successful create_ticket in the tool metrics, found {len(created)}")


def span_agents(work: Path) -> set[str]:
    """The agents that have an `agent_span` record in the app's local trace (FR-I9)."""
    trace = work / "data" / "trace.jsonl"
    events = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    return {e["agent"] for e in events if e.get("kind") == "agent_span"}


def step_recall_in_new_conversation(b: Browser, number: int) -> None:
    """The ticket made in step 3 comes back in a brand-new conversation (new thread, empty chat) from the long-term store, with
    its title and status read live, without the employee naming it (FR-H10, FR-C1)."""
    new_conversation(b)
    reply = b.send("any update on my ticket?")
    if "email" in reply.lower():
        reply = b.send(EMAIL)
    check(f"Ticket #{number} (mock)" in reply and NEW_TITLE in reply and "Status:" in reply,
          f"expected ticket #{number} ('{NEW_TITLE}') recalled in the new conversation, got {reply[:200]!r}")
    check(GAP_QUESTION.lower() not in reply.lower() and "http" not in reply, f"unexpected text in the recall: {reply[:200]!r}")
    check([m for m in b.page.locator(MESSAGE).all_inner_texts() if "tractor" in m.lower() and NEW_TITLE.lower() not in m.lower()] == [],
          "the new conversation shows messages from the first one")


def new_conversation(b: Browser) -> None:
    """Click "New conversation" and wait until the old messages are gone: the greeting can show while the previous
    conversation is still on the page, and a message sent then would be counted against it."""
    b.page.get_by_role("button", name="New conversation").click()
    b.page.get_by_text("Ask an IT or HR question").wait_for(timeout=b.timeout)
    deadline = time.time() + b.timeout / 1000
    while b.messages() and time.time() < deadline:
        time.sleep(0.2)
    check(b.messages() == 0, "the previous conversation is still on the page after New conversation")


def step_compound_question(b: Browser, work: Path) -> None:
    """One message with an IT part and an HR part: one merged answer citing both domains, and a debug panel with one
    table per part and the fan-out and synthesize steps (FR-I4, FR-A12)."""
    b.page.get_by_role("link", name="Chat").first.click()
    b.page.locator(CHAT_INPUT).wait_for(timeout=b.timeout)
    new_conversation(b)
    reply = b.send(COMPOUND_QUESTION)
    check("Sources" in reply and "IT-" in reply and "HR-" in reply,
          f"expected one answer with sources from an IT and an HR entry, got {reply[:300]!r}")
    b.page.locator('[data-testid="stExpander"]').last.locator("summary").click()
    for text in ("Part 1 (it_worker)", "Part 2 (hr_worker)", "fan_out:2", "synthesize"):
        b.page.get_by_text(text).first.wait_for(timeout=b.timeout)
    missing = {"orchestrator", "it_worker", "hr_worker", "synthesizer"} - span_agents(work)
    check(not missing, f"expected an agent span in the trace for each agent, missing {sorted(missing)}")


def step_partial_answer(b: Browser, work: Path) -> None:
    """One answerable part and one the KB cannot answer: the first is answered with a source, the second is named with a
    ticket offer for it only, and the ticket draft covers only that part (FR-I7, FR-B6)."""
    new_conversation(b)
    reply = b.send(PARTIAL_QUESTION)
    check(re.search(r"Sources\s+IT-", reply), f"expected the answerable part answered with an IT source, got {reply[:300]!r}")
    check("couldn't find an answer to this part of your question" in reply and "park my car at the Bengaluru office" in reply,
          f"expected the unanswered part to be named, got {reply[:400]!r}")
    check("support ticket for that part" in reply and "work email" in reply,
          f"expected a ticket offer for that part with the email request, got {reply[:400]!r}")
    draft = b.send(EMAIL)
    check("ready to create" in draft and "Waiting for your OK" in draft, f"expected a ticket draft, got {draft[:300]!r}")
    check("park" in draft.lower() and "password" not in draft.lower(),
          f"expected a draft about the parking part only, got {draft[:400]!r}")
    created = b.send("yes")
    check(re.search(r"Ticket #(\d+) \(mock\)", created), f"expected 'Ticket #N (mock)', got {created[:200]!r}")
    check("ticket_agent" in span_agents(work), "expected an agent span for the ticket agent in the trace")


def run(args: argparse.Namespace) -> int:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    root = Path(args.root).resolve()
    work = Path(tempfile.mkdtemp(prefix="smoke_ui_"))
    server = Server(root, free_port(), work)
    results: list[tuple[str, str]] = []
    failed = False
    try:
        server.start(args.timeout)
        base = f"http://127.0.0.1:{server.port}"
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(channel="chrome", headless=not args.headed)
            except PlaywrightError:
                browser = p.chromium.launch(headless=not args.headed)  # needs: python -m playwright install chromium
            page = browser.new_page()
            page.set_default_timeout(args.timeout * 1000)
            page.goto(base)
            page.locator(CHAT_INPUT).wait_for(timeout=args.timeout * 1000)
            side_page = browser.new_context().new_page()
            side_page.set_default_timeout(args.timeout * 1000)
            side_page.goto(base)
            side_page.locator(CHAT_INPUT).wait_for(timeout=args.timeout * 1000)
            sidekick = Sidekick(side_page, args.timeout * 1000)
            sidekick.setup()
            b = Browser(page, args.timeout * 1000, sidekick)
            number = 0

            def attempt(name: str, fn: Callable[[], object]) -> None:
                nonlocal failed, number
                if failed:
                    results.append((name, "SKIPPED"))
                    return
                try:
                    value = fn()
                    number = value if isinstance(value, int) else number
                    results.append((name, "PASS"))
                except (StepFailed, PlaywrightError, IndexError, KeyError) as error:
                    failed = True
                    results.append((name, f"FAIL: {str(error).strip().splitlines()[0][:400]}"))
                    shot = work / f"failed_{len(results)}.png"
                    try:
                        page.screenshot(path=str(shot), full_page=True)
                        results.append(("  screenshot", str(shot)))
                    except PlaywrightError:
                        pass

            attempt("1 KB question: answer with sources and debug panel", lambda: step_kb_question(b))
            attempt("2 repeated KB question: cache hit (exact) in the debug panel and the log",
                    lambda: step_repeated_kb_question(b, work))
            attempt("3 gap question -> draft -> edit -> yes -> Ticket #N (mock)", lambda: step_ticket_flow(b))
            attempt("4 status check shows the ticket", lambda: step_status(b, number))
            attempt("5 new conversation, same email: the earlier ticket is recalled (title and status live)",
                    lambda: step_recall_in_new_conversation(b, number))
            attempt("6 second session: every overlapping status lookup answered", sidekick.verify)
            attempt("7 Monitoring page: no exception, HITL 1 edit / 1 approval, cache hit rate",
                    lambda: step_monitoring(b, work))
            attempt("8 compound question: one merged answer, one panel table per part",
                    lambda: step_compound_question(b, work))
            attempt("9 partial answer: the gap is named, the ticket covers only that part",
                    lambda: step_partial_answer(b, work))
            browser.close()
    except StepFailed as error:
        failed = True
        results.append(("server", f"FAIL: {error}"))
    finally:
        server.stop()
    log = server.log_text()
    bad = [line for line in log.splitlines() if "Traceback" in line or "ProgrammingError" in line]
    results.append(("10 server log has no traceback", "PASS" if not bad else f"FAIL: {len(bad)} line(s), first: {bad[0][:200]}"))
    failed = failed or bool(bad)
    print("\nsmoke_ui results")
    for name, outcome in results:
        print(f"  {outcome.split(':')[0]:7} {name}" + (f"\n          {outcome}" if outcome.startswith('FAIL') or name.strip() == "screenshot" else ""))
    if failed:
        print(f"\nFAILED. Server log: {server.log_path}")
    elif not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    print("\nOK: every step passed." if not failed else "")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the real Streamlit app in a browser (mock mode).")
    parser.add_argument("--headed", action="store_true", help="show the browser window")
    parser.add_argument("--timeout", type=float, default=60, help="seconds to wait for the server and for each reply")
    parser.add_argument("--keep", action="store_true", help="keep the temp folder (log, data) after a pass")
    parser.add_argument("--root", default=str(ROOT), help="repo root whose app to test (default: this checkout)")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
