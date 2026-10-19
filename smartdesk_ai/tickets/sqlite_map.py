"""SQLite store for the ticket flow: the create-request log (FR-G6) and the email-to-issue map (FR-C10).

The request log holds only request IDs and issue numbers, no email. It is written before the
GitHub call, so a crash or a lost response leaves a trace to recover from. The map is a data
store, not a log (FR-G11): it holds the raw email, lower-cased, because lookup needs it.
"""
import sqlite3
import threading
from pathlib import Path


class TicketStore:
    """One connection shared by every thread: Streamlit runs each rerun on a different thread and keeps one app
    context for the whole process, so the connection is opened with `check_same_thread=False` and every operation
    runs under a lock (SQLite connections are not safe for two threads at once)."""

    def __init__(self, path: Path | str) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS create_requests ("
                "request_id TEXT PRIMARY KEY, status TEXT NOT NULL, issue_number INTEGER)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS email_tickets ("
                "email TEXT NOT NULL, issue_number INTEGER NOT NULL, PRIMARY KEY (email, issue_number))"
            )

    def begin(self, request_id: str) -> None:
        """Record the intent to create, before the GitHub call. Idempotent."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO create_requests (request_id, status) VALUES (?, 'pending')",
                (request_id,),
            )

    def complete(self, request_id: str, issue_number: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE create_requests SET status = 'done', issue_number = ? WHERE request_id = ?",
                (issue_number, request_id),
            )

    def get(self, request_id: str) -> tuple[str, int | None] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT status, issue_number FROM create_requests WHERE request_id = ?", (request_id,)
            ).fetchone()
        return None if row is None else (row[0], row[1])

    def link_email(self, email: str, issue_number: int) -> None:
        """Remember that this employee has this issue. Safe to repeat."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO email_tickets (email, issue_number) VALUES (?, ?)",
                (_key(email), issue_number),
            )

    def numbers_for(self, email: str) -> list[int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT issue_number FROM email_tickets WHERE email = ? ORDER BY issue_number", (_key(email),)
            ).fetchall()
        return [r[0] for r in rows]


def _key(email: str) -> str:
    return email.strip().lower()
