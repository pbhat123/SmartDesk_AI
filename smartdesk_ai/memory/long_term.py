"""Long-term memory: what the agent keeps about an employee across sessions (FR-H10, FR-G11).

Per employee it keeps the numbers of their past tickets and the category of the last one, and nothing else. The key is a
salted SHA-256 of the email (`MASK_SALT`), so the file holds no email. A ticket's title and status are never stored: they
are read live from GitHub by number, and no message text reaches this module. Conversation memory is the checkpointer's job.
"""
import sqlite3
import threading
from pathlib import Path

from smartdesk_ai.ops.runtime import salted_digest
from smartdesk_ai.tickets.schemas import Category


def _key(email: str) -> str:
    """A different label from the masked user ID, so a trace ID is not the start of a key in this file."""
    return salted_digest(f"long-term-memory:{email.strip().lower()}")


class LongTermStore:
    """One connection shared by every thread, under a lock, as in `TicketStore` (Streamlit reruns on different threads)."""

    def __init__(self, path: Path | str) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS employee_tickets ("
                "user_key TEXT NOT NULL, issue_number INTEGER NOT NULL, PRIMARY KEY (user_key, issue_number))")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS employee_categories ("
                "user_key TEXT PRIMARY KEY, last_category TEXT NOT NULL CHECK (last_category IN ('IT', 'HR', 'MISC')))")

    def remember(self, email: str, number: int, category: str) -> None:
        """Record that this employee has ticket `number`, filed under `category` (the latest one wins). Safe to repeat."""
        if number < 1:
            raise ValueError("a ticket number is a positive integer")
        try:
            category = Category(category).value
        except ValueError:
            raise ValueError("the category must be IT, HR or MISC") from None
        key = _key(email)
        with self._lock, self._conn:
            self._conn.execute("INSERT OR IGNORE INTO employee_tickets (user_key, issue_number) VALUES (?, ?)", (key, number))
            self._conn.execute("INSERT OR REPLACE INTO employee_categories (user_key, last_category) VALUES (?, ?)", (key, category))

    def ticket_numbers(self, email: str) -> list[int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT issue_number FROM employee_tickets WHERE user_key = ? ORDER BY issue_number", (_key(email),)).fetchall()
        return [r[0] for r in rows]

    def last_category(self, email: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT last_category FROM employee_categories WHERE user_key = ?", (_key(email),)).fetchone()
        return None if row is None else row[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
