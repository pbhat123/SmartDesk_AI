"""SQLite store for the ticket flow: the create-request log (FR-G6).

Holds only request IDs and issue numbers, no email. The email-to-issue map (FR-C10) joins this
file in T3.13. The request ID is written before the GitHub call, so a crash or a lost response
leaves a trace to recover from.
"""
import sqlite3
from pathlib import Path


class TicketStore:
    def __init__(self, path: Path | str) -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS create_requests ("
            "request_id TEXT PRIMARY KEY, status TEXT NOT NULL, issue_number INTEGER)"
        )
        self._conn.commit()

    def begin(self, request_id: str) -> None:
        """Record the intent to create, before the GitHub call. Idempotent."""
        self._conn.execute(
            "INSERT OR IGNORE INTO create_requests (request_id, status) VALUES (?, 'pending')",
            (request_id,),
        )
        self._conn.commit()

    def complete(self, request_id: str, issue_number: int) -> None:
        self._conn.execute(
            "UPDATE create_requests SET status = 'done', issue_number = ? WHERE request_id = ?",
            (issue_number, request_id),
        )
        self._conn.commit()

    def get(self, request_id: str) -> tuple[str, int | None] | None:
        row = self._conn.execute(
            "SELECT status, issue_number FROM create_requests WHERE request_id = ?", (request_id,)
        ).fetchone()
        return None if row is None else (row[0], row[1])
