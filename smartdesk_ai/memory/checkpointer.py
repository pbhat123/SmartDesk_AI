"""SQLite-backed LangGraph checkpointer: short-term memory and the approval-gate pause
(FR-H10, FR-G12).

The saver is built by hand, not with `SqliteSaver.from_conn_string`, which closes the
connection on exit; the CLI and Streamlit keep one open for the whole run (PLAN 12.1).
Every Pydantic model stored in graph state is allow-listed with the serializer, or
checkpoints stop loading once LangGraph enforces it.
"""
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel


def make_checkpointer(
    path: Path | str, state_types: Iterable[type[BaseModel]] = ()
) -> SqliteSaver:
    """Open (and create) the checkpoint database. The caller closes `saver.conn`."""
    if not str(path).strip():
        raise ValueError("checkpoint database path must not be empty")
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    allowed = [(cls.__module__, cls.__name__) for cls in state_types]
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=allowed))
