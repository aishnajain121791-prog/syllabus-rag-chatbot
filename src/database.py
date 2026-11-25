import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "app.db"


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection with foreign keys enabled, commit/rollback, and close it."""
    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create the notes, chat_sessions, and messages tables if they do not exist."""
    with _get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                subject TEXT NOT NULL DEFAULT '',
                uploaded_at TEXT NOT NULL,
                num_pages INTEGER NOT NULL DEFAULT 0,
                num_chunks INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            );
            """
        )


def add_note(
    filename: str,
    subject: str = "",
    num_pages: int = 0,
    num_chunks: int = 0,
    uploaded_at: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Insert a note record and return its generated id."""
    init_db(db_path)
    timestamp = uploaded_at or _utc_now()
    with _get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO notes (filename, subject, uploaded_at, num_pages, num_chunks)
            VALUES (?, ?, ?, ?, ?)
            """,
            (filename, subject, timestamp, num_pages, num_chunks),
        )
        return int(cursor.lastrowid)


def list_notes(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    """Return all uploaded note records ordered newest first."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, filename, subject, uploaded_at, num_pages, num_chunks
            FROM notes
            ORDER BY id DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]


def get_note_by_filename(
    filename: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict | None:
    """Return a note record matching `filename`, or None if not found."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, filename, subject, uploaded_at, num_pages, num_chunks
            FROM notes
            WHERE filename = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (filename,),
        ).fetchone()
        return dict(row) if row else None


def upsert_note(
    filename: str,
    subject: str = "",
    num_pages: int = 0,
    num_chunks: int = 0,
    uploaded_at: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> tuple[int, bool]:
    """Insert a note or update an existing note with the same filename. Returns (id, was_duplicate)."""
    init_db(db_path)
    timestamp = uploaded_at or _utc_now()
    existing = get_note_by_filename(filename, db_path=db_path)
    if existing is not None:
        with _get_connection(db_path) as conn:
            conn.execute(
                """
                UPDATE notes
                SET subject = ?, uploaded_at = ?, num_pages = ?, num_chunks = ?
                WHERE id = ?
                """,
                (subject, timestamp, num_pages, num_chunks, existing["id"]),
            )
        return existing["id"], True

    note_id = add_note(
        filename=filename,
        subject=subject,
        num_pages=num_pages,
        num_chunks=num_chunks,
        uploaded_at=timestamp,
        db_path=db_path,
    )
    return note_id, False


def delete_note(note_id: int, db_path: str | Path = DEFAULT_DB_PATH) -> bool:
    """Delete a note by id. Returns True if a row was deleted."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        return cursor.rowcount > 0


def create_session(
    title: str = "New Chat",
    created_at: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Create a chat session and return its generated id."""
    init_db(db_path)
    timestamp = created_at or _utc_now()
    with _get_connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO chat_sessions (title, created_at) VALUES (?, ?)",
            (title, timestamp),
        )
        return int(cursor.lastrowid)


def list_sessions(db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    """Return all chat sessions ordered newest first."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM chat_sessions ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def add_message(
    session_id: int,
    role: str,
    content: str,
    sources: list[dict] | dict | str | None = None,
    created_at: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Add a message to a chat session and return its generated id."""
    init_db(db_path)
    timestamp = created_at or _utc_now()
    if sources is None:
        sources_json = "[]"
    elif isinstance(sources, str):
        sources_json = sources
    else:
        sources_json = json.dumps(sources, ensure_ascii=False)

    with _get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO messages (session_id, role, content, sources, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, role, content, sources_json, timestamp),
        )
        return int(cursor.lastrowid)


def get_messages(session_id: int, db_path: str | Path = DEFAULT_DB_PATH) -> list[dict]:
    """Return all messages for a chat session in chronological order."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, session_id, role, content, sources, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()

    messages = []
    for row in rows:
        item = dict(row)
        try:
            item["sources"] = json.loads(item["sources"])
        except (TypeError, json.JSONDecodeError):
            pass
        messages.append(item)
    return messages


def delete_session(session_id: int, db_path: str | Path = DEFAULT_DB_PATH) -> bool:
    """Delete a chat session (and cascade-delete its messages). Returns True if deleted."""
    init_db(db_path)
    with _get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0
