import os
import sqlite3
from langgraph.checkpoint.memory import MemorySaver
from core.logger import get_logger
from config import settings

logger = get_logger(__name__)

# LangGraph checkpointer (MemorySaver for now — graph state checkpointing)
_checkpointer = None

# SQLite-backed conversation history, under the shared storage root so it
# survives a restart when DATA_DIR points at a mounted volume.
_db_path = settings.resolved_db_path


def _get_db_connection() -> sqlite3.Connection:
    """Get a SQLite connection, creating the DB and table if needed."""
    os.makedirs(os.path.dirname(_db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(_db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            query TEXT NOT NULL,
            report TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_thread_id ON conversation_history(thread_id)
    """)
    conn.commit()
    return conn


def get_checkpointer() -> MemorySaver:
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
        logger.info("LangGraph checkpointer initialized (MemorySaver)")
    return _checkpointer


def get_conversation_history(thread_id: str) -> list:
    """Retrieve conversation history for a thread from SQLite."""
    conn = _get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT query, report FROM conversation_history "
            "WHERE thread_id = ? ORDER BY created_at ASC LIMIT 20",
            (thread_id,),
        )
        return [{"query": row[0], "report": row[1]} for row in cursor.fetchall()]
    finally:
        conn.close()


def save_to_history(thread_id: str, query: str, report: str):
    """Save a query-report exchange to SQLite. Keeps last 20 per thread."""
    conn = _get_db_connection()
    try:
        conn.execute(
            "INSERT INTO conversation_history (thread_id, query, report) VALUES (?, ?, ?)",
            (thread_id, query, report),
        )
        # Keep only the last 20 exchanges per thread
        conn.execute("""
            DELETE FROM conversation_history
            WHERE thread_id = ? AND id NOT IN (
                SELECT id FROM conversation_history
                WHERE thread_id = ?
                ORDER BY created_at DESC
                LIMIT 20
            )
        """, (thread_id, thread_id))
        conn.commit()
        logger.info(f"Saved exchange to history | thread_id={thread_id}")
    finally:
        conn.close()


def clear_thread(thread_id: str):
    """Delete all conversation history for a thread."""
    conn = _get_db_connection()
    try:
        conn.execute("DELETE FROM conversation_history WHERE thread_id = ?", (thread_id,))
        conn.commit()
        logger.info(f"Cleared history | thread_id={thread_id}")
    finally:
        conn.close()