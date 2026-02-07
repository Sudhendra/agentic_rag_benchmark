"""SQLite-based cache for LLM responses to reduce API costs."""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class SQLiteCache:
    """Thread-safe SQLite cache for LLM API responses.

    This cache is critical for cost management - it prevents redundant
    API calls during development and experimentation.
    """

    def __init__(self, db_path: str = ".cache/llm_cache.db"):
        """Initialize the cache.

        Args:
            db_path: Path to the SQLite database file. Use ":memory:" for testing.
        """
        self.db_path = db_path
        self._local = threading.local()

        # Create cache directory if needed
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize the database schema
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._local.connection

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_created_at ON cache(created_at)
        """)
        conn.commit()

    @staticmethod
    def make_key(data: dict) -> str:
        """Create a deterministic cache key from a dictionary.

        Args:
            data: Dictionary to hash (typically contains model, messages, params)

        Returns:
            SHA256 hash of the serialized data
        """
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        """Retrieve a value from the cache.

        Args:
            key: Cache key (SHA256 hash)

        Returns:
            Cached value if exists, None otherwise
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key (SHA256 hash)
            value: Value to cache (must be JSON serializable)
        """
        conn = self._get_connection()
        serialized = json.dumps(value)
        conn.execute(
            """
            INSERT OR REPLACE INTO cache (key, value, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (key, serialized),
        )
        conn.commit()

    def delete(self, key: str) -> bool:
        """Delete a value from the cache.

        Args:
            key: Cache key to delete

        Returns:
            True if a value was deleted, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> int:
        """Clear all cached values.

        Returns:
            Number of entries cleared
        """
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM cache")
        conn.commit()
        return cursor.rowcount

    def size(self) -> int:
        """Get the number of cached entries.

        Returns:
            Number of entries in the cache
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM cache")
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection
