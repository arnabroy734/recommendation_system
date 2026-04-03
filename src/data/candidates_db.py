"""
src/data/candidates_db.py
--------------------------
Abstraction layer for all read/write operations on candidates.db (SQLite).

Write ops  — called from deploy/generate_candidates.py
Read ops   — called from serving layer
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/candidates.db")


class CandidatesDB:

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    # -----------------------------------------------------------------------
    # Connection management
    # -----------------------------------------------------------------------

    def connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        logger.info(f"Connected to candidates DB → {self.db_path.resolve()}")

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError(
                "CandidatesDB not connected. "
                "Call connect() or use as context manager."
            )
        return self._conn

    # -----------------------------------------------------------------------
    # Write ops
    # -----------------------------------------------------------------------

    def init_tables(self):
        """Drop and recreate all tables. Called once per candidate generation run."""
        self.conn.executescript("""
            DROP TABLE IF EXISTS candidates;
            DROP TABLE IF EXISTS global_candidates;

            CREATE TABLE candidates (
                user_id   INTEGER NOT NULL,
                movie_id  INTEGER NOT NULL,
                mf_score  REAL,
                rank      INTEGER,
                PRIMARY KEY (user_id, movie_id)
            );

            CREATE TABLE global_candidates (
                movie_id  INTEGER PRIMARY KEY,
                rank      INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_candidates_user
                ON candidates(user_id);
        """)
        self.conn.commit()
        logger.info("Tables initialised: candidates, global_candidates")

    def insert_candidates(self, rows: list):
        """rows: list of (user_id, movie_id, mf_score, rank)"""
        self.conn.executemany(
            "INSERT OR REPLACE INTO candidates(user_id, movie_id, mf_score, rank) "
            "VALUES (?, ?, ?, ?)",
            rows
        )
        self.conn.commit()

    def insert_global_candidates(self, rows: list):
        """rows: list of (movie_id, rank)"""
        self.conn.executemany(
            "INSERT OR REPLACE INTO global_candidates(movie_id, rank) VALUES (?, ?)",
            rows
        )
        self.conn.commit()
        logger.info(f"Inserted {len(rows)} global popular candidates")

    # -----------------------------------------------------------------------
    # Read ops
    # -----------------------------------------------------------------------

    def user_exists(self, user_id: int) -> bool:
        """Returns True if the user has candidates in the DB."""
        row = self.conn.execute(
            "SELECT 1 FROM candidates WHERE user_id = ? LIMIT 1",
            (user_id,)
        ).fetchone()
        return row is not None

    def get_candidates(self, user_id: int, top_n: int = 500) -> list[dict]:
        """
        Fetch top-N candidates for a known user, ordered by rank.
        Falls back to global popular if user not found.
        """
        if not self.user_exists(user_id):
            logger.info(
                f"user_id={user_id} not in candidates — returning global popular"
            )
            return self.get_global_candidates(top_n)

        rows = self.conn.execute(
            "SELECT movie_id, mf_score, rank "
            "FROM candidates "
            "WHERE user_id = ? "
            "ORDER BY rank ASC LIMIT ?",
            (user_id, top_n)
        ).fetchall()

        return [dict(r) for r in rows]

    def get_global_candidates(self, top_n: int = 500) -> list[dict]:
        """Fetch global popular fallback candidates, ordered by rank."""
        rows = self.conn.execute(
            "SELECT movie_id, rank "
            "FROM global_candidates "
            "ORDER BY rank ASC LIMIT ?",
            (top_n,)
        ).fetchall()

        return [dict(r) for r in rows]