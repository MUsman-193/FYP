"""SQLite-backed users, analysis history, and aggregate statistics."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_APP_DIR = Path(__file__).resolve().parents[2] / "data"
_DB_PATH = _APP_DIR / "workbench.sqlite3"
_EXPORT_DIR = _APP_DIR / "exports"

_pbkdf2_iters = 260_000

# Default admin generated on first application start (fresh database).
# NOTE: This is intentionally hard-coded per project requirement.
_DEFAULT_ADMIN_USERNAME = "admin"
_DEFAULT_ADMIN_PASSWORD = "Bc220411312@"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _pbkdf2_iters,
    )


def _verify_password(password: str, salt: bytes, expected_hash: bytes) -> bool:
    return secrets.compare_digest(_hash_password(password, salt), expected_hash)


@dataclass(frozen=True)
class UserRecord:
    id: int
    username: str
    is_admin: bool
    is_blocked: bool


class WorkbenchStore:
    """Thread-safe SQLite access."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _DB_PATH
        self._lock = threading.Lock()
        _APP_DIR.mkdir(parents=True, exist_ok=True)
        _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._ensure_default_admin()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash BLOB NOT NULL,
                    salt BLOB NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    is_blocked INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS analysis_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    run_type TEXT NOT NULL,
                    source_filename TEXT,
                    row_count INTEGER NOT NULL DEFAULT 0,
                    toxic_count INTEGER NOT NULL DEFAULT 0,
                    non_toxic_count INTEGER NOT NULL DEFAULT 0,
                    summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    results_path TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                );

                CREATE INDEX IF NOT EXISTS idx_runs_user ON analysis_runs(user_id);
                CREATE INDEX IF NOT EXISTS idx_runs_created ON analysis_runs(created_at);
                """
            )

    def _ensure_default_admin(self) -> None:
        """
        Ensure a default admin exists on first run.

        Rule: registration never grants admin. Instead, if the database is empty,
        create a known admin account when the application starts.
        """
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
            count = int(row["c"]) if row else 0
            if count > 0:
                return

            u = _DEFAULT_ADMIN_USERNAME.strip().lower()
            salt = secrets.token_bytes(16)
            pw_hash = _hash_password(_DEFAULT_ADMIN_PASSWORD, salt)
            conn.execute(
                "INSERT INTO users (username, password_hash, salt, is_admin, is_blocked, created_at) "
                "VALUES (?, ?, ?, 1, 0, ?)",
                (u, pw_hash, salt, _utc_now_iso()),
            )

    def _validate_password(self, password: str) -> tuple[bool, str]:
        if len(password) < 6:
            return False, "Password must be at least 6 characters."
        if not any(c.islower() for c in password):
            return False, "Password must include at least one lowercase letter."
        if not any(c.isupper() for c in password):
            return False, "Password must include at least one uppercase letter."
        if not any((not c.isalnum()) for c in password):
            return False, "Password must include at least one special character."
        return True, ""

    def user_count(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
            return int(row["c"]) if row else 0

    def register(self, username: str, password: str) -> tuple[bool, str]:
        u = username.strip().lower()
        if len(u) < 3:
            return False, "Username must be at least 3 characters."
        ok, msg = self._validate_password(password)
        if not ok:
            return False, msg
        salt = secrets.token_bytes(16)
        pw_hash = _hash_password(password, salt)
        with self._lock, self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO users (username, password_hash, salt, is_admin, is_blocked, created_at) "
                    "VALUES (?, ?, ?, ?, 0, ?)",
                    (u, pw_hash, salt, 0, _utc_now_iso()),
                )
            except sqlite3.IntegrityError:
                return False, "That username is already taken."
        return True, "Account created. You can sign in now."

    def authenticate(self, username: str, password: str) -> UserRecord | None:
        u = username.strip().lower()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, salt, is_admin, is_blocked FROM users WHERE username = ?",
                (u,),
            ).fetchone()
        if row is None:
            return None
        if int(row["is_blocked"]):
            return None
        if not _verify_password(password, row["salt"], row["password_hash"]):
            return None
        return UserRecord(
            id=int(row["id"]),
            username=str(row["username"]),
            is_admin=bool(int(row["is_admin"])),
            is_blocked=bool(int(row["is_blocked"])),
        )

    def list_users(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, username, is_admin, is_blocked, created_at FROM users ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def set_blocked(self, user_id: int, blocked: bool) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE users SET is_blocked = ? WHERE id = ?", (1 if blocked else 0, user_id))

    def delete_user(self, user_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM analysis_runs WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))

    def insert_run(
        self,
        *,
        user_id: int,
        run_type: str,
        source_filename: str | None,
        row_count: int,
        toxic_count: int,
        non_toxic_count: int,
        summary: str,
        results_path: str | None = None,
    ) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO analysis_runs (user_id, run_type, source_filename, row_count, toxic_count, "
                "non_toxic_count, summary, created_at, results_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    run_type,
                    source_filename,
                    row_count,
                    toxic_count,
                    non_toxic_count,
                    summary,
                    _utc_now_iso(),
                    results_path,
                ),
            )
            return int(cur.lastrowid)

    def update_run_results_path(self, run_id: int, results_path: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE analysis_runs SET results_path = ? WHERE id = ?", (results_path, run_id))

    def list_runs_for_user(self, user_id: int, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, run_type, source_filename, row_count, toxic_count, non_toxic_count, summary, created_at, results_path "
                "FROM analysis_runs WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_runs_for_user(self, user_id: int) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM analysis_runs WHERE user_id = ?", (user_id,))
            return int(cur.rowcount)

    def get_run(self, run_id: int, user_id: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_runs WHERE id = ? AND user_id = ?",
                (run_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def admin_aggregate_stats(self) -> dict[str, Any]:
        """Totals across all users for admin dashboards."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT "
                "COALESCE(SUM(row_count), 0) AS comments, "
                "COALESCE(SUM(toxic_count), 0) AS toxic, "
                "COALESCE(SUM(non_toxic_count), 0) AS nontoxic "
                "FROM analysis_runs"
            ).fetchone()
            by_day = conn.execute(
                "SELECT substr(created_at, 1, 10) AS day, SUM(row_count) AS c "
                "FROM analysis_runs GROUP BY day ORDER BY day DESC LIMIT 30"
            ).fetchall()
        comments = int(row["comments"]) if row else 0
        toxic = int(row["toxic"]) if row else 0
        nontoxic = int(row["nontoxic"]) if row else 0
        timeline = [(str(r["day"]), int(r["c"])) for r in reversed(by_day)]
        return {
            "comments_analyzed": comments,
            "toxic_comments": toxic,
            "non_toxic_comments": nontoxic,
            "timeline_days": timeline,
        }


_store: WorkbenchStore | None = None


def get_store() -> WorkbenchStore:
    global _store
    if _store is None:
        _store = WorkbenchStore()
    return _store


def exports_dir() -> Path:
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return _EXPORT_DIR
