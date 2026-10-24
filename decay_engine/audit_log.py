"""Append-only audit log for every decay, reinforcement, pruning and GDPR
purge decision made by the system.

Backed by SQLite (zero external services, trivial to inspect with any
sqlite client / the `sqlite3` CLI). Swap the connection string for a
Postgres DSN + a thin psycopg wrapper in production if centralized,
multi-writer audit storage is required; the schema below is intentionally
simple enough to port 1:1.

The log is **append-only by convention**: no method in this module updates
or deletes existing rows, only ``record`` inserts new ones.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AuditAction:
    REINFORCEMENT = "reinforcement"
    DECAY_RECOMPUTED = "decay_recomputed"
    DECAY_MARKED = "decay_marked"
    JURY_REVIEW_REQUESTED = "jury_review_requested"
    JURY_KEPT = "jury_kept"
    JURY_CONFIRMED_FORGET = "jury_confirmed_forget"
    JURY_ESCALATED = "jury_escalated"
    REACTIVATED = "reactivated"
    HARD_DELETED = "hard_deleted"
    GDPR_PURGE = "gdpr_purge"


@dataclass
class AuditEntry:
    id: int
    tenant_id: str
    node_id: str
    action: str
    decay_score: float | None
    details: dict[str, Any]
    created_at: datetime


_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    action TEXT NOT NULL,
    decay_score REAL,
    details TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_log_node ON audit_log (tenant_id, node_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log (action);
"""


class AuditLog:
    """Thread-safe append-only audit trail.

    Uses a single long-lived sqlite3 connection guarded by a lock, which is
    sufficient for this demo's write volume. ``:memory:`` is supported for
    tests (each instance gets its own isolated in-memory database).
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def record(
        self,
        *,
        tenant_id: str,
        node_id: str,
        action: str,
        decay_score: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> int:
        payload = json.dumps(details or {})
        now = datetime.now(UTC).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO audit_log (tenant_id, node_id, action, decay_score, details, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (tenant_id, node_id, action, decay_score, payload, now),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def list(
        self,
        tenant_id: str | None = None,
        node_id: str | None = None,
        action: str | None = None,
        limit: int = 500,
    ) -> list[AuditEntry]:
        query = "SELECT id, tenant_id, node_id, action, decay_score, details, created_at FROM audit_log"
        clauses, params = [], []
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if node_id is not None:
            clauses.append("node_id = ?")
            params.append(node_id)
        if action is not None:
            clauses.append("action = ?")
            params.append(action)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id ASC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [
            AuditEntry(
                id=r[0],
                tenant_id=r[1],
                node_id=r[2],
                action=r[3],
                decay_score=r[4],
                details=json.loads(r[5]),
                created_at=datetime.fromisoformat(r[6]),
            )
            for r in rows
        ]

    def connection(self) -> sqlite3.Connection:
        """Expose the raw connection so sibling tables (e.g. the jury's
        human-review queue) can live in the same sqlite file."""
        return self._conn

    def close(self) -> None:
        with self._lock:
            self._conn.close()
