"""Human-review escalation queue.

When the jury cannot reach confident consensus on a borderline forget-vs-
keep (or relevance) decision, the case is escalated here instead of being
silently resolved either way. Backed by SQLite so it can share a file with
:class:`decay_engine.audit_log.AuditLog` (or run fully in-memory for tests).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS human_review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    case_type TEXT NOT NULL,
    case_payload TEXT NOT NULL,
    votes TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    resolution TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
"""


@dataclass
class ReviewItem:
    id: int
    tenant_id: str
    node_id: str
    case_type: str
    case_payload: dict[str, Any]
    votes: list[dict[str, Any]]
    status: str
    resolution: str | None
    created_at: datetime
    resolved_at: datetime | None


class HumanReviewQueue:
    def __init__(self, db_path: str = ":memory:") -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def enqueue(
        self,
        *,
        tenant_id: str,
        node_id: str,
        case_type: str,
        case_payload: dict[str, Any],
        votes: list[dict[str, Any]],
    ) -> int:
        now = datetime.now(UTC).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO human_review_queue "
                "(tenant_id, node_id, case_type, case_payload, votes, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                (tenant_id, node_id, case_type, json.dumps(case_payload), json.dumps(votes), now),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def list_pending(self, tenant_id: str | None = None) -> list[ReviewItem]:
        return self._list(status="pending", tenant_id=tenant_id)

    def _list(self, status: str | None, tenant_id: str | None) -> list[ReviewItem]:
        query = "SELECT * FROM human_review_queue"
        clauses, params = [], []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id ASC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_item(r) for r in rows]

    def resolve(self, item_id: int, resolution: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE human_review_queue SET status = 'resolved', resolution = ?, resolved_at = ? WHERE id = ?",
                (resolution, now, item_id),
            )
            self._conn.commit()

    @staticmethod
    def _row_to_item(row) -> ReviewItem:
        (
            id_,
            tenant_id,
            node_id,
            case_type,
            case_payload,
            votes,
            status,
            resolution,
            created_at,
            resolved_at,
        ) = row
        return ReviewItem(
            id=id_,
            tenant_id=tenant_id,
            node_id=node_id,
            case_type=case_type,
            case_payload=json.loads(case_payload),
            votes=json.loads(votes),
            status=status,
            resolution=resolution,
            created_at=datetime.fromisoformat(created_at),
            resolved_at=datetime.fromisoformat(resolved_at) if resolved_at else None,
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
