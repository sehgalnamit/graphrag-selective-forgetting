"""Re-exports the canonical :class:`AuditLog` implementation from
``decay_engine.audit_log`` under the name the FastAPI app's dependency
wiring expects. Kept as a thin alias (rather than duplicating the class) so
there is exactly one audit-log schema/implementation in the repo, used by
both the API (e.g. the GDPR purge endpoint) and the background worker.
"""
from __future__ import annotations

from decay_engine.audit_log import AuditAction, AuditEntry
from decay_engine.audit_log import AuditLog as AuditLogClient

__all__ = ["AuditLogClient", "AuditAction", "AuditEntry"]
