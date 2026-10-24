"""A tiny virtual clock used so demos can *simulate* days/weeks of memory
decay passing in seconds, without waiting in real time and without faking
the system clock (which would affect logging, auth tokens, etc.).

``VirtualClock.now()`` is the single source of truth for "current time" used
by the decay engine and by access-time bookkeeping in the store adapters.
By default it tracks real wall-clock time (offset = 0). The demo-only
``/admin/time-warp`` endpoint advances the offset so a reviewer can show
decay scores dropping and pruning firing without a multi-day wait.
"""
from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta


class VirtualClock:
    def __init__(self) -> None:
        self._offset = timedelta(0)
        self._lock = threading.Lock()

    def now(self) -> datetime:
        with self._lock:
            offset = self._offset
        return datetime.now(UTC) + offset

    def advance(self, *, hours: float = 0.0, days: float = 0.0) -> datetime:
        """Jump the virtual clock forward. Demo/testing use only."""
        delta = timedelta(hours=hours, days=days)
        with self._lock:
            self._offset += delta
        return self.now()

    def reset(self) -> None:
        with self._lock:
            self._offset = timedelta(0)

    @property
    def offset_seconds(self) -> float:
        with self._lock:
            return self._offset.total_seconds()


# Process-wide singleton. In multi-process deployments each worker gets its
# own offset; for the local demo (single app process + single decay-engine
# process sharing a DB) this is sufficient to showcase the feature.
clock = VirtualClock()
