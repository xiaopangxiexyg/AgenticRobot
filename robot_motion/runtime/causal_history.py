"""Small, bounded immutable histories for causal runtime observations.

The history is deliberately independent from any hardware backend.  Producers
publish an already-read payload together with the timestamp at which the data
was measured and the timestamp at which the immutable copy became available to
the local process.  Consumers can then ask for the newest snapshot that was
known at a control cut without reading mutable backend state after that cut.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class CausalSnapshotSelection:
    """Selection metadata retained alongside a returned immutable payload."""

    sequence: int
    source_timestamp: float | None
    available_timestamp: float
    age_sec: float | None
    source_age_sec: float | None


class CausalSnapshotHistory:
    """Thread-safe bounded history with strict ``source <= T0`` semantics.

    Payloads are deep-copied both when published and when returned.  This is
    important for numpy-backed tactile arrays and for backend dictionaries that
    continue to be mutated by telemetry workers.  ``available_timestamp`` is
    recorded *after* the copy has completed, so a consumer never observes a
    partially published snapshot.
    """

    def __init__(
        self,
        *,
        maxlen: int = 256,
        clock: Any = time.time,
        name: str = "snapshot",
    ) -> None:
        if int(maxlen) <= 0:
            raise ValueError("maxlen must be positive")
        self.maxlen = int(maxlen)
        self.clock = clock
        self.name = str(name)
        self._lock = threading.RLock()
        self._sequence = 0
        self._items: deque[dict[str, Any]] = deque(maxlen=self.maxlen)

    def publish(
        self,
        payload: Mapping[str, Any] | Any,
        *,
        source_timestamp: float | None,
        available_timestamp: float | None = None,
        source_clock: str = "local_wall_time",
        available_clock: str = "local_wall_time",
    ) -> dict[str, Any]:
        """Publish and return the frozen copy.

        A missing/non-finite source timestamp is retained for diagnostics but
        can never satisfy a strict causal query.  The producer may provide an
        availability time captured at the end of its read; if omitted, it is
        taken immediately after the deep copy.
        """

        frozen = copy.deepcopy(payload)
        source = _finite_or_none(source_timestamp)
        if available_timestamp is None:
            available = _finite_or_none(self.clock())
        else:
            available = _finite_or_none(available_timestamp)
        if available is None:
            raise ValueError("available_timestamp must be finite")
        with self._lock:
            self._sequence += 1
            sequence = int(self._sequence)
            if isinstance(frozen, dict):
                record = frozen
            else:
                record = {"value": frozen}
            # Namespaced keys avoid changing device-native timestamp fields.
            record["snapshot_sequence"] = sequence
            record["source_timestamp"] = source
            record["available_timestamp"] = available
            record["source_clock"] = str(source_clock)
            record["available_clock"] = str(available_clock)
            self._items.append(record)
            return copy.deepcopy(record)

    def latest_at_or_before(
        self,
        cutoff_timestamp: float,
        *,
        require_source_timestamp: bool = True,
    ) -> tuple[dict[str, Any], CausalSnapshotSelection] | None:
        """Return the newest record satisfying the strict causal contract."""

        cutoff = _finite_or_none(cutoff_timestamp)
        if cutoff is None:
            raise ValueError("cutoff_timestamp must be finite")
        with self._lock:
            for record in reversed(self._items):
                available = _finite_or_none(record.get("available_timestamp"))
                source = _finite_or_none(record.get("source_timestamp"))
                if available is None or available > cutoff:
                    continue
                if require_source_timestamp and (source is None or source > cutoff):
                    continue
                selected = copy.deepcopy(record)
                selection = CausalSnapshotSelection(
                    sequence=int(record["snapshot_sequence"]),
                    source_timestamp=source,
                    available_timestamp=float(available),
                    age_sec=float(cutoff - available),
                    source_age_sec=(None if source is None else float(cutoff - source)),
                )
                return selected, selection
        return None

    def latest(self) -> tuple[dict[str, Any], CausalSnapshotSelection] | None:
        with self._lock:
            if not self._items:
                return None
            record = copy.deepcopy(self._items[-1])
            source = _finite_or_none(record.get("source_timestamp"))
            available = _finite_or_none(record.get("available_timestamp"))
            if available is None:
                return None
            return record, CausalSnapshotSelection(
                sequence=int(record["snapshot_sequence"]),
                source_timestamp=source,
                available_timestamp=available,
                age_sec=0.0,
                source_age_sec=None,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

