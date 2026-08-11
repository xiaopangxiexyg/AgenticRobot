"""Monotonic Station revision projection for Agent planning snapshots.

The revision changes only when the caller-supplied control fingerprint
changes.  Measurement values and capture timestamps therefore stay outside
this owner: a moving TCP does not by itself invalidate optimistic concurrency,
while authority, lifecycle, gate, operation, or binding changes do.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash


AGENT_STATION_CONTROL_FINGERPRINT_HASH_DOMAIN_V1 = (
    "dexteleop.agent_gateway.station_control_fingerprint.v1"
)


def _revision(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


@dataclass(frozen=True)
class AgentStationRevisionObservation:
    station_revision: int
    control_fingerprint_hash: str
    changed: bool


class AgentStationRevisionTracker:
    """Thread-safe single owner for one session's monotonic revision."""

    def __init__(self, *, initial_revision: int = 0) -> None:
        self._revision = _revision(
            initial_revision,
            field_name="initial_revision",
        )
        self._control_fingerprint_hash: str | None = None
        self._lock = threading.Lock()

    @property
    def current_revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def current_control_fingerprint_hash(self) -> str | None:
        with self._lock:
            return self._control_fingerprint_hash

    def observe(
        self,
        control_fingerprint: Mapping[str, Any],
    ) -> AgentStationRevisionObservation:
        if not isinstance(control_fingerprint, Mapping):
            raise TypeError("control_fingerprint must be an object.")
        fingerprint_hash = canonical_content_hash(
            dict(control_fingerprint),
            domain=AGENT_STATION_CONTROL_FINGERPRINT_HASH_DOMAIN_V1,
        )
        with self._lock:
            changed = (
                self._control_fingerprint_hash is not None
                and fingerprint_hash != self._control_fingerprint_hash
            )
            if changed:
                self._revision += 1
            self._control_fingerprint_hash = fingerprint_hash
            return AgentStationRevisionObservation(
                station_revision=self._revision,
                control_fingerprint_hash=fingerprint_hash,
                changed=changed,
            )


__all__ = [
    "AGENT_STATION_CONTROL_FINGERPRINT_HASH_DOMAIN_V1",
    "AgentStationRevisionObservation",
    "AgentStationRevisionTracker",
]
