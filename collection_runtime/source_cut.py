"""Device-neutral results at a live source-cut boundary.

Applications remain responsible for producing and interpreting source values.
This module only defines the generic rejection result returned when a
collection-time cut cannot be committed as a complete sample.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SnapshotRejection:
    """A non-committable collection cut and its stable application reasons."""

    timestamp: float
    reasons: tuple[str, ...]


__all__ = ["SnapshotRejection"]
