"""Backend-neutral B601 J7 position observation contract."""

from __future__ import annotations

import math
from dataclasses import dataclass


B601_RS_GRIPPER_SOURCE_ID = "b601_rs_gripper_motor_position_rad"
B601_RS_GRIPPER_CLOCK_DOMAIN = "b601-rs-host-monotonic"


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be finite") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


@dataclass(frozen=True)
class B601RSGripperPositionObservation:
    """One fresh J7 position reply in the shared monotonic clock domain."""

    position_rad: float
    sequence: int
    source_timestamp_sec: float
    receive_timestamp_sec: float
    source_id: str = B601_RS_GRIPPER_SOURCE_ID
    clock_domain_id: str = B601_RS_GRIPPER_CLOCK_DOMAIN

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_rad",
            _finite(self.position_rad, name="position_rad"),
        )
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a non-negative integer")
        source_time = _finite(
            self.source_timestamp_sec,
            name="source_timestamp_sec",
        )
        receive_time = _finite(
            self.receive_timestamp_sec,
            name="receive_timestamp_sec",
        )
        if receive_time < source_time:
            raise ValueError("receive_timestamp_sec must not precede source time")
        object.__setattr__(self, "source_timestamp_sec", source_time)
        object.__setattr__(self, "receive_timestamp_sec", receive_time)
        if not str(self.source_id).strip() or not str(self.clock_domain_id).strip():
            raise ValueError("source_id and clock_domain_id must be non-empty")


__all__ = [
    "B601_RS_GRIPPER_CLOCK_DOMAIN",
    "B601_RS_GRIPPER_SOURCE_ID",
    "B601RSGripperPositionObservation",
]
