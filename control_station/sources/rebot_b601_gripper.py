"""Read-only B601 J7 source for the RM65 MuJoCo hand proposal."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from hardware.rebot_b601_rs.gripper import (
    B601_RS_GRIPPER_SOURCE_ID,
    B601RSGripperPositionObservation,
)
from robot_motion.rebot_b601_rs.gripper_mapping import (
    B601RM65GripperMappingProfile,
    NormalizedGripperSample,
)


@dataclass(frozen=True)
class B601ReadOnlyGripperStep:
    observation: B601RSGripperPositionObservation | None
    sample: NormalizedGripperSample | None
    rejection_reason: str | None

    def __post_init__(self) -> None:
        if (self.sample is None) == (self.rejection_reason is None):
            raise ValueError("gripper step must contain exactly one sample or rejection")
        if self.sample is not None and self.observation is None:
            raise ValueError("an admitted gripper sample requires its observation")


@runtime_checkable
class _GripperReader(Protocol):
    def read_position(self) -> B601RSGripperPositionObservation: ...


class B601ReadOnlyGripperSource:
    """Convert fresh J7 ``mechPos`` replies into the canonical close scalar."""

    def __init__(
        self,
        reader: _GripperReader,
        mapping: B601RM65GripperMappingProfile,
        *,
        clock_domain_id: str,
        clamp: bool = False,
        calibration_tolerance_rad: float = 0.0,
    ) -> None:
        if not isinstance(reader, _GripperReader):
            raise TypeError("reader must provide the read_position port")
        if not isinstance(mapping, B601RM65GripperMappingProfile):
            raise TypeError("mapping must be B601RM65GripperMappingProfile")
        if not str(clock_domain_id).strip():
            raise ValueError("clock_domain_id must be non-empty")
        if not isinstance(clamp, bool):
            raise TypeError("clamp must be bool")
        try:
            tolerance = float(calibration_tolerance_rad)
        except (TypeError, ValueError) as error:
            raise TypeError("calibration_tolerance_rad must be finite") from error
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "calibration_tolerance_rad must be finite and non-negative"
            )
        self.reader = reader
        self.mapping = mapping
        self.clock_domain_id = str(clock_domain_id)
        self.clamp = clamp
        self.calibration_tolerance_rad = tolerance
        self.last_step: B601ReadOnlyGripperStep | None = None

    def step(self) -> B601ReadOnlyGripperStep:
        try:
            observation = self.reader.read_position()
            sample = self.mapping.b601_position_sample(
                observation.position_rad,
                sequence=observation.sequence,
                source_timestamp_sec=observation.source_timestamp_sec,
                receive_timestamp_sec=observation.receive_timestamp_sec,
                source_id=B601_RS_GRIPPER_SOURCE_ID,
                clock_domain_id=self.clock_domain_id,
                clamp=self.clamp,
                calibration_tolerance_rad=self.calibration_tolerance_rad,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            result = B601ReadOnlyGripperStep(
                observation=locals().get("observation"),
                sample=None,
                rejection_reason=f"{type(error).__name__}:{error}",
            )
        else:
            result = B601ReadOnlyGripperStep(
                observation=observation,
                sample=sample,
                rejection_reason=None,
            )
        self.last_step = result
        return result


__all__ = [
    "B601ReadOnlyGripperSource",
    "B601ReadOnlyGripperStep",
]
