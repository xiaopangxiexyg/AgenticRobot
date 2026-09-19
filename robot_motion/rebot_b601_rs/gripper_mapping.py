"""Calibrated, target-neutral B601/RM65 gripper scalar mapping.

The arm and gripper are deliberately separate control components.  This
module only defines the scalar contract used by simulation adapters; it does
not open CAN/serial devices and it never dispatches a motor command.

The canonical value is ``close`` in ``[0, 1]``: zero is fully open and one is
fully closed.  A B601 RobStride gripper reports a motor position in radians,
so its open/closed positions must be measured before a live adapter may use
that path.  Keeping those two calibration values explicit prevents a guessed
J7 range from silently driving hardware in the wrong direction.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

B601_RM65_GRIPPER_MAPPING_SCHEMA_V1 = "b601_rm65_gripper_mapping.v1"
B601_RS_GRIPPER_SOURCE_V1 = "b601_rs_gripper_motor_position_rad"
RM65_DMTAC_GRIPPER_SOURCE_V1 = "rm65_dmtac_normalized_close"
GRIPPER_CLOCK_DOMAIN_LOCAL_WALL_TIME = "local_wall_time"
NORMALIZED_CLOSE_MIN = 0.0
NORMALIZED_CLOSE_MAX = 1.0
GripperMappingDirection = Literal["b601_to_rm65", "rm65_to_b601"]


class GripperMappingCalibrationRequired(ValueError):
    """Raised when a physical B601 position mapping lacks calibration."""


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite, not bool.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _unit_close(value: object, *, name: str = "close") -> float:
    normalized = _finite(value, name=name)
    if not NORMALIZED_CLOSE_MIN <= normalized <= NORMALIZED_CLOSE_MAX:
        raise ValueError(f"{name} must be in [0, 1].")
    return normalized


def _optional_position(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    return _finite(value, name=name)


def _required_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean.")
    return value


@dataclass(frozen=True)
class NormalizedGripperSample:
    """One source observation already expressed in canonical close units."""

    close: float
    sequence: int
    source_timestamp_sec: float
    receive_timestamp_sec: float
    source_id: str
    valid: bool = True
    clock_domain_id: str = GRIPPER_CLOCK_DOMAIN_LOCAL_WALL_TIME

    def __post_init__(self) -> None:
        object.__setattr__(self, "close", _unit_close(self.close))
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise TypeError("sequence must be an integer.")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative.")
        source_timestamp = _finite(
            self.source_timestamp_sec,
            name="source_timestamp_sec",
        )
        receive_timestamp = _finite(
            self.receive_timestamp_sec,
            name="receive_timestamp_sec",
        )
        if receive_timestamp < source_timestamp:
            raise ValueError("receive_timestamp_sec must not precede source time.")
        source_id = str(self.source_id).strip()
        if not source_id:
            raise ValueError("source_id must be non-empty.")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be boolean.")
        clock_domain_id = str(self.clock_domain_id).strip()
        if not clock_domain_id:
            raise ValueError("clock_domain_id must be non-empty.")
        object.__setattr__(self, "source_timestamp_sec", source_timestamp)
        object.__setattr__(self, "receive_timestamp_sec", receive_timestamp)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "clock_domain_id", clock_domain_id)


@dataclass(frozen=True)
class B601RM65GripperMappingProfile:
    """Explicit scalar mapping and optional B601 physical calibration.

    ``invert_b601`` is direction-only calibration.  It is intentionally not
    inferred from URDF order or motor sign: the operator must choose it after
    a read-only open/close observation.  RM65 DMTAC already exposes the
    canonical normalized scalar, so both directions use the same normalized
    semantic value after conversion.
    """

    profile_id: str
    b601_open_position_rad: float | None
    b601_closed_position_rad: float | None
    invert_b601: bool = False
    rm65_min_close: float = NORMALIZED_CLOSE_MIN
    rm65_max_close: float = NORMALIZED_CLOSE_MAX
    calibration_status: str = "required_before_live"
    b601_gripper_name: str = "b601_gripper"
    rm65_gripper_name: str = "right_gripper"

    def __post_init__(self) -> None:
        profile_id = str(self.profile_id).strip()
        if not profile_id:
            raise ValueError("profile_id must be non-empty.")
        open_position = _optional_position(
            self.b601_open_position_rad,
            name="b601_open_position_rad",
        )
        closed_position = _optional_position(
            self.b601_closed_position_rad,
            name="b601_closed_position_rad",
        )
        if (open_position is None) != (closed_position is None):
            raise ValueError(
                "B601 open and closed positions must both be configured or both be null."
            )
        if (
            open_position is not None
            and closed_position is not None
            and math.isclose(open_position, closed_position, abs_tol=1e-12)
        ):
            raise ValueError("B601 open and closed positions must differ.")
        if not isinstance(self.invert_b601, bool):
            raise TypeError("invert_b601 must be boolean.")
        minimum = _unit_close(self.rm65_min_close, name="rm65_min_close")
        maximum = _unit_close(self.rm65_max_close, name="rm65_max_close")
        if minimum >= maximum:
            raise ValueError("rm65_min_close must be smaller than rm65_max_close.")
        status = str(self.calibration_status).strip()
        if not status:
            raise ValueError("calibration_status must be non-empty.")
        b601_name = str(self.b601_gripper_name).strip()
        rm65_name = str(self.rm65_gripper_name).strip()
        if not b601_name or not rm65_name:
            raise ValueError("gripper names must be non-empty.")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "b601_open_position_rad", open_position)
        object.__setattr__(self, "b601_closed_position_rad", closed_position)
        object.__setattr__(self, "rm65_min_close", minimum)
        object.__setattr__(self, "rm65_max_close", maximum)
        object.__setattr__(self, "calibration_status", status)
        object.__setattr__(self, "b601_gripper_name", b601_name)
        object.__setattr__(self, "rm65_gripper_name", rm65_name)

    @property
    def b601_calibrated(self) -> bool:
        return (
            self.b601_open_position_rad is not None
            and self.b601_closed_position_rad is not None
        )

    def _require_b601_calibration(self) -> tuple[float, float]:
        if not self.b601_calibrated:
            raise GripperMappingCalibrationRequired(
                "B601 gripper open/closed motor positions are not calibrated; "
                "live position mapping is disabled."
            )
        assert self.b601_open_position_rad is not None
        assert self.b601_closed_position_rad is not None
        return self.b601_open_position_rad, self.b601_closed_position_rad

    def normalized_rm65_to_canonical(self, close: float) -> float:
        """Map an RM65 normalized value into canonical close semantics."""

        value = _unit_close(close, name="rm65_close")
        normalized = (value - self.rm65_min_close) / (
            self.rm65_max_close - self.rm65_min_close
        )
        return _unit_close(normalized, name="canonical_close")

    def canonical_to_normalized_rm65(self, close: float) -> float:
        value = _unit_close(close, name="canonical_close")
        return _unit_close(
            self.rm65_min_close
            + value * (self.rm65_max_close - self.rm65_min_close),
            name="rm65_close",
        )

    def b601_position_to_canonical(
        self,
        position_rad: float,
        *,
        clamp: bool = False,
    ) -> float:
        """Convert a B601 motor position; reject uncalibrated/out-of-range input."""

        open_position, closed_position = self._require_b601_calibration()
        position = _finite(position_rad, name="b601_position_rad")
        denominator = closed_position - open_position
        normalized = (position - open_position) / denominator
        if not clamp and not -1e-9 <= normalized <= 1.0 + 1e-9:
            raise ValueError(
                "B601 gripper position is outside the calibrated open/closed range."
            )
        if clamp:
            normalized = min(1.0, max(0.0, normalized))
        if self.invert_b601:
            normalized = 1.0 - normalized
        return _unit_close(normalized, name="canonical_close")

    def canonical_to_b601_position(
        self,
        close: float,
        *,
        clamp: bool = True,
    ) -> float:
        open_position, closed_position = self._require_b601_calibration()
        normalized = _unit_close(close, name="canonical_close")
        if self.invert_b601:
            normalized = 1.0 - normalized
        position = open_position + normalized * (closed_position - open_position)
        if not clamp:
            lower = min(open_position, closed_position)
            upper = max(open_position, closed_position)
            if not lower - 1e-9 <= position <= upper + 1e-9:
                raise ValueError("mapped B601 position is outside calibration range.")
        return float(position)

    def map_normalized(
        self,
        value: float,
        *,
        direction: GripperMappingDirection,
    ) -> float:
        """Map either source's normalized scalar to the other target's scalar."""

        if direction == "b601_to_rm65":
            # The B601 input is already canonical only when supplied by a
            # calibrated source adapter.  This method is for that post-
            # calibration path and therefore does not guess motor units.
            return self.canonical_to_normalized_rm65(value)
        if direction == "rm65_to_b601":
            return self.normalized_rm65_to_canonical(value)
        raise ValueError(f"unsupported gripper mapping direction: {direction!r}")

    def b601_position_sample(
        self,
        position_rad: float,
        *,
        sequence: int,
        source_timestamp_sec: float,
        receive_timestamp_sec: float,
        source_id: str = B601_RS_GRIPPER_SOURCE_V1,
        clock_domain_id: str = GRIPPER_CLOCK_DOMAIN_LOCAL_WALL_TIME,
        clamp: bool = False,
    ) -> NormalizedGripperSample:
        return NormalizedGripperSample(
            close=self.b601_position_to_canonical(position_rad, clamp=clamp),
            sequence=sequence,
            source_timestamp_sec=source_timestamp_sec,
            receive_timestamp_sec=receive_timestamp_sec,
            source_id=source_id,
            clock_domain_id=clock_domain_id,
        )

    def rm65_close_sample(
        self,
        close: float,
        *,
        sequence: int,
        source_timestamp_sec: float,
        receive_timestamp_sec: float,
        source_id: str = RM65_DMTAC_GRIPPER_SOURCE_V1,
        clock_domain_id: str = GRIPPER_CLOCK_DOMAIN_LOCAL_WALL_TIME,
    ) -> NormalizedGripperSample:
        return NormalizedGripperSample(
            close=self.normalized_rm65_to_canonical(close),
            sequence=sequence,
            source_timestamp_sec=source_timestamp_sec,
            receive_timestamp_sec=receive_timestamp_sec,
            source_id=source_id,
            clock_domain_id=clock_domain_id,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": B601_RM65_GRIPPER_MAPPING_SCHEMA_V1,
            "profile_id": self.profile_id,
            "canonical_semantics": {
                "name": "normalized_close",
                "open": 0.0,
                "closed": 1.0,
            },
            "b601_rs": {
                "source": B601_RS_GRIPPER_SOURCE_V1,
                "open_position_rad": self.b601_open_position_rad,
                "closed_position_rad": self.b601_closed_position_rad,
                "invert": self.invert_b601,
                "calibration_status": self.calibration_status,
                "gripper_name": self.b601_gripper_name,
            },
            "rm65": {
                "source": RM65_DMTAC_GRIPPER_SOURCE_V1,
                "min_close": self.rm65_min_close,
                "max_close": self.rm65_max_close,
                "gripper_name": self.rm65_gripper_name,
            },
            "directions": {
                "b601_to_rm65": {
                    "source_units": B601_RS_GRIPPER_SOURCE_V1,
                    "target_units": "normalized_close",
                    "requires_b601_calibration": True,
                },
                "rm65_to_b601": {
                    "source_units": RM65_DMTAC_GRIPPER_SOURCE_V1,
                    "target_units": "normalized_close",
                    "requires_b601_calibration": False,
                },
            },
        }


def load_b601_rm65_gripper_mapping_profile(
    path: str | Path,
) -> B601RM65GripperMappingProfile:
    """Load a human-tunable JSON profile without a mandatory hash gate."""

    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{source}: profile must be a JSON object.")
    if payload.get("schema") != B601_RM65_GRIPPER_MAPPING_SCHEMA_V1:
        raise ValueError(f"{source}: unsupported gripper mapping schema.")
    b601 = payload.get("b601_rs")
    rm65 = payload.get("rm65")
    if not isinstance(b601, Mapping) or not isinstance(rm65, Mapping):
        raise TypeError(f"{source}: b601_rs and rm65 objects are required.")
    canonical = payload.get("canonical_semantics")
    if not isinstance(canonical, Mapping) or canonical.get("open") != 0.0 or canonical.get("closed") != 1.0:
        raise ValueError(f"{source}: canonical open/closed semantics must be 0/1.")
    return B601RM65GripperMappingProfile(
        profile_id=str(payload.get("profile_id", source.stem)),
        b601_open_position_rad=b601.get("open_position_rad"),
        b601_closed_position_rad=b601.get("closed_position_rad"),
    invert_b601=_required_bool(b601.get("invert", False), name="b601_rs.invert"),
        rm65_min_close=rm65.get("min_close", 0.0),
        rm65_max_close=rm65.get("max_close", 1.0),
        calibration_status=str(
            b601.get("calibration_status", "required_before_live")
        ),
        b601_gripper_name=str(b601.get("gripper_name", "b601_gripper")),
        rm65_gripper_name=str(rm65.get("gripper_name", "right_gripper")),
    )


__all__ = [
    "B601_RM65_GRIPPER_MAPPING_SCHEMA_V1",
    "B601_RS_GRIPPER_SOURCE_V1",
    "GRIPPER_CLOCK_DOMAIN_LOCAL_WALL_TIME",
    "RM65_DMTAC_GRIPPER_SOURCE_V1",
    "B601RM65GripperMappingProfile",
    "GripperMappingCalibrationRequired",
    "NormalizedGripperSample",
    "load_b601_rm65_gripper_mapping_profile",
]
