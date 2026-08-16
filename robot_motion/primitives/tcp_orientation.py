"""Pure absolute level TCP-orientation primitive semantics.

The three built-in primitives keep ``tool_tcp`` translation fixed and replace
its orientation in an explicitly supplied canonical station reference frame.
They use the existing desk-axis contract (+X forward into the tabletop, +Y to
the operator's left and +Z up) and the calibrated tool-axis convention:

* tool +Z is the gripper's forward direction;
* tool +Y is the gripper's left direction;
* tool +X points down while the gripper is level.

``center_forward`` therefore maps tool +Z to desk +X.  Positive heading yaw is
toward desk +Y (operator left), so ``forward_left_30`` is +30 degrees and
``forward_right_30`` is -30 degrees about desk +Z.  These are absolute
orientations, not incremental rotations from the input pose.

This module only constructs immutable pose intent.  It does not capture an
anchor, select an arm, choose transition timing, apply safety, solve IK,
interpolate, or dispatch.

The primitives are deliberately arm-neutral: the same primitive may be
applied to a measured ``left_arm`` or ``right_arm`` tool_tcp pose.  Arm
selection belongs to the caller and is not embedded in the primitive hash.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.coordinate_frames import CoordinateFrameSpec
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.runtime.immutable import deep_freeze, deep_thaw

TCP_ORIENTATION_PRIMITIVE_HASH_DOMAIN_V1 = (
    "dexteleop.robot_motion.tcp_orientation_primitive.v1"
)
TCP_ORIENTATION_TARGET_HASH_DOMAIN_V1 = (
    "dexteleop.robot_motion.tcp_orientation_target.v1"
)
TCP_ORIENTATION_POSE_CONVENTION_V1 = "T_reference_tool_tcp"
TCP_ORIENTATION_CONTROLLED_FRAME_ID = "tool_tcp"
TCP_FORWARD_AXIS = "+Z"
TCP_LEFT_AXIS = "+Y"
TCP_LEVEL_DOWN_AXIS = "+X"

_POSE_ATOL = 1e-9


class LevelTCPHeading(str, Enum):
    """Absolute horizontal gripper headings in canonical desk axes."""

    CENTER_FORWARD = "center_forward"
    FORWARD_LEFT_30 = "forward_left_30"
    FORWARD_RIGHT_30 = "forward_right_30"


_HEADING_YAW_DEG: Mapping[LevelTCPHeading, float] = MappingProxyType(
    {
        LevelTCPHeading.CENTER_FORWARD: 0.0,
        LevelTCPHeading.FORWARD_LEFT_30: 30.0,
        LevelTCPHeading.FORWARD_RIGHT_30: -30.0,
    }
)

_HEADING_DESCRIPTION: Mapping[LevelTCPHeading, str] = MappingProxyType(
    {
        LevelTCPHeading.CENTER_FORWARD: (
            "Set the selected arm gripper to the horizontal desk plane at "
            "the fixed absolute forward heading (0 deg), preserving TCP "
            "position; this is not a relative rotation."
        ),
        LevelTCPHeading.FORWARD_LEFT_30: (
            "Set the selected arm gripper to the horizontal desk plane at "
            "the fixed absolute forward-left 30 deg heading, preserving TCP "
            "position; this is not a relative rotation."
        ),
        LevelTCPHeading.FORWARD_RIGHT_30: (
            "Set the selected arm gripper to the horizontal desk plane at "
            "the fixed absolute forward-right 30 deg heading, preserving TCP "
            "position; this is not a relative rotation."
        ),
    }
)


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    if result != value:
        raise ValueError(f"{name} cannot contain surrounding whitespace.")
    return result


def _heading(value: object) -> LevelTCPHeading:
    if isinstance(value, LevelTCPHeading):
        return value
    if not isinstance(value, str):
        raise TypeError("heading must be a LevelTCPHeading or string.")
    try:
        return LevelTCPHeading(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in LevelTCPHeading)
        raise ValueError(
            f"Unsupported level TCP heading {value!r}; expected one of {allowed}."
        ) from exc


def _finite_yaw(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("heading_yaw_deg must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("heading_yaw_deg must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError("heading_yaw_deg must be a finite number.")
    return result


def _rigid_pose(value: object, *, name: str) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric 4x4 pose.") from exc
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be a finite 4x4 pose.")
    if not np.allclose(
        pose[3],
        (0.0, 0.0, 0.0, 1.0),
        rtol=0.0,
        atol=_POSE_ATOL,
    ):
        raise ValueError(f"{name} must be homogeneous.")
    rotation = pose[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=_POSE_ATOL,
    ):
        raise ValueError(f"{name} rotation must be orthonormal.")
    if not math.isclose(
        float(np.linalg.det(rotation)),
        1.0,
        rel_tol=0.0,
        abs_tol=_POSE_ATOL,
    ):
        raise ValueError(f"{name} rotation determinant must be +1.")
    return deep_freeze(np.ascontiguousarray(pose, dtype=np.float64))


def _level_rotation_reference(heading_yaw_deg: float) -> np.ndarray:
    """Return R_reference_tool for a level tool at the requested heading."""

    yaw = math.radians(float(heading_yaw_deg))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.asarray(
        (
            (0.0, -sine, cosine),
            (0.0, cosine, sine),
            (-1.0, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    return deep_freeze(rotation)


def _primitive_payload(
    *,
    primitive_id: str,
    primitive_version: str,
    heading: LevelTCPHeading,
    heading_yaw_deg: float,
    controlled_frame_id: str,
    reference_axis_contract: str,
    pose_convention: str,
) -> dict[str, Any]:
    return {
        "primitive_id": primitive_id,
        "primitive_version": primitive_version,
        "heading": heading.value,
        "heading_yaw_deg": heading_yaw_deg,
        "description": _HEADING_DESCRIPTION[heading],
        "controlled_frame_id": controlled_frame_id,
        "reference_axis_contract": reference_axis_contract,
        "pose_convention": pose_convention,
        "translation_application": "unchanged",
        "rotation_application": "absolute_level_heading",
        "arm_binding": "caller_selects_left_arm_or_right_arm",
        "tool_forward_axis": TCP_FORWARD_AXIS,
        "tool_left_axis": TCP_LEFT_AXIS,
        "tool_level_down_axis": TCP_LEVEL_DOWN_AXIS,
    }


@dataclass(frozen=True)
class LevelTCPOrientationTarget:
    """Immutable absolute TCP orientation target from one primitive."""

    primitive_id: str
    primitive_hash: str
    heading: LevelTCPHeading
    heading_yaw_deg: float
    reference_frame_id: str
    reference_frame_spec_hash: str
    anchor_pose: np.ndarray
    target_pose: np.ndarray
    primitive_version: str = "1"
    controlled_frame_id: str = TCP_ORIENTATION_CONTROLLED_FRAME_ID
    reference_axis_contract: str = STATION_RETARGET_FRAME_CONTRACT_V1
    pose_convention: str = TCP_ORIENTATION_POSE_CONVENTION_V1
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        text = {
            name: _required_text(getattr(self, name), name=name)
            for name in (
                "primitive_id",
                "primitive_hash",
                "primitive_version",
                "reference_frame_id",
                "reference_frame_spec_hash",
                "controlled_frame_id",
                "reference_axis_contract",
                "pose_convention",
            )
        }
        if text["controlled_frame_id"] != TCP_ORIENTATION_CONTROLLED_FRAME_ID:
            raise ValueError("TCP orientation targets must control tool_tcp.")
        if text["reference_axis_contract"] != STATION_RETARGET_FRAME_CONTRACT_V1:
            raise ValueError("Unsupported TCP orientation axis contract.")
        if text["pose_convention"] != TCP_ORIENTATION_POSE_CONVENTION_V1:
            raise ValueError("Unsupported TCP orientation pose convention.")
        heading = _heading(self.heading)
        yaw = _finite_yaw(self.heading_yaw_deg)
        expected_yaw = _HEADING_YAW_DEG[heading]
        if not math.isclose(yaw, expected_yaw, rel_tol=0.0, abs_tol=_POSE_ATOL):
            raise ValueError("heading_yaw_deg does not match heading semantics.")
        expected_primitive_hash = canonical_content_hash(
            _primitive_payload(
                primitive_id=text["primitive_id"],
                primitive_version=text["primitive_version"],
                heading=heading,
                heading_yaw_deg=yaw,
                controlled_frame_id=text["controlled_frame_id"],
                reference_axis_contract=text["reference_axis_contract"],
                pose_convention=text["pose_convention"],
            ),
            domain=TCP_ORIENTATION_PRIMITIVE_HASH_DOMAIN_V1,
        )
        if text["primitive_hash"] != expected_primitive_hash:
            raise ValueError("primitive_hash does not match primitive semantics.")
        anchor = _rigid_pose(self.anchor_pose, name="anchor_pose")
        target = _rigid_pose(self.target_pose, name="target_pose")
        if not np.allclose(
            target[:3, 3],
            anchor[:3, 3],
            rtol=0.0,
            atol=_POSE_ATOL,
        ):
            raise ValueError("A TCP orientation target must preserve translation.")
        if not np.allclose(
            target[:3, :3],
            _level_rotation_reference(yaw),
            rtol=0.0,
            atol=_POSE_ATOL,
        ):
            raise ValueError("target_pose does not match the level heading.")
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "heading", heading)
        object.__setattr__(self, "heading_yaw_deg", yaw)
        object.__setattr__(self, "anchor_pose", anchor)
        object.__setattr__(self, "target_pose", target)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=TCP_ORIENTATION_TARGET_HASH_DOMAIN_V1,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "primitive_id": self.primitive_id,
            "primitive_hash": self.primitive_hash,
            "primitive_version": self.primitive_version,
            "heading": self.heading.value,
            "heading_yaw_deg": self.heading_yaw_deg,
            "reference_frame_id": self.reference_frame_id,
            "reference_frame_spec_hash": self.reference_frame_spec_hash,
            "controlled_frame_id": self.controlled_frame_id,
            "reference_axis_contract": self.reference_axis_contract,
            "pose_convention": self.pose_convention,
            "anchor_pose": deep_thaw(self.anchor_pose),
            "target_pose": deep_thaw(self.target_pose),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class LevelTCPOrientationPrimitive:
    """One absolute level heading while holding the input TCP position."""

    primitive_id: str
    heading: LevelTCPHeading
    heading_yaw_deg: float
    primitive_version: str = "1"
    controlled_frame_id: str = TCP_ORIENTATION_CONTROLLED_FRAME_ID
    reference_axis_contract: str = STATION_RETARGET_FRAME_CONTRACT_V1
    pose_convention: str = TCP_ORIENTATION_POSE_CONVENTION_V1
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        text = {
            name: _required_text(getattr(self, name), name=name)
            for name in (
                "primitive_id",
                "primitive_version",
                "controlled_frame_id",
                "reference_axis_contract",
                "pose_convention",
            )
        }
        if text["controlled_frame_id"] != TCP_ORIENTATION_CONTROLLED_FRAME_ID:
            raise ValueError("TCP orientation primitives must control tool_tcp.")
        if text["reference_axis_contract"] != STATION_RETARGET_FRAME_CONTRACT_V1:
            raise ValueError("Unsupported TCP orientation axis contract.")
        if text["pose_convention"] != TCP_ORIENTATION_POSE_CONVENTION_V1:
            raise ValueError("Unsupported TCP orientation pose convention.")
        heading = _heading(self.heading)
        yaw = _finite_yaw(self.heading_yaw_deg)
        if not math.isclose(
            yaw,
            _HEADING_YAW_DEG[heading],
            rel_tol=0.0,
            abs_tol=_POSE_ATOL,
        ):
            raise ValueError("heading_yaw_deg does not match heading semantics.")
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "heading", heading)
        object.__setattr__(self, "heading_yaw_deg", yaw)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=TCP_ORIENTATION_PRIMITIVE_HASH_DOMAIN_V1,
            ),
        )

    @property
    def target_rotation_reference(self) -> np.ndarray:
        return _level_rotation_reference(self.heading_yaw_deg)

    @property
    def description(self) -> str:
        """Human-facing fixed, horizontal, non-relative Skill semantics."""

        return _HEADING_DESCRIPTION[self.heading]

    def apply(
        self,
        anchor_pose: object,
        *,
        reference_frame_spec: CoordinateFrameSpec,
    ) -> LevelTCPOrientationTarget:
        """Return the absolute level target for one reference-frame pose."""

        if not isinstance(reference_frame_spec, CoordinateFrameSpec):
            raise TypeError("reference_frame_spec must be a CoordinateFrameSpec.")
        supplied_axis_contract = reference_frame_spec.semantic_parameters.get(
            "axis_contract"
        )
        if supplied_axis_contract != self.reference_axis_contract:
            raise ValueError(
                "reference_frame_spec axis_contract does not match the "
                "TCP orientation primitive."
            )
        anchor = _rigid_pose(anchor_pose, name="anchor_pose")
        target = np.asarray(anchor, dtype=np.float64).copy()
        target[:3, :3] = self.target_rotation_reference
        return LevelTCPOrientationTarget(
            primitive_id=self.primitive_id,
            primitive_hash=self.semantic_content_hash,
            primitive_version=self.primitive_version,
            heading=self.heading,
            heading_yaw_deg=self.heading_yaw_deg,
            reference_frame_id=reference_frame_spec.frame_id,
            reference_frame_spec_hash=reference_frame_spec.semantic_content_hash,
            controlled_frame_id=self.controlled_frame_id,
            reference_axis_contract=self.reference_axis_contract,
            pose_convention=self.pose_convention,
            anchor_pose=anchor,
            target_pose=target,
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = _primitive_payload(
            primitive_id=self.primitive_id,
            primitive_version=self.primitive_version,
            heading=self.heading,
            heading_yaw_deg=self.heading_yaw_deg,
            controlled_frame_id=self.controlled_frame_id,
            reference_axis_contract=self.reference_axis_contract,
            pose_convention=self.pose_convention,
        )
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


LEVEL_TCP_ORIENTATION_PRIMITIVES = tuple(
    LevelTCPOrientationPrimitive(
        primitive_id=f"robot_motion.tcp.level_heading.{heading.value}",
        heading=heading,
        heading_yaw_deg=_HEADING_YAW_DEG[heading],
    )
    for heading in LevelTCPHeading
)

LEVEL_TCP_ORIENTATION_BY_HEADING: Mapping[
    LevelTCPHeading,
    LevelTCPOrientationPrimitive,
] = MappingProxyType(
    {primitive.heading: primitive for primitive in LEVEL_TCP_ORIENTATION_PRIMITIVES}
)


def level_tcp_orientation(
    heading: LevelTCPHeading | str,
) -> LevelTCPOrientationPrimitive:
    """Return one of the three immutable absolute level TCP orientations."""

    return LEVEL_TCP_ORIENTATION_BY_HEADING[_heading(heading)]


__all__ = [
    "LEVEL_TCP_ORIENTATION_BY_HEADING",
    "LEVEL_TCP_ORIENTATION_PRIMITIVES",
    "TCP_FORWARD_AXIS",
    "TCP_LEFT_AXIS",
    "TCP_LEVEL_DOWN_AXIS",
    "LevelTCPHeading",
    "LevelTCPOrientationPrimitive",
    "LevelTCPOrientationTarget",
    "level_tcp_orientation",
]
