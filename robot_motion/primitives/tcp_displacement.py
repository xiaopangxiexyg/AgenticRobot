"""Pure cardinal TCP-displacement primitive semantics.

The six built-in short-step primitives translate ``tool_tcp`` by exactly
3 cm along the axes of an explicitly supplied target reference frame:

* ``+X`` forward and ``-X`` backward;
* ``+Y`` left and ``-Y`` right;
* ``+Z`` up and ``-Z`` down.

An input pose is therefore ``T_reference_tool_tcp`` and the resulting target
keeps its orientation unchanged while adding translation in ``reference``
axes.  This module intentionally does not capture the anchor pose, select an
arm, choose duration or speed, apply a safety envelope, solve IK, interpolate,
or dispatch.  Those remain responsibilities of the authoritative runtime.
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

from ._pose import _POSE_ATOL, _rigid_pose

TCP_DISPLACEMENT_PRIMITIVE_HASH_DOMAIN_V1 = (
    "dexteleop.robot_motion.tcp_displacement_primitive.v1"
)
TCP_DISPLACEMENT_TARGET_HASH_DOMAIN_V1 = (
    "dexteleop.robot_motion.tcp_displacement_target.v1"
)
TCP_DISPLACEMENT_POSE_CONVENTION_V1 = "T_reference_tool_tcp"
TCP_CONTROLLED_FRAME_ID = "tool_tcp"
SHORT_STEP_DISTANCE_M = 0.03


class CardinalTCPPoseDirection(str, Enum):
    """Named directions in the canonical target reference-frame axes."""

    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


_UNIT_TRANSLATION_BY_DIRECTION: Mapping[
    CardinalTCPPoseDirection,
    tuple[float, float, float],
] = MappingProxyType(
    {
        CardinalTCPPoseDirection.FORWARD: (1.0, 0.0, 0.0),
        CardinalTCPPoseDirection.BACKWARD: (-1.0, 0.0, 0.0),
        CardinalTCPPoseDirection.LEFT: (0.0, 1.0, 0.0),
        CardinalTCPPoseDirection.RIGHT: (0.0, -1.0, 0.0),
        CardinalTCPPoseDirection.UP: (0.0, 0.0, 1.0),
        CardinalTCPPoseDirection.DOWN: (0.0, 0.0, -1.0),
    }
)


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    if normalized != value:
        raise ValueError(f"{name} cannot contain surrounding whitespace.")
    return normalized


def _direction(value: object) -> CardinalTCPPoseDirection:
    if isinstance(value, CardinalTCPPoseDirection):
        return value
    if not isinstance(value, str):
        raise TypeError("direction must be a CardinalTCPPoseDirection or string.")
    try:
        return CardinalTCPPoseDirection(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in CardinalTCPPoseDirection)
        raise ValueError(
            f"Unsupported cardinal TCP direction {value!r}; expected one of {allowed}."
        ) from exc


def _positive_distance(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite positive number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite positive number.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number.")
    return result


def _primitive_payload(
    *,
    primitive_id: str,
    primitive_version: str,
    direction: CardinalTCPPoseDirection,
    distance_m: float,
    controlled_frame_id: str,
    reference_axis_contract: str,
    pose_convention: str,
) -> dict[str, Any]:
    return {
        "primitive_id": primitive_id,
        "primitive_version": primitive_version,
        "direction": direction.value,
        "distance_m": distance_m,
        "controlled_frame_id": controlled_frame_id,
        "reference_axis_contract": reference_axis_contract,
        "pose_convention": pose_convention,
        "translation_application": "reference_frame_additive",
        "rotation_application": "unchanged",
    }


def _vector3(value: object, *, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite values.") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values.")
    return deep_freeze(np.ascontiguousarray(vector, dtype=np.float64))


@dataclass(frozen=True)
class TCPPoseDisplacementTarget:
    """Immutable absolute TCP target derived from one displacement primitive."""

    primitive_id: str
    primitive_hash: str
    direction: CardinalTCPPoseDirection
    distance_m: float
    reference_frame_id: str
    reference_frame_spec_hash: str
    anchor_pose: np.ndarray
    target_pose: np.ndarray
    delta_translation_reference_m: np.ndarray
    primitive_version: str = "1"
    controlled_frame_id: str = TCP_CONTROLLED_FRAME_ID
    reference_axis_contract: str = STATION_RETARGET_FRAME_CONTRACT_V1
    pose_convention: str = TCP_DISPLACEMENT_POSE_CONVENTION_V1
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
        if text["controlled_frame_id"] != TCP_CONTROLLED_FRAME_ID:
            raise ValueError("TCP displacement targets must control tool_tcp.")
        if (
            text["reference_axis_contract"]
            != STATION_RETARGET_FRAME_CONTRACT_V1
        ):
            raise ValueError("Unsupported TCP displacement axis contract.")
        if text["pose_convention"] != TCP_DISPLACEMENT_POSE_CONVENTION_V1:
            raise ValueError("Unsupported TCP displacement pose convention.")
        direction = _direction(self.direction)
        distance_m = _positive_distance(self.distance_m, name="distance_m")
        expected_primitive_hash = canonical_content_hash(
            _primitive_payload(
                primitive_id=text["primitive_id"],
                primitive_version=text["primitive_version"],
                direction=direction,
                distance_m=distance_m,
                controlled_frame_id=text["controlled_frame_id"],
                reference_axis_contract=text["reference_axis_contract"],
                pose_convention=text["pose_convention"],
            ),
            domain=TCP_DISPLACEMENT_PRIMITIVE_HASH_DOMAIN_V1,
        )
        if text["primitive_hash"] != expected_primitive_hash:
            raise ValueError("primitive_hash does not match primitive semantics.")
        anchor = _rigid_pose(self.anchor_pose, name="anchor_pose")
        target = _rigid_pose(self.target_pose, name="target_pose")
        delta = _vector3(
            self.delta_translation_reference_m,
            name="delta_translation_reference_m",
        )
        expected_delta = (
            np.asarray(_UNIT_TRANSLATION_BY_DIRECTION[direction], dtype=np.float64)
            * distance_m
        )
        if not np.allclose(delta, expected_delta, rtol=0.0, atol=_POSE_ATOL):
            raise ValueError(
                "delta_translation_reference_m does not match direction and distance."
            )
        if not np.allclose(
            target[:3, :3],
            anchor[:3, :3],
            rtol=0.0,
            atol=_POSE_ATOL,
        ):
            raise ValueError("A TCP displacement target must preserve orientation.")
        if not np.allclose(
            target[:3, 3] - anchor[:3, 3],
            delta,
            rtol=0.0,
            atol=_POSE_ATOL,
        ):
            raise ValueError(
                "target_pose translation does not match the declared displacement."
            )
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "distance_m", distance_m)
        object.__setattr__(self, "anchor_pose", anchor)
        object.__setattr__(self, "target_pose", target)
        object.__setattr__(self, "delta_translation_reference_m", delta)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=TCP_DISPLACEMENT_TARGET_HASH_DOMAIN_V1,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "primitive_id": self.primitive_id,
            "primitive_hash": self.primitive_hash,
            "primitive_version": self.primitive_version,
            "direction": self.direction.value,
            "distance_m": self.distance_m,
            "reference_frame_id": self.reference_frame_id,
            "reference_frame_spec_hash": self.reference_frame_spec_hash,
            "controlled_frame_id": self.controlled_frame_id,
            "reference_axis_contract": self.reference_axis_contract,
            "pose_convention": self.pose_convention,
            "anchor_pose": deep_thaw(self.anchor_pose),
            "target_pose": deep_thaw(self.target_pose),
            "delta_translation_reference_m": deep_thaw(
                self.delta_translation_reference_m
            ),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class TCPPoseDisplacementPrimitive:
    """One pure translation relative to an activation-local absolute pose."""

    primitive_id: str
    direction: CardinalTCPPoseDirection
    distance_m: float
    primitive_version: str = "1"
    controlled_frame_id: str = TCP_CONTROLLED_FRAME_ID
    reference_axis_contract: str = STATION_RETARGET_FRAME_CONTRACT_V1
    pose_convention: str = TCP_DISPLACEMENT_POSE_CONVENTION_V1
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
        if text["controlled_frame_id"] != TCP_CONTROLLED_FRAME_ID:
            raise ValueError("TCP displacement primitives must control tool_tcp.")
        if (
            text["reference_axis_contract"]
            != STATION_RETARGET_FRAME_CONTRACT_V1
        ):
            raise ValueError("Unsupported TCP displacement axis contract.")
        if text["pose_convention"] != TCP_DISPLACEMENT_POSE_CONVENTION_V1:
            raise ValueError("Unsupported TCP displacement pose convention.")
        direction = _direction(self.direction)
        distance_m = _positive_distance(self.distance_m, name="distance_m")
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "distance_m", distance_m)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self.to_payload(include_hash=False),
                domain=TCP_DISPLACEMENT_PRIMITIVE_HASH_DOMAIN_V1,
            ),
        )

    @property
    def delta_translation_reference_m(self) -> np.ndarray:
        vector = np.asarray(
            _UNIT_TRANSLATION_BY_DIRECTION[self.direction],
            dtype=np.float64,
        )
        return deep_freeze(vector * self.distance_m)

    def apply(
        self,
        anchor_pose: object,
        *,
        reference_frame_spec: CoordinateFrameSpec,
    ) -> TCPPoseDisplacementTarget:
        """Return the absolute target for one explicit reference-frame anchor."""

        if not isinstance(reference_frame_spec, CoordinateFrameSpec):
            raise TypeError("reference_frame_spec must be a CoordinateFrameSpec.")
        supplied_axis_contract = reference_frame_spec.semantic_parameters.get(
            "axis_contract"
        )
        if supplied_axis_contract != self.reference_axis_contract:
            raise ValueError(
                "reference_frame_spec axis_contract does not match the "
                "TCP displacement primitive."
            )
        anchor = _rigid_pose(anchor_pose, name="anchor_pose")
        delta = self.delta_translation_reference_m
        target = np.asarray(anchor, dtype=np.float64).copy()
        target[:3, 3] += delta
        return TCPPoseDisplacementTarget(
            primitive_id=self.primitive_id,
            primitive_hash=self.semantic_content_hash,
            primitive_version=self.primitive_version,
            direction=self.direction,
            distance_m=self.distance_m,
            reference_frame_id=reference_frame_spec.frame_id,
            reference_frame_spec_hash=reference_frame_spec.semantic_content_hash,
            controlled_frame_id=self.controlled_frame_id,
            reference_axis_contract=self.reference_axis_contract,
            pose_convention=self.pose_convention,
            anchor_pose=anchor,
            target_pose=target,
            delta_translation_reference_m=delta,
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = _primitive_payload(
            primitive_id=self.primitive_id,
            primitive_version=self.primitive_version,
            direction=self.direction,
            distance_m=self.distance_m,
            controlled_frame_id=self.controlled_frame_id,
            reference_axis_contract=self.reference_axis_contract,
            pose_convention=self.pose_convention,
        )
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


CARDINAL_TCP_SHORT_STEP_PRIMITIVES = tuple(
    TCPPoseDisplacementPrimitive(
        primitive_id=f"robot_motion.tcp.short_step.{direction.value}",
        direction=direction,
        distance_m=SHORT_STEP_DISTANCE_M,
    )
    for direction in CardinalTCPPoseDirection
)

CARDINAL_TCP_SHORT_STEP_BY_DIRECTION: Mapping[
    CardinalTCPPoseDirection,
    TCPPoseDisplacementPrimitive,
] = MappingProxyType(
    {
        primitive.direction: primitive
        for primitive in CARDINAL_TCP_SHORT_STEP_PRIMITIVES
    }
)


def cardinal_tcp_short_step(
    direction: CardinalTCPPoseDirection | str,
) -> TCPPoseDisplacementPrimitive:
    """Return one of the six immutable 3 cm cardinal TCP primitives."""

    return CARDINAL_TCP_SHORT_STEP_BY_DIRECTION[_direction(direction)]


__all__ = [
    "CARDINAL_TCP_SHORT_STEP_BY_DIRECTION",
    "CARDINAL_TCP_SHORT_STEP_PRIMITIVES",
    "SHORT_STEP_DISTANCE_M",
    "CardinalTCPPoseDirection",
    "TCPPoseDisplacementPrimitive",
    "TCPPoseDisplacementTarget",
    "cardinal_tcp_short_step",
]
