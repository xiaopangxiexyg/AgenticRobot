"""Resolve bounded motion primitives against lower-owned measured state.

The Skill layer names a reviewed primitive but does not author an absolute
target.  Concrete simulator and real profiles use this module at activation to
apply that primitive in its semantic frame, then return the target in the
lower execution frame.  Safety, IK, authority and dispatch remain outside this
pure resolution boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
)
from robot_motion.primitives import (
    LevelTCPOrientationPrimitive,
    TCPPoseDisplacementPrimitive,
)
from robot_motion.relative_action import (
    pose_matrix_to_xyzquat,
    xyzquat_to_pose_matrix,
)


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def _pose(value: object, *, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != 7:
        raise ValueError(f"{field_name} must contain xyz+quat_xyzw.")
    pose = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in pose):
        raise ValueError(f"{field_name} must contain finite values.")
    norm = math.sqrt(sum(item * item for item in pose[3:]))
    if abs(norm - 1.0) > 1e-5:
        raise ValueError(f"{field_name} quaternion must be normalized.")
    return pose


@dataclass(frozen=True)
class MotionPrimitiveTargetResolution:
    """One lower-derived absolute target and its immutable semantic evidence."""

    target_pose: tuple[float, ...]
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_pose",
            _pose(self.target_pose, field_name="target_pose"),
        )
        refs = tuple(
            sorted(
                {
                    _required(value, field_name="evidence_ref")
                    for value in self.evidence_refs
                }
            )
        )
        if not refs:
            raise ValueError("A derived TCP target requires semantic evidence.")
        object.__setattr__(self, "evidence_refs", refs)


def resolve_tcp_displacement_primitive_target(
    *,
    measured_pose: tuple[float, ...],
    primitive: TCPPoseDisplacementPrimitive,
    reference_frame_spec: CoordinateFrameSpec,
    frame_graph: CoordinateFrameGraph | None = None,
    measured_reference_frame_id: str | None = None,
) -> MotionPrimitiveTargetResolution:
    """Apply one displacement primitive at the lower activation boundary."""

    if not isinstance(primitive, TCPPoseDisplacementPrimitive):
        raise TypeError("primitive must be a TCP displacement primitive.")
    if not isinstance(reference_frame_spec, CoordinateFrameSpec):
        raise TypeError("reference_frame_spec must be typed.")
    measured_matrix = xyzquat_to_pose_matrix(
        measured_pose,
        name="tcp_displacement_activation_pose",
    )
    anchor, execution_from_primitive, frame_evidence = _resolve_primitive_frames(
        measured_pose=measured_matrix,
        primitive_reference_frame_spec=reference_frame_spec,
        frame_graph=frame_graph,
        measured_reference_frame_id=measured_reference_frame_id,
    )
    target = primitive.apply(
        anchor,
        reference_frame_spec=reference_frame_spec,
    )
    execution_target = (
        target.target_pose
        if execution_from_primitive is None
        else execution_from_primitive @ target.target_pose
    )
    return MotionPrimitiveTargetResolution(
        target_pose=tuple(
            float(value)
            for value in pose_matrix_to_xyzquat(
                execution_target,
                name="tcp_displacement_target_pose",
            )
        ),
        evidence_refs=(
            "tcp-displacement-primitive:" + primitive.semantic_content_hash,
            "tcp-displacement-target:" + target.semantic_content_hash,
            *frame_evidence,
        ),
    )


def resolve_level_tcp_orientation_primitive_target(
    *,
    measured_pose: tuple[float, ...],
    primitive: LevelTCPOrientationPrimitive,
    reference_frame_spec: CoordinateFrameSpec,
    frame_graph: CoordinateFrameGraph | None = None,
    measured_reference_frame_id: str | None = None,
) -> MotionPrimitiveTargetResolution:
    """Replace orientation with one fixed absolute horizontal heading.

    The measured pose contributes translation only.  The primitive replaces
    orientation with its profile-owned absolute heading, so calls never add a
    relative yaw to the current TCP orientation.
    """

    if not isinstance(primitive, LevelTCPOrientationPrimitive):
        raise TypeError("primitive must be a fixed level TCP orientation.")
    if not isinstance(reference_frame_spec, CoordinateFrameSpec):
        raise TypeError("reference_frame_spec must be typed.")
    measured_matrix = xyzquat_to_pose_matrix(
        measured_pose,
        name="tcp_level_orientation_activation_pose",
    )
    anchor, execution_from_primitive, frame_evidence = _resolve_primitive_frames(
        measured_pose=measured_matrix,
        primitive_reference_frame_spec=reference_frame_spec,
        frame_graph=frame_graph,
        measured_reference_frame_id=measured_reference_frame_id,
    )
    target = primitive.apply(
        anchor,
        reference_frame_spec=reference_frame_spec,
    )
    execution_target = (
        target.target_pose
        if execution_from_primitive is None
        else execution_from_primitive @ target.target_pose
    )
    return MotionPrimitiveTargetResolution(
        target_pose=tuple(
            float(value)
            for value in pose_matrix_to_xyzquat(
                execution_target,
                name="tcp_level_orientation_target_pose",
            )
        ),
        evidence_refs=(
            "tcp-level-orientation-primitive:" + primitive.semantic_content_hash,
            "tcp-level-orientation-target:" + target.semantic_content_hash,
            *frame_evidence,
        ),
    )


def _resolve_primitive_frames(
    *,
    measured_pose: Any,
    primitive_reference_frame_spec: CoordinateFrameSpec,
    frame_graph: CoordinateFrameGraph | None,
    measured_reference_frame_id: str | None,
) -> tuple[Any, Any | None, tuple[str, ...]]:
    """Move an execution-frame measurement into and back out of Skill axes."""

    primitive_frame_id = primitive_reference_frame_spec.frame_id
    execution_frame_id = (
        primitive_frame_id
        if measured_reference_frame_id is None
        else _required(
            measured_reference_frame_id,
            field_name="measured_reference_frame_id",
        )
    )
    if execution_frame_id == primitive_frame_id:
        return measured_pose, None, ()
    if not isinstance(frame_graph, CoordinateFrameGraph):
        raise TypeError(
            "frame_graph is required when primitive and measured frames differ."
        )
    registered_spec = frame_graph.frame_by_id.get(primitive_frame_id)
    if (
        registered_spec is None
        or registered_spec.semantic_content_hash
        != primitive_reference_frame_spec.semantic_content_hash
    ):
        raise ValueError(
            "primitive reference frame is not the exact frame-graph member."
        )
    primitive_from_execution = frame_graph.resolve(
        execution_frame_id,
        primitive_frame_id,
    )
    execution_from_primitive = frame_graph.resolve(
        primitive_frame_id,
        execution_frame_id,
    )
    return (
        primitive_from_execution.matrix_target_source @ measured_pose,
        execution_from_primitive.matrix_target_source,
        (
            "tcp-primitive-frame-resolution:"
            + primitive_from_execution.semantic_content_hash,
            "tcp-execution-frame-resolution:"
            + execution_from_primitive.semantic_content_hash,
        ),
    )


__all__ = [
    "MotionPrimitiveTargetResolution",
    "resolve_level_tcp_orientation_primitive_target",
    "resolve_tcp_displacement_primitive_target",
]
