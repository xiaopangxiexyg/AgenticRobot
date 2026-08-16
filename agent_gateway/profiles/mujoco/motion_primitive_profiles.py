"""Immutable MuJoCo bindings for the reviewed 6+3 motion primitives.

This module describes which lower primitive, semantic frame and execution
frame belong to each Agent-visible primitive Route.  It contains no runtime,
observation, safety, IK or dispatch ownership.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from agent_gateway.skills.motion.definition import (
    FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1,
    TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_SKILL_ID_V1,
)
from agent_gateway.skills.motion.schemas import CARDINAL_TCP_STEP_DIRECTIONS_V1
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.primitives import (
    CARDINAL_TCP_SHORT_STEP_PRIMITIVES,
    LevelTCPOrientationPrimitive,
    TCPPoseDisplacementPrimitive,
    cardinal_tcp_short_step,
)

MUJOCO_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1 = (
    "sim.mujoco.rm65.agent_skill_tcp_cardinal_step"
)
MUJOCO_AGENT_TCP_CARDINAL_STEP_EXECUTOR_ID_V1 = (
    "control-station.mujoco-agent-tcp-cardinal-step.v1"
)
MUJOCO_AGENT_TCP_CARDINAL_STEP_PARAMETER_PROFILE_ID_V1 = (
    "right-arm-cardinal-tool-tcp-short-step-v1"
)
MUJOCO_AGENT_TCP_LEVEL_FORWARD_ROUTE_ID_V1 = (
    "sim.mujoco.rm65.agent_skill_tcp_level_forward"
)
MUJOCO_AGENT_TCP_LEVEL_FORWARD_LEFT_30_ROUTE_ID_V1 = (
    "sim.mujoco.rm65.agent_skill_tcp_level_forward_left_30"
)
MUJOCO_AGENT_TCP_LEVEL_FORWARD_RIGHT_30_ROUTE_ID_V1 = (
    "sim.mujoco.rm65.agent_skill_tcp_level_forward_right_30"
)
MUJOCO_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1 = {
    TCP_LEVEL_FORWARD_SKILL_ID_V1: MUJOCO_AGENT_TCP_LEVEL_FORWARD_ROUTE_ID_V1,
    TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1: (
        MUJOCO_AGENT_TCP_LEVEL_FORWARD_LEFT_30_ROUTE_ID_V1
    ),
    TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1: (
        MUJOCO_AGENT_TCP_LEVEL_FORWARD_RIGHT_30_ROUTE_ID_V1
    ),
}
MUJOCO_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1 = {
    TCP_LEVEL_FORWARD_SKILL_ID_V1: "center_forward",
    TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1: "forward_left_30",
    TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1: "forward_right_30",
}
MUJOCO_AGENT_TCP_FIXED_ORIENTATION_EXECUTOR_ID_BY_SKILL_ID_V1 = {
    skill_id: (
        "control-station.mujoco-agent-"
        + skill_id.removeprefix("motion.").replace("_", "-")
        + ".v1"
    )
    for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
}
MUJOCO_AGENT_TCP_FIXED_ORIENTATION_PARAMETER_PROFILE_ID_BY_SKILL_ID_V1 = {
    skill_id: (
        "right-arm-"
        + skill_id.removeprefix("motion.").replace("_", "-")
        + "-v1"
    )
    for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
}
MUJOCO_AGENT_TCP_FIXED_ORIENTATION_MAX_ROTATION_DELTA_RAD_V1 = math.pi
MUJOCO_AGENT_TCP_FIXED_ORIENTATION_MAX_TRANSLATION_DELTA_M_V1 = 1e-6


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


@dataclass(frozen=True)
class MujocoAgentTcpCardinalStepProfile:
    """Exact lower primitive catalog and frame binding for cardinal TCP steps."""

    component_id: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str
    execution_reference_frame_id: str
    execution_reference_frame_spec_hash: str
    primitives: tuple[TCPPoseDisplacementPrimitive, ...] = (
        CARDINAL_TCP_SHORT_STEP_PRIMITIVES
    )

    def __post_init__(self) -> None:
        for field_name in (
            "component_id",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "execution_reference_frame_id",
            "execution_reference_frame_spec_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if self.component_id not in {"left_arm", "right_arm"}:
            raise ValueError("Cardinal-step component must be left_arm or right_arm.")
        primitives = tuple(self.primitives)
        if any(
            not isinstance(item, TCPPoseDisplacementPrimitive)
            for item in primitives
        ):
            raise TypeError("primitives must contain TCP displacement primitives.")
        directions = tuple(item.direction.value for item in primitives)
        if set(directions) != set(CARDINAL_TCP_STEP_DIRECTIONS_V1) or len(
            directions
        ) != len(set(directions)):
            raise ValueError(
                "Cardinal-step profile requires one primitive per direction."
            )
        object.__setattr__(
            self,
            "primitives",
            tuple(sorted(primitives, key=lambda item: item.direction.value)),
        )

    @property
    def primitive_catalog_hash(self) -> str:
        return canonical_content_hash(
            tuple(item.to_payload() for item in self.primitives),
            domain="dexteleop.robot_motion.tcp_displacement_catalog.v1",
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "profile_id": (
                    MUJOCO_AGENT_TCP_CARDINAL_STEP_PARAMETER_PROFILE_ID_V1
                ),
                "component_id": self.component_id,
                "reference_frame_id": self.reference_frame_id,
                "reference_frame_spec_hash": self.reference_frame_spec_hash,
                "controlled_frame_id": "tool_tcp",
                "frame_graph_hash": self.frame_graph_hash,
                "tool_config_hash": self.tool_config_hash,
                "calibration_hash": self.calibration_hash,
                "execution_reference_frame_id": self.execution_reference_frame_id,
                "execution_reference_frame_spec_hash": (
                    self.execution_reference_frame_spec_hash
                ),
                "primitive_catalog_hash": self.primitive_catalog_hash,
                "primitives": tuple(item.to_payload() for item in self.primitives),
                "anchor_semantics": "lower_measured_pose_at_skill_activation",
            },
            domain="dexteleop.agent_skill.mujoco_tcp_cardinal_step_profile.v1",
        )

    def primitive(self, direction: str) -> TCPPoseDisplacementPrimitive:
        selected = cardinal_tcp_short_step(direction)
        matching = tuple(
            item
            for item in self.primitives
            if item.semantic_content_hash == selected.semantic_content_hash
        )
        if len(matching) != 1:
            raise ValueError("Direction is absent from the exact primitive profile.")
        return matching[0]


@dataclass(frozen=True)
class MujocoAgentTcpFixedOrientationProfile:
    """One exact absolute-heading primitive and its semantic/execution frames."""

    skill_id: str
    component_id: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    execution_reference_frame_id: str
    execution_reference_frame_spec_hash: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str
    primitive: LevelTCPOrientationPrimitive

    def __post_init__(self) -> None:
        for field_name in (
            "skill_id",
            "component_id",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "execution_reference_frame_id",
            "execution_reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if self.skill_id not in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1:
            raise ValueError("Unsupported fixed-orientation Skill ID.")
        if self.component_id not in {"left_arm", "right_arm"}:
            raise ValueError(
                "Fixed-orientation component must be left_arm or right_arm."
            )
        if not isinstance(self.primitive, LevelTCPOrientationPrimitive):
            raise TypeError("primitive must be a fixed TCP orientation primitive.")
        expected_heading = (
            MUJOCO_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1[
                self.skill_id
            ]
        )
        if self.primitive.heading.value != expected_heading:
            raise ValueError("Skill ID and fixed-orientation primitive disagree.")

    @property
    def parameter_profile_id(self) -> str:
        return MUJOCO_AGENT_TCP_FIXED_ORIENTATION_PARAMETER_PROFILE_ID_BY_SKILL_ID_V1[
            self.skill_id
        ]

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "profile_id": self.parameter_profile_id,
                "skill_id": self.skill_id,
                "component_id": self.component_id,
                "controlled_frame_id": "tool_tcp",
                "reference_frame_id": self.reference_frame_id,
                "reference_frame_spec_hash": self.reference_frame_spec_hash,
                "execution_reference_frame_id": self.execution_reference_frame_id,
                "execution_reference_frame_spec_hash": (
                    self.execution_reference_frame_spec_hash
                ),
                "frame_graph_hash": self.frame_graph_hash,
                "tool_config_hash": self.tool_config_hash,
                "calibration_hash": self.calibration_hash,
                "primitive": self.primitive.to_payload(),
                "anchor_semantics": "lower_measured_pose_at_skill_activation",
                "target_semantics": "absolute_heading_preserve_translation",
            },
            domain=(
                "dexteleop.agent_skill.mujoco_tcp_fixed_orientation_profile.v1"
            ),
        )


__all__ = [
    "MUJOCO_AGENT_TCP_CARDINAL_STEP_EXECUTOR_ID_V1",
    "MUJOCO_AGENT_TCP_CARDINAL_STEP_PARAMETER_PROFILE_ID_V1",
    "MUJOCO_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1",
    "MUJOCO_AGENT_TCP_FIXED_ORIENTATION_EXECUTOR_ID_BY_SKILL_ID_V1",
    "MUJOCO_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1",
    "MUJOCO_AGENT_TCP_FIXED_ORIENTATION_MAX_ROTATION_DELTA_RAD_V1",
    "MUJOCO_AGENT_TCP_FIXED_ORIENTATION_MAX_TRANSLATION_DELTA_M_V1",
    "MUJOCO_AGENT_TCP_FIXED_ORIENTATION_PARAMETER_PROFILE_ID_BY_SKILL_ID_V1",
    "MUJOCO_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1",
    "MUJOCO_AGENT_TCP_LEVEL_FORWARD_LEFT_30_ROUTE_ID_V1",
    "MUJOCO_AGENT_TCP_LEVEL_FORWARD_RIGHT_30_ROUTE_ID_V1",
    "MUJOCO_AGENT_TCP_LEVEL_FORWARD_ROUTE_ID_V1",
    "MujocoAgentTcpCardinalStepProfile",
    "MujocoAgentTcpFixedOrientationProfile",
]
