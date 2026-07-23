"""Immutable request-time action-representation context for raw RTC."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.relative_action import (
    ACTION_POSE_MODE_ABSOLUTE,
    ACTION_POSE_MODE_RELATIVE,
    normalize_action_pose_mode,
    pose_matrix_to_xyzquat,
    xyzquat_to_pose_matrix,
)


POLICY_ACTION_REPRESENTATION_CONTEXT_SCHEMA_V1 = (
    "policy_action_representation_context_v1"
)
POLICY_ACTION_REPRESENTATION_CONTEXT_HASH_DOMAIN_V1 = (
    "dexteleop.policy_action_representation_context.v1"
)
ACTION_ANCHOR_POLICY_NONE_V1 = "none_v1"
ACTION_ANCHOR_POLICY_REQUEST_ACTUAL_TCP_FIXED_V1 = (
    "request_observation_actual_tcp_fixed_for_chunk_v1"
)
ACTION_POSE_TRANSFORM_CONVENTION_SE3_LEFT_V1 = (
    "left_multiply_se3_xyzquat_xyzw_v1"
)


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _active_sides(values: Sequence[str]) -> tuple[str, ...]:
    raw = tuple(str(value).strip().lower() for value in values)
    if not raw or any(value not in {"left", "right"} for value in raw):
        raise ValueError("active_sides must contain left and/or right.")
    if len(raw) != len(set(raw)):
        raise ValueError("active_sides must not contain duplicates.")
    selected = set(raw)
    return tuple(side for side in ("left", "right") if side in selected)


def _finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


def _side_text_mapping(
    values: Mapping[str, object],
    *,
    active_sides: tuple[str, ...],
    name: str,
) -> Mapping[str, str]:
    normalized = {
        str(side).strip().lower(): _nonempty(value, name=f"{name}[{side!r}]")
        for side, value in dict(values).items()
    }
    if set(normalized) != set(active_sides):
        raise ValueError(f"{name} must exactly cover active_sides.")
    return MappingProxyType({side: normalized[side] for side in active_sides})


@dataclass(frozen=True)
class PolicyActionRepresentationContext:
    """One request-bound absolute/no-anchor or relative/fixed-anchor context."""

    source_pose_mode: str
    observation_cut_id: str
    observation_contract_hash: str
    observation_timestamp: float
    observation_step_index: int
    active_sides: tuple[str, ...]
    anchor_policy: str
    arm_reference_frame_by_side: Mapping[str, str]
    arm_controlled_frame_by_side: Mapping[str, str]
    arm_anchor_pose_by_side: Mapping[str, Any]
    pose_transform_convention: str = ACTION_POSE_TRANSFORM_CONVENTION_SE3_LEFT_V1
    schema_id: str = POLICY_ACTION_REPRESENTATION_CONTEXT_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema_id != POLICY_ACTION_REPRESENTATION_CONTEXT_SCHEMA_V1:
            raise ValueError(
                f"Unsupported action representation context schema: {self.schema_id!r}."
            )
        mode = normalize_action_pose_mode(self.source_pose_mode)
        sides = _active_sides(self.active_sides)
        reference_frames = _side_text_mapping(
            self.arm_reference_frame_by_side,
            active_sides=sides,
            name="arm_reference_frame_by_side",
        )
        controlled_frames = _side_text_mapping(
            self.arm_controlled_frame_by_side,
            active_sides=sides,
            name="arm_controlled_frame_by_side",
        )
        anchor_policy = _nonempty(self.anchor_policy, name="anchor_policy")
        convention = _nonempty(
            self.pose_transform_convention,
            name="pose_transform_convention",
        )
        if convention != ACTION_POSE_TRANSFORM_CONVENTION_SE3_LEFT_V1:
            raise ValueError(f"Unsupported pose transform convention: {convention!r}.")

        raw_anchors = {
            str(side).strip().lower(): value
            for side, value in dict(self.arm_anchor_pose_by_side).items()
        }
        anchors: dict[str, np.ndarray] = {}
        if mode == ACTION_POSE_MODE_ABSOLUTE:
            if anchor_policy != ACTION_ANCHOR_POLICY_NONE_V1 or raw_anchors:
                raise ValueError(
                    "Absolute representation context requires explicit no-anchor semantics."
                )
        else:
            if anchor_policy != ACTION_ANCHOR_POLICY_REQUEST_ACTUAL_TCP_FIXED_V1:
                raise ValueError(
                    "Relative representation context requires fixed request anchor policy."
                )
            if set(raw_anchors) != set(sides):
                raise ValueError(
                    "Relative arm anchors must exactly cover active_sides."
                )
            for side in sides:
                try:
                    pose = pose_matrix_to_xyzquat(
                        xyzquat_to_pose_matrix(
                            raw_anchors[side],
                            name=f"{side}.request_anchor",
                        ),
                        name=f"{side}.request_anchor",
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid relative request anchor for {side!r}: {exc}"
                    ) from exc
                pose = np.asarray(pose, dtype=np.float32).copy()
                pose.setflags(write=False)
                anchors[side] = pose

        object.__setattr__(self, "source_pose_mode", mode)
        object.__setattr__(
            self,
            "observation_cut_id",
            _nonempty(self.observation_cut_id, name="observation_cut_id"),
        )
        object.__setattr__(
            self,
            "observation_contract_hash",
            _nonempty(
                self.observation_contract_hash,
                name="observation_contract_hash",
            ),
        )
        object.__setattr__(
            self,
            "observation_timestamp",
            _finite(self.observation_timestamp, name="observation_timestamp"),
        )
        object.__setattr__(
            self,
            "observation_step_index",
            _nonnegative_integer(
                self.observation_step_index,
                name="observation_step_index",
            ),
        )
        object.__setattr__(self, "active_sides", sides)
        object.__setattr__(self, "anchor_policy", anchor_policy)
        object.__setattr__(self, "arm_reference_frame_by_side", reference_frames)
        object.__setattr__(self, "arm_controlled_frame_by_side", controlled_frames)
        object.__setattr__(
            self,
            "arm_anchor_pose_by_side",
            MappingProxyType(anchors),
        )
        object.__setattr__(self, "pose_transform_convention", convention)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "schema_id": self.schema_id,
                "source_pose_mode": self.source_pose_mode,
                "observation_cut_id": self.observation_cut_id,
                "observation_contract_hash": self.observation_contract_hash,
                "observation_timestamp": self.observation_timestamp,
                "observation_step_index": self.observation_step_index,
                "active_sides": self.active_sides,
                "anchor_policy": self.anchor_policy,
                "arm_reference_frame_by_side": self.arm_reference_frame_by_side,
                "arm_controlled_frame_by_side": self.arm_controlled_frame_by_side,
                "arm_anchor_pose_by_side": self.arm_anchor_pose_by_side,
                "pose_transform_convention": self.pose_transform_convention,
            },
            domain=POLICY_ACTION_REPRESENTATION_CONTEXT_HASH_DOMAIN_V1,
        )

    @property
    def anchor_identity_hash(self) -> str | None:
        if self.source_pose_mode == ACTION_POSE_MODE_ABSOLUTE:
            return None
        return self.semantic_content_hash


def build_policy_action_representation_context(
    observation: Any,
    observation_cut_id: str,
    *,
    source_pose_mode: str,
    active_sides: Sequence[str],
    arm_reference_frame_by_side: Mapping[str, str],
    arm_controlled_frame_by_side: Mapping[str, str],
) -> PolicyActionRepresentationContext:
    """Build one context from the exact immutable actor observation cut."""

    from .observation_view import PolicyObservationView
    from .pose_adapter import measured_tcp_anchors_by_side

    if not isinstance(observation, PolicyObservationView):
        raise TypeError("observation must be a PolicyObservationView.")
    mode = normalize_action_pose_mode(source_pose_mode)
    sides = _active_sides(active_sides)
    anchors: Mapping[str, Any]
    anchor_policy: str
    if mode == ACTION_POSE_MODE_RELATIVE:
        anchors = measured_tcp_anchors_by_side(
            observation.to_deployment_observation(),
            sides,
        )
        anchor_policy = ACTION_ANCHOR_POLICY_REQUEST_ACTUAL_TCP_FIXED_V1
    else:
        anchors = {}
        anchor_policy = ACTION_ANCHOR_POLICY_NONE_V1
    return PolicyActionRepresentationContext(
        source_pose_mode=mode,
        observation_cut_id=observation_cut_id,
        observation_contract_hash=observation.contract_hash,
        observation_timestamp=float(observation.payload["timestamp"]),
        observation_step_index=int(observation.payload["step_index"]),
        active_sides=sides,
        anchor_policy=anchor_policy,
        arm_reference_frame_by_side=arm_reference_frame_by_side,
        arm_controlled_frame_by_side=arm_controlled_frame_by_side,
        arm_anchor_pose_by_side=anchors,
    )


__all__ = [
    "ACTION_ANCHOR_POLICY_NONE_V1",
    "ACTION_ANCHOR_POLICY_REQUEST_ACTUAL_TCP_FIXED_V1",
    "ACTION_POSE_TRANSFORM_CONVENTION_SE3_LEFT_V1",
    "POLICY_ACTION_REPRESENTATION_CONTEXT_HASH_DOMAIN_V1",
    "POLICY_ACTION_REPRESENTATION_CONTEXT_SCHEMA_V1",
    "PolicyActionRepresentationContext",
    "build_policy_action_representation_context",
]
