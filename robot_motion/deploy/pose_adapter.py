"""Deployment boundary for relative arm actions.

Policy outputs are converted to absolute poses before the existing canonical
safety/IK pipeline.  The adapter only accepts measured TCP anchors from the
current immutable observation; it never falls back to commanded targets.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.pose_frames import PoseFrameContract, PoseFrameContractError

from .actions import NormalizedAction
from .observation import DeploymentObservation
from .schema import ACTION_ARM_POSE_SLICES, ACTION_DIM
from ..relative_action import (
    ACTION_POSE_MODE_ABSOLUTE,
    ACTION_POSE_MODE_RELATIVE,
    RelativeActionError,
    convert_arm_pose_chunk,
    convert_last_action_chunk,
    normalize_action_pose_mode,
)


class PoseDeploymentError(ValueError):
    """Raised when a deployment pose representation cannot be resolved safely."""


def _frame_contract_from_observation(observation: DeploymentObservation) -> PoseFrameContract | None:
    value = observation.metadata.get("pose_frame_contract")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise PoseDeploymentError("observation pose_frame_contract must be a mapping.")
    try:
        return PoseFrameContract.from_dict(value)
    except PoseFrameContractError as exc:
        raise PoseDeploymentError(f"invalid observation pose_frame_contract: {exc}") from exc


def measured_tcp_anchors_by_side(
    observation: DeploymentObservation,
    active_sides: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return finite actual TCP anchors in the canonical desk frame."""

    contract = _frame_contract_from_observation(observation)
    anchors: dict[str, np.ndarray] = {}
    for side in tuple(dict.fromkeys(str(value).strip().lower() for value in active_sides)):
        state = observation.arm_states.get(side)
        if not isinstance(state, Mapping):
            raise PoseDeploymentError(f"missing immutable arm snapshot for side {side!r}.")
        raw = state.get("actual_tcp_pose")
        try:
            vector = np.asarray(raw, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise PoseDeploymentError(f"{side} actual_tcp_pose is unavailable.") from exc
        if vector.shape != (7,) or not np.all(np.isfinite(vector)):
            raise PoseDeploymentError(f"{side} actual_tcp_pose is invalid.")
        if contract is not None:
            try:
                vector = contract.storage_pose_vector_to_desk(vector, side=side)
            except PoseFrameContractError as exc:
                raise PoseDeploymentError(
                    f"{side} actual_tcp_pose cannot be converted to desk frame: {exc}"
                ) from exc
        anchors[side] = np.asarray(vector, dtype=np.float32).copy()
    return anchors


def convert_normalized_action_chunk_to_absolute(
    actions: Sequence[NormalizedAction],
    observation: DeploymentObservation,
    *,
    source_mode: str = ACTION_POSE_MODE_ABSOLUTE,
    active_sides: Sequence[str] | None = None,
) -> list[NormalizedAction]:
    """Convert a model action chunk to absolute pose before canonical safety."""

    source = normalize_action_pose_mode(source_mode)
    normalized_sides = (
        observation.active_sides
        if active_sides is None
        else tuple(str(value).strip().lower() for value in active_sides)
    )
    if source == ACTION_POSE_MODE_ABSOLUTE:
        return [
            NormalizedAction(
                vector=np.asarray(action.vector, dtype=np.float32).copy(),
                arm_joint_targets=dict(action.arm_joint_targets),
                metadata={**dict(action.metadata), "pose_action_source_mode": source, "pose_action_mode": "absolute"},
            )
            for action in actions
        ]
    if not actions:
        return []
    anchors = measured_tcp_anchors_by_side(observation, normalized_sides)
    values = np.stack(
        [np.asarray(action.vector, dtype=np.float32).reshape(-1) for action in actions], axis=0
    )
    if values.shape[1] != ACTION_DIM:
        raise PoseDeploymentError(
            f"deployment action chunk must have width {ACTION_DIM}, got {values.shape}."
        )
    try:
        converted = convert_arm_pose_chunk(
            values,
            anchors_by_side=anchors,
            source_mode=source,
            target_mode=ACTION_POSE_MODE_ABSOLUTE,
            arm_pose_slices=ACTION_ARM_POSE_SLICES,
            active_sides=normalized_sides,
        )
    except RelativeActionError as exc:
        raise PoseDeploymentError(f"relative deployment action conversion failed: {exc}") from exc
    output: list[NormalizedAction] = []
    for index, action in enumerate(actions):
        output.append(
            NormalizedAction(
                vector=converted[index].copy(),
                arm_joint_targets=dict(action.arm_joint_targets),
                metadata={
                    **dict(action.metadata),
                    "pose_action_source_mode": source,
                    "pose_action_mode": "absolute",
                    "pose_anchor_policy": "current_actual_tcp_fixed_for_chunk",
                    "pose_anchor_sides": list(normalized_sides),
                },
            )
        )
    return output


def convert_normalized_action_chunk_to_relative(
    actions: Sequence[NormalizedAction],
    observation: DeploymentObservation,
    *,
    active_sides: Sequence[str] | None = None,
) -> list[NormalizedAction]:
    """Express an absolute policy chunk in the current fixed TCP frames.

    This is primarily the v2.2 replay/debug bridge: the source dataset remains
    absolute, while the policy-facing values are calculated at prediction time
    from the measured TCP snapshot at the start of the chunk.  The deployment
    adapter can then decode the values back to the same absolute targets.
    """

    if not actions:
        return []
    normalized_sides = (
        observation.active_sides
        if active_sides is None
        else tuple(str(value).strip().lower() for value in active_sides)
    )
    anchors = measured_tcp_anchors_by_side(observation, normalized_sides)
    values = np.stack(
        [np.asarray(action.vector, dtype=np.float32).reshape(-1) for action in actions], axis=0
    )
    if values.shape[1] != ACTION_DIM:
        raise PoseDeploymentError(
            f"deployment action chunk must have width {ACTION_DIM}, got {values.shape}."
        )
    try:
        converted = convert_arm_pose_chunk(
            values,
            anchors_by_side=anchors,
            source_mode=ACTION_POSE_MODE_ABSOLUTE,
            target_mode=ACTION_POSE_MODE_RELATIVE,
            arm_pose_slices=ACTION_ARM_POSE_SLICES,
            active_sides=normalized_sides,
        )
    except RelativeActionError as exc:
        raise PoseDeploymentError(f"absolute replay action conversion failed: {exc}") from exc
    return [
        NormalizedAction(
            vector=converted[index].copy(),
            arm_joint_targets=dict(action.arm_joint_targets),
            metadata={
                **dict(action.metadata),
                "pose_action_source_mode": ACTION_POSE_MODE_ABSOLUTE,
                "pose_action_mode": ACTION_POSE_MODE_RELATIVE,
                "pose_anchor_policy": "current_actual_tcp_fixed_for_chunk",
                "pose_anchor_sides": list(normalized_sides),
            },
        )
        for index, action in enumerate(actions)
    ]


def convert_last_action_to_policy_mode(
    last_action: Any,
    observation: DeploymentObservation,
    *,
    target_mode: str = ACTION_POSE_MODE_ABSOLUTE,
    active_sides: Sequence[str] | None = None,
) -> np.ndarray:
    """Convert the previous absolute action using the current TCP anchor."""

    target = normalize_action_pose_mode(target_mode)
    vector = np.asarray(last_action, dtype=np.float32).reshape(-1)
    if vector.shape != (ACTION_DIM,):
        raise PoseDeploymentError(f"last_action must have shape ({ACTION_DIM},), got {vector.shape}.")
    if target == ACTION_POSE_MODE_ABSOLUTE:
        return vector.copy()
    sides = observation.active_sides if active_sides is None else tuple(active_sides)
    anchors = measured_tcp_anchors_by_side(observation, sides)
    try:
        return convert_last_action_chunk(
            vector.reshape(1, -1),
            current_anchors_by_side={side: value for side, value in anchors.items()},
            source_mode=ACTION_POSE_MODE_ABSOLUTE,
            target_mode=target,
            arm_pose_slices=ACTION_ARM_POSE_SLICES,
            active_sides=sides,
        )[0]
    except RelativeActionError as exc:
        raise PoseDeploymentError(f"last_action pose conversion failed: {exc}") from exc


def prepare_observation_for_policy_pose_mode(
    observation: DeploymentObservation,
    *,
    target_mode: str = ACTION_POSE_MODE_ABSOLUTE,
    active_sides: Sequence[str] | None = None,
) -> DeploymentObservation:
    """Return an immutable observation whose ``last_action`` matches policy mode.

    Startup observations may intentionally carry an all-NaN last action.  Such
    unknown slots remain unknown; every finite arm-pose slot is converted with
    the current measured TCP anchor, preserving the existing validity contract.
    """

    target = normalize_action_pose_mode(target_mode)
    if target == ACTION_POSE_MODE_ABSOLUTE:
        return observation
    vector = np.asarray(observation.last_action, dtype=np.float32).reshape(-1)
    if vector.shape != (ACTION_DIM,):
        raise PoseDeploymentError(f"last_action must have shape ({ACTION_DIM},), got {vector.shape}.")
    sides = (
        observation.active_sides
        if active_sides is None
        else tuple(str(value).strip().lower() for value in active_sides)
    )
    valid_sides = tuple(
        side
        for side in sides
        if np.asarray(vector[ACTION_ARM_POSE_SLICES[side]], dtype=np.float32).shape == (7,)
        and np.all(np.isfinite(vector[ACTION_ARM_POSE_SLICES[side]]))
    )
    if valid_sides:
        converted = convert_last_action_to_policy_mode(
            vector,
            observation,
            target_mode=target,
            active_sides=valid_sides,
        )
    else:
        converted = vector.copy()
    metadata = dict(observation.metadata)
    metadata["policy_pose_action_mode"] = target
    metadata["policy_last_action_pose_mode"] = target
    return replace(observation, last_action=converted, metadata=metadata)


__all__ = [
    "PoseDeploymentError",
    "convert_last_action_to_policy_mode",
    "convert_normalized_action_chunk_to_absolute",
    "convert_normalized_action_chunk_to_relative",
    "measured_tcp_anchors_by_side",
    "prepare_observation_for_policy_pose_mode",
]
