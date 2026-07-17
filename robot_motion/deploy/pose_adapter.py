"""Deployment boundary for relative arm actions.

Policy outputs are converted to absolute poses before the existing canonical
safety/IK pipeline.  The adapter only accepts measured TCP anchors from the
current immutable observation; it never falls back to commanded targets.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.pose_frames import PoseFrameContract, PoseFrameContractError

from .actions import NormalizedAction
from .observation import DeploymentObservation
from .schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    STATE_ARM_POSE_SLICES,
    pose_matrix_to_xyzquat,
)
from .session_contract import (
    POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
    POLICY_ABSOLUTE_FRAME_RUNTIME_RAW,
)
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


def _runtime_pose_contract(
    value: PoseFrameContract | Mapping[str, Any] | None,
) -> PoseFrameContract | None:
    if value is None:
        return None
    if isinstance(value, PoseFrameContract):
        return value
    if isinstance(value, Mapping):
        try:
            return PoseFrameContract.from_dict(value)
        except PoseFrameContractError as exc:
            raise PoseDeploymentError(f"invalid runtime pose frame contract: {exc}") from exc
    raise TypeError("runtime_pose_frame_contract must be a PoseFrameContract, mapping, or None.")


def _pose7(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise PoseDeploymentError(f"{name} is not a numeric pose.") from exc
    if array.shape == (4, 4):
        vector = pose_matrix_to_xyzquat(array)
        if vector is None:
            raise PoseDeploymentError(f"{name} is invalid.")
        return np.asarray(vector, dtype=np.float32)
    vector = array.reshape(-1)
    if vector.shape != (7,) or not np.all(np.isfinite(vector)):
        raise PoseDeploymentError(f"{name} must be one finite pose7 or 4x4 pose.")
    return vector.copy()


def _raw_pose_to_desk(
    value: Any,
    *,
    side: str,
    contract: PoseFrameContract,
    name: str,
) -> np.ndarray:
    try:
        return np.asarray(
            contract.storage_pose_vector_to_canonical_desk(
                _pose7(value, name=name), side=side
            ),
            dtype=np.float32,
        )
    except PoseFrameContractError as exc:
        raise PoseDeploymentError(f"{name} cannot be converted raw->desk: {exc}") from exc


def _desk_pose_to_raw(
    value: Any,
    *,
    side: str,
    contract: PoseFrameContract,
    name: str,
) -> np.ndarray:
    try:
        return np.asarray(
            contract.canonical_desk_pose_vector_to_storage(
                _pose7(value, name=name), side=side
            ),
            dtype=np.float32,
        )
    except PoseFrameContractError as exc:
        raise PoseDeploymentError(f"{name} cannot be converted desk->raw: {exc}") from exc


def convert_policy_absolute_vector_to_runtime_raw(
    value: Any,
    *,
    active_sides: Sequence[str],
    policy_absolute_frame: str = POLICY_ABSOLUTE_FRAME_RUNTIME_RAW,
    runtime_pose_frame_contract: PoseFrameContract | Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Convert one dense absolute policy action into the execution raw frame."""

    vector = np.asarray(value, dtype=np.float32).reshape(-1).copy()
    if vector.shape != (ACTION_DIM,):
        raise PoseDeploymentError(
            f"policy absolute action must have shape ({ACTION_DIM},), got {vector.shape}."
        )
    absolute_frame = str(policy_absolute_frame).strip().lower()
    if absolute_frame == POLICY_ABSOLUTE_FRAME_RUNTIME_RAW:
        return vector
    if absolute_frame != POLICY_ABSOLUTE_FRAME_CANONICAL_DESK:
        raise PoseDeploymentError(
            f"unsupported policy_absolute_frame {policy_absolute_frame!r}."
        )
    runtime_contract = _runtime_pose_contract(runtime_pose_frame_contract)
    if runtime_contract is None:
        raise PoseDeploymentError(
            "canonical_desk policy actions require a runtime pose frame contract."
        )
    for side in tuple(dict.fromkeys(str(item).strip().lower() for item in active_sides)):
        pose_slice = ACTION_ARM_POSE_SLICES[side]
        if np.all(np.isfinite(vector[pose_slice])):
            vector[pose_slice] = _desk_pose_to_raw(
                vector[pose_slice],
                side=side,
                contract=runtime_contract,
                name=f"policy_action.{side}",
            )
    return vector


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
            array = np.asarray(raw, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise PoseDeploymentError(f"{side} actual_tcp_pose is unavailable.") from exc
        if array.shape == (4, 4):
            try:
                vector = pose_matrix_to_xyzquat(array)
            except (TypeError, ValueError) as exc:
                raise PoseDeploymentError(f"{side} actual_tcp_pose is invalid.") from exc
            if vector is None:
                raise PoseDeploymentError(f"{side} actual_tcp_pose is invalid.")
        else:
            vector = array.reshape(-1)
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
    policy_absolute_frame: str = POLICY_ABSOLUTE_FRAME_RUNTIME_RAW,
    runtime_pose_frame_contract: PoseFrameContract | Mapping[str, Any] | None = None,
) -> list[NormalizedAction]:
    """Convert a model action chunk to runtime-raw absolute before safety/IK."""

    source = normalize_action_pose_mode(source_mode)
    normalized_sides = (
        observation.active_sides
        if active_sides is None
        else tuple(str(value).strip().lower() for value in active_sides)
    )
    absolute_frame = str(policy_absolute_frame).strip().lower()
    if absolute_frame not in {
        POLICY_ABSOLUTE_FRAME_RUNTIME_RAW,
        POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
    }:
        raise PoseDeploymentError(
            f"unsupported policy_absolute_frame {policy_absolute_frame!r}."
        )
    runtime_contract = _runtime_pose_contract(runtime_pose_frame_contract)
    if (
        absolute_frame == POLICY_ABSOLUTE_FRAME_CANONICAL_DESK
        and runtime_contract is None
    ):
        raise PoseDeploymentError(
            "canonical_desk policy actions require a runtime pose frame contract."
        )
    if source == ACTION_POSE_MODE_ABSOLUTE:
        output: list[NormalizedAction] = []
        for action in actions:
            vector = convert_policy_absolute_vector_to_runtime_raw(
                action.vector,
                active_sides=normalized_sides,
                policy_absolute_frame=absolute_frame,
                runtime_pose_frame_contract=runtime_contract,
            )
            output.append(
                NormalizedAction(
                    vector=vector,
                    arm_joint_targets=dict(action.arm_joint_targets),
                    metadata={
                        **dict(action.metadata),
                        "pose_action_source_mode": source,
                        "pose_action_mode": "absolute",
                        "pose_action_source_absolute_frame": absolute_frame,
                        "pose_action_output_absolute_frame": POLICY_ABSOLUTE_FRAME_RUNTIME_RAW,
                    },
                )
            )
        return output
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
                    "pose_action_source_absolute_frame": absolute_frame,
                    "pose_action_output_absolute_frame": POLICY_ABSOLUTE_FRAME_RUNTIME_RAW,
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
    policy_absolute_frame: str = POLICY_ABSOLUTE_FRAME_RUNTIME_RAW,
    runtime_pose_frame_contract: PoseFrameContract | Mapping[str, Any] | None = None,
) -> DeploymentObservation:
    """Return an immutable model view in raw-legacy or canonical desk space.

    Startup observations may intentionally carry an all-NaN last action.  Such
    unknown slots remain unknown; every finite arm-pose slot is converted with
    the current measured TCP anchor, preserving the existing validity contract.
    """

    target = normalize_action_pose_mode(target_mode)
    absolute_frame = str(policy_absolute_frame).strip().lower()
    if absolute_frame not in {
        POLICY_ABSOLUTE_FRAME_RUNTIME_RAW,
        POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
    }:
        raise PoseDeploymentError(
            f"unsupported policy_absolute_frame {policy_absolute_frame!r}."
        )
    runtime_contract = _runtime_pose_contract(runtime_pose_frame_contract)
    canonical_desk = absolute_frame == POLICY_ABSOLUTE_FRAME_CANONICAL_DESK
    if canonical_desk and runtime_contract is None:
        raise PoseDeploymentError(
            "canonical_desk policy observation requires a runtime pose frame contract."
        )
    sides = (
        observation.active_sides
        if active_sides is None
        else tuple(str(value).strip().lower() for value in active_sides)
    )
    policy_observation = observation
    if canonical_desk:
        state = np.asarray(observation.state, dtype=np.float32).copy()
        state_dict = deepcopy(observation.state_dict)
        arm_states = deepcopy(observation.arm_states)
        for side in sides:
            state_slice = STATE_ARM_POSE_SLICES[side]
            if np.all(np.isfinite(state[state_slice])):
                desk_pose = _raw_pose_to_desk(
                    state[state_slice],
                    side=side,
                    contract=runtime_contract,
                    name=f"observation.state.{side}",
                )
                state[state_slice] = desk_pose
                state_dict[f"{side}_arm.pose"] = desk_pose.copy()
                nested = state_dict.get(f"{side}_arm")
                if isinstance(nested, Mapping):
                    nested_copy = dict(nested)
                    nested_copy["pose"] = desk_pose.copy()
                    state_dict[f"{side}_arm"] = nested_copy
            side_state = arm_states.get(side)
            if isinstance(side_state, Mapping):
                side_copy = dict(side_state)
                for key in ("actual_tcp_pose", "current_tcp_pose"):
                    value = side_copy.get(key)
                    if value is None:
                        continue
                    side_copy[f"runtime_raw_{key}"] = deepcopy(value)
                    side_copy[key] = _raw_pose_to_desk(
                        value,
                        side=side,
                        contract=runtime_contract,
                        name=f"arm_states.{side}.{key}",
                    ).tolist()
                arm_states[side] = side_copy
        raw_last = np.asarray(observation.last_action, dtype=np.float32).reshape(-1)
        if raw_last.shape != (ACTION_DIM,):
            raise PoseDeploymentError(
                f"last_action must have shape ({ACTION_DIM},), got {raw_last.shape}."
            )
        absolute_last = raw_last.copy()
        for side in sides:
            pose_slice = ACTION_ARM_POSE_SLICES[side]
            if np.all(np.isfinite(absolute_last[pose_slice])):
                absolute_last[pose_slice] = _raw_pose_to_desk(
                    absolute_last[pose_slice],
                    side=side,
                    contract=runtime_contract,
                    name=f"last_action.{side}",
                )
        metadata = dict(observation.metadata)
        metadata.update(
            {
                "policy_absolute_frame": POLICY_ABSOLUTE_FRAME_CANONICAL_DESK,
                "runtime_pose_frame_contract": runtime_contract.to_dict(),
                "runtime_pose_frame_contract_hash": runtime_contract.contract_hash,
            }
        )
        policy_observation = replace(
            observation,
            state=state,
            state_dict=state_dict,
            arm_states=arm_states,
            last_action=absolute_last,
            metadata=metadata,
        )

    vector = np.asarray(policy_observation.last_action, dtype=np.float32).reshape(-1)
    if vector.shape != (ACTION_DIM,):
        raise PoseDeploymentError(f"last_action must have shape ({ACTION_DIM},), got {vector.shape}.")
    if target == ACTION_POSE_MODE_ABSOLUTE:
        converted = vector.copy()
    else:
        valid_sides = tuple(
            side
            for side in sides
            if np.asarray(vector[ACTION_ARM_POSE_SLICES[side]], dtype=np.float32).shape == (7,)
            and np.all(np.isfinite(vector[ACTION_ARM_POSE_SLICES[side]]))
        )
        if valid_sides:
            converted = convert_last_action_to_policy_mode(
                vector,
                policy_observation,
                target_mode=target,
                active_sides=valid_sides,
            )
        else:
            converted = vector.copy()
    metadata = dict(policy_observation.metadata)
    metadata["policy_pose_action_mode"] = target
    metadata["policy_last_action_pose_mode"] = target
    metadata["policy_absolute_frame"] = absolute_frame
    return replace(policy_observation, last_action=converted, metadata=metadata)


__all__ = [
    "PoseDeploymentError",
    "convert_last_action_to_policy_mode",
    "convert_policy_absolute_vector_to_runtime_raw",
    "convert_normalized_action_chunk_to_absolute",
    "convert_normalized_action_chunk_to_relative",
    "measured_tcp_anchors_by_side",
    "prepare_observation_for_policy_pose_mode",
]
