from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from openpi05_deployment.errors import InferenceResultError
from openpi05_deployment.metadata import ServerContract
from teleop_runtime.deploy.actions import NormalizedAction
from teleop_runtime.deploy.observation import DeploymentObservation
from teleop_runtime.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM, ACTION_HAND_SLICES


def _safety_overrides(disable_rate_limits: bool) -> dict[str, float]:
    if not disable_rate_limits:
        return {}
    return {
        "max_arm_xyz_step_m": 0.0,
        "max_arm_rot_step_deg": 0.0,
        "max_arm_joint_step_rad": 0.0,
    }


def compact_action_to_full_vector(
    compact_action: np.ndarray,
    contract: ServerContract,
    *,
    seed_vector: np.ndarray | None = None,
) -> np.ndarray:
    if seed_vector is None:
        vector = np.full((ACTION_DIM,), np.nan, dtype=np.float32)
    else:
        vector = np.asarray(seed_vector, dtype=np.float32).reshape(ACTION_DIM).copy()
    sides = contract.embodiment.active_sides
    expected_dim = contract.embodiment.output_action_dim
    if compact_action.shape != (expected_dim,):
        raise InferenceResultError(
            f"OpenPI compact action shape mismatch: expected {(expected_dim,)}, got {compact_action.shape}."
        )
    offset = 0
    for side in sides:
        pose_slice = ACTION_ARM_POSE_SLICES[side]
        pose_dim = pose_slice.stop - pose_slice.start
        vector[pose_slice] = compact_action[offset : offset + pose_dim]
        offset += pose_dim
    for side in sides:
        hand_slice = ACTION_HAND_SLICES[side]
        hand_dim = hand_slice.stop - hand_slice.start
        vector[hand_slice] = compact_action[offset : offset + hand_dim]
        offset += hand_dim
    return vector


def adapt_openpi_result(
    result: Mapping[str, Any],
    observation: DeploymentObservation,
    contract: ServerContract,
    *,
    disable_deploy_rate_limits: bool = True,
) -> list[NormalizedAction]:
    if "actions" not in result:
        raise InferenceResultError("OpenPI response is missing the 'actions' field.")
    actions = np.asarray(result["actions"], dtype=np.float32)
    if actions.ndim != 2:
        raise InferenceResultError(f"OpenPI actions must be rank-2, got shape {actions.shape}.")
    if actions.shape[0] <= 0:
        raise InferenceResultError("OpenPI actions must contain at least one step.")
    if actions.shape[1] != contract.embodiment.output_action_dim:
        raise InferenceResultError(
            "OpenPI action width mismatch: "
            f"expected {contract.embodiment.output_action_dim}, got {actions.shape[1]}."
        )
    if actions.shape[0] != contract.action_horizon:
        raise InferenceResultError(
            f"OpenPI action horizon mismatch: expected {contract.action_horizon}, got {actions.shape[0]}."
        )
    if not np.all(np.isfinite(actions)):
        raise InferenceResultError("OpenPI actions contain non-finite values.")

    timing_metadata = {}
    if isinstance(result.get("policy_timing"), Mapping):
        timing_metadata["policy_timing"] = dict(result["policy_timing"])
    if isinstance(result.get("server_timing"), Mapping):
        timing_metadata["server_timing"] = dict(result["server_timing"])

    overrides = _safety_overrides(bool(disable_deploy_rate_limits))
    chunk: list[NormalizedAction] = []
    for index in range(actions.shape[0]):
        vector = compact_action_to_full_vector(
            actions[index],
            contract,
            seed_vector=observation.hold_action(),
        )
        metadata: dict[str, Any] = {"source": "openpi"}
        if overrides:
            metadata["safety_overrides"] = dict(overrides)
        if timing_metadata:
            metadata["openpi"] = dict(timing_metadata)
        metadata["chunk_index"] = int(index)
        chunk.append(
            NormalizedAction(
                vector=vector.astype(np.float32, copy=True),
                metadata=metadata,
            )
        )
    return chunk
