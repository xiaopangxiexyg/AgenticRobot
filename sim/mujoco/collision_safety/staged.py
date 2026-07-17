"""Staged projection for independent gripper and arm command progress."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .projection import CollisionProjectionResult, KinematicCollisionProjector


def project_grippers_then_arms(
    scene,
    projector: KinematicCollisionProjector,
    source_data,
    *,
    start_qpos: np.ndarray,
    qpos_by_side: Mapping[str, Sequence[float]],
    gripper_close_by_name: Mapping[str, float],
) -> CollisionProjectionResult:
    """Project gripper closure first, then move arms from that accepted state.

    A saturated/blocked gripper dimension can therefore be rejected without
    preventing a simultaneously safe arm retreat or lift.
    """

    start = np.asarray(start_qpos, dtype=np.float64).reshape(-1)
    combined_requested = scene.kinematic_target_vector(
        start,
        qpos_by_side,
        gripper_close_by_name=gripper_close_by_name,
    )
    gripper_requested = scene.kinematic_target_vector(
        start,
        {},
        gripper_close_by_name=gripper_close_by_name,
    )
    gripper_result = projector.project(
        source_data,
        start_qpos=start,
        requested_qpos=gripper_requested,
    )
    if gripper_result.status == "invalid_safe_state":
        return gripper_result
    arm_requested = scene.kinematic_target_vector(
        gripper_result.accepted_qpos,
        qpos_by_side,
    )
    arm_result = projector.project(
        source_data,
        start_qpos=gripper_result.accepted_qpos,
        requested_qpos=arm_requested,
    )
    if arm_result.status == "invalid_safe_state":
        return arm_result
    modified = not np.array_equal(arm_result.accepted_qpos, combined_requested)
    moved = not np.array_equal(arm_result.accepted_qpos, start)
    if not modified:
        status = "accepted"
    elif moved:
        status = "projected"
    else:
        status = "rejected"
    blocking = (
        gripper_result.blocking_check
        if gripper_result.modified
        else arm_result.blocking_check
    )
    return CollisionProjectionResult(
        status=status,
        requested_qpos=combined_requested,
        accepted_qpos=arm_result.accepted_qpos.copy(),
        projection_alpha=float(
            min(gripper_result.projection_alpha, arm_result.projection_alpha)
        ),
        modified=modified,
        check=arm_result.check,
        sweep_samples=gripper_result.sweep_samples + arm_result.sweep_samples,
        blocking_check=blocking,
    )


__all__ = ["project_grippers_then_arms"]
