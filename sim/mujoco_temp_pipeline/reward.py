"""Temporary reward shaping computed only from the existing 31-D observation."""

from __future__ import annotations

import numpy as np

from .config import RewardShapingConfig


CUBE_TO_GRASP_SLICE = slice(20, 23)
GRIPPER_CLOSE_INDEX = 6
CUBE_Z_INDEX = 9
LAST_ACTION_SLICE = slice(23, 30)


def shape_rewards(
    observations: np.ndarray,
    next_observations: np.ndarray,
    raw_rewards: np.ndarray,
    config: RewardShapingConfig,
    safety_projection_alpha: np.ndarray | None = None,
    resting_cube_z_m: np.ndarray | float | None = None,
    lift_delta_m: float = 0.08,
    success: np.ndarray | None = None,
    grasp_latched: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute staged reach, grasp and lift rewards from the 31-D state."""

    previous = np.asarray(observations, dtype=np.float32)
    current = np.asarray(next_observations, dtype=np.float32)
    raw = np.asarray(raw_rewards, dtype=np.float32).reshape(-1)
    if previous.shape != current.shape or previous.ndim != 2 or previous.shape[1] != 31:
        raise ValueError("Reward shaping expects matching (N, 31) observations.")
    if raw.shape != (previous.shape[0],):
        raise ValueError("Reward shaping expects one raw reward per environment.")

    if lift_delta_m <= 0.0:
        raise ValueError("lift_delta_m must be positive.")
    if resting_cube_z_m is None:
        resting_z = previous[:, CUBE_Z_INDEX]
    else:
        resting_z = np.broadcast_to(
            np.asarray(resting_cube_z_m, dtype=np.float32), raw.shape
        )
    if success is None:
        succeeded = np.zeros_like(raw, dtype=np.float32)
    else:
        succeeded = np.asarray(success, dtype=np.float32).reshape(-1)
        if succeeded.shape != raw.shape:
            raise ValueError("success must match the reward batch.")
    if grasp_latched is None:
        latched = np.zeros_like(raw, dtype=np.float32)
    else:
        latched = np.asarray(grasp_latched, dtype=np.float32).reshape(-1)
        if latched.shape != raw.shape:
            raise ValueError("grasp_latched must match the reward batch.")
        latched = np.clip(latched, 0.0, 1.0)
    pre_latch = 1.0 - latched

    previous_gripper_close = np.clip(
        previous[:, GRIPPER_CLOSE_INDEX], 0.0, 1.0
    )
    gripper_close = np.clip(current[:, GRIPPER_CLOSE_INDEX], 0.0, 1.0)
    previous_error = previous[:, CUBE_TO_GRASP_SLICE].copy()
    current_error = current[:, CUBE_TO_GRASP_SLICE].copy()
    # Observation stores cube - grasp. A positive height offset therefore has
    # the target relative z = -offset, which keeps the fingers above the table
    # while approaching a cube resting on it.
    previous_error[:, 2] += config.grasp_height_offset_m
    current_error[:, 2] += config.grasp_height_offset_m
    previous_distance = np.linalg.norm(previous_error, axis=1)
    current_distance = np.linalg.norm(current_error, axis=1)
    reach_progress = previous_distance - current_distance
    # The home pose is roughly 0.4 m from the cube, so use a long-range
    # exponential rather than a narrow Gaussian whose gradient vanishes before
    # the policy discovers the workspace containing the object.
    reach_score = np.exp(-current_distance / config.reach_sigma_m)
    # Do not move the reach target as a function of gripper closure. That
    # coupling creates a local optimum where the policy hovers at pre-grasp and
    # keeps the fingers open to preserve its reach score. Instead, use an
    # independent closed-grasp target: arriving at pre-grasp already exposes a
    # strong closure gradient, while descending the final few millimetres makes
    # bilateral contact progressively easier.
    grasp_error = current[:, CUBE_TO_GRASP_SLICE].copy()
    grasp_error[:, 2] += config.closed_grasp_height_offset_m
    grasp_distance = np.linalg.norm(grasp_error, axis=1)
    grasp_gate = 1.0 / (
        1.0
        + np.exp(
            np.clip(
                (grasp_distance - config.grasp_distance_m)
                / config.grasp_gate_width_m,
                -60.0,
                60.0,
            )
        )
    )
    gripper_progress = gripper_close - previous_gripper_close
    previous_lift = np.clip(
        (previous[:, CUBE_Z_INDEX] - resting_z) / lift_delta_m, 0.0, 1.0
    )
    current_lift = np.clip(
        (current[:, CUBE_Z_INDEX] - resting_z) / lift_delta_m, 0.0, 1.0
    )
    lift_progress = current_lift - previous_lift

    reach_term = config.reach_reward_scale * reach_score * pre_latch
    reach_progress_term = config.reach_progress_scale * reach_progress * pre_latch
    grasp_region_term = config.grasp_region_reward_scale * grasp_gate * pre_latch
    gripper_term = (
        config.gripper_reward_scale * grasp_gate * gripper_close * pre_latch
    )
    gripper_progress_term = (
        config.gripper_progress_scale * grasp_gate * gripper_progress * pre_latch
    )
    premature_close_term = (
        -config.premature_close_penalty
        * (1.0 - grasp_gate)
        * gripper_close
        * pre_latch
    )
    # A lifted cube is the reliable grasp signal available in the current
    # observation. Gate the sustained lift reward by proximity and closure so
    # table jitter alone cannot be harvested as task progress.
    latch_term = config.latch_reward_scale * latched
    grasp_confidence = np.maximum(grasp_gate * gripper_close, latched)
    lift_term = config.lift_reward_scale * current_lift * grasp_confidence
    lift_progress_term = (
        config.lift_progress_scale * lift_progress * grasp_confidence
    )
    action_term = -config.action_penalty * np.mean(
        np.square(current[:, LAST_ACTION_SLICE]), axis=1
    )
    success_term = config.success_bonus * succeeded
    if safety_projection_alpha is None:
        projection_alpha = np.ones_like(raw)
    else:
        projection_alpha = np.asarray(safety_projection_alpha, dtype=np.float32).reshape(-1)
        if projection_alpha.shape != raw.shape:
            raise ValueError("safety_projection_alpha must match the reward batch.")
        projection_alpha = np.clip(projection_alpha, 0.0, 1.0)
    safety_term = -config.safety_projection_penalty * (1.0 - projection_alpha)
    shaped = (
        reach_term
        + reach_progress_term
        + grasp_region_term
        + gripper_term
        + gripper_progress_term
        + premature_close_term
        + latch_term
        + lift_term
        + lift_progress_term
        + action_term
        + success_term
        + safety_term
        if config.enabled
        else raw + safety_term
    )
    return shaped.astype(np.float32), {
        "distance_m": current_distance.astype(np.float32),
        "grasp_distance_m": grasp_distance.astype(np.float32),
        "horizontal_distance_m": np.linalg.norm(
            current_error[:, :2], axis=1
        ).astype(np.float32),
        "vertical_error_m": np.abs(current_error[:, 2]).astype(np.float32),
        "grasp_gate": grasp_gate.astype(np.float32),
        "gripper_close": gripper_close.astype(np.float32),
        "grasp_latched": latched.astype(np.float32),
        "lift_fraction": current_lift.astype(np.float32),
        "reach_term": reach_term.astype(np.float32),
        "reach_progress_term": reach_progress_term.astype(np.float32),
        "grasp_region_term": grasp_region_term.astype(np.float32),
        "gripper_term": gripper_term.astype(np.float32),
        "gripper_progress_term": gripper_progress_term.astype(np.float32),
        "premature_close_term": premature_close_term.astype(np.float32),
        "latch_term": latch_term.astype(np.float32),
        "lift_term": lift_term.astype(np.float32),
        "lift_progress_term": lift_progress_term.astype(np.float32),
        "action_term": action_term.astype(np.float32),
        "success_term": success_term.astype(np.float32),
        "safety_projection_term": safety_term.astype(np.float32),
    }


__all__ = ["shape_rewards"]
