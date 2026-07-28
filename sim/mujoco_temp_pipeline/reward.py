"""Temporary reward shaping from a simulator-only 31-D training state."""

from __future__ import annotations

import numpy as np

from sim.mujoco.tasks.cube_lift.reward_adapter import CubeLiftRewardState

from .config import RewardShapingConfig


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
    bilateral_contact: np.ndarray | None = None,
    previous_bilateral_contact: np.ndarray | None = None,
    side_contact_score: np.ndarray | None = None,
    top_bottom_contact_score: np.ndarray | None = None,
    inner_contact_score: np.ndarray | None = None,
    bilateral_inner_contact_score: np.ndarray | None = None,
    outer_contact_score: np.ndarray | None = None,
    previous_inner_contact_score: np.ndarray | None = None,
    previous_bilateral_inner_contact_score: np.ndarray | None = None,
    accepted_action: np.ndarray | None = None,
    approach_action: np.ndarray | None = None,
    time_limit: np.ndarray | None = None,
    failure_termination: np.ndarray | None = None,
    unsafe_termination: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute staged rewards without exposing this state to a normal policy."""

    previous = np.asarray(observations, dtype=np.float32)
    current = np.asarray(next_observations, dtype=np.float32)
    raw = np.asarray(raw_rewards, dtype=np.float32).reshape(-1)
    if previous.shape != current.shape or previous.ndim != 2 or previous.shape[1] != 31:
        raise ValueError("Reward shaping expects matching (N, 31) observations.")
    if raw.shape != (previous.shape[0],):
        raise ValueError("Reward shaping expects one raw reward per environment.")
    previous_state = CubeLiftRewardState.from_flat(previous)
    current_state = CubeLiftRewardState.from_flat(current)
    previous_state.validate_batch(raw.shape[0])
    current_state.validate_batch(raw.shape[0])
    if accepted_action is None:
        accepted = current_state.last_action
    else:
        accepted = np.asarray(accepted_action, dtype=np.float32)
        if accepted.shape != (raw.shape[0], 7):
            raise ValueError("accepted_action must have shape (N, 7).")
        if not np.all(np.isfinite(accepted)):
            raise ValueError("accepted_action must be finite.")
    if approach_action is None:
        scripted_approach = np.zeros((raw.shape[0], 7), dtype=np.float32)
    else:
        scripted_approach = np.asarray(approach_action, dtype=np.float32)
        if scripted_approach.shape != (raw.shape[0], 7):
            raise ValueError("approach_action must have shape (N, 7).")
        if not np.all(np.isfinite(scripted_approach)):
            raise ValueError("approach_action must be finite.")
        scripted_approach = np.clip(scripted_approach, -1.0, 1.0)

    if lift_delta_m <= 0.0:
        raise ValueError("lift_delta_m must be positive.")
    if resting_cube_z_m is None:
        resting_z = previous_state.cube_z
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
    if bilateral_contact is None:
        bilateral = np.zeros_like(raw, dtype=np.float32)
    else:
        bilateral = np.asarray(bilateral_contact, dtype=np.float32).reshape(-1)
        if bilateral.shape != raw.shape:
            raise ValueError("bilateral_contact must match the reward batch.")
        bilateral = np.clip(bilateral, 0.0, 1.0)
    if previous_bilateral_contact is None:
        previous_bilateral = np.zeros_like(raw, dtype=np.float32)
    else:
        previous_bilateral = np.asarray(
            previous_bilateral_contact, dtype=np.float32
        ).reshape(-1)
        if previous_bilateral.shape != raw.shape:
            raise ValueError(
                "previous_bilateral_contact must match the reward batch."
            )
        previous_bilateral = np.clip(previous_bilateral, 0.0, 1.0)
    if side_contact_score is None:
        side_contact = np.zeros_like(raw, dtype=np.float32)
    else:
        side_contact = np.asarray(side_contact_score, dtype=np.float32).reshape(-1)
        if side_contact.shape != raw.shape:
            raise ValueError("side_contact_score must match the reward batch.")
        if not np.all(np.isfinite(side_contact)):
            raise ValueError("side_contact_score must be finite.")
        side_contact = np.clip(side_contact, 0.0, 1.0)
    if top_bottom_contact_score is None:
        top_bottom_contact = np.zeros_like(raw, dtype=np.float32)
    else:
        top_bottom_contact = np.asarray(
            top_bottom_contact_score, dtype=np.float32
        ).reshape(-1)
        if top_bottom_contact.shape != raw.shape:
            raise ValueError(
                "top_bottom_contact_score must match the reward batch."
            )
        if not np.all(np.isfinite(top_bottom_contact)):
            raise ValueError("top_bottom_contact_score must be finite.")
        top_bottom_contact = np.clip(top_bottom_contact, 0.0, 1.0)
    if inner_contact_score is None:
        inner_contact = np.zeros_like(raw, dtype=np.float32)
    else:
        inner_contact = np.asarray(inner_contact_score, dtype=np.float32).reshape(-1)
        if inner_contact.shape != raw.shape:
            raise ValueError("inner_contact_score must match the reward batch.")
        if not np.all(np.isfinite(inner_contact)):
            raise ValueError("inner_contact_score must be finite.")
        inner_contact = np.clip(inner_contact, 0.0, 1.0)
    if bilateral_inner_contact_score is None:
        bilateral_inner_contact = np.zeros_like(raw, dtype=np.float32)
    else:
        bilateral_inner_contact = np.asarray(
            bilateral_inner_contact_score, dtype=np.float32
        ).reshape(-1)
        if bilateral_inner_contact.shape != raw.shape:
            raise ValueError(
                "bilateral_inner_contact_score must match the reward batch."
            )
        if not np.all(np.isfinite(bilateral_inner_contact)):
            raise ValueError("bilateral_inner_contact_score must be finite.")
        bilateral_inner_contact = np.clip(bilateral_inner_contact, 0.0, 1.0)
    if outer_contact_score is None:
        outer_contact = np.zeros_like(raw, dtype=np.float32)
    else:
        outer_contact = np.asarray(outer_contact_score, dtype=np.float32).reshape(-1)
        if outer_contact.shape != raw.shape:
            raise ValueError("outer_contact_score must match the reward batch.")
        if not np.all(np.isfinite(outer_contact)):
            raise ValueError("outer_contact_score must be finite.")
        outer_contact = np.clip(outer_contact, 0.0, 1.0)
    generic_contact_enabled = any(
        value is not None
        for value in (
            inner_contact_score,
            bilateral_inner_contact_score,
            outer_contact_score,
        )
    )
    if previous_inner_contact_score is None:
        previous_inner_contact = np.zeros_like(raw, dtype=np.float32)
    else:
        previous_inner_contact = np.asarray(
            previous_inner_contact_score, dtype=np.float32
        ).reshape(-1)
        if previous_inner_contact.shape != raw.shape:
            raise ValueError(
                "previous_inner_contact_score must match the reward batch."
            )
        if not np.all(np.isfinite(previous_inner_contact)):
            raise ValueError("previous_inner_contact_score must be finite.")
        previous_inner_contact = np.clip(previous_inner_contact, 0.0, 1.0)
    if previous_bilateral_inner_contact_score is None:
        previous_bilateral_inner_contact = np.zeros_like(raw, dtype=np.float32)
    else:
        previous_bilateral_inner_contact = np.asarray(
            previous_bilateral_inner_contact_score, dtype=np.float32
        ).reshape(-1)
        if previous_bilateral_inner_contact.shape != raw.shape:
            raise ValueError(
                "previous_bilateral_inner_contact_score must match the reward batch."
            )
        if not np.all(np.isfinite(previous_bilateral_inner_contact)):
            raise ValueError(
                "previous_bilateral_inner_contact_score must be finite."
            )
        previous_bilateral_inner_contact = np.clip(
            previous_bilateral_inner_contact, 0.0, 1.0
        )
    if time_limit is None:
        timed_out = np.zeros_like(raw, dtype=np.float32)
    else:
        timed_out = np.asarray(time_limit, dtype=np.float32).reshape(-1)
        if timed_out.shape != raw.shape:
            raise ValueError("time_limit must match the reward batch.")
        timed_out = np.clip(timed_out, 0.0, 1.0)
    if failure_termination is None:
        failed = np.zeros_like(raw, dtype=np.float32)
    else:
        failed = np.asarray(failure_termination, dtype=np.float32).reshape(-1)
        if failed.shape != raw.shape:
            raise ValueError("failure_termination must match the reward batch.")
        failed = np.clip(failed, 0.0, 1.0)
    if unsafe_termination is None:
        unsafe = np.zeros_like(raw, dtype=np.float32)
    else:
        unsafe = np.asarray(unsafe_termination, dtype=np.float32).reshape(-1)
        if unsafe.shape != raw.shape:
            raise ValueError("unsafe_termination must match the reward batch.")
        unsafe = np.clip(unsafe, 0.0, 1.0)

    previous_gripper_close = np.clip(
        previous_state.right_gripper_close, 0.0, 1.0
    )
    gripper_close = np.clip(current_state.right_gripper_close, 0.0, 1.0)
    previous_error = previous_state.cube_to_grasp.copy()
    current_error = current_state.cube_to_grasp.copy()
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
    grasp_error = current_state.cube_to_grasp.copy()
    grasp_error[:, 2] += config.closed_grasp_height_offset_m
    grasp_distance = np.linalg.norm(grasp_error, axis=1)
    previous_grasp_error = previous_state.cube_to_grasp.copy()
    previous_grasp_error[:, 2] += config.closed_grasp_height_offset_m
    previous_grasp_distance = np.linalg.norm(previous_grasp_error, axis=1)
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
    previous_grasp_gate = 1.0 / (
        1.0
        + np.exp(
            np.clip(
                (previous_grasp_distance - config.grasp_distance_m)
                / config.grasp_gate_width_m,
                -60.0,
                60.0,
            )
        )
    )
    if config.staged_grasp:
        close_gate = 1.0 / (
            1.0
            + np.exp(
                np.clip(
                    (grasp_distance - config.close_gate_distance_m)
                    / config.close_gate_width_m,
                    -60.0,
                    60.0,
                )
            )
        )
        reach_phase_weight = 1.0 - config.near_reach_decay * close_gate
    else:
        close_gate = grasp_gate
        reach_phase_weight = np.ones_like(grasp_gate)
    gripper_progress = gripper_close - previous_gripper_close
    previous_lift = np.clip(
        (previous_state.cube_z - resting_z) / lift_delta_m, 0.0, 1.0
    )
    current_lift = np.clip(
        (current_state.cube_z - resting_z) / lift_delta_m, 0.0, 1.0
    )
    lift_progress = current_lift - previous_lift

    # The generic contact score is a geometric signal.  For the contact-aware
    # profile, only its positive onset is rewarded; otherwise a stationary
    # finger/object contact would pay on every policy tick.
    inner_contact_gain = np.maximum(inner_contact - previous_inner_contact, 0.0)
    bilateral_inner_contact_gain = np.maximum(
        bilateral_inner_contact - previous_bilateral_inner_contact,
        0.0,
    )
    if generic_contact_enabled:
        contact_gate = bilateral_inner_contact
    else:
        contact_gate = bilateral
    contact_gate = np.clip(contact_gate, 0.0, 1.0)
    contact_ready = (
        contact_gate >= float(config.contact_gate_threshold)
    ).astype(np.float32)
    closure_gate = (
        contact_gate if config.gripper_reward_requires_contact else close_gate
    )

    # The grasp-center distance is useful for long-range approach shaping, but
    # it must not become a target for driving a closed gripper through the
    # object.  Once the center frame enters the configured exclusion radius,
    # charge a strong penalty only while the gripper is meaningfully closed and
    # a bilateral inner-surface grasp has not yet been established.  Valid
    # bilateral contact therefore remains reachable; unilateral/deep impacts
    # do not receive a free pass from the proximity term.
    center_distance = np.linalg.norm(current_state.cube_to_grasp, axis=1)
    if (
        config.center_proximity_limit_m > 0.0
        and config.center_proximity_penalty > 0.0
    ):
        center_proximity_violation = np.clip(
            (
                config.center_proximity_limit_m - center_distance
            )
            / config.center_proximity_limit_m,
            0.0,
            1.0,
        )
        close_threshold = float(config.center_proximity_close_threshold)
        if close_threshold >= 1.0:
            close_fraction = np.zeros_like(gripper_close)
        else:
            close_fraction = np.clip(
                (gripper_close - close_threshold) / max(1.0 - close_threshold, 1e-6),
                0.0,
                1.0,
            )
        center_proximity_violation *= close_fraction * (1.0 - contact_ready)
    else:
        center_proximity_violation = np.zeros_like(raw)
    center_proximity_term = (
        -config.center_proximity_penalty
        * center_proximity_violation
        * pre_latch
    )

    reach_term = (
        config.reach_reward_scale
        * reach_score
        * reach_phase_weight
        * pre_latch
    )
    reach_progress_term = (
        config.reach_progress_scale
        * reach_progress
        * reach_phase_weight
        * pre_latch
    )
    grasp_region_term = config.grasp_region_reward_scale * grasp_gate * pre_latch
    grasp_region_progress_term = (
        config.grasp_region_progress_scale
        * (grasp_gate - previous_grasp_gate)
        * pre_latch
    )
    approach_alignment = np.mean(
        accepted[:, :3] * scripted_approach[:, :3], axis=1
    )
    approach_alignment = np.clip(approach_alignment, 0.0, 1.0)
    approach_alignment_term = (
        config.approach_alignment_reward_scale
        * approach_alignment
        * (1.0 - contact_ready)
        * pre_latch
    )
    gripper_term = (
        config.gripper_reward_scale * closure_gate * gripper_close * pre_latch
    )
    gripper_progress_for_reward = (
        np.maximum(gripper_progress, 0.0)
        if config.gripper_reward_requires_contact
        else gripper_progress
    )
    gripper_progress_term = (
        config.gripper_progress_scale
        * closure_gate
        * gripper_progress_for_reward
        * pre_latch
    )
    gripper_command_term = (
        config.gripper_command_reward_scale
        * closure_gate
        * accepted[:, 6]
        * pre_latch
    )
    if config.gripper_reward_requires_contact:
        # Penalize the positive closing movement itself before inner contact;
        # penalizing the absolute open/closed state would make exploration
        # unable to discover the contact event.
        premature_close_term = (
            -config.premature_close_penalty
            * np.maximum(gripper_progress, 0.0)
            * (1.0 - contact_gate)
            * pre_latch
        )
    else:
        premature_close_term = (
            -config.premature_close_penalty
            * (1.0 - close_gate)
            * gripper_close
            * pre_latch
        )
    premature_close_state_term = (
        -config.premature_close_state_penalty
        * gripper_close
        * (1.0 - contact_gate)
        * pre_latch
    )
    close_confidence = (
        1.0
        / (
            1.0
            + np.exp(
                np.clip(
                    (config.grasp_acquire_close - gripper_close)
                    / config.grasp_close_confidence_width,
                    -60.0,
                    60.0,
                )
            )
        )
        if config.staged_grasp
        else gripper_close
    )
    grasp_acquire = (
        (
            (previous_gripper_close < config.grasp_acquire_close)
            & (gripper_close >= config.grasp_acquire_close)
            & (grasp_gate >= config.grasp_acquire_gate)
            & (
                (not config.grasp_acquire_requires_contact)
                | (contact_ready >= 0.5)
            )
        ).astype(np.float32)
        if config.staged_grasp
        else np.zeros_like(raw)
    )
    grasp_acquire_term = config.grasp_acquire_bonus * grasp_acquire * pre_latch
    bilateral_contact_term = config.bilateral_contact_reward_scale * bilateral
    bilateral_contact_acquire = (
        bilateral
        * (previous_bilateral < 0.5)
        * (
            (not config.grasp_acquire_requires_contact)
            | (contact_ready >= 0.5)
        )
    )
    bilateral_contact_acquire_term = (
        config.bilateral_contact_acquire_bonus * bilateral_contact_acquire
    )
    missing_reference = (
        previous_bilateral
        if config.contact_missing_requires_previous
        else np.ones_like(previous_bilateral)
    )
    bilateral_contact_missing_term = (
        -config.bilateral_contact_missing_penalty
        * (1.0 - bilateral)
        * missing_reference
    )
    side_contact_term = (
        config.side_contact_reward_scale * side_contact * pre_latch
    )
    top_bottom_contact_penalty_term = (
        -config.top_bottom_contact_penalty * top_bottom_contact * pre_latch
    )
    inner_contact_signal = (
        inner_contact_gain
        if config.contact_reward_onset_only
        else inner_contact
    )
    bilateral_inner_contact_signal = (
        bilateral_inner_contact_gain
        if config.contact_reward_onset_only
        else bilateral_inner_contact
    )
    inner_contact_term = (
        config.inner_contact_reward_scale * inner_contact_signal * pre_latch
    )
    bilateral_inner_contact_term = (
        config.bilateral_inner_contact_reward_scale
        * bilateral_inner_contact_signal
        * pre_latch
    )
    outer_contact_penalty_term = (
        -config.outer_contact_penalty * outer_contact * pre_latch
    )
    # A lifted cube is the reliable grasp signal available in the current
    # observation. Gate the sustained lift reward by proximity and closure so
    # table jitter alone cannot be harvested as task progress.
    latch_term = config.latch_reward_scale * latched
    grasp_confidence = np.maximum(grasp_gate * close_confidence, latched)
    if config.require_generic_contact_for_lift and generic_contact_enabled:
        grasp_confidence = np.maximum(
            latched,
            grasp_confidence * bilateral_inner_contact,
        )
    if config.require_bilateral_contact_for_lift_reward:
        grasp_confidence = np.maximum(latched, grasp_confidence * bilateral)
    lift_term = config.lift_reward_scale * current_lift * grasp_confidence
    lift_progress_confidence = grasp_confidence
    if config.ungated_negative_lift_progress:
        # Positive lift remains contact-gated so free-object jitter cannot be
        # harvested. A subsequent drop must still repay that progress even
        # after contact has already been lost; otherwise repeated tiny lifts
        # followed by ungated falls create a one-way reward ratchet.
        lift_progress_confidence = np.where(
            lift_progress < 0.0,
            1.0,
            grasp_confidence,
        )
    lift_progress_term = (
        config.lift_progress_scale
        * lift_progress
        * lift_progress_confidence
    )
    action_term = -config.action_penalty * np.mean(
        np.square(current_state.last_action), axis=1
    )
    success_term = config.success_bonus * succeeded
    time_limit_term = -config.time_limit_penalty * timed_out
    failure_termination_term = -config.failure_termination_penalty * failed
    unsafe_termination_term = -config.unsafe_termination_penalty * unsafe
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
        + grasp_region_progress_term
        + approach_alignment_term
        + gripper_term
        + gripper_progress_term
        + gripper_command_term
        + premature_close_term
        + premature_close_state_term
        + grasp_acquire_term
        + bilateral_contact_term
        + bilateral_contact_acquire_term
        + bilateral_contact_missing_term
        + side_contact_term
        + top_bottom_contact_penalty_term
        + inner_contact_term
        + bilateral_inner_contact_term
        + outer_contact_penalty_term
        + center_proximity_term
        + latch_term
        + lift_term
        + lift_progress_term
        + action_term
        + success_term
        + time_limit_term
        + failure_termination_term
        + unsafe_termination_term
        + safety_term
        if config.enabled
        else raw + safety_term
    )
    return shaped.astype(np.float32), {
        "distance_m": current_distance.astype(np.float32),
        "center_distance_m": center_distance.astype(np.float32),
        "center_proximity_violation": center_proximity_violation.astype(np.float32),
        "center_proximity_term": center_proximity_term.astype(np.float32),
        "grasp_distance_m": grasp_distance.astype(np.float32),
        "horizontal_distance_m": np.linalg.norm(
            current_error[:, :2], axis=1
        ).astype(np.float32),
        "vertical_error_m": np.abs(current_error[:, 2]).astype(np.float32),
        "grasp_gate": grasp_gate.astype(np.float32),
        "close_gate": close_gate.astype(np.float32),
        "reach_phase_weight": reach_phase_weight.astype(np.float32),
        "gripper_close": gripper_close.astype(np.float32),
        "grasp_latched": latched.astype(np.float32),
        "bilateral_contact": bilateral.astype(np.float32),
        "bilateral_contact_acquire": bilateral_contact_acquire.astype(np.float32),
        "lift_fraction": current_lift.astype(np.float32),
        "reach_term": reach_term.astype(np.float32),
        "reach_progress_term": reach_progress_term.astype(np.float32),
        "grasp_region_term": grasp_region_term.astype(np.float32),
        "grasp_region_progress_term": grasp_region_progress_term.astype(
            np.float32
        ),
        "approach_alignment_term": approach_alignment_term.astype(np.float32),
        "gripper_term": gripper_term.astype(np.float32),
        "gripper_progress_term": gripper_progress_term.astype(np.float32),
        "gripper_command_term": gripper_command_term.astype(np.float32),
        "premature_close_term": premature_close_term.astype(np.float32),
        "premature_close_state_term": premature_close_state_term.astype(
            np.float32
        ),
        "grasp_close_confidence": close_confidence.astype(np.float32),
        "grasp_acquire_term": grasp_acquire_term.astype(np.float32),
        "bilateral_contact_term": bilateral_contact_term.astype(np.float32),
        "bilateral_contact_acquire_term": bilateral_contact_acquire_term.astype(
            np.float32
        ),
        "bilateral_contact_missing_term": (
            bilateral_contact_missing_term.astype(np.float32)
        ),
        "side_contact_score": side_contact.astype(np.float32),
        "top_bottom_contact_score": top_bottom_contact.astype(np.float32),
        "side_contact_term": side_contact_term.astype(np.float32),
        "top_bottom_contact_penalty_term": (
            top_bottom_contact_penalty_term.astype(np.float32)
        ),
        "inner_contact_score": inner_contact.astype(np.float32),
        "bilateral_inner_contact_score": bilateral_inner_contact.astype(np.float32),
        "outer_contact_score": outer_contact.astype(np.float32),
        "inner_contact_gain": inner_contact_gain.astype(np.float32),
        "bilateral_inner_contact_gain": bilateral_inner_contact_gain.astype(
            np.float32
        ),
        "inner_contact_term": inner_contact_term.astype(np.float32),
        "bilateral_inner_contact_term": bilateral_inner_contact_term.astype(
            np.float32
        ),
        "outer_contact_penalty_term": outer_contact_penalty_term.astype(
            np.float32
        ),
        "latch_term": latch_term.astype(np.float32),
        "lift_term": lift_term.astype(np.float32),
        "lift_progress_term": lift_progress_term.astype(np.float32),
        "action_term": action_term.astype(np.float32),
        "success_term": success_term.astype(np.float32),
        "time_limit_term": time_limit_term.astype(np.float32),
        "failure_termination_term": failure_termination_term.astype(np.float32),
        "unsafe_termination_term": unsafe_termination_term.astype(np.float32),
        "safety_projection_term": safety_term.astype(np.float32),
    }


__all__ = ["shape_rewards"]
