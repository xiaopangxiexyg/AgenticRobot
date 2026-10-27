"""Closed state of the original coupled retargeter, with native solver history.

Only caller-fenced, complete solves are eligible. The private constructor seam
restores an unpublished independent owner; this module never acquires authority,
resets tracking, issues commands, or changes the ordinary solver policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dataset_contracts.canonical_encoding import canonical_bytes, canonical_loads

from .ik_checkpoint import (
    IKSolverCheckpoint,
    capture_ik_solver,
    check_data_aliases,
    copy_data,
    pack_data,
    restore_ik_solver,
    state_hash,
    unpack_data,
)

_STATE = {
    "_last_admitted_velocity_full",
    "_last_target_poses",
    "_jump_hold_active",
    "_jump_hold_full_q",
    "_jump_hold_anchor_targets",
    "_jump_hold_frames",
    "_profile_frame_index",
    "_time_budget_warning_frames",
    "_last_solve_iteration_budget_exhausted",
    "_last_solve_iteration_stop_reason",
    "_last_raw_solver_q",
    "_last_branch_continuity_anchor_metadata",
    "_last_wrist_coaxial_metadata",
    "_last_directional_singularity_diagnostics",
    "_last_output_qpos_smoothing_full_q",
    "_output_qpos_smoothing_effective_alpha",
    "_output_qpos_smoothing_update_time",
    "_output_qpos_normal_fast_path_last_quality_valid_time",
    "_output_qpos_admittance_pending_reversal_direction",
    "_position_singularity_metadata_cache",
    "_singularity_escape_direction_latch",
    "_singularity_escape_direction_failures",
    "_singularity_escape_axis_latch",
    "_joint_limit_escape_direction_latch",
    "_joint_limit_escape_release_guard",
    "_wrist_q5_hysteresis_escape_latch",
    "_candidate_certificate_sequence",
    "_last_safe_full_q",
}
_STATIC = {
    "ik_tuning",
    "ik_safety_profile",
    "posture_preference",
    "joint_velocity_profile",
    "profile",
    "_arm_joint_weights",
    "_full_joint_weights",
    "_execution_dt_sec",
    "_full_execution_joint_step",
    "_branch_continuity_anchor_enabled",
    "_acceleration_limit_enabled",
    "_admitted_reference_motion_limit_enabled",
    "_left_arm_limits",
    "_right_arm_limits",
    "_tool_tcp_from_solver",
    "_solver_from_tool_tcp",
}
_NONE = {
    "left_pose_retargeter",
    "right_pose_retargeter",
    "target_override_filter",
    "pre_ik_target_safety_filter",
    "_target_pose_smoother",
    "_ik_shadow_trace_sink",
}
_OWNERS = {"config", "ik_solver", "_position_singularity_analyzers"}


@dataclass(frozen=True)
class RetargeterCheckpoint:
    configuration_hash: str
    binding_hash: str
    state: dict
    solver: IKSolverCheckpoint
    content_hash: str


def _qualify(owner, *, constructing=False):
    from .dual_arm_target_solver import DualArmCollisionAwareRetargeter

    layout = _STATE | _STATIC | _NONE | _OWNERS
    if constructing:
        layout -= {"_last_safe_full_q"}
    if type(owner) is not DualArmCollisionAwareRetargeter or set(vars(owner)) != layout:
        raise ValueError("Unknown retargeter owner or field layout.")
    if (
        any(getattr(owner, n) is not None for n in _NONE)
        or owner._position_singularity_analyzers
    ):
        raise ValueError(
            "Retargeter checkpoint does not support custom filters/smoothers/observers."
        )
    if any(
        getattr(owner, n)
        for n in (
            "_branch_continuity_anchor_enabled",
            "_acceleration_limit_enabled",
            "_admitted_reference_motion_limit_enabled",
        )
    ):
        raise ValueError(
            "Retargeter checkpoint does not support additional solver constraints."
        )


def _binding(owner):
    return state_hash({n: getattr(owner, n) for n in sorted(_STATIC)})


def _contents(configuration, binding, state, solver):
    return state_hash((configuration, binding, state, solver.content_hash))


def _validate_state(state, size):
    if type(state) is not dict or set(state) != _STATE:
        raise ValueError("Unknown retargeter checkpoint state.")
    for name in (
        "_jump_hold_frames",
        "_profile_frame_index",
        "_time_budget_warning_frames",
        "_candidate_certificate_sequence",
    ):
        if type(state[name]) is not int or state[name] < 0:
            raise ValueError("Invalid retargeter checkpoint counter.")
    for name in ("_jump_hold_active", "_last_solve_iteration_budget_exhausted"):
        if type(state[name]) is not bool:
            raise ValueError("Invalid retargeter checkpoint boolean.")
    for name in (
        "_last_safe_full_q",
        "_last_admitted_velocity_full",
        "_last_raw_solver_q",
        "_jump_hold_full_q",
        "_last_output_qpos_smoothing_full_q",
    ):
        value = state[name]
        if value is None and name not in (
            "_last_safe_full_q",
            "_last_admitted_velocity_full",
        ):
            continue
        if (
            type(value) is not np.ndarray
            or value.shape != (size,)
            or not np.all(np.isfinite(value))
        ):
            raise ValueError("Invalid retargeter joint history.")
    if state["_jump_hold_active"] and (
        state["_jump_hold_full_q"] is None or state["_jump_hold_anchor_targets"] is None
    ):
        raise ValueError("Incomplete retargeter HOLD state.")


def capture_retargeter(owner, *, max_native_bytes):
    _qualify(owner)
    raw_state = {n: getattr(owner, n) for n in sorted(_STATE)}
    check_data_aliases(raw_state)
    state = copy_data(raw_state)
    _validate_state(state, owner.ik_solver.robot.model.nq)
    configuration, binding = state_hash(owner.config), _binding(owner)
    solver = capture_ik_solver(owner.ik_solver, max_native_bytes=max_native_bytes)
    return RetargeterCheckpoint(
        configuration,
        binding,
        state,
        solver,
        _contents(configuration, binding, state, solver),
    )


def restore_constructor_state(owner, checkpoint, *, max_native_bytes):
    """Called only by the original constructor before its initial tracking reset."""
    _qualify(owner, constructing=True)
    if (
        type(checkpoint) is not RetargeterCheckpoint
        or type(checkpoint.solver) is not IKSolverCheckpoint
    ):
        raise ValueError("Expected a typed retargeter checkpoint.")
    if type(checkpoint.state) is not dict or type(checkpoint.solver.state) is not dict:
        raise ValueError("Invalid retargeter checkpoint state fields.")
    if checkpoint.content_hash != _contents(
        checkpoint.configuration_hash,
        checkpoint.binding_hash,
        checkpoint.state,
        checkpoint.solver,
    ):
        raise ValueError("Retargeter checkpoint content changed.")
    if checkpoint.configuration_hash != state_hash(
        owner.config
    ) or checkpoint.binding_hash != _binding(owner):
        raise ValueError("Retargeter configuration or derived binding differs.")
    check_data_aliases(checkpoint.state)
    state = copy_data(checkpoint.state)
    _validate_state(state, owner.ik_solver.robot.model.nq)
    restore_ik_solver(
        owner.ik_solver, checkpoint.solver, max_native_bytes=max_native_bytes
    )
    for name, value in state.items():
        setattr(owner, name, value)
    _qualify(owner)


def restore_retargeter(checkpoint, *, config, max_native_bytes):
    """Create a fresh owner with the source's original, caller-trusted config."""
    from .dual_arm_target_solver import DualArmCollisionAwareRetargeter

    return DualArmCollisionAwareRetargeter(
        config, _checkpoint=checkpoint, _checkpoint_max_native_bytes=max_native_bytes
    )


_DOMAIN = "rm65.retargeter-checkpoint.v1"


def encode_retargeter_checkpoint(checkpoint):
    """Encode fixed typed state; parent owns provenance, fencing and publication."""
    if (
        type(checkpoint) is not RetargeterCheckpoint
        or type(checkpoint.solver) is not IKSolverCheckpoint
    ):
        raise ValueError("Expected a typed retargeter checkpoint.")
    if type(checkpoint.state) is not dict or type(checkpoint.solver.state) is not dict:
        raise ValueError("Invalid retargeter checkpoint state fields.")
    check_data_aliases(checkpoint.state)
    check_data_aliases(
        checkpoint.solver.state,
        barrier_q_alias=checkpoint.solver.state.get("barrier_q_is_current") is True,
    )
    return canonical_bytes(
        {
            "configuration": checkpoint.configuration_hash,
            "binding": checkpoint.binding_hash,
            "state": pack_data(checkpoint.state),
            "solver": {
                "binding": checkpoint.solver.binding_hash,
                "state": pack_data(checkpoint.solver.state),
                "native": checkpoint.solver.native,
                "hash": checkpoint.solver.content_hash,
            },
            "hash": checkpoint.content_hash,
        },
        domain=_DOMAIN,
    )


def decode_retargeter_checkpoint(encoded, *, max_bytes):
    """Decode data only, without constructing native objects or importing types."""
    if (
        type(max_bytes) is not int
        or max_bytes <= 0
        or type(encoded) is not bytes
        or len(encoded) > max_bytes
    ):
        raise ValueError("Retargeter encoding exceeds its byte budget.")
    value = canonical_loads(encoded, expected_domain=_DOMAIN)
    if type(value) is not dict or set(value) != {
        "configuration",
        "binding",
        "state",
        "solver",
        "hash",
    }:
        raise ValueError("Invalid retargeter checkpoint fields.")
    solver = value["solver"]
    if type(solver) is not dict or set(solver) != {
        "binding",
        "state",
        "native",
        "hash",
    }:
        raise ValueError("Invalid IK solver checkpoint fields.")
    return RetargeterCheckpoint(
        value["configuration"],
        value["binding"],
        unpack_data(value["state"]),
        IKSolverCheckpoint(
            solver["binding"],
            unpack_data(solver["state"]),
            solver["native"],
            solver["hash"],
        ),
        value["hash"],
    )
