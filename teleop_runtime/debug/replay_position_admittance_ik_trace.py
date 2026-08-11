"""Replay sampled force evidence through C0 and the shared RM65 IK offline.

The action-debug stream records admitted dual-arm qref at 15 Hz and the
post-frame-transform filtered force.  This diagnostic replays that force
through the current bounded 60 Hz profile-compliance core, reconstructs the
admitted qref FK with the shared RM65 model, applies the current C0 translation
gate, and compares baseline versus force-perturbed one-step IK from the same
recorded qref.

The continuous 60 Hz rollout advances only the modified branch.  On every
tick it forks a nominal preview from that branch's exact admitted IK state;
the nominal preview never owns an independent history.  This is required for
an incremental modifier measurement: subtracting two independently stateful
IK trajectories would include ordinary branch/hold/release path dependence.

Old traces do not contain the exact post-smoother target rotation.  For those
traces the replay deliberately uses the admitted-qref FK rotation for both the
baseline and candidate, isolating the incremental translation effect.  New
traces can use ``raw_target_pose_solver_world`` directly.  No device, backend,
dispatch, or hardware process is constructed by this module.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.position_control.cartesian_admittance import (
    BoundedCartesianAdmittance,
    CartesianAdmittanceWrenchSample,
)
from robot_motion.rm65.ik_tuning import IKObjectiveWeights, load_ik_tuning_config
from robot_motion.rm65.position_profile import (
    RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
    resolve_rm65_bimanual_position_profile,
)
from robot_motion.rm65.tool_profiles import (
    load_rm65_tool_profile,
    tool_tcp_pose_flange_m_rad,
)
from robot_motion.runtime import EffectorCommand
from robot_motion.transforms import interpolate_pose
from teleop_runtime.real_teleop.position_admittance_shadow import (
    POSITION_ADMITTANCE_ACTIVE_QPOS_FAST_PATH_ALPHA,
    POSITION_ADMITTANCE_ACTIVE_QPOS_MICRO_REVERSAL_HOLD_DEG,
)
from teleop_runtime.retargeters.dual_arm_collision_aware import (
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
    DualArmTargetContext,
)

from .replay_position_admittance_force_trace import (
    DEFAULT_C0_MAX_OFFSET_M,
    DEFAULT_C0_MAX_SPEED_M_S,
    DEFAULT_REPLAY_HZ,
    _append_release_tail,
    _candidate_config,
    _force_events,
    _position_admittance_receipt,
    _read_rows,
    _resolve_log,
)


SIDES = ("left", "right")
DEFAULT_LEFT_TOOL_PROFILE = (
    "hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json"
)
DEFAULT_RIGHT_TOOL_PROFILE = (
    "hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json"
)
ROTATION_GATE_DEG = 3.0
INCREMENTAL_FK_ROTATION_GATE_DEG = 0.5
RELEASE_TARGET_TRANSLATION_GATE_M = 1e-3
RELEASE_TARGET_ROTATION_GATE_DEG = 0.5


@dataclass(frozen=True)
class _ForceOutput:
    observed_at_sec: float
    output_offset_m: tuple[float, float, float]


@dataclass(frozen=True)
class _ForceTick:
    now_sec: float
    output_offset_m: tuple[float, float, float]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _vector3(value: Any, *, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float).reshape(3)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite values") from exc
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    return vector


def _matrix(value: Any) -> np.ndarray | None:
    if isinstance(value, Mapping) and "xyz_m" in value and "rpy_deg" in value:
        try:
            pose = np.eye(4, dtype=float)
            pose[:3, 3] = np.asarray(value["xyz_m"], dtype=float).reshape(3)
            pose[:3, :3] = Rotation.from_euler(
                "xyz",
                np.asarray(value["rpy_deg"], dtype=float).reshape(3),
                degrees=True,
            ).as_matrix()
            return pose
        except (TypeError, ValueError):
            return None
    try:
        pose = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return None
    return pose.copy()


def _qpos(command: Mapping[str, Any], *, side: str) -> np.ndarray:
    payload = _mapping(_mapping(command.get(side)).get("value_qpos"))
    try:
        qpos = np.deg2rad(np.asarray(payload.get("deg"), dtype=float).reshape(6))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{side} frame is missing six finite qref degrees") from exc
    if not np.all(np.isfinite(qpos)):
        raise ValueError(f"{side} frame is missing six finite qref degrees")
    return qpos


def _config_payload(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for row in rows:
        if row.get("event") == "config":
            return _mapping(row.get("payload"))
    return {}


def _frame_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("event") == "frame"]


def _raw_target_translation(frame: Mapping[str, Any], *, side: str) -> np.ndarray:
    payload = _mapping(
        _mapping(_mapping(frame.get("position_admittance")).get("sides")).get(side)
    )
    raw_pose = _matrix(payload.get("raw_target_pose_solver_world"))
    if raw_pose is not None:
        return raw_pose[:3, 3].copy()
    applied = _vector3(
        payload.get("applied_target_translation"),
        name=f"{side}.applied_target_translation",
    )
    total = _vector3(
        payload.get("applied_total_offset_m", (0.0, 0.0, 0.0)),
        name=f"{side}.applied_total_offset_m",
    )
    return applied - total


def _raw_target_rotation(
    frame: Mapping[str, Any],
    *,
    side: str,
) -> tuple[np.ndarray | None, str]:
    payload = _mapping(
        _mapping(_mapping(frame.get("position_admittance")).get("sides")).get(side)
    )
    raw_pose = _matrix(payload.get("raw_target_pose_solver_world"))
    if raw_pose is not None:
        return raw_pose[:3, :3].copy(), "recorded_post_smoother_solver_world"
    metadata = _mapping(_mapping(_mapping(frame.get("command_by_side")).get(side)).get("metadata"))
    source = _matrix(metadata.get("source_target_pose_desk"))
    if source is not None:
        return source[:3, :3].copy(), "source_pose_proxy_before_target_smoother"
    return None, "unavailable"


def _rotation_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = np.asarray(left, dtype=float).T @ np.asarray(right, dtype=float)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=float,
    )
    if finite.size == 0:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    return {
        "n": int(finite.size),
        "p50": float(np.percentile(finite, 50.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "max": float(np.max(finite)),
    }


def _sample_force_outputs(
    rows: Sequence[Mapping[str, Any]],
    *,
    side: str,
    max_offset_m: float,
    max_speed_m_s: float,
    max_return_speed_m_s: float | None,
    max_output_slew_m_s: float | None,
    replay_hz: float,
    release_tail_sec: float = 6.0,
) -> tuple[
    list[_ForceOutput],
    list[_ForceTick],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    receipt = _position_admittance_receipt(rows)
    recorded_events = _force_events(rows, side=side)
    frames = _frame_rows(rows)
    if len(recorded_events) != len(frames):
        raise ValueError(
            "force/IK replay requires one sampled force event per action frame; "
            f"events={len(recorded_events)} frames={len(frames)}"
        )
    events = _append_release_tail(
        recorded_events,
        duration_sec=float(release_tail_sec),
    )
    config = _candidate_config(
        receipt,
        max_offset_m=max_offset_m,
        max_speed_m_s=max_speed_m_s,
        max_return_speed_m_s=max_return_speed_m_s,
        max_output_slew_m_s=max_output_slew_m_s,
    )
    controller = BoundedCartesianAdmittance(config)
    tick_period = 1.0 / float(replay_hz)
    first = events[0]
    initial_source = first.source_time_sec - min(
        1.0 / 15.0,
        config.max_force_dt_sec * 0.5,
    )
    controller.reset(
        tare_sample=CartesianAdmittanceWrenchSample(
            values_n_nm=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            frame=config.control_frame,
            source_time_sec=initial_source,
            receive_time_sec=first.observed_at_sec,
            tare_generation=1,
        ),
        output_time_sec=first.observed_at_sec - tick_period,
    )
    outputs: list[_ForceOutput | None] = [None] * len(recorded_events)
    tick_outputs: list[_ForceTick] = []
    event_index = 0
    tolerance = 1e-9
    end_tick = events[-1].observed_at_sec + tick_period
    tick_count = int(
        math.ceil((end_tick - first.observed_at_sec) * float(replay_hz))
    ) + 1
    for tick_index in range(tick_count):
        now = first.observed_at_sec + tick_index * tick_period
        applied: list[int] = []
        while (
            event_index < len(events)
            and events[event_index].observed_at_sec <= now + tolerance
        ):
            event = events[event_index]
            controller.step_new_sample(
                CartesianAdmittanceWrenchSample(
                    values_n_nm=(
                        *event.filtered_force_control_n,
                        0.0,
                        0.0,
                        0.0,
                    ),
                    frame=config.control_frame,
                    source_time_sec=event.source_time_sec,
                    receive_time_sec=now,
                    tare_generation=1,
                ),
                now_sec=now,
                active=True,
            )
            applied.append(event_index)
            event_index += 1
        state = controller.sample_output(now_sec=now, active=True)
        offset = tuple(float(value) for value in state.output_offset_m)
        tick_outputs.append(_ForceTick(now_sec=now, output_offset_m=offset))
        for index in applied:
            if index < len(recorded_events):
                outputs[index] = _ForceOutput(
                    observed_at_sec=events[index].observed_at_sec,
                    output_offset_m=offset,
                )
    if event_index != len(events) or any(output is None for output in outputs):
        raise RuntimeError("force replay did not sample every recorded action frame")
    return (
        [output for output in outputs if output is not None],
        tick_outputs,
        receipt,
        asdict(config),
    )


def _profile_name(config: Mapping[str, Any]) -> str:
    profile = _mapping(config.get("position_control_profile"))
    return str(profile.get("name") or "")


def _tool_pose(path: Path, *, side: str) -> tuple[float, ...]:
    profile = load_rm65_tool_profile(path, side=side)
    return tuple(float(value) for value in tool_tcp_pose_flange_m_rad(profile))


def _build_retargeter(
    *,
    initial_qpos: Mapping[str, np.ndarray],
    config_payload: Mapping[str, Any],
    left_tool_pose: tuple[float, ...],
    right_tool_pose: tuple[float, ...],
) -> tuple[DualArmCollisionAwareRetargeter, np.ndarray]:
    profile = resolve_rm65_bimanual_position_profile(
        RM65_BIMANUAL_POSITION_V2_BALANCED_NAME
    )
    ik = profile.ik_execution
    command = profile.command_execution
    shaping = profile.source_shaping
    tuning = load_ik_tuning_config()
    mode_name = "position_admittance_offline_balanced"
    objective_modes = dict(tuning.objective_modes)
    objective_modes[mode_name] = IKObjectiveWeights(
        single_arm_position_weight=ik.position_weight,
        single_arm_orientation_weight=ik.orientation_weight,
        dual_arm_position_cost=ik.position_weight,
        dual_arm_orientation_cost=ik.orientation_weight,
    )
    tuning = replace(
        tuning,
        ik_objective_mode=mode_name,
        objective_modes=objective_modes,
    )
    effective_velocity = np.asarray(
        config_payload.get(
            "arm_max_angvel_effective_rad_s",
            (5.6, 5.6, 7.2, 12.0, 12.0, 16.0),
        ),
        dtype=float,
    ).reshape(6)
    step_limit = effective_velocity / float(profile.rates.ik_hz)
    scene_profile = str(
        _mapping(config_payload.get("position_control_profile")).get(
            "scene_profile",
            "dual_arm_hand_proxy_open",
        )
        or "dual_arm_hand_proxy_open"
    )
    retargeter = DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            control_mode="absolute",
            scene_profile=scene_profile,
            contact_mode="boundary_slide",
            solver_frequency=float(profile.rates.ik_hz),
            enforce_realistic_joint_step=True,
            left_execution_max_joint_step_rad=step_limit,
            right_execution_max_joint_step_rad=step_limit,
            left_tool_tcp_pose_flange_m_rad=left_tool_pose,
            right_tool_tcp_pose_flange_m_rad=right_tool_pose,
            left_initial_qpos=initial_qpos["left"],
            right_initial_qpos=initial_qpos["right"],
            ik_tuning=tuning,
            joint_velocity_limit_profile=ik.joint_velocity_limit_profile,
            solve_max_iterations=ik.solve_max_iterations,
            solve_min_iterations=ik.solve_min_iterations,
            solve_step_mode="single_step_singularity_aware",
            branch_continuity_anchor_enabled=False,
            joint_limit_guard_mode="legacy",
            joint_acceleration_limit_rad_s2=command.max_acceleration_rad_s2,
            admitted_reference_motion_limit_enabled=False,
            damping=ik.damping,
            solve_adaptive_iterations_enabled=ik.adaptive_iterations,
            solve_iteration_reserve_ms=ik.iteration_reserve_ms,
            candidate_certification_reserve_ms=(
                ik.candidate_certification_reserve_ms
            ),
            # The offline host may be contended; a warning-only wall-clock
            # budget must not make a deterministic geometric replay emit
            # non-JSON text or change candidate certification coverage.
            solve_time_budget_ms=1000.0,
            position_singularity_candidate_certificate_enabled=(
                ik.require_singularity_candidate_certificate
            ),
            output_command_type="joint_vector",
            target_pose_smoother=None,
            output_qpos_smoothing_alpha=shaping.post_ik_qpos_smoothing_alpha,
            output_qpos_smoothing_adaptive_enabled=(
                shaping.post_ik_qpos_smoothing_adaptive
            ),
            output_qpos_smoothing_fast_alpha=(
                shaping.post_ik_qpos_smoothing_fast_alpha
            ),
            output_qpos_normal_fast_path_enabled=(
                shaping.post_ik_qpos_normal_fast_path
            ),
            output_qpos_normal_fast_path_alpha=(
                shaping.post_ik_qpos_normal_fast_path_alpha
            ),
            output_qpos_normal_fast_path_guard_deg=(
                shaping.post_ik_qpos_normal_fast_path_guard_deg
            ),
            output_qpos_normal_fast_path_quality_grace_sec=(
                shaping.post_ik_qpos_normal_fast_path_quality_grace_sec
            ),
            output_qpos_admittance_fast_path_enabled=(
                True
            ),
            output_qpos_admittance_fast_path_alpha=(
                POSITION_ADMITTANCE_ACTIVE_QPOS_FAST_PATH_ALPHA
            ),
            output_qpos_admittance_micro_reversal_hold_deg=(
                POSITION_ADMITTANCE_ACTIVE_QPOS_MICRO_REVERSAL_HOLD_DEG
            ),
            output_qpos_smoothing_singularity_alpha=0.65,
            current_target_hold_enabled=True,
            local_step_candidate_selection_enabled=True,
            prefer_non_wrist_singular_branch=(
                ik.prefer_non_wrist_singular_branch
            ),
            wrist_singularity_threshold_deg=3.0,
            wrist_branch_switch_guard_deg=12.0,
            wrist_q5_hysteresis_release_deg=6.0,
            wrist_q5_hysteresis_requires_escape_latch=(
                shaping.wrist_q5_hysteresis_requires_escape_latch
            ),
            prefer_non_elbow_singular_branch=(
                ik.prefer_non_elbow_singular_branch
            ),
            elbow_singularity_threshold_deg=3.0,
            elbow_singularity_selection_margin_deg=0.25,
            prefer_non_shoulder_singular_branch=(
                ik.prefer_non_shoulder_singular_branch
            ),
            shoulder_singularity_threshold_m=0.05,
            local_step_candidate_clip_scale=(
                ik.local_step_candidate_clip_scale
            ),
            inactive_side_hold_enabled=True,
            inactive_side_hold_execution_cap_enabled=True,
            inactive_side_hold_execution_release_enabled=True,
            inactive_side_hold_execution_release_translation_static_m=(
                shaping.inactive_execution_release_translation_static_m
            ),
            inactive_side_hold_execution_release_translation_full_m=(
                shaping.inactive_execution_release_translation_full_m
            ),
            inactive_side_hold_execution_release_rotation_static_deg=(
                shaping.inactive_execution_release_rotation_static_deg
            ),
            inactive_side_hold_execution_release_rotation_full_deg=(
                shaping.inactive_execution_release_rotation_full_deg
            ),
        )
    )
    return retargeter, step_limit


def _source_command(
    *,
    frame: Mapping[str, Any],
    side: str,
    target: np.ndarray,
    modifier_active: bool = False,
) -> EffectorCommand:
    metadata = _mapping(
        _mapping(_mapping(frame.get("command_by_side")).get(side)).get("metadata")
    )
    activity = dict(_mapping(metadata.get("target_activity")))
    if not _mapping(activity.get("motion_window")):
        tracker_channel = _mapping(
            _mapping(frame.get("tracker_channels")).get(
                f"{side}_wrist_pose"
            )
        )
        recorded_motion_window = _mapping(
            tracker_channel.get("motion_window")
        )
        if recorded_motion_window:
            activity["motion_window"] = dict(recorded_motion_window)
    source_metadata: dict[str, Any] = {"tracker_motion_intent": activity}
    if modifier_active:
        modifier_intent: dict[str, Any] = {
            "source": "position_admittance_active_modifier",
            "state": "ACTIVE",
            "inactive": False,
            "hold_weight": 0.0,
            "modifier_active": True,
        }
        for key in ("evidence", "motion_window"):
            value = activity.get(key)
            if isinstance(value, Mapping):
                modifier_intent[key] = dict(value)
        source_metadata["pose_action_motion_intent"] = modifier_intent
    if bool(activity.get("fixed_by_authority", False)):
        source_metadata.update(
            {
                "fix_arm_frozen": True,
                "fix_arm_fixed_side": side,
            }
        )
    return EffectorCommand(
        effector_name=f"{side}_arm",
        side=side,
        command_type="cartesian_pose",
        value=target.copy(),
        frame_name="solver_world",
        metadata=source_metadata,
    )


def _context(
    *,
    frame: Mapping[str, Any],
    targets: Mapping[str, np.ndarray],
    modifier_active_side: str | None = None,
) -> DualArmTargetContext:
    return DualArmTargetContext(
        timestamp=float(
            frame.get("teleop_command_timestamp")
            or frame.get("sample_timestamp")
            or 0.0
        ),
        left_target=targets["left"].copy(),
        right_target=targets["right"].copy(),
        left_world_to_base=None,
        right_world_to_base=None,
        left_cartesian=_source_command(
            frame=frame,
            side="left",
            target=targets["left"],
            modifier_active=modifier_active_side == "left",
        ),
        right_cartesian=_source_command(
            frame=frame,
            side="right",
            target=targets["right"],
            modifier_active=modifier_active_side == "right",
        ),
        left_input_skipped=None,
        right_input_skipped=None,
        target_smoothing_metadata={
            "enabled": False,
            "source": "recorded_post_smoother_translation_teacher_forced_qref",
        },
    )


def _command_qpos(command: Any, *, side: str) -> tuple[np.ndarray, Mapping[str, Any]]:
    effector = command.commands[f"{side}_arm"]
    qpos = np.asarray(effector.value, dtype=float).reshape(6)
    if not np.all(np.isfinite(qpos)):
        raise ValueError(f"{side} IK output is non-finite")
    return qpos, _mapping(effector.metadata)


def _target_at_tick(
    *,
    tick_time: float,
    frame_times: Sequence[float],
    targets: Sequence[Mapping[str, np.ndarray]],
    frame_index: int,
    mode: str,
) -> tuple[dict[str, np.ndarray], int]:
    index = int(frame_index)
    while (
        index + 1 < len(frame_times)
        and frame_times[index + 1] <= tick_time + 1e-9
    ):
        index += 1
    if mode == "zoh" or index + 1 >= len(frame_times):
        return (
            {side: targets[index][side].copy() for side in SIDES},
            index,
        )
    interval = float(frame_times[index + 1] - frame_times[index])
    alpha = (
        0.0
        if interval <= 0.0
        else max(0.0, min(1.0, (tick_time - frame_times[index]) / interval))
    )
    return (
        {
            side: interpolate_pose(
                targets[index][side],
                targets[index + 1][side],
                alpha,
            )
            for side in SIDES
        },
        index,
    )


_COUNTERFACTUAL_STATE_FIELDS = (
    "_last_admitted_velocity_full",
    "_last_target_poses",
    "_jump_hold_active",
    "_jump_hold_full_q",
    "_jump_hold_anchor_targets",
    "_jump_hold_frames",
    "_last_safe_full_q",
    "_last_output_qpos_smoothing_full_q",
    "_output_qpos_smoothing_effective_alpha",
    "_output_qpos_smoothing_update_time",
    "_output_qpos_normal_fast_path_last_quality_valid_time",
    "_singularity_escape_direction_latch",
    "_singularity_escape_direction_failures",
    "_singularity_escape_axis_latch",
    "_joint_limit_escape_direction_latch",
    "_joint_limit_escape_release_guard",
    "_wrist_q5_hysteresis_escape_latch",
    "_candidate_certificate_sequence",
)


def _synchronize_counterfactual_state(
    *,
    authoritative: DualArmCollisionAwareRetargeter,
    counterfactual: DualArmCollisionAwareRetargeter,
    authoritative_qpos: Mapping[str, np.ndarray],
) -> float:
    """Fork one same-tick nominal preview from the admitted Active state.

    The nominal branch is not an independently advancing controller.  It must
    start from the exact state that would receive the modified target on this
    tick; otherwise ordinary stateful IK history, hold tolerance, and release
    path dependence are mislabelled as a persistent force-control effect.

    Only replay-owned retargeters are accepted here.  The helper deliberately
    copies the state that can affect candidate admission while keeping each
    retargeter's model/solver resources independent.  No production runtime
    object or hardware state is touched.
    """

    if authoritative.target_override_filter is not None:
        raise ValueError("authoritative replay retargeter must have no target filter")
    if counterfactual.target_override_filter is not None:
        raise ValueError("counterfactual replay retargeter must have no target filter")
    if authoritative._target_pose_smoother is not None:  # noqa: SLF001
        raise ValueError("continuous replay requires recorded post-smoother targets")
    if counterfactual._target_pose_smoother is not None:  # noqa: SLF001
        raise ValueError("continuous replay requires recorded post-smoother targets")

    counterfactual.set_current_qpos_pair(
        left_qpos=authoritative_qpos["left"],
        right_qpos=authoritative_qpos["right"],
        reset_tracking=False,
    )
    for field in _COUNTERFACTUAL_STATE_FIELDS:
        setattr(
            counterfactual,
            field,
            copy.deepcopy(getattr(authoritative, field)),
        )
    counterfactual.ik_solver.set_configuration_q(
        authoritative.ik_solver.current_q.copy()
    )
    counterfactual._set_solver_motion_reference(  # noqa: SLF001
        counterfactual.ik_solver.current_q
    )
    return float(
        np.max(
            np.abs(
                counterfactual.ik_solver.current_q
                - authoritative.ik_solver.current_q
            )
        )
    )


def _continuous_rollout(
    *,
    mode: str,
    frames: Sequence[Mapping[str, Any]],
    frame_times: Sequence[float],
    raw_targets: Sequence[Mapping[str, np.ndarray]],
    force_ticks: Sequence[_ForceTick],
    selected_side: str,
    initial_qpos: Mapping[str, np.ndarray],
    config_payload: Mapping[str, Any],
    left_tool_pose: tuple[float, ...],
    right_tool_pose: tuple[float, ...],
    max_offset_m: float,
) -> dict[str, Any]:
    """Run one Active qref loop plus a same-state nominal preview at replay_hz."""

    opposite_side = "right" if selected_side == "left" else "left"
    baseline, step_limit = _build_retargeter(
        initial_qpos=initial_qpos,
        config_payload=config_payload,
        left_tool_pose=left_tool_pose,
        right_tool_pose=right_tool_pose,
    )
    candidate, _ = _build_retargeter(
        initial_qpos=initial_qpos,
        config_payload=config_payload,
        left_tool_pose=left_tool_pose,
        right_tool_pose=right_tool_pose,
    )
    candidate_q = {side: initial_qpos[side].copy() for side in SIDES}
    frame_index = 0
    branch_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    candidate_step_ratios: list[float] = []
    baseline_step_ratios: list[float] = []
    selected_incremental_fk: list[float] = []
    opposite_incremental_fk: list[float] = []
    selected_incremental_rotation: list[float] = []
    opposite_incremental_rotation: list[float] = []
    nominal_residuals: list[float] = []
    composed_residuals: list[float] = []
    modifier_norms: list[float] = []
    target_tracking_residuals: list[float] = []
    target_tracking_rotation_residuals: list[float] = []
    model_errors: list[dict[str, Any]] = []
    first_gate_fault: dict[str, Any] | None = None
    completed_ticks = 0
    active_modifier_ticks = 0
    max_incremental_joint_deg = 0.0
    common_seed_q_error_rad: list[float] = []

    for tick_index, tick in enumerate(force_ticks):
        raw, frame_index = _target_at_tick(
            tick_time=tick.now_sec,
            frame_times=frame_times,
            targets=raw_targets,
            frame_index=frame_index,
            mode=mode,
        )
        offset = np.asarray(tick.output_offset_m, dtype=float)
        modifier_norm = float(np.linalg.norm(offset))
        modifier_norms.append(modifier_norm)
        active_modifier_ticks += int(modifier_norm > 1e-12)
        admitted_fk = candidate.solver_world_targets_for_qpos_pair(
            left_qpos=candidate_q["left"],
            right_qpos=candidate_q["right"],
        )
        nominal = float(
            np.linalg.norm(
                admitted_fk[selected_side][:3, 3]
                - raw[selected_side][:3, 3]
            )
        )
        composed = float(
            np.linalg.norm(
                admitted_fk[selected_side][:3, 3]
                - (raw[selected_side][:3, 3] + offset)
            )
        )
        nominal_residuals.append(nominal)
        composed_residuals.append(composed)
        if modifier_norm > 1e-12:
            reason = None
            if modifier_norm > float(max_offset_m) + 1e-9:
                reason = "modifier_envelope"
            elif nominal > float(max_offset_m) + 1e-9:
                reason = "nominal_translation_residual"
            elif composed > float(max_offset_m) + modifier_norm + 1e-9:
                reason = "composed_translation_residual"
            if reason is not None:
                first_gate_fault = {
                    "tick_index": tick_index,
                    "now_sec": tick.now_sec,
                    "frame_index": frame_index,
                    "step_index": frames[frame_index].get("step_index"),
                    "reason": reason,
                    "modifier_norm_m": modifier_norm,
                    "nominal_residual_m": nominal,
                    "composed_residual_m": composed,
                }
                break
        candidate_targets = {side: raw[side].copy() for side in SIDES}
        candidate_targets[selected_side][:3, 3] += offset
        previous_authoritative_q = {
            side: candidate_q[side].copy() for side in SIDES
        }
        try:
            common_seed_q_error_rad.append(
                _synchronize_counterfactual_state(
                    authoritative=candidate,
                    counterfactual=baseline,
                    authoritative_qpos=previous_authoritative_q,
                )
            )
            baseline_command = baseline.build_command_for_context(
                _context(frame=frames[frame_index], targets=raw)
            )
            candidate_command = candidate.build_command_for_context(
                _context(
                    frame=frames[frame_index],
                    targets=candidate_targets,
                    modifier_active_side=(
                        selected_side if modifier_norm > 1e-12 else None
                    ),
                )
            )
            baseline_q: dict[str, np.ndarray] = {}
            baseline_metadata: dict[str, Mapping[str, Any]] = {}
            candidate_metadata: dict[str, Mapping[str, Any]] = {}
            for side in SIDES:
                baseline_q[side], baseline_metadata[side] = _command_qpos(
                    baseline_command,
                    side=side,
                )
                candidate_q[side], candidate_metadata[side] = _command_qpos(
                    candidate_command,
                    side=side,
                )
            candidate.set_current_qpos_pair(
                left_qpos=candidate_q["left"],
                right_qpos=candidate_q["right"],
                reset_tracking=False,
            )
            baseline_fk = baseline.solver_world_targets_for_qpos_pair(
                left_qpos=baseline_q["left"],
                right_qpos=baseline_q["right"],
            )
            candidate_fk = candidate.solver_world_targets_for_qpos_pair(
                left_qpos=candidate_q["left"],
                right_qpos=candidate_q["right"],
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic report path.
            model_errors.append(
                {
                    "tick_index": tick_index,
                    "frame_index": frame_index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            break

        completed_ticks += 1
        baseline_step_ratios.append(
            float(
                max(
                    np.max(
                        np.abs(
                            baseline_q[side] - previous_authoritative_q[side]
                        )
                        / step_limit
                    )
                    for side in SIDES
                )
            )
        )
        candidate_step_ratios.append(
            float(
                max(
                    np.max(
                        np.abs(
                            candidate_q[side] - previous_authoritative_q[side]
                        )
                        / step_limit
                    )
                    for side in SIDES
                )
            )
        )
        max_incremental_joint_deg = max(
            max_incremental_joint_deg,
            max(
                float(
                    np.rad2deg(
                        np.max(
                            np.abs(candidate_q[side] - baseline_q[side])
                        )
                    )
                )
                for side in SIDES
            ),
        )
        selected_incremental_fk.append(
            float(
                np.linalg.norm(
                    candidate_fk[selected_side][:3, 3]
                    - baseline_fk[selected_side][:3, 3]
                )
            )
        )
        opposite_incremental_fk.append(
            float(
                np.linalg.norm(
                    candidate_fk[opposite_side][:3, 3]
                    - baseline_fk[opposite_side][:3, 3]
                )
            )
        )
        selected_incremental_rotation.append(
            _rotation_error_deg(
                baseline_fk[selected_side][:3, :3],
                candidate_fk[selected_side][:3, :3],
            )
        )
        opposite_incremental_rotation.append(
            _rotation_error_deg(
                baseline_fk[opposite_side][:3, :3],
                candidate_fk[opposite_side][:3, :3],
            )
        )
        target_tracking_residuals.append(
            float(
                np.linalg.norm(
                    candidate_fk[selected_side][:3, 3]
                    - candidate_targets[selected_side][:3, 3]
                )
            )
        )
        target_tracking_rotation_residuals.append(
            _rotation_error_deg(
                candidate_fk[selected_side][:3, :3],
                candidate_targets[selected_side][:3, :3],
            )
        )
        for side in SIDES:
            branch_counts[
                str(candidate_metadata[side].get("ik_branch_candidate"))
            ] += 1
            status_counts[
                str(candidate_metadata[side].get("collision_avoidance_status"))
            ] += 1

    final_incremental_m = (
        selected_incremental_fk[-1] if selected_incremental_fk else None
    )
    final_target_tracking_m = (
        target_tracking_residuals[-1]
        if target_tracking_residuals
        else None
    )
    final_target_tracking_rotation_deg = (
        target_tracking_rotation_residuals[-1]
        if target_tracking_rotation_residuals
        else None
    )
    completed_all = completed_ticks == len(force_ticks)
    local_limit_m = float(max_offset_m) + 5e-4
    checks = [
        {
            "name": "continuous_same_tick_counterfactual_seed",
            "passed": bool(
                common_seed_q_error_rad
                and max(common_seed_q_error_rad) <= 1e-12
            ),
            "detail": {
                "max_full_q_error_rad": (
                    max(common_seed_q_error_rad)
                    if common_seed_q_error_rad
                    else None
                ),
                "semantics": (
                    "nominal_and_modified_start_from_same_admitted_state"
                ),
            },
        },
        {
            "name": "continuous_rollout_completed",
            "passed": completed_all,
            "detail": {
                "expected_ticks": len(force_ticks),
                "completed_ticks": completed_ticks,
                "first_gate_fault": first_gate_fault,
                "model_errors": model_errors[:5],
            },
        },
        {
            "name": "continuous_formal_joint_step",
            "passed": bool(
                not candidate_step_ratios
                or (
                    max(candidate_step_ratios) <= 1.0 + 1e-6
                    and max(baseline_step_ratios) <= 1.0 + 1e-6
                )
            ),
            "detail": {
                "candidate_max_ratio": (
                    max(candidate_step_ratios)
                    if candidate_step_ratios
                    else None
                ),
                "baseline_max_ratio": (
                    max(baseline_step_ratios)
                    if baseline_step_ratios
                    else None
                ),
            },
        },
        {
            "name": "continuous_incremental_fk_envelope",
            "passed": bool(
                not selected_incremental_fk
                or (
                    max(selected_incremental_fk) <= local_limit_m
                    and max(opposite_incremental_fk) <= local_limit_m
                    and max(selected_incremental_rotation)
                    <= INCREMENTAL_FK_ROTATION_GATE_DEG
                    and max(opposite_incremental_rotation)
                    <= INCREMENTAL_FK_ROTATION_GATE_DEG
                )
            ),
            "detail": {
                "selected_translation_max_m": (
                    max(selected_incremental_fk)
                    if selected_incremental_fk
                    else None
                ),
                "opposite_translation_max_m": (
                    max(opposite_incremental_fk)
                    if opposite_incremental_fk
                    else None
                ),
                "translation_limit_m": local_limit_m,
                "selected_rotation_max_deg": (
                    max(selected_incremental_rotation)
                    if selected_incremental_rotation
                    else None
                ),
                "opposite_rotation_max_deg": (
                    max(opposite_incremental_rotation)
                    if opposite_incremental_rotation
                    else None
                ),
                "rotation_limit_deg": INCREMENTAL_FK_ROTATION_GATE_DEG,
            },
        },
        {
            "name": "continuous_release_tail",
            "passed": bool(
                not completed_all
                or final_incremental_m is not None
                and final_incremental_m <= RELEASE_TARGET_TRANSLATION_GATE_M
                and final_target_tracking_m is not None
                and final_target_tracking_m
                <= RELEASE_TARGET_TRANSLATION_GATE_M
                and final_target_tracking_rotation_deg is not None
                and final_target_tracking_rotation_deg
                <= RELEASE_TARGET_ROTATION_GATE_DEG
            ),
            "detail": {
                "final_selected_incremental_fk_m": final_incremental_m,
                "final_target_tracking_translation_m": (
                    final_target_tracking_m
                ),
                "final_target_tracking_rotation_deg": (
                    final_target_tracking_rotation_deg
                ),
                "translation_limit_m": RELEASE_TARGET_TRANSLATION_GATE_M,
                "rotation_limit_deg": RELEASE_TARGET_ROTATION_GATE_DEG,
                "final_modifier_norm_m": (
                    modifier_norms[-1] if modifier_norms else None
                ),
                "semantics": (
                    "same_tick_modifier_effect_and_absolute_target_rejoin_"
                    "after_zero_force_release"
                ),
            },
        },
    ]
    return {
        "mode": mode,
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "tick_count": len(force_ticks),
        "completed_tick_count": completed_ticks,
        "active_modifier_tick_count": active_modifier_ticks,
        "first_gate_fault": first_gate_fault,
        "model_errors": model_errors,
        "modifier_norm_m": _stats(modifier_norms),
        "nominal_translation_residual_m": _stats(nominal_residuals),
        "composed_translation_residual_m": _stats(composed_residuals),
        "selected_incremental_fk_translation_m": _stats(
            selected_incremental_fk
        ),
        "opposite_incremental_fk_translation_m": _stats(
            opposite_incremental_fk
        ),
        "target_tracking_residual_m": _stats(target_tracking_residuals),
        "target_tracking_rotation_residual_deg": _stats(
            target_tracking_rotation_residuals
        ),
        "max_incremental_joint_deg": max_incremental_joint_deg,
        "counterfactual_baseline": (
            "same_tick_same_admitted_state_nominal_preview"
        ),
        "common_seed_q_error_rad": _stats(common_seed_q_error_rad),
        "branch_counts": dict(branch_counts),
        "status_counts": dict(status_counts),
    }


def replay_recorded_force_through_shared_ik(
    path: Path,
    *,
    side: str = "left",
    max_offset_m: float = DEFAULT_C0_MAX_OFFSET_M,
    max_speed_m_s: float = DEFAULT_C0_MAX_SPEED_M_S,
    max_return_speed_m_s: float | None = None,
    max_output_slew_m_s: float | None = None,
    replay_hz: float = DEFAULT_REPLAY_HZ,
    release_tail_sec: float = 6.0,
    left_tool_profile: Path = Path(DEFAULT_LEFT_TOOL_PROFILE),
    right_tool_profile: Path = Path(DEFAULT_RIGHT_TOOL_PROFILE),
) -> dict[str, Any]:
    """Run a teacher-forced incremental C0-to-IK counterfactual."""

    selected_side = str(side).strip().lower()
    if selected_side not in SIDES:
        raise ValueError("side must be left or right")
    opposite_side = "right" if selected_side == "left" else "left"
    log_path = _resolve_log(path)
    rows = _read_rows(log_path)
    frames = _frame_rows(rows)
    if not frames:
        raise ValueError("IK replay requires action-debug frame rows")
    config_payload = _config_payload(rows)
    force_outputs, force_ticks, receipt, candidate_config = _sample_force_outputs(
        rows,
        side=selected_side,
        max_offset_m=max_offset_m,
        max_speed_m_s=max_speed_m_s,
        max_return_speed_m_s=max_return_speed_m_s,
        max_output_slew_m_s=max_output_slew_m_s,
        replay_hz=replay_hz,
        release_tail_sec=release_tail_sec,
    )
    first_q = {
        side_name: _qpos(_mapping(frames[0].get("command_by_side")), side=side_name)
        for side_name in SIDES
    }
    left_tool_pose = _tool_pose(Path(left_tool_profile), side="left")
    right_tool_pose = _tool_pose(Path(right_tool_profile), side="right")
    baseline_retargeter, step_limit = _build_retargeter(
        initial_qpos=first_q,
        config_payload=config_payload,
        left_tool_pose=left_tool_pose,
        right_tool_pose=right_tool_pose,
    )
    candidate_retargeter, _ = _build_retargeter(
        initial_qpos=first_q,
        config_payload=config_payload,
        left_tool_pose=left_tool_pose,
        right_tool_pose=right_tool_pose,
    )

    gate_reasons: Counter[str] = Counter()
    baseline_branches: Counter[str] = Counter()
    candidate_branches: Counter[str] = Counter()
    baseline_statuses: Counter[str] = Counter()
    candidate_statuses: Counter[str] = Counter()
    model_errors: list[dict[str, Any]] = []
    nonzero_count = 0
    eligible_count = 0
    exact_rotation_count = 0
    rotation_proxy_counts: Counter[str] = Counter()
    nominal_residuals: list[float] = []
    composed_residuals: list[float] = []
    rotation_proxy_residuals: list[float] = []
    force_offsets: list[float] = []
    selected_incremental_joint_deg: list[float] = []
    opposite_incremental_joint_deg: list[float] = []
    selected_raw_candidate_step_deg: list[float] = []
    selected_incremental_fk_m: list[float] = []
    selected_incremental_fk_rotation_deg: list[float] = []
    opposite_incremental_fk_m: list[float] = []
    opposite_incremental_fk_rotation_deg: list[float] = []
    baseline_seed_step_ratios: list[float] = []
    candidate_seed_step_ratios: list[float] = []
    incremental_step_ratios: list[float] = []
    jump_gate_regression_count = 0
    collision_status_regression_count = 0
    first_eligible_diagnostic: dict[str, Any] | None = None
    first_raw_motion_diagnostic: dict[str, Any] | None = None
    raw_targets_by_frame: list[dict[str, np.ndarray]] = []
    tolerance = 1e-9

    for frame_index, (frame, force_output) in enumerate(zip(frames, force_outputs)):
        qref = {
            side_name: _qpos(
                _mapping(frame.get("command_by_side")),
                side=side_name,
            )
            for side_name in SIDES
        }
        seed_fk = baseline_retargeter.solver_world_targets_for_qpos_pair(
            left_qpos=qref["left"],
            right_qpos=qref["right"],
        )
        raw_target_pair: dict[str, np.ndarray] = {}
        for side_name in SIDES:
            raw_target_pair[side_name] = seed_fk[side_name].copy()
            raw_target_pair[side_name][:3, 3] = _raw_target_translation(
                frame,
                side=side_name,
            )
            side_rotation, _ = _raw_target_rotation(
                frame,
                side=side_name,
            )
            if side_rotation is not None:
                raw_target_pair[side_name][:3, :3] = side_rotation
        raw_targets_by_frame.append(raw_target_pair)
        raw_translation = _raw_target_translation(frame, side=selected_side)
        offset = np.asarray(force_output.output_offset_m, dtype=float)
        offset_norm = float(np.linalg.norm(offset))
        force_offsets.append(offset_norm)
        nominal_residual = float(
            np.linalg.norm(seed_fk[selected_side][:3, 3] - raw_translation)
        )
        composed_translation = raw_translation + offset
        composed_residual = float(
            np.linalg.norm(
                seed_fk[selected_side][:3, 3] - composed_translation
            )
        )
        nominal_residuals.append(nominal_residual)
        composed_residuals.append(composed_residual)
        raw_rotation, rotation_source = _raw_target_rotation(
            frame,
            side=selected_side,
        )
        rotation_proxy_counts[rotation_source] += 1
        if rotation_source == "recorded_post_smoother_solver_world":
            exact_rotation_count += 1
        if raw_rotation is not None:
            rotation_proxy_residuals.append(
                _rotation_error_deg(
                    seed_fk[selected_side][:3, :3],
                    raw_rotation,
                )
            )
        if offset_norm <= 1e-12:
            gate_reasons["zero_modifier"] += 1
            continue
        nonzero_count += 1
        if offset_norm > float(max_offset_m) + tolerance:
            gate_reasons["modifier_envelope"] += 1
            continue
        if nominal_residual > float(max_offset_m) + tolerance:
            gate_reasons["nominal_translation_residual"] += 1
            continue
        if composed_residual > float(max_offset_m) + offset_norm + tolerance:
            gate_reasons["composed_translation_residual"] += 1
            continue
        if (
            raw_rotation is not None
            and rotation_source == "recorded_post_smoother_solver_world"
            and rotation_proxy_residuals[-1] > ROTATION_GATE_DEG + tolerance
        ):
            gate_reasons["nominal_rotation_residual"] += 1
            continue
        gate_reasons["eligible"] += 1
        eligible_count += 1

        baseline_targets = {
            side_name: seed_fk[side_name].copy() for side_name in SIDES
        }
        baseline_targets[selected_side][:3, 3] = raw_translation
        candidate_targets = {
            side_name: target.copy()
            for side_name, target in baseline_targets.items()
        }
        candidate_targets[selected_side][:3, 3] = composed_translation
        try:
            for retargeter in (baseline_retargeter, candidate_retargeter):
                retargeter.set_current_qpos_pair(
                    left_qpos=qref["left"],
                    right_qpos=qref["right"],
                    reset_tracking=True,
                )
                retargeter.reset_target_pose_smoothing()
                # A real Active tick never starts from an uninitialized
                # post-IK smoother: the preceding accepted qref is already
                # its committed state.  Prime that exact zero-motion state,
                # then preserve it while teacher-forcing the recorded qref.
                retargeter.build_command_for_context(
                    _context(frame=frame, targets=seed_fk)
                )
                retargeter.set_current_qpos_pair(
                    left_qpos=qref["left"],
                    right_qpos=qref["right"],
                    reset_tracking=False,
                )
            baseline_command = baseline_retargeter.build_command_for_context(
                _context(frame=frame, targets=baseline_targets)
            )
            candidate_command = candidate_retargeter.build_command_for_context(
                _context(
                    frame=frame,
                    targets=candidate_targets,
                    modifier_active_side=selected_side,
                )
            )
            baseline_q: dict[str, np.ndarray] = {}
            candidate_q: dict[str, np.ndarray] = {}
            baseline_metadata: dict[str, Mapping[str, Any]] = {}
            candidate_metadata: dict[str, Mapping[str, Any]] = {}
            for side_name in SIDES:
                baseline_q[side_name], baseline_metadata[side_name] = _command_qpos(
                    baseline_command,
                    side=side_name,
                )
                candidate_q[side_name], candidate_metadata[side_name] = _command_qpos(
                    candidate_command,
                    side=side_name,
                )
            baseline_fk = baseline_retargeter.solver_world_targets_for_qpos_pair(
                left_qpos=baseline_q["left"],
                right_qpos=baseline_q["right"],
            )
            candidate_fk = candidate_retargeter.solver_world_targets_for_qpos_pair(
                left_qpos=candidate_q["left"],
                right_qpos=candidate_q["right"],
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic report path.
            model_errors.append(
                {
                    "frame_index": frame_index,
                    "step_index": frame.get("step_index"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        for side_name in SIDES:
            baseline_branches[str(baseline_metadata[side_name].get("ik_branch_candidate"))] += 1
            candidate_branches[str(candidate_metadata[side_name].get("ik_branch_candidate"))] += 1
            baseline_status = str(
                baseline_metadata[side_name].get("collision_avoidance_status")
            )
            candidate_status = str(
                candidate_metadata[side_name].get("collision_avoidance_status")
            )
            baseline_statuses[baseline_status] += 1
            candidate_statuses[candidate_status] += 1
            baseline_jump = str(
                baseline_metadata[side_name].get("ik_jump_gate_status")
            )
            candidate_jump = str(
                candidate_metadata[side_name].get("ik_jump_gate_status")
            )
            jump_gate_regression_count += int(
                baseline_jump == "pass" and candidate_jump != "pass"
            )
            collision_status_regression_count += int(
                baseline_status == "tracking" and candidate_status != "tracking"
            )

        raw_candidate = candidate_metadata[selected_side].get("q_ik_raw")
        try:
            raw_candidate_q = np.asarray(raw_candidate, dtype=float).reshape(6)
        except (TypeError, ValueError):
            raw_candidate_q = None
        if raw_candidate_q is not None and np.all(np.isfinite(raw_candidate_q)):
            selected_raw_candidate_step_deg.append(
                float(
                    np.rad2deg(
                        np.max(np.abs(raw_candidate_q - qref[selected_side]))
                    )
                )
            )
            if first_raw_motion_diagnostic is None:
                first_raw_motion_diagnostic = {
                    "frame_index": frame_index,
                    "step_index": frame.get("step_index"),
                    "offset_m": offset.astype(float).tolist(),
                    "branch": candidate_metadata[selected_side].get(
                        "ik_branch_candidate"
                    ),
                    "raw_candidate_step_max_deg": (
                        selected_raw_candidate_step_deg[-1]
                    ),
                    "output_step_max_deg": float(
                        np.rad2deg(
                            np.max(
                                np.abs(
                                    candidate_q[selected_side]
                                    - qref[selected_side]
                                )
                            )
                        )
                    ),
                    "target_activity": candidate_metadata[selected_side].get(
                        "target_activity"
                    ),
                }
        if first_eligible_diagnostic is None:
            smoothing = _mapping(
                candidate_metadata[selected_side].get("output_qpos_smoothing")
            )
            first_eligible_diagnostic = {
                "frame_index": frame_index,
                "step_index": frame.get("step_index"),
                "offset_m": offset.astype(float).tolist(),
                "nominal_residual_m": nominal_residual,
                "composed_residual_m": composed_residual,
                "branch": candidate_metadata[selected_side].get(
                    "ik_branch_candidate"
                ),
                "output_qpos_smoothing": {
                    key: smoothing.get(key)
                    for key in (
                        "enabled",
                        "applied",
                        "initialized",
                        "stage",
                        "raw_target_joint_deg",
                        "output_target_joint_deg",
                        "raw_to_output_max_delta_deg",
                    )
                },
                "raw_candidate_step_max_deg": (
                    selected_raw_candidate_step_deg[-1]
                    if selected_raw_candidate_step_deg
                    else None
                ),
                "output_step_max_deg": float(
                    np.rad2deg(
                        np.max(
                            np.abs(
                                candidate_q[selected_side]
                                - qref[selected_side]
                            )
                        )
                    )
                ),
            }

        selected_incremental_joint_deg.append(
            float(
                np.rad2deg(
                    np.max(
                        np.abs(
                            candidate_q[selected_side]
                            - baseline_q[selected_side]
                        )
                    )
                )
            )
        )
        opposite_incremental_joint_deg.append(
            float(
                np.rad2deg(
                    np.max(
                        np.abs(
                            candidate_q[opposite_side]
                            - baseline_q[opposite_side]
                        )
                    )
                )
            )
        )
        selected_incremental_fk_m.append(
            float(
                np.linalg.norm(
                    candidate_fk[selected_side][:3, 3]
                    - baseline_fk[selected_side][:3, 3]
                )
            )
        )
        selected_incremental_fk_rotation_deg.append(
            _rotation_error_deg(
                baseline_fk[selected_side][:3, :3],
                candidate_fk[selected_side][:3, :3],
            )
        )
        opposite_incremental_fk_m.append(
            float(
                np.linalg.norm(
                    candidate_fk[opposite_side][:3, 3]
                    - baseline_fk[opposite_side][:3, 3]
                )
            )
        )
        opposite_incremental_fk_rotation_deg.append(
            _rotation_error_deg(
                baseline_fk[opposite_side][:3, :3],
                candidate_fk[opposite_side][:3, :3],
            )
        )
        baseline_seed_step_ratios.append(
            float(
                max(
                    np.max(
                        np.abs(baseline_q[side_name] - qref[side_name])
                        / step_limit
                    )
                    for side_name in SIDES
                )
            )
        )
        candidate_seed_step_ratios.append(
            float(
                max(
                    np.max(
                        np.abs(candidate_q[side_name] - qref[side_name])
                        / step_limit
                    )
                    for side_name in SIDES
                )
            )
        )
        incremental_step_ratios.append(
            float(
                max(
                    np.max(
                        np.abs(candidate_q[side_name] - baseline_q[side_name])
                        / (2.0 * step_limit)
                    )
                    for side_name in SIDES
                )
            )
        )

    frame_times = [output.observed_at_sec for output in force_outputs]
    continuous_rollouts = {
        mode: _continuous_rollout(
            mode=mode,
            frames=frames,
            frame_times=frame_times,
            raw_targets=raw_targets_by_frame,
            force_ticks=force_ticks,
            selected_side=selected_side,
            initial_qpos=first_q,
            config_payload=config_payload,
            left_tool_pose=left_tool_pose,
            right_tool_pose=right_tool_pose,
            max_offset_m=max_offset_m,
        )
        for mode in ("zoh", "linear")
    }
    incident_like = bool(
        nonzero_count > 0
        and nominal_residuals
        and max(nominal_residuals) > float(max_offset_m) + tolerance
    )

    def _continuous_mode_safe(rollout: Mapping[str, Any]) -> bool:
        if bool(rollout.get("passed")):
            return True
        fault = _mapping(rollout.get("first_gate_fault"))
        return bool(
            incident_like
            and fault.get("reason")
            in {
                "nominal_translation_residual",
                "composed_translation_residual",
            }
            and not rollout.get("model_errors")
        )

    continuous_safe = all(
        _continuous_mode_safe(rollout)
        for rollout in continuous_rollouts.values()
    )

    selected_fk_limit_m = float(max_offset_m) + 5e-4
    opposite_fk_limit_m = float(max_offset_m) + 5e-4
    model_success_count = eligible_count - len(model_errors)
    checks = [
        {
            "name": "recorded_profile_and_force_receipt",
            "passed": bool(
                receipt
                and receipt.get("enabled") is True
                and _profile_name(config_payload)
                == RM65_BIMANUAL_POSITION_V2_BALANCED_NAME
            ),
            "detail": {
                "recorded_position_profile": _profile_name(config_payload),
                "recorded_admittance_mode": receipt.get("requested_mode"),
            },
        },
        {
            "name": "c0_force_output_envelope",
            "passed": bool(
                force_offsets
                and max(force_offsets) <= float(max_offset_m) + tolerance
            ),
            "detail": {
                "max_output_offset_m": max(force_offsets) if force_offsets else None,
                "limit_m": float(max_offset_m),
            },
        },
        {
            "name": "translation_gate_accounts_for_every_nonzero_candidate",
            "passed": nonzero_count
            == eligible_count
            + sum(
                count
                for reason, count in gate_reasons.items()
                if reason
                not in {
                    "zero_modifier",
                    "eligible",
                }
            ),
            "detail": {
                "nonzero": nonzero_count,
                "eligible": eligible_count,
                "reasons": dict(gate_reasons),
            },
        },
        {
            "name": "shared_ik_has_nonzero_candidate_coverage",
            "passed": eligible_count > 0,
            "detail": {
                "eligible": eligible_count,
                "nonzero": nonzero_count,
                "meaning": (
                    "a zero-output trace is a no-op check, not force-to-IK "
                    "coverage"
                ),
            },
        },
        {
            "name": "shared_ik_completed_for_every_eligible_candidate",
            "passed": model_success_count == eligible_count,
            "detail": {
                "eligible": eligible_count,
                "completed": model_success_count,
                "errors": model_errors[:10],
            },
        },
        {
            "name": "formal_per_tick_joint_step_preserved",
            "passed": (
                not baseline_seed_step_ratios
                or (
                    max(baseline_seed_step_ratios) <= 1.0 + 1e-6
                    and max(candidate_seed_step_ratios) <= 1.0 + 1e-6
                    and max(incremental_step_ratios) <= 1.0 + 1e-6
                )
            ),
            "detail": {
                "baseline_max_ratio": (
                    max(baseline_seed_step_ratios)
                    if baseline_seed_step_ratios
                    else None
                ),
                "candidate_max_ratio": (
                    max(candidate_seed_step_ratios)
                    if candidate_seed_step_ratios
                    else None
                ),
                "candidate_vs_baseline_two_step_max_ratio": (
                    max(incremental_step_ratios)
                    if incremental_step_ratios
                    else None
                ),
            },
        },
        {
            "name": "incremental_fk_response_stays_local",
            "passed": (
                not selected_incremental_fk_m
                or (
                    max(selected_incremental_fk_m) <= selected_fk_limit_m
                    and max(opposite_incremental_fk_m) <= opposite_fk_limit_m
                    and max(selected_incremental_fk_rotation_deg) <= 0.5
                    and max(opposite_incremental_fk_rotation_deg) <= 0.5
                )
            ),
            "detail": {
                "selected_translation_max_m": (
                    max(selected_incremental_fk_m)
                    if selected_incremental_fk_m
                    else None
                ),
                "opposite_translation_max_m": (
                    max(opposite_incremental_fk_m)
                    if opposite_incremental_fk_m
                    else None
                ),
                "translation_limit_m": selected_fk_limit_m,
                "selected_rotation_max_deg": (
                    max(selected_incremental_fk_rotation_deg)
                    if selected_incremental_fk_rotation_deg
                    else None
                ),
                "opposite_rotation_max_deg": (
                    max(opposite_incremental_fk_rotation_deg)
                    if opposite_incremental_fk_rotation_deg
                    else None
                ),
                "rotation_limit_deg": 0.5,
            },
        },
        {
            "name": "no_jump_or_collision_status_regression",
            "passed": (
                jump_gate_regression_count == 0
                and collision_status_regression_count == 0
            ),
            "detail": {
                "jump_gate_regression_count": jump_gate_regression_count,
                "collision_status_regression_count": (
                    collision_status_regression_count
                ),
            },
        },
        {
            "name": "continuous_60hz_closed_loop",
            "passed": continuous_safe,
            "detail": {
                "incident_like_recorded_residual": incident_like,
                "semantics": (
                    "teacher_forced_recorded_qref_gate_plus_"
                    "same_tick_same_state_continuous_incremental_force_effect"
                ),
                "modes": {
                    mode: {
                        "passed": rollout.get("passed"),
                        "completed_tick_count": rollout.get(
                            "completed_tick_count"
                        ),
                        "tick_count": rollout.get("tick_count"),
                        "first_gate_fault": rollout.get("first_gate_fault"),
                    }
                    for mode, rollout in continuous_rollouts.items()
                },
            },
        },
    ]
    return {
        "schema": "dexteleop.position_admittance_shared_ik_replay/v2",
        "passed": all(bool(check["passed"]) for check in checks),
        "evidence_level": "offline_15hz_teacher_forced_shared_rm65_ik",
        "path": str(log_path),
        "side": selected_side,
        "candidate_position_profile": RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
        "recorded_position_profile": _profile_name(config_payload),
        "candidate_config": candidate_config,
        "tool_profiles": {
            "left": {
                "path": str(Path(left_tool_profile)),
                "tcp_pose_flange_m_rad": list(left_tool_pose),
            },
            "right": {
                "path": str(Path(right_tool_profile)),
                "tcp_pose_flange_m_rad": list(right_tool_pose),
            },
        },
        "limitations": [
            "action-debug samples qref and force at 15 Hz, not every 60 Hz IK tick",
            (
                "the replay is teacher-forced from recorded qref and does "
                "not model hardware realization"
            ),
            (
                "the continuous nominal preview is forked every tick from the modified "
                "branch's exact admitted state; it has no independent history, so their "
                "difference isolates the same-tick force-modifier effect"
            ),
            (
                "the same-state fork is an offline counterfactual; production Active "
                "remains disabled and does not yet execute a dual-solve post-IK gate"
            ),
            (
                "old logs lack exact post-smoother target rotation; admitted-qref FK rotation "
                "isolates the incremental translation response"
            ),
            (
                "saved force is already transformed and filtered, so raw "
                "tool-frame conversion is not re-evaluated"
            ),
        ],
        "checks": checks,
        "frame_count": len(frames),
        "nonzero_candidate_count": nonzero_count,
        "translation_gate_eligible_count": eligible_count,
        "shared_ik_completed_count": model_success_count,
        "shared_ik_coverage_available": eligible_count > 0,
        "gate_reason_counts": dict(gate_reasons),
        "exact_recorded_rotation_count": exact_rotation_count,
        "rotation_source_counts": dict(rotation_proxy_counts),
        "nominal_translation_residual_m": _stats(nominal_residuals),
        "composed_translation_residual_m": _stats(composed_residuals),
        "source_rotation_proxy_residual_deg": _stats(rotation_proxy_residuals),
        "force_output_offset_norm_m": _stats(force_offsets),
        "selected_incremental_qref_max_deg": _stats(
            selected_incremental_joint_deg
        ),
        "selected_raw_candidate_step_max_deg": _stats(
            selected_raw_candidate_step_deg
        ),
        "opposite_incremental_qref_max_deg": _stats(
            opposite_incremental_joint_deg
        ),
        "selected_incremental_fk_translation_m": _stats(
            selected_incremental_fk_m
        ),
        "selected_incremental_fk_rotation_deg": _stats(
            selected_incremental_fk_rotation_deg
        ),
        "opposite_incremental_fk_translation_m": _stats(
            opposite_incremental_fk_m
        ),
        "opposite_incremental_fk_rotation_deg": _stats(
            opposite_incremental_fk_rotation_deg
        ),
        "baseline_branch_counts": dict(baseline_branches),
        "candidate_branch_counts": dict(candidate_branches),
        "baseline_status_counts": dict(baseline_statuses),
        "candidate_status_counts": dict(candidate_statuses),
        "first_eligible_diagnostic": first_eligible_diagnostic,
        "first_raw_motion_diagnostic": first_raw_motion_diagnostic,
        "continuous_rollouts": continuous_rollouts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--side", choices=SIDES, default="left")
    parser.add_argument(
        "--max-offset-mm",
        type=float,
        default=DEFAULT_C0_MAX_OFFSET_M * 1000.0,
    )
    parser.add_argument(
        "--max-speed-mm-s",
        type=float,
        default=DEFAULT_C0_MAX_SPEED_M_S * 1000.0,
    )
    parser.add_argument(
        "--max-return-speed-mm-s",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-output-slew-mm-s",
        type=float,
        default=None,
    )
    parser.add_argument("--replay-hz", type=float, default=DEFAULT_REPLAY_HZ)
    parser.add_argument(
        "--release-tail-sec",
        type=float,
        default=6.0,
        help="zero-force tail appended to the offline continuous rollout",
    )
    parser.add_argument(
        "--left-tool-profile",
        type=Path,
        default=Path(DEFAULT_LEFT_TOOL_PROFILE),
    )
    parser.add_argument(
        "--right-tool-profile",
        type=Path,
        default=Path(DEFAULT_RIGHT_TOOL_PROFILE),
    )
    args = parser.parse_args()
    report = replay_recorded_force_through_shared_ik(
        args.log,
        side=args.side,
        max_offset_m=float(args.max_offset_mm) / 1000.0,
        max_speed_m_s=float(args.max_speed_mm_s) / 1000.0,
        max_return_speed_m_s=(
            None
            if args.max_return_speed_mm_s is None
            else float(args.max_return_speed_mm_s) / 1000.0
        ),
        max_output_slew_m_s=(
            None
            if args.max_output_slew_mm_s is None
            else float(args.max_output_slew_mm_s) / 1000.0
        ),
        replay_hz=float(args.replay_hz),
        release_tail_sec=float(args.release_tail_sec),
        left_tool_profile=args.left_tool_profile,
        right_tool_profile=args.right_tool_profile,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
