"""Construction root for the real-robot teleoperation component graph.

This module assembles configured sources, retargeters, backends, safety
components, recording, and optional preview rendering.  It deliberately does
not start the session or any device; lifecycle ordering remains owned by the
canonical real-teleop entrypoint.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import numpy as np

from dataset_contracts.pose_frames import legacy_rm65_source_id
from robot_motion.position_control import (
    DEFAULT_POSITION_DEGRADED_SPEED_SCALE,
    DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
    DEFAULT_POSITION_VELOCITY_RAMP_SEC,
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
    POSITION_COMMAND_PLANNER_LINEAR,
    POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
    normalize_position_command_planner,
)
from robot_motion.rm65 import (
    DEFAULT_LEFT_ARM_QPOS,
    DEFAULT_RIGHT_ARM_QPOS,
    DualArmSceneProfile,
    build_dual_arm_scene_profile,
)
from robot_motion.rm65.assets import require_asset_path
from robot_motion.rm65.ik_tuning import load_ik_tuning_config
from sim.rm65.sapien_renderer import DebugBoxTemplate, SapienArmHandRenderer

from ..action.human_shadow_frontend import (
    RealTeleopHumanShadowProfile,
    build_real_teleop_human_shadow_frontend,
)
from ..backends.arms import (
    RM65ArmBackend,
    RM65ArmBackendConfig,
    RM65ImpedanceArmBackend,
    RM65ImpedanceArmBackendConfig,
)
from ..backends.grippers import DMTacGripperBackend
from ..backends.hands import (
    DEFAULT_INSPIRE_HAND_MAX_FORCE,
    InspireTacHandBackend,
    InspireTacHandBackendConfig,
)
from ..calibration import TeleopCalibration
from ..cli_value_parsers import _active_sides_from_args
from ..configs import (
    build_annotated_dual_rm65_calibration,
    build_bimanual_pose_smoother,
    load_default_pose_smoothing_config,
    load_impedance_pose_smoothing_config,
)
from ..hand_mapping import DEFAULT_INSPIRE_HAND_JOINT_ORDER
from ..hand_render import (
    LEFT_HAND_RENDER_JOINT_NAMES,
    RIGHT_HAND_RENDER_JOINT_NAMES,
)
from ..hand_tcp_alignment import (
    DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG,
    tracker_hand_to_rm65_tcp_rotation,
    vive_tracker_local_translation_offset_from_args as _vive_tracker_local_translation_offset,
)
from ..initial_alignment import TeleopInitialAlignment
from ..log_utils import log_event as _log
from ..position_control import (
    PositionReferencePublisher,
    PositionReferencePublisherConfig,
)
from ..target_safety.tcp_pose_envelope_runtime import (
    install_tcp_pose_envelope_filter,
    load_tcp_pose_envelope_profile,
)
from .position_admittance_shadow import (
    POSITION_ADMITTANCE_ACTIVE_QPOS_FAST_PATH_ALPHA,
    POSITION_ADMITTANCE_ACTIVE_QPOS_MICRO_REVERSAL_CREEP_DEG,
    POSITION_ADMITTANCE_ACTIVE_QPOS_MICRO_REVERSAL_HOLD_DEG,
    PositionAdmittanceEndpointExcursionGuard,
    ResolvedPositionAdmittanceProfile,
    position_admittance_active_uses_home_endpoint_guard,
    resolve_position_admittance_profile,
)
from ..recording import SessionRecorder
from ..retargeters import (
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
    JointsToDexHandRetargeter,
    JointsToDexHandRetargeterConfig,
    PoseToArmRetargeter,
    PoseToArmRetargeterConfig,
    ScalarToGripperRetargeter,
    ScalarToGripperRetargeterConfig,
)
from ..rigs import BimanualRig
from ..safety import TeleopSafetyConfig
from ..session import TeleopSession, TeleopSessionConfig
from ..sources.hand import AngleTriggerSource, HandQ6Source
from ..sources.misc import KeyboardButtonSource
from ..sources.pose import DEFAULT_VIVE_CLOCK_SYNC_PORT, VivePoseSource
from ..special_tasks.fix_arm import FixArmCommandController
from ..special_tasks.init_arm import InitArmCommandController, load_init_arm_targets
from .arm_tool_config import (
    _alignment_qpos_pair,
    _force_tare_sec_arg,
    _impedance_mode_arg,
    _strict_tool_profile_arg,
    _tool_profile_arg,
    _tool_tcp_pose_flange_arg,
)
from .command_filters import (
    _compose_real_command_filter,
    _compose_real_command_guard,
    _compose_real_fault_filter,
    _fix_arm_required_effectors_by_side,
)
from .components import RealTeleopComponents
from .startup_home_continuity import (
    StartupHomeContinuityConfig,
    StartupHomeContinuityGate,
)
from .control_profile import (
    arm_control_mode as _arm_control_mode_arg,
    canfd_follow_runtime_config as _canfd_follow_runtime_config,
    effective_arm_canfd_follow as _effective_arm_canfd_follow,
    effective_arm_control_hz as _effective_arm_control_hz,
    effective_arm_follow_qpos_alpha as _effective_arm_follow_qpos_alpha,
    effective_dual_arm_ik_hz as _effective_dual_arm_ik_hz,
    effective_dual_arm_temporary_safety as _effective_dual_arm_temporary_safety,
    effective_outer_control_hz as _effective_outer_control_hz,
    effective_position_post_ik_qpos_smoothing_alpha as _effective_position_post_ik_qpos_smoothing_alpha,
    output_qpos_smoothing_singularity_alpha as _output_qpos_smoothing_singularity_alpha,
    position_retargeter_stability_switches as _position_retargeter_stability_switches,
)
from .debug_timing import (
    debug_timing_metadata as _debug_timing_metadata,
    debug_timing_modules as _debug_timing_modules,
)
from .defaults import (
    DEFAULT_POSITION_MAX_ACCELERATION_RAD_S2,
    DEFAULT_REAL_TELEOP_MAX_TRACKER_ROTATION_JUMP_DEG,
    DEFAULT_REAL_TELEOP_MAX_TRACKER_TRANSLATION_JUMP_M,
)
from .gripper_policy import (
    resolved_gripper_policies_by_side as _resolved_gripper_policies_by_side,
    resolved_gripper_policy as _resolved_gripper_policy,
)
from .hand_config import (
    HAND_BACKEND_DMTAC_GRIPPER,
    HAND_BACKEND_INSPIRE,
    HAND_BACKEND_NONE,
    dmtac_sides as _dmtac_sides,
    hand_backend_mode as _hand_backend_mode,
    hand_backend_mode_for_side as _hand_backend_mode_for_side,
)
from .hand_runtime import (
    _dmtac_command_contract_configured,
    _parse_hand_tactile_regions,
)
from .movement import (
    movement_multiple_value as _movement_multiple_value,
    resolve_movement_multiple_args as _resolve_movement_multiple_args,
)
from .position_tuning import (
    _impedance_bimanual_ik_safety_profile,
    _position_ik_tuning_for_args,
    _position_tracker_motion_intent_config,
)
from .runtime_metadata import (
    arm_execution_profile_metadata as _arm_execution_profile_metadata,
    position_ik_iteration_metadata as _position_ik_iteration_metadata,
)
from .session_recording import (
    _session_record_root_dir,
    _session_recording_config,
    _session_recording_enabled,
)
from .side_mapping import (
    build_vive_tracker_configs_for_args as _build_vive_tracker_configs_for_args,
    resolve_logical_side_assignments as _resolve_logical_side_assignments,
    tracker_name_for_source_side as _tracker_name_for_source_side,
    tracker_source_side_for_logical_side as _tracker_source_side_for_logical_side,
)
from .temporary_dual_arm_safety import (
    TemporaryDualArmSafetyConfig,
    TemporaryDualArmSafetyLimiter,
)
from .workspace import (
    apply_workspace_config as _apply_workspace_config,
    apply_workspace_overrides as _apply_workspace_overrides,
)


def _collision_debug_box_templates(
    profile: DualArmSceneProfile,
) -> dict[str, DebugBoxTemplate]:
    boxes = {
        spec.name: DebugBoxTemplate(
            size_xyz=spec.box_size_xyz,
            color_rgba=spec.color_rgba,
        )
        for spec in profile.hand_proxy_boxes
    }
    boxes.update(
        {
            spec.name: DebugBoxTemplate(
                size_xyz=spec.box_size_xyz,
                color_rgba=spec.color_rgba,
            )
            for spec in profile.fixed_scene_boxes
        }
    )
    return boxes


def _active_arm_ip_by_side(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> dict[str, tuple[str, str]]:
    """Return only arm-IP assignments admitted by ``active_sides``.

    A shared command may contain configuration for both arms even when this
    runtime is intentionally single-arm. Active-side selection owns
    admission; inactive values remain inert configuration context.
    """

    return {
        logical_side: (source_name, arm_ip)
        for source_name, (logical_side, arm_ip) in _resolve_logical_side_assignments(args).items()
        if logical_side in active_sides and arm_ip
    }


def build_real_teleop_components(
    args: argparse.Namespace,
    *,
    dmtac_backend_configs: Callable[..., dict[str, object]],
) -> RealTeleopComponents:
    _dmtac_backend_configs = dmtac_backend_configs
    position_admittance_profile = getattr(
        args,
        "_resolved_position_admittance_profile",
        None,
    )
    if position_admittance_profile is None:
        position_admittance_profile = resolve_position_admittance_profile(args)
    elif not isinstance(
        position_admittance_profile,
        ResolvedPositionAdmittanceProfile,
    ):
        raise TypeError(
            "_resolved_position_admittance_profile must be a "
            "ResolvedPositionAdmittanceProfile"
        )
    _resolve_movement_multiple_args(args)
    debug_timing_modules = _debug_timing_modules(args)
    active_sides = _active_sides_from_args(args)
    arm_control_mode = _arm_control_mode_arg(args)
    impedance_mode = _impedance_mode_arg(args)
    strict_tool_profile = _strict_tool_profile_arg(args)
    force_tare_sec = _force_tare_sec_arg(args)
    tcp_pose_envelope_profile_path = str(
        getattr(args, "tcp_pose_envelope_profile", "") or ""
    ).strip()
    tcp_pose_envelope_active = bool(
        getattr(args, "tcp_pose_envelope_active", False)
    )
    if tcp_pose_envelope_active and not tcp_pose_envelope_profile_path:
        raise ValueError(
            "--tcp-pose-envelope-active requires "
            "--tcp-pose-envelope-profile"
        )
    # ``active_sides`` is the authority for this runtime.  The command may
    # intentionally carry both saved/configured arm IPs while operating one
    # arm (for example, a shared command template or a later side switch).
    # Inactive-side IPs are configuration context and must not make component
    # construction fail; only active sides need a usable IP.
    logical_ip_by_side = _active_arm_ip_by_side(args, active_sides)
    missing_arm_sides = [side for side in active_sides if side not in logical_ip_by_side]
    if missing_arm_sides:
        raise ValueError(
            "Real teleop requires an RM65 IP for every active logical side; "
            f"missing={missing_arm_sides} active_sides={list(active_sides)} "
            f"swap_arm_sides={bool(args.swap_arm_sides)}."
        )
    for side in ("left", "right"):
        if side in active_sides:
            continue
        present = [
            f"--{side}-hand-ip"
            if str(getattr(args, f"{side}_hand_ip", "") or "")
            else "",
            f"--{side}-hand-port"
            if str(getattr(args, f"{side}_hand_port", "") or "")
            else "",
            f"--{side}-gripper-port"
            if str(getattr(args, f"{side}_gripper_port", "") or "")
            else "",
        ]
        present = [value for value in present if value]
        if present:
            raise ValueError(
                f"Inactive side '{side}' received hand configuration {present}; "
                "remove those flags or include the side in --active-sides."
            )
    outer_control_hz = _effective_outer_control_hz(args)
    action_dt = 1.0 / outer_control_hz
    arm_control_hz = _effective_arm_control_hz(args)
    arm_control_dt = 1.0 / arm_control_hz
    arm_canfd_follow = _effective_arm_canfd_follow(args)
    impedance_overrides: dict[str, object] = {}
    impedance_overrides["canfd_follow"] = bool(arm_canfd_follow)
    if getattr(args, "arm_canfd_follow_stream_hz", None) is not None:
        impedance_overrides["canfd_follow_stream_hz"] = float(args.arm_canfd_follow_stream_hz)
    if getattr(args, "arm_canfd_follow_warning_ratio", None) is not None:
        impedance_overrides["canfd_follow_warning_ratio"] = float(args.arm_canfd_follow_warning_ratio)
    if getattr(args, "arm_canfd_trajectory_mode", None) is not None:
        impedance_overrides["trajectory_mode"] = int(args.arm_canfd_trajectory_mode)
    if getattr(args, "arm_canfd_radio", None) is not None:
        impedance_overrides["radio"] = int(args.arm_canfd_radio)
    if getattr(args, "arm_state_read_every", None) is not None:
        impedance_overrides["state_read_every"] = int(args.arm_state_read_every)
    if bool(getattr(args, "arm_disable_force_admittance", False)):
        impedance_overrides.update(
            {
                "passthrough_position_control": True,
                "linear_gain_m_s_per_n": 0.0,
                "angular_gain_rad_s_per_nm": 0.0,
                "linear_damping_n_s_per_m": 0.0,
                "angular_damping_nm_s_per_rad": 0.0,
                "linear_compliance_m_per_n": 0.0,
                "angular_compliance_rad_per_nm": 0.0,
                "max_linear_offset_m": 0.0,
                "max_angular_offset_rad": 0.0,
            }
        )
    canfd_runtime_config = _canfd_follow_runtime_config(
        args,
        arm_control_hz=arm_control_hz,
        arm_canfd_follow=arm_canfd_follow,
    )
    position_background_streaming_enabled = bool(
        canfd_runtime_config["position_background_streaming_enabled"]
    )
    position_background_streaming_hz = float(
        canfd_runtime_config["position_background_streaming_hz"]
    )
    canfd_follow_warning_ratio = float(
        canfd_runtime_config["canfd_follow_warning_ratio"]
    )
    canfd_trajectory_mode = int(canfd_runtime_config["canfd_trajectory_mode"])
    canfd_radio = int(canfd_runtime_config["canfd_radio"])
    position_canfd_passthrough = bool(
        arm_control_mode == "position" and position_background_streaming_enabled
    )
    position_command_planner = normalize_position_command_planner(
        getattr(args, "position_command_planner", POSITION_COMMAND_PLANNER_LINEAR)
    )
    if (
        position_command_planner
        in {
            POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
            POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
        }
        and arm_control_mode != "position"
    ):
        raise ValueError(
            "the selected limited position-command planner requires position arm control mode."
        )
    position_max_command_step_deg = getattr(
        args,
        "position_max_command_step_deg",
        None,
    )
    if position_max_command_step_deg is not None and arm_control_mode != "position":
        raise ValueError(
            "--position-max-command-step-deg is a position-mode option and cannot alter "
            "force/impedance control."
        )
    if (
        position_max_command_step_deg is not None
        and position_command_planner
        not in {
            POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
            POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
        }
    ):
        raise ValueError(
            "--position-max-command-step-deg requires an acceleration/jerk-limited "
            "position command planner."
        )
    if (
        position_command_planner
        in {
            POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
            POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
        }
        and not position_background_streaming_enabled
    ):
        raise ValueError(
            "the selected limited position-command planner requires the high-rate position "
            "background stream; enable the normal position CANFD follow path."
        )
    if (
        position_command_planner
        in {
            POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
            POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
        }
        and position_background_streaming_hz < outer_control_hz
    ):
        raise ValueError(
            "Acceleration-limited position control requires command_hz >= reference_hz; "
            f"got command_hz={position_background_streaming_hz:g}, "
            f"reference_hz={outer_control_hz:g}."
        )
    safe_debug_enabled = bool(getattr(args, "safe_debug", False))
    position_stability = _position_retargeter_stability_switches(
        args,
        position_canfd_passthrough=position_canfd_passthrough,
    )
    safe_debug_disable_hold = bool(position_stability["hold_disabled"])
    position_post_ik_qpos_smoothing_alpha = (
        _effective_position_post_ik_qpos_smoothing_alpha(args)
    )
    arm_follow_qpos_alpha = _effective_arm_follow_qpos_alpha(args)
    strict_degree_limit_enabled = bool(getattr(args, "strict_degree_limit", False))
    qpos_limit_scale = 0.2 if strict_degree_limit_enabled else 1.0
    arm_io_threads = max(1, int(args.arm_io_threads))
    collision_guard_requested = (
        set(active_sides) == {"left", "right"}
        and not args.disable_dual_arm_collision_avoidance
    )
    temporary_safety_effective = _effective_dual_arm_temporary_safety(
        args,
        position_canfd_passthrough=position_canfd_passthrough,
    )
    temporary_safety_enabled = bool(
        temporary_safety_effective["enabled"]
        and set(active_sides) == {"left", "right"}
    )
    if (
        not temporary_safety_enabled
        and temporary_safety_effective["enabled"]
        and set(active_sides) != {"left", "right"}
    ):
        temporary_safety_effective["disabled_reason"] = "requires_both_active_sides"
    temporary_safety_limiter = (
        TemporaryDualArmSafetyLimiter(
            TemporaryDualArmSafetyConfig(
                enabled=True,
                active_sides=tuple(active_sides),
                max_xyz_step_m=float(args.dual_arm_temp_max_xyz_step_m),
                max_rot_step_deg=float(args.dual_arm_temp_max_rot_step_deg),
                max_joint_step_deg=float(args.dual_arm_temp_max_joint_step_deg)
                * qpos_limit_scale,
                command_scale=float(args.dual_arm_temp_command_scale),
                evidence_enabled=(
                    str(getattr(args, "human_action_arbiter_mode", "off"))
                    == "shadow"
                ),
            )
        )
        if temporary_safety_enabled
        else None
    )
    arm_max_angvel = np.asarray(
        getattr(args, "arm_max_angvel", (0.35, 0.35, 0.35, 0.5, 0.5, 0.5)),
        dtype=float,
    ).reshape(6) * qpos_limit_scale
    movement_multiple = _movement_multiple_value(args)
    realistic_joint_step = tuple((arm_max_angvel * action_dt).tolist())
    collision_scene_profile = build_dual_arm_scene_profile(args.collision_scene_profile)
    ik_tuning = _position_ik_tuning_for_args(
        args,
        load_ik_tuning_config(args.ik_tuning_json or None),
        arm_control_mode=arm_control_mode,
    )
    impedance_bimanual_ik_safety_profile = (
        _impedance_bimanual_ik_safety_profile()
        if arm_control_mode == "impedance" and set(active_sides) == {"left", "right"}
        else None
    )
    target_pose_smoothing_config = (
        load_impedance_pose_smoothing_config()
        if arm_control_mode == "impedance"
        else load_default_pose_smoothing_config()
    )
    if safe_debug_disable_hold:
        target_pose_smoothing_config = replace(
            target_pose_smoothing_config,
            deadband_translation_m=0.0,
            deadband_rotation_deg=0.0,
        )
    target_pose_smoother = (
        build_bimanual_pose_smoother(target_pose_smoothing_config)
        if position_stability["target_pose_smoothing_enabled"]
        else None
    )
    target_pose_smoothing_metadata = (
        {
            "enabled": False,
            "reason": "disabled_by_runtime_config",
            "safe_debug": bool(safe_debug_enabled),
            "base_config": target_pose_smoothing_config.to_metadata(),
        }
        if target_pose_smoother is None
        else target_pose_smoothing_config.to_metadata()
    )
    if position_canfd_passthrough:
        target_pose_smoothing_metadata = {
            **target_pose_smoothing_metadata,
            "safe_debug": bool(safe_debug_enabled),
            "restored_for_position_canfd_passthrough": True,
            "disable_hold": bool(safe_debug_disable_hold),
        }
    current_target_hold_enabled = not safe_debug_disable_hold
    local_step_candidate_selection_enabled = bool(
        position_stability["local_step_candidate_selection_enabled"]
    )
    inactive_side_hold_enabled = bool(
        position_stability["inactive_side_hold_enabled"]
    )
    inactive_side_hold_execution_cap_enabled = bool(
        position_stability["inactive_side_hold_execution_cap_enabled"]
    )
    loaded_initial_alignment = (
        TeleopInitialAlignment.load_json(args.load_initial_alignment)
        if args.load_initial_alignment
        else None
    )
    if args.load_profile:
        calibration = TeleopCalibration.load_json(args.load_profile)
    else:
        calibration = build_annotated_dual_rm65_calibration(
            left_right_base_distance=args.left_right_base_distance
        )
    if impedance_bimanual_ik_safety_profile is not None:
        calibration.metadata["impedance_bimanual_local_step_bounds"] = dict(
            impedance_bimanual_ik_safety_profile.metadata.get(
                "impedance_bimanual_local_step_bounds",
                {},
            )
        )
    workspace_config = _apply_workspace_config(args, calibration)
    workspace_overrides = _apply_workspace_overrides(args, calibration)
    if workspace_config:
        _log(
            "Workspace",
            f"workspace config applied: {json.dumps(workspace_config, sort_keys=True)}",
        )
    if workspace_overrides:
        _log(
            "Workspace",
            f"runtime workspace override applied: {json.dumps(workspace_overrides, sort_keys=True)}",
        )
    if _dmtac_sides(args, active_sides):
        try:
            _dmtac_backend_configs(args, active_sides)
        except (OSError, RuntimeError, ValueError) as exc:
            if str(getattr(args, "hardware_admission_mode", "strict")) == "strict":
                raise
            args.dmtac_startup_error = f"{type(exc).__name__}: {exc}"
            args._dmtac_backend_configs = {}
            args.dmtac_config_loaded_sides = []
            _log(
                "DMTAC",
                "WARNING: DMTAC initialization was isolated from arm/GUI startup: "
                f"{args.dmtac_startup_error}",
            )
    tracker_translation_offset = _vive_tracker_local_translation_offset(args)
    calibration.metadata.update(
        {
            "source_entrypoint": "teleop_runtime.debug.run_vive_rm65_real_teleop",
            **_arm_execution_profile_metadata(args),
            **_position_ik_iteration_metadata(args),
            "left_tracker_name": args.left_tracker_name,
            "right_tracker_name": args.right_tracker_name,
            "aux_tracker_name": args.aux_tracker_name,
            "active_sides": list(active_sides),
            "arm_mode": "real",
            "movement_multiple": movement_multiple,
            "translation_scale": movement_multiple,
            "left_arm_ip": args.left_arm_ip,
            "right_arm_ip": args.right_arm_ip,
            "left_hand_ip": args.left_hand_ip,
            "right_hand_ip": args.right_hand_ip,
            "left_hand_port": getattr(args, "left_hand_port", ""),
            "right_hand_port": getattr(args, "right_hand_port", ""),
            "hand_backend": _hand_backend_mode(args),
            "hand_backend_by_side": {
                side: _hand_backend_mode_for_side(args, side) for side in active_sides
            },
            "session_record_commands": not bool(getattr(args, "no_session_record_commands", False)),
            "session_record_flush_every": int(getattr(args, "session_record_flush_every", 1) or 1),
            "session_record_compact_json": bool(getattr(args, "session_record_compact_json", False)),
            "q6_config": str(getattr(args, "q6_config", "latest")),
            "q6_config_loaded_ports": dict(getattr(args, "q6_config_loaded_ports", {}) or {}),
            "dmtac_config": str(getattr(args, "dmtac_config", "latest")),
            "dmtac_default_config_enabled": bool(getattr(args, "dmtac_default_config", True)),
            "dmtac_config_loaded_sides": list(getattr(args, "dmtac_config_loaded_sides", []) or []),
            "dmtac_tactile_fields": list(getattr(args, "dmtac_tactile_fields", None) or []),
            "dmtac_tactile_compute_profile": str(
                getattr(args, "dmtac_tactile_compute_profile", None) or "dist-only"
            ),
            "dmtac_tactile_parallel_read": getattr(args, "dmtac_tactile_parallel_read", None),
            "dmtac_tactile_global_read_lock": getattr(args, "dmtac_tactile_global_read_lock", None),
            "dmtac_tactile_read_hz": getattr(args, "dmtac_tactile_read_hz", None),
            "dmtac_tactile_process_read": getattr(args, "dmtac_tactile_process_read", None),
            "dmtac_tactile_worker_cpus": list(getattr(args, "dmtac_tactile_worker_cpus", None) or []),
            "dmtac_tactile_worker_nice": getattr(args, "dmtac_tactile_worker_nice", None),
            "dmtac_tactile_worker_queue_size": getattr(args, "dmtac_tactile_worker_queue_size", None),
            "dmtac_tactile_max_fps": getattr(args, "dmtac_tactile_max_fps", None),
            "dmtac_tactile_warmup_frames": getattr(args, "dmtac_tactile_warmup_frames", None),
            "dmtac_tactile_denoise": getattr(args, "dmtac_tactile_denoise", None),
            "dmtac_tactile_flat_vector": getattr(args, "dmtac_tactile_flat_vector", None),
            "gripper_task_profile": str(getattr(args, "gripper_task_profile", "") or "") or None,
            "gripper_task": str(getattr(args, "gripper_task", "") or "") or None,
            "gripper_policy": (
                dict(getattr(args, "_resolved_gripper_policy", {}) or _resolved_gripper_policy(args))
                if _dmtac_sides(args, active_sides)
                else {}
            ),
            "gripper_policy_by_side": (
                _resolved_gripper_policies_by_side(args, active_sides)
                if _dmtac_sides(args, active_sides)
                else {}
            ),
            "left_logical_tracker_name": _tracker_name_for_source_side(
                args,
                _tracker_source_side_for_logical_side(args, "left"),
            ),
            "right_logical_tracker_name": _tracker_name_for_source_side(
                args,
                _tracker_source_side_for_logical_side(args, "right"),
            ),
            "hand_reader_baudrate": int(getattr(args, "hand_reader_baudrate", 115200)),
            "exo_hand_calibration": bool(getattr(args, "exo_hand_calibration", False)),
            "exo_hand_calibration_path": getattr(args, "exo_hand_calibration_path", "") or None,
            "hand_joint_order": list(getattr(args, "hand_joint_order", DEFAULT_INSPIRE_HAND_JOINT_ORDER)),
            "hand_invert": bool(getattr(args, "hand_invert", True)),
            "hand_command_alpha": float(getattr(args, "hand_command_alpha", 1.0)),
            "hand_send_threshold": float(getattr(args, "hand_send_threshold", 0.002)),
            "hand_telemetry_hz": float(getattr(args, "hand_telemetry_hz", 30.0)),
            "hand_tactile_regions": list(_parse_hand_tactile_regions(getattr(args, "hand_tactile_regions", "all"))),
            "max_fingertip_force": int(
                getattr(args, "max_fingertip_force", DEFAULT_INSPIRE_HAND_MAX_FORCE)
            ),
            "teleop_hardware_ip": args.teleop_hardware_ip,
            "teleop_hardware_port": int(args.teleop_hardware_port),
            "swap_arm_sides": bool(args.swap_arm_sides),
            "swap_operator_sides": bool(getattr(args, "swap_operator_sides", False)),
            "tracker_input_mapping": {
                side: _tracker_source_side_for_logical_side(args, side)
                for side in active_sides
            },
            "action_dt": action_dt,
            "arm_control_hz": arm_control_hz,
            "arm_control_dt": arm_control_dt,
            "arm_canfd_follow": bool(arm_canfd_follow),
            "arm_canfd_follow_arg": getattr(args, "arm_canfd_follow", None),
            "position_post_ik_qpos_smoothing_alpha": getattr(
                args,
                "position_post_ik_qpos_smoothing_alpha",
                None,
            ),
            "position_post_ik_qpos_smoothing_alpha_effective": float(
                position_post_ik_qpos_smoothing_alpha
            ),
            "arm_follow_qpos_alpha": getattr(args, "arm_follow_qpos_alpha", None),
            "arm_follow_qpos_alpha_effective": float(arm_follow_qpos_alpha),
            "arm_canfd_follow_stream_hz": getattr(args, "arm_canfd_follow_stream_hz", None),
            "arm_canfd_follow_warning_ratio": getattr(args, "arm_canfd_follow_warning_ratio", None),
            "position_background_streaming_enabled": bool(position_background_streaming_enabled),
            "position_background_streaming_hz": float(position_background_streaming_hz),
            "position_command_planner": position_command_planner,
            "position_velocity_ramp_sec": float(
                getattr(
                    args,
                    "position_velocity_ramp_sec",
                    DEFAULT_POSITION_VELOCITY_RAMP_SEC,
                )
            ),
            "position_target_velocity_filter_tau_sec": float(
                getattr(
                    args,
                    "position_target_velocity_filter_tau_sec",
                    DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
                )
            ),
            "position_max_acceleration_rad_s2": [
                float(value)
                for value in np.asarray(
                    getattr(
                        args,
                        "position_max_acceleration_rad_s2",
                        DEFAULT_POSITION_MAX_ACCELERATION_RAD_S2,
                    ),
                    dtype=float,
                ).reshape(6).tolist()
            ],
            "position_max_command_step_deg": (
                None
                if getattr(args, "position_max_command_step_deg", None) is None
                else float(args.position_max_command_step_deg)
            ),
            "position_degraded_speed_scale": float(
                getattr(
                    args,
                    "position_degraded_speed_scale",
                    DEFAULT_POSITION_DEGRADED_SPEED_SCALE,
                )
            ),
            "position_sample_aligned_scale_degraded_target": bool(
                getattr(
                    args,
                    "position_sample_aligned_scale_degraded_target",
                    True,
                )
            ),
            "position_sample_aligned_source_latch_mode": str(
                getattr(
                    args,
                    "position_sample_aligned_source_latch_mode",
                    "time_bucket",
                )
            ),
            "position_sample_aligned_governor_regularization": {
                "jerk": float(
                    getattr(
                        args,
                        "position_sample_aligned_jerk_regularization",
                        1e-9,
                    )
                ),
                "jerk_delta": float(
                    getattr(
                        args,
                        "position_sample_aligned_jerk_delta_regularization",
                        0.0,
                    )
                ),
                "initial_jerk_delta": float(
                    getattr(
                        args,
                        "position_sample_aligned_initial_jerk_delta_regularization",
                        0.0,
                    )
                ),
                "terminal_velocity": float(
                    getattr(
                        args,
                        "position_sample_aligned_terminal_velocity_regularization",
                        0.0,
                    )
                ),
                "terminal_acceleration": float(
                    getattr(
                        args,
                        "position_sample_aligned_terminal_acceleration_regularization",
                        0.0,
                    )
                ),
                "reference_mode": str(
                    getattr(
                        args,
                        "position_sample_aligned_reference_mode",
                        "linear_hold",
                    )
                ),
                "observer_window": int(
                    getattr(
                        args,
                        "position_sample_aligned_observer_window",
                        4,
                    )
                ),
                "observer_stop_threshold_deg": float(
                    getattr(
                        args,
                        "position_sample_aligned_observer_stop_threshold_deg",
                        0.01,
                    )
                ),
                "observer_stop_velocity_decay": float(
                    getattr(
                        args,
                        "position_sample_aligned_observer_stop_velocity_decay",
                        0.0,
                    )
                ),
            },
            "canfd_follow_warning_ratio_effective": float(canfd_follow_warning_ratio),
            "arm_canfd_trajectory_mode": getattr(args, "arm_canfd_trajectory_mode", None),
            "canfd_trajectory_mode_effective": int(canfd_trajectory_mode),
            "arm_canfd_radio": getattr(args, "arm_canfd_radio", None),
            "canfd_radio_effective": int(canfd_radio),
            "record_actual_wrist_camera_pose": bool(
                getattr(args, "record_actual_wrist_camera_pose", True)
            ),
            "actual_wrist_camera_pose_source": "rm65_feedback_qpos_fk",
            "actual_wrist_camera_pose_hz": float(
                getattr(args, "actual_wrist_camera_pose_hz", 30.0)
            ),
            "actual_wrist_camera_pose_max_age_sec": float(
                getattr(args, "actual_wrist_camera_pose_max_age_sec", 0.2)
            ),
            "arm_control_mode": arm_control_mode,
            "impedance_mode": impedance_mode,
            "strict_tool_profile": strict_tool_profile,
            "auto_sync_tool_frame": bool(getattr(args, "auto_sync_tool_frame", True)),
            "force_tare_sec": force_tare_sec,
            "left_tool_profile": _tool_profile_arg(args, "left"),
            "right_tool_profile": _tool_profile_arg(args, "right"),
            "arm_io_threads": int(arm_io_threads),
            "arm_step_dt_scaling": bool(args.arm_step_dt_scaling),
            "arm_step_dt_alpha": float(args.arm_step_dt_alpha),
            "arm_step_dt_max_scale": float(args.arm_step_dt_max_scale),
            "arm_step_dt_stall_reset_sec": float(args.arm_step_dt_stall_reset_sec),
            "arm_singularity_protection": bool(
                getattr(args, "arm_singularity_protection", True)
            ),
            **_debug_timing_metadata(args),
            "rm65_ik_framework": str(getattr(args, "rm65_ik_framework", "robot_motion")),
            "dual_arm_collision_avoidance_requested": collision_guard_requested,
            "dual_arm_temporary_safety_enabled": bool(temporary_safety_limiter is not None),
            "dual_arm_temporary_safety_requested": bool(
                temporary_safety_effective["requested"]
            ),
            "dual_arm_temporary_safety_disabled_reason": temporary_safety_effective[
                "disabled_reason"
            ],
            "dual_arm_temporary_safety": {
                "max_xyz_step_m": float(getattr(args, "dual_arm_temp_max_xyz_step_m", 0.03)),
                "max_rot_step_deg": float(getattr(args, "dual_arm_temp_max_rot_step_deg", 8.0)),
                "max_joint_step_deg": float(getattr(args, "dual_arm_temp_max_joint_step_deg", 6.0)),
                "max_joint_step_effective_deg": float(
                    getattr(args, "dual_arm_temp_max_joint_step_deg", 6.0)
                )
                * qpos_limit_scale,
                "command_scale": float(getattr(args, "dual_arm_temp_command_scale", 1.0)),
            },
            "arm_max_angvel_rad_s": [
                float(value)
                for value in np.asarray(args.arm_max_angvel, dtype=float).reshape(-1).tolist()
            ],
            "arm_max_angvel_effective_rad_s": [
                float(value) for value in arm_max_angvel.reshape(-1).tolist()
            ],
            "position_canfd_passthrough": bool(position_canfd_passthrough),
            "safe_debug": bool(safe_debug_enabled),
            "disable_hold": bool(getattr(args, "disable_hold", False)),
            "safe_debug_disable_hold": bool(safe_debug_disable_hold),
            "strict_degree_limit": bool(strict_degree_limit_enabled),
            "qpos_limit_scale": float(qpos_limit_scale),
            "position_canfd_passthrough_safety_restored": bool(
                position_canfd_passthrough
                and position_stability["target_pose_smoothing_enabled"]
                and position_stability["local_step_candidate_selection_enabled"]
            ),
            "current_target_hold_enabled": bool(current_target_hold_enabled),
            "local_step_candidate_selection_enabled": bool(
                local_step_candidate_selection_enabled
            ),
            "inactive_side_hold_enabled": bool(inactive_side_hold_enabled),
            "inactive_side_hold_execution_cap_enabled": bool(
                inactive_side_hold_execution_cap_enabled
            ),
            "target_pose_smoothing": target_pose_smoothing_metadata,
            "collision_avoidance_mode": args.collision_avoidance_mode,
            "collision_scene_profile": args.collision_scene_profile,
            "ik_tuning_source": ik_tuning.source_path,
            "ik_objective_mode": ik_tuning.ik_objective_mode,
            "ik_objective_weights": asdict(ik_tuning.objective_weights()),
            "position_ik_branch_continuity_anchor": {
                "enabled": bool(
                    arm_control_mode == "position"
                    and bool(
                        getattr(
                            args,
                            "position_ik_branch_continuity_anchor",
                            False,
                        )
                    )
                    and str(
                        getattr(
                            args,
                            "position_ik_step_mode",
                            "legacy_iterative",
                        )
                    )
                    == "single_step_singularity_aware"
                ),
                "cost": 0.5,
                "joint_weights_per_arm": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                "anchor": "previous_admitted_qref",
            },
            "position_joint_limit_guard": {
                "mode": str(
                    getattr(
                        args,
                        "position_joint_limit_guard_mode",
                        "legacy",
                    )
                ),
                "boundary_margin_deg": float(
                    getattr(
                        args,
                        "position_joint_limit_boundary_margin_deg",
                        1.5,
                    )
                ),
            },
            "position_ik_orientation_weight_override": getattr(
                args,
                "position_ik_orientation_weight",
                None,
            ),
            "initial_alignment_source": args.load_initial_alignment,
            "vive_tracker_height_correction_enabled": tracker_translation_offset is not None,
            "vive_tracker_height_correction_m": float(
                getattr(
                    args,
                    "vive_tracker_height_correction_m",
                    DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG.height_correction_m,
                )
            ),
            "vive_tracker_local_translation_offset_m": None
            if tracker_translation_offset is None
            else tracker_translation_offset.tolist(),
        }
    )

    keyboard_source = KeyboardButtonSource(name="terminal_clutch")
    default_clutch_pressed = False
    keyboard_source.set_button(
        "left_clutch_button",
        default_clutch_pressed,
        side="left",
        critical=False,
    )
    keyboard_source.set_button(
        "right_clutch_button",
        default_clutch_pressed,
        side="right",
        critical=False,
    )

    tracker_configs = _build_vive_tracker_configs_for_args(args, active_sides)

    tracker_motion_intent_config = _position_tracker_motion_intent_config(
        args,
        arm_control_mode=arm_control_mode,
    )
    left_retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="left",
            input_channel_name="left_wrist_pose",
            effector_name="left_arm",
            control_mode=args.control_mode,
            translation_scale=movement_multiple,
            input_local_translation_offset=tracker_translation_offset,
            input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
            tracker_motion_intent_enabled=arm_control_mode == "position",
            tracker_motion_intent_config=tracker_motion_intent_config,
        )
    )
    right_retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="right",
            input_channel_name="right_wrist_pose",
            effector_name="right_arm",
            control_mode=args.control_mode,
            translation_scale=movement_multiple,
            input_local_translation_offset=tracker_translation_offset,
            input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
            tracker_motion_intent_enabled=arm_control_mode == "position",
            tracker_motion_intent_config=tracker_motion_intent_config,
        )
    )
    calibration.metadata["tracker_motion_intent"] = {
        "enabled": arm_control_mode == "position",
        "activation_scope": "position_control_only",
        "disabled_reason": (
            None
            if arm_control_mode == "position"
            else "impedance_control_preserves_legacy_inactive_behavior"
        ),
        "window_sec": 0.1,
        "speed_statistic": "p90",
        "inactive_enter_threshold_scale": float(
            getattr(args, "position_inactive_enter_threshold_scale", 1.0)
        ),
        "config": left_retargeter.config.tracker_motion_intent_config.to_metadata(),
        "debug_path": "target_activity.<side>",
    }

    left_seed_qpos = DEFAULT_LEFT_ARM_QPOS
    right_seed_qpos = DEFAULT_RIGHT_ARM_QPOS
    alignment_qpos_pair = _alignment_qpos_pair(loaded_initial_alignment)
    if alignment_qpos_pair is not None:
        left_seed_qpos = alignment_qpos_pair["left"]
        right_seed_qpos = alignment_qpos_pair["right"]

    dual_arm_retargeter = None
    collision_guard_setup_error = None
    retargeters = []
    if collision_guard_requested:
        try:
            debug_ik_profile = bool(debug_timing_modules.ik_profile)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="no QP solver found on your system.*",
                    category=UserWarning,
                )
                dual_arm_retargeter = DualArmCollisionAwareRetargeter(
                    DualArmCollisionAwareRetargeterConfig(
                        control_mode=args.control_mode,
                        translation_scale=movement_multiple,
                        scene_profile=collision_scene_profile.name,
                        contact_mode=args.collision_avoidance_mode,
                        solver_frequency=(
                            _effective_dual_arm_ik_hz(args)
                            if str(getattr(args, "arm_mode", "real")) == "real"
                            else outer_control_hz
                        ),
                        enforce_realistic_joint_step=bool(
                            str(getattr(args, "arm_mode", "real")) == "real"
                        ),
                        ik_framework=str(getattr(args, "rm65_ik_framework", "robot_motion")),
                        left_execution_max_joint_step_rad=realistic_joint_step,
                        right_execution_max_joint_step_rad=realistic_joint_step,
                        left_tool_tcp_pose_flange_m_rad=_tool_tcp_pose_flange_arg(args, "left"),
                        right_tool_tcp_pose_flange_m_rad=_tool_tcp_pose_flange_arg(args, "right"),
                        left_initial_qpos=left_seed_qpos,
                        right_initial_qpos=right_seed_qpos,
                        ik_tuning=ik_tuning,
                        ik_safety_profile=impedance_bimanual_ik_safety_profile,
                        joint_velocity_limit_profile=(
                            str(getattr(args, "position_joint_velocity_limit_profile", "rm65"))
                            if arm_control_mode == "position"
                            else "rm65"
                        ),
                        debug_profile=debug_ik_profile,
                        debug_profile_print_every=(
                            int(getattr(args, "debug_ik_timing_print_every", 0))
                            if debug_ik_profile
                            else 0
                        ),
                        solve_time_budget_ms=getattr(args, "ik_time_budget_ms", None),
                        solve_max_iterations=(
                            int(getattr(args, "position_ik_solve_max_iterations", 2))
                            if arm_control_mode == "position"
                            else 1
                        ),
                        solve_min_iterations=(
                            int(getattr(args, "position_ik_solve_min_iterations", 2))
                            if arm_control_mode == "position"
                            else 1
                        ),
                        solve_step_mode=(
                            str(
                                getattr(
                                    args,
                                    "position_ik_step_mode",
                                    "legacy_iterative",
                                )
                            )
                            if arm_control_mode == "position"
                            else "legacy_iterative"
                        ),
                        # The single-step position path must keep the raw QP
                        # on the last admitted RM65 structural branch.  This
                        # is a soft q3/q5 objective only; candidate safety,
                        # collision checks, execution limits, and dispatch
                        # ownership remain unchanged.
                        branch_continuity_anchor_enabled=bool(
                            arm_control_mode == "position"
                            and bool(
                                getattr(
                                    args,
                                    "position_ik_branch_continuity_anchor",
                                    False,
                                )
                            )
                            and str(
                                getattr(
                                    args,
                                    "position_ik_step_mode",
                                    "legacy_iterative",
                                )
                            )
                            == "single_step_singularity_aware"
                        ),
                        branch_continuity_anchor_cost=0.5,
                        joint_limit_guard_mode=(
                            str(
                                getattr(
                                    args,
                                    "position_joint_limit_guard_mode",
                                    "legacy",
                                )
                            )
                            if arm_control_mode == "position"
                            else "legacy"
                        ),
                        joint_limit_boundary_guard_margin_deg=float(
                            getattr(
                                args,
                                "position_joint_limit_boundary_margin_deg",
                                1.5,
                            )
                        ),
                        joint_acceleration_limit_rad_s2=(
                            np.asarray(
                                getattr(
                                    args,
                                    "position_max_acceleration_rad_s2",
                                ),
                                dtype=float,
                            ).reshape(6)
                            if (
                                arm_control_mode == "position"
                                and str(
                                    getattr(
                                        args,
                                        "position_ik_step_mode",
                                        "legacy_iterative",
                                    )
                                )
                                == "single_step_singularity_aware"
                            )
                            else None
                        ),
                        admitted_reference_motion_limit_enabled=bool(
                            arm_control_mode == "position"
                            and getattr(
                                args,
                                "position_ik_admitted_motion_limit",
                                False,
                            )
                        ),
                        damping=(
                            float(getattr(args, "position_ik_damping", 0.03))
                            if arm_control_mode == "position"
                            else 0.01
                        ),
                        solve_adaptive_iterations_enabled=(
                            bool(getattr(args, "position_ik_adaptive_iterations", False))
                            if arm_control_mode == "position"
                            else False
                        ),
                        solve_iteration_reserve_ms=(
                            float(getattr(args, "position_ik_iteration_reserve_ms", 5.0))
                            if arm_control_mode == "position"
                            else 5.0
                        ),
                        candidate_certification_reserve_ms=(
                            float(
                                getattr(
                                    args,
                                    "position_ik_candidate_certification_reserve_ms",
                                    0.0,
                                )
                            )
                            if arm_control_mode == "position"
                            else 0.0
                        ),
                        position_singularity_candidate_certificate_enabled=bool(
                            arm_control_mode == "position"
                            and getattr(
                                args,
                                "position_singularity_candidate_certificate",
                                False,
                            )
                        ),
                        output_command_type=(
                            "cartesian_pose"
                            if arm_control_mode == "impedance"
                            else "joint_vector"
                        ),
                        target_pose_smoother=target_pose_smoother,
                        output_qpos_smoothing_alpha=float(
                            position_post_ik_qpos_smoothing_alpha
                        ),
                        output_qpos_smoothing_adaptive_enabled=bool(
                            arm_control_mode == "position"
                            and getattr(
                                args,
                                "position_post_ik_qpos_smoothing_adaptive",
                                False,
                            )
                        ),
                        output_qpos_smoothing_fast_alpha=float(
                            getattr(
                                args,
                                "position_post_ik_qpos_smoothing_fast_alpha",
                                0.45,
                            )
                        ),
                        output_qpos_normal_fast_path_enabled=bool(
                            arm_control_mode == "position"
                            and getattr(
                                args,
                                "position_post_ik_qpos_normal_fast_path",
                                False,
                            )
                        ),
                        output_qpos_normal_fast_path_alpha=float(
                            getattr(
                                args,
                                "position_post_ik_qpos_normal_fast_path_alpha",
                                1.0,
                            )
                        ),
                        output_qpos_normal_fast_path_guard_deg=float(
                            getattr(
                                args,
                                "position_post_ik_qpos_normal_fast_path_guard_deg",
                                4.0,
                            )
                        ),
                        output_qpos_normal_fast_path_quality_grace_sec=float(
                            getattr(
                                args,
                                "position_post_ik_qpos_normal_fast_path_quality_grace_sec",
                                0.0,
                            )
                        ),
                        output_qpos_admittance_fast_path_enabled=bool(
                            arm_control_mode == "position"
                            and getattr(
                                args,
                                "position_admittance_active",
                                False,
                            )
                        ),
                        output_qpos_admittance_fast_path_alpha=float(
                            POSITION_ADMITTANCE_ACTIVE_QPOS_FAST_PATH_ALPHA
                        ),
                        output_qpos_admittance_micro_reversal_hold_deg=float(
                            POSITION_ADMITTANCE_ACTIVE_QPOS_MICRO_REVERSAL_HOLD_DEG
                        ),
                        output_qpos_admittance_micro_reversal_creep_deg=float(
                            POSITION_ADMITTANCE_ACTIVE_QPOS_MICRO_REVERSAL_CREEP_DEG
                        ),
                        # A singular wrist uses a bounded, responsive endpoint
                        # only inside the narrow q5 escape band.  This avoids
                        # the normal 0.14 alpha making the guarded wrist feel
                        # latched, while the existing per-cycle joint step and
                        # acceleration limits remain authoritative.
                        output_qpos_smoothing_singularity_alpha=(
                            _output_qpos_smoothing_singularity_alpha(
                                arm_control_mode,
                                slow_alpha=position_post_ik_qpos_smoothing_alpha,
                                fast_alpha=float(
                                    getattr(
                                        args,
                                        "position_post_ik_qpos_smoothing_fast_alpha",
                                        0.45,
                                    )
                                ),
                            )
                        ),
                        # Force control keeps an interior envelope so an
                        # otherwise valid tracking qpos cannot settle on a
                        # physical joint stop. Position control retains the
                        # historical hard-envelope-only behavior.
                        joint_operating_margin_deg=(
                            3.0 if impedance_bimanual_ik_safety_profile is not None else 0.0
                        ),
                        current_target_hold_enabled=current_target_hold_enabled,
                        local_step_candidate_selection_enabled=local_step_candidate_selection_enabled,
                        prefer_non_wrist_singular_branch=bool(
                            arm_control_mode == "position"
                            and getattr(
                                args,
                                "position_prefer_non_wrist_singular_branch",
                                True,
                            )
                        ),
                        wrist_singularity_threshold_deg=3.0,
                        # Keep +/-pi wrist-flip candidates out of the local
                        # candidate pool while q5 is near the coaxial
                        # neighborhood.  This is position/single-step only;
                        # zero remains the atomic rollback to old ordering.
                        wrist_branch_switch_guard_deg=(
                            float(
                                getattr(
                                    args,
                                    "position_wrist_branch_switch_guard_deg",
                                    12.0,
                                )
                            )
                            if arm_control_mode == "position"
                            else 0.0
                        ),
                        wrist_q5_hysteresis_release_deg=(
                            float(
                                getattr(
                                    args,
                                    "position_wrist_q5_hysteresis_release_deg",
                                    6.0,
                                )
                            )
                            if arm_control_mode == "position"
                            else 0.0
                        ),
                        wrist_q5_hysteresis_requires_escape_latch=bool(
                            arm_control_mode == "position"
                            and getattr(
                                args,
                                "position_wrist_q5_hysteresis_requires_escape_latch",
                                False,
                            )
                        ),
                        prefer_non_elbow_singular_branch=bool(
                            arm_control_mode == "position"
                            and getattr(
                                args,
                                "position_prefer_non_elbow_singular_branch",
                                True,
                            )
                        ),
                        elbow_singularity_threshold_deg=3.0,
                        # Keep the local q3 escape band just above the RM65
                        # SDK elbow gate.  The previous 1 deg selection
                        # margin made ordinary 4--6 deg motion compete with
                        # the escape candidate and produced the alternating
                        # q3 trajectory seen in the 2026-07-26 trace.
                        elbow_singularity_selection_margin_deg=0.25,
                        prefer_non_shoulder_singular_branch=bool(
                            arm_control_mode == "position"
                            and getattr(
                                args,
                                "position_prefer_non_shoulder_singular_branch",
                                False,
                            )
                        ),
                        shoulder_singularity_threshold_m=0.05,
                        local_step_candidate_clip_scale=(
                            float(getattr(args, "position_ik_local_step_clip_scale", 1.0))
                            if arm_control_mode == "position"
                            else 1.0
                        ),
                        inactive_side_hold_enabled=inactive_side_hold_enabled,
                        inactive_side_hold_execution_cap_enabled=inactive_side_hold_execution_cap_enabled,
                        inactive_side_hold_execution_release_enabled=(
                            arm_control_mode == "position"
                        ),
                        inactive_side_hold_execution_release_translation_static_m=float(
                            getattr(
                                args,
                                "position_inactive_execution_release_translation_static_m",
                                0.03,
                            )
                        ),
                        inactive_side_hold_execution_release_translation_full_m=float(
                            getattr(
                                args,
                                "position_inactive_execution_release_translation_full_m",
                                0.10,
                            )
                        ),
                        inactive_side_hold_execution_release_rotation_static_deg=float(
                            getattr(
                                args,
                                "position_inactive_execution_release_rotation_static_deg",
                                5.0,
                            )
                        ),
                        inactive_side_hold_execution_release_rotation_full_deg=float(
                            getattr(
                                args,
                                "position_inactive_execution_release_rotation_full_deg",
                                30.0,
                            )
                        ),
                    ),
                    left_pose_retargeter=left_retargeter,
                    right_pose_retargeter=right_retargeter,
                )
        except (ModuleNotFoundError, RuntimeError, ValueError) as exc:
            collision_guard_setup_error = f"{type(exc).__name__}: {exc}"
        else:
            retargeters.append(dual_arm_retargeter)
    if collision_guard_requested and dual_arm_retargeter is None:
        raise RuntimeError(
            "Dual-arm collision avoidance was requested but failed to initialize; "
            "refusing real teleoperation without the requested safety layer. "
            f"setup_error={collision_guard_setup_error or 'unknown'}. "
            "Fix the collision solver/profile, or pass --no-collision explicitly "
            "only for an intentionally unguarded test."
        )
    if dual_arm_retargeter is None:
        side_retargeters = {"left": left_retargeter, "right": right_retargeter}
        retargeters.extend(side_retargeters[side] for side in active_sides)

    tcp_pose_envelope_filter = None
    if tcp_pose_envelope_profile_path:
        if dual_arm_retargeter is None:
            raise RuntimeError(
                "TCP pose envelope requires the coupled dual-arm retargeter so "
                "the final pre-IK pair has one authoritative safety owner"
            )
        tcp_pose_envelope_profile = load_tcp_pose_envelope_profile(
            tcp_pose_envelope_profile_path,
            calibration=calibration,
            retargeter=dual_arm_retargeter,
            active_sides=active_sides,
        )
        tcp_pose_envelope_filter = install_tcp_pose_envelope_filter(
            retargeter=dual_arm_retargeter,
            profile=tcp_pose_envelope_profile,
            calibration=calibration,
            execution_epoch=0,
            anchor_receipt=None,
            apply_to_target=tcp_pose_envelope_active,
        )
        calibration.metadata["tcp_pose_envelope"] = {
            "configured": True,
            "apply_to_target": tcp_pose_envelope_active,
            "profile_id": tcp_pose_envelope_profile.profile_id,
            "profile_hash": tcp_pose_envelope_profile.semantic_content_hash,
            "controlled_frame_id": tcp_pose_envelope_profile.controlled_frame_id,
            "enabled_sides": list(tcp_pose_envelope_profile.enabled_sides),
            "anchor_required_sides": list(
                tcp_pose_envelope_profile.anchor_required_sides
            ),
            "profile_path": tcp_pose_envelope_profile_path,
        }
        _log(
            "TCPPoseEnvelope",
            "installed final pre-IK "
            f"mode={'active' if tcp_pose_envelope_active else 'shadow'} "
            f"profile={tcp_pose_envelope_profile.profile_id} "
            f"sides={list(tcp_pose_envelope_profile.enabled_sides)}",
        )

    position_admittance_endpoint_guard = None
    if position_admittance_profile.active:
        if dual_arm_retargeter is None:
            raise RuntimeError(
                "Active position admittance endpoint excursion guard requires "
                "the coupled dual-arm retargeter"
            )
        if position_admittance_active_uses_home_endpoint_guard(
            position_admittance_profile.endpoint_policy
        ):
            endpoint_guard_kwargs: dict[str, float] = {}
            if (
                position_admittance_profile.maximum_allowed_tool_rotation_excursion_deg
                is not None
            ):
                endpoint_guard_kwargs[
                    "maximum_allowed_tool_rotation_excursion_deg"
                ] = (
                    position_admittance_profile.maximum_allowed_tool_rotation_excursion_deg
                )
            assert position_admittance_profile.max_tool_excursion_m is not None
            position_admittance_endpoint_guard = PositionAdmittanceEndpointExcursionGuard(
                retargeter=dual_arm_retargeter,
                active_side=position_admittance_profile.apply_sides[0],
                max_tool_excursion_m=(
                    position_admittance_profile.max_tool_excursion_m
                ),
                max_tool_rotation_excursion_deg=(
                    position_admittance_profile.max_tool_rotation_excursion_deg
                ),
                **endpoint_guard_kwargs,
            )

    calibration.metadata.update(
        {
            "dual_arm_collision_avoidance_enabled": dual_arm_retargeter is not None,
            "collision_avoidance_setup_error": collision_guard_setup_error,
        }
    )

    hand_backend_mode_by_side = {
        side: _hand_backend_mode_for_side(args, side) for side in active_sides
    }
    dmtac_active_sides = _dmtac_sides(args, active_sides)
    hand_backend_mode = (
        next(iter(set(hand_backend_mode_by_side.values())))
        if len(set(hand_backend_mode_by_side.values())) == 1
        else "mixed"
    )
    gripper_policy = _resolved_gripper_policy(args) if dmtac_active_sides else {}
    gripper_policies_by_side = (
        _resolved_gripper_policies_by_side(args, dmtac_active_sides)
        if dmtac_active_sides
        else {}
    )
    dmtac_configs = _dmtac_backend_configs(args, active_sides) if dmtac_active_sides else {}
    left_hand_port = getattr(args, "left_hand_port", "") or ""
    right_hand_port = getattr(args, "right_hand_port", "") or ""
    gripper_input_source = str(getattr(args, "gripper_input_source", "q6") or "q6")
    hand_source = None
    if gripper_input_source == "q6" and any(mode != HAND_BACKEND_NONE for mode in hand_backend_mode_by_side.values()) and (
        left_hand_port or right_hand_port
    ):
        hand_source = HandQ6Source.from_bimanual(
            left_port=(left_hand_port or None) if "left" in active_sides else None,
            right_port=(right_hand_port or None) if "right" in active_sides else None,
            baudrate=int(getattr(args, "hand_reader_baudrate", 115200)),
            timeout=float(getattr(args, "hand_reader_timeout", 0.05)),
            show_raw_lines=bool(getattr(args, "show_hand_raw_lines", False)),
            exo_hand_calibration=bool(getattr(args, "exo_hand_calibration", False)),
            calibration_path=str(getattr(args, "exo_hand_calibration_path", "") or "") or None,
            critical=False,
            strict_expected=False,
        )
    angle_trigger_source = None
    if gripper_input_source == "angle_trigger" and dmtac_active_sides:
        from hardware.angle_trigger.angle_trigger import prepare_reader

        config_root = getattr(args, "gripper_angle_trigger_config_root", None)
        try:
            reader = prepare_reader(
                config_root=config_root,
                sample_hz=float(getattr(args, "gripper_angle_trigger_rate_hz", 30.0)),
                active_sides=tuple(dmtac_active_sides),
            )
            angle_trigger_source = AngleTriggerSource(
                reader,
                rate_hz=float(getattr(args, "gripper_angle_trigger_rate_hz", 30.0)),
                config_root=config_root,
                active_sides=tuple(dmtac_active_sides),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            if str(getattr(args, "hardware_admission_mode", "strict")) == "strict":
                raise
            args.angle_trigger_startup_error = f"{type(exc).__name__}: {exc}"
            _log(
                "AngleTrigger",
                "WARNING: input initialization was isolated from arm/GUI startup: "
                f"{args.angle_trigger_startup_error}",
            )

    left_hand_retargeter = None
    right_hand_retargeter = None
    left_gripper_retargeter = None
    right_gripper_retargeter = None
    hand_retargeters_by_side: dict[str, Any] = {}
    for side in active_sides:
        mode = hand_backend_mode_by_side[side]
        hand_port = left_hand_port if side == "left" else right_hand_port
        if mode == HAND_BACKEND_DMTAC_GRIPPER:
            dmtac_config = dmtac_configs.get(side)
            policy = gripper_policies_by_side.get(side, gripper_policy)
            # The command/measured-state contract must remain stable when Q6
            # is absent at startup and connects later.  A missing Q6 channel
            # makes live input invalid; it must not erase the configured
            # ScalarToGripperRetargeter and make the dataset schema depend on
            # device connection timing.
            input_ready = _dmtac_command_contract_configured(
                gripper_input_source=gripper_input_source,
                angle_trigger_source=angle_trigger_source,
            )
            if input_ready and dmtac_config is not None and bool(dmtac_config.connect_hardware):
                hand_retargeters_by_side[side] = ScalarToGripperRetargeter(
                    ScalarToGripperRetargeterConfig(
                        side=side,
                        input_channel_name=(
                            f"{side}_hand_q6" if gripper_input_source == "q6" else f"{side}_angle_trigger"
                        ),
                        effector_name=f"{side}_hand",
                        source_dim=(int(policy["source_dim"]) if gripper_input_source == "q6" else None),
                        output_dim=int(policy["output_dim"]),
                        invert=bool(policy["invert"]),
                        alpha=float(policy["alpha"]),
                        input_min=float(policy["input_min"]),
                        input_max=float(policy["input_max"]),
                        output_max=float(policy["max_action"]),
                    )
                )
        elif mode == HAND_BACKEND_INSPIRE and hand_port:
            hand_retargeters_by_side[side] = JointsToDexHandRetargeter(
                JointsToDexHandRetargeterConfig(
                    side=side,
                    input_channel_name=f"{side}_hand_q6",
                    effector_name=f"{side}_hand",
                    joint_order=tuple(getattr(args, "hand_joint_order", DEFAULT_INSPIRE_HAND_JOINT_ORDER)),
                    invert=bool(getattr(args, "hand_invert", True)),
                    alpha=float(getattr(args, "hand_command_alpha", 1.0)),
                )
            )
    left_gripper_retargeter = hand_retargeters_by_side.get("left") if isinstance(
        hand_retargeters_by_side.get("left"), ScalarToGripperRetargeter
    ) else None
    right_gripper_retargeter = hand_retargeters_by_side.get("right") if isinstance(
        hand_retargeters_by_side.get("right"), ScalarToGripperRetargeter
    ) else None
    left_hand_retargeter = hand_retargeters_by_side.get("left") if isinstance(
        hand_retargeters_by_side.get("left"), JointsToDexHandRetargeter
    ) else None
    right_hand_retargeter = hand_retargeters_by_side.get("right") if isinstance(
        hand_retargeters_by_side.get("right"), JointsToDexHandRetargeter
    ) else None
    if left_hand_retargeter is not None:
        retargeters.append(left_hand_retargeter)
    if right_hand_retargeter is not None:
        retargeters.append(right_hand_retargeter)

    hand_tactile_regions = _parse_hand_tactile_regions(getattr(args, "hand_tactile_regions", "all"))
    hand_telemetry_hz = float(getattr(args, "hand_telemetry_hz", max(float(args.hz), 10.0)))

    hand_backends_by_side: dict[str, InspireTacHandBackend | DMTacGripperBackend] = {}
    for side in active_sides:
        mode = hand_backend_mode_by_side[side]
        hand_port = left_hand_port if side == "left" else right_hand_port
        hand_ip = str(getattr(args, f"{side}_hand_ip", "") or "")
        if mode == HAND_BACKEND_DMTAC_GRIPPER:
            dmtac_config = dmtac_configs.get(side)
            if dmtac_config is not None:
                hand_backends_by_side[side] = DMTacGripperBackend(dmtac_config)
        elif mode == HAND_BACKEND_INSPIRE and (hand_port or hand_ip):
            hand_backends_by_side[side] = InspireTacHandBackend(
                InspireTacHandBackendConfig(
                    hand_ip=hand_ip or "0.0.0.0",
                    side=side,
                    connect_hardware=bool(hand_ip),
                    max_fingertip_force=int(
                        getattr(args, "max_fingertip_force", DEFAULT_INSPIRE_HAND_MAX_FORCE)
                    ),
                    send_threshold=float(getattr(args, "hand_send_threshold", 0.002)),
                    enable_telemetry=bool(hand_ip),
                    telemetry_hz=hand_telemetry_hz,
                    tactile_regions=hand_tactile_regions,
                )
            )
    left_hand_backend = hand_backends_by_side.get("left")
    right_hand_backend = hand_backends_by_side.get("right")

    session_sources = [
        VivePoseSource(
            tracker_configs=tracker_configs,
            host=args.host,
            port=args.port,
            clock_sync_bind_host=(
                str(getattr(args, "vive_clock_sync_bind_host", "")).strip()
                or str(args.host)
            ),
            clock_sync_port=int(
                getattr(args, "vive_clock_sync_port", DEFAULT_VIVE_CLOCK_SYNC_PORT)
            ),
            clock_sync_max_round_trip_sec=(
                float(getattr(args, "vive_clock_sync_max_rtt_ms", 25.0)) / 1000.0
            ),
            clock_sync_stale_after_sec=float(
                getattr(args, "vive_clock_sync_stale_after_sec", 15.0)
            ),
            clock_sync_min_samples=int(
                getattr(args, "vive_clock_sync_min_samples", 4)
            ),
            clock_sync_expected_sender_host=(
                str(getattr(args, "vive_clock_sync_expected_sender_host", "")).strip()
                or None
            ),
        ),
        keyboard_source,
    ]
    if hand_source is not None:
        session_sources.append(hand_source)
    if angle_trigger_source is not None:
        session_sources.append(angle_trigger_source)

    arm_urdf_path = require_asset_path("rm65_arm.urdf")
    side_assignments = _resolve_logical_side_assignments(args)
    logical_arm_ip_mapping: dict[str, str] = {}
    left_command_backend = None
    right_command_backend = None
    left_state_backend = None
    right_state_backend = None
    left_preview_backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            default_tcp_frame=args.tcp_mode,
            arm_urdf_path=arm_urdf_path,
            connect_hardware=False,
            control_dt=action_dt,
            max_angvel=tuple(arm_max_angvel.tolist()),
            dry_run_fast=True,
            initial_qpos=tuple(np.asarray(left_seed_qpos, dtype=float).reshape(6).tolist()),
            use_actual_send_dt_scaling=False,
            tool_profile=_tool_profile_arg(args, "left"),
            ik_tuning=ik_tuning,
            ik_framework=str(getattr(args, "rm65_ik_framework", "robot_motion")),
        )
    )
    right_preview_backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="right",
            default_tcp_frame=args.tcp_mode,
            arm_urdf_path=arm_urdf_path,
            connect_hardware=False,
            control_dt=action_dt,
            max_angvel=tuple(arm_max_angvel.tolist()),
            dry_run_fast=True,
            initial_qpos=tuple(np.asarray(right_seed_qpos, dtype=float).reshape(6).tolist()),
            use_actual_send_dt_scaling=False,
            tool_profile=_tool_profile_arg(args, "right"),
            ik_tuning=ik_tuning,
            ik_framework=str(getattr(args, "rm65_ik_framework", "robot_motion")),
        )
    )
    for _source_name, (logical_side, arm_ip) in side_assignments.items():
        if not arm_ip or logical_side not in active_sides:
            continue
        if arm_control_mode == "impedance":
            command_backend = RM65ImpedanceArmBackend(
                RM65ImpedanceArmBackendConfig(
                    arm_ip=arm_ip,
                    side=logical_side,
                    connect_hardware=True,
                    tool_profile=_tool_profile_arg(args, logical_side),
                    strict_tool_profile=strict_tool_profile,
                    auto_sync_tool_frame=bool(getattr(args, "auto_sync_tool_frame", True)),
                    impedance_mode=impedance_mode,
                    force_tare_sec=force_tare_sec,
                    control_dt=arm_control_dt,
                    action_hz=outer_control_hz,
                    max_angvel=tuple(arm_max_angvel.tolist()),
                    actual_send_dt_alpha=float(args.arm_step_dt_alpha),
                    actual_send_dt_min_scale=1.0,
                    actual_send_dt_max_scale=1.0,
                    actual_send_dt_stall_reset_sec=float(args.arm_step_dt_stall_reset_sec),
                    scale_outer_execution_step_by_actual_dt=False,
                    debug_timing=bool(debug_timing_modules.rm65_backend),
                    impedance_overrides=dict(impedance_overrides),
                )
            )
            if tcp_pose_envelope_filter is not None:
                command_backend.set_final_tcp_pose_projector(
                    tcp_pose_envelope_filter.project_arm_base_tool_tcp
                )
        else:
            command_backend = RM65ArmBackend(
                RM65ArmBackendConfig(
                    arm_ip=arm_ip,
                    side=logical_side,
                    default_tcp_frame=args.tcp_mode,
                    arm_urdf_path=arm_urdf_path,
                    connect_hardware=True,
                    control_dt=action_dt,
                    max_angvel=tuple(arm_max_angvel.tolist()),
                    dry_run_fast=False,
                    hardware_read_retries=int(args.rm65_hardware_read_retries),
                    hardware_read_retry_delay_sec=float(args.rm65_hardware_read_retry_delay_sec),
                    initial_qpos=None,
                    use_actual_send_dt_scaling=bool(args.arm_step_dt_scaling),
                    actual_send_dt_alpha=float(args.arm_step_dt_alpha),
                    actual_send_dt_min_scale=1.0,
                    actual_send_dt_max_scale=float(args.arm_step_dt_max_scale),
                    actual_send_dt_stall_reset_sec=float(args.arm_step_dt_stall_reset_sec),
                    background_streaming_enabled=position_background_streaming_enabled,
                    background_streaming_hz=position_background_streaming_hz,
                    background_streaming_warning_ratio=canfd_follow_warning_ratio,
                    position_command_planner=position_command_planner,
                    position_velocity_ramp_sec=float(
                        getattr(
                            args,
                            "position_velocity_ramp_sec",
                            DEFAULT_POSITION_VELOCITY_RAMP_SEC,
                        )
                    ),
                    position_target_velocity_filter_tau_sec=float(
                        getattr(
                            args,
                            "position_target_velocity_filter_tau_sec",
                            DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
                        )
                    ),
                    position_max_acceleration_rad_s2=tuple(
                        np.asarray(
                            getattr(
                                args,
                                "position_max_acceleration_rad_s2",
                                DEFAULT_POSITION_MAX_ACCELERATION_RAD_S2,
                            ),
                            dtype=float,
                        ).reshape(6).tolist()
                    ),
                    position_max_command_step_deg=(
                        None
                        if getattr(args, "position_max_command_step_deg", None) is None
                        else float(args.position_max_command_step_deg)
                    ),
                    position_degraded_speed_scale=float(
                        getattr(
                            args,
                            "position_degraded_speed_scale",
                            DEFAULT_POSITION_DEGRADED_SPEED_SCALE,
                        )
                    ),
                    position_sample_aligned_source_hz=float(
                        getattr(
                            args,
                            "position_sample_aligned_source_hz",
                            args.hz,
                        )
                    ),
                    position_sample_aligned_source_latch_mode=str(
                        getattr(
                            args,
                            "position_sample_aligned_source_latch_mode",
                            "time_bucket",
                        )
                    ),
                    position_sample_aligned_horizon_source_steps=int(
                        getattr(
                            args,
                            "position_sample_aligned_horizon_source_steps",
                            2,
                        )
                    ),
                    position_sample_aligned_safety_horizon_source_steps=(
                        None
                        if getattr(
                            args,
                            "position_sample_aligned_safety_horizon_source_steps",
                            None,
                        )
                        is None
                        else int(
                            args.position_sample_aligned_safety_horizon_source_steps
                        )
                    ),
                    position_sample_aligned_scale_degraded_target=bool(
                        getattr(
                            args,
                            "position_sample_aligned_scale_degraded_target",
                            True,
                        )
                    ),
                    position_sample_aligned_jerk_regularization=float(
                        getattr(
                            args,
                            "position_sample_aligned_jerk_regularization",
                            1e-9,
                        )
                    ),
                    position_sample_aligned_jerk_delta_regularization=float(
                        getattr(
                            args,
                            "position_sample_aligned_jerk_delta_regularization",
                            0.0,
                        )
                    ),
                    position_sample_aligned_initial_jerk_delta_regularization=float(
                        getattr(
                            args,
                            "position_sample_aligned_initial_jerk_delta_regularization",
                            0.0,
                        )
                    ),
                    position_sample_aligned_terminal_velocity_regularization=float(
                        getattr(
                            args,
                            "position_sample_aligned_terminal_velocity_regularization",
                            0.0,
                        )
                    ),
                    position_sample_aligned_terminal_acceleration_regularization=float(
                        getattr(
                            args,
                            "position_sample_aligned_terminal_acceleration_regularization",
                            0.0,
                        )
                    ),
                    position_sample_aligned_reference_mode=str(
                        getattr(
                            args,
                            "position_sample_aligned_reference_mode",
                            "linear_hold",
                        )
                    ),
                    position_sample_aligned_observer_window=int(
                        getattr(
                            args,
                            "position_sample_aligned_observer_window",
                            4,
                        )
                    ),
                    position_sample_aligned_observer_efficiency_power=float(
                        getattr(
                            args,
                            "position_sample_aligned_observer_efficiency_power",
                            2.0,
                        )
                    ),
                    position_sample_aligned_observer_velocity_ema_alpha=float(
                        getattr(
                            args,
                            "position_sample_aligned_observer_velocity_ema_alpha",
                            0.5,
                        )
                    ),
                    position_sample_aligned_observer_velocity_cap_scale=float(
                        getattr(
                            args,
                            "position_sample_aligned_observer_velocity_cap_scale",
                            0.5,
                        )
                    ),
                    position_sample_aligned_observer_coherence_min=float(
                        getattr(
                            args,
                            "position_sample_aligned_observer_coherence_min",
                            0.5,
                        )
                    ),
                    position_sample_aligned_observer_speed_guard_scale=float(
                        getattr(
                            args,
                            "position_sample_aligned_observer_speed_guard_scale",
                            0.4,
                        )
                    ),
                    position_sample_aligned_terminal_velocity_displacement_scale=float(
                        getattr(
                            args,
                            "position_sample_aligned_terminal_velocity_displacement_scale",
                            0.75,
                        )
                    ),
                    position_sample_aligned_observer_stop_threshold_deg=float(
                        getattr(
                            args,
                            "position_sample_aligned_observer_stop_threshold_deg",
                            0.01,
                        )
                    ),
                    position_sample_aligned_observer_stop_velocity_decay=float(
                        getattr(
                            args,
                            "position_sample_aligned_observer_stop_velocity_decay",
                            0.0,
                        )
                    ),
                    arm_follow_qpos_alpha=float(arm_follow_qpos_alpha),
                    canfd_follow=bool(arm_canfd_follow),
                    canfd_trajectory_mode=canfd_trajectory_mode,
                    canfd_radio=canfd_radio,
                    tool_profile=_tool_profile_arg(args, logical_side),
                    singularity_protection_enabled=bool(args.arm_singularity_protection),
                    allow_position_wrist_singularity_fallback=bool(
                        arm_control_mode == "position"
                        and getattr(
                            args,
                            "position_prefer_non_wrist_singular_branch",
                            True,
                        )
                    ),
                    allow_position_elbow_singularity_fallback=bool(
                        arm_control_mode == "position"
                        and getattr(
                            args,
                            "position_prefer_non_elbow_singular_branch",
                            True,
                        )
                    ),
                    allow_position_shoulder_singularity_fallback=bool(
                        arm_control_mode == "position"
                        and getattr(
                            args,
                            "position_prefer_non_shoulder_singular_branch",
                            False,
                        )
                    ),
                    require_position_singularity_candidate_certificate=bool(
                        arm_control_mode == "position"
                        and getattr(
                            args,
                            "position_singularity_candidate_certificate",
                            False,
                        )
                    ),
                    position_sdk_singularity_advisory=bool(
                        arm_control_mode == "position"
                        and getattr(
                            args,
                            "position_singularity_candidate_certificate",
                            False,
                        )
                    ),
                    ik_tuning=ik_tuning,
                    ik_framework=str(getattr(args, "rm65_ik_framework", "robot_motion")),
                )
            )
        state_backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip=arm_ip,
                side=logical_side,
                default_tcp_frame=args.tcp_mode,
                arm_urdf_path=arm_urdf_path,
                connect_hardware=True,
                control_dt=arm_control_dt,
                max_angvel=tuple(arm_max_angvel.tolist()),
                dry_run_fast=False,
                hardware_read_retries=int(args.rm65_hardware_read_retries),
                hardware_read_retry_delay_sec=float(args.rm65_hardware_read_retry_delay_sec),
                initial_qpos=None,
                use_actual_send_dt_scaling=False,
                tool_profile=_tool_profile_arg(args, logical_side),
                ik_tuning=ik_tuning,
                ik_framework=str(getattr(args, "rm65_ik_framework", "robot_motion")),
            )
        )
        logical_arm_ip_mapping[logical_side] = arm_ip
        if logical_side == "left":
            left_command_backend = command_backend
            left_state_backend = state_backend
        else:
            right_command_backend = command_backend
            right_state_backend = state_backend
    for side in active_sides:
        command_backend = left_command_backend if side == "left" else right_command_backend
        state_backend = left_state_backend if side == "left" else right_state_backend
        if command_backend is None or state_backend is None:
            raise RuntimeError(f"Failed to resolve logical {side} arm assignment.")

    left_hand_control_via_rig = left_hand_retargeter is not None and left_hand_backend is not None
    right_hand_control_via_rig = right_hand_retargeter is not None and right_hand_backend is not None
    fix_arm_controller = None
    if bool(getattr(args, "fix_arm", False)):
        if set(active_sides) != {"left", "right"}:
            raise ValueError("--fix-arm requires --active-sides both.")
        fix_arm_controller = FixArmCommandController(
            active_sides=active_sides,
            required_effectors_by_side=_fix_arm_required_effectors_by_side(
                active_sides=active_sides,
                left_hand_control_via_rig=left_hand_control_via_rig,
                right_hand_control_via_rig=right_hand_control_via_rig,
                left_gripper_retargeter=left_gripper_retargeter,
                right_gripper_retargeter=right_gripper_retargeter,
                left_hand_backend=left_hand_backend,
                right_hand_backend=right_hand_backend,
            ),
            selected_side=str(getattr(args, "fix_arm_side", "left") or "left"),
        )
        calibration.metadata["fix_arm_mode"] = {
            "enabled": True,
            "initial_selected_side": fix_arm_controller.selected_side,
            "required_effectors_by_side": {
                side: list(effectors)
                for side, effectors in fix_arm_controller.required_effectors_by_side.items()
            },
        }
    init_arm_controller = None
    if bool(getattr(args, "init_arm_available", False)):
        if set(active_sides) != {"left", "right"}:
            raise ValueError("--init-arm-available requires --active-sides both.")
        init_arm_profile = str(getattr(args, "init_arm_profile", "") or "").strip()
        init_targets = load_init_arm_targets(
            init_arm_profile,
            calibration=calibration,
            dual_arm_retargeter=dual_arm_retargeter,
        )
        init_arm_controller = InitArmCommandController(
            targets_by_side=init_targets,
            selected_side=str(getattr(args, "init_arm_side", "left") or "left"),
            use_dual_target_override=dual_arm_retargeter is not None,
            qpos_final_collision_checker=(
                None
                if dual_arm_retargeter is None
                else dual_arm_retargeter.check_qpos_pair_collision
            ),
        )
        if dual_arm_retargeter is not None:
            dual_arm_retargeter.target_override_filter = init_arm_controller.filter_dual_arm_targets
        calibration.metadata["init_arm_mode"] = {
            "enabled": True,
            "initial_selected_side": init_arm_controller.selected_side,
            "profile_path": init_arm_profile,
            "dual_target_override": dual_arm_retargeter is not None,
            "resolved_sides": sorted(init_targets),
        }
    rig_backends = {}
    if "left" in active_sides and left_command_backend is not None:
        rig_backends["left_arm"] = left_command_backend
    if "right" in active_sides and right_command_backend is not None:
        rig_backends["right_arm"] = right_command_backend
    if left_hand_control_via_rig:
        rig_backends["left_hand"] = left_hand_backend
    if right_hand_control_via_rig:
        rig_backends["right_hand"] = right_hand_backend

    rig = BimanualRig(
        backends=rig_backends,
        parallel_dispatch=arm_io_threads > 1,
        parallel_workers=max(arm_io_threads, len(rig_backends)),
        fail_stop_on_dispatch_error=arm_control_mode == "position",
    )
    session_record_enabled = _session_recording_enabled(args)
    session_record_root = _session_record_root_dir(args) if session_record_enabled else None
    session_recorder = (
        SessionRecorder(
            root_dir=session_record_root,
            config=_session_recording_config(args),
        )
        if session_record_enabled
        else None
    )
    human_shadow_frontend = build_real_teleop_human_shadow_frontend(
        RealTeleopHumanShadowProfile(
            mode=str(getattr(args, "human_action_arbiter_mode", "off")),
            pose_source_id=(
                getattr(args, "pose_frame_contract").source.source_id
                if getattr(args, "pose_frame_contract", None) is not None
                else legacy_rm65_source_id(active_sides)
            ),
            active_sides=active_sides,
            arm_reference_frame_by_side={
                side: str(
                    (left_retargeter if side == "left" else right_retargeter).config.output_frame_name
                )
                for side in active_sides
            },
            hand_backend_by_side=dict(hand_backend_mode_by_side),
            session_hand_sides=tuple(
                side
                for side, enabled in (
                    ("left", left_hand_control_via_rig),
                    ("right", right_hand_control_via_rig),
                )
                if side in active_sides and enabled
            ),
            out_of_session_effectors=tuple(
                f"{side}_hand"
                for side, retargeter in (
                    ("left", left_gripper_retargeter),
                    ("right", right_gripper_retargeter),
                )
                if side in active_sides and retargeter is not None
            ),
            gripper_slot_by_side={
                side: int(policy.get("output_dim", 0))
                for side, policy in gripper_policies_by_side.items()
            },
            action_space=(
                "bimanual_qpos_v1"
                if dual_arm_retargeter is not None and arm_control_mode == "position"
                else "bimanual_ee_pose_v1"
            ),
            arm_joint_policy=(
                "qpos"
                if dual_arm_retargeter is not None and arm_control_mode == "position"
                else "source_pose_when_available"
            ),
            decision_hz=float(getattr(args, "human_action_arbiter_hz", 60.0)),
            trace_capacity=int(
                getattr(args, "human_action_shadow_trace_capacity", 262_144)
            ),
            observer_queue_capacity=int(
                getattr(args, "human_action_shadow_observer_queue_size", 4096)
            ),
            trace_root=getattr(args, "human_action_shadow_trace_root", None),
            wal_max_bytes=int(
                getattr(args, "human_action_shadow_wal_max_mb", 128)
            )
            * 1024
            * 1024,
            control_domain_id=str(
                getattr(args, "human_action_shadow_control_domain_id", "") or ""
            ),
            embodiment_instance_id=str(
                getattr(args, "human_action_shadow_embodiment_instance_id", "") or ""
            ),
        )
    )
    startup_home_continuity_gate = None
    if (
        arm_control_mode == "position"
        and position_command_planner
        in {
            POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
            POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
        }
    ):
        startup_command_backends = {
            "left": left_command_backend,
            "right": right_command_backend,
        }
        startup_home_continuity_gate = StartupHomeContinuityGate(
            config=StartupHomeContinuityConfig(active_sides=active_sides),
            fk_by_side={
                side: (
                    lambda qpos, backend=startup_command_backends[side]:
                    backend.compute_tcp_pose(qpos=qpos)
                )
                for side in active_sides
            },
            logger=lambda message: _log("StartupHomeContinuity", message),
        )

    session = TeleopSession(
        config=TeleopSessionConfig(
            mode="vive_rm65_real_teleop",
            tcp_mode=args.tcp_mode,
            record_enabled=session_record_enabled,
            auto_recovery_enabled=True,
            recovery_healthy_frame_count=max(1, int(args.recovery_healthy_frames)),
            hold_motion_on_freeze=True,
            tracker_loss_hold_enabled=True,
            tracker_auto_hold_sec=float(getattr(args, "tracker_auto_hold_sec", 0.10)),
            tracker_manual_hold_sec=float(getattr(args, "tracker_manual_hold_sec", 1.0)),
            non_freezing_fault_codes=("low_control_rate",),
        ),
        sources=session_sources,
        retargeters=retargeters,
        rig=rig,
        calibration=calibration,
        recorder=session_recorder,
        safety_config=TeleopSafetyConfig(
            freeze_on_fault=True,
            required_channels=tuple(f"{side}_wrist_pose" for side in active_sides),
            max_translation_jump_m=float(
                getattr(
                    args,
                    "max_tracker_translation_jump_m",
                    DEFAULT_REAL_TELEOP_MAX_TRACKER_TRANSLATION_JUMP_M,
                )
            ),
            max_rotation_jump_deg=float(
                getattr(
                    args,
                    "max_tracker_rotation_jump_deg",
                    DEFAULT_REAL_TELEOP_MAX_TRACKER_ROTATION_JUMP_DEG,
                )
            ),
            invalid_frame_limit=3,
            minimum_control_hz=float(args.minimum_control_hz),
        ),
        channel_filter=(
            fix_arm_controller.filter_channels
            if fix_arm_controller is not None
            else None
        ),
        command_filter=_compose_real_command_filter(
            temporary_safety_limiter=temporary_safety_limiter,
            fix_arm_controller=fix_arm_controller,
            init_arm_controller=init_arm_controller,
            position_admittance_endpoint_guard=(
                position_admittance_endpoint_guard
            ),
        ),
        command_guard=_compose_real_command_guard(
            fix_arm_controller=fix_arm_controller,
            init_arm_controller=init_arm_controller,
            dual_arm_retargeter=dual_arm_retargeter,
        ),
        command_dispatch_admission=(
            None
            if startup_home_continuity_gate is None
            else startup_home_continuity_gate.admit
        ),
        fault_filter=_compose_real_fault_filter(
            fix_arm_controller=fix_arm_controller,
            init_arm_controller=init_arm_controller,
        ),
        human_command_observer=(
            None
            if human_shadow_frontend is None
            else human_shadow_frontend.observe_emission
        ),
        human_command_filter_observer=(
            None
            if human_shadow_frontend is None or temporary_safety_limiter is None
            else (
                lambda emission, command, observed_at: (
                    human_shadow_frontend.observe_filtered_command(
                        emission,
                        command,
                        observed_at,
                        temporary_safety_limiter.last_evaluation,
                    )
                )
            )
        ),
        human_execution_observer=(
            None
            if human_shadow_frontend is None
            else human_shadow_frontend.observe_legacy_execution
        ),
    )
    if human_shadow_frontend is not None:
        _log(
            "ActionArbiter",
            "human shadow configured observe-only "
            f"runtime_session={human_shadow_frontend.runtime_session_id} "
            f"control_domain={human_shadow_frontend.control_domain_id} "
            f"decision_hz={float(args.human_action_arbiter_hz):g} "
            f"trace={'scoped_wal' if human_shadow_frontend.wal_directory is not None else 'in_memory'}",
        )
    _log(
        "Safety",
        "safety_freeze_hold=on auto_recovery=on for RM65 real teleop; "
        "FROZEN requests rig hold_motion and stops new command dispatch until healthy frames enter RECOVERING. "
        f"tracker_jump_limits={float(getattr(args, 'max_tracker_translation_jump_m', DEFAULT_REAL_TELEOP_MAX_TRACKER_TRANSLATION_JUMP_M)):.3f}m/"
        f"{float(getattr(args, 'max_tracker_rotation_jump_deg', DEFAULT_REAL_TELEOP_MAX_TRACKER_ROTATION_JUMP_DEG)):.1f}deg",
    )

    renderer = None
    if not args.skip_renderer:
        try:
            renderer = SapienArmHandRenderer(
                left_urdf_path=require_asset_path("rm65_inspire_tac_left.urdf"),
                left_retargeting_joint_names=LEFT_HAND_RENDER_JOINT_NAMES,
                right_urdf_path=require_asset_path("rm65_inspire_tac_right.urdf"),
                right_retargeting_joint_names=RIGHT_HAND_RENDER_JOINT_NAMES,
                left_right_base_distance=args.left_right_base_distance,
                show_viewer=not args.headless,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to create the SAPIEN renderer. Check that the current environment "
                "has a usable Vulkan/OpenGL render device, or rerun with --skip-renderer "
                "to validate the real teleop loop without visualization."
            ) from exc
        renderer.reserve_debug_frames(
            (
                "desk",
                "left_arm_base",
                "right_arm_base",
                "left_tracker",
                "right_tracker",
                "left_hand_home",
                "right_hand_home",
                "left_robot_home",
                "right_robot_home",
                "left_tcp_actual",
                "right_tcp_actual",
                "left_tcp_target",
                "right_tcp_target",
            )
        )
        renderer.reserve_debug_boxes(_collision_debug_box_templates(collision_scene_profile))

    if dual_arm_retargeter is not None:
        dual_arm_retargeter.set_current_qpos_pair(
            left_qpos=left_seed_qpos,
            right_qpos=right_seed_qpos,
        )

    arm_state_read_executor = (
        ThreadPoolExecutor(
            max_workers=min(arm_io_threads, len(active_sides)),
            thread_name_prefix="rm65_state",
        )
        if arm_io_threads > 1 and len(active_sides) > 1
        else None
    )

    position_reference_publisher = None
    if arm_control_mode == "position":
        command_backends = {
            "left": left_command_backend,
            "right": right_command_backend,
        }
        if dual_arm_retargeter is not None:
            shoulder_preference_enabled = bool(
                getattr(
                    args,
                    "position_prefer_non_shoulder_singular_branch",
                    False,
                )
            )
            dual_arm_retargeter.set_position_singularity_analyzers(
                {
                    side: command_backends[side].analyze_position_singularity_candidate
                    for side in active_sides
                    if shoulder_preference_enabled
                    and isinstance(command_backends[side], RM65ArmBackend)
                }
            )
        urdf_root_from_base_by_side: dict[str, np.ndarray] = {}
        if str(getattr(args, "control_debug", "off")).strip().lower() == "full":
            for side in active_sides:
                root_from_base = calibration.get_arm_for_side(side).T_D_B
                if root_from_base is None:
                    continue
                transform = np.asarray(root_from_base, dtype=float)
                if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                    raise ValueError(
                        f"{side} arm calibration T_D_B must be a finite 4x4 pose for "
                        "full position debug."
                    )
                urdf_root_from_base_by_side[side] = transform.copy()
                setter = getattr(
                    command_backends[side],
                    "set_position_audit_urdf_root_from_base",
                    None,
                )
                if callable(setter):
                    setter(transform)
        position_reference_publisher = PositionReferencePublisher(
            config=PositionReferencePublisherConfig(
                active_sides=active_sides,
                source="teleop_runtime.debug.run_vive_rm65_real_teleop",
                reference_max_age_sec=float(args.position_reference_max_age_sec),
            ),
            fk_by_side={
                side: (
                    lambda qpos, backend=command_backends[side]:
                    backend.compute_tcp_pose(qpos=qpos)
                )
                for side in active_sides
            },
            urdf_root_from_base_by_side=urdf_root_from_base_by_side,
        )

    return RealTeleopComponents(
        session=session,
        active_sides=active_sides,
        left_command_backend=left_command_backend,
        right_command_backend=right_command_backend,
        left_preview_backend=left_preview_backend,
        right_preview_backend=right_preview_backend,
        left_state_backend=left_state_backend,
        right_state_backend=right_state_backend,
        left_hand_backend=left_hand_backend,
        right_hand_backend=right_hand_backend,
        initial_alignment=None,
        loaded_initial_alignment=loaded_initial_alignment,
        logical_arm_ip_mapping=logical_arm_ip_mapping,
        collision_scene_profile=collision_scene_profile,
        dual_arm_retargeter=dual_arm_retargeter,
        collision_guard_requested=collision_guard_requested,
        collision_guard_setup_error=collision_guard_setup_error,
        renderer=renderer,
        keyboard_source=keyboard_source,
        left_retargeter=left_retargeter,
        right_retargeter=right_retargeter,
        left_hand_retargeter=left_hand_retargeter,
        right_hand_retargeter=right_hand_retargeter,
        left_gripper_retargeter=left_gripper_retargeter,
        right_gripper_retargeter=right_gripper_retargeter,
        left_hand_control_via_rig=left_hand_control_via_rig,
        right_hand_control_via_rig=right_hand_control_via_rig,
        hand_backend_mode=hand_backend_mode,
        gripper_input_source=gripper_input_source,
        gripper_policy=dict(gripper_policy),
        joint_distance_weights=np.asarray(
            ik_tuning.branch_normalization.joint_distance_weights,
            dtype=float,
        ),
        arm_state_read_executor=arm_state_read_executor,
        arm_loop_step_dt_tracker={
            "last_loop_time": None,
            "smoothed_loop_dt": action_dt,
        },
        human_shadow_frontend=human_shadow_frontend,
        temporary_safety_limiter=temporary_safety_limiter,
        position_admittance_endpoint_guard=position_admittance_endpoint_guard,
        agent_motion_port=None,
        fix_arm_controller=fix_arm_controller,
        init_arm_controller=init_arm_controller,
        position_reference_publisher=position_reference_publisher,
        startup_home_continuity_gate=startup_home_continuity_gate,
    )

__all__ = ["build_real_teleop_components"]
