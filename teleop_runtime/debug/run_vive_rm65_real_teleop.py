from __future__ import annotations

import argparse
import json
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
import re
import sys
import threading
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation

from cameras.configs import DEFAULT_REALSENSE_DEVICE_CONFIG_PATH
from dataset_contracts.pose_frames import load_pose_frame_contract
from ..backends.arms import (
    RM65ArmBackend,
    RM65ArmBackendConfig,
    RM65ImpedanceArmBackend,
    RM65ImpedanceArmBackendConfig,
)
from ..backends.grippers import (
    DEFAULT_DMTAC_GRIPPER_COMMAND_HZ,
    DMTAC_TACTILE_FIELDS,
    DMTacGripperBackend,
    DMTacGripperBackendConfig,
    GRIPPER_ACTION_MAX,
    normalize_dmtac_tactile_fields,
)
from ..backends.hands import (
    DEFAULT_INSPIRE_HAND_MAX_FORCE,
    InspireTacHandBackend,
    InspireTacHandBackendConfig,
)
from ..arm_frames import TOOL_TCP_FRAME
from robot_motion.action import HandActionSpec
from robot_motion.profiles import IKSafetyProfile, load_ik_safety_profiles
from robot_motion.position_control import (
    DEFAULT_POSITION_DEGRADED_SPEED_SCALE,
    DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
    DEFAULT_POSITION_VELOCITY_RAMP_SEC,
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
    POSITION_COMMAND_PLANNER_CHOICES,
    POSITION_COMMAND_PLANNER_LINEAR,
    PositionAuditFanout,
    normalize_position_command_planner,
)
from robot_motion.runtime import (
    EffectorCommand,
    LifecycleStartupError,
    TeleopCommand,
    collect_lifecycle_failures,
    merge_teleop_commands,
    run_best_effort_cleanup,
)
from ..calibration import TeleopCalibration
from ..collection.diagnostics import (
    PhaseTimer,
    build_end_to_end_latency_breakdown,
    format_end_to_end_latency_breakdown,
    format_phase_timings,
    should_report_debug_frozen,
)
from ..collection.async_writer import AsyncWriterConfig, AsyncWriterPool
from ..collection.arm_state_polling import ArmHardwareStatePoller, ArmHardwareStatePollerConfig
from ..collection.hand_state_polling import HandStateHistoryPoller, HandStateHistoryPollerConfig
from ..collection.camera_preview import CameraPreviewConfig, CameraPreviewer
from ..collection.build_collect_components import CollectorComponents
from ..collection.observers import CollectionWriterObserverConfig
from ..collection.policy_io_runtime import resolve_collection_policy_io_layout
from ..collection.process_writer import (
    ProcessCollectionConfig,
    ProcessCollectionWorker,
    camera_shared_slot_bytes,
)
from ..collection.preview_rate import PreviewRateLimiter
from ..collection.rate_gate import PeriodicRateGate
from ..collection.sample_assembler import (
    DEFAULT_CAMERA_FRAME_MAX_AGE_SEC,
    CollectorSampleBuilder,
    CollectorSampleBuilderConfig,
    _build_pose_quality_metadata as _collection_build_pose_quality_metadata,
)
from ..collection.status_snapshot import (
    camera_debug_text as _collection_camera_debug_text,
    hand_debug_text as _collection_hand_debug_text,
    writer_debug_text as _collection_writer_debug_text,
)
from ..collection.writer_runtime import CollectionWriterRuntime
from ..collection.writer_v21 import LeRobotV21Writer, LeRobotV21WriterConfig
from ..configs import (
    ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
    DEFAULT_ARM_WORKSPACE_CONFIG_PATH,
    build_bimanual_pose_smoother,
    build_annotated_dual_rm65_calibration,
    load_default_pose_smoothing_config,
    load_impedance_pose_smoothing_config,
)
from ..configs.dmtac_hardware import (
    dmtac_backend_configs_from_assignment,
    load_dmtac_hardware_assignment,
)
from ..configs.q6_hardware import load_latest_q6_ports
from ..configs.rm65_arm_ips import (
    arm_ips_for_sides,
    build_assignment as build_rm65_arm_ip_assignment,
    load_assignment as load_rm65_arm_ip_assignment,
    save_assignment_for_config as save_rm65_arm_ip_assignment_for_config,
)
from ..defaults import (
    DEFAULT_IMPEDANCE_TELEOP_CONTROL_HZ,
    DEFAULT_RM65_ARM_CONTROL_HZ,
    DEFAULT_RM65_POSITION_ARM_CONTROL_HZ,
    DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
    DEFAULT_TELEOP_SAMPLE_HZ,
)
from ..device_preflight import (
    DevicePreflightError,
    DevicePreflightOptions,
    prepare_devices_for_teleop,
)
from ..diagnostics import TeleopRunState
from ..episodes import EpisodeRecorder, EpisodeRecorderConfig, compare_start_qpos_pair
from ..hand_mapping import (
    DEFAULT_INSPIRE_HAND_JOINT_ORDER,
    TARGET_INSPIRE_HAND_Q6_ORDER,
    parse_tactile_regions,
)
from ..hand_render import (
    LEFT_HAND_RENDER_JOINT_NAMES,
    RIGHT_HAND_RENDER_JOINT_NAMES,
)
from ..hand_tcp_alignment import (
    DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG,
    add_vive_tracker_height_correction_args,
    tracker_hand_to_rm65_tcp_rotation,
)
from ..tracker_hand import TRACKER_OFFSET_CALIBRATION_VERSION, canonical_hand_rpy_default
from ..initial_alignment import TeleopInitialAlignment
from ..retargeters import (
    DEFAULT_GRIPPER_INPUT_MAX,
    DEFAULT_GRIPPER_INPUT_MIN,
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
    JointsToDexHandRetargeter,
    JointsToDexHandRetargeterConfig,
    PoseToArmRetargeter,
    PoseToArmRetargeterConfig,
    ScalarToGripperRetargeter,
    ScalarToGripperRetargeterConfig,
    map_scalar_to_gripper_target,
)
from ..recording import RecordingConfig, SessionRecorder
from ..position_control import (
    ARM_EXECUTION_PROFILE_CHOICES,
    CONTROL_DEBUG_LEVELS,
    DEFAULT_CONTROL_DEBUG_FLUSH_EVERY,
    DEFAULT_CONTROL_DEBUG_QUEUE_SIZE,
    DEFAULT_CONTROL_DEBUG_ROOT,
    PositionDebugRecorder,
    PositionDebugRecorderConfig,
    PositionReferencePublisher,
    PositionReferencePublisherConfig,
    RealArmExecutionConfig,
    resolve_control_debug_config,
    resolve_real_arm_execution_config,
)
from ..rigs import BimanualRig
from ..safety import TeleopSafetyConfig
from ..session import (
    TRACKER_HOLD_AUTOMATIC,
    TRACKER_HOLD_GRACE,
    SessionStepResult,
    TeleopSession,
    TeleopSessionConfig,
)
from ..sources.camera import RealSenseRGBSource, load_realsense_camera_config_map
from ..sources.hand import AngleTriggerSource, HandQ6Source
from ..sources.misc import KeyboardButtonSource
from ..sources.pose import DEFAULT_VIVE_CLOCK_SYNC_PORT, VivePoseSource
from ..sources.pose.vive_motion import TrackerMotionIntentConfig
from ..special_tasks.fix_arm import (
    FixArmCommandController,
    FixArmPoseSnapshotRecorder,
    advance_fix_arm_home_capture,
    build_fix_arm_pose_snapshot_payload,
)
from ..special_tasks.init_arm import (
    InitArmCommandController,
    advance_init_arm_home_capture,
    load_init_arm_targets,
)
from ..tactile_visualizer import (
    AsyncTactileUVVisualizer,
    DMTacDistributedForceVisualizer,
    DMTacForceVisualizerConfig,
    TactileUVVisualizer,
    TactileVisualizerConfig,
    dmtac_force_by_side_from_backends,
    tactile_by_side_from_backends,
)
from ..tcp import legacy_tcp_mode_requested
from ..transforms import as_pose_matrix
from .calibration_flow import (
    average_pose,
    euler_xyz_deg_to_matrix,
    fit_signed_axis_from_poses,
    fit_virtual_desk_frame,
    pose_from_rotation_translation,
    tracker_offset_from_neutral_pose,
)
from ..gui import collection_tasks as gui_collection_tasks
from ..gui.calibration_web_ui import CalibrationWebUi
from ..real_teleop.foot_pad_controls import (
    FootPadSpecialArmInput,
    SPECIAL_ARM_CONTROL_MODE_CHOICES,
    SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
    SPECIAL_ARM_CONTROL_MODE_MANUAL,
    resolve_special_arm_control_mode,
)
from ..real_teleop.fix_arm_collision_guard import fixed_arm_final_collision_fault
from ..real_teleop import dataset_layout as real_teleop_dataset_layout
from .log_utils import ChangeGate
from .run_vive_sapien_virtual import (
    DEFAULT_RUNTIME_HOME_START_DELAY_SEC,
    RecoveryPlan,
    TerminalClutchInput,
    _active_sides_from_args,
    _advance_recovery_plan,
    _build_episode_frame,
    _build_renderer_debug_boxes,
    _build_renderer_debug_frames,
    _channel_pose_in_desk,
    _collision_debug_box_templates,
    _emit_collision_proxy_logs,
    _emit_home_alignment_logs,
    _emit_runtime_orientation_logs,
    _emit_tracker_axis_mapping_logs,
    _log,
    _manual_calibration_confirmation_required,
    _mean_pose,
    _non_negative_int,
    _alpha_float,
    _parse_active_sides,
    _parse_arm_max_angvel,
    _parse_joint_order,
    _positive_float,
    _qpos_summary,
    _resolve_robot_home_for_side,
    _runtime_home_calibration,
    _side_retargeter,
    _start_recovery_plan,
    _state_qpos,
    _sync_dual_arm_retargeter_from_preview,
    _hand_render_qpos,
    _vive_tracker_local_translation_offset,
)
from robot_motion.rm65.ik_tuning import (
    IKObjectiveWeights,
    RM65IKTuningConfig,
    load_ik_tuning_config,
)
from robot_motion.rm65.assets import require_asset_path
from robot_motion.rm65 import (
    DEFAULT_LEFT_ARM_QPOS,
    DEFAULT_RIGHT_ARM_QPOS,
    DualArmSceneProfile,
    build_dual_arm_scene_profile,
)
from sim.rm65.sapien_renderer import SapienArmHandRenderer
from project_paths import calibration_profiles_dir, default_dataset_root, teleop_records_root


DEFAULT_CAMERA_CONFIG_PATH = (
    DEFAULT_REALSENSE_DEVICE_CONFIG_PATH
)
DEFAULT_DATASET_ROOT = default_dataset_root()
DEFAULT_DMTAC_ROOT = Path(__file__).resolve().parents[2] / "hardware" / "DMTAC_gripper"
DEFAULT_POSITION_MAX_ACCELERATION_RAD_S2 = (7.0, 7.0, 9.0, 15.0, 15.0, 20.0)
DEFAULT_CALIBRATION_PROFILE_DIR = calibration_profiles_dir()
DEFAULT_TASKS_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "tasks"
    / "vive_rm65_real_teleop_tasks.json"
)
WRIST_DEPTH_CAMERA_KEYS = ("left_wrist", "right_wrist")
DEFAULT_WRIST_DEPTH_SAVE_FPS = 15
DEFAULT_WRIST_DEPTH_STREAM_FPS = 15
DEFAULT_WRIST_DEPTH_FPS = DEFAULT_WRIST_DEPTH_SAVE_FPS
DEFAULT_WRIST_DEPTH_MIN_M = 0.05
DEFAULT_WRIST_DEPTH_MAX_M = 0.5
DEFAULT_EPISODE_START_COUNTDOWN_SEC = 3.0
# MIGRATION(real-teleop, batch 6): recorder filenames and their default roots
# are owned by the extracted diagnostics component.  Importing the constants
# here deliberately retains the legacy module-level patch point used by local
# tooling while avoiding two independently maintained defaults.
from ..real_teleop.debug_recorders import (
    DEFAULT_ACTION_DEBUG_LOG_ROOT,
    DEFAULT_DEBUG_TIMING_LOG_ROOT,
    DEFAULT_IK_DEBUG_LOG_ROOT,
)


def _episode_start_countdown_remaining(now: float, deadline: float | None) -> float:
    if deadline is None:
        return 0.0
    return max(0.0, float(deadline) - float(now))


def _episode_start_countdown_ready_to_record(
    *,
    now: float,
    deadline: float | None,
    warmup_ready: bool,
) -> bool:
    return bool(
        warmup_ready
        and deadline is not None
        and _episode_start_countdown_remaining(now, deadline) <= 0.0
    )


HAND_BACKEND_INSPIRE = "inspire"
HAND_BACKEND_DMTAC_GRIPPER = "dmtac-gripper"
HAND_BACKEND_NONE = "none"
HAND_BACKEND_CHOICES = (
    HAND_BACKEND_INSPIRE,
    HAND_BACKEND_DMTAC_GRIPPER,
    HAND_BACKEND_NONE,
)
GRIPPER_POLICY_DEFAULTS = {
    "source_dim": 2,
    "output_dim": 0,
    "invert": False,
    "alpha": 0.95,
    "input_min": DEFAULT_GRIPPER_INPUT_MIN,
    "input_max": DEFAULT_GRIPPER_INPUT_MAX,
    "speed": 1.0,
    "torque": 0.2,
    "max_action": GRIPPER_ACTION_MAX,
    "send_threshold": 0.002,
    "command_hz": DEFAULT_DMTAC_GRIPPER_COMMAND_HZ,
    "continuous_grasp": True,
    "telemetry_hz": 10.0,
}
EPISODE_GUI_EVENTS = (
    "episode_start",
    "episode_stop",
)
POSE_SNAPSHOT_GUI_EVENT = "record_pose_snapshot"
VISUALIZATION_GUI_EVENTS = (
    "toggle_camera_vis",
    "toggle_tactile_vis",
)
VIVE_CALIBRATION_GUI_EVENTS = (
    "calibrate_vive_coordinate_profile",
    "save_vive_calibration_profile",
    "load_vive_calibration_profile",
)
MOVEMENT_MULTIPLE_DEFAULT = 1.0
MOVEMENT_MULTIPLE_GUI_MIN = 0.05
MOVEMENT_MULTIPLE_GUI_MAX = 10.0
MOVEMENT_MULTIPLE_GUI_STEP = 0.1
MOVEMENT_MULTIPLE_GUI_EVENTS = (
    "movement_multiple_down",
    "movement_multiple_up",
    "movement_multiple_reset",
    "movement_multiple_set",
)
VIRTUAL_DESK_SWEEP_CONVENTION_VERSION = 2
DEFAULT_REAL_TELEOP_MAX_TRACKER_TRANSLATION_JUMP_M = 0.45
DEFAULT_REAL_TELEOP_MAX_TRACKER_ROTATION_JUMP_DEG = 160.0
TRACKER_GUI_WARN_AGE_SEC = 0.25
TRACKER_GUI_STALE_AGE_SEC = 0.50
CAMERA_STARTUP_SELF_CHECK_DURATION_SEC = 3.0
CAMERA_STARTUP_SELF_CHECK_INTERVAL_SEC = 0.2
CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES = 3
CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC = DEFAULT_CAMERA_FRAME_MAX_AGE_SEC
CAMERA_START_STAGGER_SEC = 0.5
# MIGRATION(real-teleop, batch 1): wrist-depth freshness now lives in the
# dependency-light component below.  This legacy entrypoint deliberately
# re-exports the former names so parallel callers and external scripts do not
# need to change import paths during the staged refactor.  Remove this note
# only after the complete real-teleop extraction is accepted.
from ..real_teleop.wrist_depth_freshness import (
    DEFAULT_EPOCH_TIMESTAMP_STALE_SEC as WRIST_DEPTH_EPOCH_TIMESTAMP_STALE_SEC,
    DEFAULT_STALE_REPEAT_FAILURE_FRAMES as WRIST_DEPTH_STALE_REPEAT_FAILURE_FRAMES,
    WristDepthFreshnessMonitor,
    WristDepthFreshnessState,
    _epoch_depth_timestamp_age_sec,
    _frame_source_marker,
    _normalize_depth_signature_value,
    _wrist_depth_signature,
)



# MIGRATION(real-teleop, batch 3): re-export the observational timing
# component.  Do not add timing implementation back here while the staged
# extraction is in progress; callers may continue to import these legacy names.
from ..real_teleop.debug_timing import (
    DebugTimingModules,
    RealDebugTimingState,
    debug_timing_metadata as _debug_timing_metadata,
    debug_timing_modules as _debug_timing_modules,
    merge_control_debug_runtime_metadata as _merge_control_debug_runtime_metadata,
)


def _arm_execution_profile_metadata(args: argparse.Namespace) -> dict[str, object]:
    resolved = getattr(args, "arm_execution_config", None)
    if isinstance(resolved, RealArmExecutionConfig):
        return {"arm_execution_profile": resolved.to_metadata()}
    return {}


def _position_ik_iteration_metadata(args: argparse.Namespace) -> dict[str, object]:
    """Return position-only IK tuning settings for audit/config metadata."""
    position_mode = _arm_control_mode_arg(args) == "position"
    return {
        "position_ik_orientation_weight": (
            None
            if not position_mode
            else getattr(args, "position_ik_orientation_weight", None)
        ),
        "position_ik_solve_max_iterations": (
            int(getattr(args, "position_ik_solve_max_iterations", 2))
            if position_mode
            else None
        ),
        "position_ik_solve_min_iterations": (
            int(getattr(args, "position_ik_solve_min_iterations", 2))
            if position_mode
            else None
        ),
        "position_ik_damping": (
            float(getattr(args, "position_ik_damping", 0.03))
            if position_mode
            else None
        ),
        "position_ik_adaptive_iterations": (
            bool(getattr(args, "position_ik_adaptive_iterations", False))
            if position_mode
            else False
        ),
        "position_ik_iteration_reserve_ms": (
            float(getattr(args, "position_ik_iteration_reserve_ms", 5.0))
            if position_mode
            else None
        ),
        # This applies only while selecting a local IK candidate. It does not
        # raise the final qref execution velocity cap.
        "position_ik_local_step_clip_scale": (
            float(getattr(args, "position_ik_local_step_clip_scale", 1.0))
            if position_mode
            else None
        ),
        "position_joint_velocity_limit_profile": (
            str(getattr(args, "position_joint_velocity_limit_profile", "rm65"))
            if position_mode
            else None
        ),
        "position_inactive_enter_threshold_scale": (
            float(getattr(args, "position_inactive_enter_threshold_scale", 1.0))
            if position_mode
            else None
        ),
        "position_prefer_non_wrist_singular_branch": (
            bool(
                getattr(
                    args,
                    "position_prefer_non_wrist_singular_branch",
                    True,
                )
            )
            if position_mode
            else None
        ),
        "position_prefer_non_elbow_singular_branch": (
            bool(
                getattr(
                    args,
                    "position_prefer_non_elbow_singular_branch",
                    True,
                )
            )
            if position_mode
            else None
        ),
    }


def _attach_position_audit_sink(
    components: RealTeleopComponents,
    sink: PositionAuditFanout | None,
) -> None:
    """Attach one observational sink to the reference publisher and command workers."""

    publisher = components.position_reference_publisher
    if publisher is not None:
        publisher.set_audit_observer(sink)
    for backend in (
        components.left_command_backend,
        components.right_command_backend,
    ):
        setter = getattr(backend, "set_position_audit_sink", None)
        if callable(setter):
            setter(sink)


def _new_position_debug_recorder(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    thread_name: str,
    components: RealTeleopComponents | None = None,
    recorder_scope: str = "session",
    episode_index: int | None = None,
) -> PositionDebugRecorder:
    resolved = getattr(args, "arm_execution_config", None)
    execution = (
        resolved.to_metadata()
        if isinstance(resolved, RealArmExecutionConfig)
        else {"mode": _arm_control_mode_arg(args)}
    )
    canfd_runtime = _canfd_follow_runtime_config(args)
    collision_context = {
        "requested": bool(
            getattr(components, "collision_guard_requested", False)
        ),
        "enabled": bool(
            components is not None
            and components.dual_arm_retargeter is not None
            and components.collision_guard_setup_error is None
        ),
        "scene_profile": str(getattr(args, "collision_scene_profile", "")),
        "setup_error": (
            None
            if components is None
            else components.collision_guard_setup_error
        ),
    }
    context = {
        "recorder_scope": str(recorder_scope),
        "episode_index": None if episode_index is None else int(episode_index),
        "active_sides": list(_active_sides_from_args(args)),
        "arm_execution": execution,
        **_position_ik_iteration_metadata(args),
        "position_max_command_step_deg": (
            None
            if getattr(args, "position_max_command_step_deg", None) is None
            else float(args.position_max_command_step_deg)
        ),
        "position_target_velocity_filter_tau_sec": float(
            getattr(
                args,
                "position_target_velocity_filter_tau_sec",
                DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
            )
        ),
        "effective_transport": {
            "position_background_streaming_enabled": bool(
                canfd_runtime["position_background_streaming_enabled"]
            ),
            "position_background_streaming_hz": float(
                canfd_runtime["position_background_streaming_hz"]
            ),
            "canfd_follow": bool(_effective_arm_canfd_follow(args)),
            "canfd_trajectory_mode": int(canfd_runtime["canfd_trajectory_mode"]),
            "canfd_radio": int(canfd_runtime["canfd_radio"]),
        },
        "semantic_switches": {
            "position_post_ik_qpos_smoothing_alpha": float(
                _effective_position_post_ik_qpos_smoothing_alpha(args)
            ),
            "position_post_ik_qpos_smoothing_adaptive": bool(
                getattr(args, "position_post_ik_qpos_smoothing_adaptive", False)
            ),
            "position_post_ik_qpos_smoothing_fast_alpha": float(
                getattr(args, "position_post_ik_qpos_smoothing_fast_alpha", 0.45)
            ),
            "safe_debug": bool(getattr(args, "safe_debug", False)),
            "disable_hold": bool(getattr(args, "disable_hold", False)),
            "singularity_protection_enabled": bool(
                getattr(args, "arm_singularity_protection", True)
            ),
            "collision_guard": collision_context,
        },
    }
    config = PositionDebugRecorderConfig.from_control_debug_config(
        args.control_debug_config,
        output_dir=output_dir,
        thread_name=thread_name,
        context=context,
    )
    return PositionDebugRecorder(config, auto_start=True)


# MIGRATION(real-teleop, batch 3): the legacy helper remains an alias so
# existing audit tests and integrations retain their import path.
from ..real_teleop.debug_timing import (
    record_position_loop_timings as _record_position_loop_timings,
)


# MIGRATION(real-teleop, batch 2): the gesture implementation now lives in the
# dependency-light component below.  Re-export former names from this legacy
# entrypoint until the staged refactor is complete, so GUI/control collaborators
# and external scripts keep their existing imports.
from ..real_teleop.q6_hold_gesture import (
    Q6HoldRecalibrationGesture,
    Q6HoldRecalibrationGestureConfig,
)

# MIGRATION(real-teleop, batch 4): the temporary dual-arm limiter is now a
# real-teleop-specific component.  Keep these legacy re-exports so the
# established entrypoint and its callers retain their existing import path;
# do not alter safety limits in this compatibility bridge.
from ..real_teleop.temporary_dual_arm_safety import (
    TemporaryDualArmSafetyConfig,
    TemporaryDualArmSafetyLimiter,
)


def _compose_real_command_filter(
    *,
    temporary_safety_limiter: TemporaryDualArmSafetyLimiter | None,
    fix_arm_controller: FixArmCommandController | None,
    init_arm_controller: InitArmCommandController | None,
):
    if (
        temporary_safety_limiter is None
        and fix_arm_controller is None
        and init_arm_controller is None
    ):
        return None

    def filter_command(command: TeleopCommand, observation, now: float) -> TeleopCommand:
        filtered = command
        if init_arm_controller is not None:
            filtered = init_arm_controller.filter_command(filtered, observation, now)
        if temporary_safety_limiter is not None:
            filtered = temporary_safety_limiter.filter_command(filtered, observation, now)
        if fix_arm_controller is not None:
            filtered = fix_arm_controller.filter_command(filtered, observation, now)
        if temporary_safety_limiter is not None and (
            fix_arm_controller is not None or init_arm_controller is not None
        ):
            temporary_safety_limiter.sync_from_command(filtered)
        return filtered

    return filter_command


def _compose_real_fault_filter(
    *,
    fix_arm_controller: FixArmCommandController | None,
    init_arm_controller: InitArmCommandController | None,
):
    if fix_arm_controller is None and init_arm_controller is None:
        return None

    def filter_faults(faults, channels, now: float):
        filtered = list(faults)
        if fix_arm_controller is not None:
            filtered = fix_arm_controller.filter_faults(filtered, channels, now)
        if init_arm_controller is not None:
            filtered = init_arm_controller.filter_faults(filtered, channels, now)
        return filtered

    return filter_faults


def _update_init_arm_arrival_from_hardware(
    components: "RealTeleopComponents",
    *,
    hardware_state_by_side: Mapping[str, Mapping[str, object]],
    now: float,
) -> bool:
    """Feed measured TCP feedback into the display-only init arrival state."""

    controller = getattr(components, "init_arm_controller", None)
    if controller is None or not bool(getattr(controller, "active", False)):
        return False
    side = str(getattr(controller, "active_side", "") or "")
    state = hardware_state_by_side.get(side)
    backend = (
        components.left_state_backend
        if side == "left"
        else components.right_state_backend
        if side == "right"
        else None
    )
    invalid_reason = ""
    pose_base = None
    timestamp = float(now)
    valid = False
    if not isinstance(state, Mapping) or backend is None:
        invalid_reason = "hardware_state_unavailable"
    elif not bool(state.get("hardware_read_success")):
        invalid_reason = "hardware_read_unsuccessful"
    elif not bool(state.get("hardware_measurement_valid")):
        invalid_reason = "hardware_measurement_invalid"
    else:
        try:
            qpos = np.asarray(state.get("hardware_qpos"), dtype=float).reshape(6)
            measurement_time = float(
                state.get("hardware_measurement_timestamp", state.get("hardware_refresh_time"))
            )
            if not np.all(np.isfinite(qpos)) or not np.isfinite(measurement_time):
                raise ValueError("non-finite measured qpos/timestamp")
            age_sec = max(0.0, float(now) - measurement_time)
            if age_sec > 0.2:
                invalid_reason = f"hardware_measurement_stale:{age_sec:.3f}s"
            else:
                pose_base = backend.compute_tcp_pose(qpos=qpos)
                timestamp = measurement_time
                valid = True
        except Exception as exc:  # noqa: BLE001 - UI-only feedback must never affect control.
            if not invalid_reason:
                invalid_reason = f"actual_tcp_unavailable:{type(exc).__name__}"
    controller.observe_actual_feedback(
        pose_base,
        timestamp=timestamp,
        valid=valid,
        invalid_reason=invalid_reason,
    )
    return True


def _filter_manual_real_command(
    components: "RealTeleopComponents",
    command: TeleopCommand | None,
    observation,
    now: float,
) -> TeleopCommand | None:
    if command is None or observation is None:
        return command
    filter_command = _compose_real_command_filter(
        temporary_safety_limiter=components.temporary_safety_limiter,
        fix_arm_controller=components.fix_arm_controller,
        init_arm_controller=getattr(components, "init_arm_controller", None),
    )
    if filter_command is None:
        return command
    return filter_command(command, observation, now)


def _fix_arm_required_effectors_by_side(
    *,
    active_sides: Sequence[str],
    left_hand_control_via_rig: bool,
    right_hand_control_via_rig: bool,
    left_gripper_retargeter: ScalarToGripperRetargeter | None,
    right_gripper_retargeter: ScalarToGripperRetargeter | None,
    left_hand_backend: InspireTacHandBackend | DMTacGripperBackend | None,
    right_hand_backend: InspireTacHandBackend | DMTacGripperBackend | None,
) -> dict[str, tuple[str, ...]]:
    required: dict[str, tuple[str, ...]] = {}
    for side in active_sides:
        names = [f"{side}_arm"]
        if side == "left":
            hand_controlled = bool(
                left_hand_control_via_rig
                or (left_gripper_retargeter is not None and left_hand_backend is not None)
            )
        else:
            hand_controlled = bool(
                right_hand_control_via_rig
                or (right_gripper_retargeter is not None and right_hand_backend is not None)
            )
        if hand_controlled:
            names.append(f"{side}_hand")
        required[str(side)] = tuple(names)
    return required












def _resolve_movement_multiple_args(
    args: argparse.Namespace,
    *,
    raw_arg_set: set[str] | None = None,
) -> float:
    raw_arg_set = raw_arg_set or set()
    movement_multiple = getattr(args, "movement_multiple", None)
    if movement_multiple is None:
        movement_multiple = getattr(args, "translation_scale", MOVEMENT_MULTIPLE_DEFAULT)
    value = _movement_multiple_arg(str(movement_multiple))
    if "--movement-multiple" in raw_arg_set and "--translation-scale" in raw_arg_set:
        _log(
            "MovementMultiple",
            "--movement-multiple overrides legacy --translation-scale.",
        )
    setattr(args, "movement_multiple", value)
    setattr(args, "translation_scale", value)
    return value


# Compatibility bridge: retain established workspace helper names while the
# extracted real-teleop workspace policy owns their behavior.
from ..real_teleop.workspace import (
    apply_workspace_config as _apply_workspace_config,
    apply_workspace_overrides as _apply_workspace_overrides,
    arm_workspace_config_path as _arm_workspace_config_path,
    parse_vector3_arg as _parse_vector3_arg,
    resolve_workspace_override_args as _resolve_workspace_override_args,
    workspace_override_for_side as _workspace_override_for_side,
)










def _hand_backend_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "hand_backend", HAND_BACKEND_INSPIRE) or HAND_BACKEND_INSPIRE)
    if mode not in HAND_BACKEND_CHOICES:
        raise ValueError(f"Unsupported hand backend {mode!r}; expected one of {HAND_BACKEND_CHOICES}.")
    return mode


def _hand_backend_mode_for_side(args: argparse.Namespace, side: str) -> str:
    normalized_side = str(side).strip().lower()
    if normalized_side not in {"left", "right"}:
        raise ValueError(f"Unsupported hand backend side {side!r}.")
    override = getattr(args, f"{normalized_side}_hand_backend", None)
    mode = _hand_backend_mode(args) if override in (None, "") else str(override)
    if mode not in HAND_BACKEND_CHOICES:
        raise ValueError(
            f"Unsupported {normalized_side} hand backend {mode!r}; "
            f"expected one of {HAND_BACKEND_CHOICES}."
        )
    return mode


def _dmtac_sides(args: argparse.Namespace, active_sides: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        str(side)
        for side in active_sides
        if _hand_backend_mode_for_side(args, str(side)) == HAND_BACKEND_DMTAC_GRIPPER
    )


def _profile_value(profile: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in profile:
            return profile[key]
    return None


def _strip_side_gripper_overrides(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in profile.items()
        if str(key) not in {"left", "right", "sides"}
    }


def _side_gripper_overrides_from_profile(profile: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(profile, Mapping):
        return {}
    by_side: dict[str, dict[str, Any]] = {}
    sides = profile.get("sides")
    if isinstance(sides, Mapping):
        for side in ("left", "right"):
            item = sides.get(side)
            if isinstance(item, Mapping):
                by_side[side] = dict(item)
    for side in ("left", "right"):
        item = profile.get(side)
        if isinstance(item, Mapping):
            merged = dict(by_side.get(side, {}))
            merged.update(dict(item))
            by_side[side] = merged
    return by_side


def _merge_gripper_profile(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = _strip_side_gripper_overrides(base or {})
    side_overrides = _side_gripper_overrides_from_profile(base)
    if isinstance(override, Mapping):
        merged.update(_strip_side_gripper_overrides(override))
        for side, item in _side_gripper_overrides_from_profile(override).items():
            side_merged = dict(side_overrides.get(side, {}))
            side_merged.update(item)
            side_overrides[side] = side_merged
    for side, item in side_overrides.items():
        merged[side] = dict(item)
    return merged


def _load_gripper_task_profile(args: argparse.Namespace) -> dict[str, Any]:
    path_text = str(getattr(args, "gripper_task_profile", "") or "").strip()
    if not path_text:
        return {}
    path = Path(path_text).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Gripper task profile {path} must contain a JSON object.")
    base_gripper = payload.get("gripper")
    if base_gripper is not None and not isinstance(base_gripper, Mapping):
        raise ValueError(f"Profile {path} field 'gripper' must be an object.")
    selected: Mapping[str, Any] | None = None
    tasks = payload.get("tasks")
    if isinstance(tasks, Mapping):
        task_keys = (
            str(getattr(args, "gripper_task", "") or ""),
            str(getattr(args, "task_id", "") or ""),
            str(getattr(args, "task_name", "") or ""),
            str(getattr(args, "task", "") or ""),
        )
        for key in task_keys:
            item = tasks.get(key)
            if isinstance(item, Mapping):
                selected = item
                setattr(args, "_resolved_gripper_task", key)
                break
    if selected is not None:
        gripper = selected.get("gripper")
        if gripper is not None and not isinstance(gripper, Mapping):
            raise ValueError(f"Profile {path} selected task field 'gripper' must be an object.")
        return _merge_gripper_profile(
            base_gripper if isinstance(base_gripper, Mapping) else None,
            gripper if isinstance(gripper, Mapping) else selected,
        )
    if isinstance(base_gripper, Mapping):
        return _merge_gripper_profile(base_gripper, None)
    if "gripper" in payload:
        raise ValueError(f"Profile {path} field 'gripper' must be an object.")
    return _merge_gripper_profile(payload, None)


def _resolved_gripper_policy(args: argparse.Namespace) -> dict[str, Any]:
    cached = getattr(args, "_resolved_gripper_policy", None)
    if isinstance(cached, dict):
        return dict(cached)

    profile = _load_gripper_task_profile(args)
    policy: dict[str, Any] = dict(GRIPPER_POLICY_DEFAULTS)
    profile_keys = {
        "source_dim": ("source_dim", "q6_dim"),
        "output_dim": ("output_dim", "gripper_slot"),
        "invert": ("invert",),
        "alpha": ("alpha", "command_alpha"),
        "input_min": ("input_min", "q6_min", "source_min"),
        "input_max": ("input_max", "q6_max", "source_max"),
        "speed": ("speed",),
        "torque": ("torque", "force"),
        "max_action": ("max_action", "max_close"),
        "send_threshold": ("send_threshold",),
        "command_hz": ("command_hz", "max_command_hz"),
        "continuous_grasp": ("continuous_grasp", "continuous"),
        "telemetry_hz": ("telemetry_hz",),
        "tactile_fields": ("tactile_fields", "dmtac_tactile_fields"),
        "tactile_compute_profile": ("tactile_compute_profile", "dmtac_tactile_compute_profile"),
    }
    for key, aliases in profile_keys.items():
        value = _profile_value(profile, *aliases)
        if value is not None:
            policy[key] = value

    cli_overrides = {
        "source_dim": getattr(args, "gripper_source_dim", None),
        "output_dim": getattr(args, "gripper_output_dim", None),
        "invert": getattr(args, "gripper_invert", None),
        "alpha": getattr(args, "gripper_alpha", None),
        "input_min": getattr(args, "gripper_input_min", None),
        "input_max": getattr(args, "gripper_input_max", None),
        "speed": getattr(args, "gripper_speed", None),
        "torque": getattr(args, "gripper_torque", None),
        "max_action": getattr(args, "gripper_max_action", None),
        "send_threshold": getattr(args, "gripper_send_threshold", None),
        "command_hz": getattr(args, "gripper_command_hz", None),
        "continuous_grasp": getattr(args, "gripper_continuous_grasp", None),
        "telemetry_hz": getattr(args, "gripper_telemetry_hz", None),
    }
    for key, value in cli_overrides.items():
        if value is not None:
            policy[key] = value

    policy["source_dim"] = int(policy["source_dim"])
    policy["output_dim"] = int(policy["output_dim"])
    policy["invert"] = bool(policy["invert"])
    policy["alpha"] = float(policy["alpha"])
    policy["input_min"] = float(policy["input_min"])
    policy["input_max"] = float(policy["input_max"])
    policy["speed"] = float(policy["speed"])
    policy["torque"] = float(policy["torque"])
    # Old task profiles used 0.9/0.95 as hardware workarounds.  v2.2 stores
    # the canonical normalized DM-Tac range, so force the runtime to 1.0.
    policy["max_action"] = float(GRIPPER_ACTION_MAX)
    policy["send_threshold"] = float(policy["send_threshold"])
    policy["command_hz"] = float(policy["command_hz"])
    policy["continuous_grasp"] = bool(policy["continuous_grasp"])
    policy["telemetry_hz"] = float(policy["telemetry_hz"])
    if policy.get("tactile_fields") is not None:
        policy["tactile_fields"] = normalize_dmtac_tactile_fields(policy["tactile_fields"])
    if policy.get("tactile_compute_profile") is not None:
        policy["tactile_compute_profile"] = str(policy["tactile_compute_profile"])
    if not 0 <= int(policy["source_dim"]) < 6:
        raise ValueError("gripper source_dim must be in [0, 5].")
    if int(policy["output_dim"]) != 0:
        raise ValueError("dmtac-gripper output_dim/gripper_slot must be 0.")
    if not 0.0 < float(policy["alpha"]) <= 1.0:
        raise ValueError("gripper alpha must be in (0, 1].")
    if not 0.0 <= float(policy["input_min"]) < float(policy["input_max"]) <= 1.0:
        raise ValueError("gripper input_min/input_max must satisfy 0 <= input_min < input_max <= 1.")
    if float(policy["send_threshold"]) < 0.0:
        raise ValueError("gripper send_threshold must be >= 0.")
    if float(policy["command_hz"]) <= 0.0:
        raise ValueError("gripper command_hz must be > 0.")
    if float(policy["telemetry_hz"]) <= 0.0:
        raise ValueError("gripper telemetry_hz must be > 0.")
    setattr(args, "_resolved_gripper_policy", dict(policy))
    return dict(policy)


def _gripper_side_override(profile: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    item = profile.get(side)
    if isinstance(item, Mapping):
        return item
    sides = profile.get("sides")
    if isinstance(sides, Mapping):
        item = sides.get(side)
        if isinstance(item, Mapping):
            return item
    return {}


def _resolved_gripper_policy_for_side(args: argparse.Namespace, side: str) -> dict[str, Any]:
    cached = getattr(args, "_resolved_gripper_policy_by_side", None)
    if isinstance(cached, dict) and side in cached and isinstance(cached[side], dict):
        return dict(cached[side])

    base_policy = _resolved_gripper_policy(args)
    profile = _load_gripper_task_profile(args)
    side_override = _gripper_side_override(profile, side)
    policy = dict(base_policy)
    profile_keys = {
        "source_dim": ("source_dim", "q6_dim"),
        "output_dim": ("output_dim", "gripper_slot"),
        "invert": ("invert",),
        "alpha": ("alpha", "command_alpha"),
        "input_min": ("input_min", "q6_min", "source_min"),
        "input_max": ("input_max", "q6_max", "source_max"),
        "speed": ("speed",),
        "torque": ("torque", "force"),
        "max_action": ("max_action", "max_close"),
        "send_threshold": ("send_threshold",),
        "command_hz": ("command_hz", "max_command_hz"),
        "continuous_grasp": ("continuous_grasp", "continuous"),
        "telemetry_hz": ("telemetry_hz",),
        "tactile_fields": ("tactile_fields", "dmtac_tactile_fields"),
        "tactile_compute_profile": ("tactile_compute_profile", "dmtac_tactile_compute_profile"),
    }
    for key, aliases in profile_keys.items():
        value = _profile_value(side_override, *aliases)
        if value is not None:
            policy[key] = value
    policy["source_dim"] = int(policy["source_dim"])
    policy["output_dim"] = int(policy["output_dim"])
    policy["invert"] = bool(policy["invert"])
    policy["alpha"] = float(policy["alpha"])
    policy["input_min"] = float(policy["input_min"])
    policy["input_max"] = float(policy["input_max"])
    policy["speed"] = float(policy["speed"])
    policy["torque"] = float(policy["torque"])
    policy["max_action"] = float(GRIPPER_ACTION_MAX)
    policy["send_threshold"] = float(policy["send_threshold"])
    policy["command_hz"] = float(policy["command_hz"])
    policy["continuous_grasp"] = bool(policy["continuous_grasp"])
    policy["telemetry_hz"] = float(policy["telemetry_hz"])
    if policy.get("tactile_fields") is not None:
        policy["tactile_fields"] = normalize_dmtac_tactile_fields(policy["tactile_fields"])
    if policy.get("tactile_compute_profile") is not None:
        policy["tactile_compute_profile"] = str(policy["tactile_compute_profile"])
    if not 0 <= int(policy["source_dim"]) < 6:
        raise ValueError("gripper source_dim must be in [0, 5].")
    if int(policy["output_dim"]) != 0:
        raise ValueError("dmtac-gripper output_dim/gripper_slot must be 0.")
    if not 0.0 < float(policy["alpha"]) <= 1.0:
        raise ValueError("gripper alpha must be in (0, 1].")
    if not 0.0 <= float(policy["input_min"]) < float(policy["input_max"]) <= 1.0:
        raise ValueError("gripper input_min/input_max must satisfy 0 <= input_min < input_max <= 1.")
    if not 0.0 <= float(policy["speed"]) <= 1.0:
        raise ValueError("gripper speed must be in [0, 1].")
    if not 0.0 <= float(policy["torque"]) <= 1.0:
        raise ValueError("gripper force/torque must be in [0, 1].")
    by_side = dict(cached) if isinstance(cached, dict) else {}
    by_side[side] = dict(policy)
    setattr(args, "_resolved_gripper_policy_by_side", by_side)
    return dict(policy)


def _resolved_gripper_policies_by_side(
    args: argparse.Namespace,
    active_sides: Sequence[str],
) -> dict[str, dict[str, Any]]:
    return {
        str(side): _resolved_gripper_policy_for_side(args, str(side))
        for side in active_sides
        if str(side) in {"left", "right"}
    }


def _clear_gripper_policy_cache(args: argparse.Namespace) -> None:
    for name in (
        "_resolved_gripper_policy",
        "_resolved_gripper_policy_by_side",
        "_dmtac_backend_configs",
    ):
        if hasattr(args, name):
            delattr(args, name)


def _apply_default_q6_ports(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> None:
    args.q6_config_loaded_ports = {}
    if not _dmtac_sides(args, _active_sides_from_args(args)):
        return
    if str(getattr(args, "gripper_input_source", "q6")) != "q6":
        return
    if not bool(getattr(args, "q6_default_config", True)):
        return
    active_sides = tuple(str(side) for side in _active_sides_from_args(args))
    missing_sides = tuple(
        side
        for side in active_sides
        if side in {"left", "right"} and not str(getattr(args, f"{side}_hand_port", "") or "")
    )
    if not missing_sides:
        return
    try:
        loaded_ports = load_latest_q6_ports(
            active_sides=missing_sides,
            config=str(getattr(args, "q6_config", "latest") or "latest"),
            allow_missing=not bool(getattr(args, "q6_config_required", False)),
        )
    except Exception as exc:  # noqa: BLE001
        message = f"Failed to load hand Q6 assignment {getattr(args, 'q6_config', 'latest')!r}: {exc}"
        if parser is not None:
            parser.error(message)
        raise RuntimeError(message) from exc
    args.q6_config_loaded_ports = dict(loaded_ports)
    for side, port in loaded_ports.items():
        if side in missing_sides and port:
            setattr(args, f"{side}_hand_port", str(port))


def _dmtac_tactile_enabled(args: argparse.Namespace) -> bool:
    explicit = getattr(args, "dmtac_tactile", None)
    if explicit is not None:
        return bool(explicit)
    return False


def _run_gui_device_preflight(
    args: argparse.Namespace,
    *,
    active_sides: tuple[str, ...],
    parser: argparse.ArgumentParser,
) -> None:
    if not bool(getattr(args, "gui_requested", False)):
        return
    if not bool(getattr(args, "device_preflight", True)):
        return
    dmtac_sides = _dmtac_sides(args, active_sides)
    require_dmtac_stack = bool(dmtac_sides)
    require_q6 = require_dmtac_stack and str(getattr(args, "gripper_input_source", "q6")) == "q6"
    if (
        require_dmtac_stack
        and bool(getattr(args, "visualize_tactile", False))
        and getattr(args, "dmtac_tactile", None) is None
    ):
        args.dmtac_tactile = True
    require_tactile = bool(getattr(args, "device_preflight_require_tactile", True)) and _dmtac_tactile_enabled(args)
    if getattr(args, "dmtac_tactile", None) is False:
        require_tactile = False

    options = DevicePreflightOptions(
        active_sides=active_sides,
        camera_config=str(getattr(args, "camera_config", DEFAULT_CAMERA_CONFIG_PATH)),
        require_camera=not bool(getattr(args, "no_camera", False)),
        require_q6=require_q6,
        require_dmtac_gripper=require_dmtac_stack,
        require_dmtac_tactile=bool(require_dmtac_stack and require_tactile),
        auto_configure=bool(getattr(args, "device_auto_configure", True)),
        force_reconfigure=bool(getattr(args, "device_preflight_force_reconfigure", False)),
        assume_yes=bool(getattr(args, "device_preflight_yes", False)),
        fix_permissions=bool(getattr(args, "device_fix_permissions", False)),
        probe_all_serial=bool(getattr(args, "device_preflight_probe_all_serial", False)),
        q6_config=str(getattr(args, "q6_config", "latest") or "latest"),
        dmtac_config=str(getattr(args, "dmtac_config", "latest") or "latest"),
        dmtac_tactile_backend=str(getattr(args, "dmtac_tactile_backend", "cpu") or "cpu"),
    )
    try:
        result = prepare_devices_for_teleop(options)
    except DevicePreflightError as exc:
        parser.error(f"device preflight failed: {exc}")
    if not bool(getattr(result, "ok", True)):
        parser.error("device preflight permission fix failed; rerun the permission script with sudo.")
    for message in result.messages:
        for line in str(message).splitlines():
            _log("DevicePreflight", line)
    if getattr(result, "dmtac_assignment", None) is not None:
        args._dmtac_preflight_assignment = result.dmtac_assignment
    if result.q6_ports:
        args.q6_config_loaded_ports = dict(result.q6_ports)
        for side, port in result.q6_ports.items():
            if side in active_sides:
                setattr(args, f"{side}_hand_port", str(port))


def _resolve_gripper_input_source(args: argparse.Namespace, active_sides: tuple[str, ...]) -> None:
    """Prefer usable Q6 input; fall back to the calibrated angle triggers."""
    if not _dmtac_sides(args, active_sides):
        args.gripper_input_source = "none"
        return
    if bool(getattr(args, "gripper_angle_trigger", False)):
        args.gripper_input_source = "angle_trigger"
        return
    try:
        from hardware.teleop_hand.q6_assignment import scan_q6_devices

        scan = scan_q6_devices(
            probe_all_serial=True,
            show_raw_lines=False,
        )
        expected = len(active_sides)
        if len(scan.get("q6_devices") or ()) >= expected:
            args.gripper_input_source = "q6"
            return
        if len(scan.get("q6_devices") or ()):
            raise ValueError(
                f"Detected only {len(scan.get('q6_devices') or ())} Q6 device(s), expected {expected}."
            )
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - Q6 probing is optional when triggers are available.
        _log("GripperInput", f"Q6 probe unavailable; trying angle trigger: {type(exc).__name__}: {exc}")
    args.gripper_input_source = "angle_trigger"


def _argv_has_option(raw_argv: Sequence[str], option: str) -> bool:
    return any(str(item) == option or str(item).startswith(f"{option}=") for item in raw_argv)


def _apply_real_loop_timing_defaults(
    args: argparse.Namespace,
    *,
    raw_argv: Sequence[str],
) -> None:
    hz = max(1e-6, _effective_outer_control_hz(args))
    if not _argv_has_option(raw_argv, "--minimum-control-hz"):
        setattr(args, "minimum_control_hz", max(1.0, min(15.0, 0.9 * hz)))
        setattr(args, "minimum_control_hz_auto", True)
    else:
        setattr(args, "minimum_control_hz_auto", False)

    if not _argv_has_option(raw_argv, "--arm-step-dt-max-scale"):
        arm_control_hz = max(1e-6, _effective_arm_control_hz(args))
        required_scale = 1.1 * arm_control_hz / hz
        setattr(
            args,
            "arm_step_dt_max_scale",
            max(float(getattr(args, "arm_step_dt_max_scale", 3.0)), required_scale),
        )
        setattr(args, "arm_step_dt_max_scale_auto", True)
    else:
        setattr(args, "arm_step_dt_max_scale_auto", False)


def _impedance_bimanual_ik_safety_profile() -> IKSafetyProfile:
    """Return force-control-only bounds for degraded coupled-IK candidates.

    These are deliberately *not* a Cartesian output rate limiter.  Applying
    3 mm / 0.5 deg after IK at the outer target rate creates a multi-second
    target chase.  Here the values only constrain a local candidate that would
    otherwise regress from the currently safe coupled solution; normal target
    tracking still uses the coupled solver and execution joint-rate limits.
    """

    profiles = load_ik_safety_profiles()
    try:
        base_profile = profiles["dual_arm_teleop"]
    except KeyError as exc:  # pragma: no cover - guarded by shipped profiles
        raise RuntimeError("Missing required dual_arm_teleop IK safety profile.") from exc
    bounds = {
        "max_position_worsen_m": 0.003,
        "max_orientation_worsen_deg": 0.5,
        "dynamic_worsen_enabled": False,
        "scope": "impedance_bimanual_degraded_ik_candidate_only",
    }
    return replace(
        base_profile,
        name=f"{base_profile.name}_impedance_local_bounds",
        local_step_max_position_worsen_m=float(bounds["max_position_worsen_m"]),
        local_step_max_orientation_worsen_deg=float(bounds["max_orientation_worsen_deg"]),
        local_step_dynamic_worsen_enabled=False,
        metadata={
            **dict(base_profile.metadata),
            "impedance_bimanual_local_step_bounds": bounds,
        },
    )


def _source_arm_ip_fields_for_active_sides(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> dict[str, str]:
    fields: dict[str, str] = {}
    for source_name, (logical_side, _arm_ip) in _resolve_logical_side_assignments(args).items():
        if logical_side in active_sides:
            fields[str(source_name).replace("_arm_ip", "")] = str(source_name)
    return fields


def _apply_default_arm_ips(
    args: argparse.Namespace,
    *,
    active_sides: tuple[str, ...],
) -> None:
    setattr(args, "arm_ip_config_loaded_sides", [])
    if not bool(getattr(args, "arm_ip_default_config", True)):
        return
    source_fields = _source_arm_ip_fields_for_active_sides(args, active_sides)
    missing_sources = [
        source_side
        for source_side, field in source_fields.items()
        if not str(getattr(args, field, "") or "").strip()
    ]
    if not missing_sources:
        return
    try:
        assignment = load_rm65_arm_ip_assignment(str(getattr(args, "arm_ip_config", "latest") or "latest"))
    except FileNotFoundError:
        return
    loaded = arm_ips_for_sides(assignment, sides=missing_sources)
    loaded_sides: list[str] = []
    for source_side, arm_ip in loaded.items():
        field = source_fields.get(source_side)
        if field and not str(getattr(args, field, "") or "").strip():
            setattr(args, field, str(arm_ip))
            loaded_sides.append(str(source_side))
    setattr(args, "arm_ip_config_loaded_sides", loaded_sides)


def _save_explicit_arm_ips_latest(
    args: argparse.Namespace,
    *,
    raw_argv: Sequence[str],
) -> None:
    if not bool(getattr(args, "arm_ip_save_latest", True)):
        return
    explicit_left = _argv_has_option(raw_argv, "--left-arm-ip") and bool(str(args.left_arm_ip or "").strip())
    explicit_right = _argv_has_option(raw_argv, "--right-arm-ip") and bool(str(args.right_arm_ip or "").strip())
    if not (explicit_left or explicit_right):
        return
    try:
        base = load_rm65_arm_ip_assignment(str(getattr(args, "arm_ip_config", "latest") or "latest"))
    except FileNotFoundError:
        base = None
    assignment = build_rm65_arm_ip_assignment(
        base=base,
        left_arm_ip=str(args.left_arm_ip).strip() if str(args.left_arm_ip or "").strip() else None,
        right_arm_ip=str(args.right_arm_ip).strip() if str(args.right_arm_ip or "").strip() else None,
    )
    saved = save_rm65_arm_ip_assignment_for_config(
        assignment,
        config=str(getattr(args, "arm_ip_config", "latest") or "latest"),
    )
    setattr(args, "arm_ip_config_saved_path", str(saved))


def _validate_required_arm_ips(args: argparse.Namespace, active_sides: tuple[str, ...]) -> None:
    logical_ip_by_side: dict[str, tuple[str, str]] = {}
    inactive_config: list[str] = []
    for source_name, (logical_side, arm_ip) in _resolve_logical_side_assignments(args).items():
        if not arm_ip:
            continue
        if logical_side not in active_sides:
            inactive_config.append(f"{source_name}->{logical_side}")
            continue
        logical_ip_by_side[logical_side] = (source_name, arm_ip)
    missing = [side for side in active_sides if side not in logical_ip_by_side]
    if missing:
        raise ValueError(
            "Real teleop requires an RM65 IP for every active logical side; "
            f"missing={missing} active_sides={list(active_sides)} "
            f"swap_arm_sides={bool(args.swap_arm_sides)}. "
            "Check that LEFT_ARM_IP/RIGHT_ARM_IP are exported in this shell, or pass "
            "--left-arm-ip/--right-arm-ip literal IPs."
        )
    if inactive_config:
        raise ValueError(
            "Inactive logical side received an arm IP; remove it or include that side "
            f"in --active-sides. inactive={inactive_config}"
        )


def _manual_dmtac_config_for_side(
    args: argparse.Namespace,
    side: str,
    policy: Mapping[str, Any],
) -> DMTacGripperBackendConfig | None:
    port = str(getattr(args, f"{side}_gripper_port", "") or "").strip()
    if not port:
        return None
    return DMTacGripperBackendConfig(
        side=side,
        effector_name=f"{side}_hand",
        gripper_port=port,
        gripper_id=int(getattr(args, f"{side}_gripper_id", 1) or 1),
        command_dim=int(policy["output_dim"]),
        speed=float(policy["speed"]),
        torque=float(policy["torque"]),
        max_action=float(policy["max_action"]),
        send_threshold=float(policy["send_threshold"]),
        command_hz=float(policy["command_hz"]),
        continuous_grasp=bool(policy["continuous_grasp"]),
        enable_telemetry=True,
        telemetry_hz=float(policy["telemetry_hz"]),
        connect_tactile=False,
        tactile_ids=(),
    )


def _apply_gripper_policy_to_dmtac_config(
    config: DMTacGripperBackendConfig,
    *,
    args: argparse.Namespace,
    policy: Mapping[str, Any],
) -> DMTacGripperBackendConfig:
    config.speed = float(policy["speed"])
    config.torque = float(policy["torque"])
    config.max_action = float(policy["max_action"])
    config.command_dim = int(policy["output_dim"])
    config.send_threshold = float(policy["send_threshold"])
    config.command_hz = float(policy["command_hz"])
    config.continuous_grasp = bool(policy["continuous_grasp"])
    config.telemetry_hz = float(policy["telemetry_hz"])
    tactile_timeout_ms = getattr(args, "dmtac_tactile_timeout_ms", None)
    if tactile_timeout_ms is not None:
        config.tactile_timeout_ms = int(tactile_timeout_ms)
    tactile_wait_for_new = getattr(args, "dmtac_tactile_wait_for_new", None)
    if tactile_wait_for_new is not None:
        config.tactile_wait_for_new = bool(tactile_wait_for_new)
    tactile_read_hz = getattr(args, "dmtac_tactile_read_hz", None)
    if tactile_read_hz is not None:
        config.tactile_read_hz = float(tactile_read_hz)
    tactile_process_read = getattr(args, "dmtac_tactile_process_read", None)
    if tactile_process_read is not None:
        config.tactile_process_read = bool(tactile_process_read)
    tactile_worker_cpus = getattr(args, "dmtac_tactile_worker_cpus", None)
    if tactile_worker_cpus is not None:
        config.tactile_process_cpu_affinity = tuple(int(cpu) for cpu in tactile_worker_cpus)
    tactile_worker_nice = getattr(args, "dmtac_tactile_worker_nice", None)
    if tactile_worker_nice is not None:
        config.tactile_process_nice = int(tactile_worker_nice)
    tactile_worker_queue_size = getattr(args, "dmtac_tactile_worker_queue_size", None)
    if tactile_worker_queue_size is not None:
        config.tactile_process_queue_size = int(tactile_worker_queue_size)
    tactile_fields = getattr(args, "dmtac_tactile_fields", None)
    if tactile_fields is None:
        tactile_fields = policy.get("tactile_fields")
    if tactile_fields is not None:
        config.tactile_fields = normalize_dmtac_tactile_fields(tactile_fields)
    tactile_compute_profile = getattr(args, "dmtac_tactile_compute_profile", None)
    if tactile_compute_profile:
        config.tactile_compute_profile = str(tactile_compute_profile)
    elif policy.get("tactile_compute_profile"):
        config.tactile_compute_profile = str(policy["tactile_compute_profile"])
    tactile_max_fps = getattr(args, "dmtac_tactile_max_fps", None)
    if tactile_max_fps is not None:
        config.tactile_max_fps = int(tactile_max_fps)
    tactile_warmup_frames = getattr(args, "dmtac_tactile_warmup_frames", None)
    if tactile_warmup_frames is not None:
        config.tactile_warmup_frames = int(tactile_warmup_frames)
    tactile_denoise = getattr(args, "dmtac_tactile_denoise", None)
    if tactile_denoise is not None:
        config.tactile_denoise = bool(tactile_denoise)
    tactile_flat_vector = getattr(args, "dmtac_tactile_flat_vector", None)
    if tactile_flat_vector is not None:
        config.tactile_store_flat_vector = bool(tactile_flat_vector)
    tactile_parallel_read = getattr(args, "dmtac_tactile_parallel_read", None)
    if tactile_parallel_read is not None:
        config.tactile_parallel_read = bool(tactile_parallel_read)
    elif bool(getattr(args, "dmtac_tactile_sequential_read", False)):
        config.tactile_parallel_read = False
    tactile_global_read_lock = getattr(args, "dmtac_tactile_global_read_lock", None)
    if tactile_global_read_lock is not None:
        config.tactile_global_read_lock = bool(tactile_global_read_lock)
    if not _dmtac_tactile_enabled(args):
        config.connect_tactile = False
        config.tactile_ids = ()
    config.enable_telemetry = bool(config.connect_hardware or config.connect_tactile)
    return config


def _dmtac_backend_configs(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> dict[str, DMTacGripperBackendConfig]:
    cached = getattr(args, "_dmtac_backend_configs", None)
    if isinstance(cached, dict):
        return dict(cached)
    dmtac_sides = _dmtac_sides(args, active_sides)
    if not dmtac_sides:
        setattr(args, "_dmtac_backend_configs", {})
        setattr(args, "dmtac_config_loaded_sides", [])
        return {}

    policy = _resolved_gripper_policy(args)
    policies_by_side = _resolved_gripper_policies_by_side(args, active_sides)
    configs: dict[str, DMTacGripperBackendConfig] = {}
    if bool(getattr(args, "dmtac_default_config", True)):
        preflight_assignment = getattr(args, "_dmtac_preflight_assignment", None)
        try:
            assignment = (
                dict(preflight_assignment)
                if isinstance(preflight_assignment, Mapping)
                else load_dmtac_hardware_assignment(
                    str(getattr(args, "dmtac_config", "latest") or "latest"),
                    validate_current_usb=bool(getattr(args, "dmtac_validate_current_usb", True)),
                    allow_missing=bool(getattr(args, "dmtac_allow_missing", False)),
                    assume_yes_for_extra=bool(getattr(args, "dmtac_assume_yes_for_extra", True)),
                )
            )
            for side in dmtac_sides:
                side_policy = policies_by_side.get(side, policy)
                configs.update(
                    dmtac_backend_configs_from_assignment(
                        assignment,
                        active_sides=(side,),
                        tactile_backend=str(getattr(args, "dmtac_tactile_backend", "cpu") or "cpu"),
                        speed=float(side_policy["speed"]),
                        torque=float(side_policy["torque"]),
                        max_action=float(side_policy["max_action"]),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            if bool(getattr(args, "dmtac_config_required", False)):
                raise RuntimeError(
                    f"Failed to load DMTAC hardware assignment {getattr(args, 'dmtac_config', 'latest')!r}: {exc}"
                ) from exc
            _log("DMTAC", f"WARNING: could not load DMTAC hardware assignment: {exc}")

    for side in dmtac_sides:
        manual = _manual_dmtac_config_for_side(args, side, policies_by_side.get(side, policy))
        if manual is not None:
            configs[side] = manual

    resolved = {
        side: _apply_gripper_policy_to_dmtac_config(
            config,
            args=args,
            policy=policies_by_side.get(side, policy),
        )
        for side, config in configs.items()
        if side in dmtac_sides and (config.connect_hardware or config.connect_tactile)
    }
    setattr(args, "_dmtac_backend_configs", dict(resolved))
    setattr(args, "dmtac_config_loaded_sides", sorted(resolved))
    return dict(resolved)


# Compatibility bridge: callers retain the established entrypoint helper names
# while the pure real-teleop profile implementation owns their behavior.
from ..real_teleop.control_profile import (
    arm_control_mode as _arm_control_mode_arg,
    canfd_follow_runtime_config as _canfd_follow_runtime_config,
    effective_arm_canfd_follow as _effective_arm_canfd_follow,
    effective_arm_control_hz as _effective_arm_control_hz,
    effective_arm_follow_qpos_alpha as _effective_arm_follow_qpos_alpha,
    effective_dual_arm_ik_hz as _effective_dual_arm_ik_hz,
    effective_dual_arm_temporary_safety as _effective_dual_arm_temporary_safety,
    effective_outer_control_hz as _effective_outer_control_hz,
    effective_position_post_ik_qpos_smoothing_alpha as _effective_position_post_ik_qpos_smoothing_alpha,
    effective_position_reference_hz as _effective_position_reference_hz,
    position_retargeter_stability_switches as _position_retargeter_stability_switches,
)


def _impedance_mode_arg(args: argparse.Namespace) -> str:
    return str(getattr(args, "impedance_mode", "pose"))


def _strict_tool_profile_arg(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "strict_tool_profile", True))


def _force_tare_sec_arg(args: argparse.Namespace) -> float:
    return float(getattr(args, "force_tare_sec", 3.0))


def _tool_profile_arg(args: argparse.Namespace, side: str) -> str | None:
    return getattr(args, f"{side}_tool_profile", "") or None


def _tool_tcp_pose_flange_arg(args: argparse.Namespace, side: str) -> tuple[float, ...] | None:
    profile_arg = _tool_profile_arg(args, side)
    if not profile_arg:
        return None
    try:
        from hardware.rm65.calibration_profiles import load_profile

        profile = load_profile(profile_arg, arm=side)
        tcp = profile.get("tcp") if isinstance(profile.get("tcp"), dict) else {}
        pose = tcp.get("pose_flange_m_rad")
        if pose is None:
            return None
        values = np.asarray(pose, dtype=float).reshape(6)
        if not np.all(np.isfinite(values)):
            return None
        return tuple(float(value) for value in values)
    except Exception as exc:
        _log("TCP", f"WARNING: could not load {side} tool TCP for collision solver: {exc}")
        return None


def _alignment_qpos_pair(alignment: TeleopInitialAlignment | None) -> dict[str, np.ndarray] | None:
    if alignment is None:
        return None
    qpos_pair: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        side_state = alignment.get_side(side)
        if side_state is None or side_state.arm_qpos is None:
            return None
        qpos_pair[side] = np.asarray(side_state.arm_qpos, dtype=float).reshape(6).copy()
    return qpos_pair


def _start_state_summary(checks: dict[str, object]) -> str:
    parts: list[str] = []
    for side in ("left", "right"):
        if side not in checks:
            continue
        check = checks[side]
        parts.append(
            f"{side}: max_abs={check.max_abs_joint_error_deg:.2f}deg "
            f"weighted={check.weighted_joint_error_deg:.2f}deg "
            f"within_limits={check.within_limits}"
        )
    return "; ".join(parts)


_expand_cli_path = real_teleop_dataset_layout.expand_cli_path
_sanitize_dataset_name_for_path = real_teleop_dataset_layout.sanitize_dataset_name_for_path
_default_task_id_from_name = real_teleop_dataset_layout.default_task_id_from_name


def _load_task_registry(args: argparse.Namespace) -> list[dict[str, str]]:
    return real_teleop_dataset_layout.load_task_registry(args, expand_path=_expand_cli_path)


def _task_definition_for_key(args: argparse.Namespace, key: str) -> dict[str, str] | None:
    return real_teleop_dataset_layout.task_definition_for_key(args, key, expand_path=_expand_cli_path)


def _collection_task_ui_state(args: argparse.Namespace) -> dict[str, object]:
    return real_teleop_dataset_layout.collection_task_ui_state(args, expand_path=_expand_cli_path)


def _resolve_formal_dataset_layout(args: argparse.Namespace) -> tuple[Path, str, str, str, str, str]:
    layout = real_teleop_dataset_layout.resolve_formal_dataset_layout(
        args,
        date_prefix=datetime.now().strftime("%Y-%m-%d"),
    )
    return (
        layout.dataset_root,
        layout.task_name,
        layout.task_name_zh,
        layout.task_id,
        layout.task_description,
        layout.dataset_name,
    )


def _apply_formal_dataset_layout(args: argparse.Namespace) -> None:
    real_teleop_dataset_layout.apply_formal_dataset_layout(
        args,
        date_prefix=datetime.now().strftime("%Y-%m-%d"),
    )


def _refresh_formal_dataset_layout(args: argparse.Namespace) -> None:
    real_teleop_dataset_layout.refresh_formal_dataset_layout(
        args,
        date_prefix=datetime.now().strftime("%Y-%m-%d"),
    )


def _toggle_collection_round_trip_phase(args: argparse.Namespace) -> bool:
    return real_teleop_dataset_layout.toggle_collection_round_trip_phase(
        args,
        date_prefix=datetime.now().strftime("%Y-%m-%d"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Control real dual RM65 arms from live VIVE wrist trackers, with optional "
            "additional hand q6 teleop and InspireTac hand control."
        )
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="UDP bind host for the VIVE receiver")
    parser.add_argument("--port", type=int, default=9999, help="UDP bind port for the VIVE receiver")
    parser.add_argument(
        "--vive-clock-sync-port",
        type=int,
        default=DEFAULT_VIVE_CLOCK_SYNC_PORT,
        help=(
            "Separate VIVE clock-control UDP port for the default new protocol; "
            "set 0 to select the legacy v1-only receiver."
        ),
    )
    parser.add_argument(
        "--vive-clock-sync-bind-host",
        type=str,
        default="",
        help="Optional VIVE clock-control bind host; defaults to --host.",
    )
    parser.add_argument(
        "--vive-clock-sync-expected-sender-host",
        type=str,
        default="",
        help="Optional VIVE sender IP allowed to submit clock-control results.",
    )
    parser.add_argument(
        "--vive-clock-sync-min-samples",
        type=int,
        default=4,
        help="Accepted four-timestamp samples required before sync is ready.",
    )
    parser.add_argument(
        "--vive-clock-sync-max-rtt-ms",
        type=float,
        default=25.0,
        help="Reject VIVE clock samples above this RTT; <=0 disables the cap.",
    )
    parser.add_argument(
        "--vive-clock-sync-stale-after-sec",
        type=float,
        default=15.0,
        help="Mark remote capture-time mapping stale after this interval.",
    )
    parser.add_argument(
        "--vive-clock-sync-required",
        action="store_true",
        help="Fail startup when VIVE clock-control does not become ready.",
    )
    parser.add_argument(
        "--vive-clock-sync-startup-timeout-sec",
        type=float,
        default=5.0,
        help="Maximum startup wait for required VIVE clock calibration.",
    )
    parser.add_argument(
        "--active-sides",
        type=_parse_active_sides,
        default=("left", "right"),
        help="One of: left, right, both.",
    )
    parser.add_argument("--left-arm-ip", type=str, default="", help="Left RM65 IP")
    parser.add_argument("--right-arm-ip", type=str, default="", help="Right RM65 IP")
    parser.add_argument(
        "--arm-ip-config",
        type=str,
        default="latest",
        help="RM65 arm IP assignment config path, or 'latest'. Missing --left/right-arm-ip values are filled from it.",
    )
    parser.add_argument(
        "--no-arm-ip-default-config",
        dest="arm_ip_default_config",
        action="store_false",
        default=True,
        help="Do not auto-fill missing RM65 arm IPs from the latest saved assignment.",
    )
    parser.add_argument(
        "--no-arm-ip-save-latest",
        dest="arm_ip_save_latest",
        action="store_false",
        default=True,
        help="Do not save explicitly provided RM65 arm IPs to the latest assignment.",
    )
    parser.add_argument("--left-hand-ip", type=str, default="", help="Optional left InspireTac IP")
    parser.add_argument("--right-hand-ip", type=str, default="", help="Optional right InspireTac IP")
    parser.add_argument("--left-hand-port", type=str, default="", help="Optional left hand teleop serial port")
    parser.add_argument("--right-hand-port", type=str, default="", help="Optional right hand teleop serial port")
    parser.add_argument(
        "--fix-arm",
        action="store_true",
        help=(
            "Enable fixed-arm collection mode. In this mode GUI C freezes the selected "
            "arm/hand at the last dispatched action, and C again recaptures that side's "
            "runtime home before releasing it."
        ),
    )
    parser.add_argument(
        "--fix-arm-side",
        choices=("left", "right"),
        default="left",
        help="Initial side selected by --fix-arm. The GUI can switch it while no side is fixed.",
    )
    parser.add_argument(
        "--init-arm-available",
        "--init-arm-availabe",
        dest="init_arm_available",
        action="store_true",
        help=(
            "Enable init-arm mode. GUI G routes the selected arm to the init pose, "
            "and G again recaptures that side's runtime home before resuming teleop."
        ),
    )
    parser.add_argument(
        "--init-arm-profile",
        type=str,
        default="",
        help=(
            "Dual-arm init pose profile JSON used by --init-arm-available. "
            "Each side may provide arm_qpos, tcp_pose_base, or tcp_pose_desk."
        ),
    )
    parser.add_argument(
        "--init-arm-side",
        choices=("left", "right"),
        default="left",
        help="Initial side selected by --init-arm-available. The GUI can switch it while init-arm is idle.",
    )
    parser.add_argument(
        "--special-arm-control-mode",
        choices=SPECIAL_ARM_CONTROL_MODE_CHOICES,
        default=None,
        help=(
            "Special-arm GUI input mode. When both --fix-arm and --init-arm-available "
            "are set, defaults to foot-pad (F13/F14/F15/F16); otherwise defaults to manual."
        ),
    )
    parser.add_argument(
        "--hand-backend",
        choices=HAND_BACKEND_CHOICES,
        default=HAND_BACKEND_INSPIRE,
        help=(
            "Hand/gripper backend type. Use dmtac-gripper to map Q6 to the DMTAC "
            "1D gripper backend and expose browser GUI control/status."
        ),
    )
    parser.add_argument(
        "--left-hand-backend",
        choices=HAND_BACKEND_CHOICES,
        default=None,
        help="Optional left-side override for --hand-backend.",
    )
    parser.add_argument(
        "--right-hand-backend",
        choices=HAND_BACKEND_CHOICES,
        default=None,
        help="Optional right-side override for --hand-backend.",
    )
    parser.add_argument(
        "--q6-config",
        type=str,
        default="latest",
        help=(
            "Q6 left/right assignment config used by --hand-backend dmtac-gripper. "
            "Explicit --left/right-hand-port wins."
        ),
    )
    parser.add_argument(
        "--no-q6-default-config",
        dest="q6_default_config",
        action="store_false",
        default=True,
        help="Do not auto-fill missing Q6 serial ports from the latest saved assignment.",
    )
    parser.add_argument(
        "--q6-config-required",
        action="store_true",
        help="Fail if --q6-config cannot be loaded while a DMTAC gripper side is missing a Q6 port.",
    )
    parser.add_argument(
        "--gripper-angle-trigger",
        action="store_true",
        help="Use calibrated hardware.angle_trigger input directly instead of Q6 discovery.",
    )
    parser.add_argument(
        "--gripper-angle-trigger-config-root",
        type=Path,
        default=None,
        help="Optional angle-trigger assignment/angle-calibration directory.",
    )
    parser.add_argument(
        "--gripper-angle-trigger-rate-hz",
        type=_positive_float,
        default=30.0,
        help="Angle-trigger reader polling rate.",
    )
    parser.add_argument(
        "--q6-hold-recalibration-gesture",
        action="store_true",
        help=(
            "Enable dual-arm Q6 gesture control: close both hands' q6 dims 4/5/6 for "
            "1s to hold, release them for 1s while holding, then close again for 1s "
            "to re-capture runtime home and resume teleop."
        ),
    )
    parser.add_argument(
        "--q6-gesture-hold-sec",
        type=_positive_float,
        default=1.0,
        help="Required continuous Q6 gesture duration for hold/release/restart.",
    )
    parser.add_argument(
        "--q6-gesture-close-threshold",
        type=float,
        default=0.95,
        help="Q6 dims 4/5/6 must be above this value on both hands to count as closed.",
    )
    parser.add_argument(
        "--q6-gesture-open-threshold",
        type=float,
        default=0.85,
        help=(
            "Q6 dims 4/5/6 must be below this value on both hands to count as released "
            "between hold/restart gestures."
        ),
    )
    parser.add_argument("--left-gripper-port", type=str, default="", help="Optional left DMTAC gripper serial port.")
    parser.add_argument("--right-gripper-port", type=str, default="", help="Optional right DMTAC gripper serial port.")
    parser.add_argument("--left-gripper-id", type=int, default=1, help="Left gripper Modbus id.")
    parser.add_argument("--right-gripper-id", type=int, default=1, help="Right gripper Modbus id.")
    parser.add_argument(
        "--dmtac-config",
        type=str,
        default="latest",
        help="DMTAC gripper/tactile assignment config used by --hand-backend dmtac-gripper.",
    )
    parser.add_argument(
        "--no-dmtac-default-config",
        dest="dmtac_default_config",
        action="store_false",
        default=True,
        help="Do not auto-load DMTAC gripper/tactile ports from latest assignment.",
    )
    parser.add_argument(
        "--dmtac-config-required",
        action="store_true",
        help="Fail if --dmtac-config cannot be loaded for --hand-backend dmtac-gripper.",
    )
    parser.add_argument(
        "--dmtac-allow-missing",
        action="store_true",
        help="Allow configured DMTAC gripper/tactile devices to be missing during USB validation.",
    )
    parser.add_argument(
        "--dmtac-no-validate-current-usb",
        dest="dmtac_validate_current_usb",
        action="store_false",
        default=True,
        help="Load DMTAC assignment without probing current USB devices.",
    )
    parser.add_argument(
        "--dmtac-confirm-extra",
        dest="dmtac_assume_yes_for_extra",
        action="store_false",
        default=True,
        help="Ask before continuing when extra unassigned DMTAC devices are detected.",
    )
    parser.add_argument(
        "--dmtac-tactile-backend",
        type=str,
        default="cpu",
        help="DM-Tac tactile backend passed to portable_dmtac_gripper.",
    )
    parser.add_argument(
        "--dmtac-tactile-timeout-ms",
        type=_positive_int,
        default=None,
        help="Per-sensor DM-Tac read timeout. Runtime telemetry does not wait for new frames by default.",
    )
    parser.add_argument(
        "--dmtac-tactile-max-fps",
        type=_positive_int,
        default=None,
        help="DM-Tac camera max_fps. Lower values reduce USB/CPU pressure; default follows backend config.",
    )
    parser.add_argument(
        "--dmtac-tactile-warmup-frames",
        type=_non_negative_int,
        default=None,
        help="DM-Tac tactile filter warmup frames. Runtime default is 0 like the assignment GUI.",
    )
    parser.add_argument(
        "--dmtac-tactile-wait-for-new",
        action="store_true",
        default=None,
        help="Wait for a new DM-Tac frame on each telemetry read. Higher latency; off by default.",
    )
    parser.add_argument(
        "--dmtac-tactile-read-hz",
        type=_non_negative_float,
        default=None,
        help=(
            "Max DM-Tac tactile read rate per gripper backend. Default is backend config "
            "1 Hz; use 0 to read every gripper telemetry cycle."
        ),
    )
    parser.add_argument(
        "--dmtac-tactile-process-read",
        dest="dmtac_tactile_process_read",
        action="store_true",
        default=None,
        help="Read DM-Tac tactile frames in a separate low-priority process. Runtime default is on.",
    )
    parser.add_argument(
        "--dmtac-tactile-in-process-read",
        dest="dmtac_tactile_process_read",
        action="store_false",
        help="Read DM-Tac tactile frames in the main teleop process.",
    )
    parser.add_argument(
        "--dmtac-tactile-worker-cpus",
        type=_dmtac_worker_cpu_affinity,
        default=None,
        help=(
            "Comma/space-separated CPU cores for low-priority DM-Tac tactile worker processes, "
            "for example '6,7'. Unset lets the OS schedule normally."
        ),
    )
    parser.add_argument(
        "--dmtac-tactile-worker-nice",
        type=_dmtac_worker_nice,
        default=None,
        help="Lower scheduling priority for DM-Tac tactile worker processes; 0 is normal, 19 is lowest.",
    )
    parser.add_argument(
        "--dmtac-tactile-worker-queue-size",
        type=_positive_int,
        default=None,
        help="Latest-frame queue size between each DM-Tac tactile worker process and the main process.",
    )
    parser.add_argument(
        "--dmtac-tactile-parallel-read",
        dest="dmtac_tactile_parallel_read",
        action="store_true",
        default=None,
        help="Read sensors in each DM-Tac gripper backend concurrently. Higher USB/CPU pressure.",
    )
    parser.add_argument(
        "--dmtac-tactile-sequential-read",
        dest="dmtac_tactile_parallel_read",
        action="store_false",
        help="Read sensors in each DM-Tac gripper backend sequentially. This is the runtime default.",
    )
    parser.add_argument(
        "--dmtac-tactile-global-read-lock",
        dest="dmtac_tactile_global_read_lock",
        action="store_true",
        default=None,
        help="Serialize DMTAC tactile reads across left/right gripper backends. Runtime default is on.",
    )
    parser.add_argument(
        "--dmtac-tactile-no-global-read-lock",
        dest="dmtac_tactile_global_read_lock",
        action="store_false",
        help="Allow left/right DMTAC tactile reads to overlap.",
    )
    parser.add_argument(
        "--dmtac-tactile-fields",
        type=_parse_dmtac_tactile_fields,
        default=None,
        metavar="FIELDS",
        help=(
            "DM-Tac fields to read/store. Default is raw-image-only collection; pass "
            "'distributed_force' for real-time force output. Valid fields: "
            + ", ".join(DMTAC_TACTILE_FIELDS)
        ),
    )
    parser.add_argument(
        "--dmtac-tactile-compute-profile",
        choices=("dist-only", "legacy", "requested", "shear-depth"),
        default=None,
        help=(
            "SDK compute channels for DMTAC telemetry. Default follows backend config; "
            "use dist-only with distributed_force for the real-time force path."
        ),
    )
    parser.add_argument(
        "--dmtac-tactile-denoise",
        dest="dmtac_tactile_denoise",
        action="store_true",
        default=None,
        help="Enable runtime distributed_force denoise. This is heavier than the assignment GUI default.",
    )
    parser.add_argument(
        "--dmtac-tactile-no-denoise",
        dest="dmtac_tactile_denoise",
        action="store_false",
        help="Disable runtime distributed_force denoise.",
    )
    parser.add_argument(
        "--dmtac-tactile-flat-vector",
        dest="dmtac_tactile_flat_vector",
        action="store_true",
        default=None,
        help="Keep building the legacy flattened tactile vector in telemetry.",
    )
    parser.add_argument(
        "--dmtac-tactile-no-flat-vector",
        dest="dmtac_tactile_flat_vector",
        action="store_false",
        help="Do not build the legacy flattened tactile vector in telemetry.",
    )
    parser.add_argument(
        "--dmtac-with-tactile",
        dest="dmtac_tactile",
        action="store_true",
        default=None,
        help=(
            "Opt in to DM-Tac tactile sensors during real teleop. Off by default so gripper/Q6 "
            "control does not depend on the DM-Tac SDK camera threads."
        ),
    )
    parser.add_argument("--dmtac-no-tactile", dest="dmtac_tactile", action="store_false")
    parser.add_argument(
        "--no-device-preflight",
        dest="device_preflight",
        action="store_false",
        default=True,
        help="Skip GUI startup USB/camera/Q6/DMTAC assignment validation.",
    )
    parser.add_argument(
        "--device-fix-permissions",
        action="store_true",
        help="Before GUI startup preflight, try setfacl on serial/video nodes for the current user.",
    )
    parser.add_argument(
        "--no-device-auto-configure",
        dest="device_auto_configure",
        action="store_false",
        default=True,
        help="Only validate devices; do not enter Q6/DMTAC/RealSense assignment flows.",
    )
    parser.add_argument(
        "--device-preflight-yes",
        action="store_true",
        help="Use default left/right assignment choices during GUI device preflight.",
    )
    parser.add_argument(
        "--device-preflight-probe-all-serial",
        action="store_true",
        help=(
            "Probe all current USB serial ports during GUI device preflight. Useful after USB "
            "disconnect/reconnect when descriptors or by-id aliases changed."
        ),
    )
    parser.add_argument(
        "--device-preflight-force-reconfigure",
        action="store_true",
        help=(
            "Ignore existing valid Q6/DMTAC assignments during GUI preflight and re-run "
            "their selection flows, saving refreshed configs. RealSense and arm IP configs stay unchanged."
        ),
    )
    parser.add_argument(
        "--device-preflight-no-tactile",
        dest="device_preflight_require_tactile",
        action="store_false",
        default=True,
        help="Do not require DMTAC tactile sensors during GUI device preflight.",
    )
    parser.add_argument(
        "--gripper-task-profile",
        type=str,
        default="",
        help=(
            "Optional JSON profile with a top-level gripper object and/or tasks.<name>.gripper "
            "overrides. It can set speed, force/torque, max_action, send_threshold, command_hz, "
            "continuous_grasp, source_dim, output_dim, invert, alpha, input_min, input_max."
        ),
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default=str(DEFAULT_TASKS_CONFIG_PATH),
        help=(
            "JSON task registry for the GUI. Each task defines key, english_name, "
            "chinese_name, and gripper_task."
        ),
    )
    parser.add_argument(
        "--gripper-task",
        "--task-id",
        "--task-name",
        dest="gripper_task",
        type=str,
        default="",
        help="Task key used to select tasks.<key>.gripper from --gripper-task-profile.",
    )
    parser.add_argument("--gripper-source-dim", type=int, default=None, help="Q6 source dimension for 1D gripper.")
    parser.add_argument("--gripper-output-dim", type=int, default=None, help="Output slot in the 6-wide hand vector.")
    parser.add_argument("--gripper-invert", dest="gripper_invert", action="store_true", default=None)
    parser.add_argument("--gripper-no-invert", dest="gripper_invert", action="store_false")
    parser.add_argument("--gripper-alpha", type=_alpha_float, default=None, help="EMA alpha for 1D gripper command.")
    parser.add_argument(
        "--gripper-input-min",
        type=float,
        default=None,
        help="Q6 value mapped to fully open gripper before optional inversion.",
    )
    parser.add_argument(
        "--gripper-input-max",
        type=float,
        default=None,
        help="Q6 value mapped to fully closed gripper before optional inversion.",
    )
    parser.add_argument("--gripper-speed", type=float, default=None, help="Gripper move speed for this task.")
    parser.add_argument(
        "--gripper-torque",
        "--gripper-force",
        dest="gripper_torque",
        type=float,
        default=None,
        help="Gripper torque/force setting for this task.",
    )
    parser.add_argument("--gripper-max-action", type=float, default=None, help="Max normalized close command.")
    parser.add_argument(
        "--gripper-send-threshold",
        type=float,
        default=None,
        help="Minimum normalized delta before sending another gripper command.",
    )
    parser.add_argument("--gripper-continuous", dest="gripper_continuous_grasp", action="store_true", default=None)
    parser.add_argument("--gripper-no-continuous", dest="gripper_continuous_grasp", action="store_false")
    parser.add_argument("--gripper-command-hz", type=_positive_float, default=None)
    parser.add_argument("--gripper-telemetry-hz", type=_positive_float, default=None)
    parser.add_argument(
        "--gripper-warmup-probe",
        dest="gripper_warmup_probe",
        action="store_true",
        default=True,
        help=(
            "During recording warmup, command DMTAC grippers open->half-close->open "
            "and reconnect any gripper whose measured state does not follow."
        ),
    )
    parser.add_argument(
        "--no-gripper-warmup-probe",
        dest="gripper_warmup_probe",
        action="store_false",
        help="Disable the DMTAC gripper motion probe during recording warmup.",
    )
    parser.add_argument(
        "--gripper-warmup-probe-tolerance",
        type=float,
        default=0.18,
        help="Allowed normalized error when validating DMTAC gripper warmup motion.",
    )
    parser.add_argument(
        "--gripper-warmup-probe-step-timeout-sec",
        type=_positive_float,
        default=2.0,
        help="Timeout per open/half/open gripper warmup step.",
    )
    parser.add_argument(
        "--gripper-warmup-probe-settle-sec",
        type=float,
        default=0.20,
        help="Settling delay after each gripper warmup command before reading state.",
    )
    parser.add_argument(
        "--gripper-warmup-probe-poll-sec",
        type=_positive_float,
        default=0.05,
        help="Polling interval while waiting for DMTAC gripper warmup state.",
    )
    parser.add_argument("--hand-reader-baudrate", type=int, default=115200, help="Hand teleop serial baudrate")
    parser.add_argument("--hand-reader-timeout", type=float, default=0.05, help="Hand teleop serial timeout in seconds")
    parser.add_argument(
        "--exo-hand-calibration",
        action="store_true",
        help="Re-run exoskeleton hand zero-pose calibration and update the fixed calibration file.",
    )
    parser.add_argument(
        "--exo-hand-calibration-path",
        type=str,
        default="",
        help="Optional fixed exoskeleton hand zero-pose calibration JSON path.",
    )
    parser.add_argument("--show-hand-raw-lines", action="store_true", help="Print every raw line from hand teleop serial devices")
    parser.add_argument(
        "--hand-joint-order",
        type=_parse_joint_order,
        default=DEFAULT_INSPIRE_HAND_JOINT_ORDER,
        help=(
            "Permutation applied to hand q6 before mapping to InspireTac q6. "
            f"Default maps into InspireTac order {list(TARGET_INSPIRE_HAND_Q6_ORDER)}."
        ),
    )
    parser.add_argument(
        "--hand-invert",
        dest="hand_invert",
        action="store_true",
        help="Invert normalized q6 before sending to the InspireTac hand. Default is enabled.",
    )
    parser.add_argument(
        "--hand-no-invert",
        dest="hand_invert",
        action="store_false",
        help="Disable q6 inversion if the hand already moves in the correct direction.",
    )
    parser.set_defaults(hand_invert=True)
    parser.add_argument(
        "--hand-command-alpha",
        type=_alpha_float,
        default=1.0,
        help="EMA smoothing factor for hand q6 mapping.",
    )
    parser.add_argument(
        "--hand-send-threshold",
        type=float,
        default=0.002,
        help="Only send/update hand q6 when max delta exceeds this threshold.",
    )
    parser.add_argument(
        "--hand-telemetry-hz",
        type=_positive_float,
        default=30.0,
        help="Telemetry polling rate for each real InspireTac hand backend.",
    )
    parser.add_argument(
        "--hand-tactile-regions",
        type=str,
        default="all",
        help=(
            "Comma-separated tactile regions to poll, for example "
            "'thumb_1,index_1,palm' or 'all'. Defaults to all; pass an empty string to skip tactile reads."
        ),
    )
    parser.add_argument(
        "--print-tactile-sum-only",
        action="store_true",
        help=(
            "Suppress normal stdout logs and print only real-time left/right hand tactile sums. "
            "Teleop behavior is unchanged. Disabled by default."
        ),
    )
    parser.add_argument(
        "--visualize-tactile",
        action="store_true",
        help=(
            "Show an OpenCV tactile window during real teleop. InspireTac uses the UV layout; "
            "DMTAC gripper uses 20x30x3 force grids: color=max(Fy,0), arrows=(Fx,Fz)."
        ),
    )
    parser.add_argument(
        "--tactile-vis-value-scale",
        type=float,
        default=0.05,
        help="Scale used by the tactile visualizer response curve.",
    )
    parser.add_argument(
        "--tactile-vis-size",
        type=int,
        default=640,
        help="Per-hand tactile visualizer panel size in pixels.",
    )
    parser.add_argument(
        "--tactile-vis-hz",
        type=_positive_float,
        default=5.0,
        help="Maximum tactile visualization update rate.",
    )
    parser.add_argument(
        "--camera-config",
        type=str,
        default=str(DEFAULT_CAMERA_CONFIG_PATH),
        help="JSON config for RealSense RGB capture/video and optional preview windows.",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Disable RealSense capture, preview, and video writing.",
    )
    parser.add_argument(
        "--vis_cam",
        "--vis-cam",
        dest="vis_cam",
        action="store_true",
        help="Open optional RealSense RGB preview windows at startup.",
    )
    parser.add_argument(
        "--vis-cam-key",
        "--vis-camera-key",
        type=str,
        default="",
        help=(
            "Camera key to visualize when camera preview is opened. "
            "Defaults to all configured cameras; use 'front', 'left_wrist', or 'right_wrist' to show one."
        ),
    )
    parser.add_argument(
        "--vis-cam-hz",
        type=_positive_float,
        default=10.0,
        help="Maximum camera preview update rate.",
    )
    parser.add_argument(
        "--inspire-hand-max-force",
        "--max-force",
        "--max-fingertip-force",
        dest="max_fingertip_force",
        type=_non_negative_int,
        default=DEFAULT_INSPIRE_HAND_MAX_FORCE,
        help="Max force value written to every InspireTac finger on startup.",
    )
    parser.add_argument(
        "--teleop-hardware-ip",
        type=str,
        default="",
        help="Reserved teleop-hardware endpoint IP, currently unused.",
    )
    parser.add_argument(
        "--teleop-hardware-port",
        type=int,
        default=0,
        help="Reserved teleop-hardware endpoint port, currently unused.",
    )
    parser.add_argument(
        "--swap-arm-sides",
        action="store_true",
        help=(
            "Swap logical left/right assignment for the provided arm IPs. "
            "Useful when the physical arm connected as --left-arm-ip should behave as logical right."
        ),
    )
    parser.add_argument(
        "--swap-operator-sides",
        dest="swap_operator_sides",
        action="store_true",
        help=(
            "Before teleop starts, swap only VIVE tracker inputs: logical left uses the right VIVE "
            "tracker and logical right uses the left VIVE tracker. Arms, hands, serial ports, and "
            "profiles are unchanged. This mapping is fixed after startup."
        ),
    )
    parser.add_argument(
        "--control-mode",
        choices=["absolute"],
        default="absolute",
        help=(
            "Arm retargeting mode. The active runtime uses the archived absolute pose mapping."
        ),
    )
    parser.add_argument(
        "--arm-execution-profile",
        choices=ARM_EXECUTION_PROFILE_CHOICES,
        default=None,
        help=(
            "Mode-aware real-arm defaults. impedance_40hz resolves to a 30 Hz outer "
            "reference and 40 Hz impedance backend; position resolves to the "
            "configurable position chain (defaults: 15 Hz sampling, 75 Hz references, "
            "150 Hz CANFD stream). "
            "position_45_150hz remains an explicit legacy profile, and fine-grained "
            "flags remain compatible overrides."
        ),
    )
    parser.add_argument(
        "--arm-control-mode",
        choices=["impedance", "position"],
        default="impedance",
        help=(
            "Real-arm execution mode. Default impedance uses the outer-loop pose impedance profile; "
            "position uses the legacy IK + Movej_CANFD path."
        ),
    )
    parser.add_argument(
        "--safe-debug",
        dest="safe_debug",
        action="store_true",
        help=(
            "Compatibility diagnostic marker for position CANFD passthrough. Target pose "
            "smoothing, local-step candidate selection, and inactive-side hold/cap are now "
            "enabled by default in live teleop."
        ),
    )
    parser.add_argument(
        "--disable-hold",
        action="store_true",
        help=(
            "When used with --safe-debug in position CANFD passthrough, disable small-motion "
            "hold behavior: current-pose hold, pose-smoother deadband hold, and inactive-side hold/cap."
        ),
    )
    parser.add_argument(
        "--strict-degree-limit",
        action="store_true",
        help="Scale qpos velocity/step limits to one fifth of their configured values.",
    )
    parser.add_argument(
        "--impedance-mode",
        choices=["pose"],
        default="pose",
        help="Impedance controller reference mode. Pose is the default and matches the unified arm action.",
    )
    parser.add_argument(
        "--arm-canfd-follow",
        dest="arm_canfd_follow",
        action="store_true",
        help=(
            "Send RM65 CANFD commands with SDK follow=True. The SDK requires a stable "
            "transparent transmission period <=10ms for high-follow use. Defaults to true "
            "for --arm-control-mode position and false for force-admittance impedance."
        ),
    )
    parser.add_argument(
        "--no-arm-canfd-follow",
        dest="arm_canfd_follow",
        action="store_false",
        help="Send RM65 impedance CANFD commands with SDK follow=False.",
    )
    parser.set_defaults(arm_canfd_follow=None)
    parser.add_argument(
        "--position-post-ik-qpos-smoothing-alpha",
        type=_alpha_float,
        default=0.14,
        help=(
            "EMA alpha applied after IK/safety step limits and before the outer qpos "
            "reference is created. Position teleop defaults to the adaptive slow endpoint 0.14. Set 1.0 "
            "to disable this pre-trajectory smoothing."
        ),
    )
    parser.add_argument(
        "--position-post-ik-qpos-smoothing-fast-alpha",
        type=_alpha_float,
        default=0.45,
        help="Fast-motion endpoint for adaptive post-IK qpos smoothing (position teleop only).",
    )
    parser.add_argument(
        "--position-post-ik-qpos-smoothing-adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Adapt post-IK qpos EMA from the slow alpha to --position-post-ik-qpos-"
            "smoothing-fast-alpha using validated tracker linear/angular velocity. "
            "Only position teleoperation enables the policy."
        ),
    )
    parser.add_argument(
        "--arm-follow-qpos-alpha",
        type=_alpha_float,
        default=None,
        help=(
            "Final high-rate qpos EMA alpha applied directly at the configured "
            "--arm-control-hz SDK send boundary. Defaults to 0.1 in position streaming; "
            "the filtered qpos is committed back to the planner. Quick-stop/hold bypasses "
            "smoothing. Set 1.0 to disable."
        ),
    )
    parser.add_argument(
        "--arm-canfd-follow-stream-hz",
        type=_non_negative_float,
        default=None,
        help=(
            "High-rate Movej_CANFD stream frequency used when RM65 CANFD follow=True. "
            "In position mode this controls the background qpos stream; in impedance "
            "joint transport this controls the follow stream. Backend default is mode-specific; "
            "set 0 to disable streaming and fall back to one command per outer/backend step."
        ),
    )
    parser.add_argument(
        "--arm-canfd-follow-warning-ratio",
        type=_alpha_float,
        default=None,
        help=(
            "Warn when actual CANFD follow stream send rate falls below "
            "ratio * --arm-canfd-follow-stream-hz. Backend default is 0.9."
        ),
    )
    parser.add_argument(
        "--arm-canfd-trajectory-mode",
        type=_canfd_trajectory_mode,
        default=None,
        help=(
            "RM65 CANFD high-follow trajectory mode: 0 passthrough, 1 curve fitting, "
            "2 filtering. Backend/profile default is 0."
        ),
    )
    parser.add_argument(
        "--arm-canfd-radio",
        type=_canfd_radio,
        default=None,
        help=(
            "RM65 CANFD smoothing parameter for curve fitting/filtering modes. "
            "Valid range is 0..100. Backend/profile default is 0."
        ),
    )
    parser.add_argument(
        "--arm-state-read-every",
        type=_positive_int,
        default=None,
        help=(
            "RM65 impedance state read decimation. 1 reads arm state every backend step. "
            "When --arm-canfd-follow is enabled, backend default is 4 to reduce SDK lock "
            "contention with the high-rate CANFD stream."
        ),
    )
    parser.add_argument(
        "--arm-disable-force-admittance",
        action="store_true",
        help=(
            "Diagnostic impedance-mode override: keep pose-action execution but zero the "
            "outer-loop force/admittance offset so RM65 commands are driven only by the "
            "retargeted pose, IK, and CANFD follow path."
        ),
    )
    parser.add_argument(
        "--force-tare-sec",
        type=_positive_float,
        default=3.0,
        help="Required stillness/tare duration after runtime home before enabling impedance force control.",
    )
    parser.add_argument(
        "--strict-tool-profile",
        dest="strict_tool_profile",
        action="store_true",
        help="Require active RM65 tool, confirmed TCP, payload, and impedance profile before arming. Default.",
    )
    parser.add_argument(
        "--no-strict-tool-profile",
        dest="strict_tool_profile",
        action="store_false",
        help="Allow incomplete tool profiles for lab debugging. Not recommended for real teleop.",
    )
    parser.set_defaults(strict_tool_profile=True)
    parser.add_argument(
        "--auto-sync-tool-frame",
        dest="auto_sync_tool_frame",
        action="store_true",
        help=(
            "Before impedance arming, write/create and activate the RM65 controller tool frame "
            "from the selected calibration profile. Default."
        ),
    )
    parser.add_argument(
        "--no-auto-sync-tool-frame",
        dest="auto_sync_tool_frame",
        action="store_false",
        help="Disable automatic RM65 controller tool-frame sync for impedance debugging.",
    )
    parser.set_defaults(auto_sync_tool_frame=True)
    parser.add_argument(
        "--left-tool-profile",
        type=str,
        default="",
        help=(
            "Optional explicit left RM65 calibration/impedance profile name or JSON path. "
            "Use position_endpoint with --arm-control-mode position for legacy endpoint-TCP position control."
        ),
    )
    parser.add_argument(
        "--right-tool-profile",
        type=str,
        default="",
        help=(
            "Optional explicit right RM65 calibration/impedance profile name or JSON path. "
            "Use position_endpoint with --arm-control-mode position for legacy endpoint-TCP position control."
        ),
    )
    parser.add_argument(
        "--tcp-mode",
        type=str,
        default=TOOL_TCP_FRAME,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--movement-multiple",
        type=_movement_multiple_arg,
        default=None,
        help=(
            "倍率 for VIVE tracker translation around runtime home. Rotation is unchanged; "
            "robot target translation = robot_home + movement_multiple * (tracker - tracker_home)."
        ),
    )
    parser.add_argument(
        "--translation-scale",
        type=_movement_multiple_arg,
        default=MOVEMENT_MULTIPLE_DEFAULT,
        help=argparse.SUPPRESS,
    )
    add_vive_tracker_height_correction_args(parser)
    parser.add_argument(
        "--virtual-desk-tracker-role",
        choices=["auto", "auxiliary", "left_wrist", "right_wrist"],
        default="auto",
        help=(
            "Tracker role used by the GUI VIVE coordinate calibration sweeps. "
            "auto asks in the GUI and defaults to right_wrist; auxiliary explicitly skips the hand prompt."
        ),
    )
    parser.add_argument(
        "--virtual-desk-post-yaw-deg",
        type=float,
        default=180.0,
        help=(
            "Post-rotation about virtual desk +Z after GUI VIVE coordinate calibration. "
            "Default 180 matches the standalone calibration GUI/SAPIEN scene convention."
        ),
    )
    parser.add_argument(
        "--sweep-capture-sec",
        type=_positive_float,
        default=5.0,
        help="Timed capture duration for each GUI VIVE coordinate X/Y/Z sweep.",
    )
    parser.add_argument(
        "--capture-prep-sec",
        type=_non_negative_float,
        default=3.0,
        help="Preparation countdown before each GUI VIVE coordinate capture starts.",
    )
    parser.add_argument(
        "--capture-extra-wait-sec",
        type=_non_negative_float,
        default=0.0,
        help="Extra wait time if GUI VIVE coordinate capture ends before enough fresh samples arrive.",
    )
    parser.add_argument(
        "--min-sweep-extent-m",
        type=_positive_float,
        default=0.15,
        help="Minimum motion extent required for each GUI VIVE coordinate sweep.",
    )
    parser.add_argument(
        "--min-axis-variance-ratio",
        type=float,
        default=0.5,
        help="Minimum PCA variance ratio required for each GUI VIVE coordinate sweep.",
    )
    parser.add_argument(
        "--min-start-end-distance-m",
        type=_positive_float,
        default=0.08,
        help="Minimum start/end separation required for each GUI VIVE coordinate sweep.",
    )
    parser.add_argument(
        "--max-axis-non-orthogonality-deg",
        type=_positive_float,
        default=20.0,
        help="Maximum allowed axis correction when fitting GUI VIVE coordinate sweeps.",
    )
    parser.add_argument(
        "--min-sweep-samples",
        type=int,
        default=3,
        help="Minimum valid fresh samples for each GUI VIVE coordinate sweep.",
    )
    parser.add_argument(
        "--wrist-capture-prep-sec",
        type=_non_negative_float,
        default=5.0,
        help="Preparation countdown before GUI wrist tracker offset capture.",
    )
    parser.add_argument(
        "--hold-capture-sec",
        type=_positive_float,
        default=4.0,
        help="Timed capture duration for GUI wrist tracker offset capture.",
    )
    parser.add_argument(
        "--min-hold-samples",
        type=int,
        default=3,
        help="Minimum valid samples for each side during GUI wrist tracker offset capture.",
    )
    parser.add_argument(
        "--no-collision",
        "--disable-collision-avoidance",
        "--disable-dual-arm-collision-avoidance",
        dest="disable_dual_arm_collision_avoidance",
        action="store_true",
        help="Disable joint-space dual-arm collision avoidance and fall back to two independent arm retargeters.",
    )
    parser.add_argument(
        "--collision-avoidance-mode",
        choices=["boundary_slide", "freeze_on_contact"],
        default="boundary_slide",
        help="Behavior when the operator pushes both arms toward collision.",
    )
    parser.add_argument(
        "--collision-scene-profile",
        choices=[
            "dual_arm",
            "dual_arm_hand_proxy_open",
            "dual_arm_hand",
            "dual_arm_cylinder_table",
        ],
        default="dual_arm_hand_proxy_open",
        help=(
            "Joint-space collision scene profile used by the dual-arm solver. "
            "'dual_arm_hand_proxy_open' keeps arm IK at 12DoF while attaching fixed open-hand "
            "proxy geometry to both wrists and a fixed table box for collision."
        ),
    )
    parser.add_argument(
        "--rm65-ik-framework",
        choices=["robot_motion"],
        default="robot_motion",
        help=(
            "RM65 IK implementation used by real/sim arm backends and dual-arm collision IK. "
            "robot_motion routes all shared RM65 IK construction and solving through robot_motion.rm65."
        ),
    )
    parser.add_argument(
        "--dual-arm-temporary-safety",
        dest="dual_arm_temporary_safety",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable a temporary conservative output limiter for initial real dual-arm bring-up. "
            "Use --no-dual-arm-temporary-safety to remove it without editing code."
        ),
    )
    parser.add_argument(
        "--dual-arm-temp-max-xyz-step-m",
        type=_positive_float,
        default=0.03,
        help="Temporary dual-arm safety max per-frame TCP translation step in meters.",
    )
    parser.add_argument(
        "--dual-arm-temp-max-rot-step-deg",
        type=_positive_float,
        default=8.0,
        help="Temporary dual-arm safety max per-frame TCP rotation step in degrees.",
    )
    parser.add_argument(
        "--dual-arm-temp-max-joint-step-deg",
        type=_positive_float,
        default=6.0,
        help="Temporary dual-arm safety max per-frame joint-vector step in degrees.",
    )
    parser.add_argument(
        "--dual-arm-temp-command-scale",
        type=_alpha_float,
        default=1.0,
        help="Temporary dual-arm safety scale applied before max-step clipping.",
    )
    parser.add_argument(
        "--hz",
        type=_positive_float,
        default=DEFAULT_TELEOP_SAMPLE_HZ,
        help=(
            "Dataset sample/save rate. Defaults to 15 Hz and remains independent from "
            "the impedance teleop/IK rate."
        ),
    )
    parser.add_argument(
        "--impedance-teleop-control-hz",
        type=_positive_float,
        default=None,
        help=(
            "Force-teleop outer command and dual-arm IK rate. Defaults to 30 Hz when "
            "--hz is left at its default; use any positive value. An explicit legacy --hz "
            "without this flag keeps its historical coupled force-control rate."
        ),
    )
    parser.add_argument(
        "--position-reference-hz",
        type=_positive_float,
        default=DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
        help="Position-mode IK/safety/qpos reference rate. Defaults to 75 Hz.",
    )
    parser.add_argument(
        "--position-reference-max-age-sec",
        type=_positive_float,
        default=0.2,
        help=(
            "Maximum causal qpos-reference age accepted by 15 Hz position-control collection. "
            "Missing or older references drop the training row."
        ),
    )
    parser.add_argument(
        "--position-command-planner",
        choices=POSITION_COMMAND_PLANNER_CHOICES,
        default=POSITION_COMMAND_PLANNER_LINEAR,
        help=(
            "Final position-command planner. 'linear' preserves the existing pure-position "
            "reference interpolation; 'acceleration_limited' adds velocity-continuous, "
            "acceleration-limited online 75->150 Hz planning."
        ),
    )
    parser.add_argument(
        "--position-velocity-ramp-sec",
        type=_positive_float,
        default=DEFAULT_POSITION_VELOCITY_RAMP_SEC,
        help=(
            "Position-error feedback horizon used to form desired velocity in the optional "
            "planner. Defaults to 0.10 s; acceleration is limited independently by "
            "--position-max-acceleration-rad-s2."
        ),
    )
    parser.add_argument(
        "--position-target-velocity-filter-tau-sec",
        type=_non_negative_float,
        default=DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
        help=(
            "Time constant in seconds for the bounded causal low-pass applied to "
            "target velocity inferred from accepted position qrefs. Defaults to 0.03 s; "
            "set 0 to reproduce the unfiltered planner behavior. The option is consumed "
            "only by the acceleration-limited position planner."
        ),
    )
    parser.add_argument(
        "--position-degraded-speed-scale",
        type=_alpha_float,
        default=DEFAULT_POSITION_DEGRADED_SPEED_SCALE,
        help=(
            "Desired speed scale for valid but anomalous position targets. Defaults to 0.50; "
            "manual/collision/no-valid-solution holds use immediate software quick stop instead."
        ),
    )
    parser.add_argument(
        "--position-max-acceleration-rad-s2",
        type=_parse_arm_max_angvel,
        default=DEFAULT_POSITION_MAX_ACCELERATION_RAD_S2,
        help=(
            "Scalar or 6 comma-separated acceleration limits for the optional position "
            "trajectory planner in rad/s^2. Defaults to 7,7,9,15,15,20 and is not "
            "scaled down when --position-degraded-speed-scale is active."
        ),
    )
    parser.add_argument(
        "--position-max-command-step-deg",
        type=_positive_float,
        default=None,
        help=(
            "Optional uniform per-joint command-step envelope, in degrees, applied by the "
            "high-rate acceleration-limited position planner before the final follow EMA. "
            "It is independent of --arm-max-angvel; omit it to retain the existing "
            "max_angvel * command_dt envelope. Position mode only."
        ),
    )
    parser.add_argument(
        "--arm-control-hz",
        type=_positive_float,
        default=None,
        help=(
            "RM65 backend control/streaming rate after each outer action update. "
            "Defaults to 40 Hz for impedance and 150 Hz for position control."
        ),
    )
    parser.add_argument(
        "--arm-max-angvel",
        type=_parse_arm_max_angvel,
        default=(0.7, 0.7, 0.9, 1.5, 1.5, 2.0),
        help="Scalar or 6 comma-separated arm joint velocity limits in rad/s.",
    )
    parser.add_argument(
        "--minimum-control-hz",
        type=_positive_float,
        default=15.0,
        help=(
            "Report low-control-rate faults below this threshold. If omitted, real teleop uses "
            "min(15, 0.9 * effective outer control Hz) so normal scheduler jitter is not flagged."
        ),
    )
    parser.add_argument(
        "--status-sec",
        type=_non_negative_float,
        default=0.0,
        help=(
            "Terminal status heartbeat interval in seconds. Use 0 (the default) to print "
            "only state changes, faults, and lifecycle events."
        ),
    )
    parser.add_argument(
        "--debug-orientation-console",
        action="store_true",
        help=(
            "Print periodic raw/retargeted/preview wrist-orientation diagnostics while not "
            "recording. Disabled by default to keep real fault output visible."
        ),
    )
    parser.add_argument(
        "--no-debug-disk-write",
        "--disable-debug-disk-write",
        dest="no_debug_disk_write",
        action="store_true",
        help=(
            "Disable all standalone debug disk outputs for this run: control/action/timing/IK "
            "sidecars, legacy debug session JSONL, fix-arm pose snapshots, and lightweight "
            "debug trajectories. Console/timing instrumentation and formal dataset collection "
            "remain available. This switch overrides individual debug-write flags."
        ),
    )
    parser.add_argument(
        "--control-debug",
        choices=CONTROL_DEBUG_LEVELS,
        default="off",
        help=(
            "Observational runtime control-audit level: off, summary, or full. It never changes "
            "safety, smoothing, hold, IK, or dispatch behavior."
        ),
    )
    parser.add_argument(
        "--control-debug-root",
        type=str,
        default=str(DEFAULT_CONTROL_DEBUG_ROOT),
        help="Output root for runtime control-audit records.",
    )
    parser.add_argument(
        "--control-debug-queue-size",
        type=_positive_int,
        default=DEFAULT_CONTROL_DEBUG_QUEUE_SIZE,
        help="Bounded queue capacity for non-blocking control-audit records.",
    )
    parser.add_argument(
        "--control-debug-flush-every",
        type=_positive_int,
        default=DEFAULT_CONTROL_DEBUG_FLUSH_EVERY,
        help="Control-audit writer flush interval in rows.",
    )
    parser.add_argument(
        "--control-debug-episode-sidecar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Attach a second episode-scoped position audit during formal collection. "
            "Disable it to retain session audit data without duplicate JSON work."
        ),
    )
    parser.add_argument(
        "--debug-timing",
        "--debug-frozen",
        dest="debug_timing",
        action="store_true",
        help=(
            "Compatibility switch that enables all timing debug modules: console slow-frame "
            "reports, JSONL timing log, RM65 backend timing, IK profiling, and the main-loop "
            "watchdog. Prefer the narrower --debug-timing-* flags for routine diagnosis."
        ),
    )
    parser.add_argument(
        "--debug-timing-console",
        action="store_true",
        help=(
            "Print slow-frame real-loop phase/latency reports without enabling timing "
            "JSONL or backend profiling."
        ),
    )
    parser.add_argument(
        "--debug-timing-log",
        "--debug-timing-light-log",
        dest="debug_timing_log",
        action="store_true",
        help="Write sampled real-loop timing JSONL frames under teleop_records/debug_timing.",
    )
    parser.add_argument(
        "--no-debug-timing-auto-log",
        "--no-debug-timing-log",
        dest="debug_timing_auto_log",
        action="store_false",
        help=(
            "Do not automatically start the JSONL timing log for narrow timing modules "
            "such as --debug-timing-ik or --debug-timing-rm65. Explicit --debug-timing-log "
            "and legacy --debug-timing still write the JSONL log."
        ),
    )
    parser.add_argument(
        "--debug-timing-rm65",
        "--rm65-backend-debug-timing",
        dest="debug_timing_rm65",
        action="store_true",
        help=(
            "Enable RM65 impedance backend timing instrumentation and start the sampled "
            "JSONL timing log unless --no-debug-timing-auto-log is set."
        ),
    )
    parser.add_argument(
        "--debug-action",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record extra action pose source/timing diagnostics to an independent JSONL log. "
            "When formal collection is active, the same diagnostics are also added to sample_metadata_json."
        ),
    )
    parser.add_argument(
        "--debug-action-log-root",
        type=str,
        default=str(DEFAULT_ACTION_DEBUG_LOG_ROOT),
        help="Directory for the standalone --debug-action JSONL log.",
    )
    parser.add_argument(
        "--debug-action-flush-every",
        type=_positive_int,
        default=1,
        help="Flush the standalone --debug-action JSONL log every N frames.",
    )
    parser.add_argument(
        "--debug-action-queue-size",
        type=_positive_int,
        default=2048,
        help="Maximum queued standalone --debug-action rows before frames are dropped.",
    )
    parser.add_argument(
        "--debug-action-sample-hz",
        type=_non_negative_float,
        default=0.0,
        help=(
            "Maximum standalone action-debug frame rate. 0 preserves every-outer-loop "
            "recording; formal dataset sampling keeps its own sample clock."
        ),
    )
    parser.add_argument(
        "--debug-timing-ik",
        dest="debug_timing_ik",
        action="store_true",
        help=(
            "Enable IK timing/profile collection as a timing debug module. "
            "This starts the sampled JSONL timing log unless --no-debug-timing-auto-log "
            "is set. --debug-ik-timing remains available for GUI/status-only IK diagnostics."
        ),
    )
    parser.add_argument(
        "--debug-timing-watchdog",
        action="store_true",
        help="Enable the background main-loop stall watchdog without enabling other timing logs.",
    )
    parser.add_argument(
        "--debug-timing-threshold-sec",
        "--debug-frozen-threshold-sec",
        dest="debug_timing_threshold_sec",
        type=_positive_float,
        default=1.0 / DEFAULT_TELEOP_SAMPLE_HZ,
        help="Slow-frame threshold for --debug-timing diagnostics; default matches one outer action period.",
    )
    parser.add_argument(
        "--debug-timing-report-interval-sec",
        "--debug-frozen-report-interval-sec",
        dest="debug_timing_report_interval_sec",
        type=_positive_float,
        default=1.0,
        help="Minimum seconds between repeated --debug-timing reports.",
    )
    parser.add_argument(
        "--debug-timing-log-sample-hz",
        type=_non_negative_float,
        default=0.0,
        help=(
            "Maximum routine timing-JSONL frame rate. 0 preserves unbounded slow-frame "
            "logging; state and fault transitions always bypass this limit."
        ),
    )
    parser.add_argument(
        "--debug-ik-timing",
        action="store_true",
        help=(
            "Enable IK timing diagnostics in status metadata and the browser GUI. "
            "This turns on the dual-arm retargeter IK profile without enabling full loop timing logs."
        ),
    )
    parser.add_argument(
        "--debug-ik-timing-print-every",
        type=_non_negative_int,
        default=0,
        help=(
            "Print one dual-arm IK timing profile every N retargeter frames when --debug-ik-timing is enabled. "
            "0 keeps the diagnostics in status/GUI only."
        ),
    )
    parser.add_argument(
        "--ik-debug-log-full",
        action="store_true",
        help=(
            "Keep the GUI IK debug log in the historical full-detail format. "
            "By default the log is compact to avoid adding main-loop latency while diagnosing teleop lag."
        ),
    )
    parser.add_argument(
        "--ik-time-budget-ms",
        type=_non_negative_float,
        default=None,
        help=(
            "Optional IK frame time budget in milliseconds. Omit to use one control period; "
            "pass 0 to disable over-budget warnings."
        ),
    )
    parser.add_argument(
        "--position-ik-orientation-weight",
        type=_positive_finite_float_arg,
        default=None,
        help=(
            "Position mode only: override both single- and dual-arm IK orientation "
            "weights while fixing the corresponding position weights at 1.0. "
            "Omit to preserve the selected IK tuning JSON."
        ),
    )
    parser.add_argument(
        "--position-ik-solve-max-iterations",
        "--ik-solve-max-iterations",
        "--solve-max-iterations",
        dest="position_ik_solve_max_iterations",
        type=_positive_int,
        default=2,
        help=(
            "Position-mode dual-arm IK solve iteration upper bound (default 2). "
            "This option never changes the force/impedance IK path."
        ),
    )
    parser.add_argument(
        "--position-ik-solve-min-iterations",
        type=_positive_int,
        default=2,
        help=(
            "Position-mode dual-arm IK hard minimum iteration count (default 2). "
            "The time budget may stop only later optional iterations; force/impedance remains at 1."
        ),
    )
    parser.add_argument(
        "--position-ik-damping",
        type=_positive_finite_float_arg,
        default=0.03,
        help=(
            "Position-mode Pink QP damping (default 0.03). "
            "Force/impedance keeps the historical 0.01."
        ),
    )
    parser.add_argument(
        "--position-ik-adaptive-iterations",
        dest="position_ik_adaptive_iterations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Position mode only: when enabled, run up to --position-ik-solve-max-iterations "
            "while the IK frame budget still has time after the safety-selection reserve."
        ),
    )
    parser.add_argument(
        "--position-ik-iteration-reserve-ms",
        type=_non_negative_float,
        default=5.0,
        help=(
            "Position adaptive-iteration time reserved for candidate/safety selection after "
            "IK solving. Defaults to 5 ms, matching the existing candidate-selection budget."
        ),
    )
    parser.add_argument(
        "--position-ik-local-step-clip-scale",
        "--position-ik-local-step-scale",
        dest="position_ik_local_step_clip_scale",
        type=_local_step_clip_scale_arg,
        default=1.0,
        help=(
            "Position mode only: multiplier for the candidate-local IK step clip used "
            "during safe-branch selection. Default 1 preserves behavior; the final qref "
            "execution velocity cap is unchanged."
        ),
    )
    parser.add_argument(
        "--position-prefer-non-wrist-singular-branch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Position mode only: prefer a feasible IK candidate with fewer RM65 wrist "
            "singular sides. If none exists, accept the best singular candidate at normal "
            "speed. Use --no-position-prefer-non-wrist-singular-branch to atomically "
            "restore the previous branch ordering and backend hold-entry gate."
        ),
    )
    parser.add_argument(
        "--position-prefer-non-elbow-singular-branch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Position mode only: prefer collision-safe candidates outside the RM65 "
            "elbow-singularity neighborhood. If no non-singular candidate is feasible "
            "within the frame budget, explicitly allow the selected elbow-singular "
            "candidate through the backend SDK gate. Use "
            "--no-position-prefer-non-elbow-singular-branch to restore the previous "
            "hard elbow hold-entry behavior."
        ),
    )
    parser.add_argument(
        "--position-inactive-enter-threshold-scale",
        type=_positive_unit_scale_arg,
        default=1.0,
        help=(
            "Position mode only: tighten tracker INACTIVE entry evidence thresholds. "
            "1 preserves the current thresholds; 0.5 requires linear/angular speed "
            "and translation/rotation span to be at most half their defaults. "
            "Exit thresholds and dwell times are unchanged."
        ),
    )
    parser.add_argument(
        "--position-joint-velocity-limit-profile",
        type=str,
        default="rm65",
        help=(
            "Position mode only: joint-velocity profile used by the 60 Hz accepted-qref "
            "final execution-step cap. Default rm65 preserves existing behavior. "
            "The rm65_position_2x/4x/8x profiles are explicit position-only test levels; "
            "the per-arm --arm-max-angvel step remains an independent upper bound."
        ),
    )
    parser.add_argument(
        "--dual-arm-ik-hz",
        type=_positive_float,
        default=None,
        help=(
            "Dual-arm IK solver nominal frequency/step budget, not a separate worker clock. "
            "Actual IK calls follow the outer loop. Defaults to --position-reference-hz in "
            "position mode and --impedance-teleop-control-hz in impedance mode; normally keep them equal."
        ),
    )
    parser.add_argument(
        "--left-right-base-distance",
        type=float,
        default=ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
        help="Annotated dual-arm base distance used by the SAPIEN scene and default calibration.",
    )
    parser.add_argument(
        "--arm-workspace-config",
        type=str,
        default=str(DEFAULT_ARM_WORKSPACE_CONFIG_PATH),
        help=(
            "Desk-frame arm workspace JSON config. Defaults to "
            "teleop_runtime/configs/arm_workspace.json; pass 'none' to skip it."
        ),
    )
    parser.add_argument(
        "--arm-workspace-min",
        nargs="+",
        default=None,
        help=(
            "Runtime override for both arms' desk-frame workspace minimum xyz. "
            "Accepts either three values or --arm-workspace-min=x,y,z."
        ),
    )
    parser.add_argument(
        "--arm-workspace-max",
        nargs="+",
        default=None,
        help=(
            "Runtime override for both arms' desk-frame workspace maximum xyz. "
            "Accepts either three values or --arm-workspace-max=x,y,z."
        ),
    )
    parser.add_argument(
        "--left-arm-workspace-min",
        nargs="+",
        default=None,
        help="Runtime override for the left arm desk-frame workspace minimum xyz.",
    )
    parser.add_argument(
        "--left-arm-workspace-max",
        nargs="+",
        default=None,
        help="Runtime override for the left arm desk-frame workspace maximum xyz.",
    )
    parser.add_argument(
        "--right-arm-workspace-min",
        nargs="+",
        default=None,
        help="Runtime override for the right arm desk-frame workspace minimum xyz.",
    )
    parser.add_argument(
        "--right-arm-workspace-max",
        nargs="+",
        default=None,
        help="Runtime override for the right arm desk-frame workspace maximum xyz.",
    )
    parser.add_argument("--left-tracker-name", type=str, default="tracker_1", help="Left wrist tracker device name")
    parser.add_argument("--right-tracker-name", type=str, default="tracker_2", help="Right wrist tracker device name")
    parser.add_argument(
        "--aux-tracker-name",
        type=str,
        default="",
        help="Optional auxiliary tracker name. Empty by default; pass tracker_3 to enable a third tracker.",
    )
    parser.add_argument(
        "--gui",
        dest="open_browser_gui",
        action="store_true",
        help=(
            "Open the browser controls for VIVE tracker swap/start. "
            "SAPIEN rendering stays disabled unless --show-viewer or --no-skip-renderer is passed."
        ),
    )
    parser.set_defaults(open_browser_gui=True)
    parser.add_argument(
        "--show-viewer",
        dest="headless",
        action="store_false",
        help="Show the SAPIEN real-teleop viewer window; also requires --no-skip-renderer.",
    )
    parser.add_argument(
        "--no-gui",
        "--headless",
        dest="headless",
        action="store_true",
        help="Disable the browser controls and the SAPIEN viewer window.",
    )
    parser.set_defaults(headless=True)
    parser.add_argument(
        "--skip-renderer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip SAPIEN renderer creation. Enabled by default to reduce real-teleop latency.",
    )
    parser.add_argument(
        "--calibration-timeout",
        type=_positive_float,
        default=10.0,
        help="Timeout for runtime home calibration after the receiver starts.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=20,
        help="Number of valid wrist frames to average per side during runtime home calibration.",
    )
    parser.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="Start runtime home calibration immediately without waiting for a key press.",
    )
    parser.add_argument(
        "--keyboard-clutch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--record",
        dest="no_record",
        action="store_false",
        help=(
            "Enable legacy debug session JSONL recording. This writes raw_inputs/faults/"
            "state_transitions/calibration under --session-record-root when provided, or "
            "teleop_records/<timestamp> otherwise."
        ),
    )
    parser.add_argument(
        "--no-record",
        dest="no_record",
        action="store_true",
        help="Disable legacy debug session JSONL recording. This is the default for real teleop.",
    )
    parser.set_defaults(no_record=True)
    parser.add_argument(
        "--no-session-record-commands",
        action="store_true",
        help=(
            "Do not write the synchronous debug commands.jsonl stream. Formal LeRobot dataset "
            "recording is unaffected."
        ),
    )
    parser.add_argument(
        "--session-record-flush-every",
        type=_positive_int,
        default=1,
        help=(
            "Flush debug session JSONL streams every N rows. Default 1 preserves the previous "
            "flush-after-each-row behavior."
        ),
    )
    parser.add_argument(
        "--session-record-compact-json",
        action="store_true",
        help="Write compact debug session JSONL without spaces to reduce serialization and disk volume.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help=(
            "Base output directory for formal LeRobot dataset collection. Actual dataset path is "
            "<dataset-root>/<YYYY-MM-DD>_<collection-task-name>."
        ),
    )
    parser.add_argument(
        "--collection-task-name",
        type=str,
        default="",
        help="Task name used for formal dataset folder naming and metadata.",
    )
    parser.add_argument(
        "--collection-task-name-zh",
        type=str,
        default="",
        help="Optional Chinese task name written into dataset/session metadata and shown in the GUI.",
    )
    parser.add_argument(
        "--collection-task-id",
        type=str,
        default="",
        help="Optional formal dataset task id. Defaults to a slug from --collection-task-name.",
    )
    parser.add_argument(
        "--collection-task-description",
        type=str,
        default="",
        help="Optional formal dataset task description.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Optional formal dataset name written into meta/info.json.",
    )
    parser.add_argument(
        "--session-record-root",
        type=str,
        default="",
        help=(
            "Optional base directory for legacy debug JSONL session output when --record is set. "
            "Session JSONL is written to <session-record-root>/<timestamp>_vive_rm65_real_teleop."
        ),
    )
    parser.add_argument(
        "--no-collection",
        action="store_true",
        help="Disable formal dataset collection; GUI episode buttons will only use debug trajectory recording.",
    )
    parser.add_argument("--write-parquet", dest="write_parquet", action="store_true")
    parser.add_argument("--no-parquet", dest="write_parquet", action="store_false")
    parser.set_defaults(write_parquet=True)
    parser.add_argument(
        "--policy-io-layout",
        type=str,
        default="",
        help=(
            "Optional DexTeleop v2.2 policy-I/O layout JSON, or literal 'auto' to resolve a "
            "strict layout from the instantiated backend/retargeter pairs. It is opt-in and "
            "leaves the existing real-teleop command path unchanged."
        ),
    )
    parser.add_argument(
        "--pose-frame-contract",
        type=load_pose_frame_contract,
        default=None,
        metavar="JSON",
        help=(
            "Optional explicit v2.3 pose-frame contract JSON for formal collection. "
            "Requires --policy-io-layout auto or a layout JSON. It changes only detached "
            "dataset rows, never the live teleop command path."
        ),
    )
    parser.add_argument("--write-video", dest="write_video", action="store_true")
    parser.add_argument("--no-video", dest="write_video", action="store_false")
    parser.set_defaults(write_video=True)
    parser.add_argument(
        "--wrist-depth-mask",
        "--wrsit-depth-mask",
        dest="wrist_depth_mask",
        action="store_true",
        help=(
            "Save 15 Hz bit-packed 0/1 depth-range masks for left/right D405 wrist cameras only. "
            "The front camera depth stream is not enabled by this option."
        ),
    )
    parser.add_argument(
        "--wrist-depth-raw",
        "--wrsit-depth-raw",
        dest="wrist_depth_raw",
        action="store_true",
        help=(
            "Debug mode: save 15 Hz aligned raw z16 depth for left/right D405 wrist cameras only. "
            "This can be large; the front camera depth stream is not enabled."
        ),
    )
    parser.add_argument(
        "--wrist-depth-min-m",
        type=_positive_float,
        default=DEFAULT_WRIST_DEPTH_MIN_M,
        help="Minimum metric depth included in --wrist-depth-mask.",
    )
    parser.add_argument(
        "--wrist-depth-max-m",
        type=_positive_float,
        default=DEFAULT_WRIST_DEPTH_MAX_M,
        help="Maximum metric depth included in --wrist-depth-mask.",
    )
    parser.add_argument(
        "--wrist-depth-stream-fps",
        type=_positive_int,
        default=DEFAULT_WRIST_DEPTH_STREAM_FPS,
        help=(
            "RealSense depth stream fps for left/right D405 wrist cameras. "
            "Saved wrist-depth mask/raw frames follow the 15 Hz collection sample rate."
        ),
    )
    parser.add_argument(
        "--wrist-cam-width",
        dest="wrist_cam_width",
        type=_positive_int,
        default=None,
        help=(
            "Requested native RealSense stream width for left/right D405 wrist cameras when "
            "wrist depth is enabled. D405 RGB and depth are started at the same resolution. "
            "Defaults to the configured wrist RGB stream width."
        ),
    )
    parser.add_argument(
        "--wrist-depth-width",
        dest="wrist_cam_width",
        type=_positive_int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--wrist-cam-height",
        dest="wrist_cam_height",
        type=_positive_int,
        default=None,
        help=(
            "Requested native RealSense stream height for left/right D405 wrist cameras when "
            "wrist depth is enabled. D405 RGB and depth are started at the same resolution. "
            "Defaults to the configured wrist RGB stream height."
        ),
    )
    parser.add_argument(
        "--wrist-depth-height",
        dest="wrist_cam_height",
        type=_positive_int,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--wrist-depth-align-to-color",
        dest="wrist_depth_align_to_color",
        action="store_true",
        default=True,
        help="Align wrist depth frames to the RGB stream before saving mask/raw depth.",
    )
    parser.add_argument(
        "--wrist-depth-no-align-to-color",
        dest="wrist_depth_align_to_color",
        action="store_false",
        help=(
            "Do not align wrist depth to RGB. Use with --wrist-cam-width/--wrist-cam-height "
            "when the saved mask/raw stream should stay at the lower wrist stream resolution."
        ),
    )
    parser.add_argument(
        "--tactile-storage",
        choices=("external", "parquet"),
        default="external",
        help="How high-dimensional tactile frames are stored in the formal dataset.",
    )
    parser.add_argument("--external-tactile", dest="tactile_storage", action="store_const", const="external")
    parser.add_argument(
        "--tactile-compression",
        choices=("zstd", "lz4", "none"),
        default="none",
        help=(
            "Compression for external tactile streams. none is the default and writes "
            "raw .bin streams; use lz4 or zstd for lossless chunk compression."
        ),
    )
    parser.add_argument(
        "--writer-workers",
        type=_positive_int,
        default=4,
        help="Number of async formal dataset writer workers.",
    )
    parser.add_argument(
        "--collection-worker-mode",
        choices=("auto", "process", "thread"),
        default="auto",
        help=(
            "Formal-collection execution boundary. auto uses a spawned process for "
            "position teleop and preserves the existing thread pipeline for force/impedance. "
            "thread is the rollback path."
        ),
    )
    parser.add_argument(
        "--collection-process-queue-size",
        type=_positive_int,
        default=4,
        help=(
            "Bounded whole-sample queue for process-isolated position collection. "
            "Overload pauses collection and never blocks motion."
        ),
    )
    parser.add_argument(
        "--collection-process-shared-slots",
        type=_positive_int,
        default=4,
        help="Preallocated shared-memory camera slots for process-isolated collection.",
    )
    parser.add_argument(
        "--collection-process-extra-array-mb",
        type=_positive_int,
        default=8,
        help=(
            "Extra bytes per shared slot, in MiB, for tactile and other nested ndarray "
            "payloads. These arrays bypass multiprocessing.Queue pickle."
        ),
    )
    parser.add_argument(
        "--collection-worker-cpus",
        type=_dmtac_worker_cpu_affinity,
        default=(),
        help=(
            "Optional comma-separated CPU affinity for the formal collection process. "
            "Leave empty for scheduler default."
        ),
    )
    parser.add_argument(
        "--collection-worker-nice",
        type=_dmtac_worker_nice,
        default=5,
        help=(
            "Nice increment for the process-isolated collection worker, in [0,19]. "
            "Defaults to 5 so control remains preferred."
        ),
    )
    parser.add_argument(
        "--writer-queue-size",
        type=_positive_int,
        default=512,
        help="Max async formal dataset writer queue size.",
    )
    parser.add_argument(
        "--storage-queue-size",
        type=_non_negative_int,
        default=64,
        help=(
            "Per-camera video storage queue size before the async writer pool. "
            "When full, the oldest non-critical video frame is dropped."
        ),
    )
    parser.add_argument(
        "--collection-warning-hz",
        type=_positive_float,
        default=15.0,
        help="GUI warning threshold for actual dataset recording rate. Defaults to 15 Hz.",
    )
    parser.add_argument(
        "--record-actual-wrist-camera-pose",
        "--record-actual-wrist-pose",
        dest="record_actual_wrist_camera_pose",
        action="store_true",
        default=True,
        help=(
            "Annotate each dataset frame with the latest background-polled RM65 feedback "
            "TCP pose as observation.wrist_camera.* without changing training state/action fields. "
            "Enabled by default; pass --no-record-actual-wrist-camera-pose to disable."
        ),
    )
    parser.add_argument(
        "--no-record-actual-wrist-camera-pose",
        "--no-record-actual-wrist-pose",
        dest="record_actual_wrist_camera_pose",
        action="store_false",
    )
    parser.add_argument(
        "--actual-wrist-camera-pose-hz",
        "--actual-wrist-pose-hz",
        type=_positive_float,
        default=30.0,
        help="Background feedback polling rate for actual wrist camera pose annotation.",
    )
    parser.add_argument(
        "--actual-wrist-camera-pose-max-age-sec",
        "--actual-wrist-pose-max-age-sec",
        type=_positive_float,
        default=0.2,
        help="Mark actual wrist camera pose annotations stale when their sample-time age exceeds this value.",
    )
    parser.add_argument(
        "--no-episode-recording",
        action="store_true",
        help="Disable lightweight debug trajectory recording support for this run.",
    )
    parser.add_argument(
        "--debug-trajectory-recording",
        action="store_true",
        help=(
            "Also save lightweight debug trajectory episode files. Formal LeRobot dataset "
            "collection remains independent; debug trajectories still default on when "
            "--no-collection is used."
        ),
    )
    parser.add_argument(
        "--episode-root",
        type=str,
        default="",
        help=(
            "Optional directory used to store episode_XXXXXX subdirectories. "
            "Providing this option also enables lightweight debug trajectory recording; "
            "it overrides --dataset-root for episode trajectory files only."
        ),
    )
    parser.add_argument(
        "--record-auto-start",
        action="store_true",
        help="Start an episode automatically when the control loop begins.",
    )
    parser.add_argument(
        "--load-profile",
        type=str,
        default="",
        help=(
            "Calibration profile to load before runtime home calibration. "
            "When omitted, the newest JSON under teleop_records/calibration_profiles is loaded."
        ),
    )
    parser.add_argument(
        "--load-initial-alignment",
        type=str,
        default="",
        help=(
            "Optional initial-alignment JSON used only for startup joint mismatch validation. "
            "Runtime home capture still uses the live real-arm pose at 'p' time."
        ),
    )
    parser.add_argument(
        "--ik-tuning-json",
        type=str,
        default="",
        help="Optional RM65 IK tuning JSON. Defaults to robot_motion/configs/rm65_ik_tuning.json.",
    )
    parser.add_argument(
        "--rm65-hardware-read-retries",
        type=_non_negative_int,
        default=0,
        help=(
            "Extra retry count for each Get_Current_Arm_State call. "
            "Default 0 avoids retry-induced control-rate collapse when the controller returns transient nonzero codes."
        ),
    )
    parser.add_argument(
        "--rm65-hardware-read-retry-delay-sec",
        type=_positive_float,
        default=0.02,
        help="Delay inserted between repeated Get_Current_Arm_State retries.",
    )
    parser.add_argument(
        "--arm-io-threads",
        type=_non_negative_int,
        default=2,
        help=(
            "Worker count used for left/right arm state refresh and dispatch. "
            "Set 0 or 1 to force sequential IO."
        ),
    )
    parser.add_argument(
        "--disable-arm-step-dt-scaling",
        dest="arm_step_dt_scaling",
        action="store_false",
        help="Disable real-arm step scaling based on measured send dt.",
    )
    parser.set_defaults(arm_step_dt_scaling=True)
    parser.add_argument(
        "--arm-step-dt-alpha",
        type=_alpha_float,
        default=0.25,
        help="EMA factor used when estimating real-arm send dt for step scaling.",
    )
    parser.add_argument(
        "--arm-step-dt-max-scale",
        type=_positive_float,
        default=3.0,
        help=(
            "Maximum multiplier applied to arm step limits when real send dt runs slower than target. "
            "This applies to position control; impedance force control keeps its outer IK step at "
            "the nominal control dt. "
            "If omitted, real teleop uses max(default, 1.1 * arm_control_hz / effective_outer_hz) "
            "so a slower outer reference loop with a faster inner arm stream is not capped as if too few "
            "inner cycles elapsed."
        ),
    )
    parser.add_argument(
        "--arm-step-dt-stall-reset-sec",
        type=_positive_float,
        default=0.25,
        help="Reset actual-dt scaling back to the nominal control dt after longer send stalls.",
    )
    parser.add_argument(
        "--arm-singularity-protection",
        dest="arm_singularity_protection",
        action="store_true",
        help="Enable RM65 non-impedance backend singularity entry gate.",
    )
    parser.add_argument(
        "--no-arm-singularity-protection",
        dest="arm_singularity_protection",
        action="store_false",
        help=(
            "Disable the RM65 non-impedance backend singularity entry gate. "
            "Use only for diagnostics; this does not affect impedance mode."
        ),
    )
    parser.set_defaults(arm_singularity_protection=True)
    parser.add_argument(
        "--max-start-joint-error-deg",
        type=_positive_float,
        default=5.0,
        help="Abort teleop startup if any current joint differs from the initial-alignment qpos by more than this threshold.",
    )
    parser.add_argument(
        "--max-start-weighted-error-deg",
        type=_positive_float,
        default=12.0,
        help="Abort teleop startup if the weighted 6D start-state error exceeds this threshold.",
    )
    parser.add_argument(
        "--ignore-start-mismatch",
        action="store_true",
        help="Allow teleop startup to continue even when the initial-alignment start-state check fails.",
    )
    parser.add_argument(
        "--recovery-healthy-frames",
        type=int,
        default=8,
        help=(
            "Number of consecutive fault-free control frames required before auto recovery starts. "
            "During FROZEN, RM65 real teleop stops new command dispatch and then enters RECOVERING "
            "after this many healthy frames."
        ),
    )
    parser.add_argument(
        "--recovery-ramp-sec",
        type=_positive_float,
        default=1.0,
        help="Absolute-control recovery ramp duration.",
    )
    parser.add_argument(
        "--tracker-auto-hold-sec",
        type=_positive_float,
        default=0.10,
        help="Automatically hold arm motion after a tracker target is unavailable for this long.",
    )
    parser.add_argument(
        "--tracker-manual-hold-sec",
        type=_positive_float,
        default=1.0,
        help=(
            "Escalate tracker loss to a latched manual hold after this long; "
            "runtime home must then be captured again. The default leaves a 0.1 s "
            "automatic-hold/recovery window before escalating at 1.0 s."
        ),
    )
    parser.add_argument(
        "--max-tracker-translation-jump-m",
        type=_positive_float,
        default=DEFAULT_REAL_TELEOP_MAX_TRACKER_TRANSLATION_JUMP_M,
        help=(
            "Maximum single-frame VIVE tracker translation discontinuity before FROZEN. "
            "This is intentionally larger than normal fast hand motion; lower it only when "
            "tracker teleport jumps slip through."
        ),
    )
    parser.add_argument(
        "--max-tracker-rotation-jump-deg",
        type=_positive_float,
        default=DEFAULT_REAL_TELEOP_MAX_TRACKER_ROTATION_JUMP_DEG,
        help=(
            "Maximum single-frame VIVE tracker rotation discontinuity before FROZEN. "
            "Large fast wrist rotations should be allowed; near-flip discontinuities should freeze."
        ),
    )
    parser.add_argument("--max-steps", type=_non_negative_int, default=0, help="Stop after N control steps, 0 means infinite")
    raw_argv = list(sys.argv[1:])
    args = parser.parse_args()
    if not str(getattr(args, "dataset_root", "")).strip():
        parser.error(
            "--dataset-root must not be empty; omit it to use the configured default."
        )
    raw_arg_set = set(raw_argv)
    if bool(getattr(args, "no_debug_disk_write", False)):
        args.control_debug = "off"
        args.debug_action = False
        args.debug_timing_log = False
        args.debug_timing_auto_log = False
        args.no_record = True
        args.no_episode_recording = True
        args.debug_trajectory_recording = False
    try:
        args.arm_execution_config = resolve_real_arm_execution_config(
            args,
            raw_argv=raw_argv,
            impedance_default_outer_control_hz=DEFAULT_IMPEDANCE_TELEOP_CONTROL_HZ,
        )
        args.control_debug_config = resolve_control_debug_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    # A named profile may select a mode without requiring the legacy mode flag.
    # Fine-grained legacy fields retain their original raw values so callers can
    # still distinguish explicit overrides from mode-derived defaults.
    args.arm_control_mode = args.arm_execution_config.mode
    requested_collection_worker_mode = str(args.collection_worker_mode)
    args.collection_worker_mode_resolved = (
        "process"
        if requested_collection_worker_mode == "auto"
        and args.arm_control_mode == "position"
        else "thread"
        if requested_collection_worker_mode == "auto"
        else requested_collection_worker_mode
    )
    if (
        args.collection_worker_mode_resolved == "process"
        and args.arm_control_mode != "position"
    ):
        parser.error(
            "--collection-worker-mode process is currently position-only; "
            "force/impedance collection intentionally retains the established thread path."
        )
    if int(args.position_ik_solve_min_iterations) > int(
        args.position_ik_solve_max_iterations
    ):
        parser.error(
            "--position-ik-solve-min-iterations must be <= "
            "--position-ik-solve-max-iterations."
        )
    if args.arm_control_mode != "position" and (
        args.position_ik_orientation_weight is not None
        or int(args.position_ik_solve_max_iterations) != 2
        or int(args.position_ik_solve_min_iterations) != 2
        or float(args.position_ik_damping) != 0.03
        or bool(args.position_ik_adaptive_iterations)
        or float(args.position_ik_iteration_reserve_ms) != 5.0
        or float(args.position_ik_local_step_clip_scale) != 1.0
        or str(args.position_joint_velocity_limit_profile) != "rm65"
        or float(args.position_inactive_enter_threshold_scale) != 1.0
    ):
        parser.error(
            "--position-ik-orientation-weight, --position-ik-solve-max-iterations, "
            "--position-ik-solve-min-iterations, --position-ik-damping, "
            "--position-ik-adaptive-iterations, --position-ik-iteration-reserve-ms, "
            "--position-ik-local-step-clip-scale, and "
            "--position-joint-velocity-limit-profile, and "
            "--position-inactive-enter-threshold-scale are position-mode options and "
            "cannot alter force/impedance IK."
        )
    args.position_command_planner = normalize_position_command_planner(
        args.position_command_planner
    )
    if (
        args.position_command_planner
        == POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED
        and args.arm_control_mode != "position"
    ):
        parser.error(
            "--position-command-planner acceleration_limited requires position arm control mode."
        )
    if args.position_max_command_step_deg is not None and args.arm_control_mode != "position":
        parser.error(
            "--position-max-command-step-deg is a position-mode option and cannot alter "
            "force/impedance control."
        )
    if (
        args.position_max_command_step_deg is not None
        and args.position_command_planner
        != POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED
    ):
        parser.error(
            "--position-max-command-step-deg requires "
            "--position-command-planner acceleration_limited."
        )
    if (
        args.position_command_planner
        == POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED
        and not args.arm_execution_config.position_background_streaming_enabled
    ):
        parser.error(
            "--position-command-planner acceleration_limited requires the high-rate position "
            "background stream."
        )
    if (
        args.position_command_planner
        == POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED
        and float(args.arm_execution_config.position_background_streaming_hz)
        < float(args.arm_execution_config.position_reference_hz)
    ):
        parser.error(
            "--position-command-planner acceleration_limited requires the final command "
            "rate to be at least the position reference rate; otherwise safe references "
            "can be preempted before any qcmd sample is emitted."
        )
    for compatibility_warning in args.arm_execution_config.compatibility_warnings:
        print(f"WARNING: arm execution profile: {compatibility_warning}", file=sys.stderr)
    _apply_real_loop_timing_defaults(args, raw_argv=raw_argv)
    args._dataset_name_explicit = _argv_has_option(raw_argv, "--dataset-name")
    try:
        _resolve_movement_multiple_args(args, raw_arg_set=raw_arg_set)
        _resolve_workspace_override_args(args)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    _apply_formal_dataset_layout(args)
    if args.pose_frame_contract is not None and not str(args.policy_io_layout).strip():
        parser.error(
            "--pose-frame-contract requires --policy-io-layout auto or an explicit layout JSON."
        )
    if args.pose_frame_contract is not None:
        if not bool(args.record_actual_wrist_camera_pose):
            parser.error(
                "--pose-frame-contract requires measured arm feedback; remove "
                "--no-record-actual-wrist-camera-pose."
            )
        contract_sides = tuple(args.pose_frame_contract.source.active_sides)
        requested_sides = _active_sides_from_args(args)
        if contract_sides != requested_sides:
            parser.error(
                "--pose-frame-contract active_sides disagree with collection --active-sides: "
                f"{contract_sides!r} != {requested_sides!r}."
            )
    args.gui_requested = bool(getattr(args, "open_browser_gui", True)) and not (
        "--no-gui" in raw_arg_set or "--headless" in raw_arg_set
    )
    if not bool(args.headless):
        args.skip_renderer = False
    if int(args.tactile_vis_size) <= 0:
        parser.error("--tactile-vis-size must be positive.")
    if bool(args.no_camera) and bool(args.vis_cam):
        parser.error("--vis_cam requires cameras; remove --no-camera.")
    if float(args.wrist_depth_max_m) <= float(args.wrist_depth_min_m):
        parser.error("--wrist-depth-max-m must be greater than --wrist-depth-min-m.")
    wrist_cam_width = getattr(args, "wrist_cam_width", None)
    wrist_cam_height = getattr(args, "wrist_cam_height", None)
    if (wrist_cam_width is None) != (wrist_cam_height is None):
        parser.error("--wrist-cam-width and --wrist-cam-height must be set together.")
    if bool(args.no_camera) and (bool(args.wrist_depth_mask) or bool(args.wrist_depth_raw)):
        parser.error("--wrist-depth-mask/--wrist-depth-raw require cameras; remove --no-camera.")
    if bool(getattr(args, "init_arm_available", False)) and not str(
        getattr(args, "init_arm_profile", "") or ""
    ).strip():
        parser.error("--init-arm-available requires --init-arm-profile.")
    try:
        args.special_arm_control_mode = resolve_special_arm_control_mode(
            getattr(args, "special_arm_control_mode", None),
            fix_arm_enabled=bool(getattr(args, "fix_arm", False)),
            init_arm_enabled=bool(getattr(args, "init_arm_available", False)),
        )
    except ValueError as exc:
        parser.error(str(exc))
    special_arm_modes = {
        flag
        for flag, enabled in (
            ("--fix-arm", bool(getattr(args, "fix_arm", False))),
            ("--init-arm-available", bool(getattr(args, "init_arm_available", False))),
        )
        if enabled
    }
    if special_arm_modes and set(_active_sides_from_args(args)) != {"left", "right"}:
        parser.error(
            f"{', '.join(sorted(special_arm_modes))} requires --active-sides both."
        )
    if not str(getattr(args, "load_profile", "") or "").strip():
        latest_profile = _latest_vive_calibration_profile_path()
        if latest_profile is None:
            parser.error(
                "--load-profile was not provided and no calibration profile JSON was found in "
                f"{DEFAULT_CALIBRATION_PROFILE_DIR}"
            )
        args.load_profile = str(latest_profile)
    args.requested_tcp_mode = str(getattr(args, "tcp_mode", TOOL_TCP_FRAME))
    args.tcp_mode = TOOL_TCP_FRAME
    active_sides = _active_sides_from_args(args)
    try:
        for side in active_sides:
            _hand_backend_mode_for_side(args, side)
        if _dmtac_sides(args, active_sides):
            _resolved_gripper_policy(args)
        _resolve_gripper_input_source(args, active_sides)
        _apply_default_arm_ips(args, active_sides=active_sides)
        _validate_required_arm_ips(args, active_sides)
        _save_explicit_arm_ips_latest(args, raw_argv=raw_argv)
        _run_gui_device_preflight(args, active_sides=active_sides, parser=parser)
        _apply_default_q6_ports(args, parser=parser)
    except ValueError as exc:
        parser.error(str(exc))
    if bool(args.visualize_tactile):
        if _dmtac_sides(args, active_sides):
            if getattr(args, "dmtac_tactile", None) is False:
                parser.error(
                    "--visualize-tactile with --hand-backend dmtac-gripper requires "
                    "DMTAC tactile reads; remove --dmtac-no-tactile."
                )
            args.dmtac_tactile = True
        if any(
            _hand_backend_mode_for_side(args, side) == HAND_BACKEND_INSPIRE
            for side in active_sides
        ):
            tactile_regions = _parse_hand_tactile_regions(args.hand_tactile_regions)
            if not tactile_regions:
                parser.error(
                    "--visualize-tactile requires non-empty --hand-tactile-regions, "
                    "for example 'thumb_1,index_1,palm' or 'all'."
                )
            missing_hand_ip_sides = [
                side
                for side in active_sides
                if _hand_backend_mode_for_side(args, side) == HAND_BACKEND_INSPIRE
                and not str(getattr(args, f"{side}_hand_ip") or "")
            ]
            if missing_hand_ip_sides:
                parser.error(
                    "--visualize-tactile requires real InspireTac hand IPs for active sides; "
                    f"missing={missing_hand_ip_sides}."
                )
    if not 0 <= int(args.vive_clock_sync_port) <= 65535:
        parser.error("--vive-clock-sync-port must be in [0, 65535].")
    if (
        int(args.vive_clock_sync_port)
        and int(args.vive_clock_sync_port) == int(args.port)
    ):
        parser.error("--vive-clock-sync-port must differ from the VIVE --port.")
    if bool(args.vive_clock_sync_required) and int(args.vive_clock_sync_port) <= 0:
        parser.error("--vive-clock-sync-required requires --vive-clock-sync-port.")
    if int(args.vive_clock_sync_min_samples) < 1:
        parser.error("--vive-clock-sync-min-samples must be >= 1.")
    if float(args.vive_clock_sync_max_rtt_ms) < 0.0:
        parser.error("--vive-clock-sync-max-rtt-ms must be >= 0.")
    if float(args.vive_clock_sync_stale_after_sec) < 0.0:
        parser.error("--vive-clock-sync-stale-after-sec must be >= 0.")
    if float(args.vive_clock_sync_startup_timeout_sec) < 0.0:
        parser.error("--vive-clock-sync-startup-timeout-sec must be >= 0.")
    return args


# Compatibility bridge: retain the established entrypoint helper names while
# the extracted component owns logical-side and tracker mapping behavior.
from ..real_teleop.side_mapping import (
    build_vive_tracker_configs_for_args as _build_vive_tracker_configs_for_args,
    opposite_side as _opposite_side,
    resolve_logical_side_assignments as _resolve_logical_side_assignments,
    tracker_legacy_index_for_source_side as _tracker_legacy_index_for_source_side,
    tracker_name_for_source_side as _tracker_name_for_source_side,
    tracker_role_for_source_side as _tracker_role_for_source_side,
    tracker_source_side_for_logical_side as _tracker_source_side_for_logical_side,
)


def _movement_multiple_value(args: argparse.Namespace) -> float:
    value = getattr(args, "movement_multiple", None)
    if value is None:
        value = getattr(args, "translation_scale", MOVEMENT_MULTIPLE_DEFAULT)
    return _movement_multiple_arg(str(value))


def _movement_multiple_ui_state(args: argparse.Namespace) -> dict[str, object]:
    value = _movement_multiple_value(args)
    return {
        "movement_multiple": value,
        "movement_multiple_min": MOVEMENT_MULTIPLE_GUI_MIN,
        "movement_multiple_max": max(MOVEMENT_MULTIPLE_GUI_MAX, value),
        "movement_multiple_step": MOVEMENT_MULTIPLE_GUI_STEP,
    }


def _apply_movement_multiple(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    value: float,
) -> float:
    movement_multiple = _movement_multiple_arg(str(value))
    setattr(args, "movement_multiple", movement_multiple)
    setattr(args, "translation_scale", movement_multiple)
    for retargeter in (components.left_retargeter, components.right_retargeter):
        retargeter.config.translation_scale = movement_multiple
    if components.dual_arm_retargeter is not None:
        components.dual_arm_retargeter.config.translation_scale = movement_multiple
        components.dual_arm_retargeter.left_pose_retargeter.config.translation_scale = movement_multiple
        components.dual_arm_retargeter.right_pose_retargeter.config.translation_scale = movement_multiple
    calibration = components.session.calibration
    if calibration is not None:
        calibration.metadata["movement_multiple"] = movement_multiple
        calibration.metadata["translation_scale"] = movement_multiple
    return movement_multiple


def _vive_pose_source_from_components(components: RealTeleopComponents) -> VivePoseSource | None:
    for source in components.session.sources:
        if isinstance(source, VivePoseSource):
            return source
    return None


def _hand_q6_source_from_components(components: RealTeleopComponents) -> HandQ6Source | None:
    for source in components.session.sources:
        if isinstance(source, HandQ6Source):
            return source
    return None


def _gripper_input_channel_name(components: RealTeleopComponents, side: str) -> str:
    source = str(getattr(components, "gripper_input_source", "q6") or "q6")
    return f"{side}_angle_trigger" if source == "angle_trigger" else f"{side}_hand_q6"


def _gripper_input_source_from_components(components: RealTeleopComponents):
    for source in components.session.sources:
        if isinstance(source, (HandQ6Source, AngleTriggerSource)):
            return source
    return None


def _apply_vive_tracker_mapping(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> None:
    source = _vive_pose_source_from_components(components)
    if source is None:
        return
    source.set_tracker_configs(
        _build_vive_tracker_configs_for_args(args, components.active_sides)
    )


# Compatibility bridge: retain legacy threshold globals through thin wrappers
# while the extracted module owns tracker-status payload construction.
from ..real_teleop.gui_tracker_status import (
    pose_xyz_rpy_summary as _ExtractedPoseXyzRpySummary,
    publish_raw_tracker_pose_state as _extracted_publish_raw_tracker_pose_state,
    raw_tracker_pose_state_from_channels as _extracted_raw_tracker_pose_state_from_channels,
    tracker_gui_status as _extracted_tracker_gui_status,
)


def _tracker_gui_status(
    *,
    enabled: bool,
    valid: bool,
    age_sec: float | None,
) -> tuple[str, str, bool]:
    return _extracted_tracker_gui_status(
        enabled=enabled,
        valid=valid,
        age_sec=age_sec,
        warn_age_sec=TRACKER_GUI_WARN_AGE_SEC,
        stale_age_sec=TRACKER_GUI_STALE_AGE_SEC,
    )


def _pose_xyz_rpy_summary(
    channel: object | None,
    *,
    now: float,
    enabled: bool = True,
) -> dict[str, object]:
    return _ExtractedPoseXyzRpySummary(
        channel,
        now=now,
        enabled=enabled,
        warn_age_sec=TRACKER_GUI_WARN_AGE_SEC,
        stale_age_sec=TRACKER_GUI_STALE_AGE_SEC,
    )


def _raw_tracker_pose_state_from_channels(
    channels: dict[str, object],
    *,
    active_sides: tuple[str, ...],
    now: float,
) -> dict[str, object]:
    return _extracted_raw_tracker_pose_state_from_channels(
        channels,
        active_sides=active_sides,
        now=now,
        warn_age_sec=TRACKER_GUI_WARN_AGE_SEC,
        stale_age_sec=TRACKER_GUI_STALE_AGE_SEC,
    )


def _publish_raw_tracker_pose_state(
    ui: CalibrationWebUi | None,
    channels: dict[str, object],
    *,
    active_sides: tuple[str, ...],
    now: float | None = None,
) -> None:
    _extracted_publish_raw_tracker_pose_state(
        ui,
        channels,
        active_sides=active_sides,
        now=now,
        warn_age_sec=TRACKER_GUI_WARN_AGE_SEC,
        stale_age_sec=TRACKER_GUI_STALE_AGE_SEC,
    )


# Compatibility bridge: retain legacy profile-status helper names while the
# extracted read-only module owns Vive calibration GUI projection.
from ..real_teleop.gui_vive_calibration import (
    vive_calibration_profile_ready as _vive_calibration_profile_ready,
    vive_calibration_profile_status_text as _vive_calibration_profile_status_text,
    vive_calibration_profile_ui_state as _vive_calibration_profile_ui_state,
    vive_calibration_tracker_roles as _vive_calibration_tracker_roles,
)


def _calibration_profile_latest_sort_key(path: Path) -> tuple[int, str, float, str]:
    match = re.match(r"^(\d{8}_\d{6})", path.name)
    timestamp_key = match.group(1) if match else ""
    return (
        1 if timestamp_key else 0,
        timestamp_key,
        path.stat().st_mtime,
        path.name,
    )


def _latest_vive_calibration_profile_path() -> Path | None:
    if not DEFAULT_CALIBRATION_PROFILE_DIR.exists():
        return None
    candidates: list[Path] = []
    seen: set[Path] = set()
    for path in DEFAULT_CALIBRATION_PROFILE_DIR.glob("*.json"):
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=_calibration_profile_latest_sort_key)


def _gui_vive_calibration_load_path(args: argparse.Namespace) -> Path:
    explicit = str(getattr(args, "load_profile", "") or "").strip()
    if explicit:
        return Path(os.path.expandvars(explicit)).expanduser()
    last_saved = str(getattr(args, "gui_last_vive_calibration_profile", "") or "").strip()
    if last_saved:
        return Path(os.path.expandvars(last_saved)).expanduser()
    latest = _latest_vive_calibration_profile_path()
    if latest is None:
        raise FileNotFoundError(
            f"No VIVE calibration profile found in {DEFAULT_CALIBRATION_PROFILE_DIR}"
        )
    return latest


def _runtime_home_was_available(calibration: TeleopCalibration) -> bool:
    if calibration.captured_home_qpos:
        return True
    for arm in calibration.arms.values():
        if arm.home_qpos is not None or arm.home_pose_for_tcp_mode() is not None:
            return True
    return False


def _calibration_profile_payload_without_workspace(
    calibration: TeleopCalibration,
    *,
    workspace_config_path: Path | None,
) -> dict[str, Any]:
    payload = calibration.to_dict()
    for arm_payload in payload.get("arms", {}).values():
        if not isinstance(arm_payload, dict):
            continue
        arm_payload.pop("workspace_min", None)
        arm_payload.pop("workspace_max", None)
        metadata = arm_payload.get("metadata")
        if isinstance(metadata, dict):
            metadata.pop("workspace_config_source", None)
            metadata.pop("workspace_override_runtime", None)
    metadata = payload.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata.pop("workspace_config_runtime", None)
        metadata.pop("workspace_override_runtime", None)
        metadata["workspace_policy"] = "managed_by_teleop_runtime_config"
        metadata["workspace_config_path"] = (
            None if workspace_config_path is None else str(workspace_config_path)
        )
    return payload


def _clear_runtime_home_state(components: RealTeleopComponents) -> None:
    calibration = components.session.calibration
    if calibration is not None:
        calibration.clear_home_state()
    components.left_retargeter.clear_home_anchors()
    components.right_retargeter.clear_home_anchors()


def _channel_sample_key(channel: object) -> tuple[object, object, object]:
    metadata = getattr(channel, "metadata", None) or {}
    return (
        metadata.get("seq"),
        metadata.get("receiver_timestamp"),
        getattr(channel, "timestamp", None),
    )


def _format_wall_time(timestamp: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(float(timestamp)))


def _source_debug_stats(source: object) -> dict[str, object]:
    get_stats = getattr(source, "debug_stats", None)
    if get_stats is None:
        return {}
    stats = get_stats()
    return stats if isinstance(stats, dict) else {}


def _format_vive_source_rates(
    source: object,
    *,
    start_stats: Mapping[str, object],
    channel_names: tuple[str, ...],
    elapsed_sec: float,
    now: float,
) -> str:
    current_stats = _source_debug_stats(source)
    if not current_stats:
        return "source_stats=unavailable"
    elapsed = max(float(elapsed_sec), 1e-6)
    packet_count = int(current_stats.get("packet_count") or 0)
    start_packet_count = int(start_stats.get("packet_count") or 0)
    packet_hz = (packet_count - start_packet_count) / elapsed
    update_counts = current_stats.get("channel_update_counts") or {}
    start_update_counts = start_stats.get("channel_update_counts") or {}
    last_updates = current_stats.get("last_channel_update_at") or {}
    channel_parts: list[str] = []
    if isinstance(update_counts, Mapping) and isinstance(start_update_counts, Mapping):
        for channel_name in channel_names:
            count = int(update_counts.get(channel_name) or 0)
            start_count = int(start_update_counts.get(channel_name) or 0)
            channel_hz = (count - start_count) / elapsed
            last_update_at = last_updates.get(channel_name) if isinstance(last_updates, Mapping) else None
            age = (now - float(last_update_at)) if last_update_at is not None else float("nan")
            channel_parts.append(f"{channel_name}_hz={channel_hz:.1f} age={age:.2f}s")
    last_packet_at = current_stats.get("last_packet_received_at")
    packet_age = (now - float(last_packet_at)) if last_packet_at is not None else float("nan")
    return f"source_udp_hz={packet_hz:.1f} packet_age={packet_age:.2f}s " + " ".join(channel_parts)


def _format_vive_sweep_quality(poses: list[np.ndarray]) -> str:
    try:
        quality = fit_signed_axis_from_poses(poses).quality
    except Exception as exc:  # noqa: BLE001
        return f"quality=unavailable({type(exc).__name__}: {exc})"
    return (
        f"quality extent={quality.extent_m:.3f}m "
        f"variance_ratio={quality.variance_ratio:.3f} "
        f"start_end_distance={quality.start_end_distance_m:.3f}m"
    )


def _virtual_desk_post_rotation_pose(yaw_deg: float) -> np.ndarray:
    return pose_from_rotation_translation(
        euler_xyz_deg_to_matrix([0.0, 0.0, float(yaw_deg)]),
        [0.0, 0.0, 0.0],
    )


def _channel_for_tracker_role(role: str) -> str:
    return {
        "auxiliary": "auxiliary_tracker_pose",
        "left_wrist": "left_wrist_pose",
        "right_wrist": "right_wrist_pose",
    }[str(role)]


def _tracker_role_for_channel(channel_name: str) -> str:
    return {
        "auxiliary_tracker_pose": "auxiliary",
        "left_wrist_pose": "left_wrist",
        "right_wrist_pose": "right_wrist",
    }.get(str(channel_name), str(channel_name))


def _resolve_virtual_desk_channel_name(args: argparse.Namespace, channels: Mapping[str, object]) -> str:
    requested = str(getattr(args, "virtual_desk_tracker_role", "auto") or "auto")
    if requested != "auto":
        return _channel_for_tracker_role(requested)
    for candidate in ("right_wrist_pose", "left_wrist_pose", "auxiliary_tracker_pose"):
        channel = channels.get(candidate)
        if channel is not None and bool(getattr(channel, "valid", False)):
            return candidate
    if str(getattr(args, "aux_tracker_name", "") or "").strip():
        return "auxiliary_tracker_pose"
    return "right_wrist_pose"


def _default_gui_vive_calibration_side(args: argparse.Namespace) -> str:
    requested = str(getattr(args, "virtual_desk_tracker_role", "auto") or "auto")
    return "left" if requested == "left_wrist" else "right"


def _select_gui_vive_coordinate_sweep_hand(
    *,
    ui: CalibrationWebUi,
    args: argparse.Namespace,
    components: RealTeleopComponents,
    source: VivePoseSource,
) -> tuple[str, str]:
    requested = str(getattr(args, "virtual_desk_tracker_role", "auto") or "auto")
    if requested == "auxiliary":
        channel_name = _channel_for_tracker_role("auxiliary")
        ui.append_message(
            "ViveCalibration",
            f"已通过 --virtual-desk-tracker-role auxiliary 指定 {channel_name} 做三轴 sweep。",
        )
        _log("ViveCalibration", f"using explicit auxiliary tracker channel={channel_name}")
        return "auxiliary", channel_name

    stage_name = "VIVE 标定手选择"
    stage_id = "ViveCalibrationHand"
    selected_side = _default_gui_vive_calibration_side(args)
    last_status_hint = ""
    ui.append_message(
        "ViveCalibration",
        "请选择用于 VIVE 坐标系三轴 sweep 的手：默认右手。点击“左手/右手”或按 1/2，点击“下一步”或按 n 确认。",
    )
    _log(
        "ViveCalibration",
        f"hand selection ready default={selected_side}; press 1/2 then n to continue",
    )
    while True:
        loop_start = time.time()
        channels = source.get_latest_channels()
        events = _consume_vive_capture_events(ui)
        if "q" in events:
            raise KeyboardInterrupt
        for event in events:
            if event == "1":
                selected_side = "left"
                last_status_hint = "已选择左手；点击“下一步”或按 n 开始三轴 sweep。"
                ui.append_message("ViveCalibration", "已选择左手进行 VIVE 坐标系标定。")
            elif event == "2":
                selected_side = "right"
                last_status_hint = "已选择右手；点击“下一步”或按 n 开始三轴 sweep。"
                ui.append_message("ViveCalibration", "已选择右手进行 VIVE 坐标系标定。")

        channel_name = f"{selected_side}_wrist_pose"
        channel = channels.get(channel_name)
        channel_valid = bool(channel is not None and getattr(channel, "valid", False))
        side_text = "左手" if selected_side == "left" else "右手"
        if last_status_hint:
            control_hint = last_status_hint
        else:
            control_hint = f"请选择用于三轴 sweep 的手；默认右手，当前为{side_text}。点击“下一步”或按 n 确认。"
        if not channel_valid:
            control_hint += f" 当前 {channel_name} 暂无有效数据。"

        ui.publish(
            stage=stage_name,
            selected_side=selected_side,
            active_controls=["1", "2", "n", "q"],
            control_hint=control_hint,
            capture_stage=stage_id,
            capture_phase="ready",
            capturing=False,
            capture_channel=channel_name,
            capture_samples=None,
            capture_left_samples=None,
            capture_right_samples=None,
            capture_min_samples=None,
            capture_valid=channel_valid,
            capture_left_valid=None,
            capture_right_valid=None,
            capture_elapsed_sec=None,
            capture_duration_sec=None,
            capture_prep_remaining_sec=None,
            capture_prep_total_sec=float(getattr(args, "capture_prep_sec", 3.0)),
            capture_waiting_for_samples=False,
            teleop_running=False,
            home_ready=False,
            gripper_control_enabled=False,
            raw_tracker_poses=_raw_tracker_pose_state_from_channels(
                dict(channels),
                active_sides=components.active_sides,
                now=loop_start,
            ),
            **_vive_calibration_profile_ui_state(args, components),
        )
        if "n" in events:
            if channel_valid:
                ui.append_message(
                    "ViveCalibration",
                    f"将使用{side_text} {channel_name} 做 VIVE 坐标系三轴 sweep。",
                )
                _log(
                    "ViveCalibration",
                    f"selected {selected_side} hand for coordinate sweep channel={channel_name}",
                    now=loop_start,
                )
                return selected_side, channel_name
            last_status_hint = (
                f"{side_text} {channel_name} 当前无有效数据；请等待 tracker 在线，"
                "或选择另一只手后再点“下一步”。"
            )
            ui.append_message("ViveCalibration", last_status_hint)

        time.sleep(max(0.0, min(0.02, (1.0 / float(args.hz)) - (time.time() - loop_start))))


def _publish_vive_capture_state(
    ui: CalibrationWebUi,
    args: argparse.Namespace,
    components: RealTeleopComponents,
    channels: Mapping[str, object],
    *,
    now: float,
    stage: str,
    active_controls: list[str],
    control_hint: str,
    capture_stage: str,
    capture_phase: str,
    capturing: bool,
    capture_channel: str,
    capture_samples: int | None = None,
    capture_left_samples: int | None = None,
    capture_right_samples: int | None = None,
    capture_min_samples: int | None = None,
    capture_valid: bool | None = None,
    capture_left_valid: bool | None = None,
    capture_right_valid: bool | None = None,
    capture_elapsed_sec: float | None = None,
    capture_duration_sec: float | None = None,
    capture_prep_remaining_sec: float | None = None,
    capture_prep_total_sec: float | None = None,
    capture_waiting_for_samples: bool = False,
) -> None:
    ui.publish(
        stage=stage,
        active_controls=active_controls,
        control_hint=control_hint,
        capture_stage=capture_stage,
        capture_phase=capture_phase,
        capturing=capturing,
        capture_channel=capture_channel,
        capture_samples=capture_samples,
        capture_left_samples=capture_left_samples,
        capture_right_samples=capture_right_samples,
        capture_min_samples=capture_min_samples,
        capture_valid=capture_valid,
        capture_left_valid=capture_left_valid,
        capture_right_valid=capture_right_valid,
        capture_elapsed_sec=capture_elapsed_sec,
        capture_duration_sec=capture_duration_sec,
        capture_prep_remaining_sec=capture_prep_remaining_sec,
        capture_prep_total_sec=capture_prep_total_sec,
        capture_waiting_for_samples=capture_waiting_for_samples,
        teleop_running=False,
        home_ready=False,
        gripper_control_enabled=False,
        raw_tracker_poses=_raw_tracker_pose_state_from_channels(
            dict(channels),
            active_sides=components.active_sides,
            now=now,
        ),
        **_vive_calibration_profile_ui_state(args, components),
    )


def _consume_vive_capture_events(ui: CalibrationWebUi) -> list[str]:
    return ui.consume_events()


def _capture_single_vive_channel_from_gui(
    *,
    ui: CalibrationWebUi,
    args: argparse.Namespace,
    components: RealTeleopComponents,
    source: VivePoseSource,
    stage_name: str,
    instructions: str,
    timed_flow_hint: str,
    direction_hint: str,
    channel_name: str,
    min_samples: int,
    capture_duration_sec: float,
    validator,
) -> list[np.ndarray]:
    ui.append_message(stage_name, instructions)
    ui.append_message(stage_name, timed_flow_hint)
    ui.append_message(stage_name, "本阶段需要点击“下一步”或按 n 才开始准备倒计时；r 重置当前阶段，q 取消。")
    _log(
        "ViveCalibration",
        f"{stage_name} ready channel={channel_name} direction={direction_hint} "
        f"prep={float(getattr(args, 'capture_prep_sec', 3.0)):.1f}s "
        f"window={float(capture_duration_sec):.1f}s min_samples={int(min_samples)}; wait_for_next=true; "
        f"{instructions}",
    )
    poses: list[np.ndarray] = []
    preparing = False
    capturing = False
    capture_start_time = 0.0
    capture_deadline = 0.0
    capture_extension_deadline = 0.0
    source_stats_at_start: dict[str, object] = {}
    last_sample_key: tuple[object, object, object] | None = None
    stage_gate = ChangeGate()
    min_samples_reported = False
    waiting_for_samples_reported = False
    last_status_hint = ""
    while True:
        loop_start = time.time()
        channels = source.get_latest_channels()
        events = _consume_vive_capture_events(ui)
        if "q" in events:
            raise KeyboardInterrupt
        if "r" in events:
            poses = []
            preparing = False
            capturing = False
            min_samples_reported = False
            waiting_for_samples_reported = False
            last_sample_key = None
            last_status_hint = f"已重置；点击“下一步”或按 n 重采，方向仍为 {direction_hint}。"
            ui.append_message(stage_name, "当前阶段已重置。")
        start_requested = bool("n" in events)
        if start_requested:
            if not capturing and not preparing:
                poses = []
                preparing = True
                last_status_hint = ""
                capture_start_time = loop_start + float(getattr(args, "capture_prep_sec", 3.0))
                capture_deadline = capture_start_time + float(capture_duration_sec)
                capture_extension_deadline = capture_deadline + float(getattr(args, "capture_extra_wait_sec", 0.0))
                source_stats_at_start = {}
                last_sample_key = None
                min_samples_reported = False
                waiting_for_samples_reported = False
                ui.append_message(
                    stage_name,
                    f"准备 {float(getattr(args, 'capture_prep_sec', 3.0)):.1f}s，"
                    f"{_format_wall_time(capture_start_time)} 开始采集。",
                )
            else:
                ui.append_message(stage_name, "采集已经在进行，请等待自动结束或按 r 重置。")

        channel = channels.get(channel_name)
        channel_valid = bool(channel is not None and getattr(channel, "valid", False))
        if preparing and loop_start >= capture_start_time:
            preparing = False
            capturing = True
            source_stats_at_start = _source_debug_stats(source)
            if channel_valid:
                last_sample_key = _channel_sample_key(channel)
                poses.append(as_pose_matrix(getattr(channel, "value")))
            ui.append_message(
                stage_name,
                f"采集开始，窗口 {float(capture_duration_sec):.1f}s，"
                f"自动结束时间 {_format_wall_time(capture_deadline)}。",
            )
        elapsed = max(0.0, loop_start - capture_start_time) if capturing else 0.0
        waiting_for_samples = bool(capturing and loop_start >= capture_deadline and len(poses) < int(min_samples))
        capture_active = preparing or capturing
        capture_phase = (
            "preparing"
            if preparing
            else "waiting"
            if waiting_for_samples
            else "capturing"
            if capturing
            else "ready"
        )
        if preparing:
            control_hint = f"{direction_hint}；准备中，{max(0.0, capture_start_time - loop_start):.1f}s 后开始。"
        elif waiting_for_samples:
            control_hint = (
                f"{direction_hint}；样本不足 {len(poses)}/{int(min_samples)}，"
                "继续等待 fresh samples。"
            )
        elif capturing:
            control_hint = f"{direction_hint}；采集中 {len(poses)}/{int(min_samples)}，请按方向扫动。"
        else:
            control_hint = last_status_hint or f"{direction_hint}；点击“下一步”或按 n 开始本阶段。"
        _publish_vive_capture_state(
            ui,
            args,
            components,
            channels,
            now=loop_start,
            stage=stage_name,
            active_controls=["n", "r", "q"] if not capture_active else ["r", "q"],
            control_hint=control_hint,
            capture_stage=stage_name,
            capture_phase=capture_phase,
            capturing=capture_active,
            capture_channel=channel_name,
            capture_samples=len(poses),
            capture_min_samples=int(min_samples),
            capture_valid=channel_valid,
            capture_elapsed_sec=elapsed if capturing else None,
            capture_duration_sec=(
                float(capture_duration_sec) + float(getattr(args, "capture_extra_wait_sec", 0.0))
                if waiting_for_samples
                else float(capture_duration_sec)
            ),
            capture_prep_remaining_sec=max(0.0, capture_start_time - loop_start) if preparing else None,
            capture_prep_total_sec=float(getattr(args, "capture_prep_sec", 3.0)),
            capture_waiting_for_samples=waiting_for_samples,
        )
        if capturing and stage_gate.should_emit(f"{stage_name}_channel_valid", channel_valid, now=loop_start):
            _log("ViveCalibration", f"{stage_name} channel_valid={channel_valid}", now=loop_start)
        if capturing and channel_valid:
            sample_key = _channel_sample_key(channel)
            if sample_key != last_sample_key:
                last_sample_key = sample_key
                poses.append(as_pose_matrix(getattr(channel, "value")))
            if len(poses) >= int(min_samples) and not min_samples_reported:
                min_samples_reported = True
                ui.append_message(stage_name, f"已达到最小样本数 {len(poses)}/{int(min_samples)}。")
        if capturing and stage_gate.should_emit(
            f"{stage_name}_live",
            channel_valid,
            now=loop_start,
            heartbeat_sec=3.0,
        ):
            _log(
                "ViveCalibration",
                f"{stage_name} live time={elapsed:.1f}/{float(capture_duration_sec):.1f}s "
                f"samples={len(poses)} valid={channel_valid} "
                f"{_format_vive_source_rates(source, start_stats=source_stats_at_start, channel_names=(channel_name,), elapsed_sec=elapsed, now=loop_start)}",
                now=loop_start,
            )
        if capturing and loop_start >= capture_deadline:
            capturing = False
            effective_hz = len(poses) / max(elapsed, 1e-6)
            if len(poses) < int(min_samples):
                if loop_start < capture_extension_deadline:
                    capturing = True
                    if not waiting_for_samples_reported:
                        waiting_for_samples_reported = True
                        ui.append_message(
                            stage_name,
                            f"样本不足 {len(poses)}/{int(min_samples)}，继续等待 fresh samples。",
                        )
                    time.sleep(max(0.0, min(0.02, (1.0 / float(args.hz)) - (time.time() - loop_start))))
                    continue
                last_status_hint = (
                    f"未通过：样本不足 {len(poses)}/{int(min_samples)}。"
                    f"当前仍在 {stage_name}，不会进入下一轴；请重采，方向仍为 {direction_hint}。"
                )
                _log(
                    "ViveCalibration",
                    f"{stage_name} AUTO-END INSUFFICIENT samples={len(poses)} need>={int(min_samples)} "
                    f"effective_hz={effective_hz:.1f}; stay_on={stage_name}; direction={direction_hint}; "
                    f"{_format_vive_source_rates(source, start_stats=source_stats_at_start, channel_names=(channel_name,), elapsed_sec=elapsed, now=loop_start)}",
                    now=loop_start,
                )
                ui.append_message(stage_name, f"{last_status_hint}")
                _publish_vive_capture_state(
                    ui,
                    args,
                    components,
                    channels,
                    now=loop_start,
                    stage=stage_name,
                    active_controls=["n", "r", "q"],
                    control_hint=last_status_hint,
                    capture_stage=stage_name,
                    capture_phase="ready",
                    capturing=False,
                    capture_channel=channel_name,
                    capture_samples=0,
                    capture_min_samples=int(min_samples),
                    capture_valid=channel_valid,
                    capture_waiting_for_samples=False,
                )
                poses = []
                min_samples_reported = False
                waiting_for_samples_reported = False
                last_sample_key = None
            else:
                quality_text = _format_vive_sweep_quality(poses)
                error = validator(poses) if validator is not None else None
                if error is None:
                    _log(
                        "ViveCalibration",
                        f"{stage_name} AUTO-END ACCEPTED samples={len(poses)} effective_hz={effective_hz:.1f}; "
                        f"{quality_text}; "
                        f"{_format_vive_source_rates(source, start_stats=source_stats_at_start, channel_names=(channel_name,), elapsed_sec=elapsed, now=loop_start)}",
                        now=loop_start,
                    )
                    ui.append_message(
                        stage_name,
                        f"采集接受：samples={len(poses)} hz={effective_hz:.1f}；{quality_text}。",
                    )
                    ui.publish(
                        stage=stage_name,
                        active_controls=[],
                        control_hint=f"{stage_name} 已接受，正在进入下一阶段。",
                        capture_phase="ready",
                        capturing=False,
                    )
                    return poses
                last_status_hint = (
                    f"未通过：{error}。当前仍在 {stage_name}，"
                    f"不会进入下一轴；请重采，方向仍为 {direction_hint}。{quality_text}。"
                )
                _log(
                    "ViveCalibration",
                    f"{stage_name} AUTO-END REJECTED: {error}; samples={len(poses)} "
                    f"effective_hz={effective_hz:.1f}; {quality_text}; "
                    f"stay_on={stage_name}; direction={direction_hint}; "
                    f"{_format_vive_source_rates(source, start_stats=source_stats_at_start, channel_names=(channel_name,), elapsed_sec=elapsed, now=loop_start)}",
                    now=loop_start,
                )
                ui.append_message(stage_name, f"{last_status_hint}")
                _publish_vive_capture_state(
                    ui,
                    args,
                    components,
                    channels,
                    now=loop_start,
                    stage=stage_name,
                    active_controls=["n", "r", "q"],
                    control_hint=last_status_hint,
                    capture_stage=stage_name,
                    capture_phase="ready",
                    capturing=False,
                    capture_channel=channel_name,
                    capture_samples=0,
                    capture_min_samples=int(min_samples),
                    capture_valid=channel_valid,
                    capture_waiting_for_samples=False,
                )
                poses = []
                min_samples_reported = False
                waiting_for_samples_reported = False
                last_sample_key = None
        time.sleep(max(0.0, min(0.02, (1.0 / float(args.hz)) - (time.time() - loop_start))))


def _capture_dual_wrist_offsets_from_gui(
    *,
    ui: CalibrationWebUi,
    args: argparse.Namespace,
    components: RealTeleopComponents,
    source: VivePoseSource,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    stage_id = "WristOffset"
    stage_name = "Wrist Offset Capture"
    ui.append_message(
        stage_id,
        "把左右 wrist tracker 放到标准中立姿态：前臂朝前、手背朝上、手指大致沿 desk +X。"
    )
    ui.append_message(stage_id, "点击“下一步”或按 n 开始定时采集；这里只写 tracker offset，不保存 runtime Home。")
    left_poses: list[np.ndarray] = []
    right_poses: list[np.ndarray] = []
    preparing = False
    capturing = False
    capture_start_time = 0.0
    capture_deadline = 0.0
    capture_extension_deadline = 0.0
    source_stats_at_start: dict[str, object] = {}
    left_last_sample_key: tuple[object, object, object] | None = None
    right_last_sample_key: tuple[object, object, object] | None = None
    stage_gate = ChangeGate()
    left_min_reported = False
    right_min_reported = False
    waiting_for_samples_reported = False
    min_samples = int(getattr(args, "min_hold_samples", 3))
    while True:
        loop_start = time.time()
        channels = source.get_latest_channels()
        events = _consume_vive_capture_events(ui)
        if "q" in events:
            raise KeyboardInterrupt
        if "r" in events:
            left_poses = []
            right_poses = []
            preparing = False
            capturing = False
            left_min_reported = False
            right_min_reported = False
            waiting_for_samples_reported = False
            left_last_sample_key = None
            right_last_sample_key = None
            ui.append_message(stage_id, "当前 wrist offset 阶段已重置。")
        if "n" in events:
            if not capturing and not preparing:
                left_poses = []
                right_poses = []
                preparing = True
                capture_start_time = loop_start + float(getattr(args, "wrist_capture_prep_sec", 5.0))
                capture_deadline = capture_start_time + float(getattr(args, "hold_capture_sec", 4.0))
                capture_extension_deadline = capture_deadline + float(getattr(args, "capture_extra_wait_sec", 0.0))
                source_stats_at_start = {}
                left_last_sample_key = None
                right_last_sample_key = None
                left_min_reported = False
                right_min_reported = False
                waiting_for_samples_reported = False
                ui.append_message(
                    stage_id,
                    f"准备 {float(getattr(args, 'wrist_capture_prep_sec', 5.0)):.1f}s，"
                    f"{_format_wall_time(capture_start_time)} 开始采集。",
                )
            else:
                ui.append_message(stage_id, "采集已经在进行，请等待自动结束或按 r 重置。")

        left_channel = channels.get("left_wrist_pose")
        right_channel = channels.get("right_wrist_pose")
        left_valid = bool(left_channel is not None and getattr(left_channel, "valid", False))
        right_valid = bool(right_channel is not None and getattr(right_channel, "valid", False))
        validity_snapshot = (left_valid, right_valid)
        if preparing and loop_start >= capture_start_time:
            preparing = False
            capturing = True
            source_stats_at_start = _source_debug_stats(source)
            if left_valid:
                left_last_sample_key = _channel_sample_key(left_channel)
                left_poses.append(
                    components.session.calibration.world_to_desk(as_pose_matrix(getattr(left_channel, "value")))
                )
            if right_valid:
                right_last_sample_key = _channel_sample_key(right_channel)
                right_poses.append(
                    components.session.calibration.world_to_desk(as_pose_matrix(getattr(right_channel, "value")))
                )
            ui.append_message(stage_id, f"采集开始，窗口 {float(getattr(args, 'hold_capture_sec', 4.0)):.1f}s。")
        elapsed = max(0.0, loop_start - capture_start_time) if capturing else 0.0
        waiting_for_samples = bool(
            capturing
            and loop_start >= capture_deadline
            and (len(left_poses) < min_samples or len(right_poses) < min_samples)
        )
        capture_active = preparing or capturing
        capture_phase = (
            "preparing"
            if preparing
            else "waiting"
            if waiting_for_samples
            else "capturing"
            if capturing
            else "ready"
        )
        _publish_vive_capture_state(
            ui,
            args,
            components,
            channels,
            now=loop_start,
            stage=stage_name,
            active_controls=["n", "r", "q"] if not capture_active else ["r", "q"],
            control_hint=(
                "左右手中立姿态；点击“下一步”或按 n 开始本阶段。"
                if not capture_active
                else "左右手中立姿态；该阶段只标定 tracker offset，不保存 runtime Home。"
            ),
            capture_stage=stage_id,
            capture_phase=capture_phase,
            capturing=capture_active,
            capture_channel="left_wrist_pose,right_wrist_pose",
            capture_left_samples=len(left_poses),
            capture_right_samples=len(right_poses),
            capture_min_samples=min_samples,
            capture_left_valid=left_valid,
            capture_right_valid=right_valid,
            capture_elapsed_sec=elapsed if capturing else None,
            capture_duration_sec=(
                float(getattr(args, "hold_capture_sec", 4.0)) + float(getattr(args, "capture_extra_wait_sec", 0.0))
                if waiting_for_samples
                else float(getattr(args, "hold_capture_sec", 4.0))
            ),
            capture_prep_remaining_sec=max(0.0, capture_start_time - loop_start) if preparing else None,
            capture_prep_total_sec=float(getattr(args, "wrist_capture_prep_sec", 5.0)),
            capture_waiting_for_samples=waiting_for_samples,
        )
        if capturing and stage_gate.should_emit("wrist_offset_validity", validity_snapshot, now=loop_start):
            _log("ViveCalibration", f"{stage_id} channel_valid={{'left': {left_valid}, 'right': {right_valid}}}", now=loop_start)
        if capturing and left_valid:
            left_sample_key = _channel_sample_key(left_channel)
            if left_sample_key != left_last_sample_key:
                left_last_sample_key = left_sample_key
                left_poses.append(components.session.calibration.world_to_desk(as_pose_matrix(getattr(left_channel, "value"))))
            if len(left_poses) >= min_samples and not left_min_reported:
                left_min_reported = True
                ui.append_message(stage_id, f"左手已达到最小样本数 {len(left_poses)}/{min_samples}。")
        if capturing and right_valid:
            right_sample_key = _channel_sample_key(right_channel)
            if right_sample_key != right_last_sample_key:
                right_last_sample_key = right_sample_key
                right_poses.append(components.session.calibration.world_to_desk(as_pose_matrix(getattr(right_channel, "value"))))
            if len(right_poses) >= min_samples and not right_min_reported:
                right_min_reported = True
                ui.append_message(stage_id, f"右手已达到最小样本数 {len(right_poses)}/{min_samples}。")
        if capturing and stage_gate.should_emit(
            "wrist_offset_live",
            validity_snapshot,
            now=loop_start,
            heartbeat_sec=3.0,
        ):
            _log(
                "ViveCalibration",
                f"{stage_id} live time={elapsed:.1f}/{float(getattr(args, 'hold_capture_sec', 4.0)):.1f}s "
                f"left={len(left_poses)} right={len(right_poses)} valid={{'left': {left_valid}, 'right': {right_valid}}} "
                f"{_format_vive_source_rates(source, start_stats=source_stats_at_start, channel_names=('left_wrist_pose', 'right_wrist_pose'), elapsed_sec=elapsed, now=loop_start)}",
                now=loop_start,
            )
        if capturing and loop_start >= capture_deadline:
            capturing = False
            left_hz = len(left_poses) / max(elapsed, 1e-6)
            right_hz = len(right_poses) / max(elapsed, 1e-6)
            if len(left_poses) >= min_samples and len(right_poses) >= min_samples:
                ui.append_message(stage_id, f"采集接受：left={len(left_poses)} right={len(right_poses)}。")
                return left_poses, right_poses
            if loop_start < capture_extension_deadline:
                capturing = True
                if not waiting_for_samples_reported:
                    waiting_for_samples_reported = True
                    ui.append_message(stage_id, f"样本不足 left={len(left_poses)} right={len(right_poses)}，继续等待。")
                time.sleep(max(0.0, min(0.02, (1.0 / float(args.hz)) - (time.time() - loop_start))))
                continue
            ui.append_message(
                stage_id,
                f"样本不足 left={len(left_poses)} right={len(right_poses)} "
                f"need>={min_samples} left_hz={left_hz:.1f} right_hz={right_hz:.1f}，请按 n 重采。",
            )
            left_poses = []
            right_poses = []
            left_min_reported = False
            right_min_reported = False
            waiting_for_samples_reported = False
            left_last_sample_key = None
            right_last_sample_key = None
        time.sleep(max(0.0, min(0.02, (1.0 / float(args.hz)) - (time.time() - loop_start))))


def _calibrate_gui_vive_coordinate_profile(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    ui: CalibrationWebUi,
) -> Path:
    if not {"left", "right"}.issubset(set(components.active_sides)):
        raise RuntimeError("VIVE coordinate calibration requires active-sides both.")
    source = _vive_pose_source_from_components(components)
    if source is None:
        raise RuntimeError("No VIVE pose source is attached to the teleop session.")
    calibration = components.session.calibration
    if calibration is None:
        raise RuntimeError("No active calibration is attached to the teleop session.")

    def _sweep_validator(poses: list[np.ndarray]) -> str | None:
        fit = fit_signed_axis_from_poses(poses)
        quality = fit.quality
        if quality.extent_m < float(getattr(args, "min_sweep_extent_m", 0.15)):
            return f"extent {quality.extent_m:.3f}m < {float(getattr(args, 'min_sweep_extent_m', 0.15)):.3f}m"
        if quality.variance_ratio < float(getattr(args, "min_axis_variance_ratio", 0.5)):
            return (
                f"variance_ratio {quality.variance_ratio:.3f} "
                f"< {float(getattr(args, 'min_axis_variance_ratio', 0.5)):.3f}"
            )
        if quality.start_end_distance_m < float(getattr(args, "min_start_end_distance_m", 0.08)):
            return (
                f"start_end_distance {quality.start_end_distance_m:.3f}m "
                f"< {float(getattr(args, 'min_start_end_distance_m', 0.08)):.3f}m"
            )
        return None

    _, desk_channel_name = _select_gui_vive_coordinate_sweep_hand(
        ui=ui,
        args=args,
        components=components,
        source=source,
    )
    desk_tracker_role = _tracker_role_for_channel(desk_channel_name)
    ui.append_message("ViveCalibration", f"使用 {desk_channel_name} ({desk_tracker_role}) 做 VIVE 坐标系三轴 sweep。")

    x_sweep = _capture_single_vive_channel_from_gui(
        ui=ui,
        args=args,
        components=components,
        source=source,
        stage_name="VirtualDeskX",
        instructions=(
            f"使用 {desk_tracker_role} 做 X 轴扫动，让拟合后的 +X 指向身体前方："
            "从靠近身体后侧的位置开始，向身体前方移动并结束。"
        ),
        timed_flow_hint="按一次采集后，按“后 -> 前”扫动；采集自动结束前停在身体前方。",
        direction_hint="后 -> 前",
        channel_name=desk_channel_name,
        min_samples=int(getattr(args, "min_sweep_samples", 3)),
        capture_duration_sec=float(getattr(args, "sweep_capture_sec", 5.0)),
        validator=_sweep_validator,
    )
    y_sweep = _capture_single_vive_channel_from_gui(
        ui=ui,
        args=args,
        components=components,
        source=source,
        stage_name="VirtualDeskY",
        instructions="做 Y 轴扫动，让拟合后的 +Y 指向你的左侧：从身体右侧开始，向身体左侧移动并结束。",
        timed_flow_hint="按一次采集后，按“右 -> 左”扫动；采集自动结束前停在身体左侧。",
        direction_hint="右 -> 左",
        channel_name=desk_channel_name,
        min_samples=int(getattr(args, "min_sweep_samples", 3)),
        capture_duration_sec=float(getattr(args, "sweep_capture_sec", 5.0)),
        validator=_sweep_validator,
    )
    z_sweep = _capture_single_vive_channel_from_gui(
        ui=ui,
        args=args,
        components=components,
        source=source,
        stage_name="VirtualDeskZ",
        instructions="做 Z 轴扫动，让拟合后的 +Z 指向上方：从低处开始，向上移动并结束。",
        timed_flow_hint="按一次采集后，按“低 -> 高”扫动；采集自动结束前不要向下返回。",
        direction_hint="低 -> 高",
        channel_name=desk_channel_name,
        min_samples=int(getattr(args, "min_sweep_samples", 3)),
        capture_duration_sec=float(getattr(args, "sweep_capture_sec", 5.0)),
        validator=_sweep_validator,
    )

    result = fit_virtual_desk_frame(
        x_sweep_poses=x_sweep,
        y_sweep_poses=y_sweep,
        z_sweep_poses=z_sweep,
    )
    max_raw_non_orthogonality = max(
        result.xy_non_orthogonality_deg,
        result.xz_non_orthogonality_deg,
        result.yz_non_orthogonality_deg,
    )
    max_axis_correction = max(
        result.x_axis_correction_deg,
        result.y_axis_correction_deg,
        result.z_axis_correction_deg,
    )
    max_allowed_axis_correction = float(getattr(args, "max_axis_non_orthogonality_deg", 20.0))
    if max_axis_correction > max_allowed_axis_correction:
        worst_axis_name, worst_axis_correction = max(
            (
                ("x", result.x_axis_correction_deg),
                ("y", result.y_axis_correction_deg),
                ("z", result.z_axis_correction_deg),
            ),
            key=lambda item: item[1],
        )
        raise RuntimeError(
            "Virtual desk fit failed axis-correction check: "
            f"worst_axis={worst_axis_name} correction={worst_axis_correction:.2f}deg "
            f"> {max_allowed_axis_correction:.2f}deg. "
            f"raw xy={result.xy_non_orthogonality_deg:.2f}deg "
            f"xz={result.xz_non_orthogonality_deg:.2f}deg "
            f"yz={result.yz_non_orthogonality_deg:.2f}deg. "
            "请按提示方向重新扫 X/Y/Z。"
        )
    if max_raw_non_orthogonality > max_allowed_axis_correction:
        ui.append_message(
            "ViveCalibration",
            f"raw sweep 非正交较大 {max_raw_non_orthogonality:.2f}deg，但正交拟合已接受。",
        )
    base_world_to_desk = result.T_W_V.copy()
    yaw_deg = float(getattr(args, "virtual_desk_post_yaw_deg", 180.0))
    world_to_desk = base_world_to_desk @ _virtual_desk_post_rotation_pose(yaw_deg)
    calibration.set_world_to_desk(world_to_desk)
    calibration.metadata.update(
        {
            "calibration_mode": "vive_rm65_virtual_desk",
            "virtual_desk": result.to_metadata(),
            "virtual_desk_tracker_role": desk_tracker_role,
            "virtual_desk_channel_name": desk_channel_name,
            "virtual_desk_sweep_convention_version": VIRTUAL_DESK_SWEEP_CONVENTION_VERSION,
            "virtual_desk_post_yaw_deg": yaw_deg,
            "runtime_gui_vive_calibrated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    ui.append_message(
        "ViveCalibration",
        f"VIVE desk 坐标系拟合完成，post_yaw={yaw_deg:.0f}deg；接下来采左右 wrist tracker offset。",
    )

    left_poses, right_poses = _capture_dual_wrist_offsets_from_gui(
        ui=ui,
        args=args,
        components=components,
        source=source,
    )
    tracker_metadata = calibration.metadata.setdefault("tracker_offset_calibration", {})
    if not isinstance(tracker_metadata, dict):
        tracker_metadata = {}
        calibration.metadata["tracker_offset_calibration"] = tracker_metadata
    per_side_poses = {"left": left_poses, "right": right_poses}
    for side, poses in per_side_poses.items():
        tracker_role = f"{side}_wrist"
        canonical_rpy_deg = canonical_hand_rpy_default()
        hand_rotation_virtual = euler_xyz_deg_to_matrix(canonical_rpy_deg)
        tracker_pose_virtual = average_pose(poses)
        offset = tracker_offset_from_neutral_pose(
            tracker_pose_virtual,
            hand_rotation_virtual=hand_rotation_virtual,
            translation_tracker_to_hand=np.zeros(3, dtype=float),
        )
        calibration.set_tracker_offset(tracker_role, offset)
        tracker_metadata[tracker_role] = {
            "method": "runtime_gui_wrist_neutral_capture",
            "canonical_hand_rpy_deg": canonical_rpy_deg.tolist(),
            "canonical_hand_frame_version": TRACKER_OFFSET_CALIBRATION_VERSION,
            "translation_tracker_to_hand": [0.0, 0.0, 0.0],
            "coarse_offset": offset.tolist(),
            "fine_tune_rpy_deg": [0.0, 0.0, 0.0],
            "final_offset": offset.tolist(),
            "captured_tracker_pose_virtual": tracker_pose_virtual.tolist(),
        }
    _clear_runtime_home_state(components)
    path = _save_gui_vive_calibration_profile(args, components)
    setattr(args, "load_profile", str(path))
    return path


def _save_gui_vive_calibration_profile(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> Path:
    calibration = components.session.calibration
    if calibration is None:
        raise RuntimeError("No active calibration is attached to the teleop session.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = DEFAULT_CALIBRATION_PROFILE_DIR / f"{timestamp}_vive_rm65_real_teleop.json"
    profile = TeleopCalibration.from_dict(calibration.to_dict())
    runtime_home_was_available = _runtime_home_was_available(profile)
    profile.clear_home_state()
    profile.metadata.update(
        {
            "profile_scope": "tracker_station_only",
            "runtime_home_used_for_visual_check": runtime_home_was_available,
            "home_policy": (
                "session home anchors are intentionally not saved; real teleop must recapture wrist home"
            ),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "saved_from_entrypoint": "teleop_runtime.debug.run_vive_rm65_real_teleop",
            "saved_from_gui": True,
            "active_sides": list(components.active_sides),
            "movement_multiple": _movement_multiple_value(args),
            "translation_scale": _movement_multiple_value(args),
            "tracker_input_mapping": {
                side: _tracker_source_side_for_logical_side(args, side)
                for side in components.active_sides
            },
        }
    )
    payload = _calibration_profile_payload_without_workspace(
        profile,
        workspace_config_path=_arm_workspace_config_path(args),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    setattr(args, "gui_last_vive_calibration_profile", str(output_path))
    return output_path


def _load_gui_vive_calibration_profile(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> Path:
    path = _gui_vive_calibration_load_path(args)
    loaded = TeleopCalibration.load_json(path)
    current = components.session.calibration
    if current is None:
        current = loaded
        components.session.calibration = current
    else:
        current.T_W_O = None if loaded.T_W_O is None else loaded.T_W_O.copy()
        current.T_O_D = None if loaded.T_O_D is None else loaded.T_O_D.copy()
        current.tracker_offsets = {
            str(role): np.asarray(offset, dtype=float).copy()
            for role, offset in loaded.tracker_offsets.items()
        }
        for arm_name, loaded_arm in loaded.arms.items():
            current_arm = current.arms.get(arm_name)
            if current_arm is None:
                continue
            current_arm.workspace_min = (
                None
                if loaded_arm.workspace_min is None
                else np.asarray(loaded_arm.workspace_min, dtype=float).copy()
            )
            current_arm.workspace_max = (
                None
                if loaded_arm.workspace_max is None
                else np.asarray(loaded_arm.workspace_max, dtype=float).copy()
            )
        calibration_metadata_keys = (
            "calibration_mode",
            "world_to_desk_mode",
            "virtual_desk",
            "virtual_desk_tracker_role",
            "virtual_desk_channel_name",
            "virtual_desk_sweep_convention_version",
            "virtual_desk_post_yaw_deg",
            "virtual_desk_scene_alignment_defaulted",
            "tracker_offset_calibration",
            "profile_scope",
            "home_policy",
            "workspace_policy",
            "workspace_config_path",
            "saved_at",
        )
        for key in calibration_metadata_keys:
            if key in loaded.metadata:
                current.metadata[key] = loaded.metadata[key]
            else:
                current.metadata.pop(key, None)
    current.clear_home_state()
    current.metadata.update(
        {
            "runtime_gui_loaded_profile": str(path),
            "runtime_gui_loaded_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _apply_workspace_config(args, current)
    _apply_workspace_overrides(args, current)
    components.left_retargeter.clear_home_anchors()
    components.right_retargeter.clear_home_anchors()
    setattr(args, "load_profile", str(path))
    setattr(args, "gui_last_vive_calibration_profile", str(path))
    return path


def _build_camera_sources(args: argparse.Namespace) -> dict[str, RealSenseRGBSource]:
    if bool(getattr(args, "no_camera", False)):
        return {}
    config_text = str(getattr(args, "camera_config", "") or "").strip()
    if not config_text:
        return {}
    config_path = Path(config_text).expanduser()
    if not config_path.exists():
        if bool(getattr(args, "vis_cam", False)):
            raise FileNotFoundError(f"Camera config not found: {config_path}")
        _log("Camera", f"optional camera config not found; preview disabled: {config_path}")
        return {}
    camera_configs = load_realsense_camera_config_map(config_path)
    wrist_depth_requested = bool(getattr(args, "wrist_depth_mask", False)) or bool(
        getattr(args, "wrist_depth_raw", False)
    )
    if wrist_depth_requested:
        available_wrist_keys = sorted(set(camera_configs) & set(WRIST_DEPTH_CAMERA_KEYS))
        missing_wrist_keys = sorted(set(WRIST_DEPTH_CAMERA_KEYS) - set(camera_configs))
        requested_wrist_cam_width = getattr(args, "wrist_cam_width", None)
        requested_wrist_cam_height = getattr(args, "wrist_cam_height", None)
        align_depth_to_color = bool(getattr(args, "wrist_depth_align_to_color", True))
        if missing_wrist_keys:
            _log(
                "Camera",
                "wrist depth requested but some wrist cameras are absent from config: "
                f"missing={missing_wrist_keys}",
            )
        for camera_key, config in camera_configs.items():
            if camera_key in WRIST_DEPTH_CAMERA_KEYS:
                config.enable_depth = True
                config.depth_width = int(
                    requested_wrist_cam_width
                    if requested_wrist_cam_width is not None
                    else config.width
                )
                config.depth_height = int(
                    requested_wrist_cam_height
                    if requested_wrist_cam_height is not None
                    else config.height
                )
                config.depth_fps = int(getattr(args, "wrist_depth_stream_fps", DEFAULT_WRIST_DEPTH_STREAM_FPS))
                config.align_depth_to_color = align_depth_to_color
                if requested_wrist_cam_width is not None and requested_wrist_cam_height is not None:
                    old_rgb_resolution = (int(config.width), int(config.height))
                    new_rgb_resolution = (int(config.depth_width), int(config.depth_height))
                    if old_rgb_resolution != new_rgb_resolution:
                        config.metadata = dict(config.metadata)
                        config.metadata.update(
                            {
                                "wrist_cam_original_rgb_width": old_rgb_resolution[0],
                                "wrist_cam_original_rgb_height": old_rgb_resolution[1],
                                "wrist_cam_matched_rgb_depth_resolution": True,
                            }
                        )
                        config.width, config.height = new_rgb_resolution
                        _log(
                            "Camera",
                            f"{camera_key} wrist camera stream resolution changed for RGB+depth: "
                            f"{old_rgb_resolution[0]}x{old_rgb_resolution[1]} -> "
                            f"{new_rgb_resolution[0]}x{new_rgb_resolution[1]}. "
                            "D405 color/depth profiles cannot always resolve mixed resolutions.",
                        )
            else:
                config.enable_depth = False
        wrist_cam_resolution_text = (
            f"{int(requested_wrist_cam_width)}x{int(requested_wrist_cam_height)}"
            if requested_wrist_cam_width is not None and requested_wrist_cam_height is not None
            else "match_rgb"
        )
        _log(
            "Camera",
            "wrist depth enabled for D405 wrist cameras only: "
            f"keys={available_wrist_keys} save_fps={DEFAULT_WRIST_DEPTH_SAVE_FPS} "
            f"depth_stream_fps={int(getattr(args, 'wrist_depth_stream_fps', DEFAULT_WRIST_DEPTH_STREAM_FPS))} "
            f"wrist_cam_stream_resolution={wrist_cam_resolution_text} "
            f"align_depth_to_color={align_depth_to_color} "
            f"mask={bool(getattr(args, 'wrist_depth_mask', False))} "
            f"raw={bool(getattr(args, 'wrist_depth_raw', False))}",
        )
    return {
        camera_key: RealSenseRGBSource(config)
        for camera_key, config in camera_configs.items()
    }


def _select_preview_camera_sources(
    camera_sources: dict[str, RealSenseRGBSource],
    requested_key: str,
) -> dict[str, RealSenseRGBSource]:
    if not camera_sources:
        return {}
    key = str(requested_key or "").strip()
    if not key or key.lower() in {"all", "*"}:
        return dict(camera_sources)
    if key:
        if key not in camera_sources:
            available = ", ".join(sorted(camera_sources))
            raise ValueError(
                f"--vis-cam-key {key!r} not found. Available camera keys: {available}"
            )
        return {key: camera_sources[key]}


def _start_camera_sources(
    camera_sources: Mapping[str, RealSenseRGBSource],
    *,
    stagger_sec: float = CAMERA_START_STAGGER_SEC,
) -> None:
    for index, (camera_key, source) in enumerate(camera_sources.items()):
        if index > 0 and float(stagger_sec) > 0.0:
            _log("Camera", f"waiting {float(stagger_sec):.2f}s before starting camera {camera_key}")
            time.sleep(float(stagger_sec))
        _log("Camera", f"starting camera {camera_key}")
        source.start()


def _close_camera_sources(camera_sources: Mapping[str, RealSenseRGBSource]) -> None:
    run_best_effort_cleanup(
        "real_teleop.camera_sources.close",
        (
            (f"camera[{camera_key}]", source.close)
            for camera_key, source in reversed(tuple(camera_sources.items()))
        ),
    )


def _parse_hand_tactile_regions(value: object) -> tuple[str, ...]:
    return parse_tactile_regions(value)


# Compatibility bridge: retain legacy status helper names while the extracted
# read-only projection owns special-arm GUI status formatting.
from ..real_teleop.gui_special_arm_status import (
    fix_arm_debug_state as _fix_arm_debug_state,
    fix_arm_status_text as _fix_arm_status_text,
    fix_arm_ui_state as _fix_arm_ui_state,
    init_arm_debug_state as _init_arm_debug_state,
    init_arm_status_text as _init_arm_status_text,
    init_arm_ui_state as _init_arm_ui_state,
)


def _cancel_special_arm_modes_for_hold(
    components: RealTeleopComponents,
    *,
    now: float,
    reason: str,
) -> list[str]:
    messages: list[str] = []
    fix_controller = getattr(components, "fix_arm_controller", None)
    if fix_controller is not None and bool(getattr(fix_controller, "fixed", False)):
        transition = fix_controller.cancel_active(
            now=now,
            reason=reason,
            clear_snapshots=True,
        )
        if transition.ok:
            side_label = "左臂" if transition.side == "left" else "右臂"
            messages.append(f"{side_label} fix-arm 已停止")
    init_controller = getattr(components, "init_arm_controller", None)
    if init_controller is not None and bool(getattr(init_controller, "active", False)):
        transition = init_controller.cancel_active(now=now, reason=reason)
        if transition.ok:
            side_label = "左臂" if transition.side == "left" else "右臂"
            messages.append(f"{side_label} init-arm 已停止")
    if messages:
        reset_smoothing = getattr(
            getattr(components, "dual_arm_retargeter", None),
            "reset_target_pose_smoothing",
            None,
        )
        if callable(reset_smoothing):
            try:
                reset_smoothing()
            except Exception as exc:
                _log(
                    "Hold",
                    "WARNING: retargeter target smoothing reset failed after special-mode cancel: "
                    f"{type(exc).__name__}: {exc}",
                    now=now,
                )
    return messages


def _disengage_position_backends_for_manual_interaction(
    components: RealTeleopComponents,
    *,
    now: float,
    reason: str,
) -> None:
    """Withdraw position qpos streams after entering an operator Hold state."""

    failures: list[str] = []
    for side in components.active_sides:
        backend = getattr(components, f"{side}_command_backend", None)
        disengage = getattr(backend, "disengage_position_command_stream", None)
        if not callable(disengage):
            continue
        try:
            disengage(reason)
        except Exception as exc:  # noqa: BLE001 - report both arms before failing.
            failures.append(f"{side}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError(
            "position command stream disengage failed: " + "; ".join(failures)
        )
    _log(
        "Hold",
        "position command streams disengaged; no persistent qpos command remains "
        "active during manual interaction",
        now=now,
    )


def _session_record_root_dir(args: argparse.Namespace) -> Path | None:
    cached = str(getattr(args, "_session_record_root_dir", "") or "").strip()
    if cached:
        return Path(cached)
    session_record_root = str(getattr(args, "session_record_root", "") or "").strip()
    if not session_record_root:
        return None
    stamp = str(getattr(args, "_session_record_stamp", "") or "").strip()
    if not stamp:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        setattr(args, "_session_record_stamp", stamp)
    root_dir = _expand_cli_path(session_record_root) / f"{stamp}_vive_rm65_real_teleop"
    setattr(args, "_session_record_root_dir", str(root_dir))
    return root_dir


def _session_recording_enabled(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "no_record", True))


def _session_recording_config(args: argparse.Namespace) -> RecordingConfig:
    return RecordingConfig(
        enabled=True,
        save_commands=not bool(getattr(args, "no_session_record_commands", False)),
        flush_every=max(1, int(getattr(args, "session_record_flush_every", 1) or 1)),
        compact_json=bool(getattr(args, "session_record_compact_json", False)),
    )


def _debug_episode_recording_enabled(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "no_episode_recording", False)):
        return False
    if bool(getattr(args, "no_collection", False)):
        return True
    if bool(getattr(args, "debug_trajectory_recording", False)):
        return True
    return bool(str(getattr(args, "episode_root", "") or "").strip())


def _episode_root_dir(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> Path | None:
    episode_root = str(getattr(args, "episode_root", "") or "").strip()
    if episode_root:
        return _expand_cli_path(episode_root)
    if components.session.recorder is not None:
        return Path(components.session.recorder.root_dir) / "trajectories"
    session_record_root = _session_record_root_dir(args)
    if session_record_root is not None:
        return session_record_root / "trajectories"
    return None


def _fix_arm_pose_snapshot_recorder(
    components: RealTeleopComponents,
) -> FixArmPoseSnapshotRecorder | None:
    if getattr(components, "fix_arm_controller", None) is None:
        return None
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    return FixArmPoseSnapshotRecorder(
        teleop_records_root() / "fix_arm_pose_snapshots" / session_name
    )


def _build_episode_metadata(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    start_reason: str,
) -> dict[str, object]:
    calibration = components.session.calibration
    assert calibration is not None
    session_record_enabled = _session_recording_enabled(args)
    session_record_root = (
        None if components.session.recorder is None else str(components.session.recorder.root_dir)
    )
    debug_episode_enabled = _debug_episode_recording_enabled(args)
    episode_root = _episode_root_dir(args, components) if debug_episode_enabled else None
    return {
        "source_entrypoint": "teleop_runtime.debug.run_vive_rm65_real_teleop",
        "start_reason": start_reason,
        "control_mode": args.control_mode,
        "arm_control_mode": _arm_control_mode_arg(args),
        **_arm_execution_profile_metadata(args),
        **_position_ik_iteration_metadata(args),
        "impedance_mode": _impedance_mode_arg(args),
        "strict_tool_profile": _strict_tool_profile_arg(args),
        "force_tare_sec": _force_tare_sec_arg(args),
        **_debug_timing_metadata(args),
        "arm_mode": "real",
        "active_sides": list(components.active_sides),
        "runtime_arm_state_source": (
            "causal_measured_hardware_preferred"
            if _arm_control_mode_arg(args) == "position"
            else "last_commanded_action"
        ),
        "runtime_arm_state_fallback": (
            "last_commanded_action"
            if _arm_control_mode_arg(args) == "position"
            else None
        ),
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
        "hz": float(args.hz),
        "sample_hz": float(args.hz),
        "collection_worker_mode_requested": str(
            getattr(args, "collection_worker_mode", "auto")
        ),
        "collection_worker_mode": str(
            getattr(args, "collection_worker_mode_resolved", "thread")
        ),
        "collection_process_queue_size": int(
            getattr(args, "collection_process_queue_size", 4)
        ),
        "collection_process_shared_slots": int(
            getattr(args, "collection_process_shared_slots", 4)
        ),
        "collection_process_extra_array_mb": int(
            getattr(args, "collection_process_extra_array_mb", 8)
        ),
        "collection_worker_cpus": list(
            getattr(args, "collection_worker_cpus", ()) or ()
        ),
        "collection_worker_nice": int(
            getattr(args, "collection_worker_nice", 5)
        ),
        "position_reference_hz": (
            _effective_position_reference_hz(args)
            if _arm_control_mode_arg(args) == "position"
            else None
        ),
        "outer_control_hz": _effective_outer_control_hz(args),
        "arm_control_hz": _effective_arm_control_hz(args),
        "position_reference_max_age_sec": (
            float(args.position_reference_max_age_sec)
            if _arm_control_mode_arg(args) == "position"
            else None
        ),
        "movement_multiple": _movement_multiple_value(args),
        "translation_scale": float(args.translation_scale),
        "left_right_base_distance_m": float(args.left_right_base_distance),
        "collision_scene_profile": args.collision_scene_profile,
        "collision_avoidance_mode": args.collision_avoidance_mode,
        "collision_guard_requested": bool(components.collision_guard_requested),
        "collision_guard_enabled": bool(components.dual_arm_retargeter is not None),
        "dataset_root": str(getattr(args, "dataset_root", "") or "") or None,
        "wrist_depth_mask": bool(getattr(args, "wrist_depth_mask", False)),
        "wrist_depth_raw": bool(getattr(args, "wrist_depth_raw", False)),
        "wrist_depth_camera_keys": list(WRIST_DEPTH_CAMERA_KEYS),
        "wrist_depth_fps": DEFAULT_WRIST_DEPTH_FPS,
        "wrist_depth_save_fps": DEFAULT_WRIST_DEPTH_SAVE_FPS,
        "wrist_depth_stream_fps": int(getattr(args, "wrist_depth_stream_fps", DEFAULT_WRIST_DEPTH_STREAM_FPS)),
        "wrist_cam_stream_width": getattr(args, "wrist_cam_width", None),
        "wrist_cam_stream_height": getattr(args, "wrist_cam_height", None),
        "wrist_depth_align_to_color": bool(getattr(args, "wrist_depth_align_to_color", True)),
        "wrist_depth_min_m": float(getattr(args, "wrist_depth_min_m", DEFAULT_WRIST_DEPTH_MIN_M)),
        "wrist_depth_max_m": float(getattr(args, "wrist_depth_max_m", DEFAULT_WRIST_DEPTH_MAX_M)),
        "session_record_enabled": session_record_enabled,
        "session_record_root": session_record_root,
        "session_record_commands": (
            session_record_enabled and not bool(getattr(args, "no_session_record_commands", False))
        ),
        "session_record_flush_every": int(getattr(args, "session_record_flush_every", 1) or 1),
        "session_record_compact_json": bool(getattr(args, "session_record_compact_json", False)),
        "debug_episode_recording_enabled": debug_episode_enabled,
        "episode_root": None if episode_root is None else str(episode_root),
        "load_profile": args.load_profile or None,
        "load_initial_alignment": args.load_initial_alignment or None,
        "logical_arm_ip_mapping": dict(components.logical_arm_ip_mapping),
        "left_hand_ip": args.left_hand_ip or None,
        "right_hand_ip": args.right_hand_ip or None,
        "left_hand_port": getattr(args, "left_hand_port", "") or None,
        "right_hand_port": getattr(args, "right_hand_port", "") or None,
        "hand_backend": _hand_backend_mode(args),
        "hand_backend_by_side": {
            side: _hand_backend_mode_for_side(args, side) for side in components.active_sides
        },
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
        "collection_task_id": str(getattr(args, "collection_task_id", "") or "") or None,
        "collection_task_name": str(getattr(args, "collection_task_name", "") or "") or None,
        "collection_task_name_zh": str(getattr(args, "collection_task_name_zh", "") or "") or None,
        "collection_task_description": str(getattr(args, "collection_task_description", "") or "") or None,
        "gripper_policy": (
            dict(getattr(args, "_resolved_gripper_policy", {}) or _resolved_gripper_policy(args))
            if _dmtac_sides(args, components.active_sides)
            else {}
        ),
        "gripper_policy_by_side": (
            _resolved_gripper_policies_by_side(args, components.active_sides)
            if _dmtac_sides(args, components.active_sides)
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
        "teleop_hardware_ip": args.teleop_hardware_ip or None,
        "teleop_hardware_port": int(args.teleop_hardware_port) if args.teleop_hardware_port else None,
        "swap_arm_sides": bool(args.swap_arm_sides),
        "swap_operator_sides": bool(getattr(args, "swap_operator_sides", False)),
        "tracker_input_mapping": {
            side: _tracker_source_side_for_logical_side(args, side)
            for side in components.active_sides
        },
        "calibration_snapshot": calibration.to_dict(),
        "initial_alignment_snapshot": (
            None
            if components.loaded_initial_alignment is None
            else components.loaded_initial_alignment.to_dict()
        ),
    }


def _start_episode_recording(
    recorder: EpisodeRecorder,
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    now: float,
    start_reason: str,
) -> Path | None:
    episode_dir = recorder.start_episode(
        metadata=_build_episode_metadata(
            args,
            components,
            start_reason=start_reason,
        ),
        timestamp=now,
    )
    if episode_dir is not None:
        _log("Episode", f"recording started -> {episode_dir}", now=now)
    return episode_dir


def _stop_episode_recording(
    recorder: EpisodeRecorder,
    *,
    now: float,
    end_reason: str,
    summary: dict[str, object] | None = None,
) -> Path | None:
    episode_dir = recorder.stop_episode(
        end_reason=end_reason,
        summary=summary,
        timestamp=now,
    )
    if episode_dir is not None:
        _log("Episode", f"recording stopped reason={end_reason} -> {episode_dir}", now=now)
    return episode_dir




def _collection_hand_action_specs(
    args: argparse.Namespace,
    active_sides: Sequence[str],
) -> Mapping[str, HandActionSpec] | None:
    dmtac_sides = _dmtac_sides(args, active_sides)
    if not dmtac_sides:
        return None
    return {side: HandActionSpec.dmtac_gripper(side) for side in dmtac_sides}


# Compatibility bridge: retain established collection adapter helper names
# while the extracted real-teleop bridge owns their behavior.
from ..real_teleop.collection_bridge import (
    collection_backends_from_components as _collection_backends_from_components,
    collection_sample_step_result as _collection_sample_step_result,
    collection_sources_from_components as _collection_sources_from_components,
    collection_step_result as _collection_step_result,
    copy_teleop_command as _copy_teleop_command,
)






def _action_debug_backend_snapshot(
    backend: Any | None,
    *,
    side: str,
    role: str,
    sample_timestamp: float | None,
    record_wall_time: float,
) -> dict[str, Any]:
    if backend is None:
        return {
            "available": False,
            "side": str(side),
            "role": str(role),
            "reason": "missing_backend",
        }
    read_start = time.time()
    try:
        state = backend.get_state()
    except Exception as exc:  # noqa: BLE001
        read_end = time.time()
        return {
            "available": False,
            "side": str(side),
            "role": str(role),
            "backend_class": type(backend).__name__,
            "reason": "get_state_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "read_start_wall_time": float(read_start),
            "read_end_wall_time": float(read_end),
            "read_elapsed_s": float(read_end - read_start),
        }
    read_end = time.time()
    return _action_debug_backend_snapshot_from_state(
        state,
        backend=backend,
        side=side,
        role=role,
        sample_timestamp=sample_timestamp,
        record_wall_time=record_wall_time,
        read_start=read_start,
        read_end=read_end,
    )


def _preloaded_action_debug_backend_snapshot(
    backend: Any | None,
    state: Mapping[str, object] | None,
    *,
    side: str,
    role: str,
    sample_timestamp: float | None,
    record_wall_time: float,
) -> dict[str, Any]:
    if isinstance(state, Mapping):
        return _action_debug_backend_snapshot_from_state(
            state,
            backend=backend,
            side=side,
            role=role,
            sample_timestamp=sample_timestamp,
            record_wall_time=record_wall_time,
            read_start=record_wall_time,
            read_end=record_wall_time,
        )
    return _action_debug_backend_snapshot(
        backend,
        side=side,
        role=role,
        sample_timestamp=sample_timestamp,
        record_wall_time=record_wall_time,
    )






class RealTeleopDatasetRecorder:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        components: RealTeleopComponents,
        camera_sources: Mapping[str, RealSenseRGBSource],
    ) -> None:
        self.args = args
        self.components = components
        self.camera_sources = dict(camera_sources)
        self.dataset_root = Path(args.dataset_root)
        self.arm_state_poller = self._build_arm_state_poller()
        self.hand_state_poller = self._build_hand_state_poller()
        policy_io_resolution = resolve_collection_policy_io_layout(
            getattr(args, "policy_io_layout", ""),
            active_sides=components.active_sides,
            hand_backends_by_side={
                "left": components.left_hand_backend,
                "right": components.right_hand_backend,
            },
            hand_retargeters_by_side={
                "left": components.left_gripper_retargeter or components.left_hand_retargeter,
                "right": components.right_gripper_retargeter or components.right_hand_retargeter,
            },
        )
        writer_config = LeRobotV21WriterConfig(
            root_dir=self.dataset_root,
            nominal_fps=float(args.hz),
            task_id=str(args.collection_task_id),
            task_description=str(args.collection_task_description),
            dataset_name=str(args.dataset_name),
            write_parquet=bool(args.write_parquet),
            write_video=bool(args.write_video and not args.no_camera),
            tactile_storage=str(args.tactile_storage),
            tactile_compression=str(getattr(args, "tactile_compression", "none")),
            allow_missing_video_frames=int(args.storage_queue_size) > 0,
            write_wrist_depth_mask=bool(getattr(args, "wrist_depth_mask", False)),
            write_wrist_depth_raw=bool(getattr(args, "wrist_depth_raw", False)),
            hand_action_specs=_collection_hand_action_specs(args, components.active_sides),
            policy_io_layout=policy_io_resolution.layout,
            policy_io_layout_provenance=policy_io_resolution.provenance,
            strict_v22_policy_io=bool(policy_io_resolution.layout is not None),
            pose_frame_contract=args.pose_frame_contract,
            wrist_depth_camera_keys=WRIST_DEPTH_CAMERA_KEYS,
            wrist_depth_min_m=float(
                getattr(args, "wrist_depth_min_m", DEFAULT_WRIST_DEPTH_MIN_M)
            ),
            wrist_depth_max_m=float(
                getattr(args, "wrist_depth_max_m", DEFAULT_WRIST_DEPTH_MAX_M)
            ),
        )
        self.writer_storage_config = writer_config
        self.collection_worker_mode = str(
            getattr(args, "collection_worker_mode_resolved", "thread") or "thread"
        )
        self.process_worker: ProcessCollectionWorker | None = None
        self.writer_pool: AsyncWriterPool | None = None
        self.writer_runtime: CollectionWriterRuntime | None = None
        self.writer_observer: Any | None = None
        self.collector_components = CollectorComponents(
            session=components.session,
            sources=_collection_sources_from_components(components, self.camera_sources),
            retargeters={},
            backends=_collection_backends_from_components(components),
            rig=components.session.rig,
            metadata={
                "source_entrypoint": "teleop_runtime.debug.run_vive_rm65_real_teleop",
                **_arm_execution_profile_metadata(args),
                **_position_ik_iteration_metadata(args),
                "active_sides": list(components.active_sides),
                "camera_frame_alignment_policy": "latest_before_or_at_sample_timestamp",
                "hand_tactile_alignment_policy": "causal_hand_history_at_T0",
                "hand_state_alignment_policy": (
                    "source_timestamp<=T0_and_available_timestamp<=T0"
                ),
                "causal_snapshot_contract": "dexteleop_causal_snapshot_v1",
                "movement_multiple": _movement_multiple_value(args),
                "sample_hz": float(args.hz),
                "collection_worker_mode": self.collection_worker_mode,
                "collection_process_queue_size": int(
                    getattr(args, "collection_process_queue_size", 4)
                ),
                "collection_process_shared_slots": int(
                    getattr(args, "collection_process_shared_slots", 4)
                ),
                "collection_process_extra_array_mb": int(
                    getattr(args, "collection_process_extra_array_mb", 8)
                ),
                "impedance_teleop_control_hz": (
                    _effective_outer_control_hz(args)
                    if _arm_control_mode_arg(args) == "impedance"
                    else None
                ),
                "collection_action_source": (
                    "latest_dispatched_main_command"
                    if _arm_control_mode_arg(args) == "impedance"
                    else "causal_position_reference"
                ),
                "position_reference_hz": (
                    _effective_position_reference_hz(args)
                    if _arm_control_mode_arg(args) == "position"
                    else None
                ),
                "position_command_hz": float(_effective_arm_control_hz(args)),
                **_debug_timing_metadata(args),
                "record_actual_wrist_camera_pose": bool(self.arm_state_poller is not None),
                "actual_wrist_camera_pose_hz": float(
                    getattr(args, "actual_wrist_camera_pose_hz", 30.0)
                ),
                "actual_wrist_camera_pose_max_age_sec": float(
                    getattr(args, "actual_wrist_camera_pose_max_age_sec", 0.2)
                ),
                "wrist_depth_mask": bool(getattr(args, "wrist_depth_mask", False)),
                "wrist_depth_raw": bool(getattr(args, "wrist_depth_raw", False)),
                "wrist_depth_camera_keys": list(WRIST_DEPTH_CAMERA_KEYS),
                "wrist_depth_fps": DEFAULT_WRIST_DEPTH_FPS,
                "wrist_depth_save_fps": DEFAULT_WRIST_DEPTH_SAVE_FPS,
                "wrist_depth_stream_fps": int(getattr(args, "wrist_depth_stream_fps", DEFAULT_WRIST_DEPTH_STREAM_FPS)),
                "wrist_cam_stream_width": getattr(args, "wrist_cam_width", None),
                "wrist_cam_stream_height": getattr(args, "wrist_cam_height", None),
                "wrist_depth_align_to_color": bool(getattr(args, "wrist_depth_align_to_color", True)),
                "wrist_depth_min_m": float(getattr(args, "wrist_depth_min_m", DEFAULT_WRIST_DEPTH_MIN_M)),
                "wrist_depth_max_m": float(getattr(args, "wrist_depth_max_m", DEFAULT_WRIST_DEPTH_MAX_M)),
                "tactile_compression": str(getattr(args, "tactile_compression", "none")),
            },
        )
        # The process worker discovers persisted episode/sample indices during
        # startup.  The frontend builder owns only causal snapshot providers in
        # that mode, so zero placeholders are sufficient until start_episode.
        initial_episode_index = 0
        initial_sample_index = 0
        if self.collection_worker_mode == "thread":
            initial_writer = LeRobotV21Writer(writer_config)
            initial_episode_index = int(initial_writer.next_episode_index)
            initial_sample_index = int(initial_writer.next_sample_index)
            self.writer = initial_writer
        sample_builder_config = CollectorSampleBuilderConfig(
            active_sides=components.active_sides,
            camera_buffers={
                camera_key: source.frame_buffer()
                for camera_key, source in self.camera_sources.items()
            },
            require_camera_frames=bool(args.write_video and not args.no_camera),
            camera_frame_max_age_sec=DEFAULT_CAMERA_FRAME_MAX_AGE_SEC,
            episode_index=initial_episode_index,
            sample_index_start=initial_sample_index,
            tactile_array_snapshot_policy=(
                "always"
                if str(args.tactile_storage).strip().lower().replace("-", "_")
                == "parquet"
                else "on_change"
            ),
            position_reference_max_age_sec=float(
                getattr(args, "position_reference_max_age_sec", 0.2)
            ),
            # State semantics must not depend on the controller mode.  Both
            # force/impedance and position collection use the latest eligible
            # measured hardware snapshot at T0; commanded state remains only
            # in the explicit commanded_* sidecars.
            prefer_causal_measured_arm_state=True,
        )
        self.sample_builder = CollectorSampleBuilder(
            sample_builder_config,
            arm_state_metadata_provider=self._latest_actual_arm_state_metadata,
            hand_state_provider=(
                None
                if self.hand_state_poller is None
                else lambda _components, sample_timestamp: self.hand_state_poller.latest_by_side(
                    sample_timestamp=float(sample_timestamp)
                )
            ),
            # The causal history returns a deep-copied immutable selection.
            # Avoid copying ~1.23 MiB of tactile arrays a second time only on
            # the position-process data plane; the established thread/force
            # path retains its previous copy semantics.
            hand_state_provider_returns_isolated=(
                self.collection_worker_mode == "process"
                and self.hand_state_poller is not None
            ),
            position_reference_provider=(
                None
                if components.position_reference_publisher is None
                else components.position_reference_publisher.reference_for_sample
            ),
        )
        if self.collection_worker_mode == "process":
            self.process_worker = ProcessCollectionWorker(
                writer_config=writer_config,
                sample_builder_config=sample_builder_config,
                config=ProcessCollectionConfig(
                    queue_size=int(getattr(args, "collection_process_queue_size", 4)),
                    shared_slot_count=int(
                        getattr(args, "collection_process_shared_slots", 4)
                    ),
                    shared_slot_bytes=camera_shared_slot_bytes(
                        self.camera_sources,
                        extra_array_bytes=(
                            int(getattr(args, "collection_process_extra_array_mb", 8))
                            * 1024
                            * 1024
                        ),
                    ),
                    worker_cpus=tuple(
                        int(cpu)
                        for cpu in (
                            getattr(args, "collection_worker_cpus", ()) or ()
                        )
                    ),
                    worker_nice=int(getattr(args, "collection_worker_nice", 5)),
                ),
            )
            self.writer = self.process_worker
        else:
            assert isinstance(self.writer, LeRobotV21Writer)
            self.writer_pool = AsyncWriterPool(
                self.writer.handle_job,
                config=AsyncWriterConfig(
                    worker_count=int(args.writer_workers),
                    max_queue_size=int(args.writer_queue_size),
                    put_timeout_sec=0.25,
                    backpressure_policy="block",
                    fail_fast=True,
                ),
                name="real_teleop_dataset_writer",
            )
            self.writer_runtime = CollectionWriterRuntime(
                components=self.collector_components,
                sample_builder=self.sample_builder,
                write_planner=lambda sample, _components: self.writer.plan_sample_jobs(sample),
                writer_pool=self.writer_pool,
                config=CollectionWriterObserverConfig(
                    name="real_teleop_dataset_observer",
                    storage_queue_size=int(args.storage_queue_size),
                ),
            )
            # Keep the established attribute for diagnostics and downstream tools;
            # the shared writer runtime now owns its lifecycle.
            self.writer_observer = self.writer_runtime.writer_observer
        self.active_episode_index: int | None = None
        self.active_started_at: float | None = None
        self.finalize_tasks: list[RealDatasetFinalizeTask] = []
        self.saved_episodes = 0
        self.discarded_episodes = 0
        self.frequency_warning_hz = float(getattr(args, "collection_warning_hz", 15.0))
        self.frequency_window_sec = 3.0
        self.frequency_min_elapsed_sec = 2.0
        self._record_timestamps: deque[float] = deque()
        self._episode_start_monotonic: float | None = None
        self._pause_started_at: float | None = None
        self._pause_started_monotonic: float | None = None
        self._pause_reason: str | None = None
        self._pause_started_frame_index: int | None = None
        self._pause_started_step_index: int | None = None
        self._pause_started_last_state: str | None = None
        self._pause_segments: list[dict[str, Any]] = []
        self._paused_accumulated_sec = 0.0
        # Sample assembly and write planning are CPU-heavy enough to miss a
        # teleop/IK deadline. Keep exactly one causal sample request in flight
        # for both modes: force teleop may run at 30 Hz while collection stays
        # at 15 Hz.
        self._async_sample_enabled = True
        self._sample_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="teleop_sample_assembly")
            if self._async_sample_enabled
            else None
        )
        self._sample_future: Any | None = None
        self._sample_async_skipped_busy = 0
        self._sample_process_rejected_busy = 0
        self._sample_overload_reason: str | None = None
        self._process_episode_submitted_count = 0
        self._collection_integrity_reasons: list[str] = []

    def _build_arm_state_poller(self) -> ArmHardwareStatePoller | None:
        if not bool(getattr(self.args, "record_actual_wrist_camera_pose", True)):
            return None
        backends_by_side = {
            "left": self.components.left_state_backend,
            "right": self.components.right_state_backend,
        }
        active_backends = {
            side: backend
            for side, backend in backends_by_side.items()
            if side in self.components.active_sides and backend is not None
        }
        if not active_backends:
            return None
        return ArmHardwareStatePoller(
            backends_by_side=active_backends,
            config=ArmHardwareStatePollerConfig(
                poll_hz=float(getattr(self.args, "actual_wrist_camera_pose_hz", 30.0)),
                stale_after_sec=float(
                    getattr(self.args, "actual_wrist_camera_pose_max_age_sec", 0.2)
                ),
            ),
        )

    def _build_hand_state_poller(self) -> HandStateHistoryPoller | None:
        backends_by_side = {
            "left": self.components.left_hand_backend,
            "right": self.components.right_hand_backend,
        }
        active_backends = {
            side: backend
            for side, backend in backends_by_side.items()
            if side in self.components.active_sides and backend is not None
        }
        if not active_backends:
            return None
        return HandStateHistoryPoller(
            backends_by_side=active_backends,
            config=HandStateHistoryPollerConfig(
                poll_hz=max(60.0, 2.0 * float(self.args.hz)),
            ),
        )

    def _latest_actual_arm_state_metadata(
        self,
        _components: CollectorComponents,
        sample_timestamp: float,
    ) -> Mapping[str, Mapping[str, Any]]:
        if self.arm_state_poller is None:
            return {}
        return self.arm_state_poller.latest_by_side(sample_timestamp=float(sample_timestamp))

    def reconfigure_for_current_args(self) -> None:
        desired_root = Path(self.args.dataset_root)
        if (
            desired_root == self.dataset_root
            and str(self.writer_storage_config.task_id) == str(self.args.collection_task_id)
            and str(self.writer_storage_config.task_description) == str(self.args.collection_task_description)
            and str(self.writer_storage_config.dataset_name) == str(self.args.dataset_name)
        ):
            return
        if self.is_recording:
            raise RuntimeError("Cannot reconfigure dataset recorder while recording.")
        self.close()
        self.__init__(
            args=self.args,
            components=self.components,
            camera_sources=self.camera_sources,
        )
        self.start()
        self.write_camera_device_config()

    @property
    def is_recording(self) -> bool:
        return self.active_episode_index is not None

    @property
    def is_paused(self) -> bool:
        return bool(self.is_recording and getattr(self, "_pause_started_at", None) is not None)

    @property
    def can_record_samples(self) -> bool:
        return bool(self.is_recording and not self.is_paused)

    @property
    def pause_reason(self) -> str:
        return str(getattr(self, "_pause_reason", None) or "")

    @property
    def frame_count(self) -> int:
        if getattr(self, "collection_worker_mode", "thread") == "process":
            return int(self._process_episode_submitted_count)
        return int(self.sample_builder.frame_index)

    @property
    def finalizing(self) -> bool:
        return any(not task.done for task in self.finalize_tasks)

    def _writer_lifecycle(self) -> Any:
        """Return the shared writer runtime while preserving old observer seams."""

        if getattr(self, "collection_worker_mode", "thread") == "process":
            return self.process_worker
        runtime = getattr(self, "writer_runtime", None)
        observer = getattr(self, "writer_observer", None)
        if runtime is not None and observer is getattr(runtime, "writer_observer", None):
            return runtime
        # Existing tests and downstream diagnostics may replace the established
        # writer_observer attribute with a fake.  Keep that seam functional
        # without making the shared runtime depend on the GUI entrypoint.
        return observer

    def start(self) -> None:
        writer_lifecycle = self._writer_lifecycle()
        if writer_lifecycle is None:
            raise RuntimeError("formal collection writer lifecycle is unavailable")
        rollback_steps = [("writer_runtime", lambda: writer_lifecycle.close(flush=False))]
        try:
            writer_lifecycle.start()
            if self.arm_state_poller is not None:
                rollback_steps.append(("arm_state_poller", self.arm_state_poller.close))
                self.arm_state_poller.start()
            if self.hand_state_poller is not None:
                rollback_steps.append(("hand_state_poller", self.hand_state_poller.close))
                self.hand_state_poller.start()
        except BaseException as start_error:  # noqa: BLE001 - rollback before propagating.
            rollback_failures = collect_lifecycle_failures(reversed(rollback_steps))
            if rollback_failures:
                raise LifecycleStartupError(
                    "real_teleop_dataset_recorder.start",
                    start_error,
                    rollback_failures,
                ) from start_error
            raise

    def write_camera_device_config(self) -> None:
        if not self.camera_sources:
            return
        self.writer.write_device_config(
            "realsense",
            {
                camera_key: source.device_info()
                for camera_key, source in self.camera_sources.items()
            },
        )

    def start_episode(self, metadata: Mapping[str, Any], *, now: float) -> int:
        if self.is_recording:
            raise RuntimeError("dataset episode is already recording")
        if self.finalizing:
            raise RuntimeError("previous dataset episode is still finalizing")
        ready, message = self.actual_feedback_warmup_status(now)
        if not ready:
            raise RuntimeError(f"actual TCP feedback is not ready: {message}")
        episode_index = self.writer.start_episode(metadata=metadata)
        if getattr(self, "collection_worker_mode", "thread") == "thread":
            self.sample_builder.set_episode(episode_index)
        self._process_episode_submitted_count = 0
        self._sample_overload_reason = None
        self._collection_integrity_reasons = []
        self.active_episode_index = int(episode_index)
        self.active_started_at = float(now)
        self._record_timestamps.clear()
        self._episode_start_monotonic = time.monotonic()
        self._pause_started_at = None
        self._pause_started_monotonic = None
        self._pause_reason = None
        self._pause_started_frame_index = None
        self._pause_started_step_index = None
        self._pause_started_last_state = None
        self._pause_segments = []
        self._paused_accumulated_sec = 0.0
        return int(episode_index)

    def actual_feedback_warmup_status(self, now: float) -> tuple[bool, str]:
        """Return the feedback-start gate without issuing a synchronous hardware read."""

        if self.arm_state_poller is None:
            return True, "actual TCP feedback recording is disabled"
        readiness = self.arm_state_poller.readiness_by_side(
            required_sides=tuple(self.components.active_sides),
            sample_timestamp=float(now),
        )
        pending = {side: reason for side, reason in readiness.items() if reason is not None}
        if not pending:
            return True, "actual TCP feedback ready"
        details = ", ".join(f"{side}={reason}" for side, reason in sorted(pending.items()))
        return False, f"等待 actual TCP feedback：{details}。"

    def _note_recorded_sample(self) -> None:
        current_time = time.monotonic()
        self._record_timestamps.append(current_time)
        cutoff = current_time - self.frequency_window_sec
        while self._record_timestamps and self._record_timestamps[0] < cutoff:
            self._record_timestamps.popleft()

    def _mark_collection_integrity_issue(self, reason: str) -> None:
        """Persist any causal sample loss instead of silently accepting it."""

        normalized = str(reason).strip()
        if not normalized:
            return
        reasons = getattr(self, "_collection_integrity_reasons", None)
        if reasons is None:
            reasons = []
            self._collection_integrity_reasons = reasons
        if normalized not in reasons:
            reasons.append(normalized)

    def _current_pause_elapsed_sec(self, *, current_time: float | None = None) -> float:
        pause_started = getattr(self, "_pause_started_monotonic", None)
        if pause_started is None:
            return 0.0
        now_monotonic = time.monotonic() if current_time is None else float(current_time)
        return max(0.0, now_monotonic - float(pause_started))

    def _total_paused_sec(self, *, current_time: float | None = None) -> float:
        accumulated = max(0.0, float(getattr(self, "_paused_accumulated_sec", 0.0) or 0.0))
        return accumulated + self._current_pause_elapsed_sec(current_time=current_time)

    def _active_elapsed_sec(self, *, current_time: float | None = None) -> tuple[float, float]:
        start_time = getattr(self, "_episode_start_monotonic", None)
        now_monotonic = time.monotonic() if current_time is None else float(current_time)
        wall_elapsed_sec = (
            max(0.0, now_monotonic - float(start_time))
            if start_time is not None
            else 0.0
        )
        paused_sec = min(wall_elapsed_sec, self._total_paused_sec(current_time=now_monotonic))
        return max(0.0, wall_elapsed_sec - paused_sec), wall_elapsed_sec

    def pause_episode(
        self,
        *,
        reason: str,
        now: float,
        step_index: int,
        last_state: str | None,
    ) -> bool:
        if not self.is_recording:
            return False
        if self.is_paused:
            return False
        # Process mode pauses must never wait on the snapshot thread from the
        # 60 Hz control loop.  At most the already accepted causal sample may
        # complete after this boundary; stop/finalize still performs a drain.
        if getattr(self, "collection_worker_mode", "thread") == "thread":
            self._drain_async_sample()
        if not hasattr(self, "_pause_segments"):
            self._pause_segments = []
        self._pause_started_at = float(now)
        self._pause_started_monotonic = time.monotonic()
        self._pause_reason = str(reason)
        self._pause_started_frame_index = int(self.frame_count)
        self._pause_started_step_index = int(step_index)
        self._pause_started_last_state = last_state
        self._record_timestamps.clear()
        return True

    def _close_pause_segment(
        self,
        *,
        resume_reason: str,
        now: float,
        step_index: int,
    ) -> dict[str, Any] | None:
        if not self.is_paused:
            return None
        pause_started_at = self._pause_started_at
        pause_started_monotonic = self._pause_started_monotonic
        duration_sec = (
            max(0.0, time.monotonic() - float(pause_started_monotonic))
            if pause_started_monotonic is not None
            else max(0.0, float(now) - float(pause_started_at or now))
        )
        segment = {
            "reason": str(self._pause_reason or ""),
            "resume_reason": str(resume_reason),
            "start_timestamp": float(pause_started_at or now),
            "end_timestamp": float(now),
            "duration_sec": float(duration_sec),
            "start_frame_index": self._pause_started_frame_index,
            "end_frame_index": int(self.frame_count),
            "start_step_index": self._pause_started_step_index,
            "end_step_index": int(step_index),
            "last_state_at_pause": self._pause_started_last_state,
        }
        if not hasattr(self, "_pause_segments"):
            self._pause_segments = []
        self._pause_segments.append(segment)
        self._paused_accumulated_sec = max(
            0.0,
            float(getattr(self, "_paused_accumulated_sec", 0.0) or 0.0) + float(duration_sec),
        )
        self._pause_started_at = None
        self._pause_started_monotonic = None
        self._pause_reason = None
        self._pause_started_frame_index = None
        self._pause_started_step_index = None
        self._pause_started_last_state = None
        self._record_timestamps.clear()
        return segment

    def resume_episode(
        self,
        *,
        reason: str,
        now: float,
        step_index: int,
    ) -> bool:
        resumed = self._close_pause_segment(
            resume_reason=reason,
            now=now,
            step_index=step_index,
        ) is not None
        if resumed:
            self._sample_overload_reason = None
        return resumed

    def pause_summary(self) -> dict[str, Any]:
        active_elapsed_sec, wall_elapsed_sec = self._active_elapsed_sec()
        return {
            "recording_pause_schema": "dexteleop_recording_pause_v1",
            "recording_paused": bool(self.is_paused),
            "recording_pause_reason": self.pause_reason,
            "recording_pause_segments": [dict(segment) for segment in getattr(self, "_pause_segments", [])],
            "recording_pause_count": len(getattr(self, "_pause_segments", [])) + (1 if self.is_paused else 0),
            "recording_paused_duration_sec": self._total_paused_sec(),
            "recording_active_duration_sec": active_elapsed_sec,
            "recording_wall_duration_sec": wall_elapsed_sec,
        }

    def collection_frequency_status(self) -> dict[str, Any]:
        frame_count = int(self.frame_count)
        current_time = time.monotonic()
        elapsed_sec, wall_elapsed_sec = self._active_elapsed_sec(current_time=current_time)
        average_hz = (float(frame_count) / elapsed_sec) if elapsed_sec > 0.0 else None
        rolling_hz = None
        if len(self._record_timestamps) >= 2:
            span = float(self._record_timestamps[-1] - self._record_timestamps[0])
            if span > 0.0:
                rolling_hz = float(len(self._record_timestamps) - 1) / span
        sample_hz = rolling_hz if rolling_hz is not None else average_hz
        enough_data = (
            self.can_record_samples
            and frame_count >= 4
            and elapsed_sec >= self.frequency_min_elapsed_sec
            and sample_hz is not None
        )
        threshold_hz = float(self.frequency_warning_hz)
        warning_active = bool(enough_data and float(sample_hz) < threshold_hz)
        warning = {
            "active": warning_active,
            "threshold_hz": threshold_hz,
            "message": "",
        }
        if warning_active:
            warning["message"] = (
                f"WARNING: 数采实际频率 {float(sample_hz):.1f} Hz，"
                f"低于目标 {threshold_hz:.1f} Hz；请检查相机、写盘或 CPU 负载。"
            )
        tactile_frame_counts: dict[str, int] = {}
        tactile_write_hz: dict[str, float] = {}
        if self.active_episode_index is not None:
            try:
                tactile_frame_counts = self.writer.tactile_frame_counts(self.active_episode_index)
            except Exception:  # noqa: BLE001
                tactile_frame_counts = {}
        if elapsed_sec > 0.0:
            tactile_write_hz = {
                side: float(count) / elapsed_sec
                for side, count in tactile_frame_counts.items()
            }
        tactile_write_hz_mean = (
            sum(tactile_write_hz.values()) / float(len(tactile_write_hz))
            if tactile_write_hz
            else None
        )
        return {
            "sample_hz": sample_hz,
            "average_hz": average_hz,
            "rolling_hz": rolling_hz,
            "target_hz": threshold_hz,
            "elapsed_sec": elapsed_sec,
            "wall_elapsed_sec": wall_elapsed_sec,
            "paused": bool(self.is_paused),
            "pause_reason": self.pause_reason,
            "paused_duration_sec": self._total_paused_sec(current_time=current_time),
            "frame_count": frame_count,
            "camera_frame_max_age_sec": DEFAULT_CAMERA_FRAME_MAX_AGE_SEC,
            "camera_frame_rejection": self.sample_builder.last_camera_rejection,
            "camera_frame_age_sec_by_key": dict(
                self.sample_builder.last_camera_frame_age_sec_by_key
            ),
            "tactile_frame_counts": tactile_frame_counts,
            "tactile_write_hz": tactile_write_hz,
            "tactile_write_hz_mean": tactile_write_hz_mean,
            "window_sec": float(self.frequency_window_sec),
            "sample_assembly_async": bool(
                getattr(self, "_async_sample_enabled", False)
            ),
            "sample_assembly_skipped_busy": int(
                getattr(self, "_sample_async_skipped_busy", 0)
            ),
            "collection_worker_mode": getattr(
                self, "collection_worker_mode", "thread"
            ),
            "collection_process_rejected_busy": int(
                getattr(self, "_sample_process_rejected_busy", 0)
            ),
            "collection_overload_reason": self.collection_overload_reason,
            "collection_integrity_valid": not bool(
                getattr(self, "_collection_integrity_reasons", [])
            ),
            "collection_integrity_reasons": list(
                getattr(self, "_collection_integrity_reasons", [])
            ),
            "collection_process": (
                {}
                if getattr(self, "process_worker", None) is None
                else self.process_worker.quick_status()
            ),
            "warning": warning,
        }

    def _action_debug_metadata(
        self,
        step_result: SessionStepResult,
        *,
        record_wall_time: float,
    ) -> dict[str, Any]:
        return _build_action_debug_metadata(
            step_result,
            components=getattr(self, "components", None),
            record_wall_time=record_wall_time,
        )

    def record_step(
        self,
        step_result: SessionStepResult,
        *,
        include_debug_metadata: bool = False,
        include_action_debug_metadata: bool = False,
        phase_recorder: Any | None = None,
        metadata_extra: Mapping[str, Any] | None = None,
    ) -> Any | None:
        if not self.can_record_samples:
            return None
        if include_action_debug_metadata:
            metadata_with_action_debug = dict(metadata_extra or {})
            record_wall_time = time.time()
            try:
                metadata_with_action_debug["action_debug"] = self._action_debug_metadata(
                    step_result,
                    record_wall_time=record_wall_time,
                )
            except Exception as exc:  # noqa: BLE001
                metadata_with_action_debug["action_debug"] = {
                    "schema": "dexteleop_action_debug_v1",
                    "enabled": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "record_wall_time": float(record_wall_time),
                }
            metadata_extra = metadata_with_action_debug
        previous_include_debug_metadata = bool(self.sample_builder.include_debug_metadata)
        previous_include_action_debug_metadata = bool(
            getattr(self.sample_builder, "include_action_debug_metadata", False)
        )
        self.sample_builder.include_debug_metadata = bool(include_debug_metadata)
        self.sample_builder.include_action_debug_metadata = bool(include_action_debug_metadata)
        try:
            if getattr(self, "collection_worker_mode", "thread") == "process":
                if self.process_worker is None:
                    raise RuntimeError("formal collection process worker is unavailable")
                self._sample_overload_reason = None
                snapshot = self.sample_builder.snapshot_inputs(
                    step_result,
                    self.collector_components,
                )
                if snapshot is None:
                    result = None
                elif self.process_worker.submit_snapshot(
                    snapshot,
                    metadata_extra=metadata_extra,
                ):
                    self._process_episode_submitted_count += 1
                    result = True
                else:
                    self._sample_process_rejected_busy += 1
                    self._sample_overload_reason = (
                        "collection_process_queue_or_shared_slots_busy"
                    )
                    self._mark_collection_integrity_issue(self._sample_overload_reason)
                    result = None
            else:
                result = self._writer_lifecycle().handle_step(
                    step_result,
                    phase_recorder=phase_recorder,
                    metadata_extra=metadata_extra,
                )
        finally:
            self.sample_builder.include_debug_metadata = previous_include_debug_metadata
            self.sample_builder.include_action_debug_metadata = previous_include_action_debug_metadata
        if result is not None:
            self._note_recorded_sample()
        return result

    def _drain_async_sample(self) -> Any | None:
        future = getattr(self, "_sample_future", None)
        if future is None:
            return None
        self._sample_future = None
        return future.result()

    def record_step_async(
        self,
        step_result: SessionStepResult,
        *,
        include_debug_metadata: bool = False,
        include_action_debug_metadata: bool = False,
        metadata_extra: Mapping[str, Any] | None = None,
    ) -> bool:
        """Queue one causal dataset sample without blocking the control/IK loop."""

        if not bool(getattr(self, "_async_sample_enabled", False)):
            self.record_step(
                step_result,
                include_debug_metadata=include_debug_metadata,
                include_action_debug_metadata=include_action_debug_metadata,
                metadata_extra=metadata_extra,
            )
            return True
        if not self.can_record_samples:
            return False
        previous = getattr(self, "_sample_future", None)
        if previous is not None:
            if not previous.done():
                # At 15 Hz a healthy assembly (~8-20 ms) finishes well before
                # the next request.  Never stall motion if storage is unhealthy.
                self._sample_async_skipped_busy += 1
                self._sample_overload_reason = "collection_snapshot_thread_busy"
                self._mark_collection_integrity_issue(self._sample_overload_reason)
                return False
            self._sample_future = None
            try:
                previous_result = previous.result()
            except Exception as exc:
                error_kind = (
                    "collection_process_error"
                    if getattr(self, "collection_worker_mode", "thread") == "process"
                    else "collection_writer_error"
                )
                self._sample_overload_reason = (
                    f"{error_kind}:{type(exc).__name__}:{exc}"
                )
                self._mark_collection_integrity_issue(self._sample_overload_reason)
                return False
            if (
                getattr(self, "collection_worker_mode", "thread") == "process"
                and previous_result is None
                and self._sample_overload_reason is not None
            ):
                return False
        executor = getattr(self, "_sample_executor", None)
        if executor is None:
            raise RuntimeError("teleop sample executor is not available")
        self._sample_future = executor.submit(
            self.record_step,
            step_result,
            include_debug_metadata=include_debug_metadata,
            include_action_debug_metadata=include_action_debug_metadata,
            phase_recorder=None,
            metadata_extra=None if metadata_extra is None else dict(metadata_extra),
        )
        return True

    @property
    def collection_overload_reason(self) -> str:
        return str(getattr(self, "_sample_overload_reason", None) or "")

    def poll_collection_overload(self) -> str:
        """Nonblocking 60 Hz poll for a rejected/failed collection sample."""

        future = getattr(self, "_sample_future", None)
        if future is not None and future.done():
            self._sample_future = None
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - isolate collection from control.
                error_kind = (
                    "collection_process_error"
                    if getattr(self, "collection_worker_mode", "thread") == "process"
                    else "collection_writer_error"
                )
                self._sample_overload_reason = (
                    f"{error_kind}:{type(exc).__name__}:{exc}"
                )
                self._mark_collection_integrity_issue(self._sample_overload_reason)
            else:
                if result is None and self._sample_overload_reason is None:
                    # Camera/reference rejection is accounted by the normal
                    # sample-frequency diagnostics, not process overload.
                    return ""
        return self.collection_overload_reason

    def collection_worker_ready_status(self) -> tuple[bool, str]:
        if getattr(self, "process_worker", None) is None:
            return True, "thread collection writer ready"
        status = self.process_worker.quick_status()
        error = status.get("worker_error")
        if error:
            return False, f"数采子进程失败：{error}"
        if not bool(status.get("worker_alive")):
            return False, "数采子进程已退出。"
        if int(status.get("slots_available", 0) or 0) <= 0:
            return False, "等待数采子进程释放共享内存槽。"
        return True, "数采子进程队列可用。"

    def _writer_stats_payload(self) -> dict[str, Any]:
        if getattr(self, "process_worker", None) is not None:
            try:
                return {
                    "mode": "process",
                    **asdict(self.process_worker.stats()),
                    "process_status": dict(self.process_worker.process_status),
                }
            except Exception as exc:  # noqa: BLE001
                return {"mode": "process", "error": f"{type(exc).__name__}: {exc}"}
        try:
            if getattr(self, "writer_pool", None) is None:
                raise RuntimeError("dataset writer pool is unavailable")
            return asdict(self.writer_pool.stats())
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _storage_queue_stats_payload(self) -> dict[str, Any]:
        if getattr(self, "process_worker", None) is not None:
            return {
                "enabled": False,
                "mode": "process_whole_sample_queue",
                "storage_queue_size": 0,
                "routes": {},
            }
        try:
            if getattr(self, "writer_observer", None) is None:
                raise RuntimeError("dataset writer observer is unavailable")
            return dict(self.writer_observer.storage_stats())
        except Exception as exc:  # noqa: BLE001
            return {"enabled": False, "error": f"{type(exc).__name__}: {exc}", "routes": {}}

    def _finalize_summary(
        self,
        *,
        reason: str,
        step_index: int,
        last_state: str | None,
        frame_count: int,
        queued_at: float,
        frequency_status: Mapping[str, Any] | None = None,
        pause_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        pause_payload = self.pause_summary() if pause_summary is None else dict(pause_summary)
        return {
            "stop_reason": str(reason),
            "last_step_index": int(step_index),
            "last_state": last_state,
            "episode_frame_count": int(frame_count),
            "dataset_root": str(self.dataset_root),
            "camera_keys": sorted(self.camera_sources),
            "writer_pool_stats_at_stop": self._writer_stats_payload(),
            "storage_queue_stats_at_stop": self._storage_queue_stats_payload(),
            "collection_frequency": {} if frequency_status is None else dict(frequency_status),
            "collection_integrity_valid": not bool(
                getattr(self, "_collection_integrity_reasons", [])
            ),
            "collection_integrity_reasons": list(
                getattr(self, "_collection_integrity_reasons", [])
            ),
            **pause_payload,
            "background_finalize": True,
            "finalize_queued_at": float(queued_at),
        }

    def _run_finalize_task(
        self,
        task: RealDatasetFinalizeTask,
    ) -> None:
        summary_payload = dict(task.summary)

        def run_stage(
            name: str,
            callback: Callable[[], Any],
        ) -> Any:
            started_at = time.perf_counter()
            try:
                return callback()
            finally:
                task.record_stage_timing(name, time.perf_counter() - started_at)

        try:
            if task.before_finalize is not None:
                audit_result = run_stage("position_audit_close", task.before_finalize)
                if audit_result is not None:
                    summary_payload["position_audit_finalize"] = dict(audit_result)
            run_stage("writer_flush", self._writer_lifecycle().flush)
            summary_payload["writer_pool_stats_after_flush"] = self._writer_stats_payload()
            summary_payload["storage_queue_stats"] = self._storage_queue_stats_payload()
            # This is persisted with the episode record; the final writer stage
            # itself is measured on task completion and emitted in the Dataset
            # completion log because it cannot be known before the record is
            # atomically finalized.
            summary_payload["finalize_prewrite_timing_ms"] = dict(task.stage_timings_ms)
            summary_payload["finalize_started_at"] = float(
                task.started_at if task.started_at is not None else time.time()
            )
            run_stage(
                "writer_finalize_episode",
                lambda: self.writer.finalize_episode(task.episode_index, summary=summary_payload),
            )
        except BaseException as exc:  # noqa: BLE001
            task.mark_done(success=False, error=f"{type(exc).__name__}: {exc}")
            return
        task.mark_done(success=True)

    def _start_finalize_task(self, task: RealDatasetFinalizeTask) -> bool:
        """Start one previously queued finalizer without blocking control."""

        if task.done or task.thread is not None:
            return False
        task.mark_started()
        thread = threading.Thread(
            target=self._run_finalize_task,
            name=f"real_dataset_finalize_{task.episode_index}",
            args=(task,),
            daemon=True,
        )
        task.thread = thread
        try:
            thread.start()
        except BaseException as exc:  # noqa: BLE001 - preserve task completion semantics.
            task.mark_done(success=False, error=f"thread_start {type(exc).__name__}: {exc}")
            return False
        return True

    def start_pending_finalize_tasks(self, *, max_tasks: int = 1) -> list[RealDatasetFinalizeTask]:
        """Launch at most ``max_tasks`` queued finalizers in FIFO order.

        Real teleoperation currently admits one active recording at a time.  The
        explicit bound nevertheless prevents a future caller from starting
        multiple CPU/IO-heavy finalizers together at an episode boundary.
        """

        started: list[RealDatasetFinalizeTask] = []
        for task in self.finalize_tasks:
            if len(started) >= max(0, int(max_tasks)):
                break
            if task.pending and self._start_finalize_task(task):
                started.append(task)
        return started

    def stop_episode(
        self,
        *,
        reason: str,
        step_index: int,
        last_state: str | None,
        defer_finalize_start: bool = False,
        before_finalize: Callable[[], Mapping[str, Any] | None] | None = None,
    ) -> RealDatasetFinalizeTask | None:
        if self.active_episode_index is None:
            return None
        self._drain_async_sample()
        stop_time = time.time()
        if self.is_paused:
            self._close_pause_segment(
                resume_reason=f"stop:{reason}",
                now=stop_time,
                step_index=step_index,
            )
        episode_index = int(self.active_episode_index)
        frame_count = int(self.frame_count)
        frequency_status = self.collection_frequency_status()
        pause_summary = self.pause_summary()
        self.active_episode_index = None
        self.active_started_at = None
        self._episode_start_monotonic = None
        self._pause_started_at = None
        self._pause_started_monotonic = None
        self._pause_reason = None
        self._pause_started_frame_index = None
        self._pause_started_step_index = None
        self._pause_started_last_state = None
        if frame_count <= 0:
            self._writer_lifecycle().flush()
            self.writer.discard_episode(episode_index)
            self.discarded_episodes += 1
            self._record_timestamps.clear()
            self._pause_segments = []
            self._paused_accumulated_sec = 0.0
            return None
        task = RealDatasetFinalizeTask(
            episode_index=episode_index,
            frame_count=frame_count,
            reason=str(reason),
            queued_at=time.time(),
            before_finalize=before_finalize,
        )
        task.summary = self._finalize_summary(
            reason=reason,
            step_index=step_index,
            last_state=last_state,
            frame_count=frame_count,
            queued_at=task.queued_at,
            frequency_status=frequency_status,
            pause_summary=pause_summary,
        )
        self.finalize_tasks.append(task)
        if not defer_finalize_start:
            self._start_finalize_task(task)
        return task

    def poll_finalize_tasks(self) -> list[RealDatasetFinalizeTask]:
        completed: list[RealDatasetFinalizeTask] = []
        remaining: list[RealDatasetFinalizeTask] = []
        for task in self.finalize_tasks:
            if not task.done:
                remaining.append(task)
                continue
            if task.thread is not None:
                task.thread.join(timeout=0.0)
            if task.success:
                self.saved_episodes += 1
            completed.append(task)
        self.finalize_tasks = remaining
        return completed

    def wait_finalize_tasks(self) -> None:
        while self.finalizing:
            self.start_pending_finalize_tasks()
            for task in list(self.finalize_tasks):
                if task.thread is not None:
                    task.thread.join(timeout=0.1)
            self.poll_finalize_tasks()

    def close(self) -> None:
        steps = []
        if getattr(self, "_sample_executor", None) is not None:
            steps.extend(
                [
                    ("sample_assembly_drain", self._drain_async_sample),
                    (
                        "sample_assembly_executor",
                        lambda: self._sample_executor.shutdown(wait=True, cancel_futures=False),
                    ),
                ]
            )
        if self.arm_state_poller is not None:
            steps.append(("arm_state_poller", self.arm_state_poller.close))
        if self.hand_state_poller is not None:
            steps.append(("hand_state_poller", self.hand_state_poller.close))
        steps.extend(
            [
                ("finalize_tasks", self.wait_finalize_tasks),
                ("writer_runtime", lambda: self._writer_lifecycle().close(flush=True)),
            ]
        )
        run_best_effort_cleanup("real_teleop_dataset_recorder.close", steps)


def _dataset_storage_debug_text(dataset_recorder: RealTeleopDatasetRecorder) -> str:
    stats = dataset_recorder._storage_queue_stats_payload()
    if not bool(stats.get("enabled", False)):
        return "storage_queue=off"
    routes = stats.get("routes")
    if not isinstance(routes, Mapping) or not routes:
        return (
            "storage_queue=enabled"
            f":size:{int(stats.get('storage_queue_size', 0) or 0)},routes:none"
        )
    parts: list[str] = []
    for key, route in sorted(routes.items()):
        if not isinstance(route, Mapping):
            parts.append(f"{key}=invalid")
            continue
        parts.append(
            f"{key}=q:{int(route.get('queued', 0) or 0)},"
            f"enq:{int(route.get('enqueued', 0) or 0)},"
            f"sub:{int(route.get('submitted', 0) or 0)},"
            f"drop:{int(route.get('dropped', 0) or 0)},"
            f"high:{int(route.get('high_water', 0) or 0)}/"
            f"{int(route.get('max_queue_size', 0) or 0)}"
        )
    return "storage_queue=" + "; ".join(parts)


def _dataset_recording_debug_text(
    *,
    dataset_recorder: RealTeleopDatasetRecorder,
    components: RealTeleopComponents,
    camera_sources: Mapping[str, RealSenseRGBSource],
    camera_previewer: CameraPreviewer | None,
    now: float,
) -> str:
    writer_text = _collection_writer_debug_text(dataset_recorder.writer_pool)
    storage_text = _dataset_storage_debug_text(dataset_recorder)
    camera_text = _collection_camera_debug_text(
        camera_sources,
        camera_previewer=camera_previewer,
    )
    hand_text = _collection_hand_debug_text(
        components.active_sides,
        lambda side: _hand_backend_for_side(components, side),
        now=now,
    )
    return f"{writer_text} {storage_text} {camera_text} {hand_text}"


def _position_ik_tuning_for_args(
    args: argparse.Namespace,
    tuning: RM65IKTuningConfig,
    *,
    arm_control_mode: str,
) -> RM65IKTuningConfig:
    orientation_weight = getattr(args, "position_ik_orientation_weight", None)
    if arm_control_mode != "position" or orientation_weight is None:
        return tuning
    weight = float(orientation_weight)
    mode_name = "position_runtime_override"
    objective_modes = dict(tuning.objective_modes)
    objective_modes[mode_name] = IKObjectiveWeights(
        single_arm_position_weight=1.0,
        single_arm_orientation_weight=weight,
        dual_arm_position_cost=1.0,
        dual_arm_orientation_cost=weight,
    )
    return replace(
        tuning,
        ik_objective_mode=mode_name,
        objective_modes=objective_modes,
    )


def _position_tracker_motion_intent_config(
    args: argparse.Namespace,
    *,
    arm_control_mode: str,
) -> TrackerMotionIntentConfig:
    config = TrackerMotionIntentConfig()
    if arm_control_mode != "position":
        return config
    scale = float(
        getattr(args, "position_inactive_enter_threshold_scale", 1.0)
    )
    return replace(
        config,
        enter_linear_speed_m_s=config.enter_linear_speed_m_s * scale,
        enter_angular_speed_rad_s=config.enter_angular_speed_rad_s * scale,
        enter_translation_span_m=config.enter_translation_span_m * scale,
        enter_rotation_span_rad=config.enter_rotation_span_rad * scale,
    )


def _build_components(args: argparse.Namespace) -> RealTeleopComponents:
    _resolve_movement_multiple_args(args)
    debug_timing_modules = _debug_timing_modules(args)
    active_sides = _active_sides_from_args(args)
    arm_control_mode = _arm_control_mode_arg(args)
    impedance_mode = _impedance_mode_arg(args)
    strict_tool_profile = _strict_tool_profile_arg(args)
    force_tare_sec = _force_tare_sec_arg(args)
    logical_ip_by_side: dict[str, tuple[str, str]] = {}
    inactive_arm_config: list[str] = []
    for source_name, (logical_side, arm_ip) in _resolve_logical_side_assignments(args).items():
        if not arm_ip:
            continue
        if logical_side not in active_sides:
            inactive_arm_config.append(f"{source_name}->{logical_side}")
            continue
        logical_ip_by_side[logical_side] = (source_name, arm_ip)
    if inactive_arm_config:
        raise ValueError(
            "Inactive logical side received an arm IP; remove it or include that side "
            f"in --active-sides. inactive={inactive_arm_config}"
        )
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
        position_command_planner == POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED
        and arm_control_mode != "position"
    ):
        raise ValueError(
            "--position-command-planner acceleration_limited requires position arm control mode."
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
        and position_command_planner != POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED
    ):
        raise ValueError(
            "--position-max-command-step-deg requires "
            "--position-command-planner acceleration_limited."
        )
    if (
        position_command_planner == POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED
        and not position_background_streaming_enabled
    ):
        raise ValueError(
            "--position-command-planner acceleration_limited requires the high-rate position "
            "background stream; enable the normal position CANFD follow path."
        )
    if (
        position_command_planner == POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED
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
        _dmtac_backend_configs(args, active_sides)
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
                        prefer_non_elbow_singular_branch=bool(
                            arm_control_mode == "position"
                            and getattr(
                                args,
                                "position_prefer_non_elbow_singular_branch",
                                True,
                            )
                        ),
                        elbow_singularity_threshold_deg=3.0,
                        elbow_singularity_selection_margin_deg=1.0,
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
        reader = prepare_reader(
            config_root=config_root,
            sample_hz=float(getattr(args, "gripper_angle_trigger_rate_hz", 30.0)),
        )
        angle_trigger_source = AngleTriggerSource(
            reader,
            rate_hz=float(getattr(args, "gripper_angle_trigger_rate_hz", 30.0)),
            config_root=config_root,
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
            input_ready = hand_port if gripper_input_source == "q6" else angle_trigger_source is not None
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
        ),
        command_guard=(
            lambda command, _observation, now: fixed_arm_final_collision_fault(
                command,
                fix_arm_controller=fix_arm_controller,
                dual_arm_retargeter=dual_arm_retargeter,
                now=now,
            )
            if fix_arm_controller is not None and dual_arm_retargeter is not None
            else None
        ),
        fault_filter=_compose_real_fault_filter(
            fix_arm_controller=fix_arm_controller,
            init_arm_controller=init_arm_controller,
        ),
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
        temporary_safety_limiter=temporary_safety_limiter,
        fix_arm_controller=fix_arm_controller,
        init_arm_controller=init_arm_controller,
        position_reference_publisher=position_reference_publisher,
    )


def _run_initial_alignment_start_check(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> str:
    recorded_qpos_pair = _alignment_qpos_pair(components.loaded_initial_alignment)
    if recorded_qpos_pair is None:
        if components.loaded_initial_alignment is not None:
            _log(
                "StartCheck",
                "loaded initial alignment does not contain both left/right arm_qpos entries; startup joint check skipped.",
            )
        return "skipped"

    current_qpos_pair = {
        "left": _state_qpos(components.left_preview_backend, DEFAULT_LEFT_ARM_QPOS),
        "right": _state_qpos(components.right_preview_backend, DEFAULT_RIGHT_ARM_QPOS),
    }
    inactive_sides = set(("left", "right")) - set(components.active_sides)
    for side in inactive_sides:
        current_qpos_pair[side] = recorded_qpos_pair[side]
    checks = compare_start_qpos_pair(
        current_qpos_pair=current_qpos_pair,
        recorded_qpos_pair=recorded_qpos_pair,
        joint_distance_weights=components.joint_distance_weights,
        max_abs_threshold_deg=float(args.max_start_joint_error_deg),
        weighted_threshold_deg=float(args.max_start_weighted_error_deg),
    )
    checks = {side: checks[side] for side in components.active_sides}
    _log("StartCheck", f"initial-alignment check: {_start_state_summary(checks)}")
    for side in components.active_sides:
        check = checks[side]
        _log(
            "StartCheck",
            f"{side} current_q={_qpos_summary(check.current_qpos)} "
            f"alignment_q={_qpos_summary(check.recorded_qpos)} "
            f"abs_joint_error_deg={_qpos_summary(check.abs_joint_error_deg)}",
        )
    start_ok = all(check.within_limits for check in checks.values())
    if not start_ok and not args.ignore_start_mismatch:
        raise RuntimeError(
            "Real teleop aborted because the current start-state joint error exceeds the configured threshold."
        )
    if not start_ok:
        return "ignored_mismatch"
    return "passed"


def _sync_preview_from_real_state_real(
    components: RealTeleopComponents,
    *,
    now: float | None = None,
    required: bool,
) -> dict[str, bool]:
    synced = {side: False for side in components.active_sides}
    backends = (
        ("left", components.left_state_backend, components.left_preview_backend),
        ("right", components.right_state_backend, components.right_preview_backend),
    )
    active = [
        (side, state_backend, preview_backend)
        for side, state_backend, preview_backend in backends
        if side in components.active_sides and state_backend is not None
    ]
    parallel_submit_failed = False
    if components.arm_state_read_executor is not None and len(active) > 1:
        try:
            future_map = {
                components.arm_state_read_executor.submit(state_backend.refresh_state_from_hardware): (
                    side,
                    preview_backend,
                )
                for side, state_backend, preview_backend in active
            }
        except RuntimeError:
            parallel_submit_failed = True
        else:
            for future, (side, preview_backend) in future_map.items():
                state = future.result()
                qpos = state.get("current_qpos")
                if qpos is None:
                    if required:
                        raise RuntimeError(
                            f"{side}_arm real-state sync failed: no current_qpos available from hardware."
                        )
                    continue
                preview_backend.set_local_qpos(qpos)
                synced[side] = True
    if components.arm_state_read_executor is None or len(active) <= 1 or parallel_submit_failed:
        for side, state_backend, preview_backend in backends:
            if side not in components.active_sides or state_backend is None:
                continue
            state = state_backend.refresh_state_from_hardware()
            qpos = state.get("current_qpos")
            if qpos is None:
                if required:
                    raise RuntimeError(
                        f"{side}_arm real-state sync failed: no current_qpos available from hardware."
                    )
                continue
            preview_backend.set_local_qpos(qpos)
            synced[side] = True
    _sync_dual_arm_retargeter_for_real_control(components)
    return synced


def _reset_runtime_motion_state_from_real_qpos(
    components: RealTeleopComponents,
    *,
    required: bool,
) -> dict[str, bool]:
    reset_start = time.perf_counter()
    synced = {side: False for side in components.active_sides}
    timing_parts: list[str] = []
    for side in components.active_sides:
        side_start = time.perf_counter()
        state_backend = components.left_state_backend if side == "left" else components.right_state_backend
        preview_backend = components.left_preview_backend if side == "left" else components.right_preview_backend
        command_backend = _command_backend_for_side(components, side)
        if state_backend is None:
            if required:
                raise RuntimeError(f"{side}_arm has no real state backend for runtime restart.")
            timing_parts.append(f"{side}:missing_state_backend")
            continue
        refresh_start = time.perf_counter()
        state = state_backend.refresh_state_from_hardware()
        refresh_elapsed_ms = (time.perf_counter() - refresh_start) * 1000.0
        qpos = state.get("current_qpos")
        if qpos is None:
            if required:
                raise RuntimeError(
                    f"{side}_arm runtime restart failed: no current_qpos available from hardware."
                )
            timing_parts.append(f"{side}:refresh={refresh_elapsed_ms:.1f}ms no_qpos")
            continue
        try:
            qpos_array = np.asarray(qpos, dtype=float).reshape(6)
        except (TypeError, ValueError) as exc:
            if required:
                raise RuntimeError(
                    f"{side}_arm runtime restart failed: invalid current_qpos from hardware."
                ) from exc
            continue
        if not np.all(np.isfinite(qpos_array)):
            if required:
                raise RuntimeError(
                    f"{side}_arm runtime restart failed: non-finite current_qpos from hardware."
                )
            continue
        reset_side_start = time.perf_counter()
        preview_backend.set_local_qpos(qpos_array)
        preview_anchor_qpos = preview_backend.get_state().get("current_qpos")
        if preview_anchor_qpos is None:
            if required:
                raise RuntimeError(
                    f"{side}_arm runtime restart failed: preview did not retain current_qpos."
                )
            continue
        preview_anchor_qpos = np.asarray(preview_anchor_qpos, dtype=float).reshape(6)
        if not np.all(np.isfinite(preview_anchor_qpos)):
            if required:
                raise RuntimeError(
                    f"{side}_arm runtime restart failed: preview produced non-finite current_qpos."
                )
            continue
        if command_backend is not None:
            command_backend.reset_motion_state(preview_anchor_qpos)
        reset_side_elapsed_ms = (time.perf_counter() - reset_side_start) * 1000.0
        synced[side] = True
        timing_parts.append(
            f"{side}:total={(time.perf_counter() - side_start) * 1000.0:.1f}ms "
            f"refresh={refresh_elapsed_ms:.1f}ms "
            f"reset={reset_side_elapsed_ms:.1f}ms"
        )
    sync_start = time.perf_counter()
    _sync_dual_arm_retargeter_for_real_control(components)
    sync_elapsed_ms = (time.perf_counter() - sync_start) * 1000.0
    _log(
        "CalibrationTiming",
        "runtime_restart_state "
        f"total={(time.perf_counter() - reset_start) * 1000.0:.1f}ms "
        f"retargeter_sync={sync_elapsed_ms:.1f}ms "
        + " | ".join(timing_parts),
    )
    return synced


def _sync_real_execution_step_limits(
    components: RealTeleopComponents,
    *,
    now: float | None = None,
) -> None:
    current_time = time.time() if now is None else float(now)
    active_sides = tuple(getattr(components, "active_sides", ()) or ())
    if not active_sides:
        active_sides = tuple(
            side
            for side in ("left", "right")
            if _command_backend_for_side(components, side) is not None
        )
    for side in active_sides:
        backend = _command_backend_for_side(components, side)
        prepare_step = getattr(backend, "prepare_effective_max_joint_step", None)
        if callable(prepare_step):
            prepare_step(now=current_time)

    if components.dual_arm_retargeter is None:
        return
    if not components.dual_arm_retargeter.config.enforce_realistic_joint_step:
        return
    if components.left_command_backend is None or components.right_command_backend is None:
        return
    tracker = components.arm_loop_step_dt_tracker
    command_config = components.left_command_backend.config
    scale_outer_execution_step_by_actual_dt = bool(
        getattr(command_config, "scale_outer_execution_step_by_actual_dt", True)
    )
    backend_control_dt = float(command_config.control_dt)
    action_hz = getattr(command_config, "action_hz", None)
    nominal_outer_dt = (
        1.0 / float(action_hz)
        if action_hz is not None and float(action_hz) > 0.0
        else backend_control_dt
    )
    # Impedance deliberately does not expand a per-frame motion allowance
    # after a slow outer frame.  Its fixed allowance must nevertheless use the
    # nominal *outer* target cadence, not the 40 Hz inner controller.
    # Position backends retain their existing actual-dt policy.
    base_dt = (
        backend_control_dt
        if scale_outer_execution_step_by_actual_dt
        else nominal_outer_dt
    )
    smoothed_loop_dt = float(tracker.get("smoothed_loop_dt", base_dt) or base_dt)
    last_loop_time = tracker.get("last_loop_time")
    if not scale_outer_execution_step_by_actual_dt or last_loop_time is None:
        smoothed_loop_dt = base_dt
    else:
        raw_loop_dt = max(0.0, current_time - float(last_loop_time))
        if raw_loop_dt > float(command_config.actual_send_dt_stall_reset_sec):
            smoothed_loop_dt = base_dt
        else:
            alpha = float(command_config.actual_send_dt_alpha)
            smoothed_loop_dt = alpha * raw_loop_dt + (1.0 - alpha) * smoothed_loop_dt
    tracker["last_loop_time"] = current_time
    tracker["smoothed_loop_dt"] = smoothed_loop_dt
    effective_loop_dt = float(
        np.clip(
            smoothed_loop_dt,
            base_dt * float(command_config.actual_send_dt_min_scale),
            base_dt * float(command_config.actual_send_dt_max_scale),
        )
    )
    explicit_steps_configured = (
        components.dual_arm_retargeter.config.left_execution_max_joint_step_rad is not None
        and components.dual_arm_retargeter.config.right_execution_max_joint_step_rad is not None
    )
    if explicit_steps_configured or components.dual_arm_retargeter.config.joint_velocity_profile is None:
        components.dual_arm_retargeter.set_execution_max_joint_step(
            left_step_rad=np.asarray(components.left_command_backend.config.max_angvel, dtype=float)
            * effective_loop_dt,
            right_step_rad=np.asarray(components.right_command_backend.config.max_angvel, dtype=float)
            * effective_loop_dt,
        )
        return
    components.dual_arm_retargeter.set_execution_dt_sec(effective_loop_dt)


def _apply_active_arm_command_to_preview_real(
    components: RealTeleopComponents,
    active_command,
) -> None:
    if active_command is None:
        return
    left_command = active_command.commands.get("left_arm")
    if left_command is not None:
        components.left_preview_backend.send_command(left_command.copy())
    right_command = active_command.commands.get("right_arm")
    if right_command is not None:
        components.right_preview_backend.send_command(right_command.copy())


def _acceleration_limited_position_execution_active(
    components: RealTeleopComponents,
) -> bool:
    active_backends = [
        _command_backend_for_side(components, side)
        for side in components.active_sides
    ]
    return bool(active_backends) and all(
        isinstance(backend, RM65ArmBackend)
        and str(getattr(backend.config, "position_command_planner", ""))
        == POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED
        for backend in active_backends
    )


def _sync_dual_arm_retargeter_for_real_control(
    components: RealTeleopComponents,
    *,
    reset_tracking: bool = True,
) -> None:
    """Seed coupled IK from the state the position transport actually accepted.

    The legacy linear path reaches each outer qref within one reference period,
    so its historical preview-state synchronization is retained. The optional
    acceleration-limited path can intentionally lag qref (especially at the
    10% degraded speed); seeding IK from preview would then certify collision
    and joint steps around a configuration the real transport has not reached.
    """

    if components.dual_arm_retargeter is None:
        return
    if not _acceleration_limited_position_execution_active(components):
        _sync_dual_arm_retargeter_from_preview(
            components,
            reset_tracking=reset_tracking,
        )
        return

    qpos_by_side: dict[str, np.ndarray] = {}
    accepted_monotonic_by_side: dict[str, float] = {}
    source_by_side: dict[str, str] = {}
    for side in ("left", "right"):
        backend = _command_backend_for_side(components, side)
        if not isinstance(backend, RM65ArmBackend):
            raise RuntimeError(
                f"Acceleration-limited coupled IK requires an RM65 command backend for {side}."
            )
        state = backend.get_state()
        raw_qpos = state.get("last_accepted_qpos")
        source = "last_accepted_qpos"
        if raw_qpos is None:
            raw_qpos = state.get("current_qpos")
            source = "current_qpos_fallback"
        try:
            qpos = np.asarray(raw_qpos, dtype=float).reshape(6)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{side}_arm has no valid accepted qcmd for coupled IK synchronization."
            ) from exc
        if not np.all(np.isfinite(qpos)):
            raise RuntimeError(
                f"{side}_arm accepted qcmd is non-finite; refusing coupled IK synchronization."
            )
        qpos_by_side[side] = qpos.copy()
        source_by_side[side] = source
        accepted_monotonic = state.get("last_accepted_timestamp_monotonic")
        if accepted_monotonic is not None:
            try:
                normalized_time = float(accepted_monotonic)
            except (TypeError, ValueError):
                normalized_time = float("nan")
            if np.isfinite(normalized_time):
                accepted_monotonic_by_side[side] = normalized_time

    components.dual_arm_retargeter.set_current_qpos_pair(
        left_qpos=qpos_by_side["left"],
        right_qpos=qpos_by_side["right"],
        reset_tracking=reset_tracking,
    )
    timestamps = tuple(accepted_monotonic_by_side.values())
    skew_sec = None if len(timestamps) < 2 else max(timestamps) - min(timestamps)
    components.arm_loop_step_dt_tracker["retargeter_execution_state_sync"] = {
        "source_by_side": source_by_side,
        "accepted_monotonic_by_side": dict(accepted_monotonic_by_side),
        "accepted_timestamp_skew_sec": skew_sec,
        "left_qpos": qpos_by_side["left"].tolist(),
        "right_qpos": qpos_by_side["right"].tolist(),
    }


def _command_backend_for_side(
    components: RealTeleopComponents,
    side: str,
) -> RM65ArmBackend | RM65ImpedanceArmBackend | None:
    return components.left_command_backend if side == "left" else components.right_command_backend


def _active_command_backend(components: RealTeleopComponents) -> RM65ArmBackend | RM65ImpedanceArmBackend | None:
    for side in components.active_sides:
        backend = _command_backend_for_side(components, side)
        if backend is not None:
            return backend
    return None


def _active_impedance_force_backends(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> list[tuple[str, RM65ImpedanceArmBackend]]:
    if _arm_control_mode_arg(args) != "impedance":
        return []
    backends: list[tuple[str, RM65ImpedanceArmBackend]] = []
    for side in components.active_sides:
        backend = _command_backend_for_side(components, side)
        if not isinstance(backend, RM65ImpedanceArmBackend):
            raise RuntimeError(f"{side}_arm command backend is not RM65ImpedanceArmBackend.")
        backends.append((side, backend))
    return backends


def _impedance_force_control_warmup_backends(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> list[tuple[str, RM65ImpedanceArmBackend]]:
    backends = []
    for side, backend in _active_impedance_force_backends(args, components):
        if not bool(getattr(backend.config, "connect_hardware", True)):
            continue
        if not backend.force_control_warm_started():
            backends.append((side, backend))
    return backends


def _estimate_impedance_force_control_warmup_sec(
    args: argparse.Namespace,
    backends: Sequence[tuple[str, RM65ImpedanceArmBackend]],
) -> float:
    if not backends:
        return 0.0
    tare_sec = max(0.0, _force_tare_sec_arg(args))
    parallel = len(backends) > 1 and int(getattr(args, "arm_io_threads", 1)) > 1
    tare_total_sec = tare_sec if parallel else tare_sec * len(backends)
    return max(0.5, tare_total_sec + 0.5)


def _warm_start_impedance_force_control_before_home(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    clutch_input: "RealTeleopControlInput",
) -> bool:
    backends = _impedance_force_control_warmup_backends(args, components)
    if not backends:
        return True
    total_estimate_sec = _estimate_impedance_force_control_warmup_sec(args, backends)
    active_sides = tuple(side for side, _ in backends)
    warmup_started_perf = time.perf_counter()
    _log(
        "ForceControlWarmup",
        "cold_start required before runtime home "
        f"sides={list(active_sides)} estimated={total_estimate_sec:.2f}s "
        f"tare_sec={_force_tare_sec_arg(args):.2f} "
        f"io_threads={max(1, int(getattr(args, 'arm_io_threads', 1)))}",
    )

    def warm_start_one(item: tuple[str, RM65ImpedanceArmBackend]) -> tuple[str, dict[str, object], dict[str, float]]:
        side, backend = item
        side_start = time.perf_counter()
        warm_start = time.perf_counter()
        backend.warm_start_force_control()
        warm_elapsed_ms = (time.perf_counter() - warm_start) * 1000.0
        return (
            side,
            backend.get_state(),
            {
                "warm_start_ms": warm_elapsed_ms,
                "total_ms": (time.perf_counter() - side_start) * 1000.0,
            },
        )

    def run_warmup() -> list[tuple[str, dict[str, object], dict[str, float]]]:
        if len(backends) > 1 and int(getattr(args, "arm_io_threads", 1)) > 1:
            with ThreadPoolExecutor(max_workers=len(backends), thread_name_prefix="rm65_warmup_arm") as executor:
                return [future.result() for future in (executor.submit(warm_start_one, item) for item in backends)]
        return [warm_start_one(item) for item in backends]

    clutch_input.publish_force_control_warmup_progress(
        elapsed_sec=0.0,
        total_sec=total_estimate_sec,
        active_sides=active_sides,
    )
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="rm65_warmup_gate") as executor:
        future = executor.submit(run_warmup)
        while not future.done():
            elapsed_sec = time.perf_counter() - warmup_started_perf
            clutch_input.publish_force_control_warmup_progress(
                elapsed_sec=elapsed_sec,
                total_sec=total_estimate_sec,
                active_sides=active_sides,
            )
            clutch_input.terminal.poll()
            clutch_input.terminal.consume_calibration_request()
            clutch_input.terminal.consume_episode_start_request()
            clutch_input.terminal.consume_episode_stop_request()
            if clutch_input.terminal.should_quit:
                clutch_input.should_quit = True
            if clutch_input.ui is not None:
                for event in clutch_input.ui.consume_events():
                    if event == "q":
                        clutch_input.should_quit = True
            time.sleep(0.05)
        try:
            results = future.result()
        except Exception as exc:  # noqa: BLE001
            message = f"力控冷启动失败: {type(exc).__name__}: {exc}"
            _log("ForceControlWarmup", message)
            clutch_input.publish_calibration_waiting(message)
            return False

    elapsed_ms = (time.perf_counter() - warmup_started_perf) * 1000.0
    timing_parts = []
    for side, state, timing in results:
        timing_parts.append(
            f"{side}:total={timing['total_ms']:.1f}ms "
            f"warm_start={timing['warm_start_ms']:.1f}ms "
            f"status={state.get('send_result_text')}"
        )
    _log(
        "ForceControlWarmup",
        f"cold_start completed total={elapsed_ms:.1f}ms " + " | ".join(timing_parts),
    )
    clutch_input.publish_force_control_warmup_progress(
        elapsed_sec=total_estimate_sec,
        total_sec=total_estimate_sec,
        active_sides=active_sides,
    )
    if clutch_input.should_quit:
        return False
    return True


def _emergency_stop_real_arms(
    components: RealTeleopComponents,
    *,
    reason: str,
    now: float | None = None,
) -> None:
    stop_time = time.time() if now is None else float(now)
    for side in components.active_sides:
        backend = _command_backend_for_side(components, side)
        if backend is None:
            continue
        stop = getattr(backend, "emergency_stop", None)
        label = f"{side}_arm"
        try:
            if callable(stop):
                stop()
            else:
                backend.close()
        except Exception as exc:
            _log(
                "EmergencyStop",
                f"{label} stop failed reason={reason}: {type(exc).__name__}: {exc}",
                now=stop_time,
            )
        else:
            _log("EmergencyStop", f"{label} stop requested reason={reason}", now=stop_time)


def _enable_impedance_force_control_after_runtime_home(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    home_captured: bool = True,
) -> None:
    force_control_start = time.perf_counter()
    backends = _active_impedance_force_backends(args, components)
    if not backends:
        return
    _log(
        "ForceControl",
        f"runtime home {'captured' if home_captured else 'skipped'}; "
        f"hold active tools still for tare_sec={_force_tare_sec_arg(args):.2f}s "
        f"before enabling pose impedance.",
    )

    def enable_one(
        item: tuple[str, RM65ImpedanceArmBackend],
    ) -> tuple[str, dict[str, object], dict[str, float]]:
        side, backend = item
        side_start = time.perf_counter()
        enable_start = time.perf_counter()
        backend.enable_force_control()
        enable_elapsed_ms = (time.perf_counter() - enable_start) * 1000.0
        timeout_sec = max(1.0, 3.0 * float(backend.config.control_dt))
        wait_start = time.perf_counter()
        state = backend.wait_until_force_control_ready(timeout_sec=timeout_sec)
        wait_elapsed_ms = (time.perf_counter() - wait_start) * 1000.0
        return (
            side,
            state,
            {
                "enable_ms": enable_elapsed_ms,
                "wait_ready_ms": wait_elapsed_ms,
                "total_ms": (time.perf_counter() - side_start) * 1000.0,
            },
        )

    if len(backends) > 1 and int(getattr(args, "arm_io_threads", 1)) > 1:
        with ThreadPoolExecutor(max_workers=len(backends), thread_name_prefix="rm65_tare") as executor:
            results = [future.result() for future in (executor.submit(enable_one, item) for item in backends)]
    else:
        results = [enable_one(item) for item in backends]
    timing_parts = []
    for side, state, timing in results:
        timing_parts.append(
            f"{side}:total={timing['total_ms']:.1f}ms "
            f"enable={timing['enable_ms']:.1f}ms "
            f"wait_ready={timing['wait_ready_ms']:.1f}ms"
        )
        record = state.get("last_impedance_record")
        step = record.get("step") if isinstance(record, dict) else None
        force_norm = record.get("force_norm_n") if isinstance(record, dict) else None
        torque_norm = record.get("torque_norm_nm") if isinstance(record, dict) else None
        wrench_text = ""
        if force_norm is not None and torque_norm is not None:
            wrench_text = f" first_tick_force={float(force_norm):.3f}N torque={float(torque_norm):.3f}Nm"
        target_offset = record.get("target_offset") if isinstance(record, dict) else None
        if target_offset is not None:
            offset_values = np.asarray(target_offset, dtype=float).reshape(-1)
            if offset_values.size >= 6 and np.all(np.isfinite(offset_values[:6])):
                linear_m = float(np.linalg.norm(offset_values[:3]))
                angular_rad = float(np.linalg.norm(offset_values[3:6]))
                wrench_text += f" first_tick_offset={linear_m:.5f}m/{angular_rad:.5f}rad"
        _log(
            "ForceControl",
            f"{side}_arm pose impedance enabled "
            f"tool={state.get('active_tool_name')} "
            f"profile={state.get('active_tool_profile_path')} "
            f"tare_sec={state.get('force_tare_sec')} "
            f"loop={state.get('impedance_loop_state')} "
            f"tick={step} "
            f"send={state.get('send_result_text')}"
            f"{wrench_text}",
        )
    _log(
        "CalibrationTiming",
        "force_control_after_home "
        f"total={(time.perf_counter() - force_control_start) * 1000.0:.1f}ms "
        f"tare_sec={_force_tare_sec_arg(args):.2f} "
        + " | ".join(timing_parts),
    )


def _short_status_text(value: object, *, max_len: int = 160) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)] + "..."


def _command_backend_status_text(state: dict[str, object]) -> str:
    if not state:
        return "none"
    if "impedance_enabled" in state:
        bits = [
            f"imp={int(bool(state.get('impedance_enabled')))}",
            f"loop={state.get('impedance_loop_state')}",
            f"send={state.get('send_result_text')}",
        ]
        ik_metadata = state.get("ik_metadata")
        if isinstance(ik_metadata, Mapping):
            ik_status = ik_metadata.get("status") or ik_metadata.get("ik_status")
            if ik_status:
                bits.append(f"ik={_short_status_text(ik_status, max_len=48)}")
        fault = state.get("fault")
        if fault:
            bits.append(f"fault={_short_status_text(fault)}")
        return ",".join(bits)
    code = state.get("send_result_code")
    text = state.get("send_result_text")
    bits = [
        f"send={code}:{_short_status_text(text)}" if text is not None else f"send={code}"
    ]
    stream = state.get("background_streaming")
    if isinstance(stream, Mapping):
        planner = str(stream.get("position_command_planner") or "")
        if planner == POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED:
            mode = str(stream.get("trajectory_policy_mode") or "unknown")
            scale = float(stream.get("trajectory_policy_speed_scale", 1.0) or 0.0)
            bits.append(f"traj={mode}/{scale:.2f}")
            last_debug = stream.get("last_trajectory_debug")
            if isinstance(last_debug, Mapping):
                for key, label in (
                    ("trajectory_qd_command_rad_s", "qd"),
                    ("trajectory_qdd_command_rad_s2", "qdd"),
                ):
                    try:
                        values = np.asarray(last_debug.get(key), dtype=float).reshape(-1)
                    except (TypeError, ValueError):
                        continue
                    if values.size and np.all(np.isfinite(values)):
                        bits.append(f"{label}max={float(np.max(np.abs(values))):.2f}")
            if bool(stream.get("quick_stop_active", False)):
                bits.append(
                    "hold="
                    + _short_status_text(stream.get("quick_stop_reason"), max_len=64)
                )
    return ",".join(bits)


def _command_backend_health_signature(state: Mapping[str, object] | None) -> tuple[object, ...]:
    """Keep high-rate command diagnostics out of the status change gate."""

    if not state:
        return ("none",)
    if "impedance_enabled" in state:
        ik_metadata = state.get("ik_metadata")
        ik_status = None
        if isinstance(ik_metadata, Mapping):
            ik_status = ik_metadata.get("status") or ik_metadata.get("ik_status")
        return (
            "impedance",
            bool(state.get("impedance_enabled")),
            str(state.get("impedance_loop_state") or ""),
            state.get("send_result_code"),
            str(ik_status or ""),
            bool(state.get("fault")),
        )
    stream = state.get("background_streaming")
    if not isinstance(stream, Mapping):
        stream = {}
    return (
        "position",
        state.get("send_result_code"),
        str(stream.get("position_command_planner") or ""),
        str(stream.get("trajectory_policy_mode") or ""),
        bool(stream.get("quick_stop_active", False)),
        str(stream.get("quick_stop_reason") or ""),
    )


def _temporary_safety_health_signature(status: object) -> tuple[object, ...] | None:
    if not isinstance(status, Mapping):
        return None
    effectors = status.get("effectors")
    effector_statuses: list[tuple[object, ...]] = []
    if isinstance(effectors, Mapping):
        for name, value in sorted(effectors.items(), key=lambda item: str(item[0])):
            details = value if isinstance(value, Mapping) else {}
            effector_statuses.append(
                (
                    str(name),
                    str(details.get("command_type") or ""),
                    bool(details.get("limited", False)),
                    bool(details.get("initialized", False)),
                    str(details.get("reason") or ""),
                )
            )
    return (
        bool(status.get("enabled", False)),
        bool(status.get("limited", False)),
        tuple(effector_statuses),
    )


def _impedance_force_control_issue(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> str | None:
    if _arm_control_mode_arg(args) != "impedance":
        return None
    for side in components.active_sides:
        backend = _command_backend_for_side(components, side)
        if not isinstance(backend, RM65ImpedanceArmBackend):
            raise RuntimeError(f"{side}_arm command backend is not RM65ImpedanceArmBackend.")
        state = backend.get_state()
        fault = state.get("fault")
        if fault:
            return (
                f"{side}_arm pose impedance fault before command dispatch: "
                f"{_short_status_text(fault, max_len=240)}; status={backend.force_control_status_text()}"
            )
        if not bool(state.get("impedance_enabled")):
            return (
                f"{side}_arm pose impedance is disabled before command dispatch: "
                f"{backend.force_control_status_text()}"
            )
        if state.get("impedance_loop_state") == "stopped":
            return (
                f"{side}_arm pose impedance loop stopped before command dispatch: "
                f"{backend.force_control_status_text()}"
            )
    return None


def _hold_motion_after_impedance_guard_issue(
    components: RealTeleopComponents,
    *,
    reason: str,
    now: float,
) -> None:
    rig = components.session.rig
    if rig is None:
        _log(
            "ForceControl",
            f"control fault hold_motion skipped because session has no rig: {_short_status_text(reason)}",
            now=now,
        )
        return
    try:
        rig.hold_motion(f"control fault: {reason}")
    except Exception as exc:  # noqa: BLE001
        _log(
            "ForceControl",
            "control fault hold_motion failed: "
            f"{type(exc).__name__}: {exc}; reason={_short_status_text(reason)}",
            now=now,
        )
    else:
        _log(
            "ForceControl",
            f"control fault hold_motion requested for all rig backends: {_short_status_text(reason)}",
            now=now,
        )


def _guard_impedance_force_control(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    context: str,
    now: float,
) -> bool:
    if context == "main_loop" and components.session.state_machine.state not in {
        TeleopRunState.ARMED,
        TeleopRunState.RUNNING,
        TeleopRunState.RECOVERING,
    }:
        return True
    issue = _impedance_force_control_issue(args, components)
    if issue is None:
        return True
    reason = f"{context}: {issue}"
    if components.session.state_machine.state != TeleopRunState.FAULT:
        _log("ForceControl", reason, now=now)
        _hold_motion_after_impedance_guard_issue(components, reason=reason, now=now)
        components.session.fault(reason)
    return False


def _extract_hand_command_from_command(command: TeleopCommand | None) -> TeleopCommand | None:
    if command is None:
        return None
    hand_command = TeleopCommand(timestamp=command.timestamp, metadata=dict(command.metadata))
    for effector_name in ("left_hand", "right_hand"):
        effector_command = command.commands.get(effector_name)
        if effector_command is not None:
            hand_command.add_command(effector_command.copy())
    if not hand_command.commands:
        return None
    return hand_command


def _build_runtime_hand_command_for_recovery(
    components: RealTeleopComponents,
    observation,
) -> TeleopCommand | None:
    if observation is None:
        return None
    hand_commands: list[TeleopCommand] = []
    if components.left_hand_retargeter is not None:
        hand_commands.append(
            components.left_hand_retargeter.retarget(
                observation,
                components.session.calibration,
            )
        )
    if components.right_hand_retargeter is not None:
        hand_commands.append(
            components.right_hand_retargeter.retarget(
                observation,
                components.session.calibration,
            )
        )
    if not hand_commands:
        return None
    merged = merge_teleop_commands(hand_commands)
    if not merged.commands:
        return None
    return merged


def _merge_runtime_hand_command(
    active_command: TeleopCommand | None,
    hand_command: TeleopCommand | None,
) -> TeleopCommand | None:
    if hand_command is None:
        return active_command
    if active_command is None:
        return None
    return merge_teleop_commands((active_command, hand_command))


def _hand_tactile_sum(backend: InspireTacHandBackend | DMTacGripperBackend | None) -> float | None:
    if backend is None:
        return None
    if isinstance(backend, DMTacGripperBackend):
        state = backend.get_state(include_tactile_vector=False)
    else:
        state = backend.get_state()
    tactile = state.get("tactile")
    if tactile is not None:
        array = np.asarray(tactile, dtype=float).reshape(-1)
        if array.size > 0 and np.all(np.isfinite(array)):
            return float(np.sum(array))
    tactile_dict = state.get("tactile_dict")
    if isinstance(tactile_dict, dict) and tactile_dict:
        total = 0.0
        found = False
        for value in tactile_dict.values():
            array = np.asarray(value, dtype=float).reshape(-1)
            if array.size == 0 or not np.all(np.isfinite(array)):
                continue
            total += float(np.sum(array))
            found = True
        if found:
            return total
    return None


def _emit_tactile_sum_only_status(
    components: RealTeleopComponents,
    *,
    now: float,
    stream,
) -> None:
    timestamp = time.strftime("%H:%M:%S", time.localtime(now))
    left_sum = _hand_tactile_sum(components.left_hand_backend)
    right_sum = _hand_tactile_sum(components.right_hand_backend)
    left_text = "None" if left_sum is None else f"{left_sum:.4f}"
    right_text = "None" if right_sum is None else f"{right_sum:.4f}"
    stream.write(f"[{timestamp}] [TactileSum] left={left_text} right={right_text}\n")
    stream.flush()


def _hand_backend_for_side(
    components: RealTeleopComponents,
    side: str,
) -> InspireTacHandBackend | DMTacGripperBackend | None:
    if side == "left":
        return components.left_hand_backend
    if side == "right":
        return components.right_hand_backend
    return None


def _gripper_backend_for_side(
    components: RealTeleopComponents,
    side: str,
) -> DMTacGripperBackend | None:
    backend = _hand_backend_for_side(components, side)
    return backend if isinstance(backend, DMTacGripperBackend) else None


def _gripper_state_scalar_from_payload(payload: object) -> float | None:
    if not isinstance(payload, Mapping):
        return None
    for key in ("joint_pos", "gripper_measured_scalar", "gripper_state_scalar"):
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    raw_state = payload.get("raw_gripper_state")
    if isinstance(raw_state, Mapping):
        return _gripper_state_scalar_from_payload(raw_state)
    return None


def _read_gripper_backend_scalar_now(backend: object) -> float | None:
    read_now = getattr(backend, "read_gripper_state_now", None)
    if callable(read_now):
        return _gripper_state_scalar_from_payload(read_now())
    get_state = getattr(backend, "get_state", None)
    if not callable(get_state):
        return None
    try:
        payload = get_state(
            include_tactile_arrays=False,
            include_tactile_vector=False,
            copy_tactile_arrays=False,
        )
    except TypeError:
        payload = get_state()
    return _gripper_state_scalar_from_payload(payload)


def _probe_gripper_backend_motion(
    backend: object,
    *,
    targets: Sequence[float] = (0.0, 0.5, 0.0),
    tolerance: float = 0.18,
    step_timeout_s: float = 2.0,
    settle_s: float = 0.2,
    poll_s: float = 0.05,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, object]:
    send = getattr(backend, "send_fixed_joint_pos", None)
    if not callable(send):
        return {"ok": False, "error": "backend_missing_send_fixed_joint_pos", "samples": []}
    samples: list[dict[str, object]] = []
    for index, raw_target in enumerate(targets):
        if cancelled is not None and cancelled():
            return {"ok": False, "error": "cancelled", "step": int(index), "samples": samples}
        target = float(np.clip(float(raw_target), 0.0, 1.0))
        try:
            sent = bool(send(target))
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "error": f"send_failed:{type(exc).__name__}:{exc}",
                "target": target,
                "step": int(index),
                "samples": samples,
            }
        if not sent:
            return {
                "ok": False,
                "error": "send_returned_false",
                "target": target,
                "step": int(index),
                "samples": samples,
            }
        deadline = time.time() + max(0.0, float(step_timeout_s))
        if float(settle_s) > 0.0:
            time.sleep(float(settle_s))
        last_measured: float | None = None
        while True:
            if cancelled is not None and cancelled():
                return {"ok": False, "error": "cancelled", "step": int(index), "samples": samples}
            try:
                measured = _read_gripper_backend_scalar_now(backend)
            except Exception as exc:  # noqa: BLE001
                return {
                    "ok": False,
                    "error": f"read_failed:{type(exc).__name__}:{exc}",
                    "target": target,
                    "step": int(index),
                    "samples": samples,
                }
            last_measured = measured
            sample = {
                "step": int(index),
                "target": target,
                "measured": measured,
                "error_abs": None if measured is None else abs(float(measured) - target),
            }
            samples.append(sample)
            if measured is not None and abs(float(measured) - target) <= float(tolerance):
                break
            if time.time() >= deadline:
                return {
                    "ok": False,
                    "error": "target_not_reached",
                    "target": target,
                    "measured": last_measured,
                    "step": int(index),
                    "tolerance": float(tolerance),
                    "samples": samples,
                }
            time.sleep(max(0.0, float(poll_s)))
    return {"ok": True, "error": None, "samples": samples}


def _run_gripper_warmup_probe_for_side(
    side: str,
    backend: object,
    *,
    targets: Sequence[float],
    tolerance: float,
    step_timeout_s: float,
    settle_s: float,
    poll_s: float,
    reconnect_on_failure: bool = True,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, object]:
    first = _probe_gripper_backend_motion(
        backend,
        targets=targets,
        tolerance=tolerance,
        step_timeout_s=step_timeout_s,
        settle_s=settle_s,
        poll_s=poll_s,
        cancelled=cancelled,
    )
    attempts = [dict(first)]
    if bool(first.get("ok")):
        return {"side": str(side), "ok": True, "attempts": attempts, "reconnected": False}
    reconnected = False
    reconnect_error = None
    if reconnect_on_failure:
        reconnect = getattr(backend, "force_reconnect_gripper", None)
        if callable(reconnect):
            try:
                reconnected = bool(reconnect(f"warmup_probe_failed:{first.get('error')}"))
            except Exception as exc:  # noqa: BLE001
                reconnect_error = f"{type(exc).__name__}: {exc}"
    if reconnected:
        if cancelled is not None and cancelled():
            return {
                "side": str(side),
                "ok": False,
                "attempts": attempts,
                "reconnected": reconnected,
                "reconnect_error": reconnect_error,
                "error": "cancelled",
            }
        second = _probe_gripper_backend_motion(
            backend,
            targets=targets,
            tolerance=tolerance,
            step_timeout_s=step_timeout_s,
            settle_s=settle_s,
            poll_s=poll_s,
            cancelled=cancelled,
        )
        attempts.append(dict(second))
        if bool(second.get("ok")):
            return {"side": str(side), "ok": True, "attempts": attempts, "reconnected": True}
    return {
        "side": str(side),
        "ok": False,
        "attempts": attempts,
        "reconnected": reconnected,
        "reconnect_error": reconnect_error,
    }


def _real_hand_render_qpos(
    backend: InspireTacHandBackend | DMTacGripperBackend | None,
    side: str,
) -> np.ndarray:
    if isinstance(backend, DMTacGripperBackend):
        return _hand_render_qpos(None, side)
    return _hand_render_qpos(backend, side)


def _gripper_retargeter_for_side(
    components: RealTeleopComponents,
    side: str,
) -> ScalarToGripperRetargeter | None:
    if side == "left":
        return components.left_gripper_retargeter
    if side == "right":
        return components.right_gripper_retargeter
    return None


def _has_dmtac_gripper_backend(components: RealTeleopComponents) -> bool:
    return any(_gripper_backend_for_side(components, side) is not None for side in components.active_sides)


def _has_q6_gripper_control(components: RealTeleopComponents) -> bool:
    return any(_gripper_retargeter_for_side(components, side) is not None for side in components.active_sides)


def _gripper_gui_available(components: RealTeleopComponents) -> bool:
    return (
        str(getattr(components, "hand_backend_mode", "") or "") == HAND_BACKEND_DMTAC_GRIPPER
        or _has_dmtac_gripper_backend(components)
        or _has_q6_gripper_control(components)
    )


def _tactile_visualization_available(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> bool:
    if _dmtac_sides(args, components.active_sides):
        for side in components.active_sides:
            backend = _gripper_backend_for_side(components, side)
            tactile_fields = tuple(getattr(backend.config, "tactile_fields", ()) or ()) if backend is not None else ()
            if (
                backend is not None
                and bool(getattr(backend.config, "connect_tactile", False))
                and "distributed_force" in tactile_fields
            ):
                return True
        return False
    return any(_hand_backend_for_side(components, side) is not None for side in components.active_sides)












def _q6_channel_status(
    channel: object | None,
    *,
    side: str,
    now: float,
    channel_name: str | None = None,
) -> dict[str, object]:
    channel_name = channel_name or f"{side}_hand_q6"
    status: dict[str, object] = {
        "q6_channel_present": channel is not None,
        "q6_channel_name": channel_name,
        "q6_valid": False,
        "q6_age_sec": None,
        "q6_values": None,
        "q6_value_shape": None,
        "q6_port": None,
        "q6_baudrate": None,
        "q6_read_error": None,
        "q6_reader_error_count": None,
        "q6_reader_recovery_count": None,
        "q6_stale_frames_dropped": None,
        "q6_serial_read_stats": None,
    }
    if channel is None:
        return status

    status["q6_channel_name"] = str(getattr(channel, "channel_name", channel_name) or channel_name)
    status["q6_channel_source"] = str(getattr(channel, "source_name", "") or "")
    status["q6_valid"] = bool(getattr(channel, "valid", False))
    timestamp = getattr(channel, "timestamp", None)
    if timestamp is not None:
        status["q6_age_sec"] = round(max(0.0, float(now) - float(timestamp)), 3)

    try:
        value_array = np.asarray(getattr(channel, "value"), dtype=float)
        status["q6_value_shape"] = tuple(int(dim) for dim in value_array.shape)
        flat_values = value_array.reshape(-1)
        status["q6_values"] = flat_values.round(4).tolist()
    except Exception as exc:  # noqa: BLE001
        status["q6_value_error"] = f"{type(exc).__name__}: {exc}"

    metadata = getattr(channel, "metadata", {}) or {}
    if isinstance(metadata, Mapping):
        status["q6_port"] = metadata.get("port")
        status["q6_baudrate"] = metadata.get("baudrate")
        status["q6_read_error"] = metadata.get("read_error") or metadata.get("error")
        status["q6_reader_error_count"] = (
            metadata.get("reader_error_count")
            if metadata.get("reader_error_count") is not None
            else metadata.get("error_count")
        )
        status["q6_reader_recovery_count"] = metadata.get("reader_recovery_count")
        status["q6_stale_frames_dropped"] = metadata.get("stale_q6_frames_dropped")
        serial_read_stats = metadata.get("serial_read_stats")
        if isinstance(serial_read_stats, Mapping):
            status["q6_serial_read_stats"] = _json_safe_value(serial_read_stats)
    return status


def _q6_retarget_status(
    components: RealTeleopComponents,
    retargeter: ScalarToGripperRetargeter | None,
    *,
    q6_values: object,
) -> dict[str, object]:
    source_dim = int(components.gripper_policy.get("source_dim", 2))
    output_dim = int(components.gripper_policy.get("output_dim", 1))
    invert = bool(components.gripper_policy.get("invert", False))
    alpha = float(components.gripper_policy.get("alpha", 1.0))
    input_min = float(components.gripper_policy.get("input_min", DEFAULT_GRIPPER_INPUT_MIN))
    input_max = float(components.gripper_policy.get("input_max", DEFAULT_GRIPPER_INPUT_MAX))
    output_min = 0.0
    output_max = float(components.gripper_policy.get("max_action", GRIPPER_ACTION_MAX))
    if retargeter is not None:
        config = getattr(retargeter, "config", None)
        if config is not None:
            configured_source_dim = getattr(config, "source_dim", source_dim)
            source_dim = None if configured_source_dim is None else int(configured_source_dim)
            output_dim = int(getattr(config, "output_dim", output_dim))
            invert = bool(getattr(config, "invert", invert))
            alpha = float(getattr(config, "alpha", alpha))
            input_min = float(getattr(config, "input_min", input_min))
            input_max = float(getattr(config, "input_max", input_max))
            output_min = float(getattr(config, "output_min", output_min))
            output_max = float(getattr(config, "output_max", output_max))

    source_value: float | None = None
    normalized_value: float | None = None
    target_preview: float | None = None
    try:
        q6_array = np.asarray(q6_values, dtype=float).reshape(-1)
        if source_dim is None and q6_array.size == 1:
            raw_source = _safe_float_value(q6_array[0])
        elif source_dim is not None and 0 <= source_dim < q6_array.size:
            raw_source = _safe_float_value(q6_array[source_dim])
        else:
            raw_source = None
        if raw_source is not None:
            source_value = float(np.clip(raw_source, 0.0, 1.0))
            target_preview, normalized_value = map_scalar_to_gripper_target(
                source_value,
                input_min=input_min,
                input_max=input_max,
                invert=invert,
                output_min=output_min,
                output_max=output_max,
            )
    except Exception:
        source_value = None
        normalized_value = None
        target_preview = None

    filtered_target = None
    if retargeter is not None:
        filtered_target = _safe_float_value(getattr(retargeter, "_filtered_value", None))

    return {
        "q6_control_available": retargeter is not None,
        "q6_source_dim": source_dim,
        "q6_output_dim": output_dim,
        "q6_invert": invert,
        "q6_alpha": alpha,
        "q6_input_min": input_min,
        "q6_input_max": input_max,
        "q6_output_min": output_min,
        "q6_output_max": output_max,
        "q6_source_value": source_value,
        "q6_normalized_value": normalized_value,
        "q6_target_preview": target_preview,
        "q6_filtered_target": filtered_target,
    }


def _gripper_status_for_side(
    components: RealTeleopComponents,
    side: str,
    *,
    channels: dict[str, object] | None,
    now: float,
    command_status: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    backend = _gripper_backend_for_side(components, side)
    retargeter = _gripper_retargeter_for_side(components, side)
    input_channel_name = _gripper_input_channel_name(components, side)
    channel = None if channels is None else channels.get(input_channel_name)
    q6_status = _q6_channel_status(channel, side=side, now=now, channel_name=input_channel_name)
    q6_status["input_source"] = str(getattr(components, "gripper_input_source", "q6") or "q6")
    q6_status.update(
        _q6_retarget_status(
            components,
            retargeter,
            q6_values=q6_status.get("q6_values"),
        )
    )
    if command_status is not None:
        q6_status["q6_command"] = _json_safe_value(command_status.get(side) or {})
    if backend is None:
        return {
            "configured": False,
            **q6_status,
        }
    try:
        state = backend.get_state(include_tactile_arrays=False)
    except Exception as exc:  # noqa: BLE001
        return {
            "configured": True,
            "is_connected": False,
            "telemetry_error": f"state_failed: {exc}",
            **q6_status,
        }
    return {
        "configured": True,
        "backend_type": state.get("backend_type"),
        "is_connected": bool(state.get("is_connected")),
        "connect_hardware": bool(state.get("connect_hardware")),
        "connect_tactile": bool(state.get("connect_tactile")),
        "send_commands": bool(state.get("send_commands")),
        "port": state.get("gripper_port"),
        "id": state.get("gripper_id"),
        "command_dim": state.get("gripper_command_dim"),
        "max_action": state.get("gripper_max_action"),
        "speed": state.get("gripper_speed"),
        "torque": state.get("gripper_torque"),
        "command_hz": state.get("gripper_command_hz"),
        "continuous": state.get("gripper_continuous"),
        "commanded": state.get("gripper_commanded_scalar"),
        "measured": state.get("gripper_measured_scalar"),
        "state": state.get("gripper_state_scalar"),
        "last_sent_at": state.get("last_sent_command_timestamp"),
        "telemetry_at": state.get("telemetry_timestamp"),
        "telemetry_error": state.get("telemetry_error"),
        "gripper_read_elapsed_s": state.get("gripper_read_elapsed_s"),
        "tactile_read_elapsed_s": state.get("tactile_read_elapsed_s"),
        "tactile_gate_wait_s": state.get("tactile_gate_wait_s"),
        "tactile_timestamp": state.get("tactile_timestamp"),
        "telemetry_elapsed_s": state.get("telemetry_elapsed_s"),
        "tactile_fields": _json_safe_value(state.get("tactile_fields") or ()),
        "tactile_devices": _json_safe_value(state.get("tactile_devices") or ()),
        "tactile_frame_ids": _json_safe_value(state.get("tactile_frame_ids") or {}),
        "tactile_field_by_key": _json_safe_value(state.get("tactile_field_by_key") or {}),
        "tactile_shapes": _json_safe_value(state.get("tactile_shapes") or {}),
        "tactile_raw_shape_per_sensor": _json_safe_value(
            state.get("tactile_raw_shape_per_sensor") or {}
        ),
        "tactile_timings_ms": _json_safe_value(state.get("tactile_timings_ms") or {}),
        "tactile_read_hz": state.get("tactile_read_hz"),
        "tactile_read_skipped": state.get("tactile_read_skipped"),
        "tactile_process_read": state.get("tactile_process_read"),
        "tactile_worker_alive": state.get("tactile_worker_alive"),
        "tactile_worker_receiver_alive": state.get("tactile_worker_receiver_alive"),
        "tactile_process_nice": state.get("tactile_process_nice"),
        "tactile_process_cpu_affinity": _json_safe_value(
            state.get("tactile_process_cpu_affinity") or ()
        ),
        "tactile_process_queue_size": state.get("tactile_process_queue_size"),
        "tactile_max_fps": state.get("tactile_max_fps"),
        "tactile_warmup_frames": state.get("tactile_warmup_frames"),
        "tactile_denoise": state.get("tactile_denoise"),
        "tactile_store_flat_vector": state.get("tactile_store_flat_vector"),
        "tactile_global_read_lock": state.get("tactile_global_read_lock"),
        "tactile_ready": _json_safe_value(state.get("tactile_ready") or {}),
        "tactile_version": state.get("tactile_version"),
        **q6_status,
    }


def _build_gripper_status(
    components: RealTeleopComponents,
    *,
    channels: dict[str, object] | None = None,
    now: float | None = None,
    command_status: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    current_time = time.time() if now is None else float(now)
    return {
        side: _gripper_status_for_side(
            components,
            side,
            channels=channels,
            now=current_time,
            command_status=command_status,
        )
        for side in ("left", "right")
    }




def _camera_status_for_source(
    camera_key: str,
    source: RealSenseRGBSource,
    *,
    now: float,
    preview_selected: bool,
    preview_active: bool,
    depth_monitor: WristDepthFreshnessMonitor | None = None,
) -> dict[str, object]:
    config = getattr(source, "config", None)
    fallback_serial = getattr(config, "serial", None)
    try:
        info = source.device_info()
    except Exception as exc:  # noqa: BLE001
        return {
            "camera_key": str(camera_key),
            "status_level": "error",
            "status_label": "状态读取失败",
            "status_message": f"{type(exc).__name__}: {exc}",
            "serial": fallback_serial,
            "preview_selected": bool(preview_selected),
            "preview_active": bool(preview_active and preview_selected),
        }
    if not isinstance(info, Mapping):
        info = {}

    frame_count = _safe_int_value(info.get("frame_count"))
    frame_count = 0 if frame_count is None else max(0, int(frame_count))
    timeout_count = _safe_int_value(info.get("timeout_count"))
    consecutive_timeout_count = _safe_int_value(info.get("consecutive_timeout_count"))
    restart_count = _safe_int_value(info.get("restart_count"))
    last_frame_age_sec = _safe_float_value(info.get("last_frame_age_sec"))
    stale_after_sec = _safe_float_value(info.get("stale_after_sec"))
    if stale_after_sec is None or stale_after_sec <= 0.0:
        stale_after_sec = 2.0
    last_error = _as_status_error_text(info.get("last_error"))
    last_runtime_error = _as_status_error_text(info.get("last_runtime_error"))
    depth_enabled = bool(info.get("depth_enabled") or getattr(config, "enable_depth", False))
    depth_required = bool(depth_enabled and str(camera_key) in WRIST_DEPTH_CAMERA_KEYS)
    depth_scale = _safe_float_value(info.get("depth_scale_m_per_unit"))
    depth_frame_present: bool | None = None
    depth_status_level = "off"
    depth_status_label = "未启用"
    depth_status_message = ""
    latest_depth_metadata: Mapping[str, object] = {}
    depth_freshness: dict[str, object] = {}
    if depth_required:
        latest_frame = None
        try:
            latest_frame = source.frame_buffer().get_latest()
        except Exception as exc:  # noqa: BLE001
            depth_status_level = "error"
            depth_status_label = "Depth 状态失败"
            depth_status_message = f"{type(exc).__name__}: {exc}"
        else:
            if latest_frame is None:
                depth_status_level = "waiting"
                depth_status_label = "等待 Wrist Depth"
                depth_status_message = "wrist depth 已启用，但还没有 color+depth 同步帧。"
            else:
                raw_depth_metadata = getattr(latest_frame, "depth_metadata", {})
                if isinstance(raw_depth_metadata, Mapping):
                    latest_depth_metadata = raw_depth_metadata
                depth_frame_present = getattr(latest_frame, "depth_image", None) is not None
                frame_depth_scale = _safe_float_value(latest_depth_metadata.get("depth_scale_m_per_unit"))
                if depth_scale is None and frame_depth_scale is not None:
                    depth_scale = frame_depth_scale
                if not depth_frame_present:
                    depth_status_level = "error"
                    depth_status_label = "Wrist Depth 失效"
                    depth_status_message = "最新 wrist 相机帧缺少 depth_image。"
                elif depth_scale is None and frame_depth_scale is None:
                    depth_status_level = "warn"
                    depth_status_label = "Depth scale 缺失"
                    depth_status_message = "depth frame 存在，但缺少有效 depth_scale_m_per_unit，mask 不能可靠生成。"
                else:
                    if depth_monitor is not None:
                        depth_freshness = depth_monitor.update(
                            str(camera_key),
                            latest_frame,
                            now=float(now),
                            depth_metadata=latest_depth_metadata,
                        )
                    freshness_level = str(depth_freshness.get("depth_freshness_status_level") or "ok")
                    freshness_label = str(depth_freshness.get("depth_freshness_status_label") or "")
                    freshness_message = str(depth_freshness.get("depth_freshness_status_message") or "")
                    if freshness_level in {"error", "warn", "waiting"}:
                        depth_status_level = freshness_level
                        depth_status_label = freshness_label or "Wrist Depth 检查中"
                        depth_status_message = freshness_message
                    else:
                        depth_status_level = "ok"
                        depth_status_label = "Wrist Depth 正常"

    status_level = "ok"
    status_label = "正常"
    status_message = ""
    if last_error:
        status_level = "error"
        status_label = "异常"
        status_message = last_error
    elif frame_count <= 0:
        status_level = "waiting"
        status_label = "等待首帧"
        status_message = last_runtime_error
    elif last_frame_age_sec is None:
        status_level = "warn"
        status_label = "缺少时间戳"
    elif last_frame_age_sec >= stale_after_sec:
        status_level = "error"
        status_label = "数据停滞"
        status_message = f"last frame age {last_frame_age_sec:.2f}s >= stale {stale_after_sec:.2f}s"
    elif last_frame_age_sec >= max(0.5, stale_after_sec * 0.5):
        status_level = "warn"
        status_label = "帧延迟"
        status_message = f"last frame age {last_frame_age_sec:.2f}s"
    elif (consecutive_timeout_count or 0) > 0 or last_runtime_error:
        status_level = "warn"
        status_label = "临时 timeout"
        status_message = last_runtime_error
    if depth_required:
        severity_rank = {"off": 0, "ok": 0, "waiting": 1, "warn": 2, "error": 3}
        if severity_rank.get(depth_status_level, 0) > severity_rank.get(status_level, 0):
            status_level = depth_status_level
            status_label = depth_status_label
            status_message = depth_status_message
        elif depth_status_message and depth_status_message not in status_message:
            status_message = (
                f"{status_message}; {depth_status_message}"
                if status_message
                else depth_status_message
            )

    metadata = info.get("metadata") if isinstance(info.get("metadata"), Mapping) else {}
    return {
        "camera_key": str(camera_key),
        "status_level": status_level,
        "status_label": status_label,
        "status_message": status_message,
        "name": info.get("name") or getattr(source, "name", str(camera_key)),
        "serial": info.get("serial") or fallback_serial,
        "serial_number": info.get("serial_number"),
        "device_name": info.get("device_name"),
        "product_line": info.get("product_line"),
        "firmware_version": info.get("firmware_version"),
        "width": info.get("width") if info.get("width") is not None else getattr(config, "width", None),
        "height": info.get("height") if info.get("height") is not None else getattr(config, "height", None),
        "fps": info.get("fps") if info.get("fps") is not None else getattr(config, "fps", None),
        "color_space": info.get("color_space") or getattr(config, "color_space", None),
        "depth_enabled": depth_enabled,
        "depth_required": depth_required,
        "depth_width": info.get("depth_width") if info.get("depth_width") is not None else getattr(config, "depth_width", None),
        "depth_height": info.get("depth_height") if info.get("depth_height") is not None else getattr(config, "depth_height", None),
        "depth_fps": info.get("depth_fps") if info.get("depth_fps") is not None else getattr(config, "depth_fps", None),
        "depth_scale_m_per_unit": depth_scale,
        "depth_frame_present": depth_frame_present,
        "depth_status_level": depth_status_level,
        "depth_status_label": depth_status_label,
        "depth_status_message": depth_status_message,
        "depth_metadata": _json_safe_value(dict(latest_depth_metadata)),
        "depth_frame_number": latest_depth_metadata.get("frame_number"),
        "depth_camera_timestamp_sec": latest_depth_metadata.get("camera_timestamp_sec"),
        "depth_sensor_timestamp_ms": latest_depth_metadata.get("sensor_timestamp_ms"),
        **depth_freshness,
        "frame_count": frame_count,
        "last_frame_age_sec": last_frame_age_sec,
        "stale_after_sec": stale_after_sec,
        "timeout_count": timeout_count,
        "consecutive_timeout_count": consecutive_timeout_count,
        "restart_count": restart_count,
        "last_error": last_error,
        "last_runtime_error": last_runtime_error,
        "last_restart_timestamp": info.get("last_restart_timestamp"),
        "preview_selected": bool(preview_selected),
        "preview_active": bool(preview_active and preview_selected),
        "metadata": _json_safe_value(metadata),
        "updated_at": float(now),
    }


def _build_camera_status(
    camera_sources: Mapping[str, RealSenseRGBSource] | None,
    *,
    preview_camera_sources: Mapping[str, RealSenseRGBSource] | None = None,
    camera_previewer: CameraPreviewer | None = None,
    now: float | None = None,
    depth_monitor: WristDepthFreshnessMonitor | None = None,
) -> dict[str, object]:
    if not camera_sources:
        return {}
    current_time = time.time() if now is None else float(now)
    preview_keys = set(preview_camera_sources or {})
    preview_active = camera_previewer is not None
    return {
        str(camera_key): _camera_status_for_source(
            str(camera_key),
            source,
            now=current_time,
            preview_selected=str(camera_key) in preview_keys,
            preview_active=preview_active,
            depth_monitor=depth_monitor,
        )
        for camera_key, source in sorted(camera_sources.items(), key=lambda item: str(item[0]))
    }


def _wrist_depth_requested(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "wrist_depth_mask", False)) or bool(
        getattr(args, "wrist_depth_raw", False)
    )


def _with_expected_wrist_depth_camera_statuses(
    camera_status: Mapping[str, object],
    *,
    args: argparse.Namespace,
    now: float,
) -> dict[str, object]:
    merged = dict(camera_status)
    if not _wrist_depth_requested(args):
        return merged
    for camera_key in WRIST_DEPTH_CAMERA_KEYS:
        if camera_key in merged:
            continue
        merged[camera_key] = {
            "camera_key": camera_key,
            "status_level": "error",
            "status_label": "Wrist 相机缺失",
            "status_message": (
                f"--wrist-depth-mask/--wrist-depth-raw requested but {camera_key} "
                "is absent from camera_sources/config."
            ),
            "name": camera_key,
            "serial": None,
            "depth_enabled": True,
            "depth_required": True,
            "depth_frame_present": False,
            "depth_status_level": "error",
            "depth_status_label": "Wrist Depth 缺失",
            "depth_status_message": f"{camera_key} 没有可用 RealSense source，无法生成 wrist depth mask。",
            "frame_count": 0,
            "last_frame_age_sec": None,
            "stale_after_sec": None,
            "timeout_count": None,
            "consecutive_timeout_count": None,
            "restart_count": None,
            "preview_selected": False,
            "preview_active": False,
            "updated_at": float(now),
        }
    return merged
















def _dispatch_q6_gripper_control(
    components: RealTeleopComponents,
    observation,
    *,
    now: float,
    status_out: dict[str, dict[str, object]] | None = None,
) -> TeleopCommand | None:
    if observation is None:
        if status_out is not None:
            for side in components.active_sides:
                status_out[side] = {
                    "status": "no_observation",
                    "sent": False,
                    "timestamp": float(now),
                }
        return None
    commands: list[TeleopCommand] = []

    def backend_send_fields(backend: DMTacGripperBackend) -> dict[str, object]:
        elapsed_s = _safe_float_value(getattr(backend, "_last_send_elapsed_s", None))
        lock_wait_s = _safe_float_value(getattr(backend, "_last_send_lock_wait_s", None))
        return {
            "backend_status": str(getattr(backend, "_last_send_status", "") or ""),
            "skip_reason": str(getattr(backend, "_last_send_skip_reason", "") or ""),
            "send_elapsed_s": elapsed_s,
            "send_elapsed_ms": None if elapsed_s is None else float(elapsed_s) * 1000.0,
            "send_lock_wait_s": lock_wait_s,
            "send_lock_wait_ms": None if lock_wait_s is None else float(lock_wait_s) * 1000.0,
            "send_target": _safe_float_value(getattr(backend, "_last_send_target_scalar", None)),
        }

    for side in components.active_sides:
        side_status: dict[str, object] = {
            "status": "pending",
            "sent": False,
            "timestamp": float(now),
        }

        def finish(status: str, **fields: object) -> None:
            side_status["status"] = status
            side_status.update(fields)
            if status_out is not None:
                status_out[side] = dict(side_status)

        backend = _gripper_backend_for_side(components, side)
        retargeter = _gripper_retargeter_for_side(components, side)
        if backend is None:
            finish("no_backend")
            continue
        if retargeter is None:
            finish("no_retargeter")
            continue
        fix_arm_controller = getattr(components, "fix_arm_controller", None)
        fixed_side_active = bool(
            fix_arm_controller is not None
            and fix_arm_controller.fixed
            and fix_arm_controller.fixed_side == side
        )
        if fixed_side_active:
            command = TeleopCommand(timestamp=float(now))
            command = fix_arm_controller.filter_command(
                command,
                observation,
                now,
                include_missing_fixed_effectors=True,
                allowed_effectors=(f"{side}_hand",),
            )
        else:
            try:
                command = retargeter.retarget(observation, components.session.calibration)
            except Exception as exc:  # noqa: BLE001
                finish("retarget_exception", error=f"{type(exc).__name__}: {exc}")
                _log("Gripper", f"{side} Q6 retarget failed: {exc}", now=now)
                continue
            if fix_arm_controller is not None:
                command = fix_arm_controller.filter_command(command, observation, now)
        if not command.commands:
            finish(
                "no_retarget_command",
                skipped=_json_safe_value(command.metadata.get("skipped")),
            )
            continue
        effector_command = command.commands.get(f"{side}_hand")
        if effector_command is None:
            finish(
                "missing_effector_command",
                effector_names=tuple(str(name) for name in command.commands),
            )
            continue
        metadata = effector_command.metadata or {}
        if fixed_side_active:
            finish(
                "fixed_command",
                target=_safe_float_value(metadata.get("gripper_joint_target")),
            )
        else:
            finish(
                "retargeted",
                source_dim=_safe_int_value(metadata.get("source_dim")),
                output_dim=_safe_int_value(metadata.get("output_dim")),
                source_value=_safe_float_value(metadata.get("source_value")),
                target=_safe_float_value(metadata.get("gripper_joint_target")),
            )
        try:
            sent_ok = bool(backend.send_command(effector_command))
        except Exception as exc:  # noqa: BLE001
            finish("send_exception", error=f"{type(exc).__name__}: {exc}")
            _log("Gripper", f"{side} Q6 command failed: {exc}", now=now)
            continue
        commands.append(command)
        if not sent_ok:
            fields = backend_send_fields(backend)
            backend_status = str(fields.get("backend_status") or "send_returned_false")
            finish(backend_status, **fields)
            continue
        finish("sent", sent=True, **backend_send_fields(backend))
    if not commands:
        return None
    return merge_teleop_commands(commands)


def _hand_channel_status(
    components: RealTeleopComponents,
    channels: dict[str, object],
) -> tuple[str, ...]:
    statuses: list[str] = []
    for side in components.active_sides:
        channel = channels.get(_gripper_input_channel_name(components, side))
        if channel is None:
            continue
        valid = bool(getattr(channel, "valid", False))
        if valid:
            statuses.append(f"{side}=ok")
            continue
        metadata = getattr(channel, "metadata", {}) or {}
        read_error = str(metadata.get("read_error") or "invalid")
        statuses.append(f"{side}=invalid:{read_error}")
    return tuple(statuses)


def _gripper_status_signature(components: RealTeleopComponents) -> tuple[str, ...]:
    return _gripper_runtime_status_signature(components)


def _gripper_runtime_status_signature(
    components: RealTeleopComponents,
    *,
    channels: Mapping[str, object] | None = None,
    command_status: Mapping[str, Mapping[str, object]] | None = None,
    now: float | None = None,
) -> tuple[str, ...]:
    statuses: list[str] = []
    current_time = time.time() if now is None else float(now)
    for side in components.active_sides:
        backend = _gripper_backend_for_side(components, side)
        retargeter = _gripper_retargeter_for_side(components, side)
        channel_name = _gripper_input_channel_name(components, side)
        channel = None if channels is None else channels.get(channel_name)
        q6_status = _q6_channel_status(channel, side=side, now=current_time, channel_name=channel_name)
        q6_status.update(
            _q6_retarget_status(
                components,
                retargeter,
                q6_values=q6_status.get("q6_values"),
            )
        )
        cmd_status = dict(command_status.get(side) or {}) if isinstance(command_status, Mapping) else {}
        cmd_text = str(cmd_status.get("status") or "-")
        sent_text = _compact_runtime_bool(cmd_status.get("sent", False))
        cmd_target = _compact_runtime_float(cmd_status.get("target"))
        cmd_source = _compact_runtime_float(cmd_status.get("source_value"))
        cmd_send_ms = _compact_runtime_float(cmd_status.get("send_elapsed_ms"), digits=1)
        cmd_skip = str(cmd_status.get("skip_reason") or "")
        q6_valid = "ok" if bool(q6_status.get("q6_valid")) else (
            "missing" if not bool(q6_status.get("q6_channel_present")) else "invalid"
        )
        q6_source_dim = q6_status.get("q6_source_dim")
        q6_source = _compact_runtime_float(q6_status.get("q6_source_value"))
        q6_target = _compact_runtime_float(q6_status.get("q6_target_preview"))
        q6_filtered = _compact_runtime_float(q6_status.get("q6_filtered_target"))
        q6_values = q6_status.get("q6_values")
        q6_values_text = "-"
        if isinstance(q6_values, Sequence) and not isinstance(q6_values, (str, bytes)):
            q6_values_text = "[" + ",".join(_compact_runtime_float(value, digits=2) for value in q6_values) + "]"
        q6_error = q6_status.get("q6_read_error") or q6_status.get("q6_value_error") or ""

        prefix = f"{side}=no_backend"
        measured_text = "-"
        commanded_text = "-"
        suffix = ""
        try:
            state = backend.get_state(include_tactile_arrays=False) if backend is not None else {}
        except Exception as exc:  # noqa: BLE001
            prefix = f"{side}=state_error:{exc}"
        else:
            if backend is not None:
                if cmd_send_ms == "-":
                    send_elapsed_s = _safe_float_value(state.get("gripper_send_elapsed_s"))
                    if send_elapsed_s is not None:
                        cmd_send_ms = _compact_runtime_float(float(send_elapsed_s) * 1000.0, digits=1)
                if not cmd_skip:
                    cmd_skip = str(state.get("gripper_last_send_skip_reason") or "")
                connected = "connected" if bool(state.get("is_connected")) else "disconnected"
                error = state.get("telemetry_error")
                measured = state.get("gripper_measured_scalar")
                commanded = state.get("gripper_commanded_scalar")
                measured_text = _compact_runtime_float(measured)
                commanded_text = _compact_runtime_float(commanded)
                q6_mode = str(getattr(components, "gripper_input_source", "q6")) if retargeter is not None else "manual"
                suffix = f":{error}" if error else ""
                prefix = f"{side}={connected}/{q6_mode}/cmd={commanded_text}/meas={measured_text}{suffix}"
        statuses.append(
            f"{prefix} input={getattr(components, 'gripper_input_source', 'q6')} valid={q6_valid} dim={q6_source_dim} src={q6_source} "
            f"target={q6_target} filt={q6_filtered} vals={q6_values_text} "
            f"send={cmd_text}/{sent_text} send_src={cmd_source} send_target={cmd_target} "
            f"send_ms={cmd_send_ms}"
            + (f" skip={cmd_skip}" if cmd_skip else "")
            + (f" q6_error={q6_error}" if q6_error else "")
        )
    return tuple(statuses)


def _gripper_runtime_health_signature(statuses: Sequence[str]) -> tuple[str, ...]:
    """Return only the stable health fields used to decide whether to print status.

    The full status intentionally contains live q6, target, measured, and send-time
    values.  Including those values in the change gate turns ordinary hand motion
    into a terminal log on every control frame.
    """

    health_statuses: list[str] = []
    for status in statuses:
        text = str(status)
        prefix, _, remainder = text.partition(" input=")
        route = "/".join(prefix.split("/")[:2])
        valid = "unknown"
        for token in remainder.split():
            if token.startswith("valid="):
                valid = token.removeprefix("valid=")
                break
        lower_text = text.lower()
        unhealthy = any(
            marker in lower_text
            for marker in (
                "disconnected",
                "no_backend",
                "state_error",
                "invalid",
                "error",
                "failed",
            )
        )
        health_statuses.append(f"{route} valid={valid} unhealthy={int(unhealthy)}")
    return tuple(health_statuses)


_DEBUG_TIMING_RED = "\033[1;31;40m"
_DEBUG_TIMING_RESET = "\033[0m"








def _build_pose_quality_display_payload(active_command: TeleopCommand | None) -> dict[str, object]:
    quality = _collection_build_pose_quality_metadata(active_command)
    if not isinstance(quality, Mapping):
        quality = {}
    metrics = quality.get("metrics")
    metrics_dict = dict(metrics) if isinstance(metrics, Mapping) else {}
    pose_error = metrics_dict.get("ik_solve_pose_error")
    pose_error_dict = dict(pose_error) if isinstance(pose_error, Mapping) else {}
    joint_margin = metrics_dict.get("joint_limit_margin")
    joint_margin_dict = dict(joint_margin) if isinstance(joint_margin, Mapping) else {}
    available = bool(quality.get("available", False))
    status = str(quality.get("status") or "unknown")
    bad = bool(quality.get("bad", False))
    score = _float_or_none(quality.get("score"))
    return {
        "available": available,
        "status": status,
        "score": score,
        "bad": bad,
        "level": _pose_quality_level(status, available=available, bad=bad),
        "source": str(quality.get("source") or ""),
        "reasons": _pose_quality_reason_list(quality.get("reasons")),
        "bad_reasons": _pose_quality_reason_list(quality.get("bad_reasons")),
        "warning_reasons": _pose_quality_reason_list(quality.get("warning_reasons")),
        "metrics": {
            "collision_avoidance_status": metrics_dict.get("collision_avoidance_status"),
            "collision_distance_m": _float_or_none(metrics_dict.get("collision_distance_m")),
            "collision_min_distance_m": _float_or_none(metrics_dict.get("collision_min_distance_m")),
            "collision_margin_m": _float_or_none(metrics_dict.get("collision_margin_m")),
            "ik_jump_gate_status": metrics_dict.get("ik_jump_gate_status"),
            "ik_branch_candidate": metrics_dict.get("ik_branch_candidate"),
            "ik_translation_error_m": _float_or_none(pose_error_dict.get("max_translation_m")),
            "ik_rotation_error_deg": _float_or_none(pose_error_dict.get("max_rotation_deg")),
            "joint_nearest_margin_deg": _float_or_none(joint_margin_dict.get("nearest_margin_deg")),
            "joint_near_limit": bool(joint_margin_dict.get("near_limit", False)),
            "execution_step_limited": bool(metrics_dict.get("execution_step_limited", False)),
            "ik_time_budget_exceeded": bool(metrics_dict.get("ik_time_budget_exceeded", False)),
        },
    }




































def _debug_timing_report(
    state: RealDebugTimingState,
    *,
    reason: str,
    elapsed_sec: float,
    phases: Sequence[tuple[str, float]],
    step_index: int,
    result,
    active_command,
    left_hw_state: dict[str, object],
    right_hw_state: dict[str, object],
    left_command_state: dict[str, object],
    right_command_state: dict[str, object],
    target_step_scale: float,
    force: bool = False,
) -> None:
    if not state.enabled:
        return
    now = time.time()
    report_interval = max(0.0, float(state.report_interval_sec))
    if (
        not force
        and report_interval > 0.0
        and state.last_report_at > 0.0
        and (now - state.last_report_at) < report_interval
    ):
        state.suppressed_reports += 1
        return
    suppressed = int(state.suppressed_reports)
    state.suppressed_reports = 0
    state.last_report_at = now
    prefix = (
        f"[RealDebugTiming] slow reason={reason} elapsed={elapsed_sec * 1000.0:.1f}ms "
        f"active_hz={(1.0 / elapsed_sec) if elapsed_sec > 0.0 else 0.0:.2f} "
        f"threshold={float(state.threshold_sec) * 1000.0:.1f}ms "
        f"step={int(step_index)}"
    )
    if result is not None:
        prefix += f" control_hz={float(result.control_hz):.2f}"
    if suppressed:
        prefix += f" suppressed={suppressed}"
    lines = [
        prefix,
        "  " + format_phase_timings(phases, limit=14),
        "  "
        + format_end_to_end_latency_breakdown(
            build_end_to_end_latency_breakdown(result, reference_time=now),
            limit=6,
        ),
        "  " + _debug_timing_ik_text(active_command),
        "  "
        + _debug_timing_arm_text(
            left_hw_state=left_hw_state,
            right_hw_state=right_hw_state,
            left_command_state=left_command_state,
            right_command_state=right_command_state,
            target_step_scale=target_step_scale,
        ),
        "  " + _debug_timing_fault_text(result),
    ]
    print(_DEBUG_TIMING_RED + "\n".join(lines) + _DEBUG_TIMING_RESET, flush=True)


def _debug_timing_maybe_report(
    state: RealDebugTimingState,
    *,
    reason: str,
    elapsed_sec: float,
    phases: Sequence[tuple[str, float]],
    step_index: int,
    result,
    active_command,
    left_hw_state: dict[str, object],
    right_hw_state: dict[str, object],
    left_command_state: dict[str, object],
    right_command_state: dict[str, object],
    target_step_scale: float,
    force: bool = False,
) -> None:
    if not state.enabled:
        return
    if should_report_debug_frozen(
        elapsed_sec=elapsed_sec,
        phases=phases,
        threshold_sec=float(state.threshold_sec),
        force=force,
    ):
        _debug_timing_report(
            state,
            reason=reason,
            elapsed_sec=elapsed_sec,
            phases=phases,
            step_index=step_index,
            result=result,
            active_command=active_command,
            left_hw_state=left_hw_state,
            right_hw_state=right_hw_state,
            left_command_state=left_command_state,
            right_command_state=right_command_state,
            target_step_scale=target_step_scale,
            force=force,
        )


def _build_tactile_visualizer(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
    *,
    force: bool = False,
) -> TactileUVVisualizer | AsyncTactileUVVisualizer | DMTacDistributedForceVisualizer | None:
    if not (bool(args.visualize_tactile) or bool(force)):
        return None
    if _dmtac_sides(args, active_sides):
        return DMTacDistributedForceVisualizer(
            DMTacForceVisualizerConfig(
                enabled_sides=active_sides,
                dmtac_root=DEFAULT_DMTAC_ROOT,
            )
        )
    config = TactileVisualizerConfig(
        enabled_sides=active_sides,
        image_size=(int(args.tactile_vis_size), int(args.tactile_vis_size)),
        value_scale=float(args.tactile_vis_value_scale),
    )
    if not bool(args.skip_renderer):
        return AsyncTactileUVVisualizer(config)
    return TactileUVVisualizer(config)
















def _command_for_side(active_command: TeleopCommand | None, side: str):
    if active_command is None:
        return None
    return active_command.commands.get(f"{side}_arm")


def _build_ik_debug_log_frame(
    *,
    components: RealTeleopComponents,
    step_index: int,
    loop_start: float,
    step_result,
    active_command: TeleopCommand | None,
    left_command_state: Mapping[str, object] | None,
    right_command_state: Mapping[str, object] | None,
    left_hw_state: Mapping[str, object] | None,
    right_hw_state: Mapping[str, object] | None,
    ik_timing_payload: Mapping[str, object] | None,
    target_step_scale: float,
    left_clutch: bool,
    right_clutch: bool,
    full: bool = False,
) -> dict[str, object]:
    fault_codes = [
        f"{fault.code}:{fault.channel_name}" if fault.channel_name else fault.code
        for fault in getattr(step_result, "faults", ())
    ]
    base = {
        "schema": "dexteleop_ik_debug_v1",
        "event": "frame",
        "step_index": int(step_index),
        "timestamp": float(loop_start),
        "state": getattr(getattr(step_result, "state", None), "value", str(getattr(step_result, "state", ""))),
        "control_hz": float(getattr(step_result, "control_hz", 0.0) or 0.0),
        "faults": fault_codes,
        "clutch": {"left": bool(left_clutch), "right": bool(right_clutch)},
        "fix_arm": _fix_arm_debug_state(components),
        "init_arm": _init_arm_debug_state(components),
        "target_step_scale": float(target_step_scale),
        "ik_timing": dict(ik_timing_payload or {}),
    }
    if full:
        return {
            **base,
            "log_mode": "full",
            "retargeter_command": {
                "metadata": _teleop_command_metadata_debug(active_command),
                "left": _command_debug_payload(_command_for_side(active_command, "left")),
                "right": _command_debug_payload(_command_for_side(active_command, "right")),
            },
            "backend": {
                "left_command": _backend_debug_payload(left_command_state),
                "right_command": _backend_debug_payload(right_command_state),
                "left_hardware": _backend_debug_payload(left_hw_state),
                "right_hardware": _backend_debug_payload(right_hw_state),
            },
            "preview_qpos": {
                "left": _qpos_debug(_state_qpos(components.left_preview_backend, DEFAULT_LEFT_ARM_QPOS)),
                "right": _qpos_debug(_state_qpos(components.right_preview_backend, DEFAULT_RIGHT_ARM_QPOS)),
            },
            "observation": _observation_pose_debug(step_result),
        }
    return {
        **base,
        "log_mode": "compact",
        "retargeter_command": {
            "metadata": _teleop_command_metadata_compact_debug(active_command),
            "left": _command_compact_debug_payload(_command_for_side(active_command, "left")),
            "right": _command_compact_debug_payload(_command_for_side(active_command, "right")),
        },
        "backend": {
            "left_command": _backend_compact_debug_payload(left_command_state),
            "right_command": _backend_compact_debug_payload(right_command_state),
            "left_hardware": _backend_compact_debug_payload(left_hw_state),
            "right_hardware": _backend_compact_debug_payload(right_hw_state),
        },
        "preview_qpos": {
            "left": _qpos_compact_debug(_state_qpos(components.left_preview_backend, DEFAULT_LEFT_ARM_QPOS)),
            "right": _qpos_compact_debug(_state_qpos(components.right_preview_backend, DEFAULT_RIGHT_ARM_QPOS)),
        },
        "observation": _observation_pose_compact_debug(step_result),
    }














def _camera_status_log_payload(
    camera_sources: Mapping[str, RealSenseRGBSource] | None,
    *,
    preview_camera_sources: Mapping[str, RealSenseRGBSource] | None,
    camera_previewer: CameraPreviewer | None,
    now: float,
) -> dict[str, object]:
    statuses = _build_camera_status(
        camera_sources,
        preview_camera_sources=preview_camera_sources,
        camera_previewer=camera_previewer,
        now=now,
    )
    selected_keys = (
        "status_level",
        "status_label",
        "status_message",
        "serial",
        "width",
        "height",
        "fps",
        "frame_count",
        "last_frame_age_sec",
        "stale_after_sec",
        "timeout_count",
        "consecutive_timeout_count",
        "restart_count",
        "last_error",
        "last_runtime_error",
        "preview_active",
        "depth_required",
        "depth_status_level",
        "depth_status_label",
        "depth_status_message",
        "depth_frame_number",
        "depth_camera_timestamp_sec",
        "depth_sensor_timestamp_ms",
        "depth_signature_unique_count",
        "depth_signature_repeat_count",
        "depth_timestamp_age_sec",
    )
    return {
        str(camera_key): {
            key: _compact_debug_value(status.get(key), max_items=8)
            for key in selected_keys
            if key in status
        }
        for camera_key, status in statuses.items()
        if isinstance(status, Mapping)
    }


def _backend_timing_log_payload(state: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(state, Mapping):
        return {}
    selected_keys = (
        "hardware_result_code",
        "hardware_result_text",
        "send_result_code",
        "send_result_text",
        "effective_control_dt_sec",
        "effective_step_scale",
        "impedance_enabled",
        "impedance_loop_state",
        "active_tool_name",
        "active_tool_profile_path",
        "active_tool_frame",
        "profile_tcp",
        "last_command_controlled_frame",
        "latest_target_tcp_pose6",
        "latest_target_joint_deg",
        "fault",
    )
    out: dict[str, object] = {
        key: _compact_debug_value(state.get(key), max_items=12)
        for key in selected_keys
        if key in state and state.get(key) is not None
    }
    if state.get("last_send_command_timing") is not None:
        out["last_send_command_timing"] = _compact_debug_value(
            state.get("last_send_command_timing"),
            max_items=24,
        )
    if state.get("last_impedance_loop_timing") is not None:
        out["last_impedance_loop_timing"] = _compact_debug_value(
            state.get("last_impedance_loop_timing"),
            max_items=24,
        )
    if state.get("active_impedance_config") is not None:
        out["active_impedance_config"] = _compact_debug_value(
            state.get("active_impedance_config"),
            max_items=80,
        )
    impedance_record = _impedance_record_compact_debug(state.get("last_impedance_record"))
    if impedance_record:
        out["last_impedance_record"] = impedance_record
    recent_records = [
        item
        for item in (
            _impedance_record_compact_debug(record)
            for record in list(state.get("recent_impedance_records") or [])[-4:]
        )
        if item
    ]
    if recent_records:
        out["recent_impedance_records"] = recent_records
    ik_metadata = _backend_ik_metadata_compact_debug(state.get("ik_metadata"))
    if ik_metadata:
        out["ik_metadata"] = ik_metadata
    return out


def _build_debug_timing_log_frame(
    *,
    components: RealTeleopComponents,
    step_index: int,
    loop_start: float,
    elapsed_sec: float,
    phases: Sequence[tuple[str, float]],
    reason: str,
    step_result: object,
    active_command: TeleopCommand | None,
    left_command_state: Mapping[str, object] | None,
    right_command_state: Mapping[str, object] | None,
    left_hw_state: Mapping[str, object] | None,
    right_hw_state: Mapping[str, object] | None,
    ik_timing_payload: Mapping[str, object] | None,
    target_step_scale: float,
    left_clutch: bool,
    right_clutch: bool,
    camera_sources: Mapping[str, RealSenseRGBSource] | None,
    preview_camera_sources: Mapping[str, RealSenseRGBSource] | None,
    camera_previewer: CameraPreviewer | None,
    hand_input_status: object,
    gripper_status: object,
    recording_status: Mapping[str, object] | None = None,
) -> dict[str, object]:
    faults = tuple(getattr(step_result, "faults", ()) or ())
    return {
        "schema": "dexteleop_debug_timing_v1",
        "event": "frame",
        "reason": str(reason),
        "step_index": int(step_index),
        "timestamp": float(loop_start),
        "loop_elapsed_ms": float(elapsed_sec) * 1000.0,
        "control_hz": float(getattr(step_result, "control_hz", 0.0) or 0.0),
        "state": _state_value(getattr(step_result, "state", "")),
        "session_fault_reason": _short_status_text(
            getattr(getattr(components, "session", None), "last_fault_reason", ""),
            max_len=400,
        ),
        "fault_codes": list(_fault_signature(faults)),
        "faults": _faults_log_payload(step_result),
        "clutch": {"left": bool(left_clutch), "right": bool(right_clutch)},
        "fix_arm": _fix_arm_debug_state(components),
        "init_arm": _init_arm_debug_state(components),
        "target_step_scale": float(target_step_scale),
        "recording": _compact_debug_value(dict(recording_status or {}), max_items=16),
        "phases": _phase_timings_log_payload(phases),
        "latency": _latency_log_payload(step_result, now=loop_start),
        "channels": _channels_log_payload(step_result, now=loop_start),
        "camera_status": _camera_status_log_payload(
            camera_sources,
            preview_camera_sources=preview_camera_sources,
            camera_previewer=camera_previewer,
            now=loop_start,
        ),
        "ik_timing": _compact_debug_value(dict(ik_timing_payload or {}), max_items=24),
        "retargeter_command": {
            "metadata": _teleop_command_metadata_compact_debug(active_command),
        },
        "backend": {
            "left_command": _backend_timing_log_payload(left_command_state),
            "right_command": _backend_timing_log_payload(right_command_state),
            "left_hardware": _backend_timing_log_payload(left_hw_state),
            "right_hardware": _backend_timing_log_payload(right_hw_state),
        },
        "hand_input_status": _compact_debug_value(hand_input_status, max_items=16),
        "gripper_status": _compact_debug_value(gripper_status, max_items=16),
        "active_sides": list(getattr(components, "active_sides", ()) or ()),
    }


# MIGRATION(real-teleop, batch 6): route legacy recorder construction through
# the extracted component.  These wrappers intentionally preserve the legacy
# module-level default-root patch point used by tests and local tooling; no
# recorder behavior is implemented here.
from ..real_teleop.debug_recorders import (
    ActionDebugLogRecorder as _ExtractedActionDebugLogRecorder,
    DebugTimingLogRecorder as _ExtractedDebugTimingLogRecorder,
    IkDebugLogRecorder as _ExtractedIkDebugLogRecorder,
    json_safe as _json_safe,
    state_value as _state_value,
    stop_debug_writer_thread as _stop_debug_writer_thread,
)


class DebugTimingLogRecorder(_ExtractedDebugTimingLogRecorder):
    def __init__(
        self,
        *,
        root_dir: Path | str | None = None,
        max_queue_size: int = 512,
        heartbeat_sec: float = 1.0,
        sample_hz: float = 0.0,
    ) -> None:
        super().__init__(
            root_dir=DEFAULT_DEBUG_TIMING_LOG_ROOT if root_dir is None else root_dir,
            max_queue_size=max_queue_size,
            heartbeat_sec=heartbeat_sec,
            sample_hz=sample_hz,
        )


class ActionDebugLogRecorder(_ExtractedActionDebugLogRecorder):
    def __init__(
        self,
        *,
        root_dir: Path | str | None = None,
        max_queue_size: int = 2048,
        flush_every: int = 1,
        sample_hz: float = 0.0,
    ) -> None:
        super().__init__(
            root_dir=DEFAULT_ACTION_DEBUG_LOG_ROOT if root_dir is None else root_dir,
            max_queue_size=max_queue_size,
            flush_every=flush_every,
            sample_hz=sample_hz,
        )


class IkDebugLogRecorder(_ExtractedIkDebugLogRecorder):
    def __init__(
        self,
        *,
        root_dir: Path | str | None = None,
        max_queue_size: int = 256,
    ) -> None:
        super().__init__(
            root_dir=DEFAULT_IK_DEBUG_LOG_ROOT if root_dir is None else root_dir,
            max_queue_size=max_queue_size,
        )


# MIGRATION(real-teleop, batch 5): compatibility re-export for the
# observational watchdog.  This entrypoint intentionally keeps the legacy
# symbol available while the larger staged extraction is underway.
from ..real_teleop.main_loop_watchdog import MainLoopStallWatchdog

# GUI batch 1 compatibility bridge: the imported policy is pure and the
# coordinator below continues to own all state transitions and event effects.
from ..real_teleop.gui_controls import (
    GuiControlEvents,
    GuiControlState,
    collection_task_controls as _gui_collection_task_controls,
    current_active_controls as _gui_current_active_controls,
    episode_busy as _gui_episode_busy,
    episode_controls as _gui_episode_controls,
    episode_countdown_controls as _gui_episode_countdown_controls,
    finalizing_controls as _gui_finalizing_controls,
    fix_arm_controls as _gui_fix_arm_controls,
    fix_init_allowed_in_controls as _gui_fix_init_allowed_in_controls,
    hold_controls as _gui_hold_controls,
    init_arm_controls as _gui_init_arm_controls,
    movement_multiple_controls as _gui_movement_multiple_controls,
    pose_snapshot_controls as _gui_pose_snapshot_controls,
    pre_home_controls as _gui_pre_home_controls,
    recording_controls as _gui_recording_controls,
    single_arm_tracker_swap_allowed_in_hold as _gui_single_arm_tracker_swap_allowed_in_hold,
    teleop_controls as _gui_teleop_controls,
    vive_calibration_controls as _gui_vive_calibration_controls,
    visualization_controls as _gui_visualization_controls,
)


# Batch 7 compatibility imports: these modules contain pure value/state
# primitives.  Runtime assembly and all device side effects remain below.
from ..real_teleop.components import RealTeleopComponents

RealTeleopComponents.__module__ = __name__

from ..real_teleop.cli_value_parsers import (
    _canfd_radio,
    _canfd_trajectory_mode,
    _dmtac_worker_cpu_affinity,
    _dmtac_worker_nice,
    _local_step_clip_scale_arg,
    _movement_multiple_arg,
    _non_negative_float,
    _positive_finite_float_arg,
    _positive_unit_scale_arg,
    _parse_dmtac_tactile_fields,
    _positive_int,
)
from ..real_teleop.dataset_finalize import RealDatasetFinalizeTask

RealDatasetFinalizeTask.__module__ = __name__

from ..real_teleop.debug_payload_primitives import (
    _compact_debug_value,
    _compact_numeric_list,
    _pose_compact_debug,
    _pose_delta_compact_debug,
    _pose_matrix_debug,
    _qpos_compact_debug,
    _qpos_debug,
)

from ..real_teleop.debug_ik_payload import (
    _IK_PROFILE_COMPACT_KEYS,
    _IMPEDANCE_RECORD_BASE_KEYS,
    _IMPEDANCE_RECORD_BACKEND_TARGET_KEYS,
    _IMPEDANCE_RECORD_FORCE_KEYS,
    _IMPEDANCE_RECORD_TARGET_OFFSET_KEYS,
    _IMPEDANCE_RECORD_REALIZATION_KEYS,
    _IMPEDANCE_RECORD_REALIZATION_TAIL_KEYS,
    _IMPEDANCE_RECORD_RETURN_LIMIT_KEYS,
    _IMPEDANCE_RECORD_JOINT_LIMIT_PROFILE_KEYS,
    _IMPEDANCE_RECORD_COMMON_TAIL_KEYS,
    _IMPEDANCE_RECORD_POSE_TRACKING_KEYS,
    _IMPEDANCE_RECORD_COMPACT_KEYS,
    _IMPEDANCE_RECORD_DEBUG_KEYS,
    _backend_compact_debug_payload,
    _backend_debug_payload,
    _backend_ik_metadata_compact_debug,
    _candidate_selection_compact_debug,
    _command_compact_debug_payload,
    _command_debug_payload,
    _command_metadata_compact_debug,
    _execution_step_compact_debug,
    _ik_profile_compact_debug,
    _impedance_record_compact_debug,
    _impedance_record_debug,
    _observation_pose_compact_debug,
    _observation_pose_debug,
    _teleop_command_metadata_compact_debug,
    _teleop_command_metadata_debug,
)

# GUI batch 2: the coordinator body lives in ``teleop_runtime.gui``.  The
# explicit support binding below keeps legacy imports and monkeypatch points
# stable while this entrypoint retains runtime assembly and control-loop code.
from ..gui.status_payloads import (
    _IK_RED_WARNING_LABELS,
    _IK_RED_WARNING_SINGULARITY_GATE_CODES,
    _IK_RED_WARNING_STATUS_CODES,
    _as_status_error_text,
    _bool_or_false,
    _build_ik_timing_payload,
    _build_ik_warning_payload,
    _build_tactile_camera_status,
    _camera_self_check_payload,
    _camera_status_blocking_message,
    _compact_runtime_bool,
    _compact_runtime_float,
    _debug_timing_arm_text,
    _debug_timing_fault_text,
    _debug_timing_ik_text,
    _float_or_none,
    _ik_timing_status_text,
    _ik_timing_summaries,
    _ik_timing_summary_from_metadata,
    _ik_warning_entries_from_metadata,
    _ik_warning_payload_from_entries,
    _int_or_zero,
    _json_safe_value,
    _mapping_or_empty,
    _pose_quality_level,
    _pose_quality_reason_list,
    _pose_quality_status_text,
    _runtime_fault_label,
    _safe_float_value,
    _safe_int_value,
    _stripped_code,
    _tactile_camera_status_for_side,
    _tactile_device_count,
    _tactile_frame_count,
    _tactile_ready_counts,
    bind_legacy_support as _bind_status_payload_support,
)

_bind_status_payload_support(globals())

from ..real_teleop.debug_timing_payload import (
    _channels_log_payload,
    _fault_signature,
    _faults_log_payload,
    _latency_items_log_payload,
    _latency_log_payload,
    _phase_timings_log_payload,
    bind_legacy_support as _bind_debug_timing_payload_support,
)

_bind_debug_timing_payload_support(globals())

from ..real_teleop.action_debug_payload import (
    _action_debug_active_sides,
    _action_debug_backend_snapshot_from_state,
    _action_debug_jitter_focus,
    _build_action_debug_metadata,
    bind_legacy_support as _bind_action_debug_payload_support,
)

_bind_action_debug_payload_support(globals())

from ..gui.real_teleop_control import (
    RealTeleopControlInput,
    bind_legacy_support as _bind_real_teleop_gui_support,
)

_bind_real_teleop_gui_support(globals())
RealTeleopControlInput.__module__ = __name__

def _run_camera_startup_self_check(
    *,
    camera_sources: Mapping[str, RealSenseRGBSource] | None,
    clutch_input: RealTeleopControlInput,
    preview_camera_sources: Mapping[str, RealSenseRGBSource] | None = None,
    camera_previewer: CameraPreviewer | None = None,
    duration_sec: float = CAMERA_STARTUP_SELF_CHECK_DURATION_SEC,
    interval_sec: float = CAMERA_STARTUP_SELF_CHECK_INTERVAL_SEC,
) -> dict[str, object]:
    started_at = time.time()
    duration_sec = max(0.0, float(duration_sec))
    interval_sec = max(0.01, float(interval_sec))
    final_payload: dict[str, object] = {}
    camera_sources = camera_sources or {}
    if not camera_sources:
        expected_status = _with_expected_wrist_depth_camera_statuses(
            {},
            args=clutch_input.args,
            now=started_at,
        )
        final_payload = _camera_self_check_payload(
            expected_status,
            started_at=started_at,
            now=started_at,
            duration_sec=duration_sec,
            completed=True,
        )
        if clutch_input.ui is not None:
            clutch_input.ui.publish(
                camera_self_check=dict(final_payload),
                camera_status=expected_status,
            )
        _log("CameraSelfCheck", str(final_payload.get("message") or final_payload), now=started_at)
        return final_payload
    _log(
        "CameraSelfCheck",
        "startup self-check "
        f"timeout={duration_sec:.1f}s cameras={sorted(camera_sources)} "
        f"early_success={CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES} "
        "consecutive new frames/camera "
        f"age<={CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC:.3f}s",
        now=started_at,
    )
    previous_frame_counts: dict[str, int | None] = {
        str(camera_key): None for camera_key in camera_sources
    }
    fresh_frame_streaks: dict[str, int] = {
        str(camera_key): 0 for camera_key in camera_sources
    }
    while True:
        now = time.time()
        timed_out = now - started_at >= duration_sec
        status = clutch_input.publish_camera_status(
            camera_sources=camera_sources,
            preview_camera_sources=preview_camera_sources,
            camera_previewer=camera_previewer,
            now=now,
            force=True,
        )
        for camera_key in camera_sources:
            key = str(camera_key)
            item = status.get(key)
            frame_count = _safe_int_value(item.get("frame_count")) if isinstance(item, Mapping) else None
            frame_age_sec = (
                _safe_float_value(item.get("last_frame_age_sec"))
                if isinstance(item, Mapping)
                else None
            )
            status_level = str(item.get("status_level") or "") if isinstance(item, Mapping) else ""
            previous_count = previous_frame_counts[key]
            is_new_fresh_frame = bool(
                frame_count is not None
                and int(frame_count) > 0
                and (previous_count is None or int(frame_count) > previous_count)
                and frame_age_sec is not None
                and 0.0 <= float(frame_age_sec) <= CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC
                and status_level == "ok"
            )
            fresh_frame_streaks[key] = fresh_frame_streaks[key] + 1 if is_new_fresh_frame else 0
            previous_frame_counts[key] = frame_count
        early_success = bool(fresh_frame_streaks) and all(
            streak >= CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES
            for streak in fresh_frame_streaks.values()
        )
        completed = bool(timed_out or early_success)
        final_payload = _camera_self_check_payload(
            status,
            started_at=started_at,
            now=now,
            duration_sec=duration_sec,
            completed=completed,
        )
        final_payload.update(
            {
                "completion_reason": (
                    "fresh_frames" if early_success else "timeout" if timed_out else None
                ),
                "fresh_frame_max_age_sec": CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC,
                "fresh_frame_required_consecutive_samples": CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES,
                "fresh_frame_streaks": dict(fresh_frame_streaks),
            }
        )
        if clutch_input.ui is not None:
            updates: dict[str, object] = {"camera_self_check": dict(final_payload)}
            if completed and final_payload.get("status") in {"failed", "warning"}:
                updates["control_hint"] = str(final_payload.get("message") or "")
            clutch_input.ui.publish(**updates)
        if completed:
            break
        clutch_input.poll()
        if clutch_input.should_quit:
            break
        sleep_sec = min(interval_sec, max(0.0, duration_sec - (time.time() - started_at)))
        if sleep_sec > 0.0:
            time.sleep(sleep_sec)

    message = str(final_payload.get("message") or "")
    _log("CameraSelfCheck", message or str(final_payload), now=time.time())
    if clutch_input.ui is not None and message:
        clutch_input.ui.append_message("CameraSelfCheck", message)
    return final_payload


def _release_runtime_state_readers(components: RealTeleopComponents) -> None:
    if components.arm_state_read_executor is not None:
        components.arm_state_read_executor.shutdown(wait=True)


def _log_dual_arm_readiness(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> None:
    active_sides = tuple(components.active_sides)
    if set(active_sides) != {"left", "right"}:
        _log(
            "DualArmReady",
            f"active_sides={list(active_sides)}; dual-arm collision guard is not used for single-side teleop.",
        )
        return
    if not components.collision_guard_requested:
        _log(
            "DualArmReady",
            "WARNING: dual-arm collision guard explicitly disabled; "
            "the session will use two independent arm retargeters.",
        )
        return
    retargeter = components.dual_arm_retargeter
    if retargeter is None:
        error = components.collision_guard_setup_error or "unknown setup error"
        _log(
            "DualArmReady",
            "WARNING: dual-arm collision guard was requested but is not active; "
            f"setup_error={error}. Do not run close-range dual-arm tests in this mode.",
        )
        return

    safety_profile = retargeter.ik_safety_profile
    joint_velocity_profile = retargeter.joint_velocity_profile
    hard_indices = tuple(int(index) for index in safety_profile.hard_branch_joint_indices)
    j1_relaxed = 0 not in hard_indices
    time_budget_ms = retargeter.config.solve_time_budget_ms
    _log(
        "DualArmReady",
        "guard=enabled "
        f"scene_profile={retargeter.profile.name} "
        f"collision_pairs={len(retargeter.profile.collision_pairs)} "
        f"mode={retargeter.config.contact_mode} "
        f"output={retargeter.config.output_command_type} "
        f"ik_hz={float(retargeter.config.solver_frequency):.1f} "
        f"time_budget_ms={0.0 if time_budget_ms is None else float(time_budget_ms):.1f} "
        f"safety_profile={safety_profile.name} "
        f"hard_branch_joint_indices={list(hard_indices)} "
        f"j1_hard_branch_relaxed={j1_relaxed}",
    )
    local_bounds = safety_profile.metadata.get(
        "impedance_bimanual_local_step_bounds"
    )
    if isinstance(local_bounds, Mapping):
        _log(
            "DualArmReady",
            "impedance_local_candidate_bounds="
            f"{float(local_bounds.get('max_position_worsen_m', 0.0)) * 1000.0:.1f}mm/"
            f"{float(local_bounds.get('max_orientation_worsen_deg', 0.0)):.1f}deg "
            f"dynamic_worsen={bool(local_bounds.get('dynamic_worsen_enabled', False))} "
            f"scope={local_bounds.get('scope', 'unknown')}",
        )
    if joint_velocity_profile is not None:
        _log(
            "DualArmReady",
            "joint_velocity_limit "
            f"profile={joint_velocity_profile.name} "
            f"max_velocity_deg_s={np.round(np.asarray(joint_velocity_profile.max_velocity_deg_s, dtype=float), 3).tolist()} "
            f"max_dt_sec={float(joint_velocity_profile.max_dt_sec):.3f}",
        )
    posture = retargeter.posture_preference
    if posture is not None:
        _log(
            "DualArmReady",
            "rm65_posture_preference "
            f"name={posture.name} "
            f"local_escape={bool(posture.local_escape_enabled)} "
            f"wrist_unload={bool(posture.wrist_unload_enabled)} "
            f"j2_preferred=[{posture.j2_preferred_min_deg:.1f},{posture.j2_preferred_max_deg:.1f}]deg "
            f"j3_preferred=[{posture.j3_preferred_min_deg:.1f},{posture.j3_preferred_max_deg:.1f}]deg "
            f"hard_escape_allowed={list(posture.hard_branch_escape_allowed_joint_indices)}",
        )
    _log(
        "DualArmReady",
        "runtime_home will be captured from live real-arm qpos after startup; "
        "the loaded calibration profile defines coordinate frames, not reusable wrist home.",
    )


def _handle_ik_debug_log_toggle_request(
    *,
    args: argparse.Namespace,
    components: RealTeleopComponents,
    clutch_input: RealTeleopControlInput,
    ik_debug_log_recorder: IkDebugLogRecorder,
    control_ui: CalibrationWebUi | None,
    ik_debug_profile_base_enabled: bool,
    now: float,
) -> None:
    if not clutch_input.consume_ik_debug_log_toggle_request():
        return
    if bool(getattr(args, "no_debug_disk_write", False)):
        clutch_input.set_ik_debug_logging_state(active=False, path="")
        message = "本次运行已通过 --no-debug-disk-write 禁止 debug 写盘。"
        _log("IKDebugLog", message, now=now)
        if control_ui is not None:
            control_ui.append_message("IKDebugLog", message)
        return
    if ik_debug_log_recorder.active:
        stopped_path = ik_debug_log_recorder.stop(reason="gui_toggle_stop", now=now)
        if components.dual_arm_retargeter is not None and len(components.active_sides) > 1:
            components.dual_arm_retargeter.config.debug_profile = ik_debug_profile_base_enabled
            components.dual_arm_retargeter.ik_solver.debug_profile = ik_debug_profile_base_enabled
        clutch_input.set_ik_debug_logging_state(
            active=False,
            path="" if stopped_path is None else str(stopped_path),
        )
        _log(
            "IKDebugLog",
            (
                f"stopped path={stopped_path} frames={ik_debug_log_recorder.frame_count} "
                f"dropped={ik_debug_log_recorder.dropped_frame_count}"
            ),
            now=now,
        )
        if control_ui is not None:
            control_ui.append_message(
                "IKDebugLog",
                f"已停止 IK 记录：{stopped_path}",
            )
        return

    try:
        started_path = ik_debug_log_recorder.start(now=now)
    except Exception as exc:  # noqa: BLE001 - debug logging must not break teleop.
        clutch_input.set_ik_debug_logging_state(active=False, path="")
        message = f"IK 记录启动失败：{type(exc).__name__}: {exc}"
        _log("IKDebugLog", message, now=now)
        if control_ui is not None:
            control_ui.append_message("IKDebugLog", message)
        return
    ik_debug_log_mode = (
        "full" if bool(getattr(args, "ik_debug_log_full", False)) else "compact"
    )
    ik_debug_log_recorder.write_event(
        "mode",
        {"mode": ik_debug_log_mode},
        now=now,
    )
    if components.dual_arm_retargeter is not None and len(components.active_sides) > 1:
        components.dual_arm_retargeter.config.debug_profile = True
        components.dual_arm_retargeter.ik_solver.debug_profile = True
    clutch_input.set_ik_debug_logging_state(active=True, path=str(started_path))
    _log(
        "IKDebugLog",
        f"started path={started_path} mode={ik_debug_log_mode}",
        now=now,
    )
    if control_ui is not None:
        control_ui.append_message(
            "IKDebugLog",
            f"已开始 IK 记录：{started_path} mode={ik_debug_log_mode}",
        )


def _wait_for_real_teleop_start_request(
    components: RealTeleopComponents,
    clutch_input: RealTeleopControlInput,
    *,
    arm_mode: str,
    args: argparse.Namespace,
    control_ui: CalibrationWebUi | None,
    ik_debug_log_recorder: IkDebugLogRecorder,
    ik_debug_profile_base_enabled: bool,
) -> str | None:
    if not clutch_input.enabled:
        _log(
            "Calibration",
            "stdin/gui is not interactive; starting runtime home calibration automatically.",
        )
        return "home"

    _log(
        "Calibration",
        "HOME CAPTURE IS ARMED. Use GUI/keyboard: p=home capture, q=quit.",
    )
    status_gate = ChangeGate()
    while True:
        loop_start = time.time()
        clutch_input.poll()
        _handle_ik_debug_log_toggle_request(
            args=args,
            components=components,
            clutch_input=clutch_input,
            ik_debug_log_recorder=ik_debug_log_recorder,
            control_ui=control_ui,
            ik_debug_profile_base_enabled=ik_debug_profile_base_enabled,
            now=loop_start,
        )
        if clutch_input.should_quit:
            return None
        if clutch_input.consume_calibration_request():
            if _impedance_force_control_warmup_backends(args, components):
                if _warm_start_impedance_force_control_before_home(args, components, clutch_input):
                    clutch_input.publish_calibration_waiting(
                        "力控冷启动完成；请再次点击“开始遥操”进入准备、采集 runtime home 并开始遥操。"
                    )
                    continue
                if clutch_input.should_quit:
                    return None
                continue
            if clutch_input.before_calibration_start_delay(
                DEFAULT_RUNTIME_HOME_START_DELAY_SEC,
                arm_mode=arm_mode,
            ):
                return "home"
            if clutch_input.should_quit:
                return None
            clutch_input.publish_calibration_waiting("Home 标定已取消，回到等待状态。")

        _sync_preview_from_real_state_real(components, required=False)

        channels, _ = components.session.poll_sources(loop_start)
        clutch_input.publish_raw_tracker_pose_state(channels=channels, now=loop_start)
        ready = {
            name: bool(channels.get(name) is not None and channels[name].valid)
            for name in tuple(f"{side}_wrist_pose" for side in components.active_sides)
        }
        if status_gate.should_emit("wait_for_real_start_status", tuple(sorted(ready.items())), now=loop_start):
            _log(
                "Calibration",
                f"waiting_for_gui tracker_ready={ready} ready_key='p'",
                now=loop_start,
            )
        elif status_gate.should_emit(
            "wait_for_real_start_heartbeat",
            "waiting",
            now=loop_start,
            heartbeat_sec=3.0,
        ):
            _log(
                "Calibration",
                f"still_waiting ready={ready} ready_key='p'",
                now=loop_start,
            )
        time.sleep(0.02)


def _runtime_home_failure_message(exc: BaseException) -> str:
    return f"runtime home 标定失败: {type(exc).__name__}: {exc}"


def _arm_after_start_calibration(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    clutch_input: RealTeleopControlInput,
    *,
    home_captured: bool,
) -> None:
    arm_after_start = time.perf_counter()
    reset_elapsed_ms = 0.0
    if home_captured:
        reset_start = time.perf_counter()
        synced = _reset_runtime_motion_state_from_real_qpos(components, required=True)
        reset_elapsed_ms = (time.perf_counter() - reset_start) * 1000.0
        _log("Calibration", f"runtime restart motion state reset from live real qpos synced={synced}")
    state_machine_start = time.perf_counter()
    if home_captured:
        components.session.mark_home_recalibrated("RM65 runtime home captured")
    components.session.set_calibration_ready()
    components.session.mark_debug_passed()
    if components.position_reference_publisher is not None:
        components.position_reference_publisher.reset_epoch()
    state_machine_elapsed_ms = (time.perf_counter() - state_machine_start) * 1000.0
    force_control_start = time.perf_counter()
    _enable_impedance_force_control_after_runtime_home(
        args,
        components,
        home_captured=home_captured,
    )
    force_control_elapsed_ms = (time.perf_counter() - force_control_start) * 1000.0
    arm_hardware_start = time.perf_counter()
    components.session.arm_hardware()
    arm_hardware_elapsed_ms = (time.perf_counter() - arm_hardware_start) * 1000.0
    publish_start = time.perf_counter()
    clutch_input.publish_teleop_running(
        home_ready=home_captured,
        message=(
            "runtime home 完成，已进入真机遥操。"
            if home_captured
            else "runtime home 未就绪，未进入正常真机遥操。"
        ),
    )
    publish_elapsed_ms = (time.perf_counter() - publish_start) * 1000.0
    _log(
        "CalibrationTiming",
        f"arm_after_home total={(time.perf_counter() - arm_after_start) * 1000.0:.1f}ms "
        f"reset_motion={reset_elapsed_ms:.1f}ms "
        f"state_machine={state_machine_elapsed_ms:.1f}ms "
        f"force_control={force_control_elapsed_ms:.1f}ms "
        f"arm_hardware={arm_hardware_elapsed_ms:.1f}ms "
        f"publish_gui={publish_elapsed_ms:.1f}ms",
    )


def main() -> None:
    args = parse_args()
    foot_pad_input = (
        FootPadSpecialArmInput.wait_until_online()
        if str(getattr(args, "special_arm_control_mode", ""))
        == SPECIAL_ARM_CONTROL_MODE_FOOT_PAD
        else None
    )
    tactile_visualizer = None
    if bool(args.visualize_tactile) and not bool(args.skip_renderer):
        tactile_visualizer = _build_tactile_visualizer(args, _active_sides_from_args(args))
    try:
        components = _build_components(args)
    except Exception:
        if tactile_visualizer is not None:
            tactile_visualizer.close()
        raise
    try:
        camera_sources = _build_camera_sources(args)
        try:
            preview_camera_sources = _select_preview_camera_sources(
                camera_sources,
                str(getattr(args, "vis_cam_key", "") or ""),
            )
        except ValueError:
            if bool(getattr(args, "vis_cam", False)):
                raise
            _log(
                "Camera",
                f"preview camera key {getattr(args, 'vis_cam_key', '')!r} is unavailable; preview disabled.",
            )
            preview_camera_sources = {}
    except Exception:
        if tactile_visualizer is not None:
            tactile_visualizer.close()
        raise
    setattr(args, "camera_visualization_available", bool(preview_camera_sources))
    setattr(args, "tactile_visualization_available", _tactile_visualization_available(args, components))
    if tactile_visualizer is None:
        tactile_visualizer = _build_tactile_visualizer(args, components.active_sides)
    elif not bool(getattr(args, "tactile_visualization_available", False)):
        tactile_visualizer.close()
        tactile_visualizer = None
    if tactile_visualizer is not None:
        tactile_vis_mode = (
            "subprocess"
            if isinstance(tactile_visualizer, AsyncTactileUVVisualizer)
            else "dmtac_force"
            if isinstance(tactile_visualizer, DMTacDistributedForceVisualizer)
            else "same_process"
        )
        tactile_vis_detail = (
            "source=dmtac_distributed_force"
            if isinstance(tactile_visualizer, DMTacDistributedForceVisualizer)
            else f"regions={list(_parse_hand_tactile_regions(getattr(args, 'hand_tactile_regions', 'all')))}"
        )
        _log(
            "TactileVis",
            f"enabled mode={tactile_vis_mode} active_sides={list(components.active_sides)} "
            f"{tactile_vis_detail}",
        )
    control_ui = (
        CalibrationWebUi(open_browser=True)
        if bool(getattr(args, "gui_requested", False))
        else None
    )
    pose_snapshot_recorder = (
        None
        if bool(getattr(args, "no_debug_disk_write", False))
        else _fix_arm_pose_snapshot_recorder(components)
    )
    clutch_input = RealTeleopControlInput(
        terminal=TerminalClutchInput(components.keyboard_source),
        ui=control_ui,
        args=args,
        components=components,
        pose_snapshot_recorder=pose_snapshot_recorder,
        foot_pad_input=foot_pad_input,
    )
    debug_episode_recording_enabled = _debug_episode_recording_enabled(args)
    episode_recorder = EpisodeRecorder(
        root_dir=_episode_root_dir(args, components) if debug_episode_recording_enabled else None,
        config=EpisodeRecorderConfig(enabled=debug_episode_recording_enabled),
    )
    control_debug_requested = bool(args.control_debug_config.enabled)
    _merge_control_debug_runtime_metadata(
        args,
        attached=False,
        session_sidecar_requested=control_debug_requested,
        session_sidecar_attached=False,
        session_sidecar_error=None,
        session_sidecar_output_dir=None,
        session_sidecar_attached_wall_time=None,
        session_sidecar_attached_monotonic_time=None,
        episode_sidecar_requested=bool(
            control_debug_requested
            and bool(getattr(args, "control_debug_episode_sidecar", True))
            and not bool(getattr(args, "no_collection", False))
        ),
        episode_sidecar_attached=False,
        episode_sidecar_error=None,
        episode_sidecar_output_dir=None,
        episode_sidecar_episode_index=None,
        episode_sidecar_attached_wall_time=None,
        episode_sidecar_attached_monotonic_time=None,
        episode_sidecars=False,
    )
    dataset_recorder = (
        None
        if bool(getattr(args, "no_collection", False))
        else RealTeleopDatasetRecorder(
            args=args,
            components=components,
            camera_sources=camera_sources,
        )
    )

    def update_control_debug_runtime_metadata(**updates: object) -> dict[str, object]:
        runtime_metadata = _merge_control_debug_runtime_metadata(args, **updates)
        if dataset_recorder is not None:
            control_debug_metadata = _debug_timing_metadata(args).get("control_debug", {})
            if isinstance(control_debug_metadata, Mapping):
                dataset_recorder.collector_components.metadata["control_debug"] = dict(
                    control_debug_metadata
                )
        return runtime_metadata

    camera_previewer: CameraPreviewer | None = None
    camera_preview_active = False
    console_stream = sys.__stdout__
    silenced_stdout = None
    original_stdout = None
    if args.print_tactile_sum_only:
        console_stream.write(
            "[TactileSum] tactile-sum-only mode enabled; press 'p' or Enter for calibration, 'q' to quit.\n"
        )
        console_stream.flush()
        silenced_stdout = open(os.devnull, "w")
        original_stdout = sys.stdout
        sys.stdout = silenced_stdout
    position_audit_fanout: PositionAuditFanout | None = None
    position_debug_session_recorder: PositionDebugRecorder | None = None
    position_debug_episode_recorder: PositionDebugRecorder | None = None
    position_debug_session_dir: Path | None = None
    position_debug_close_threads: list[threading.Thread] = []
    if bool(args.control_debug_config.enabled):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            position_debug_session_dir = (
                _expand_cli_path(args.control_debug_config.root)
                / f"{timestamp}_vive_rm65_real_teleop"
            )
            position_debug_session_recorder = _new_position_debug_recorder(
                args,
                output_dir=position_debug_session_dir,
                thread_name="position-debug-session-writer",
                components=components,
                recorder_scope="session",
            )
            position_audit_fanout = PositionAuditFanout((position_debug_session_recorder,))
            _attach_position_audit_sink(components, position_audit_fanout)
            attached_wall_time = time.time()
            attached_monotonic_time = time.monotonic()
            update_control_debug_runtime_metadata(
                attached=True,
                session_output_dir=str(position_debug_session_dir),
                session_sidecar_attached=True,
                session_sidecar_error=None,
                session_sidecar_output_dir=str(position_debug_session_dir),
                session_sidecar_attached_wall_time=attached_wall_time,
                session_sidecar_attached_monotonic_time=attached_monotonic_time,
            )
            _log(
                "ControlDebug",
                f"async audit attached output={position_debug_session_dir} "
                f"mode={args.control_debug_config.level}",
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics must not alter control behavior.
            cleanup_errors: list[str] = []
            if position_debug_session_recorder is not None:
                try:
                    position_debug_session_recorder.close(timeout_sec=1.0)
                except Exception as cleanup_exc:  # noqa: BLE001
                    cleanup_errors.append(
                        f"recorder_close:{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
            position_debug_session_recorder = None
            position_audit_fanout = None
            try:
                _attach_position_audit_sink(components, None)
            except Exception as cleanup_exc:  # noqa: BLE001
                cleanup_errors.append(
                    f"sink_detach:{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
            setup_error = f"{type(exc).__name__}: {exc}"
            if cleanup_errors:
                setup_error += "; cleanup=" + "; ".join(cleanup_errors)
            update_control_debug_runtime_metadata(
                attached=False,
                error=setup_error,
                session_sidecar_attached=False,
                session_sidecar_error=setup_error,
                session_sidecar_output_dir=(
                    None
                    if position_debug_session_dir is None
                    else str(position_debug_session_dir)
                ),
                session_sidecar_attached_wall_time=None,
                session_sidecar_attached_monotonic_time=None,
            )
            _log(
                "ControlDebug",
                f"WARNING: async audit setup failed and was isolated: {type(exc).__name__}: {exc}",
            )
    try:
        if components.left_state_backend is not None:
            components.left_state_backend.start()
        if components.right_state_backend is not None:
            components.right_state_backend.start()
        if components.left_hand_backend is not None and not components.left_hand_control_via_rig:
            components.left_hand_backend.start()
        if components.right_hand_backend is not None and not components.right_hand_control_via_rig:
            components.right_hand_backend.start()
        components.session.start()
        vive_source = _vive_pose_source_from_components(components)
        if vive_source is not None and vive_source.clock_sync_enabled:
            sync_wait_sec = (
                float(args.vive_clock_sync_startup_timeout_sec)
                if bool(args.vive_clock_sync_required)
                else 0.0
            )
            sync_ready = vive_source.wait_for_clock_sync(
                sync_wait_sec,
                min_samples=int(args.vive_clock_sync_min_samples),
            )
            sync_stats = vive_source.clock_sync_debug_stats()
            _log(
                "ViveClockSync",
                "startup "
                f"ready={sync_ready} required={bool(args.vive_clock_sync_required)} "
                f"stats={sync_stats}",
            )
            if bool(args.vive_clock_sync_required) and not sync_ready:
                raise RuntimeError(
                    "Required VIVE clock synchronization did not become ready before "
                    "real teleoperation startup. Check sender --clock-sync-* settings "
                    "and the separate UDP control port."
                )
        if camera_sources:
            _start_camera_sources(camera_sources)
        if dataset_recorder is not None:
            dataset_recorder.start()
            dataset_recorder.write_camera_device_config()
        clutch_input.start()
        clutch_input.set_visualization_state(
            camera_active=False,
            tactile_active=tactile_visualizer is not None,
            status="",
        )
        clutch_input.publish_camera_status(
            camera_sources=camera_sources,
            preview_camera_sources=preview_camera_sources,
            camera_previewer=camera_previewer,
            force=True,
        )
        _run_camera_startup_self_check(
            camera_sources=camera_sources,
            clutch_input=clutch_input,
            preview_camera_sources=preview_camera_sources,
            camera_previewer=camera_previewer,
        )
        if clutch_input.should_quit:
            return
        if camera_sources:
            _log(
                "Camera",
                f"configured cameras={sorted(camera_sources)} "
                f"config={getattr(args, 'camera_config', '')} "
                f"selected={sorted(preview_camera_sources)}",
            )
        if bool(getattr(args, "vis_cam", False)) and preview_camera_sources:
            try:
                _start_camera_sources(preview_camera_sources)
                camera_previewer = CameraPreviewer(CameraPreviewConfig())
                if camera_previewer.ensure_available():
                    camera_preview_active = True
                    clutch_input.set_visualization_state(
                        camera_active=True,
                        status=f"图像窗口已打开: {', '.join(sorted(preview_camera_sources))}",
                    )
                    _log("Camera", f"preview active cameras={sorted(preview_camera_sources)}")
                else:
                    reason = camera_previewer.disabled_reason or "display unavailable"
                    camera_previewer.close()
                    camera_previewer = None
                    clutch_input.set_visualization_state(camera_active=False, status=reason)
                    _log("Camera", f"preview disabled: {reason}")
            except Exception as exc:  # noqa: BLE001
                if camera_previewer is not None:
                    camera_previewer.close()
                    camera_previewer = None
                clutch_input.set_visualization_state(camera_active=False, status=f"图像窗口启动失败: {exc}")
                _log("Camera", f"preview startup failed: {type(exc).__name__}: {exc}")
        _sync_preview_from_real_state_real(components, required=True)
        _sync_dual_arm_retargeter_for_real_control(components)
        _sync_real_execution_step_limits(components)

        for source_name, (logical_side, arm_ip) in _resolve_logical_side_assignments(args).items():
            if not arm_ip or logical_side not in components.active_sides:
                continue
            _log("ArmMode", f"{source_name}={arm_ip} -> logical_{logical_side}_arm")
        _log(
            "InputMapping",
            "vive_tracker_sides "
            f"swap={bool(getattr(args, 'swap_operator_sides', False))} "
            f"mapping={{"
            + ", ".join(
                f"logical_{side}:{_tracker_source_side_for_logical_side(args, side)}"
                for side in components.active_sides
            )
            + "} arms_hands_serial_profiles_unchanged fixed_after_start",
        )
        if _has_dmtac_gripper_backend(components):
            policy = components.gripper_policy
            _log(
                "Gripper",
                "DMTAC backend "
                f"q6_control_sides={[side for side in components.active_sides if _gripper_retargeter_for_side(components, side) is not None]} "
                f"backend_sides={[side for side in components.active_sides if _gripper_backend_for_side(components, side) is not None]} "
                f"left_q6_port={getattr(args, 'left_hand_port', '') or None} "
                f"right_q6_port={getattr(args, 'right_hand_port', '') or None} "
                f"dmtac_config={getattr(args, 'dmtac_config', 'latest')} "
                f"loaded_sides={list(getattr(args, 'dmtac_config_loaded_sides', []) or [])} "
                f"source_dim={policy.get('source_dim')} output_dim={policy.get('output_dim')} "
                f"invert={policy.get('invert')} max_action={float(policy.get('max_action', GRIPPER_ACTION_MAX)):.3f} "
                f"speed={float(policy.get('speed', 0.2)):.3f} torque={float(policy.get('torque', 0.2)):.3f} "
                "dispatch=gui_gated",
            )
        elif components.left_hand_retargeter is not None or components.right_hand_retargeter is not None:
            _log(
                "HandTeleop",
                "enabled "
                f"left_port={getattr(args, 'left_hand_port', '') or None} "
                f"right_port={getattr(args, 'right_hand_port', '') or None} "
                f"left_hand_ip={args.left_hand_ip or None} "
                f"right_hand_ip={args.right_hand_ip or None} "
                f"joint_order={list(getattr(args, 'hand_joint_order', DEFAULT_INSPIRE_HAND_JOINT_ORDER))} "
                f"invert={bool(getattr(args, 'hand_invert', True))} "
                f"alpha={float(getattr(args, 'hand_command_alpha', 1.0)):.3f} "
                f"telemetry_hz={float(getattr(args, 'hand_telemetry_hz', 30.0)):.1f} "
                f"tactile_regions={list(_parse_hand_tactile_regions(getattr(args, 'hand_tactile_regions', 'all'))) or []} "
                f"dispatch=rig_parallel",
            )
        elif components.left_hand_backend is not None or components.right_hand_backend is not None:
            _log(
                "HandTeleop",
                "telemetry-only "
                f"left_hand_ip={args.left_hand_ip or None} "
                f"right_hand_ip={args.right_hand_ip or None} "
                f"telemetry_hz={float(getattr(args, 'hand_telemetry_hz', 30.0)):.1f} "
                f"tactile_regions={list(_parse_hand_tactile_regions(getattr(args, 'hand_tactile_regions', 'all'))) or []}",
            )
        else:
            _log("HandTeleop", "disabled; arm teleop path remains unchanged.")
        if args.teleop_hardware_ip or args.teleop_hardware_port:
            _log(
                "ReservedIO",
                "teleop-hardware endpoint is still reserved on this entrypoint and remains inactive.",
            )
        if components.loaded_initial_alignment is not None:
            _log(
                "InitialAlignment",
                f"source={args.load_initial_alignment} "
                f"left_q={_qpos_summary(components.loaded_initial_alignment.get_side('left').arm_qpos if components.loaded_initial_alignment.get_side('left') is not None else None)} "
                f"right_q={_qpos_summary(components.loaded_initial_alignment.get_side('right').arm_qpos if components.loaded_initial_alignment.get_side('right') is not None else None)}",
            )
            start_check_status = _run_initial_alignment_start_check(args, components)
            if start_check_status == "passed":
                _log(
                    "InitialAlignment",
                    "startup joint check passed; runtime home capture at 'p' will still use the live real-arm pose.",
                )
            elif start_check_status == "ignored_mismatch":
                _log(
                    "InitialAlignment",
                    "startup joint mismatch was ignored by CLI; runtime home capture at 'p' will still use the live real-arm pose.",
                )
        _emit_tracker_axis_mapping_logs(components)
        _log(
            "IK",
            f"objective_mode={components.session.calibration.metadata.get('ik_objective_mode')} "
            f"tuning={components.session.calibration.metadata.get('ik_tuning_source')}",
        )
        _log(
            "ArmMode",
            f"execution_profile={args.arm_execution_config.resolved_profile} "
            f"requested_profile={args.arm_execution_config.requested_profile} "
            f"control_mode={_arm_control_mode_arg(args)} "
            f"impedance_mode={_impedance_mode_arg(args)} "
            f"strict_tool_profile={_strict_tool_profile_arg(args)} "
            f"force_tare_sec={_force_tare_sec_arg(args):.2f} "
            f"state_read_retries={args.rm65_hardware_read_retries} "
            f"retry_delay_sec={args.rm65_hardware_read_retry_delay_sec:.3f} "
            f"io_threads={max(1, int(args.arm_io_threads))} "
            f"sample_hz={float(args.hz):.1f} "
            f"outer_control_hz={_effective_outer_control_hz(args):.1f} "
            f"arm_control_hz={_effective_arm_control_hz(args):.1f} "
            f"minimum_control_hz={args.minimum_control_hz:.1f} "
            f"arm_max_angvel_rad_s={np.round(np.asarray(args.arm_max_angvel, dtype=float), 3).tolist()} "
            f"arm_max_angvel_effective_rad_s={np.round(np.asarray(args.arm_max_angvel, dtype=float) * (0.2 if bool(getattr(args, 'strict_degree_limit', False)) else 1.0), 3).tolist()} "
            f"position_max_command_step_deg={getattr(args, 'position_max_command_step_deg', None)} "
            f"position_target_velocity_filter_tau_sec={getattr(args, 'position_target_velocity_filter_tau_sec', DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC)} "
            f"strict_degree_limit={'on' if bool(getattr(args, 'strict_degree_limit', False)) else 'off'} "
            f"actual_dt_scaling={'on' if args.arm_step_dt_scaling else 'off'} "
            f"dt_alpha={args.arm_step_dt_alpha:.2f} "
            f"dt_max_scale={args.arm_step_dt_max_scale:.2f} "
            f"dt_stall_reset_sec={args.arm_step_dt_stall_reset_sec:.3f}",
        )
        _log(
            "ControlDebug",
            f"level={args.control_debug_config.level} "
            f"root={args.control_debug_config.root} "
            f"queue_size={args.control_debug_config.queue_size} "
            f"flush_every={args.control_debug_config.flush_every} "
            "observational_only=true",
        )
        if bool(getattr(args, "no_debug_disk_write", False)):
            _log(
                "DebugDiskWrite",
                "disabled by --no-debug-disk-write; formal dataset collection is unaffected",
            )
        if components.temporary_safety_limiter is not None:
            temp_config = components.temporary_safety_limiter.config
            _log(
                "TempSafety",
                (
                    "temporary dual-arm output limiter enabled "
                    f"max_xyz_step_m={temp_config.max_xyz_step_m:.3f} "
                    f"max_rot_step_deg={temp_config.max_rot_step_deg:.1f} "
                    f"max_joint_step_deg={temp_config.max_joint_step_deg:.1f} "
                    f"command_scale={temp_config.command_scale:.2f}; "
                    "disable with --no-dual-arm-temporary-safety"
                ),
            )
        else:
            disabled_reason = components.session.calibration.metadata.get(
                "dual_arm_temporary_safety_disabled_reason"
            )
            reason_text = "" if not disabled_reason else f" reason={disabled_reason}"
            _log("TempSafety", f"temporary dual-arm output limiter disabled{reason_text}")
        if components.collision_guard_setup_error is not None:
            _log(
                "CollisionGuard",
                "disabled automatically; "
                f"{components.collision_guard_setup_error}. "
                "Falling back to two independent arm retargeters. "
                "Use --no-collision to disable this explicitly.",
            )
        if components.dual_arm_retargeter is not None:
            _log(
                "CollisionGuard",
                "enabled "
                f"scene_profile={args.collision_scene_profile} "
                f"pairs={len(components.dual_arm_retargeter.profile.collision_pairs)} "
                f"mode={args.collision_avoidance_mode}",
            )
        else:
            _log("CollisionGuard", "disabled; using two independent arm retargeters.")
        _log_dual_arm_readiness(args, components)
        _emit_collision_proxy_logs(components)
        debug_timing_modules = _debug_timing_modules(args)
        debug_timing_state = RealDebugTimingState(
            enabled=bool(debug_timing_modules.console),
            threshold_sec=float(getattr(args, "debug_timing_threshold_sec", 0.025)),
            report_interval_sec=float(getattr(args, "debug_timing_report_interval_sec", 1.0)),
        )
        ik_debug_profile_base_enabled = bool(debug_timing_modules.ik_profile)
        ik_debug_log_recorder = IkDebugLogRecorder()
        debug_timing_log_recorder = DebugTimingLogRecorder(
            heartbeat_sec=max(0.1, float(getattr(args, "debug_timing_report_interval_sec", 1.0))),
            sample_hz=float(getattr(args, "debug_timing_log_sample_hz", 0.0)),
        )
        action_debug_log_recorder = ActionDebugLogRecorder(
            root_dir=_expand_cli_path(getattr(args, "debug_action_log_root", DEFAULT_ACTION_DEBUG_LOG_ROOT)),
            max_queue_size=int(getattr(args, "debug_action_queue_size", 2048)),
            flush_every=int(getattr(args, "debug_action_flush_every", 1)),
            sample_hz=float(getattr(args, "debug_action_sample_hz", 0.0)),
        )
        canfd_runtime_config = _canfd_follow_runtime_config(args)
        action_debug_position_canfd_passthrough = bool(
            _arm_control_mode_arg(args) == "position"
            and canfd_runtime_config["position_background_streaming_enabled"]
        )
        action_debug_temporary_safety = _effective_dual_arm_temporary_safety(
            args,
            position_canfd_passthrough=action_debug_position_canfd_passthrough,
        )
        main_loop_watchdog = MainLoopStallWatchdog(
            enabled=bool(debug_timing_modules.watchdog),
            timeout_sec=max(5.0, float(debug_timing_state.report_interval_sec) * 5.0),
            report_interval_sec=max(5.0, float(debug_timing_state.report_interval_sec) * 5.0),
        )
        if debug_timing_modules.light_log:
            try:
                debug_timing_path = debug_timing_log_recorder.start(now=time.time())
                debug_timing_log_recorder.write_event(
                    "config",
                    {
                        "modules": list(debug_timing_modules.enabled_names()),
                        "legacy_all": bool(debug_timing_modules.legacy_all),
                        "threshold_sec": float(debug_timing_state.threshold_sec),
                        "report_interval_sec": float(debug_timing_state.report_interval_sec),
                        "sample_hz": float(
                            getattr(args, "debug_timing_log_sample_hz", 0.0)
                        ),
                        "console": bool(debug_timing_modules.console),
                        "rm65_backend_debug_timing": bool(debug_timing_modules.rm65_backend),
                        "ik_profile": bool(debug_timing_modules.ik_profile),
                        "main_loop_watchdog_enabled": bool(debug_timing_modules.watchdog),
                        "main_loop_watchdog_timeout_sec": float(main_loop_watchdog.timeout_sec),
                    },
                    now=time.time(),
                )
            except Exception as exc:  # noqa: BLE001 - diagnostics must not block teleop.
                debug_timing_path = None
                _log(
                    "DebugTiming",
                    f"light log disabled: {type(exc).__name__}: {exc}",
                )
        else:
            debug_timing_path = None
        if bool(getattr(args, "debug_action", False)):
            try:
                action_debug_path = action_debug_log_recorder.start(now=time.time())
                action_debug_log_recorder.write_event(
                    "config",
                    {
                        "standalone": True,
                        **_arm_execution_profile_metadata(args),
                        **_position_ik_iteration_metadata(args),
                        "control_debug": args.control_debug_config.to_metadata(),
                        "dataset_metadata_also_enabled_when_recording": True,
                        "no_collection": bool(getattr(args, "no_collection", False)),
                        "no_record": bool(getattr(args, "no_record", True)),
                        "sample_hz": float(getattr(args, "debug_action_sample_hz", 0.0)),
                        "hz": float(getattr(args, "hz", 0.0) or 0.0),
                        "arm_control_hz": float(_effective_arm_control_hz(args)),
                        "safe_debug": bool(getattr(args, "safe_debug", False)),
                        "disable_hold": bool(getattr(args, "disable_hold", False)),
                        "strict_degree_limit": bool(getattr(args, "strict_degree_limit", False)),
                        "qpos_limit_scale": (
                            0.2 if bool(getattr(args, "strict_degree_limit", False)) else 1.0
                        ),
                        "arm_canfd_follow": bool(_effective_arm_canfd_follow(args)),
                        "arm_canfd_follow_arg": getattr(args, "arm_canfd_follow", None),
                        "position_post_ik_qpos_smoothing_alpha": getattr(
                            args,
                            "position_post_ik_qpos_smoothing_alpha",
                            None,
                        ),
                        "position_post_ik_qpos_smoothing_alpha_effective": float(
                            _effective_position_post_ik_qpos_smoothing_alpha(args)
                        ),
                        "arm_follow_qpos_alpha": getattr(args, "arm_follow_qpos_alpha", None),
                        "arm_follow_qpos_alpha_effective": float(
                            _effective_arm_follow_qpos_alpha(args)
                        ),
                        "arm_canfd_follow_stream_hz": getattr(
                            args,
                            "arm_canfd_follow_stream_hz",
                            None,
                        ),
                        "arm_canfd_follow_warning_ratio": getattr(
                            args,
                            "arm_canfd_follow_warning_ratio",
                            None,
                        ),
                        "position_background_streaming_enabled": bool(
                            canfd_runtime_config["position_background_streaming_enabled"]
                        ),
                        "position_background_streaming_hz": float(
                            canfd_runtime_config["position_background_streaming_hz"]
                        ),
                        "canfd_follow_warning_ratio_effective": float(
                            canfd_runtime_config["canfd_follow_warning_ratio"]
                        ),
                        "arm_canfd_trajectory_mode": getattr(args, "arm_canfd_trajectory_mode", None),
                        "canfd_trajectory_mode_effective": int(canfd_runtime_config["canfd_trajectory_mode"]),
                        "arm_canfd_radio": getattr(args, "arm_canfd_radio", None),
                        "canfd_radio_effective": int(canfd_runtime_config["canfd_radio"]),
                        "arm_disable_force_admittance": bool(
                            getattr(args, "arm_disable_force_admittance", False)
                        ),
                        "arm_max_angvel_rad_s": [
                            float(value)
                            for value in np.asarray(
                                getattr(args, "arm_max_angvel", ()),
                                dtype=float,
                            ).reshape(-1).tolist()
                        ],
                        "arm_max_angvel_effective_rad_s": [
                            float(value)
                            for value in (
                                np.asarray(
                                    getattr(args, "arm_max_angvel", ()),
                                    dtype=float,
                                ).reshape(-1)
                                * (
                                    0.2
                                    if bool(getattr(args, "strict_degree_limit", False))
                                    else 1.0
                                )
                            ).tolist()
                        ],
                        "position_max_command_step_deg": getattr(
                            args,
                            "position_max_command_step_deg",
                            None,
                        ),
                        "position_target_velocity_filter_tau_sec": float(
                            getattr(
                                args,
                                "position_target_velocity_filter_tau_sec",
                                DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
                            )
                        ),
                        "arm_step_dt_scaling": bool(
                            getattr(args, "arm_step_dt_scaling", False)
                        ),
                        "arm_step_dt_alpha": float(
                            getattr(args, "arm_step_dt_alpha", 0.0) or 0.0
                        ),
                        "arm_step_dt_max_scale": float(
                            getattr(args, "arm_step_dt_max_scale", 0.0) or 0.0
                        ),
                        "arm_step_dt_max_scale_auto": bool(
                            getattr(args, "arm_step_dt_max_scale_auto", False)
                        ),
                        "arm_step_dt_stall_reset_sec": float(
                            getattr(args, "arm_step_dt_stall_reset_sec", 0.0) or 0.0
                        ),
                        "arm_singularity_protection": bool(
                            getattr(args, "arm_singularity_protection", True)
                        ),
                        "dual_arm_temporary_safety_requested": bool(
                            action_debug_temporary_safety["requested"]
                        ),
                        "dual_arm_temporary_safety_enabled_effective": bool(
                            action_debug_temporary_safety["enabled"]
                        ),
                        "dual_arm_temporary_safety_disabled_reason": action_debug_temporary_safety[
                            "disabled_reason"
                        ],
                        "dual_arm_temp_max_xyz_step_m": float(
                            getattr(args, "dual_arm_temp_max_xyz_step_m", 0.0) or 0.0
                        ),
                        "dual_arm_temp_max_rot_step_deg": float(
                            getattr(args, "dual_arm_temp_max_rot_step_deg", 0.0) or 0.0
                        ),
                        "dual_arm_temp_max_joint_step_deg": float(
                            getattr(args, "dual_arm_temp_max_joint_step_deg", 0.0) or 0.0
                        ),
                        "dual_arm_temp_max_joint_step_effective_deg": float(
                            (getattr(args, "dual_arm_temp_max_joint_step_deg", 0.0) or 0.0)
                            * (
                                0.2
                                if bool(getattr(args, "strict_degree_limit", False))
                                else 1.0
                            )
                        ),
                        "dual_arm_temp_command_scale": float(
                            getattr(args, "dual_arm_temp_command_scale", 0.0) or 0.0
                        ),
                    },
                    now=time.time(),
                )
                _log("DebugAction", f"standalone log path={action_debug_path}")
            except Exception as exc:  # noqa: BLE001 - diagnostics must not block teleop.
                action_debug_path = None
                _log("DebugAction", f"standalone log disabled: {type(exc).__name__}: {exc}")
        else:
            action_debug_path = None
        startup_home_captured = False
        manual_startup_request = _manual_calibration_confirmation_required(args, components)
        while True:
            startup_request = "home"
            if manual_startup_request:
                requested = _wait_for_real_teleop_start_request(
                    components,
                    clutch_input,
                    arm_mode="real",
                    args=args,
                    control_ui=control_ui,
                    ik_debug_log_recorder=ik_debug_log_recorder,
                    ik_debug_profile_base_enabled=ik_debug_profile_base_enabled,
                )
                if requested is None:
                    return
                startup_request = requested
            try:
                _runtime_home_calibration(
                    components,
                    arm_mode="real",
                    timeout_sec=args.calibration_timeout,
                    num_samples=args.calibration_samples,
                    progress_callback=clutch_input.publish_runtime_home_progress,
                )
            except Exception as exc:  # noqa: BLE001
                message = _runtime_home_failure_message(exc)
                _log("Calibration", message)
                clutch_input.publish_calibration_waiting(
                    f"{message}\n请检查 VIVE sender/tracker_role 后重新选择操作。"
                )
                if not clutch_input.enabled:
                    raise
                manual_startup_request = True
                continue
            startup_home_captured = True
            break
        _release_runtime_state_readers(components)
        _log(
            "ArmMode",
            "runtime arm state source=last_commanded_action; startup hardware-state readers are now idle after calibration.",
        )
        if startup_home_captured:
            _emit_home_alignment_logs(components)
        else:
            _log("Calibration", "runtime home alignment logs skipped because Home was not captured.")
        if legacy_tcp_mode_requested(args.requested_tcp_mode):
            _log(
                "TCP",
                "legacy --tcp-mode rm_eef was requested; runtime now uses tool_tcp actions.",
            )
        _arm_after_start_calibration(
            args,
            components,
            clutch_input,
            home_captured=startup_home_captured,
        )
        clutch_input.consume_episode_start_request()
        clutch_input.consume_episode_stop_request()

        if args.keyboard_clutch:
            _log(
                "Keys",
                "p/Enter=start calibration, t=start episode, s=stop episode, z=left clutch, x=right clutch, b=both clutch, r=release, q=quit",
            )
        if episode_recorder.config.enabled:
            auto_start_note = " auto_start=on" if args.record_auto_start else " auto_start=off"
            _log(
                "Episode",
                f"recording ready root={episode_recorder.root_dir or _episode_root_dir(args, components)} "
                f"start_key='t' stop_key='s'{auto_start_note}",
            )
        if dataset_recorder is not None:
            _log(
                "Dataset",
                f"formal collection ready root={dataset_recorder.dataset_root} "
                f"cameras={sorted(camera_sources)} write_video={bool(args.write_video and not args.no_camera)} "
                f"tactile_storage={args.tactile_storage} "
                f"tactile_compression={getattr(args, 'tactile_compression', 'none')} "
                f"worker_mode={dataset_recorder.collection_worker_mode} "
                f"start_key='t' stop_key='s'",
            )

        status_gate = ChangeGate()
        orientation_gate = ChangeGate()
        tactile_sum_gate = ChangeGate()
        status_heartbeat_sec = (
            float(args.status_sec) if float(args.status_sec) > 0.0 else None
        )
        if debug_timing_modules.enabled:
            _log(
                "DebugTiming",
                "enabled "
                f"modules={list(debug_timing_modules.enabled_names())} "
                f"threshold={debug_timing_state.threshold_sec:.3f}s "
                f"report_interval={debug_timing_state.report_interval_sec:.3f}s "
                f"light_log={debug_timing_path}",
            )
        if bool(debug_timing_modules.ik_profile):
            budget = getattr(args, "ik_time_budget_ms", None)
            budget_text = "one_control_period" if budget is None else f"{float(budget):.3f}ms"
            _log(
                "DebugIKTiming",
                "enabled "
                f"gui={'on' if control_ui is not None else 'off'} "
                f"print_every={int(getattr(args, 'debug_ik_timing_print_every', 0))} "
                f"time_budget={budget_text}",
            )
        step_index = 0
        recovery_plan: RecoveryPlan | None = None
        last_dispatched_main_command: TeleopCommand | None = None
        episode_last_state: str | None = None
        last_collection_frequency_warning_active = False
        last_collection_frequency_debug_at = 0.0
        episode_start_countdown_deadline: float | None = None
        episode_start_countdown_reason = "manual_key_start"
        episode_start_warmup_ready_since: float | None = None
        episode_start_warmup_last_message: str | None = None
        episode_recording_paused = False
        episode_recording_pause_reason = ""
        episode_resume_warmup_ready_count = 0
        episode_resume_warmup_required_frames = 2
        tactile_preview_rate = PreviewRateLimiter(float(getattr(args, "tactile_vis_hz", 5.0)))
        camera_preview_rate = PreviewRateLimiter(float(getattr(args, "vis_cam_hz", 10.0)))
        sample_rate_gate = PeriodicRateGate(float(args.hz))
        outer_control_hz = _effective_outer_control_hz(args)
        gripper_warmup_probe_lock = threading.Lock()
        gripper_warmup_probe_thread: threading.Thread | None = None
        gripper_warmup_probe_generation = 0
        gripper_warmup_probe_state: dict[str, object] = {
            "status": "idle",
            "message": "",
            "reason": "",
            "started_at": None,
            "completed_at": None,
            "results": {},
        }

        def close_camera_preview(status: str = "") -> None:
            nonlocal camera_previewer, camera_preview_active
            if camera_previewer is not None:
                camera_previewer.close()
                camera_previewer = None
            camera_preview_active = False
            clutch_input.set_visualization_state(camera_active=False, status=status)

        def open_camera_preview(now: float) -> None:
            nonlocal camera_previewer, camera_preview_active
            if not preview_camera_sources:
                message = "没有可用相机配置；图像窗口未打开。"
                clutch_input.set_visualization_state(camera_active=False, status=message)
                if control_ui is not None:
                    control_ui.append_message("Camera", message)
                return
            try:
                _start_camera_sources(preview_camera_sources)
                if camera_previewer is None:
                    camera_previewer = CameraPreviewer(CameraPreviewConfig())
                if not camera_previewer.ensure_available():
                    reason = camera_previewer.disabled_reason or "display unavailable"
                    close_camera_preview(f"图像窗口不可用: {reason}")
                    if control_ui is not None:
                        control_ui.append_message("Camera", f"图像窗口不可用：{reason}")
                    return
                camera_preview_active = True
                camera_preview_rate.reset()
                message = f"图像窗口已打开: {', '.join(sorted(preview_camera_sources))}"
                clutch_input.set_visualization_state(camera_active=True, status=message)
                _log("Camera", message, now=now)
            except Exception as exc:  # noqa: BLE001
                close_camera_preview(f"图像窗口启动失败: {type(exc).__name__}: {exc}")
                if control_ui is not None:
                    control_ui.append_message("Camera", f"图像窗口启动失败：{exc}")

        def close_tactile_preview(status: str = "") -> None:
            nonlocal tactile_visualizer
            if tactile_visualizer is not None:
                tactile_visualizer.close()
                tactile_visualizer = None
            clutch_input.set_visualization_state(tactile_active=False, status=status)

        def open_tactile_preview(now: float) -> None:
            nonlocal tactile_visualizer
            if not bool(getattr(args, "tactile_visualization_available", False)):
                message = "没有可用触觉 backend；触觉窗口未打开。"
                clutch_input.set_visualization_state(tactile_active=False, status=message)
                if control_ui is not None:
                    control_ui.append_message("TactileVis", message)
                return
            try:
                if tactile_visualizer is None:
                    tactile_visualizer = _build_tactile_visualizer(
                        args,
                        components.active_sides,
                        force=True,
                    )
                if tactile_visualizer is None:
                    raise RuntimeError("tactile visualizer was not created")
                tactile_preview_rate.reset()
                message = "触觉窗口已打开"
                clutch_input.set_visualization_state(tactile_active=True, status=message)
                _log("TactileVis", message, now=now)
            except Exception as exc:  # noqa: BLE001
                close_tactile_preview(f"触觉窗口启动失败: {type(exc).__name__}: {exc}")
                if control_ui is not None:
                    control_ui.append_message("TactileVis", f"触觉窗口启动失败：{exc}")

        def recording_active() -> bool:
            return bool(
                (dataset_recorder is not None and dataset_recorder.is_recording)
                or episode_recorder.is_recording
            )

        def recording_paused() -> bool:
            return bool(
                (dataset_recorder is not None and dataset_recorder.is_paused)
                or episode_recording_paused
            )

        def recording_pause_reason() -> str:
            if dataset_recorder is not None and dataset_recorder.is_paused:
                return dataset_recorder.pause_reason
            return str(episode_recording_pause_reason or "")

        def recording_frame_count() -> int:
            if dataset_recorder is not None and dataset_recorder.is_recording:
                return int(dataset_recorder.frame_count)
            return int(episode_recorder.frame_count)

        def recording_path() -> str:
            if dataset_recorder is not None:
                return str(dataset_recorder.dataset_root)
            return "" if episode_recorder.episode_dir is None else str(episode_recorder.episode_dir)

        def recording_frequency_status() -> dict[str, object]:
            if dataset_recorder is not None and dataset_recorder.is_recording:
                return dataset_recorder.collection_frequency_status()
            return {
                "sample_hz": None,
                "target_hz": float(getattr(args, "collection_warning_hz", 15.0)),
                "warning": {"active": False, "message": ""},
            }

        def gripper_warmup_probe_enabled() -> bool:
            if not bool(getattr(args, "gripper_warmup_probe", True)):
                return False
            backends = [
                _gripper_backend_for_side(components, side)
                for side in components.active_sides
            ]
            return any(
                backend is not None
                and bool(getattr(getattr(backend, "config", None), "connect_hardware", True))
                for backend in backends
            )

        def reset_gripper_warmup_probe(reason: str) -> None:
            nonlocal gripper_warmup_probe_generation, gripper_warmup_probe_state
            if not gripper_warmup_probe_enabled():
                return
            with gripper_warmup_probe_lock:
                gripper_warmup_probe_generation += 1
                gripper_warmup_probe_state = {
                    "status": "idle",
                    "message": "",
                    "reason": str(reason),
                    "started_at": None,
                    "completed_at": None,
                    "results": {},
                }

        def gripper_warmup_probe_running() -> bool:
            if not gripper_warmup_probe_enabled():
                return False
            with gripper_warmup_probe_lock:
                return str(gripper_warmup_probe_state.get("status") or "") == "running"

        def gripper_warmup_probe_failure_message(results: Mapping[str, object]) -> str:
            details: list[str] = []
            for side in components.active_sides:
                result = results.get(side)
                if not isinstance(result, Mapping) or bool(result.get("ok")):
                    continue
                attempts = result.get("attempts")
                last_attempt = None
                if isinstance(attempts, Sequence) and not isinstance(attempts, (str, bytes)) and attempts:
                    maybe_attempt = attempts[-1]
                    if isinstance(maybe_attempt, Mapping):
                        last_attempt = maybe_attempt
                error = (
                    str(last_attempt.get("error") or "")
                    if isinstance(last_attempt, Mapping)
                    else ""
                )
                target = (
                    last_attempt.get("target")
                    if isinstance(last_attempt, Mapping)
                    else None
                )
                measured = (
                    last_attempt.get("measured")
                    if isinstance(last_attempt, Mapping)
                    else None
                )
                target_f = _safe_float_value(target)
                measured_f = _safe_float_value(measured)
                if target_f is not None and measured_f is not None:
                    details.append(
                        f"{side}: {error or 'state_mismatch'} "
                        f"target={target_f:.2f} measured={measured_f:.2f}"
                    )
                else:
                    details.append(f"{side}: {error or 'state_mismatch'}")
            suffix = "；".join(details) if details else "unknown failure"
            return f"gripper warmup probe 失败，已尝试重连后仍未通过：{suffix}。"

        def start_gripper_warmup_probe(now: float, *, reason: str) -> tuple[bool, str]:
            nonlocal gripper_warmup_probe_generation, gripper_warmup_probe_state
            nonlocal gripper_warmup_probe_thread
            if not gripper_warmup_probe_enabled():
                return True, "gripper warmup probe 未启用。"
            side_backends: list[tuple[str, DMTacGripperBackend]] = []
            for side in components.active_sides:
                backend = _gripper_backend_for_side(components, side)
                if backend is None:
                    continue
                if not bool(getattr(getattr(backend, "config", None), "connect_hardware", True)):
                    continue
                side_backends.append((side, backend))
            if not side_backends:
                return True, "gripper warmup probe 无硬件 backend。"
            with gripper_warmup_probe_lock:
                status = str(gripper_warmup_probe_state.get("status") or "idle")
                if status == "ready":
                    return True, str(
                        gripper_warmup_probe_state.get("message") or "gripper warmup probe 已通过。"
                    )
                if status == "running":
                    return False, str(
                        gripper_warmup_probe_state.get("message") or "等待 gripper warmup probe 完成。"
                    )
                if status == "failed":
                    return False, str(
                        gripper_warmup_probe_state.get("message") or "gripper warmup probe 失败。"
                    )
                gripper_warmup_probe_generation += 1
                generation = int(gripper_warmup_probe_generation)
                sides_text = ",".join(side for side, _backend in side_backends)
                gripper_warmup_probe_state = {
                    "status": "running",
                    "message": f"gripper warmup probe 运行中：{sides_text} open->half->open。",
                    "reason": str(reason),
                    "started_at": float(now),
                    "completed_at": None,
                    "results": {},
                }

            def cancelled() -> bool:
                with gripper_warmup_probe_lock:
                    return int(gripper_warmup_probe_generation) != generation

            def worker() -> None:
                nonlocal gripper_warmup_probe_state
                results: dict[str, object] = {}
                targets = (0.0, 0.5, 0.0)
                try:
                    for side, backend in side_backends:
                        if cancelled():
                            return
                        results[side] = _run_gripper_warmup_probe_for_side(
                            side,
                            backend,
                            targets=targets,
                            tolerance=max(
                                0.0,
                                float(getattr(args, "gripper_warmup_probe_tolerance", 0.18)),
                            ),
                            step_timeout_s=max(
                                0.01,
                                float(getattr(args, "gripper_warmup_probe_step_timeout_sec", 2.0)),
                            ),
                            settle_s=max(
                                0.0,
                                float(getattr(args, "gripper_warmup_probe_settle_sec", 0.20)),
                            ),
                            poll_s=max(
                                0.01,
                                float(getattr(args, "gripper_warmup_probe_poll_sec", 0.05)),
                            ),
                            reconnect_on_failure=True,
                            cancelled=cancelled,
                        )
                    if cancelled():
                        return
                    passed = bool(results) and all(
                        isinstance(result, Mapping) and bool(result.get("ok"))
                        for result in results.values()
                    )
                    if passed:
                        reconnected_sides = [
                            side
                            for side, result in results.items()
                            if isinstance(result, Mapping) and bool(result.get("reconnected"))
                        ]
                        if reconnected_sides:
                            message = (
                                "gripper warmup probe 通过；已自动重连 "
                                f"{','.join(reconnected_sides)}。"
                            )
                        else:
                            message = "gripper warmup probe 通过：open->half->open state matched。"
                        status = "ready"
                    else:
                        message = gripper_warmup_probe_failure_message(results)
                        status = "failed"
                except Exception as exc:  # noqa: BLE001 - warmup must fail closed, not hang forever.
                    if cancelled():
                        return
                    message = f"gripper warmup probe 内部异常：{type(exc).__name__}: {exc}。"
                    status = "failed"
                completed_at = time.time()
                with gripper_warmup_probe_lock:
                    if int(gripper_warmup_probe_generation) != generation:
                        return
                    gripper_warmup_probe_state = {
                        "status": status,
                        "message": message,
                        "reason": str(reason),
                        "started_at": float(now),
                        "completed_at": completed_at,
                        "results": results,
                    }
                _log("GripperWarmup", message, now=completed_at)

            gripper_warmup_probe_thread = threading.Thread(
                target=worker,
                name="gripper-warmup-probe",
                daemon=True,
            )
            gripper_warmup_probe_thread.start()
            _log(
                "GripperWarmup",
                f"started reason={reason} sides={','.join(side for side, _backend in side_backends)}",
                now=now,
            )
            return False, "等待 gripper warmup probe 完成。"

        def gripper_warmup_probe_status(now: float, *, reason: str) -> tuple[bool, str]:
            if not gripper_warmup_probe_enabled():
                return True, ""
            with gripper_warmup_probe_lock:
                status = str(gripper_warmup_probe_state.get("status") or "idle")
                message = str(gripper_warmup_probe_state.get("message") or "")
            if status == "ready":
                return True, message or "gripper warmup probe 已通过。"
            if status == "running":
                return False, message or "等待 gripper warmup probe 完成。"
            if status == "failed":
                return False, message or "gripper warmup probe 失败。"
            return start_gripper_warmup_probe(now, reason=reason)

        def publish_collection_frequency_warning(
            frequency_status: Mapping[str, object],
            *,
            now: float,
        ) -> None:
            nonlocal last_collection_frequency_debug_at, last_collection_frequency_warning_active
            warning = frequency_status.get("warning")
            warning_payload = warning if isinstance(warning, Mapping) else {}
            warning_active = bool(warning_payload.get("active", False))
            should_emit_debug = bool(
                warning_active
                and dataset_recorder is not None
                and dataset_recorder.is_recording
                and (
                    not last_collection_frequency_warning_active
                    or float(now) - float(last_collection_frequency_debug_at) >= 5.0
                )
            )
            if warning_active and not last_collection_frequency_warning_active:
                message = str(warning_payload.get("message") or "WARNING: 数采实际频率低于目标 15 Hz。")
                _log("DatasetFreq", message, now=now)
                if control_ui is not None:
                    control_ui.append_message("DatasetFreq", message)
            if should_emit_debug:
                assert dataset_recorder is not None
                last_collection_frequency_debug_at = float(now)
                _log(
                    "DatasetFreqDebug",
                    _dataset_recording_debug_text(
                        dataset_recorder=dataset_recorder,
                        components=components,
                        camera_sources=camera_sources,
                        camera_previewer=camera_previewer,
                        now=now,
                    ),
                    now=now,
                )
            last_collection_frequency_warning_active = warning_active

        def recording_auto_stop_reason(state: TeleopRunState | str) -> str | None:
            state_value = str(getattr(state, "value", state)).upper()
            if state_value == TeleopRunState.FAULT.value:
                return "runtime_fault"
            if state_value == TeleopRunState.ESTOP.value:
                return "runtime_estop"
            return None

        def recording_warmup_status(
            now: float,
            *,
            step_result: SessionStepResult | None,
        ) -> tuple[bool, str]:
            if step_result is None:
                return False, "等待首个控制帧。"
            stop_reason = recording_auto_stop_reason(step_result.state)
            if stop_reason is not None:
                return False, f"等待恢复，当前状态会自动停止数采：{stop_reason}。"
            if clutch_input.hold_active():
                return False, "等待 hold 解除并重采 runtime home。"
            if step_result.state != TeleopRunState.RUNNING:
                return False, f"等待遥操进入 RUNNING，当前状态：{step_result.state.value}。"
            if step_result.observation is None:
                return False, "等待 observation 可用。"
            if step_result.command is None:
                return False, "等待 action command 可用。"
            if dataset_recorder is not None:
                actual_ready, actual_message = dataset_recorder.actual_feedback_warmup_status(now)
                if not actual_ready:
                    return False, actual_message
                worker_ready, worker_message = (
                    dataset_recorder.collection_worker_ready_status()
                )
                if not worker_ready:
                    return False, worker_message
            if not bool(getattr(args, "no_camera", False)):
                for camera_key, source in camera_sources.items():
                    frame = source.frame_buffer().get_latest()
                    if frame is None:
                        return False, f"等待相机 {camera_key} 首帧。"
                    camera_age = float(now) - float(frame.source_timestamp)
                    if camera_age < -0.05:
                        return False, f"等待相机 {camera_key} 时间戳稳定。"
                    if camera_age > 1.0:
                        return False, f"等待相机 {camera_key} 最新帧，当前 age={camera_age:.2f}s。"
                camera_status = clutch_input.publish_camera_status(
                    camera_sources=camera_sources,
                    preview_camera_sources=preview_camera_sources,
                    camera_previewer=camera_previewer,
                    now=now,
                    force=False,
                )
                camera_blocking_message = _camera_status_blocking_message(camera_status)
                if camera_blocking_message:
                    return False, camera_blocking_message
            if any(
                _hand_backend_mode_for_side(args, side) != HAND_BACKEND_NONE
                for side in components.active_sides
            ):
                for side in components.active_sides:
                    if _hand_backend_mode_for_side(args, side) == HAND_BACKEND_NONE:
                        continue
                    channel = step_result.channels.get(f"{side}_hand_q6")
                    if channel is None or not bool(getattr(channel, "valid", False)):
                        return False, f"等待 {side} hand_q6 输入有效。"
                    hand_age = float(now) - float(getattr(channel, "timestamp", now))
                    if hand_age < -0.05:
                        return False, f"等待 {side} hand_q6 时间戳稳定。"
                    if hand_age > 0.5:
                        return False, f"等待 {side} hand_q6 最新输入，当前 age={hand_age:.2f}s。"
            probe_ready, probe_message = gripper_warmup_probe_status(now, reason="recording_warmup")
            if not probe_ready:
                return False, probe_message
            return True, "warmup 输入稳定。"

        def start_position_episode_debug(episode_index: int) -> bool:
            nonlocal position_debug_episode_recorder
            requested = bool(
                dataset_recorder is not None
                and args.control_debug_config.enabled
                and bool(getattr(args, "control_debug_episode_sidecar", True))
            )
            update_control_debug_runtime_metadata(
                episode_sidecar_requested=requested,
                episode_sidecar_attached=False,
                episode_sidecars=False,
            )
            if not requested:
                return False
            if position_audit_fanout is None:
                unavailable_error = "session position-audit fanout is unavailable"
                update_control_debug_runtime_metadata(
                    episode_sidecar_error=unavailable_error,
                    episode_sidecar_episode_index=int(episode_index),
                )
                return False
            if position_debug_episode_recorder is not None:
                # Defensive cleanup: normal stop removes the previous sink.
                position_audit_fanout.remove_sink(position_debug_episode_recorder)
                try:
                    position_debug_episode_recorder.close(timeout_sec=1.0)
                except Exception as exc:  # noqa: BLE001 - diagnostics remain best-effort.
                    _log(
                        "ControlDebug",
                        "WARNING: stale episode audit close failed before reattach: "
                        f"{type(exc).__name__}: {exc}",
                    )
                position_debug_episode_recorder = None
            output_dir = (
                Path(dataset_recorder.dataset_root)
                / "meta"
                / "position_control"
                / f"episode_{int(episode_index):06d}"
            )
            recorder: PositionDebugRecorder | None = None
            try:
                recorder = _new_position_debug_recorder(
                    args,
                    output_dir=output_dir,
                    thread_name=f"position-debug-episode-{int(episode_index):06d}",
                    components=components,
                    recorder_scope="episode",
                    episode_index=int(episode_index),
                )
                if not position_audit_fanout.add_sink(recorder):
                    raise RuntimeError("episode position-audit recorder was already attached")
                position_debug_episode_recorder = recorder
                attached_wall_time = time.time()
                attached_monotonic_time = time.monotonic()
                update_control_debug_runtime_metadata(
                    episode_sidecar_requested=True,
                    episode_sidecar_attached=True,
                    episode_sidecar_error=None,
                    episode_sidecar_output_dir=str(output_dir),
                    episode_sidecar_episode_index=int(episode_index),
                    episode_sidecar_attached_wall_time=attached_wall_time,
                    episode_sidecar_attached_monotonic_time=attached_monotonic_time,
                    episode_sidecars=True,
                )
                _log(
                    "ControlDebug",
                    f"episode audit attached episode={int(episode_index)} output={output_dir}",
                )
                return True
            except Exception as exc:  # noqa: BLE001 - isolate debug storage from control.
                if recorder is not None:
                    position_audit_fanout.remove_sink(recorder)
                    if position_debug_episode_recorder is recorder:
                        position_debug_episode_recorder = None
                    try:
                        recorder.close(timeout_sec=1.0)
                    except Exception:
                        pass
                setup_error = f"{type(exc).__name__}: {exc}"
                update_control_debug_runtime_metadata(
                    episode_sidecar_requested=True,
                    episode_sidecar_attached=False,
                    episode_sidecar_error=setup_error,
                    episode_sidecar_output_dir=str(output_dir),
                    episode_sidecar_episode_index=int(episode_index),
                    episode_sidecar_attached_wall_time=None,
                    episode_sidecar_attached_monotonic_time=None,
                    episode_sidecars=False,
                )
                _log(
                    "ControlDebug",
                    f"WARNING: episode audit setup failed: {type(exc).__name__}: {exc}",
                )
                return False

        def stop_position_episode_debug(
            *,
            reason: str,
        ) -> Callable[[], Mapping[str, Any] | None] | None:
            """Detach the episode audit immediately and return its close action.

            The caller decides where to run the close action.  In impedance
            collection it is executed by the *same* deferred finalizer as the
            dataset writer, avoiding two independently scheduled Python
            finalizers during the first post-stop control period.
            """

            nonlocal position_debug_episode_recorder
            recorder = position_debug_episode_recorder
            if recorder is None:
                return None
            position_debug_episode_recorder = None
            if position_audit_fanout is not None:
                position_audit_fanout.remove_sink(recorder)
            update_control_debug_runtime_metadata(
                episode_sidecar_attached=False,
                episode_sidecars=False,
                episode_sidecar_detached_wall_time=time.time(),
                episode_sidecar_detached_monotonic_time=time.monotonic(),
                episode_sidecar_detach_reason=str(reason),
            )

            def close_recorder() -> Mapping[str, Any] | None:
                close_started_at = time.perf_counter()
                try:
                    recorder.close(timeout_sec=5.0)
                    snapshot = recorder.snapshot()
                    result = {
                        "closed": bool(snapshot.get("closed", False)),
                        "enqueue_drop_count": int(snapshot.get("enqueue_drop_count", 0) or 0),
                        "writer_error": snapshot.get("writer_error"),
                        "elapsed_ms": (time.perf_counter() - close_started_at) * 1000.0,
                    }
                    _log(
                        "ControlDebug",
                        "episode audit closed "
                        f"reason={reason} output={recorder.output_dir} "
                        f"drops={result['enqueue_drop_count']} "
                        f"writer_error={result['writer_error']} "
                        f"elapsed={result['elapsed_ms']:.1f}ms",
                    )
                    return result
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "closed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_ms": (time.perf_counter() - close_started_at) * 1000.0,
                    }
                    _log(
                        "ControlDebug",
                        "WARNING: episode audit close failed: "
                        f"{type(exc).__name__}: {exc} elapsed={result['elapsed_ms']:.1f}ms",
                    )
                    return result

            return close_recorder

        def start_position_episode_debug_close(
            close_recorder: Callable[[], Mapping[str, Any] | None],
        ) -> None:
            close_thread = threading.Thread(
                target=close_recorder,
                name="position-debug-episode-close",
                daemon=True,
            )
            position_debug_close_threads.append(close_thread)
            close_thread.start()

        def start_recording(now: float, *, start_reason: str) -> bool:
            nonlocal episode_last_state, last_collection_frequency_debug_at, last_collection_frequency_warning_active
            if recording_active():
                return False
            last_collection_frequency_warning_active = False
            last_collection_frequency_debug_at = 0.0
            if dataset_recorder is None and not episode_recorder.config.enabled:
                message = "正式 dataset collection 和 debug trajectory recording 都不可用。"
                _log("Dataset", message, now=now)
                if control_ui is not None:
                    control_ui.append_message("Dataset", message)
                return False
            if dataset_recorder is not None:
                try:
                    dataset_recorder.reconfigure_for_current_args()
                except Exception as exc:  # noqa: BLE001
                    message = f"正式 dataset 重新配置失败: {type(exc).__name__}: {exc}"
                    _log("Dataset", message, now=now)
                    if control_ui is not None:
                        control_ui.append_message("Dataset", message)
                    return False
                planned_episode_index = int(dataset_recorder.writer.next_episode_index)
                # Attach before metadata is frozen and before the dataset starts
                # accepting samples.  The recorded metadata therefore describes
                # the actual sidecar state/path instead of only the request.
                start_position_episode_debug(planned_episode_index)
                metadata = _build_episode_metadata(
                    args,
                    components,
                    start_reason=start_reason,
                )
                metadata.update(
                    {
                        "formal_dataset": True,
                        "dataset_root": str(dataset_recorder.dataset_root),
                        "dataset_base_root": str(getattr(args, "dataset_base_root", "")),
                        "dataset_dir_name": str(getattr(args, "dataset_dir_name", "")),
                        "dataset_name": str(args.dataset_name),
                        "task_id": str(args.collection_task_id),
                        "task_name": str(args.collection_task_name),
                        "task_name_zh": str(getattr(args, "collection_task_name_zh", "")),
                        "task_description": str(args.collection_task_description),
                        "camera_keys": sorted(camera_sources),
                        "camera_config_path": None if args.no_camera else str(args.camera_config),
                        "write_video": bool(args.write_video and not args.no_camera),
                        "tactile_storage": str(args.tactile_storage),
                        "tactile_compression": str(getattr(args, "tactile_compression", "none")),
                        "storage_queue_size": int(args.storage_queue_size),
                    }
                )
                try:
                    episode_index = dataset_recorder.start_episode(metadata, now=now)
                except Exception as exc:  # noqa: BLE001
                    if position_debug_episode_recorder is not None:
                        audit_close = stop_position_episode_debug(
                            reason="dataset_start_failed"
                        )
                        if audit_close is not None:
                            start_position_episode_debug_close(audit_close)
                    if bool(args.control_debug_config.enabled):
                        update_control_debug_runtime_metadata(
                            episode_sidecar_error=(
                                f"dataset_start_failed:{type(exc).__name__}: {exc}"
                            )
                        )
                    message = f"正式 dataset episode 启动失败: {type(exc).__name__}: {exc}"
                    _log("Dataset", message, now=now)
                    if control_ui is not None:
                        control_ui.append_message("Dataset", message)
                    return False
                if int(episode_index) != planned_episode_index:
                    mismatch = (
                        "dataset episode index changed after audit attachment: "
                        f"planned={planned_episode_index} actual={int(episode_index)}"
                    )
                    update_control_debug_runtime_metadata(
                        episode_sidecar_error=mismatch,
                    )
                    _log("ControlDebug", f"WARNING: {mismatch}", now=now)
                _log(
                    "Dataset",
                    f"recording started episode={episode_index} root={dataset_recorder.dataset_root}",
                    now=now,
                )
                if control_ui is not None:
                    control_ui.append_message(
                        "Dataset",
                        f"正式数采开始：episode {episode_index} -> {dataset_recorder.dataset_root}",
                    )
            if episode_recorder.config.enabled:
                episode_recorder.root_dir = _episode_root_dir(args, components)
                _start_episode_recording(
                    episode_recorder,
                    args,
                    components,
                    now=now,
                    start_reason=start_reason,
                )
            episode_last_state = None
            clutch_input.set_episode_recording_state(
                active=True,
                episode_dir=recording_path(),
                frame_count=recording_frame_count(),
                frequency_status=recording_frequency_status(),
                event_status="recording",
                event_message=(
                    "正式数采正在写入；结束前请确认图像、触觉和机械臂状态正常。"
                    if dataset_recorder is not None
                    else "Debug trajectory 正在写入。"
                ),
                episode_index=(
                    int(dataset_recorder.active_episode_index)
                    if dataset_recorder is not None
                    and getattr(dataset_recorder, "active_episode_index", None) is not None
                    else None
                ),
            )
            reset_gripper_warmup_probe("recording_started")
            return True

        def schedule_recording_countdown(now: float, *, start_reason: str) -> bool:
            nonlocal episode_start_countdown_deadline, episode_start_countdown_reason
            nonlocal episode_start_warmup_ready_since, episode_start_warmup_last_message
            if recording_active():
                return False
            if episode_start_countdown_deadline is not None:
                return False
            if dataset_recorder is not None and dataset_recorder.finalizing:
                message = "上一个正式 dataset episode 仍在后台保存，暂不开始新的数采。"
                _log("Dataset", message, now=now)
                if control_ui is not None:
                    control_ui.append_message("Dataset", message)
                return False
            if dataset_recorder is None and not episode_recorder.config.enabled:
                message = "正式 dataset collection 和 debug trajectory recording 都不可用。"
                _log("Dataset", message, now=now)
                if control_ui is not None:
                    control_ui.append_message("Dataset", message)
                return False
            clutch_input._restore_gripper_task_policy_for_collection()
            countdown_sec = float(DEFAULT_EPISODE_START_COUNTDOWN_SEC)
            episode_start_countdown_deadline = float(now) + countdown_sec
            episode_start_countdown_reason = str(start_reason)
            episode_start_warmup_ready_since = None
            episode_start_warmup_last_message = None
            reset_gripper_warmup_probe(f"countdown:{start_reason}")
            message = (
                f"准备开始数采：warmup 与 {countdown_sec:.1f} 秒倒计时同时进行；"
                "两者都完成后开始写入。"
            )
            clutch_input.set_episode_start_countdown_state(remaining_sec=countdown_sec, message=message)
            _log("Dataset", f"recording warmup/countdown started delay={countdown_sec:.1f}s reason={start_reason}", now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", message)
            return True

        def cancel_recording_countdown(now: float, *, reason: str) -> bool:
            nonlocal episode_start_countdown_deadline, episode_start_warmup_ready_since
            nonlocal episode_start_warmup_last_message
            if episode_start_countdown_deadline is None:
                return False
            episode_start_countdown_deadline = None
            episode_start_warmup_ready_since = None
            episode_start_warmup_last_message = None
            reset_gripper_warmup_probe(f"countdown_cancelled:{reason}")
            message = f"数采倒计时已取消：{reason}。"
            _log("Dataset", message, now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", message)
            clutch_input.set_episode_recording_state(
                active=False,
                episode_dir=recording_path(),
                frame_count=recording_frame_count(),
                event_status="idle",
                event_message="数采倒计时已取消；点击“开始数采”或按 A 重新开始。",
            )
            return True

        def advance_recording_countdown(
            now: float,
            *,
            step_result: SessionStepResult | None,
        ) -> None:
            nonlocal episode_start_countdown_deadline, episode_start_warmup_ready_since
            nonlocal episode_start_warmup_last_message
            if episode_start_countdown_deadline is None:
                return
            if recording_active():
                episode_start_countdown_deadline = None
                episode_start_warmup_ready_since = None
                episode_start_warmup_last_message = None
                return
            ready, warmup_message = recording_warmup_status(now, step_result=step_result)
            remaining_sec = _episode_start_countdown_remaining(now, episode_start_countdown_deadline)
            if not ready:
                episode_start_warmup_ready_since = None
                if warmup_message != episode_start_warmup_last_message:
                    episode_start_warmup_last_message = warmup_message
                    _log("Dataset", f"recording warmup waiting: {warmup_message}", now=now)
                countdown_text = (
                    f"倒计时还需 {remaining_sec:.1f}s。"
                    if remaining_sec > 0.0
                    else "倒计时已完成，warmup 完成后立即开始。"
                )
                clutch_input.set_episode_start_countdown_state(
                    remaining_sec=remaining_sec,
                    message=f"数采 warmup 等待中：{warmup_message}；{countdown_text}",
                )
                return
            if episode_start_warmup_ready_since is None:
                episode_start_warmup_ready_since = float(now)
                _log("Dataset", "recording warmup ready; waiting for countdown if needed", now=now)
            if not _episode_start_countdown_ready_to_record(
                now=now,
                deadline=episode_start_countdown_deadline,
                warmup_ready=True,
            ):
                clutch_input.set_episode_start_countdown_state(
                    remaining_sec=remaining_sec,
                    message=f"数采准备中：warmup 已就绪，倒计时还需 {remaining_sec:.1f}s。",
                )
                return
            start_reason = episode_start_countdown_reason
            episode_start_countdown_deadline = None
            episode_start_warmup_ready_since = None
            episode_start_warmup_last_message = None
            _log("Dataset", "recording warmup finished; starting collection", now=now)
            if control_ui is not None:
                control_ui.append_message("Dataset", "warmup 完成，正式开始数采写入。")
            if not start_recording(now, start_reason=start_reason):
                clutch_input.set_episode_recording_state(
                    active=False,
                    episode_dir=recording_path(),
                    frame_count=recording_frame_count(),
                    event_status="error",
                    event_message="数采启动失败；请查看 Dataset 日志后重新点击“开始数采”。",
                )

        def stop_recording(
            now: float,
            *,
            end_reason: str,
            summary: Mapping[str, object] | None = None,
        ) -> None:
            nonlocal episode_last_state, last_collection_frequency_warning_active
            nonlocal episode_recording_paused, episode_recording_pause_reason
            nonlocal episode_resume_warmup_ready_count
            stopped_any = False
            frame_count = recording_frame_count()
            advance_round_trip_after_stop = False
            if dataset_recorder is not None and dataset_recorder.is_recording:
                audit_close = stop_position_episode_debug(reason=end_reason)
                impedance_finalize_deferred_launch = _arm_control_mode_arg(args) == "impedance"
                if audit_close is not None and not impedance_finalize_deferred_launch:
                    # Preserve the prior position-mode ordering and behavior:
                    # diagnostic close remains independent of its writer path.
                    start_position_episode_debug_close(audit_close)
                task = dataset_recorder.stop_episode(
                    reason=end_reason,
                    step_index=step_index,
                    last_state=episode_last_state,
                    # In force/impedance mode, launch finalization after the
                    # stop-request control frame has produced its final normal
                    # command. The worker remains asynchronous: recording
                    # finalization must never hold tracker/IK updates.
                    defer_finalize_start=impedance_finalize_deferred_launch,
                    # Serializing the observational sidecar close with the
                    # writer prevents the two post-stop Python finalizers from
                    # competing with each other or the real-time loop.
                    before_finalize=(
                        audit_close if impedance_finalize_deferred_launch else None
                    ),
                )
                if (
                    audit_close is not None
                    and task is None
                    and impedance_finalize_deferred_launch
                ):
                    # Empty impedance episodes have no dataset task to own the
                    # detached audit close, so retain a small standalone close.
                    start_position_episode_debug_close(audit_close)
                stopped_any = True
                if task is None:
                    message = "正式数采已停止：当前 episode 没有样本，已丢弃。"
                else:
                    message = (
                        f"正式数采已停止：episode {task.episode_index} "
                        f"{task.frame_count} 帧，正在后台收尾。"
                    )
                    advance_round_trip_after_stop = True
                _log("Dataset", message, now=now)
                if control_ui is not None:
                    control_ui.append_message("Dataset", message)
            if episode_recorder.is_recording:
                _stop_episode_recording(
                    episode_recorder,
                    now=now,
                    end_reason=end_reason,
                    summary={} if summary is None else dict(summary),
                )
                stopped_any = True
            if stopped_any:
                event_status = "idle"
                event_message = "数采已停止。"
                stopped_episode_index: int | None = None
                stopped_frame_count: int | None = frame_count
                if dataset_recorder is not None:
                    if "task" in locals() and task is not None:
                        event_status = "finalizing"
                        event_message = (
                            f"正式数采已停止：episode {task.episode_index} "
                            f"{task.frame_count} 帧，正在后台保存。"
                        )
                        stopped_episode_index = int(task.episode_index)
                        stopped_frame_count = int(task.frame_count)
                    elif "task" in locals() and task is None:
                        event_status = "discarded"
                        event_message = "正式数采已停止：当前 episode 没有样本，已丢弃。"
                clutch_input.set_episode_recording_state(
                    active=False,
                    episode_dir=recording_path(),
                    frame_count=stopped_frame_count,
                    event_status=event_status,
                    event_message=event_message,
                    episode_index=stopped_episode_index,
                )
                episode_last_state = None
                last_collection_frequency_warning_active = False
                episode_recording_paused = False
                episode_recording_pause_reason = ""
                episode_resume_warmup_ready_count = 0
                reset_gripper_warmup_probe(f"recording_stopped:{end_reason}")
                if advance_round_trip_after_stop:
                    clutch_input.advance_round_trip_task_after_saved_episode()

        def pause_recording(now: float, *, reason: str) -> bool:
            nonlocal episode_recording_paused, episode_recording_pause_reason
            nonlocal last_collection_frequency_warning_active
            nonlocal episode_resume_warmup_ready_count
            if not recording_active():
                return False
            episode_resume_warmup_ready_count = 0
            reset_gripper_warmup_probe(f"recording_paused:{reason}")
            newly_paused = False
            if dataset_recorder is not None and dataset_recorder.is_recording:
                newly_paused = bool(
                    dataset_recorder.pause_episode(
                        reason=reason,
                        now=now,
                        step_index=step_index,
                        last_state=episode_last_state,
                    )
                ) or newly_paused
            if episode_recorder.is_recording and not episode_recording_paused:
                episode_recorder.record_event(
                    "recording_paused",
                    payload={
                        "reason": str(reason),
                        "step_index": int(step_index),
                        "last_state": episode_last_state,
                        "frame_count": int(episode_recorder.frame_count),
                    },
                    timestamp=now,
                )
                episode_recording_paused = True
                episode_recording_pause_reason = str(reason)
                newly_paused = True
            if not recording_paused():
                return False
            last_collection_frequency_warning_active = False
            message = (
                f"数采已暂停：{reason}；当前 episode 保持打开，恢复遥操且输入稳定后会继续写入。"
            )
            if newly_paused:
                _log("Dataset", message, now=now)
                if control_ui is not None:
                    control_ui.append_message("Dataset", message)
            clutch_input.set_episode_recording_state(
                active=True,
                episode_dir=recording_path(),
                frame_count=recording_frame_count(),
                frequency_status=recording_frequency_status(),
                event_status="paused",
                event_message=message,
                episode_index=(
                    int(dataset_recorder.active_episode_index)
                    if dataset_recorder is not None
                    and getattr(dataset_recorder, "active_episode_index", None) is not None
                    else None
                ),
            )
            if control_ui is not None:
                control_ui.publish(
                    active_controls=(
                        clutch_input._hold_controls()
                        if clutch_input.hold_active()
                        else clutch_input._teleop_controls()
                    )
                )
            return True

        def resume_recording_if_ready(
            now: float,
            *,
            step_result: SessionStepResult,
        ) -> bool:
            nonlocal episode_recording_paused, episode_recording_pause_reason
            nonlocal episode_resume_warmup_ready_count
            if not recording_paused():
                return True
            ready, warmup_message = recording_warmup_status(now, step_result=step_result)
            if not ready:
                episode_resume_warmup_ready_count = 0
                pause_reason = recording_pause_reason()
                clutch_input.set_episode_recording_state(
                    active=True,
                    episode_dir=recording_path(),
                    frame_count=recording_frame_count(),
                    frequency_status=recording_frequency_status(),
                    event_status="resume_pending",
                    event_message=(
                        f"数采暂停中：{pause_reason or 'paused'}；"
                        f"等待恢复写入：{warmup_message}"
                    ),
                    episode_index=(
                        int(dataset_recorder.active_episode_index)
                        if dataset_recorder is not None
                        and getattr(dataset_recorder, "active_episode_index", None) is not None
                        else None
                    ),
                )
                return False
            episode_resume_warmup_ready_count += 1
            if episode_resume_warmup_ready_count < episode_resume_warmup_required_frames:
                pause_reason = recording_pause_reason()
                clutch_input.set_episode_recording_state(
                    active=True,
                    episode_dir=recording_path(),
                    frame_count=recording_frame_count(),
                    frequency_status=recording_frequency_status(),
                    event_status="resume_pending",
                    event_message=(
                        f"数采暂停中：{pause_reason or 'paused'}；"
                        "遥操输入已恢复，等待连续稳定帧 "
                        f"{episode_resume_warmup_ready_count}/{episode_resume_warmup_required_frames}。"
                    ),
                    episode_index=(
                        int(dataset_recorder.active_episode_index)
                        if dataset_recorder is not None
                        and getattr(dataset_recorder, "active_episode_index", None) is not None
                        else None
                    ),
                )
                return False
            resumed_any = False
            pause_reason = recording_pause_reason()
            if dataset_recorder is not None and dataset_recorder.is_paused:
                resumed_any = bool(
                    dataset_recorder.resume_episode(
                        reason="runtime_running_warmup_ready",
                        now=now,
                        step_index=step_index,
                    )
                ) or resumed_any
            if episode_recording_paused:
                episode_recorder.record_event(
                    "recording_resumed",
                    payload={
                        "reason": "runtime_running_warmup_ready",
                        "pause_reason": pause_reason,
                        "step_index": int(step_index),
                        "state": step_result.state.value,
                        "frame_count": int(episode_recorder.frame_count),
                    },
                    timestamp=now,
                )
                episode_recording_paused = False
                episode_recording_pause_reason = ""
                resumed_any = True
            if resumed_any:
                episode_resume_warmup_ready_count = 0
                reset_gripper_warmup_probe("recording_resumed")
                message = "数采已恢复：遥操 RUNNING 且输入/相机 warmup 已稳定，继续写入同一个 episode。"
                _log("Dataset", message, now=now)
                if control_ui is not None:
                    control_ui.append_message("Dataset", message)
            clutch_input.set_episode_recording_state(
                active=True,
                episode_dir=recording_path(),
                frame_count=recording_frame_count(),
                frequency_status=recording_frequency_status(),
                event_status="recording",
                event_message="正式数采正在写入；结束前请确认图像、触觉和机械臂状态正常。",
                episode_index=(
                    int(dataset_recorder.active_episode_index)
                    if dataset_recorder is not None
                    and getattr(dataset_recorder, "active_episode_index", None) is not None
                    else None
                ),
            )
            return True

        def poll_dataset_finalize(now: float) -> None:
            if dataset_recorder is None:
                return
            for task in dataset_recorder.poll_finalize_tasks():
                started_at = task.started_at if task.started_at is not None else task.queued_at
                elapsed = 0.0 if task.completed_at is None else task.completed_at - started_at
                queue_delay = max(0.0, started_at - task.queued_at)
                stage_text = " ".join(
                    f"{name}={elapsed_ms:.1f}ms"
                    for name, elapsed_ms in sorted(task.stage_timings_ms.items())
                )
                if task.success:
                    message = (
                        f"正式数据集 episode {task.episode_index} 已保存 "
                        f"({task.frame_count} 帧，后台收尾 {elapsed:.2f}s，"
                        f"启动延后 {queue_delay:.3f}s"
                        f"{('，' + stage_text) if stage_text else ''})。"
                    )
                    _log("Dataset", message, now=now)
                    if control_ui is not None:
                        control_ui.append_message("Dataset", message)
                    clutch_input.set_episode_finalize_state(
                        status="saved",
                        message=message,
                        episode_index=int(task.episode_index),
                        frame_count=int(task.frame_count),
                    )
                else:
                    message = f"正式数据集 episode {task.episode_index} 保存失败: {task.error}"
                    _log("Dataset", message, now=now)
                    if control_ui is not None:
                        control_ui.append_message("Dataset", message)
                    clutch_input.set_episode_finalize_state(
                        status="error",
                        message=message,
                        episode_index=int(task.episode_index),
                        frame_count=int(task.frame_count),
                    )

        if args.record_auto_start:
            schedule_recording_countdown(time.time(), start_reason="auto_start")

        main_loop_watchdog.start()
        emergency_stop_sent = False
        while True:
            loop_start = time.time()
            main_loop_watchdog.beat("loop.start", step_index)
            loop_phase_timer = (
                PhaseTimer()
                if (
                    debug_timing_modules.needs_loop_phase_timer
                    or recording_active()
                    or action_debug_log_recorder.active
                    or position_audit_fanout is not None
                )
                else None
            )

            def mark_loop_phase(name: str) -> None:
                main_loop_watchdog.beat(name, step_index)
                if loop_phase_timer is not None:
                    loop_phase_timer.mark(name)

            clutch_input.poll()
            poll_dataset_finalize(loop_start)
            _handle_ik_debug_log_toggle_request(
                args=args,
                components=components,
                clutch_input=clutch_input,
                ik_debug_log_recorder=ik_debug_log_recorder,
                control_ui=control_ui,
                ik_debug_profile_base_enabled=ik_debug_profile_base_enabled,
                now=loop_start,
            )
            if clutch_input.consume_camera_visualization_toggle_request():
                if camera_preview_active or camera_previewer is not None:
                    close_camera_preview("图像窗口已关闭")
                    if control_ui is not None:
                        control_ui.append_message("Camera", "图像窗口已关闭。")
                else:
                    open_camera_preview(loop_start)
            if clutch_input.consume_tactile_visualization_toggle_request():
                if tactile_visualizer is not None:
                    close_tactile_preview("触觉窗口已关闭")
                    if control_ui is not None:
                        control_ui.append_message("TactileVis", "触觉窗口已关闭。")
                else:
                    open_tactile_preview(loop_start)
            if clutch_input.should_quit:
                if not emergency_stop_sent:
                    _emergency_stop_real_arms(
                        components,
                        reason="quit_requested",
                        now=loop_start,
                    )
                    emergency_stop_sent = True
                cancel_recording_countdown(loop_start, reason="quit_requested")
                if recording_active():
                    stop_recording(
                        loop_start,
                        end_reason="quit_requested",
                        summary={
                            "last_step_index": step_index,
                            "last_state": episode_last_state,
                        },
                    )
                break
            if clutch_input.consume_hold_request():
                hold_reason = clutch_input.hold_request_reason()
                if hold_reason == "q6_gesture":
                    hold_hardware_reason = "q6 gesture hold requested"
                elif hold_reason == "tracker_loss_timeout":
                    hold_hardware_reason = "tracker target loss requires runtime home recalibration"
                else:
                    hold_hardware_reason = "gui hold requested"
                if recording_active():
                    pause_recording(loop_start, reason="hold_requested")
                special_cancel_messages = _cancel_special_arm_modes_for_hold(
                    components,
                    now=loop_start,
                    reason="hold_requested",
                )
                cancel_recording_countdown(loop_start, reason="hold_requested")
                recovery_plan = None
                if not components.session.home_recalibration_required:
                    components.session.hold_hardware(hold_hardware_reason)
                _disengage_position_backends_for_manual_interaction(
                    components,
                    now=loop_start,
                    reason=hold_hardware_reason,
                )
                _log(
                    "Hold",
                    f"{hold_hardware_reason}; waiting for explicit runtime home calibration",
                    now=loop_start,
                )
                if control_ui is not None:
                    if hold_reason == "q6_gesture":
                        message = "机械臂已进入保持状态；松开双手后三维到释放区间 1 秒，再次握紧 1 秒后会重新采集 runtime home。"
                    elif hold_reason == "tracker_loss_timeout":
                        message = "Tracker 目标持续丢失已进入手动 Hold；必须点击“开始遥操”重新采集 runtime home 后才能恢复。"
                    else:
                        message = "机械臂已进入保持状态；点击“开始遥操”后才会重新采集 runtime home。"
                    control_ui.append_message("Hold", message)
                    if special_cancel_messages:
                        control_ui.append_message(
                            "Hold",
                            "；".join(special_cancel_messages)
                            + "；重新采集 runtime home 后恢复普通遥操。",
                        )
                continue
            if clutch_input.consume_calibration_request():
                if recording_active():
                    pause_recording(loop_start, reason="runtime_recalibration_requested")
                special_cancel_messages = _cancel_special_arm_modes_for_hold(
                    components,
                    now=loop_start,
                    reason="runtime_recalibration_requested",
                )
                cancel_recording_countdown(loop_start, reason="runtime_recalibration_requested")
                recovery_plan = None
                if components.session.state_machine.state != TeleopRunState.DEBUG_ONLY:
                    components.session.hold_hardware("runtime home recalibration requested")
                _disengage_position_backends_for_manual_interaction(
                    components,
                    now=loop_start,
                    reason="runtime home recalibration requested",
                )
                _log("Calibration", "explicit runtime home recalibration requested from hold", now=loop_start)
                if control_ui is not None and special_cancel_messages:
                    control_ui.append_message(
                        "Calibration",
                        "；".join(special_cancel_messages)
                        + "；本次 runtime home 后特殊状态保持关闭。",
                    )
                if not clutch_input.before_calibration_start_delay(
                    DEFAULT_RUNTIME_HOME_START_DELAY_SEC,
                    arm_mode="real",
                ):
                    continue
                try:
                    _runtime_home_calibration(
                        components,
                        arm_mode="real",
                        timeout_sec=args.calibration_timeout,
                        num_samples=args.calibration_samples,
                        progress_callback=clutch_input.publish_runtime_home_progress,
                    )
                except Exception as exc:  # noqa: BLE001
                    message = _runtime_home_failure_message(exc)
                    _log("Calibration", message, now=loop_start)
                    clutch_input.publish_calibration_waiting(
                        f"{message}\n请检查 VIVE sender/tracker_role 后重新选择操作。"
                    )
                    continue
                _emit_home_alignment_logs(components)
                _arm_after_start_calibration(
                    args,
                    components,
                    clutch_input,
                    home_captured=True,
                )
                continue
            if clutch_input.consume_episode_stop_request():
                if recording_active():
                    stop_recording(
                        loop_start,
                        end_reason="manual_key_stop",
                        summary={
                            "last_step_index": step_index,
                            "last_state": episode_last_state,
                        },
                    )
                else:
                    cancel_recording_countdown(loop_start, reason="manual_key_stop")
            if clutch_input.consume_episode_start_request():
                if not recording_active():
                    schedule_recording_countdown(loop_start, start_reason="manual_key_start")
            mark_loop_phase("loop.keys")

            if (
                clutch_input.hold_active()
                and components.session.state_machine.state == TeleopRunState.DEBUG_ONLY
            ):
                if control_ui is not None:
                    control_ui.publish(
                        stage="Hold / 停臂保持",
                        active_controls=clutch_input._hold_controls(),
                        control_hint="机械臂已保持停止；可手动调整机械臂，之后重新采集 runtime Home 再遥操。",
                        capture_stage="runtime_home",
                        capture_phase="ready",
                        teleop_running=False,
                        capturing=False,
                        capture_waiting_for_samples=False,
                        debug_ik_timing=clutch_input.ik_debug_timing_enabled(),
                        debug_ik_logging=clutch_input._ik_debug_logging_active,
                        ik_debug_log_path=clutch_input._ik_debug_log_path,
                        gripper_control_enabled=False,
                        **_movement_multiple_ui_state(args),
                        **_fix_arm_ui_state(components),
                        **_init_arm_ui_state(components),
                    )
                if loop_phase_timer is not None:
                    mark_loop_phase("loop.hold_idle")
                clutch_input.publish_camera_status(
                    camera_sources=camera_sources,
                    preview_camera_sources=preview_camera_sources,
                    camera_previewer=camera_previewer,
                    now=loop_start,
                )
                time.sleep(max(0.0, min(0.02, (1.0 / outer_control_hz) - (time.time() - loop_start))))
                continue

            _sync_dual_arm_retargeter_for_real_control(
                components,
                reset_tracking=False,
            )
            mark_loop_phase("loop.sync_preview_before")
            _sync_real_execution_step_limits(components, now=loop_start)
            mark_loop_phase("loop.sync_step_limits")

            _guard_impedance_force_control(
                args,
                components,
                context="main_loop",
                now=loop_start,
            )
            mark_loop_phase("loop.impedance_guard")

            phase_recorder = loop_phase_timer.record if loop_phase_timer is not None else None
            step_result = components.session.run_step(
                now=loop_start,
                phase_recorder=phase_recorder,
            )
            mark_loop_phase("loop.session_run_step")
            raw_step_channels = (
                step_result.raw_channels
                if step_result.raw_channels
                else step_result.channels
            )
            clutch_input.publish_raw_tracker_pose_state(
                channels=raw_step_channels,
                now=loop_start,
            )
            if (
                step_result.home_recalibration_required
                and not clutch_input.hold_active()
            ):
                tracker_hold_payload = {
                    "step_index": int(step_index),
                    "teleop_state": str(step_result.state.value),
                    "motion_hold_mode": str(step_result.motion_hold_mode),
                    "motion_hold_reason": str(step_result.motion_hold_reason),
                    "tracker_loss_duration_sec": float(
                        step_result.tracker_loss_duration_sec
                    ),
                    "tracker_loss_channels": list(step_result.tracker_loss_channels),
                    "home_recalibration_required": True,
                    "faults": _faults_log_payload(step_result),
                    "tracker_channels": {
                        name: payload
                        for name, payload in _channels_log_payload(
                            step_result,
                            now=float(loop_start),
                        ).items()
                        if name in {"left_wrist_pose", "right_wrist_pose"}
                    },
                }
                if action_debug_log_recorder.active:
                    action_debug_log_recorder.write_event(
                        "tracker_manual_hold",
                        tracker_hold_payload,
                        now=loop_start,
                    )
                if recording_active():
                    pause_recording(
                        loop_start,
                        reason="tracker_loss_manual_hold",
                    )
                _log(
                    "TrackerHold",
                    "tracker target loss exceeded manual-hold threshold; "
                    "waiting for explicit runtime home recalibration; "
                    f"channels={list(step_result.tracker_loss_channels)} "
                    f"duration={step_result.tracker_loss_duration_sec:.3f}s "
                    f"reason={step_result.motion_hold_reason}",
                    now=loop_start,
                )
                clutch_input.request_hold(reason="tracker_loss_timeout")
                continue
            gripper_command = None
            gripper_command_status: dict[str, dict[str, object]] = {}
            if step_result.motion_hold_mode in {
                TRACKER_HOLD_GRACE,
                TRACKER_HOLD_AUTOMATIC,
            }:
                for side in components.active_sides:
                    gripper_command_status[side] = {
                        "status": "tracker_motion_hold",
                        "sent": False,
                        "reason": step_result.motion_hold_reason,
                        "timestamp": float(loop_start),
                    }
            elif clutch_input.gripper_control_enabled() and gripper_warmup_probe_running():
                for side in components.active_sides:
                    gripper_command_status[side] = {
                        "status": "warmup_probe",
                        "sent": False,
                        "reason": "gripper_warmup_probe",
                        "timestamp": float(loop_start),
                    }
            elif clutch_input.gripper_control_enabled():
                gripper_command = _dispatch_q6_gripper_control(
                    components,
                    step_result.observation,
                    now=loop_start,
                    status_out=gripper_command_status,
                )
            else:
                for side in components.active_sides:
                    gripper_command_status[side] = {
                        "status": "control_disabled",
                        "sent": False,
                        "timestamp": float(loop_start),
                    }
            clutch_input.publish_gripper_status(
                channels=step_result.channels,
                command_status=gripper_command_status,
                now=loop_start,
            )
            fix_arm_progress = advance_fix_arm_home_capture(
                controller=getattr(components, "fix_arm_controller", None),
                channels=raw_step_channels,
                calibration=components.session.calibration,
                now=loop_start,
                pose_from_channel=_channel_pose_in_desk,
                mean_pose=_mean_pose,
                resolve_robot_home=lambda side: _resolve_robot_home_for_side(components, side=side),
                side_retargeter=lambda side: _side_retargeter(components, side),
                tcp_mode=args.tcp_mode,
                log=lambda tag, message: _log(tag, message, now=loop_start),
            )
            if fix_arm_progress.active:
                if fix_arm_progress.completed and fix_arm_progress.side:
                    components.session.safety_monitor.request_pose_baseline_reset_many(
                        (f"{fix_arm_progress.side}_wrist_pose",)
                    )
                clutch_input.publish_fix_arm_home_capture_progress(fix_arm_progress)
            init_arm_progress = advance_init_arm_home_capture(
                controller=getattr(components, "init_arm_controller", None),
                channels=raw_step_channels,
                calibration=components.session.calibration,
                now=loop_start,
                pose_from_channel=_channel_pose_in_desk,
                mean_pose=_mean_pose,
                resolve_robot_home=lambda side: _resolve_robot_home_for_side(components, side=side),
                side_retargeter=lambda side: _side_retargeter(components, side),
                tcp_mode=args.tcp_mode,
                log=lambda tag, message: _log(tag, message, now=loop_start),
            )
            if init_arm_progress.active:
                if init_arm_progress.completed and init_arm_progress.side:
                    components.session.safety_monitor.request_pose_baseline_reset_many(
                        (f"{init_arm_progress.side}_wrist_pose",)
                    )
                    reset_smoothing = getattr(
                        getattr(components, "dual_arm_retargeter", None),
                        "reset_target_pose_smoothing",
                        None,
                    )
                    if callable(reset_smoothing):
                        try:
                            reset_smoothing()
                        except Exception as exc:
                            _log(
                                "InitArm",
                                "WARNING: retargeter target smoothing reset failed after home recapture: "
                                f"{type(exc).__name__}: {exc}",
                                now=loop_start,
                            )
                clutch_input.publish_init_arm_home_capture_progress(init_arm_progress)
            clutch_input.publish_camera_status(
                camera_sources=camera_sources,
                preview_camera_sources=preview_camera_sources,
                camera_previewer=camera_previewer,
                now=loop_start,
            )
            advance_recording_countdown(loop_start, step_result=step_result)
            mark_loop_phase("loop.gripper_control")
            active_command = step_result.command
            manual_recovery_command_dispatched = False
            hand_command = _extract_hand_command_from_command(step_result.command)
            if step_result.state == TeleopRunState.RECOVERING:
                hand_command = _build_runtime_hand_command_for_recovery(
                    components,
                    step_result.observation,
                )
                if step_result.observation is None:
                    active_command = None
                elif recovery_plan is None:
                    recovery_plan, active_command, recovery_done = _start_recovery_plan(
                        components,
                        step_result.observation,
                        now=loop_start,
                        arm_mode="real",
                        control_mode=args.control_mode,
                        ramp_sec=args.recovery_ramp_sec,
                    )
                    if not recording_active():
                        if recovery_plan is None:
                            _log(
                                "Recovery",
                                f"direct mode control_mode={args.control_mode}",
                                now=loop_start,
                            )
                        else:
                            _log(
                                "Recovery",
                                f"start mode={recovery_plan.mode} duration={recovery_plan.duration_sec:.2f}s",
                                now=loop_start,
                            )
                    active_command = _merge_runtime_hand_command(active_command, hand_command)
                    active_command = _filter_manual_real_command(
                        components,
                        active_command,
                        step_result.observation,
                        loop_start,
                    )
                    if active_command is not None:
                        command_guard_fault = components.session.evaluate_command_guard(
                            active_command,
                            step_result.observation,
                            loop_start,
                        )
                        if command_guard_fault is not None:
                            components.session.reject_command_for_guard_fault(command_guard_fault)
                            _log(
                                "Safety",
                                "recovery_start command rejected by pre-dispatch guard: "
                                f"{command_guard_fault.code}",
                                now=loop_start,
                            )
                            active_command = None
                        elif _guard_impedance_force_control(
                            args,
                            components,
                            context="recovery_start",
                            now=loop_start,
                        ):
                            assert components.session.rig is not None
                            components.session.dispatch_command(active_command)
                            manual_recovery_command_dispatched = True
                            if components.temporary_safety_limiter is not None:
                                components.temporary_safety_limiter.sync_from_command(active_command)
                        else:
                            active_command = None
                    if recovery_done:
                        components.session.finish_recovery("recovery completed")
                        recovery_plan = None
                else:
                    active_command, recovery_done = _advance_recovery_plan(
                        components,
                        step_result.observation,
                        recovery_plan,
                        now=loop_start,
                    )
                    active_command = _merge_runtime_hand_command(active_command, hand_command)
                    active_command = _filter_manual_real_command(
                        components,
                        active_command,
                        step_result.observation,
                        loop_start,
                    )
                    if active_command is not None:
                        command_guard_fault = components.session.evaluate_command_guard(
                            active_command,
                            step_result.observation,
                            loop_start,
                        )
                        if command_guard_fault is not None:
                            components.session.reject_command_for_guard_fault(command_guard_fault)
                            _log(
                                "Safety",
                                "recovery_advance command rejected by pre-dispatch guard: "
                                f"{command_guard_fault.code}",
                                now=loop_start,
                            )
                            active_command = None
                        elif _guard_impedance_force_control(
                            args,
                            components,
                            context="recovery_advance",
                            now=loop_start,
                        ):
                            assert components.session.rig is not None
                            components.session.dispatch_command(active_command)
                            manual_recovery_command_dispatched = True
                            if components.temporary_safety_limiter is not None:
                                components.temporary_safety_limiter.sync_from_command(active_command)
                        else:
                            active_command = None
                    if recovery_done:
                        components.session.finish_recovery("recovery completed")
                        if not recording_active():
                            _log("Recovery", "completed", now=loop_start)
                        recovery_plan = None
            else:
                recovery_plan = None
            mark_loop_phase("loop.recovery")
            if gripper_command is not None:
                hand_command = (
                    gripper_command
                    if hand_command is None
                    else merge_teleop_commands((hand_command, gripper_command))
                )

            collection_step_result = _collection_step_result(
                step_result,
                active_command=active_command,
                gripper_command=gripper_command,
            )
            if (
                bool(step_result.command_dispatched or manual_recovery_command_dispatched)
                and collection_step_result.command is not None
                and any(
                    name in collection_step_result.command.commands
                    for name in ("left_arm", "right_arm")
                )
            ):
                last_dispatched_main_command = _copy_teleop_command(
                    collection_step_result.command
                )
            if components.position_reference_publisher is not None:
                components.position_reference_publisher.publish_step(collection_step_result)
            mark_loop_phase("loop.publish_position_reference")

            _apply_active_arm_command_to_preview_real(components, active_command)
            mark_loop_phase("loop.preview_apply")
            _sync_dual_arm_retargeter_for_real_control(
                components,
                reset_tracking=False,
            )
            mark_loop_phase("loop.sync_preview_after")

            if args.print_tactile_sum_only:
                tactile_snapshot = (
                    _hand_tactile_sum(components.left_hand_backend),
                    _hand_tactile_sum(components.right_hand_backend),
                )
                if tactile_sum_gate.should_emit(
                    "tactile_sum_only_status",
                    tactile_snapshot,
                    now=loop_start,
                    heartbeat_sec=0.1,
                ):
                    _emit_tactile_sum_only_status(
                        components,
                        now=loop_start,
                        stream=console_stream,
                    )
            mark_loop_phase("loop.tactile_sum")

            if tactile_visualizer is not None and tactile_preview_rate.should_update(loop_start):
                try:
                    if isinstance(tactile_visualizer, DMTacDistributedForceVisualizer):
                        tactile_keys = tactile_visualizer.update(
                            dmtac_force_by_side_from_backends(
                                components.active_sides,
                                lambda side: _gripper_backend_for_side(components, side),
                            )
                        )
                    else:
                        tactile_visualizer.update(
                            tactile_by_side_from_backends(
                                components.active_sides,
                                lambda side: _hand_backend_for_side(components, side),
                            )
                        )
                        tactile_keys = []
                    if any(key in {"q", "\x1b"} for key in tactile_keys):
                        close_tactile_preview("触觉窗口已关闭")
                except Exception as exc:  # noqa: BLE001
                    close_tactile_preview(f"触觉窗口刷新失败: {type(exc).__name__}: {exc}")
                    if control_ui is not None:
                        control_ui.append_message("TactileVis", f"触觉窗口刷新失败：{exc}")
            mark_loop_phase("loop.tactile_visualizer")

            if camera_preview_active and camera_previewer is not None and camera_preview_rate.should_update(loop_start):
                try:
                    camera_keys = camera_previewer.update(preview_camera_sources)
                    if any(key in {"q", "\x1b"} for key in camera_keys):
                        close_camera_preview("图像窗口已关闭")
                except Exception as exc:  # noqa: BLE001
                    close_camera_preview(f"图像窗口刷新失败: {type(exc).__name__}: {exc}")
                    if control_ui is not None:
                        control_ui.append_message("Camera", f"图像窗口刷新失败：{exc}")
            mark_loop_phase("loop.camera_previewer")

            left_qpos = _state_qpos(components.left_preview_backend, DEFAULT_LEFT_ARM_QPOS)
            right_qpos = _state_qpos(components.right_preview_backend, DEFAULT_RIGHT_ARM_QPOS)
            mark_loop_phase("loop.preview_state")
            if components.renderer is not None:
                components.renderer.set_debug_frames(
                    _build_renderer_debug_frames(
                        components,
                        channels=step_result.channels,
                    )
                )
                components.renderer.set_debug_boxes(_build_renderer_debug_boxes(components))
                components.renderer.render_data(
                    left_qpos_arm=left_qpos,
                    left_qpos_hand=_real_hand_render_qpos(components.left_hand_backend, "left"),
                    right_qpos_arm=right_qpos,
                    right_qpos_hand=_real_hand_render_qpos(components.right_hand_backend, "right"),
                    capture_image=False,
                )
            mark_loop_phase("loop.render")
            if bool(args.debug_orientation_console) and not recording_active():
                _emit_runtime_orientation_logs(
                    components,
                    channels=step_result.channels,
                    now=loop_start,
                    gate=orientation_gate,
                )
            mark_loop_phase("loop.orientation_logs")

            fault_codes = tuple(_runtime_fault_label(fault) for fault in step_result.faults)
            channel_valid = tuple(
                sorted((name, bool(channel.valid)) for name, channel in step_result.channels.items())
            )
            hand_input_status = _hand_channel_status(components, step_result.channels)
            gripper_status = _gripper_runtime_status_signature(
                components,
                channels=step_result.channels,
                command_status=gripper_command_status,
                now=loop_start,
            )
            gripper_health_status = _gripper_runtime_health_signature(gripper_status)
            clutch_channels = components.keyboard_source.get_latest_channels()
            left_clutch_channel = clutch_channels.get("left_clutch_button")
            right_clutch_channel = clutch_channels.get("right_clutch_button")
            left_clutch = bool(left_clutch_channel.value) if left_clutch_channel is not None else False
            right_clutch = bool(right_clutch_channel.value) if right_clutch_channel is not None else False
            position_reference_enabled = components.position_reference_publisher is not None
            sample_due = sample_rate_gate.is_due(time.monotonic())
            auto_stop_reason = recording_auto_stop_reason(step_result.state)
            if auto_stop_reason is not None:
                cancel_recording_countdown(loop_start, reason=auto_stop_reason)
                if recording_active():
                    stop_recording(
                        loop_start,
                        end_reason=auto_stop_reason,
                        summary={
                            "last_step_index": step_index,
                            "last_state": step_result.state.value,
                        },
                    )
            if recording_active() and recording_paused():
                resume_recording_if_ready(loop_start, step_result=step_result)
            if recording_active():
                if episode_last_state != step_result.state.value:
                    if episode_recorder.is_recording:
                        episode_recorder.record_event(
                            "state_change",
                            payload={
                                "old_state": episode_last_state,
                                "new_state": step_result.state.value,
                            },
                            timestamp=loop_start,
                        )
                    episode_last_state = step_result.state.value
                if (
                    dataset_recorder is not None
                    and dataset_recorder.is_recording
                    and dataset_recorder.can_record_samples
                    and sample_due
                ):
                    collection_metadata_extra = None
                    if loop_phase_timer is not None:
                        collection_metadata_extra = {
                            "loop_phase_timings_ms": [
                                {"name": name, "elapsed_ms": elapsed * 1000.0}
                                for name, elapsed in loop_phase_timer.phases
                            ],
                            "loop_elapsed_before_record_s": loop_phase_timer.elapsed_sec(),
                        }
                    if position_reference_enabled:
                        dataset_step_result = replace(
                            collection_step_result,
                            # ``collection_step_result.timestamp`` is T0: the
                            # action/control information cut.  Sample assembly
                            # may execute asynchronously after this point, so
                            # do not move T0 to the later record call.
                            timestamp=float(collection_step_result.timestamp),
                            action_record=None,
                        )
                    elif last_dispatched_main_command is not None:
                        dataset_step_result = _collection_sample_step_result(
                            collection_step_result,
                            dispatched_command=last_dispatched_main_command,
                        )
                        collection_metadata_extra = {
                            **dict(collection_metadata_extra or {}),
                            "collection_action_source": "latest_dispatched_main_command",
                            "collection_action_timestamp": float(
                                last_dispatched_main_command.timestamp
                            ),
                            "collection_action_age_sec": max(
                                0.0,
                                float(collection_step_result.timestamp)
                                - float(last_dispatched_main_command.timestamp),
                            ),
                        }
                    else:
                        dataset_step_result = None
                    if dataset_step_result is not None:
                        sample_accepted = dataset_recorder.record_step_async(
                            dataset_step_result,
                            include_debug_metadata=bool(ik_debug_log_recorder.active),
                            include_action_debug_metadata=bool(getattr(args, "debug_action", False)),
                            metadata_extra=collection_metadata_extra,
                        )
                        overload_reason = dataset_recorder.collection_overload_reason
                        if not sample_accepted and overload_reason:
                            pause_recording(
                                loop_start,
                                reason=f"collection_overload:{overload_reason}",
                            )
                if dataset_recorder is not None and not recording_paused():
                    overload_reason = dataset_recorder.poll_collection_overload()
                    if overload_reason:
                        pause_recording(
                            loop_start,
                            reason=f"collection_overload:{overload_reason}",
                        )
                frequency_status = recording_frequency_status()
                publish_collection_frequency_warning(frequency_status, now=loop_start)
                if (
                    episode_recorder.is_recording
                    and not recording_paused()
                    and sample_due
                    and (position_reference_enabled or last_dispatched_main_command is not None)
                ):
                    episode_recorder.record_frame(
                        _build_episode_frame(
                            components,
                            step_index=step_index,
                            step_result=step_result,
                            active_command=(
                                active_command
                                if position_reference_enabled
                                else last_dispatched_main_command
                            ),
                            hand_command=hand_command,
                            recovery_plan=recovery_plan,
                            left_clutch=left_clutch,
                            right_clutch=right_clutch,
                        ),
                        timestamp=loop_start,
                    )
                paused_now = recording_paused()
                clutch_input.set_episode_recording_state(
                    active=True,
                    episode_dir=recording_path(),
                    frame_count=recording_frame_count(),
                    frequency_status=frequency_status,
                    event_status="resume_pending" if paused_now else "recording",
                    event_message=(
                        f"数采暂停中：{recording_pause_reason() or 'paused'}；等待遥操恢复稳定后继续写入。"
                        if paused_now
                        else "正式数采正在写入；结束前请确认图像、触觉和机械臂状态正常。"
                    ),
                )
            mark_loop_phase("loop.episode_record")

            collision_status = None
            collision_fallback = None
            jump_gate_status = None
            branch_candidate = None
            temporary_safety_status = None
            if active_command is not None:
                collision_status = active_command.metadata.get("collision_avoidance_status")
                collision_fallback = active_command.metadata.get(
                    "collision_avoidance_fallback_reason"
                )
                jump_gate_status = active_command.metadata.get("ik_jump_gate_status")
                branch_candidate = active_command.metadata.get("ik_branch_candidate")
                temporary_safety_status = active_command.metadata.get("temporary_dual_arm_safety")
            left_hw_state = components.left_state_backend.get_state() if components.left_state_backend is not None else {}
            right_hw_state = components.right_state_backend.get_state() if components.right_state_backend is not None else {}
            left_command_state = (
                components.left_command_backend.get_state()
                if components.left_command_backend is not None
                else {}
            )
            right_command_state = (
                components.right_command_backend.get_state()
                if components.right_command_backend is not None
                else {}
            )
            if _update_init_arm_arrival_from_hardware(
                components,
                hardware_state_by_side={
                    "left": left_hw_state,
                    "right": right_hw_state,
                },
                now=loop_start,
            ):
                clutch_input.publish_init_arm_arrival_state()
            ik_warning_payload = _build_ik_warning_payload(
                active_command=active_command,
                left_command_state=left_command_state,
                right_command_state=right_command_state,
            )
            clutch_input.publish_ik_warning_state(
                ik_warning_payload,
                now=loop_start,
            )
            pose_quality_payload = _build_pose_quality_display_payload(active_command)
            clutch_input.publish_pose_quality_state(
                pose_quality_payload,
                now=loop_start,
            )
            ik_timing_payload = None
            if clutch_input.ik_debug_timing_enabled():
                ik_timing_payload = _build_ik_timing_payload(
                    active_command=active_command,
                    left_command_state=left_command_state,
                    right_command_state=right_command_state,
                )
                clutch_input.publish_ik_timing_state(
                    ik_timing_payload,
                    now=loop_start,
                )
            active_command_backend = _active_command_backend(components)
            base_control_dt = (
                float(active_command_backend.config.control_dt)
                if active_command_backend is not None
                else 1.0 / outer_control_hz
            )
            target_step_scale = (
                float(components.arm_loop_step_dt_tracker.get("smoothed_loop_dt", base_control_dt))
                / base_control_dt
            )
            if action_debug_log_recorder.should_record_frame(now=loop_start):
                try:
                    if ik_timing_payload is None:
                        ik_timing_payload = _build_ik_timing_payload(
                            active_command=active_command,
                            left_command_state=left_command_state,
                            right_command_state=right_command_state,
                        )
                    action_debug_recording_status: dict[str, object] = {
                        "active": bool(recording_active()),
                        "path": recording_path(),
                        "frame_count": int(recording_frame_count()),
                    }
                    if recording_active():
                        action_debug_recording_status["frequency"] = recording_frequency_status()
                    action_debug_log_recorder.record_frame(
                        _build_action_debug_metadata(
                            _collection_step_result(
                                step_result,
                                active_command=active_command,
                                gripper_command=gripper_command,
                            ),
                            components=components,
                            record_wall_time=time.time(),
                            step_index=step_index,
                            loop_start=loop_start,
                            left_command_state=left_command_state,
                            right_command_state=right_command_state,
                            left_hw_state=left_hw_state,
                            right_hw_state=right_hw_state,
                            target_step_scale=target_step_scale,
                            left_clutch=left_clutch,
                            right_clutch=right_clutch,
                            ik_timing_payload=ik_timing_payload,
                            loop_phases=(
                                loop_phase_timer.phases
                                if loop_phase_timer is not None
                                else None
                            ),
                            loop_elapsed_sec=(
                                loop_phase_timer.elapsed_sec()
                                if loop_phase_timer is not None
                                else None
                            ),
                            recording_status=action_debug_recording_status,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - diagnostics must not break teleop.
                    action_debug_log_recorder.write_event(
                        "frame_error",
                        {
                            "step_index": int(step_index),
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        now=loop_start,
                    )
            mark_loop_phase("loop.action_debug_log")
            if clutch_input.consume_pose_snapshot_request():
                if pose_snapshot_recorder is None:
                    message = "Pose 抽帧不可用：当前未启用 --fix-arm。"
                    clutch_input.set_pose_snapshot_state(status=message, append_message=True)
                elif recording_active() or episode_start_countdown_deadline is not None:
                    message = "Pose 抽帧与正式数采互斥；本次请求已忽略。"
                    clutch_input.set_pose_snapshot_state(status=message, append_message=True)
                else:
                    try:
                        payload = build_fix_arm_pose_snapshot_payload(
                            timestamp=loop_start,
                            monotonic_time=time.monotonic(),
                            step_index=step_index,
                            active_sides=components.active_sides,
                            teleop_state=step_result.state.value,
                            control_hz=float(step_result.control_hz),
                            movement_multiple=_movement_multiple_value(args),
                            tcp_mode=str(args.tcp_mode),
                            calibration=components.session.calibration,
                            raw_channels=raw_step_channels,
                            filtered_channels=step_result.channels,
                            observation=step_result.observation,
                            step_command=step_result.command,
                            active_command=active_command,
                            hand_command=hand_command,
                            gripper_command=gripper_command,
                            command_dispatched=bool(
                                step_result.command_dispatched or active_command is not None
                            ),
                            faults=step_result.faults,
                            fix_arm_state=_fix_arm_ui_state(components),
                            left_retargeter=components.left_retargeter,
                            right_retargeter=components.right_retargeter,
                            left_preview_qpos=left_qpos,
                            right_preview_qpos=right_qpos,
                            left_hardware_state=left_hw_state,
                            right_hardware_state=right_hw_state,
                            left_command_state=left_command_state,
                            right_command_state=right_command_state,
                            gripper_status=gripper_status,
                            pose_quality=pose_quality_payload,
                            ik_timing=ik_timing_payload,
                            extra={
                                "recording_active": False,
                                "episode_countdown_active": False,
                            },
                        )
                        record = pose_snapshot_recorder.record(payload, now=loop_start)
                        message = f"Pose 抽帧已保存 #{record.sequence}: {record.path}"
                        _log("PoseSnapshot", message, now=loop_start)
                        clutch_input.set_pose_snapshot_state(
                            last_path=record.path,
                            status=message,
                            append_message=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        message = f"Pose 抽帧保存失败: {type(exc).__name__}: {exc}"
                        _log("PoseSnapshot", message, now=loop_start)
                        clutch_input.set_pose_snapshot_state(
                            status=message,
                            append_message=True,
                        )
            if ik_debug_log_recorder.active:
                if ik_timing_payload is None:
                    ik_timing_payload = _build_ik_timing_payload(
                        active_command=active_command,
                        left_command_state=left_command_state,
                        right_command_state=right_command_state,
                    )
                ik_debug_log_recorder.record_frame(
                    _build_ik_debug_log_frame(
                        components=components,
                        step_index=step_index,
                        loop_start=loop_start,
                        step_result=step_result,
                        active_command=active_command,
                        left_command_state=left_command_state,
                        right_command_state=right_command_state,
                        left_hw_state=left_hw_state,
                        right_hw_state=right_hw_state,
                        ik_timing_payload=ik_timing_payload,
                        target_step_scale=target_step_scale,
                        left_clutch=left_clutch,
                        right_clutch=right_clutch,
                        full=bool(getattr(args, "ik_debug_log_full", False)),
                    )
                )
            hardware_snapshot = (
                left_hw_state.get("hardware_result_code"),
                right_hw_state.get("hardware_result_code"),
            )
            command_backend_snapshot = (
                _command_backend_health_signature(left_command_state),
                _command_backend_health_signature(right_command_state),
            )
            pose_quality_snapshot = (
                pose_quality_payload.get("status"),
                bool(pose_quality_payload.get("bad", False)),
                tuple(_pose_quality_reason_list(pose_quality_payload.get("reasons"))[:3]),
            )
            fix_arm_snapshot = _fix_arm_debug_state(components)
            init_arm_snapshot = _init_arm_debug_state(components)
            status_snapshot = (
                step_result.state.value,
                fault_codes,
                channel_valid,
                left_clutch,
                right_clutch,
                collision_status,
                collision_fallback,
                jump_gate_status,
                _temporary_safety_health_signature(temporary_safety_status),
                hardware_snapshot,
                command_backend_snapshot,
                pose_quality_snapshot,
                hand_input_status,
                gripper_health_status,
                clutch_input.gripper_control_enabled(),
                tuple(sorted(fix_arm_snapshot.items())),
                tuple(sorted(init_arm_snapshot.items())),
            )
            mark_loop_phase("loop.status_snapshot")
            if status_gate.should_emit(
                "real_teleop_status",
                status_snapshot,
                now=loop_start,
                heartbeat_sec=status_heartbeat_sec,
            ):
                collision_bits = ""
                if collision_status is not None:
                    collision_bits = f" collision={collision_status}"
                    if collision_fallback:
                        collision_bits += f" fallback={collision_fallback}"
                if jump_gate_status is not None:
                    collision_bits += f" ik={jump_gate_status}"
                if branch_candidate is not None:
                    collision_bits += f" branch={branch_candidate}"
                if isinstance(temporary_safety_status, Mapping):
                    temp_limited = bool(temporary_safety_status.get("limited", False))
                    collision_bits += f" temp_safety={'limited' if temp_limited else 'on'}"
                if clutch_input.ik_debug_timing_enabled():
                    collision_bits += f" {_ik_timing_status_text(ik_timing_payload)}"
                fault_reason_bits = ""
                fault_reason = getattr(components.session, "last_fault_reason", "")
                if fault_reason:
                    fault_reason_bits = f" fault_reason={_short_status_text(fault_reason, max_len=160)}"
                fix_arm_bits = ""
                fix_arm_text = _fix_arm_status_text(fix_arm_snapshot)
                if fix_arm_text:
                    fix_arm_bits = f" fix_arm={fix_arm_text.strip()}"
                init_arm_bits = ""
                init_arm_text = _init_arm_status_text(init_arm_snapshot)
                if init_arm_text:
                    init_arm_bits = f" init_arm={init_arm_text.strip()}"
                pose_quality_bits = f" {_pose_quality_status_text(pose_quality_payload)}"
                hand_bits = "" if not hand_input_status else f" hand_q6={list(hand_input_status)}"
                gripper_bits = ""
                if gripper_status:
                    gripper_bits = (
                        f" gripper_control={'on' if clutch_input.gripper_control_enabled() else 'off'}"
                        f" gripper={list(gripper_status)}"
                    )
                _log(
                    "RealTeleop",
                    f"step={step_index} hz={step_result.control_hz:.1f} "
                    f"state={step_result.state.value} "
                    f"left_hw_init={left_hw_state.get('hardware_result_code')}:{left_hw_state.get('hardware_result_text')} "
                    f"right_hw_init={right_hw_state.get('hardware_result_code')}:{right_hw_state.get('hardware_result_text')} "
                    f"left_cmd={_command_backend_status_text(left_command_state)} "
                    f"right_cmd={_command_backend_status_text(right_command_state)} "
                    f"left_step_scale={float(left_command_state.get('effective_step_scale', 1.0)):.2f} "
                    f"right_step_scale={float(right_command_state.get('effective_step_scale', 1.0)):.2f} "
                    f"target_step_scale={target_step_scale:.2f} "
                    f"left_clutch={left_clutch} "
                    f"right_clutch={right_clutch} "
                    f"faults={list(fault_codes)}"
                    f"{fault_reason_bits}"
                    f"{fix_arm_bits}"
                    f"{init_arm_bits}"
                    f"{collision_bits}"
                    f"{pose_quality_bits}"
                    f"{hand_bits}"
                    f"{gripper_bits}",
                    now=loop_start,
                )
            mark_loop_phase("loop.status_emit")
            if loop_phase_timer is not None:
                active_elapsed_sec = loop_phase_timer.elapsed_sec()
                if debug_timing_log_recorder.active:
                    timing_log_reason = debug_timing_log_recorder.select_frame_reason(
                        now=loop_start,
                        elapsed_sec=active_elapsed_sec,
                        phases=loop_phase_timer.phases,
                        state=step_result.state,
                        fault_codes=fault_codes,
                        threshold_sec=float(debug_timing_state.threshold_sec),
                    )
                    if timing_log_reason is not None:
                        if ik_timing_payload is None:
                            ik_timing_payload = _build_ik_timing_payload(
                                active_command=active_command,
                                left_command_state=left_command_state,
                                right_command_state=right_command_state,
                            )
                        debug_recording_status: dict[str, object] = {
                            "active": bool(recording_active()),
                            "path": recording_path(),
                            "frame_count": int(recording_frame_count()),
                        }
                        if recording_active():
                            debug_recording_status["frequency"] = recording_frequency_status()
                        debug_timing_log_recorder.record_frame(
                            _build_debug_timing_log_frame(
                                components=components,
                                step_index=step_index,
                                loop_start=loop_start,
                                elapsed_sec=active_elapsed_sec,
                                phases=loop_phase_timer.phases,
                                reason=timing_log_reason,
                                step_result=step_result,
                                active_command=active_command,
                                left_command_state=left_command_state,
                                right_command_state=right_command_state,
                                left_hw_state=left_hw_state,
                                right_hw_state=right_hw_state,
                                ik_timing_payload=ik_timing_payload,
                                target_step_scale=target_step_scale,
                                left_clutch=left_clutch,
                                right_clutch=right_clutch,
                                camera_sources=camera_sources,
                                preview_camera_sources=preview_camera_sources,
                                camera_previewer=camera_previewer,
                                hand_input_status=hand_input_status,
                                gripper_status=gripper_status,
                                recording_status=debug_recording_status,
                            )
                        )
                _debug_timing_maybe_report(
                    debug_timing_state,
                    reason="main_loop",
                    elapsed_sec=active_elapsed_sec,
                    phases=loop_phase_timer.phases,
                    step_index=step_index,
                    result=step_result,
                    active_command=active_command,
                    left_hw_state=left_hw_state,
                    right_hw_state=right_hw_state,
                    left_command_state=left_command_state,
                    right_command_state=right_command_state,
                    target_step_scale=target_step_scale,
                )
                if position_audit_fanout is not None:
                    try:
                        _record_position_loop_timings(
                            position_audit_fanout,
                            timer=loop_phase_timer,
                            step_index=step_index,
                            state=step_result.state,
                            outer_control_hz=outer_control_hz,
                        )
                    except Exception as exc:  # noqa: BLE001 - debug must not interrupt control.
                        _log(
                            "ControlDebug",
                            f"WARNING: loop timing audit event failed: {type(exc).__name__}: {exc}",
                            now=loop_start,
                        )

            if dataset_recorder is not None:
                for task in dataset_recorder.start_pending_finalize_tasks():
                    _log(
                        "Dataset",
                        "starting deferred episode finalization after the completed "
                        f"control frame: episode {task.episode_index}",
                        now=loop_start,
                    )

            step_index += 1
            if args.max_steps > 0 and step_index >= args.max_steps:
                break

            remaining = (1.0 / outer_control_hz) - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        shutdown_steps: list[tuple[str, Callable[[], None]]] = []
        if "main_loop_watchdog" in locals():
            shutdown_steps.append(("main_loop_watchdog", main_loop_watchdog.stop))
        if "debug_timing_log_recorder" in locals():
            shutdown_steps.append(
                (
                    "debug_timing_log_recorder",
                    lambda: debug_timing_log_recorder.stop(
                        reason="finally",
                        now=time.time(),
                    ),
                )
            )
        if "action_debug_log_recorder" in locals():
            shutdown_steps.append(
                (
                    "action_debug_log_recorder",
                    lambda: action_debug_log_recorder.stop(
                        reason="finally",
                        now=time.time(),
                    ),
                )
            )
        if "ik_debug_log_recorder" in locals():
            shutdown_steps.append(
                (
                    "ik_debug_log_recorder",
                    lambda: ik_debug_log_recorder.stop(
                        reason="finally",
                        now=time.time(),
                    ),
                )
            )
        if "recording_active" in locals() and recording_active():
            shutdown_steps.append(
                (
                    "active_recording",
                    lambda: stop_recording(
                        time.time(),
                        end_reason="shutdown",
                        summary={"reason": "finally"},
                    ),
                )
            )
        shutdown_steps.extend(
            [
                ("episode_recorder", episode_recorder.close),
                ("clutch_input", clutch_input.close),
            ]
        )
        if dataset_recorder is not None:
            shutdown_steps.append(("dataset_recorder", dataset_recorder.close))
        if camera_previewer is not None:
            shutdown_steps.append(("camera_previewer", camera_previewer.close))
        shutdown_steps.append(("camera_sources", lambda: _close_camera_sources(camera_sources)))
        if tactile_visualizer is not None:
            shutdown_steps.append(("tactile_visualizer", tactile_visualizer.close))

        shutdown_steps.extend(
            [
                (
                    "position_audit.detach",
                    lambda: _attach_position_audit_sink(components, None),
                ),
                (
                    "position_audit.metadata_detach",
                    lambda: update_control_debug_runtime_metadata(
                        attached=False,
                        session_sidecar_attached=False,
                        session_sidecar_detached_wall_time=time.time(),
                        session_sidecar_detached_monotonic_time=time.monotonic(),
                    ),
                ),
            ]
        )
        if position_debug_episode_recorder is not None:
            if position_audit_fanout is not None:
                shutdown_steps.append(
                    (
                        "position_audit.episode.remove_sink",
                        lambda recorder=position_debug_episode_recorder: (
                            position_audit_fanout.remove_sink(recorder)
                        ),
                    )
                )
            shutdown_steps.extend(
                [
                    (
                        "position_audit.episode_recorder",
                        lambda recorder=position_debug_episode_recorder: recorder.close(
                            timeout_sec=5.0
                        ),
                    ),
                    (
                        "position_audit.episode_metadata",
                        lambda: update_control_debug_runtime_metadata(
                            episode_sidecar_attached=False,
                            episode_sidecars=False,
                            episode_sidecar_detached_wall_time=time.time(),
                            episode_sidecar_detached_monotonic_time=time.monotonic(),
                            episode_sidecar_detach_reason="shutdown",
                        ),
                    ),
                ]
            )
        if position_audit_fanout is not None:
            shutdown_steps.append(("position_audit.fanout", position_audit_fanout.clear))
        if position_debug_session_recorder is not None:
            def close_position_debug_session(
                recorder: PositionDebugRecorder = position_debug_session_recorder,
            ) -> None:
                recorder.close(timeout_sec=5.0)
                snapshot = recorder.snapshot()
                writer_error = snapshot.get("writer_error")
                update_control_debug_runtime_metadata(
                    session_sidecar_error=writer_error,
                )
                _log(
                    "ControlDebug",
                    "session audit closed "
                    f"output={recorder.output_dir} "
                    f"drops={snapshot.get('enqueue_drop_count', 0)} "
                    f"fanout_errors={0 if position_audit_fanout is None else position_audit_fanout.error_count} "
                    f"writer_error={writer_error}",
                )

            shutdown_steps.append(
                ("position_audit.session_recorder", close_position_debug_session)
            )
        for close_thread in position_debug_close_threads:
            def join_position_debug_thread(
                thread: threading.Thread = close_thread,
            ) -> None:
                thread.join(timeout=5.0)
                if thread.is_alive():
                    raise RuntimeError(
                        f"position debug close thread {thread.name!r} did not stop within 5.000s"
                    )

            shutdown_steps.append(
                (f"position_audit.close_thread[{close_thread.name}]", join_position_debug_thread)
            )
        shutdown_steps.append(("teleop_session", components.session.close))
        if components.arm_state_read_executor is not None:
            shutdown_steps.append(
                (
                    "arm_state_read_executor",
                    lambda: components.arm_state_read_executor.shutdown(wait=True),
                )
            )
        if components.right_hand_backend is not None and not components.right_hand_control_via_rig:
            shutdown_steps.append(("right_hand_backend", components.right_hand_backend.close))
        if components.left_hand_backend is not None and not components.left_hand_control_via_rig:
            shutdown_steps.append(("left_hand_backend", components.left_hand_backend.close))
        if components.right_state_backend is not None:
            shutdown_steps.append(("right_state_backend", components.right_state_backend.close))
        if components.left_state_backend is not None:
            shutdown_steps.append(("left_state_backend", components.left_state_backend.close))
        if original_stdout is not None:
            def restore_stdout() -> None:
                sys.stdout = original_stdout

            shutdown_steps.append(("stdout.restore", restore_stdout))
        if silenced_stdout is not None:
            shutdown_steps.append(("stdout.silenced_stream", silenced_stdout.close))
        run_best_effort_cleanup("run_vive_rm65_real_teleop.shutdown", shutdown_steps)


if __name__ == "__main__":
    main()
