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


def _remove_dmrobotics_package_path_shadow() -> None:
    """Prevent the vendor SDK's package directory from shadowing ``utils``.

    The DM-Tac SDK historically inserts ``.../dmrobotics`` itself into
    ``sys.path``.  That directory contains a module named ``utils.py`` and can
    therefore hijack the repository's ``utils`` package during entrypoint
    bootstrap (before the portable SDK adapter has a chance to isolate it).
    Keep the cleanup local to the process bootstrap; the adapter adds the
    SDK's parent directory when it actually imports the vendor package.
    """

    for value in tuple(sys.path):
        if not isinstance(value, (str, os.PathLike)) or not str(value):
            continue
        try:
            candidate = Path(value).resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if (
            candidate.name == "dmrobotics"
            and (candidate / "__init__.py").is_file()
            and (candidate / "utils.py").is_file()
        ):
            while value in sys.path:
                sys.path.remove(value)


_remove_dmrobotics_package_path_shadow()

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation

from teleop_runtime.configs.camera_paths import DEFAULT_REALSENSE_DEVICE_CONFIG_PATH
from dataset_contracts.pose_frames import legacy_rm65_source_id, load_pose_frame_contract
from dataset_contracts.capture_modalities import CaptureModalityContract
from dataset_contracts.gripper_force import GripperForceDatasetContract
from dataset_contracts.stats import (
    FORCE_COLLISION_LABEL_DETECTED,
    has_force_limit_collision_text,
)
from dataset_interface.capture_modalities import bind_capture_modality_contract
from dataset_interface.gripper_force import (
    bind_gripper_force_contract,
    read_gripper_force_contract,
)
from teleop_runtime.action.human_shadow_frontend import (
    RealTeleopHumanShadowProfile,
    add_human_shadow_arguments,
    build_real_teleop_human_shadow_frontend,
    validate_human_shadow_arguments,
)
from teleop_runtime.backends.arms import (
    RM65ArmBackend,
    RM65ArmBackendConfig,
    RM65ImpedanceArmBackend,
    RM65ImpedanceArmBackendConfig,
)
from teleop_runtime.backends.grippers import (
    DEFAULT_DMTAC_GRIPPER_COMMAND_HZ,
    DEFAULT_DMTAC_GRIPPER_TELEMETRY_HZ,
    DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
    DMTAC_TACTILE_FIELDS,
    DMTacGripperBackend,
    DMTacGripperBackendConfig,
    GRIPPER_ACTION_MAX,
    normalize_dmtac_tactile_fields,
)
from teleop_runtime.backends.hands import (
    DEFAULT_INSPIRE_HAND_MAX_FORCE,
    InspireTacHandBackend,
    InspireTacHandBackendConfig,
)
from teleop_runtime.arm_frames import TOOL_TCP_FRAME
from robot_motion.action import HandActionSpec
from robot_motion.joint_limits import load_joint_velocity_limit_profiles
from robot_motion.pose_frame_profiles import default_rm65_workspace_near_boundary_collection_pose_frame_contract
from robot_motion.profiles import IKSafetyProfile, load_ik_safety_profiles
from robot_motion.position_control import (
    DEFAULT_POSITION_DEGRADED_SPEED_SCALE,
    DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
    DEFAULT_POSITION_VELOCITY_RAMP_SEC,
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
    POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
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
from teleop_runtime.calibration import TeleopCalibration
from teleop_runtime.collection.diagnostics import (
    PhaseTimer,
    build_end_to_end_latency_breakdown,
    format_end_to_end_latency_breakdown,
    format_phase_timings,
    should_report_debug_frozen,
)
from teleop_runtime.collection.async_writer import AsyncWriterConfig, AsyncWriterPool
from teleop_runtime.collection.arm_state_polling import ArmHardwareStatePoller, ArmHardwareStatePollerConfig
from teleop_runtime.collection.hand_state_polling import HandStateHistoryPoller, HandStateHistoryPollerConfig
from teleop_runtime.collection.camera_preview import CameraPreviewConfig, CameraPreviewer
from teleop_runtime.collection.build_collect_components import CollectorComponents
from teleop_runtime.collection.dataset_episode_maintenance import (
    EpisodeDeleteResult,
    delete_episode_and_compact,
)
from teleop_runtime.collection.observers import CollectionWriterObserverConfig
from teleop_runtime.collection.policy_io_runtime import resolve_collection_policy_io_layout
from teleop_runtime.collection.process_writer import (
    ProcessCollectionConfig,
    ProcessCollectionWorker,
    camera_shared_slot_bytes,
)
from teleop_runtime.collection.preview_rate import PreviewRateLimiter
from teleop_runtime.collection.rate_gate import PeriodicRateGate
from teleop_runtime.collection.sample_assembler import (
    DEFAULT_CAMERA_FRAME_MAX_AGE_SEC,
    CollectorSampleBuilder,
    CollectorSampleBuilderConfig,
    _build_pose_quality_metadata as _collection_build_pose_quality_metadata,
)
from teleop_runtime.collection.status_snapshot import (
    camera_debug_text as _collection_camera_debug_text,
    hand_debug_text as _collection_hand_debug_text,
    writer_debug_text as _collection_writer_debug_text,
)
from teleop_runtime.collection.writer_runtime import CollectionWriterRuntime
from teleop_runtime.collection.writer_v21 import LeRobotV21Writer, LeRobotV21WriterConfig
from teleop_runtime.control_trace.episode import EpisodeTraceRecorderSink
from teleop_runtime.configs import (
    ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
    DEFAULT_ARM_WORKSPACE_CONFIG_PATH,
    build_bimanual_pose_smoother,
    build_annotated_dual_rm65_calibration,
    load_default_pose_smoothing_config,
    load_impedance_pose_smoothing_config,
)
from teleop_runtime.configs.dmtac_hardware import (
    add_dmtac_assignment_arguments,
    dmtac_backend_configs_from_assignment,
    load_dmtac_hardware_assignment,
    manual_dmtac_backend_config_for_side,
    require_dmtac_sides_resolved,
)
from teleop_runtime.configs.q6_hardware import load_latest_q6_ports
from teleop_runtime.configs.rm65_arm_ips import (
    arm_ips_for_sides,
    build_assignment as build_rm65_arm_ip_assignment,
    load_assignment as load_rm65_arm_ip_assignment,
    save_assignment_for_config as save_rm65_arm_ip_assignment_for_config,
)
from teleop_runtime.defaults import (
    DEFAULT_IMPEDANCE_TELEOP_CONTROL_HZ,
    DEFAULT_RM65_ARM_CONTROL_HZ,
    DEFAULT_RM65_POSITION_ARM_CONTROL_HZ,
    DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
    DEFAULT_TELEOP_SAMPLE_HZ,
)
from teleop_runtime.device_preflight import (
    DevicePreflightError,
    DevicePreflightOptions,
    prepare_devices_for_teleop,
)
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.episodes import EpisodeRecorder, EpisodeRecorderConfig, compare_start_qpos_pair
from teleop_runtime.hand_mapping import (
    DEFAULT_INSPIRE_HAND_JOINT_ORDER,
    TARGET_INSPIRE_HAND_Q6_ORDER,
    parse_tactile_regions,
)
from teleop_runtime.hand_render import (
    LEFT_HAND_RENDER_JOINT_NAMES,
    RIGHT_HAND_RENDER_JOINT_NAMES,
)
from teleop_runtime.hand_tcp_alignment import (
    DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG,
    add_vive_tracker_height_correction_args,
    tracker_hand_to_rm65_tcp_rotation,
)
from teleop_runtime.tracker_hand import TRACKER_OFFSET_CALIBRATION_VERSION, canonical_hand_rpy_default
from teleop_runtime.initial_alignment import TeleopInitialAlignment
from teleop_runtime.retargeters import (
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
from teleop_runtime.recording import RecordingConfig, SessionRecorder
from teleop_runtime.position_control import (
    ARM_EXECUTION_PROFILE_CHOICES,
    CONTROL_DEBUG_LEVELS,
    DEFAULT_CONTROL_DEBUG_FLUSH_EVERY,
    DEFAULT_CONTROL_DEBUG_QUEUE_SIZE,
    DEFAULT_CONTROL_DEBUG_ROOT,
    POSITION_15_60_120HZ_PROFILE_NAME,
    PositionDebugRecorder,
    PositionDebugRecorderConfig,
    PositionReferencePublisher,
    PositionReferencePublisherConfig,
    RealArmExecutionConfig,
    apply_rm65_position_profile_defaults,
    resolve_control_debug_config,
    resolve_real_arm_execution_config,
)
from teleop_runtime.rigs import BimanualRig
from teleop_runtime.safety import TeleopSafetyConfig
from teleop_runtime.session import (
    TRACKER_HOLD_AUTOMATIC,
    TRACKER_HOLD_GRACE,
    SessionStepResult,
    TeleopSession,
    TeleopSessionConfig,
)
from teleop_runtime.sources.camera import RealSenseRGBSource, load_realsense_camera_config_map
from teleop_runtime.sources.hand import AngleTriggerSource, HandQ6Source
from teleop_runtime.sources.misc import KeyboardButtonSource
from teleop_runtime.sources.pose import DEFAULT_VIVE_CLOCK_SYNC_PORT, VivePoseSource
from teleop_runtime.sources.pose.vive_motion import TrackerMotionIntentConfig
from teleop_runtime.special_tasks.fix_arm import (
    FixArmCommandController,
    FixArmPoseSnapshotRecorder,
    advance_fix_arm_home_capture,
    build_fix_arm_pose_snapshot_payload,
)
from teleop_runtime.special_tasks.fix_hold import SpecialArmFixHoldCoordinator
from teleop_runtime.special_tasks.init_arm import (
    InitArmCommandController,
    InitArmTransition,
    advance_init_arm_home_capture,
    load_init_arm_targets,
)
from teleop_runtime.tactile_visualizer import (
    AsyncDMTacDistributedForceVisualizer,
    AsyncTactileUVVisualizer,
    DMTacDistributedForceVisualizer,
    DMTacForceVisualizerConfig,
    TactileUVVisualizer,
    TactileVisualizerConfig,
    dmtac_visualization_snapshot_from_backends,
    tactile_by_side_from_backends,
)
from teleop_runtime.tcp import legacy_tcp_mode_requested
from teleop_runtime.transforms import as_pose_matrix
from teleop_runtime.calibration_flow import (
    average_pose,
    euler_xyz_deg_to_matrix,
    fit_signed_axis_from_poses,
    fit_virtual_desk_frame,
    pose_from_rotation_translation,
    tracker_offset_from_neutral_pose,
)
from teleop_runtime.gui import collection_tasks as gui_collection_tasks
from control_station.web.calibration_web_ui import CalibrationWebUi
from control_station.web.startup_progress import GuiStartupProgress
from teleop_runtime.real_teleop.foot_pad_controls import (
    FootPadSpecialArmInput,
    FootPadUnavailableError,
    SPECIAL_ARM_CONTROL_MODE_CHOICES,
    SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
    SPECIAL_ARM_CONTROL_MODE_MANUAL,
    configured_foot_pad_count,
    resolve_special_arm_control_mode,
)
from teleop_runtime.real_teleop.actuator_command_trace import (
    DEFAULT_ACTUATOR_TRACE_MAX_DURATION_SEC,
    DEFAULT_ACTUATOR_TRACE_ROOT,
    ActuatorCommandTraceConfig,
    ActuatorCommandTraceRecorder,
)
from teleop_runtime.real_teleop.fix_arm_collision_guard import fixed_arm_final_collision_fault
from teleop_runtime.real_teleop.position_force_guard import (
    POSITION_FORCE_SOURCES,
    PositionForceGuardConfig,
    PositionForceGuardFault,
    PositionForceGuardMonitor,
    position_force_guard_config as _position_force_guard_config,
)
from teleop_runtime.real_teleop import dataset_layout as real_teleop_dataset_layout

_expand_cli_path = real_teleop_dataset_layout.expand_cli_path
_sanitize_dataset_name_for_path = real_teleop_dataset_layout.sanitize_dataset_name_for_path
_default_task_id_from_name = real_teleop_dataset_layout.default_task_id_from_name
from teleop_runtime.debug.log_utils import ChangeGate
from teleop_runtime.debug.run_vive_sapien_virtual import (
    DEFAULT_RUNTIME_HOME_START_DELAY_SEC,
    RecoveryPlan,
    TerminalClutchInput,
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
from teleop_runtime.cli_value_parsers import (
    _active_sides_from_args,
    _alpha_float,
    _non_negative_int,
    _parse_active_sides,
    _parse_arm_max_angvel,
    _parse_joint_order,
    _positive_float,
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
    LEGACY_POSITION_CONTROL_PROFILE_NAME,
    RM65_BIMANUAL_POSITION_V1_NAME,
    RM65_POSITION_CONTROL_PROFILE_CHOICES,
    build_dual_arm_scene_profile,
)
from sim.rm65.sapien_renderer import SapienArmHandRenderer
from project_paths import calibration_profiles_dir, default_dataset_root, teleop_records_root
from teleop_runtime.real_teleop.defaults import (
    DEFAULT_CAMERA_CONFIG_PATH,
    DEFAULT_DATASET_ROOT,
    DEFAULT_INIT_ARM_PROFILE_PATH,
    DEFAULT_INIT_ARM_QPOS_SPEED_SCALE,
    DEFAULT_POSITION_MAX_ACCELERATION_RAD_S2,
    DEFAULT_REAL_TELEOP_MAX_TRACKER_ROTATION_JUMP_DEG,
    DEFAULT_REAL_TELEOP_MAX_TRACKER_TRANSLATION_JUMP_M,
    INIT_ARM_QPOS_SPEED_BASELINE_PROFILE,
)
from teleop_runtime.real_teleop.cli import build_real_teleop_argument_parser


DEFAULT_DMTAC_ROOT = Path(__file__).resolve().parents[1] / "hardware" / "DMTAC_gripper"
DEFAULT_CALIBRATION_PROFILE_DIR = calibration_profiles_dir()
DEFAULT_EPISODE_START_COUNTDOWN_SEC = 3.0
# MIGRATION(real-teleop, batch 6): recorder filenames and their default roots
# are owned by the extracted diagnostics component.  Importing the constants
# here deliberately retains the legacy module-level patch point used by local
# tooling while avoiding two independently maintained defaults.
from teleop_runtime.real_teleop.debug_recorders import (
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


from teleop_runtime.real_teleop.hand_config import (
    HAND_BACKEND_INSPIRE,
    HAND_BACKEND_DMTAC_GRIPPER,
    HAND_BACKEND_NONE,
    HAND_BACKEND_CHOICES,
    dmtac_sides as _dmtac_sides,
    hand_backend_mode as _hand_backend_mode,
    hand_backend_mode_for_side as _hand_backend_mode_for_side,
)
from teleop_runtime.real_teleop.hand_cli import (
    DEFAULT_TASKS_CONFIG_PATH,
    add_dmtac_arguments,
    add_gripper_policy_arguments,
    add_hand_backend_arguments,
    add_inspire_hand_arguments,
)
from teleop_runtime.real_teleop.gripper_policy import (
    GRIPPER_POLICY_DEFAULTS,
    clear_gripper_policy_cache as _clear_gripper_policy_cache,
    current_gripper_force_dataset_contract as _current_gripper_force_dataset_contract,
    gripper_force_dataset_contract_from_args as _gripper_force_dataset_contract_from_args,
    gripper_force_dataset_metadata as _gripper_force_dataset_metadata,
    gripper_side_override as _gripper_side_override,
    load_gripper_task_profile as _load_gripper_task_profile,
    merge_gripper_profile as _merge_gripper_profile,
    profile_value as _profile_value,
    resolved_gripper_policies_by_side as _resolved_gripper_policies_by_side,
    resolved_gripper_policy as _resolved_gripper_policy,
    resolved_gripper_policy_for_side as _resolved_gripper_policy_for_side,
    side_gripper_overrides_from_profile as _side_gripper_overrides_from_profile,
    strip_side_gripper_overrides as _strip_side_gripper_overrides,
)
from teleop_runtime.real_teleop.dmtac_config import (
    apply_gripper_policy_to_dmtac_config as _apply_gripper_policy_to_dmtac_config,
    dmtac_tactile_enabled as _dmtac_tactile_enabled,
)
EPISODE_GUI_EVENTS = (
    "episode_start",
    "episode_stop",
    "episode_delete",
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
from teleop_runtime.real_teleop.movement import (
    MOVEMENT_MULTIPLE_DEFAULT,
    MOVEMENT_MULTIPLE_GUI_EVENTS,
    MOVEMENT_MULTIPLE_GUI_MAX,
    MOVEMENT_MULTIPLE_GUI_MIN,
    MOVEMENT_MULTIPLE_GUI_STEP,
    apply_movement_multiple as _apply_movement_multiple,
    movement_multiple_ui_state as _movement_multiple_ui_state,
    movement_multiple_value as _movement_multiple_value,
    resolve_movement_multiple_args as _resolve_movement_multiple_args,
)
VIRTUAL_DESK_SWEEP_CONVENTION_VERSION = 2
TRACKER_GUI_WARN_AGE_SEC = 0.25
TRACKER_GUI_STALE_AGE_SEC = 0.50
CAMERA_STARTUP_SELF_CHECK_DURATION_SEC = 3.0
CAMERA_STARTUP_SELF_CHECK_INTERVAL_SEC = 0.2
CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES = 3
CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC = DEFAULT_CAMERA_FRAME_MAX_AGE_SEC
# MIGRATION(real-teleop, batch 1): wrist-depth freshness now lives in the
# dependency-light component below.  This legacy entrypoint deliberately
# re-exports the former names so parallel callers and external scripts do not
# need to change import paths during the staged refactor.  Remove this note
# only after the complete real-teleop extraction is accepted.
from teleop_runtime.real_teleop.wrist_depth_freshness import (
    DEFAULT_EPOCH_TIMESTAMP_STALE_SEC as WRIST_DEPTH_EPOCH_TIMESTAMP_STALE_SEC,
    DEFAULT_STALE_REPEAT_FAILURE_FRAMES as WRIST_DEPTH_STALE_REPEAT_FAILURE_FRAMES,
    DEFAULT_WRIST_DEPTH_FPS,
    DEFAULT_WRIST_DEPTH_MAX_M,
    DEFAULT_WRIST_DEPTH_MIN_M,
    DEFAULT_WRIST_DEPTH_SAVE_FPS,
    DEFAULT_WRIST_DEPTH_STREAM_FPS,
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
from teleop_runtime.real_teleop.debug_timing import (
    AsyncDebugTimingConsole,
    DebugTimingModules,
    RealDebugTimingState,
    debug_timing_metadata as _debug_timing_metadata,
    debug_timing_modules as _debug_timing_modules,
    merge_control_debug_runtime_metadata as _merge_control_debug_runtime_metadata,
)


from teleop_runtime.real_teleop.runtime_metadata import (
    arm_execution_profile_metadata as _arm_execution_profile_metadata,
    position_ik_iteration_metadata as _position_ik_iteration_metadata,
)


from teleop_runtime.real_teleop.diagnostic_setup import (
    _attach_actuator_command_trace_sink,
    _attach_ik_shadow_trace_sink,
    _attach_position_audit_sink,
    _attach_singularity_transition_trace_sink,
    _new_actuator_command_trace_recorder,
    _new_position_debug_recorder,
)










# MIGRATION(real-teleop, batch 3): the legacy helper remains an alias so
# existing audit tests and integrations retain their import path.
from teleop_runtime.real_teleop.debug_timing import (
    record_position_loop_timings as _record_position_loop_timings,
)


# MIGRATION(real-teleop, batch 2): the gesture implementation now lives in the
# dependency-light component below.  Re-export former names from this legacy
# entrypoint until the staged refactor is complete, so GUI/control collaborators
# and external scripts keep their existing imports.
from teleop_runtime.real_teleop.q6_hold_gesture import (
    Q6HoldRecalibrationGesture,
    Q6HoldRecalibrationGestureConfig,
)

# MIGRATION(real-teleop, batch 4): the temporary dual-arm limiter is now a
# real-teleop-specific component.  Keep these legacy re-exports so the
# established entrypoint and its callers retain their existing import path;
# do not alter safety limits in this compatibility bridge.
from teleop_runtime.real_teleop.temporary_dual_arm_safety import (
    TemporaryDualArmSafetyConfig,
    TemporaryDualArmSafetyLimiter,
)


from teleop_runtime.real_teleop.command_filters import (
    _compose_real_command_filter,
    _compose_real_command_guard,
    _compose_real_fault_filter,
    _filter_manual_real_command,
    _fix_arm_required_effectors_by_side,
)


# Compatibility bridge: retain established workspace helper names while the
# extracted real-teleop workspace policy owns their behavior.
from teleop_runtime.real_teleop.workspace import (
    apply_workspace_config as _apply_workspace_config,
    apply_workspace_overrides as _apply_workspace_overrides,
    arm_workspace_config_path as _arm_workspace_config_path,
    parse_vector3_arg as _parse_vector3_arg,
    resolve_workspace_override_args as _resolve_workspace_override_args,
    workspace_override_for_side as _workspace_override_for_side,
)










from teleop_runtime.real_teleop import cli_runtime_config as _cli_runtime_config


def _apply_default_q6_ports(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> None:
    return _cli_runtime_config._apply_default_q6_ports(args, parser=parser)


def _run_gui_device_preflight(
    args: argparse.Namespace,
    *,
    active_sides: tuple[str, ...],
    parser: argparse.ArgumentParser,
) -> None:
    return _cli_runtime_config._run_gui_device_preflight(
        args,
        active_sides=active_sides,
        parser=parser,
    )


def _resolve_gripper_input_source(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> None:
    return _cli_runtime_config._resolve_gripper_input_source(args, active_sides)


def _argv_has_option(raw_argv: Sequence[str], option: str) -> bool:
    return _cli_runtime_config._argv_has_option(raw_argv, option)


def _apply_real_loop_timing_defaults(
    args: argparse.Namespace,
    *,
    raw_argv: Sequence[str],
) -> None:
    return _cli_runtime_config._apply_real_loop_timing_defaults(
        args,
        raw_argv=raw_argv,
    )


def _source_arm_ip_fields_for_active_sides(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> dict[str, str]:
    return _cli_runtime_config._source_arm_ip_fields_for_active_sides(
        args,
        active_sides,
    )


def _apply_default_arm_ips(
    args: argparse.Namespace,
    *,
    active_sides: tuple[str, ...],
) -> None:
    return _cli_runtime_config._apply_default_arm_ips(
        args,
        active_sides=active_sides,
    )


def _save_explicit_arm_ips_latest(
    args: argparse.Namespace,
    *,
    raw_argv: Sequence[str],
) -> None:
    return _cli_runtime_config._save_explicit_arm_ips_latest(
        args,
        raw_argv=raw_argv,
    )


def _validate_required_arm_ips(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> None:
    return _cli_runtime_config._validate_required_arm_ips(args, active_sides)


def _dmtac_backend_configs(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> dict[str, DMTacGripperBackendConfig]:
    return _cli_runtime_config._dmtac_backend_configs(args, active_sides)


# Compatibility bridge: callers retain the established entrypoint helper names
# while the pure real-teleop profile implementation owns their behavior.
from teleop_runtime.real_teleop.control_profile import (
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


from teleop_runtime.real_teleop.arm_tool_config import (
    _alignment_qpos_pair,
    _force_tare_sec_arg,
    _impedance_mode_arg,
    _start_state_summary,
    _strict_tool_profile_arg,
    _tool_profile_arg,
    _tool_tcp_pose_flange_arg,
)
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


from teleop_runtime.real_teleop.cli_resolution import (
    bind_legacy_support as _bind_cli_resolution_support,
    parse_real_teleop_args,
    parse_real_teleop_args_with_parser,
)
from control_station.real_agent_primitive_runtime import (
    add_real_agent_primitive_arguments,
    build_real_agent_primitive_extension,
    validate_real_agent_primitive_arguments,
)


def parse_args() -> argparse.Namespace:
    _bind_cli_resolution_support(globals())
    parser = build_real_teleop_argument_parser()
    add_real_agent_primitive_arguments(parser)
    args = parse_real_teleop_args_with_parser(parser)
    validate_real_agent_primitive_arguments(args, parser)
    return args


# Compatibility bridge: retain the established entrypoint helper names while
# the extracted component owns logical-side and tracker mapping behavior.
from teleop_runtime.real_teleop.side_mapping import (
    build_vive_tracker_configs_for_args as _build_vive_tracker_configs_for_args,
    opposite_side as _opposite_side,
    resolve_logical_side_assignments as _resolve_logical_side_assignments,
    tracker_legacy_index_for_source_side as _tracker_legacy_index_for_source_side,
    tracker_name_for_source_side as _tracker_name_for_source_side,
    tracker_role_for_source_side as _tracker_role_for_source_side,
    tracker_source_side_for_logical_side as _tracker_source_side_for_logical_side,
)


def _vive_pose_source_from_components(components: RealTeleopComponents) -> VivePoseSource | None:
    for source in components.session.sources:
        if isinstance(source, VivePoseSource):
            return source
    return None


from teleop_runtime.real_teleop.hand_runtime import (
    _build_runtime_hand_command_for_recovery,
    _dmtac_command_contract_configured,
    _dmtac_tactile_backends,
    _emit_tactile_sum_only_status,
    _enable_gui_tactile_solving,
    _extract_hand_command_from_command,
    _gripper_backend_for_side,
    _gripper_gui_available,
    _gripper_input_channel_name,
    _gripper_input_source_from_components,
    _gripper_retargeter_for_side,
    _gripper_state_scalar_from_payload,
    _hand_backend_for_side,
    _hand_q6_source_from_components,
    _hand_tactile_sum,
    _has_dmtac_gripper_backend,
    _has_q6_gripper_control,
    _hold_requested_pause_reason,
    _merge_runtime_hand_command,
    _parse_hand_tactile_regions,
    _read_gripper_backend_scalar_now,
    _tactile_solving_active,
    _tactile_solving_available,
    _tactile_visualization_available,
)
def _discover_runtime_q6_ports(
    args: argparse.Namespace,
    active_sides: Sequence[str],
) -> dict[str, str]:
    return _cli_runtime_config._discover_runtime_q6_ports(args, active_sides)


def _q6_gripper_retargeter_for_side(
    args: argparse.Namespace,
    side: str,
) -> ScalarToGripperRetargeter:
    policy = _resolved_gripper_policy_for_side(args, side)
    return ScalarToGripperRetargeter(
        ScalarToGripperRetargeterConfig(
            side=side,
            input_channel_name=f"{side}_hand_q6",
            effector_name=f"{side}_hand",
            source_dim=int(policy["source_dim"]),
            output_dim=int(policy["output_dim"]),
            invert=bool(policy["invert"]),
            alpha=float(policy["alpha"]),
            input_min=float(policy["input_min"]),
            input_max=float(policy["input_max"]),
            output_max=float(policy["max_action"]),
        )
    )


from teleop_runtime.real_teleop.gripper_recovery import (
    promote_angle_trigger_gripper_input_to_q6,
)


def _promote_angle_trigger_gripper_input_to_q6(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    ready_timeout_sec: float = 2.0,
) -> tuple[bool, str]:
    return promote_angle_trigger_gripper_input_to_q6(
        args,
        components,
        ready_timeout_sec=ready_timeout_sec,
        discover_runtime_q6_ports=_discover_runtime_q6_ports,
        q6_gripper_retargeter_for_side=_q6_gripper_retargeter_for_side,
    )


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
from teleop_runtime.real_teleop.gui_tracker_status import (
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
from teleop_runtime.real_teleop.gui_vive_calibration import (
    vive_calibration_profile_ready as _vive_calibration_profile_ready,
    vive_calibration_profile_status_text as _vive_calibration_profile_status_text,
    vive_calibration_profile_ui_state as _vive_calibration_profile_ui_state,
    vive_calibration_tracker_roles as _vive_calibration_tracker_roles,
)


from teleop_runtime.real_teleop.vive_calibration_profiles import (
    _calibration_profile_latest_sort_key,
    _calibration_profile_payload_without_workspace,
    _gui_vive_calibration_load_path as _gui_vive_calibration_load_path_impl,
    _latest_vive_calibration_profile_path as _latest_vive_calibration_profile_path_impl,
    _load_gui_vive_calibration_profile as _load_gui_vive_calibration_profile_impl,
    _runtime_home_was_available,
    _save_gui_vive_calibration_profile as _save_gui_vive_calibration_profile_impl,
)


def _latest_vive_calibration_profile_path() -> Path | None:
    return _latest_vive_calibration_profile_path_impl(
        profile_dir=DEFAULT_CALIBRATION_PROFILE_DIR,
    )


def _gui_vive_calibration_load_path(args: argparse.Namespace) -> Path:
    return _gui_vive_calibration_load_path_impl(
        args,
        profile_dir=DEFAULT_CALIBRATION_PROFILE_DIR,
    )


def _save_gui_vive_calibration_profile(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> Path:
    return _save_gui_vive_calibration_profile_impl(
        args,
        components,
        profile_dir=DEFAULT_CALIBRATION_PROFILE_DIR,
    )


def _load_gui_vive_calibration_profile(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> Path:
    return _load_gui_vive_calibration_profile_impl(
        args,
        components,
        profile_dir=DEFAULT_CALIBRATION_PROFILE_DIR,
    )


from teleop_runtime.real_teleop.vive_calibration_runtime import (
    _capture_dual_wrist_offsets_from_gui,
    _capture_single_vive_channel_from_gui,
    _channel_for_tracker_role,
    _channel_sample_key,
    _clear_runtime_home_state,
    _consume_vive_capture_events,
    _default_gui_vive_calibration_side,
    _format_vive_source_rates,
    _format_vive_sweep_quality,
    _format_wall_time,
    _publish_vive_capture_state,
    _resolve_virtual_desk_channel_name,
    _select_gui_vive_coordinate_sweep_hand,
    _source_debug_stats,
    _tracker_role_for_channel,
    _virtual_desk_post_rotation_pose,
    calibrate_gui_vive_coordinate_profile,
)


def _calibrate_gui_vive_coordinate_profile(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    ui: CalibrationWebUi,
) -> Path:
    return calibrate_gui_vive_coordinate_profile(
        args,
        components,
        ui,
        save_profile=_save_gui_vive_calibration_profile,
    )




from teleop_runtime.real_teleop.camera_runtime import (
    CAMERA_START_STAGGER_SEC,
    _build_camera_sources,
    _close_camera_sources,
    _select_preview_camera_sources,
    _start_camera_sources,
    _start_camera_sources_best_effort,
)


from teleop_runtime.real_teleop.special_arm_runtime import (
    MAX_INIT_ARM_QPOS_SPEED_SCALE,
    MIN_INIT_ARM_QPOS_SPEED_SCALE,
    _cancel_special_arm_modes_for_hold,
    _disengage_position_backends_for_manual_interaction,
    _fresh_gui_init_qpos_by_side,
    _init_arm_qpos_speed_scale_for_side,
    _normalize_init_arm_qpos_speed_scale,
    _release_gui_qpos_init_transit,
    start_gui_qpos_init,
)


def _start_gui_qpos_init(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    now: float,
) -> InitArmTransition:
    return start_gui_qpos_init(
        args,
        components,
        now=now,
        fresh_qpos_reader=_fresh_gui_init_qpos_by_side,
        impedance_force_control_not_ready_sides=_impedance_force_control_not_ready_sides,
        impedance_backend_type=RM65ImpedanceArmBackend,
    )


# Compatibility bridge: retain legacy status helper names while the extracted
# read-only projection owns special-arm GUI status formatting.
from teleop_runtime.real_teleop.gui_special_arm_status import (
    fix_arm_debug_state as _fix_arm_debug_state,
    fix_arm_status_text as _fix_arm_status_text,
    fix_arm_ui_state as _fix_arm_ui_state,
    init_arm_debug_state as _init_arm_debug_state,
    init_arm_status_text as _init_arm_status_text,
    init_arm_ui_state as _init_arm_ui_state,
)




from teleop_runtime.real_teleop.session_recording import (
    _session_record_root_dir,
    _session_recording_config,
    _session_recording_enabled,
)
from teleop_runtime.real_teleop.episode_runtime import (
    _build_episode_metadata,
    _debug_episode_recording_enabled,
    _episode_root_dir,
    _fix_arm_pose_snapshot_recorder,
    _start_episode_recording,
    _stop_episode_recording,
)




# Compatibility bridge: retain established collection adapter helper names
# while the extracted real-teleop bridge owns their behavior.
from teleop_runtime.real_teleop.collection_bridge import (
    collection_backends_from_components as _collection_backends_from_components,
    collection_sample_step_result as _collection_sample_step_result,
    collection_sources_from_components as _collection_sources_from_components,
    collection_step_result as _collection_step_result,
    copy_teleop_command as _copy_teleop_command,
)






from teleop_runtime.real_teleop.runtime_debug import (
    _action_debug_backend_snapshot,
    _dataset_recording_debug_text,
    _dataset_storage_debug_text,
    _debug_timing_maybe_report,
    _debug_timing_report,
    _preloaded_action_debug_backend_snapshot,
)








from teleop_runtime.real_teleop.dataset_recorder import (
    RealTeleopDatasetRecorder,
    _collection_hand_action_specs,
)

RealTeleopDatasetRecorder.__module__ = (
    "teleop_runtime.debug.run_vive_rm65_real_teleop"
)






from teleop_runtime.real_teleop.position_tuning import (
    _impedance_bimanual_ik_safety_profile,
    _position_ik_tuning_for_args,
    _position_tracker_motion_intent_config,
)


from teleop_runtime.real_teleop.component_builder import build_real_teleop_components


def _build_components(args: argparse.Namespace) -> RealTeleopComponents:
    return build_real_teleop_components(
        args,
        dmtac_backend_configs=_dmtac_backend_configs,
    )


from teleop_runtime.real_teleop.arm_state_sync import (
    _reset_runtime_motion_state_from_real_qpos,
    _run_initial_alignment_start_check,
    _sync_preview_from_real_state_real,
    _sync_real_execution_step_limits,
)


from teleop_runtime.real_teleop.arm_runtime import (
    _acceleration_limited_position_execution_active,
    _active_command_backend,
    _active_impedance_force_backends,
    _apply_active_arm_command_to_preview_real,
    _command_backend_for_side,
    _estimate_impedance_force_control_warmup_sec,
    _impedance_force_control_not_ready_sides as _impedance_force_control_not_ready_sides_impl,
    _impedance_force_control_warmup_backends,
    _runtime_arm_backend_state,
    _sync_dual_arm_retargeter_for_real_control,
)


def _impedance_force_control_not_ready_sides(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> list[str]:
    return _impedance_force_control_not_ready_sides_impl(
        args,
        components,
        impedance_backend_type=RM65ImpedanceArmBackend,
    )


from teleop_runtime.real_teleop.impedance_force_runtime import (
    _emergency_stop_real_arms,
    _enable_impedance_force_control_after_runtime_home,
    _exception_chain_messages,
    _force_collision_recording_summary,
    _force_guard_recording_stop_context,
    _handle_init_qpos_emergency_stop,
    _impedance_force_control_issue,
    _position_force_guard_collision_summary,
    _terminal_force_collision_summary,
    _warm_start_impedance_force_control_before_home,
    guard_impedance_force_control,
    hold_motion_after_impedance_guard_issue,
)


def _hold_motion_after_impedance_guard_issue(
    components: RealTeleopComponents,
    *,
    reason: str,
    now: float,
) -> None:
    hold_motion_after_impedance_guard_issue(
        components,
        reason=reason,
        now=now,
        log=_log,
    )


def _guard_impedance_force_control(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    context: str,
    now: float,
    collision_handler: Callable[[str], None] | None = None,
) -> bool:
    return guard_impedance_force_control(
        args,
        components,
        context=context,
        now=now,
        issue_provider=_impedance_force_control_issue,
        hold_handler=_hold_motion_after_impedance_guard_issue,
        log=_log,
        collision_handler=collision_handler,
    )


from teleop_runtime.real_teleop.runtime_health import (
    _command_backend_health_signature,
    _command_backend_status_text,
    _short_status_text,
    _temporary_safety_health_signature,
)


from teleop_runtime.real_teleop.gripper_control import (
    _dispatch_q6_gripper_control,
    _probe_gripper_backend_motion,
    _run_gripper_warmup_probe_for_side,
)


def _real_hand_render_qpos(
    backend: InspireTacHandBackend | DMTacGripperBackend | None,
    side: str,
) -> np.ndarray:
    if isinstance(backend, DMTacGripperBackend):
        return _hand_render_qpos(None, side)
    return _hand_render_qpos(backend, side)


from teleop_runtime.real_teleop.gripper_status import (
    _DEBUG_TIMING_RED,
    _DEBUG_TIMING_RESET,
    _build_gripper_status,
    _gripper_runtime_health_signature,
    _gripper_runtime_status_signature,
    _gripper_status_for_side,
    _gripper_status_signature,
    _hand_channel_status,
    _q6_channel_status,
    _q6_retarget_status,
)


from teleop_runtime.real_teleop.status_payloads import _build_pose_quality_display_payload








































from teleop_runtime.real_teleop.tactile_visualization import _build_tactile_visualizer
















# MIGRATION(real-teleop, batch 6): route legacy recorder construction through
# the extracted component.  These wrappers intentionally preserve the legacy
# module-level default-root patch point used by tests and local tooling; no
# recorder behavior is implemented here.
from teleop_runtime.real_teleop.debug_recorders import (
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
from teleop_runtime.real_teleop.main_loop_watchdog import MainLoopStallWatchdog

# GUI batch 1 compatibility bridge: the imported policy is pure and the
# coordinator below continues to own all state transitions and event effects.
from control_station.real_teleop.control_availability import (
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
from control_station.real_teleop.input_mapping import prioritize_input_batch
from control_station.real_teleop_application import (
    build_real_teleop_progress_application,
)


# Batch 7 compatibility imports: these modules contain pure value/state
# primitives.  Runtime assembly and all device side effects remain below.
from teleop_runtime.real_teleop.components import RealTeleopComponents

RealTeleopComponents.__module__ = "teleop_runtime.debug.run_vive_rm65_real_teleop"

from teleop_runtime.real_teleop.cli_value_parsers import (
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
from teleop_runtime.real_teleop.dataset_finalize import RealDatasetFinalizeTask

RealDatasetFinalizeTask.__module__ = (
    "teleop_runtime.debug.run_vive_rm65_real_teleop"
)

from teleop_runtime.real_teleop.debug_payload_primitives import (
    _compact_debug_value,
    _compact_numeric_list,
    _pose_compact_debug,
    _pose_delta_compact_debug,
    _pose_matrix_debug,
    _qpos_compact_debug,
    _qpos_debug,
)

from teleop_runtime.real_teleop.debug_ik_payload import (
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

# GUI batch 2: the coordinator body lives in ``teleop_runtime.gui``.  Status
# projections import shared read-only formatting from its runtime-health owner;
# this entrypoint retains compatibility re-exports and runtime assembly.
from teleop_runtime.real_teleop.status_payloads import (
    _IK_RED_WARNING_LABELS,
    _IK_RED_WARNING_SINGULARITY_GATE_CODES,
    _IK_RED_WARNING_STATUS_CODES,
    _as_status_error_text,
    _bool_or_false,
    _build_camera_status,
    _build_ik_timing_payload,
    _build_ik_warning_payload,
    _build_tactile_camera_status,
    _camera_status_for_source,
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
    _with_expected_wrist_depth_camera_statuses,
    _wrist_depth_requested,
    WRIST_DEPTH_CAMERA_KEYS,
)

from teleop_runtime.real_teleop.debug_timing_payload import (
    _channels_log_payload,
    _fault_signature,
    _faults_log_payload,
    _latency_items_log_payload,
    _latency_log_payload,
    _phase_timings_log_payload,
)

from teleop_runtime.real_teleop.debug_log_frames import (
    _backend_timing_log_payload,
    _build_debug_timing_log_frame,
    _build_ik_debug_log_frame,
    _camera_status_log_payload,
    _command_for_side,
)

from teleop_runtime.real_teleop.action_debug_payload import (
    _action_debug_active_sides,
    _action_debug_backend_snapshot_from_state,
    _action_debug_jitter_focus,
    _build_action_debug_metadata,
)

from teleop_runtime.real_teleop.operator_control_runtime import (
    RealTeleopControlInput,
    bind_legacy_support as _bind_real_teleop_gui_support,
)

_bind_real_teleop_gui_support(globals())
RealTeleopControlInput.__module__ = (
    "teleop_runtime.debug.run_vive_rm65_real_teleop"
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
















from teleop_runtime.real_teleop.startup_runtime import (
    CAMERA_STARTUP_SELF_CHECK_CONSECUTIVE_FRESH_SAMPLES,
    CAMERA_STARTUP_SELF_CHECK_DURATION_SEC,
    CAMERA_STARTUP_SELF_CHECK_FRESH_FRAME_MAX_AGE_SEC,
    CAMERA_STARTUP_SELF_CHECK_INTERVAL_SEC,
    _advance_startup_gui_qpos_maintenance,
    _arm_after_start_calibration,
    _finish_startup_gui_qpos_terminal,
    _log_dual_arm_readiness,
    _release_runtime_state_readers,
    _restore_init_arm_after_foot_pad_fallback,
    _run_camera_startup_self_check,
    _runtime_home_failure_message,
    _startup_gui_qpos_seed_command,
    _update_init_arm_arrival_from_hardware,
    _wait_for_real_teleop_start_request,
    bind_legacy_support as _bind_startup_runtime_support,
    cleanup_active_startup,
)

_bind_startup_runtime_support(globals())

from teleop_runtime.real_teleop.runtime import (
    bind_legacy_support as _bind_real_teleop_runtime_support,
    run_real_teleop,
)


def main() -> None:
    try:
        args = parse_args()
        _bind_real_teleop_runtime_support(globals())
        runtime_extension = build_real_agent_primitive_extension(args)
        args._real_teleop_runtime_extension = runtime_extension
        run_real_teleop(
            args,
            progress_application=build_real_teleop_progress_application(args),
            input_batch_prioritizer=prioritize_input_batch,
        )
    except KeyboardInterrupt:
        # Keep Ctrl+C a normal operator action.  The runtime's regular
        # ``finally`` handles the post-startup path; this fallback covers the
        # short pre-startup window where that ``finally`` is not yet active.
        cleanup_active_startup()
        print("\n[RealTeleop] 已收到 Ctrl+C，启动已安全取消。", file=sys.stderr)
    finally:
        # Also release the registry for non-interrupt startup exceptions.  It
        # is disarmed automatically once the normal runtime shutdown owns the
        # resources, so this is idempotent on the regular path.
        cleanup_active_startup()


if __name__ == "__main__":
    main()
