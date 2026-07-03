from __future__ import annotations

import argparse
import json
import queue
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
import re
import sys
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation

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
from ..collection.camera_preview import CameraPreviewConfig, CameraPreviewer
from ..collection.build_collect_components import CollectorComponents
from ..collection.observers import CollectionWriterObserver, CollectionWriterObserverConfig
from ..collection.preview_rate import PreviewRateLimiter
from ..collection.sample_assembler import CollectorSampleBuilder, CollectorSampleBuilderConfig
from ..collection.status_snapshot import (
    camera_debug_text as _collection_camera_debug_text,
    hand_debug_text as _collection_hand_debug_text,
    writer_debug_text as _collection_writer_debug_text,
)
from ..collection.writer_v21 import LeRobotV21Writer, LeRobotV21WriterConfig
from ..configs import (
    ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
    build_bimanual_pose_smoother,
    build_annotated_dual_rm65_calibration,
    load_default_pose_smoothing_config,
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
from ..defaults import DEFAULT_RM65_ARM_CONTROL_HZ, DEFAULT_TELEOP_SAMPLE_HZ
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
from ..rigs import BimanualRig
from ..safety import TeleopSafetyConfig
from ..schemas import EffectorCommand, TeleopCommand, merge_teleop_commands
from ..session import SessionStepResult, TeleopSession, TeleopSessionConfig
from ..sources.camera import RealSenseRGBSource, load_realsense_camera_config_map
from ..sources.hand import HandQ6Source
from ..sources.misc import KeyboardButtonSource
from ..sources.pose import VivePoseSource, ViveTrackerConfig
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
from ..gui.calibration_web_ui import CalibrationWebUi
from .log_utils import ChangeGate
from .run_vive_sapien_virtual import (
    RecoveryPlan,
    TerminalClutchInput,
    _active_sides_from_args,
    _advance_recovery_plan,
    _build_episode_frame,
    _build_renderer_debug_boxes,
    _build_renderer_debug_frames,
    _collision_debug_box_templates,
    _emit_collision_proxy_logs,
    _emit_home_alignment_logs,
    _emit_runtime_orientation_logs,
    _emit_tracker_axis_mapping_logs,
    _log,
    _manual_calibration_confirmation_required,
    _non_negative_int,
    _alpha_float,
    _parse_active_sides,
    _parse_arm_max_angvel,
    _parse_joint_order,
    _positive_float,
    _qpos_summary,
    _runtime_home_calibration,
    _start_recovery_plan,
    _state_qpos,
    _sync_dual_arm_retargeter_from_preview,
    _hand_render_qpos,
    _vive_tracker_local_translation_offset,
    _wait_for_calibration_start,
)
from rm65_inspire_tac.ik_tuning import load_ik_tuning_config
from rm65_inspire_tac.paths import require_asset_path
from rm65_inspire_tac.pipeline import (
    DEFAULT_LEFT_ARM_QPOS,
    DEFAULT_RIGHT_ARM_QPOS,
    DualArmSceneProfile,
    build_dual_arm_scene_profile,
)
from rm65_inspire_tac.rendering import SapienArmHandRenderer
from project_paths import calibration_profiles_dir, default_dataset_root


DEFAULT_CAMERA_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "devices" / "realsense.json"
)
DEFAULT_DATASET_ROOT = default_dataset_root()
DEFAULT_DMTAC_ROOT = Path(__file__).resolve().parents[2] / "hardware" / "DMTAC_gripper"
DEFAULT_CALIBRATION_PROFILE_DIR = calibration_profiles_dir()
DEFAULT_TASKS_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "tasks"
    / "vive_rm65_real_teleop_tasks.json"
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
    "output_dim": 1,
    "invert": False,
    "alpha": 1.0,
    "input_min": DEFAULT_GRIPPER_INPUT_MIN,
    "input_max": DEFAULT_GRIPPER_INPUT_MAX,
    "speed": 1.0,
    "torque": 0.2,
    "max_action": GRIPPER_ACTION_MAX,
    "send_threshold": 0.002,
    "command_hz": DEFAULT_DMTAC_GRIPPER_COMMAND_HZ,
    "continuous_grasp": False,
    "telemetry_hz": 10.0,
}
EPISODE_GUI_EVENTS = (
    "episode_start",
    "episode_stop",
)
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


@dataclass(frozen=True)
class RealTeleopComponents:
    session: TeleopSession
    active_sides: tuple[str, ...]
    left_command_backend: RM65ArmBackend | RM65ImpedanceArmBackend | None
    right_command_backend: RM65ArmBackend | RM65ImpedanceArmBackend | None
    left_preview_backend: RM65ArmBackend
    right_preview_backend: RM65ArmBackend
    left_state_backend: RM65ArmBackend | None
    right_state_backend: RM65ArmBackend | None
    left_hand_backend: InspireTacHandBackend | DMTacGripperBackend | None
    right_hand_backend: InspireTacHandBackend | DMTacGripperBackend | None
    initial_alignment: TeleopInitialAlignment | None
    loaded_initial_alignment: TeleopInitialAlignment | None
    logical_arm_ip_mapping: dict[str, str]
    collision_scene_profile: DualArmSceneProfile
    dual_arm_retargeter: DualArmCollisionAwareRetargeter | None
    collision_guard_requested: bool
    collision_guard_setup_error: str | None
    renderer: SapienArmHandRenderer | None
    keyboard_source: KeyboardButtonSource
    left_retargeter: PoseToArmRetargeter
    right_retargeter: PoseToArmRetargeter
    left_hand_retargeter: JointsToDexHandRetargeter | None
    right_hand_retargeter: JointsToDexHandRetargeter | None
    left_gripper_retargeter: ScalarToGripperRetargeter | None
    right_gripper_retargeter: ScalarToGripperRetargeter | None
    left_hand_control_via_rig: bool
    right_hand_control_via_rig: bool
    hand_backend_mode: str
    gripper_policy: dict[str, object]
    joint_distance_weights: np.ndarray
    arm_state_read_executor: ThreadPoolExecutor | None
    arm_loop_step_dt_tracker: dict[str, float | None]
    temporary_safety_limiter: TemporaryDualArmSafetyLimiter | None


@dataclass
class RealDebugTimingState:
    enabled: bool = False
    threshold_sec: float = 0.025
    report_interval_sec: float = 1.0
    last_report_at: float = 0.0
    suppressed_reports: int = 0


@dataclass(frozen=True)
class Q6HoldRecalibrationGestureConfig:
    enabled: bool = False
    hold_sec: float = 1.0
    close_threshold: float = 0.95
    open_threshold: float = 0.85


class Q6HoldRecalibrationGesture:
    EVENT_HOLD = "hold"
    EVENT_REARMED = "rearmed"
    EVENT_RESTART = "restart"

    _RUNNING_WAIT_CLOSE = "running_wait_close"
    _HOLD_WAIT_OPEN = "hold_wait_open"
    _HOLD_WAIT_CLOSE = "hold_wait_close"
    _RESTART_REQUESTED = "restart_requested"
    _RUNNING_WAIT_RELEASE_TO_REARM = "running_wait_release_to_rearm"

    def __init__(self, config: Q6HoldRecalibrationGestureConfig) -> None:
        self.config = config
        self._state = self._RUNNING_WAIT_CLOSE
        self._condition_since: float | None = None

    def reset_for_running(self, *, require_release: bool = False) -> None:
        self._state = self._RUNNING_WAIT_RELEASE_TO_REARM if bool(require_release) else self._RUNNING_WAIT_CLOSE
        self._condition_since = None

    def reset_for_hold(self) -> None:
        self._state = self._HOLD_WAIT_OPEN
        self._condition_since = None

    @staticmethod
    def q6_values_from_channel(channel: object | None) -> np.ndarray | None:
        if channel is None or not bool(getattr(channel, "valid", False)):
            return None
        try:
            values = np.asarray(getattr(channel, "value"), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if values.size < 6:
            return None
        tail = values[3:6]
        if not np.all(np.isfinite(tail)):
            return None
        return values

    @staticmethod
    def _tail(values: np.ndarray | None) -> np.ndarray | None:
        if values is None:
            return None
        tail = np.asarray(values, dtype=float).reshape(-1)[3:6]
        if tail.shape != (3,) or not np.all(np.isfinite(tail)):
            return None
        return tail

    def _both_closed(self, left_q6: np.ndarray | None, right_q6: np.ndarray | None) -> bool:
        left_tail = self._tail(left_q6)
        right_tail = self._tail(right_q6)
        if left_tail is None or right_tail is None:
            return False
        threshold = float(self.config.close_threshold)
        return bool(np.all(left_tail > threshold) and np.all(right_tail > threshold))

    def _both_released(self, left_q6: np.ndarray | None, right_q6: np.ndarray | None) -> bool:
        left_tail = self._tail(left_q6)
        right_tail = self._tail(right_q6)
        if left_tail is None or right_tail is None:
            return False
        close_threshold = float(self.config.close_threshold)
        release_threshold = min(float(self.config.open_threshold), close_threshold - 1e-6)
        return bool(np.all(left_tail < release_threshold) and np.all(right_tail < release_threshold))

    def _condition_sustained(self, condition: bool, *, now: float) -> bool:
        if not condition:
            self._condition_since = None
            return False
        if self._condition_since is None:
            self._condition_since = float(now)
        return float(now) - float(self._condition_since) >= max(0.0, float(self.config.hold_sec))

    def update(
        self,
        *,
        left_q6: np.ndarray | None,
        right_q6: np.ndarray | None,
        teleop_started: bool,
        hold_active: bool,
        now: float,
    ) -> str | None:
        if not bool(self.config.enabled) or not bool(teleop_started):
            self.reset_for_running()
            return None

        closed = self._both_closed(left_q6, right_q6)
        released = self._both_released(left_q6, right_q6)

        if not bool(hold_active):
            if self._state == self._RUNNING_WAIT_RELEASE_TO_REARM:
                if self._condition_sustained(released, now=now):
                    self.reset_for_running()
                return None
            if self._state != self._RUNNING_WAIT_CLOSE:
                self.reset_for_running(require_release=True)
                if released:
                    self.reset_for_running()
            if self._condition_sustained(closed, now=now):
                self._state = self._HOLD_WAIT_OPEN
                self._condition_since = None
                return self.EVENT_HOLD
            return None

        if self._state not in {
            self._HOLD_WAIT_OPEN,
            self._HOLD_WAIT_CLOSE,
            self._RESTART_REQUESTED,
        }:
            self._state = self._HOLD_WAIT_OPEN
            self._condition_since = None

        if self._state == self._HOLD_WAIT_OPEN:
            if self._condition_sustained(released, now=now):
                self._state = self._HOLD_WAIT_CLOSE
                self._condition_since = None
                return self.EVENT_REARMED
            return None

        if self._state == self._HOLD_WAIT_CLOSE:
            if self._condition_sustained(closed, now=now):
                self._state = self._RESTART_REQUESTED
                self._condition_since = None
                return self.EVENT_RESTART
            return None

        return None

    def mark_runtime_restarted(self) -> None:
        self.reset_for_running(require_release=True)


@dataclass(frozen=True)
class TemporaryDualArmSafetyConfig:
    enabled: bool = True
    active_sides: tuple[str, ...] = ("left", "right")
    max_xyz_step_m: float = 0.015
    max_rot_step_deg: float = 5.0
    max_joint_step_deg: float = 4.0
    command_scale: float = 1.0


class TemporaryDualArmSafetyLimiter:
    """Small removable output limiter for first dual-arm hardware bring-up."""

    def __init__(self, config: TemporaryDualArmSafetyConfig):
        self.config = config
        self._previous_by_effector: dict[str, np.ndarray] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and set(self.config.active_sides) == {"left", "right"})

    def sync_from_command(self, command: TeleopCommand | None) -> None:
        """Observe commands sent outside the session filter, such as recovery ramps."""
        if not self.enabled or command is None:
            return
        for effector_name, effector_command in command.commands.items():
            if effector_name in {"left_arm", "right_arm"}:
                self._sync_effector_command(effector_command)

    def filter_command(
        self,
        command: TeleopCommand,
        _observation,
        _now: float,
    ) -> TeleopCommand:
        if not self.enabled:
            return command
        filtered = TeleopCommand(timestamp=command.timestamp, metadata=dict(command.metadata))
        summary: dict[str, object] = {
            "enabled": True,
            "limited": False,
            "max_xyz_step_m": float(self.config.max_xyz_step_m),
            "max_rot_step_deg": float(self.config.max_rot_step_deg),
            "max_joint_step_deg": float(self.config.max_joint_step_deg),
            "command_scale": float(self.config.command_scale),
            "effectors": {},
        }
        for effector_name, effector_command in command.commands.items():
            next_command = effector_command.copy()
            if effector_name in {"left_arm", "right_arm"}:
                next_command, effector_summary = self._filter_arm_command(next_command)
                if effector_summary:
                    summary["effectors"][effector_name] = effector_summary
                    summary["limited"] = bool(summary["limited"] or effector_summary.get("limited"))
            filtered.commands[effector_name] = next_command
        filtered.metadata["temporary_dual_arm_safety"] = summary
        return filtered

    def _filter_arm_command(
        self,
        command: EffectorCommand,
    ) -> tuple[EffectorCommand, dict[str, object]]:
        if command.command_type == "cartesian_pose":
            return self._filter_cartesian_pose_command(command)
        if command.command_type == "joint_vector":
            return self._filter_joint_vector_command(command)
        return command, {
            "command_type": command.command_type,
            "limited": False,
            "reason": "unsupported_command_type",
        }

    def _filter_cartesian_pose_command(
        self,
        command: EffectorCommand,
    ) -> tuple[EffectorCommand, dict[str, object]]:
        try:
            target = as_pose_matrix(command.value)
        except Exception as exc:  # noqa: BLE001
            return command, {
                "command_type": command.command_type,
                "limited": False,
                "reason": f"invalid_pose:{type(exc).__name__}",
            }
        previous = self._previous_by_effector.get(command.effector_name)
        if previous is None:
            self._previous_by_effector[command.effector_name] = target.copy()
            command.metadata = {
                **dict(command.metadata),
                "temporary_dual_arm_safety_initialized": True,
            }
            return command, {
                "command_type": command.command_type,
                "limited": False,
                "initialized": True,
            }
        limited_pose, metadata = self._limit_pose(previous, target)
        self._previous_by_effector[command.effector_name] = limited_pose.copy()
        out = command.copy()
        out.value = limited_pose
        out.metadata = {
            **dict(out.metadata),
            "temporary_dual_arm_safety": metadata,
            "temporary_dual_arm_safety_limited": bool(metadata.get("limited", False)),
        }
        return out, {"command_type": command.command_type, **metadata}

    def _sync_effector_command(self, command: EffectorCommand) -> None:
        try:
            if command.command_type == "cartesian_pose":
                self._previous_by_effector[command.effector_name] = as_pose_matrix(command.value).copy()
            elif command.command_type == "joint_vector":
                self._previous_by_effector[command.effector_name] = (
                    np.asarray(command.value, dtype=float).reshape(-1).copy()
                )
        except Exception:
            return

    def _filter_joint_vector_command(
        self,
        command: EffectorCommand,
    ) -> tuple[EffectorCommand, dict[str, object]]:
        try:
            target = np.asarray(command.value, dtype=float).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            return command, {
                "command_type": command.command_type,
                "limited": False,
                "reason": f"invalid_joint_vector:{type(exc).__name__}",
            }
        previous = self._previous_by_effector.get(command.effector_name)
        if previous is None or previous.shape != target.shape:
            self._previous_by_effector[command.effector_name] = target.copy()
            command.metadata = {
                **dict(command.metadata),
                "temporary_dual_arm_safety_initialized": True,
            }
            return command, {
                "command_type": command.command_type,
                "limited": False,
                "initialized": True,
            }
        delta = target - previous
        scale = max(0.0, min(1.0, float(self.config.command_scale)))
        delta = delta * scale
        max_step_rad = np.deg2rad(max(0.0, float(self.config.max_joint_step_deg)))
        if max_step_rad > 0.0:
            delta = np.clip(delta, -max_step_rad, max_step_rad)
        output = previous + delta
        self._previous_by_effector[command.effector_name] = output.copy()
        raw_delta = target - previous
        limited = not np.allclose(output, target, atol=1e-10)
        metadata = {
            "limited": bool(limited),
            "raw_max_step_deg": float(np.rad2deg(np.max(np.abs(raw_delta)))) if raw_delta.size else 0.0,
            "applied_max_step_deg": float(np.rad2deg(np.max(np.abs(delta)))) if delta.size else 0.0,
            "max_joint_step_deg": float(self.config.max_joint_step_deg),
            "command_scale": float(self.config.command_scale),
        }
        out = command.copy()
        out.value = output
        out.metadata = {
            **dict(out.metadata),
            "temporary_dual_arm_safety": metadata,
            "temporary_dual_arm_safety_limited": bool(limited),
        }
        return out, {"command_type": command.command_type, **metadata}

    def _limit_pose(self, previous: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        output = np.asarray(previous, dtype=float).reshape(4, 4).copy()
        target_pose = np.asarray(target, dtype=float).reshape(4, 4)
        scale = max(0.0, min(1.0, float(self.config.command_scale)))
        translation_delta = (target_pose[:3, 3] - output[:3, 3]) * scale
        raw_translation_norm = float(np.linalg.norm(target_pose[:3, 3] - output[:3, 3]))
        max_xyz = max(0.0, float(self.config.max_xyz_step_m))
        applied_translation_norm = float(np.linalg.norm(translation_delta))
        if max_xyz > 0.0 and applied_translation_norm > max_xyz:
            translation_delta = translation_delta * (max_xyz / max(1e-12, applied_translation_norm))
            applied_translation_norm = max_xyz
        prev_rot = SciPyRotation.from_matrix(output[:3, :3])
        target_rot = SciPyRotation.from_matrix(target_pose[:3, :3])
        delta_rot = target_rot * prev_rot.inv()
        raw_rot_deg = float(np.rad2deg(delta_rot.magnitude()))
        rotvec = delta_rot.as_rotvec() * scale
        applied_rot_deg = float(np.rad2deg(np.linalg.norm(rotvec)))
        max_rot_deg = max(0.0, float(self.config.max_rot_step_deg))
        if max_rot_deg > 0.0 and applied_rot_deg > max_rot_deg:
            rotvec = rotvec * (max_rot_deg / max(1e-12, applied_rot_deg))
            applied_rot_deg = max_rot_deg
        output[:3, 3] = output[:3, 3] + translation_delta
        output[:3, :3] = (SciPyRotation.from_rotvec(rotvec) * prev_rot).as_matrix()
        limited = bool(
            not np.allclose(output[:3, 3], target_pose[:3, 3], atol=1e-10)
            or not np.allclose(output[:3, :3], target_pose[:3, :3], atol=1e-10)
        )
        return output, {
            "limited": limited,
            "raw_translation_step_m": raw_translation_norm,
            "applied_translation_step_m": applied_translation_norm,
            "raw_rotation_step_deg": raw_rot_deg,
            "applied_rotation_step_deg": applied_rot_deg,
            "max_xyz_step_m": float(self.config.max_xyz_step_m),
            "max_rot_step_deg": float(self.config.max_rot_step_deg),
            "command_scale": float(self.config.command_scale),
        }


def _non_negative_float(text: str) -> float:
    value = float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def _movement_multiple_arg(text: str) -> float:
    value = float(text)
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("movement_multiple must be a positive finite value")
    return value


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


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _dmtac_worker_cpu_affinity(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in re.split(r"[,\s]+", str(text)) if part.strip()]
    if not parts:
        return ()
    cpus: list[int] = []
    for part in parts:
        try:
            cpu = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("CPU affinity must be a comma-separated integer list") from exc
        if cpu < 0:
            raise argparse.ArgumentTypeError("CPU affinity indices must be >= 0")
        cpus.append(cpu)
    return tuple(dict.fromkeys(cpus))


def _dmtac_worker_nice(text: str) -> int:
    value = int(text)
    if not 0 <= value <= 19:
        raise argparse.ArgumentTypeError("worker nice must be in [0, 19]")
    return value


def _parse_dmtac_tactile_fields(text: str) -> tuple[str, ...]:
    try:
        return normalize_dmtac_tactile_fields(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _hand_backend_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "hand_backend", HAND_BACKEND_INSPIRE) or HAND_BACKEND_INSPIRE)
    if mode not in HAND_BACKEND_CHOICES:
        raise ValueError(f"Unsupported hand backend {mode!r}; expected one of {HAND_BACKEND_CHOICES}.")
    return mode


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
    policy["max_action"] = float(policy["max_action"])
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
    if not 0 <= int(policy["output_dim"]) < 6:
        raise ValueError("gripper output_dim must be in [0, 5].")
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
    policy["max_action"] = float(policy["max_action"])
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
    if not 0 <= int(policy["output_dim"]) < 6:
        raise ValueError("gripper output_dim must be in [0, 5].")
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
    if _hand_backend_mode(args) != HAND_BACKEND_DMTAC_GRIPPER:
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
    hand_backend_mode = _hand_backend_mode(args)
    require_dmtac_stack = hand_backend_mode == HAND_BACKEND_DMTAC_GRIPPER
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
        require_q6=require_dmtac_stack,
        require_dmtac_gripper=require_dmtac_stack,
        require_dmtac_tactile=bool(require_dmtac_stack and require_tactile),
        auto_configure=bool(getattr(args, "device_auto_configure", True)),
        assume_yes=bool(getattr(args, "device_preflight_yes", False)),
        fix_permissions=bool(getattr(args, "device_fix_permissions", False)),
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
    if result.q6_ports:
        args.q6_config_loaded_ports = dict(result.q6_ports)
        for side, port in result.q6_ports.items():
            if side in active_sides:
                setattr(args, f"{side}_hand_port", str(port))


def _argv_has_option(raw_argv: Sequence[str], option: str) -> bool:
    return any(str(item) == option or str(item).startswith(f"{option}=") for item in raw_argv)


def _apply_real_loop_timing_defaults(
    args: argparse.Namespace,
    *,
    raw_argv: Sequence[str],
) -> None:
    hz = max(1e-6, float(getattr(args, "hz", DEFAULT_TELEOP_SAMPLE_HZ)))
    if not _argv_has_option(raw_argv, "--minimum-control-hz"):
        setattr(args, "minimum_control_hz", max(1.0, min(15.0, 0.9 * hz)))
        setattr(args, "minimum_control_hz_auto", True)
    else:
        setattr(args, "minimum_control_hz_auto", False)

    if not _argv_has_option(raw_argv, "--arm-step-dt-max-scale"):
        arm_control_hz = max(
            1e-6,
            float(getattr(args, "arm_control_hz", DEFAULT_RM65_ARM_CONTROL_HZ)),
        )
        required_scale = 1.1 * arm_control_hz / hz
        setattr(
            args,
            "arm_step_dt_max_scale",
            max(float(getattr(args, "arm_step_dt_max_scale", 3.0)), required_scale),
        )
        setattr(args, "arm_step_dt_max_scale_auto", True)
    else:
        setattr(args, "arm_step_dt_max_scale_auto", False)


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
    if _hand_backend_mode(args) != HAND_BACKEND_DMTAC_GRIPPER:
        setattr(args, "_dmtac_backend_configs", {})
        setattr(args, "dmtac_config_loaded_sides", [])
        return {}

    policy = _resolved_gripper_policy(args)
    policies_by_side = _resolved_gripper_policies_by_side(args, active_sides)
    configs: dict[str, DMTacGripperBackendConfig] = {}
    if bool(getattr(args, "dmtac_default_config", True)):
        try:
            assignment = load_dmtac_hardware_assignment(
                str(getattr(args, "dmtac_config", "latest") or "latest"),
                validate_current_usb=bool(getattr(args, "dmtac_validate_current_usb", True)),
                allow_missing=bool(getattr(args, "dmtac_allow_missing", False)),
                assume_yes_for_extra=bool(getattr(args, "dmtac_assume_yes_for_extra", True)),
            )
            for side in active_sides:
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

    for side in active_sides:
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
        if side in active_sides and (config.connect_hardware or config.connect_tactile)
    }
    setattr(args, "_dmtac_backend_configs", dict(resolved))
    setattr(args, "dmtac_config_loaded_sides", sorted(resolved))
    return dict(resolved)


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
        "--hand-backend",
        choices=HAND_BACKEND_CHOICES,
        default=HAND_BACKEND_INSPIRE,
        help=(
            "Hand/gripper backend type. Use dmtac-gripper to map Q6 to the DMTAC "
            "1D gripper backend and expose browser GUI control/status."
        ),
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
            "DM-Tac fields to read/store, e.g. 'distributed_force' for the original path "
            "or 'raw' for raw-image-only collection. Valid fields: "
            + ", ".join(DMTAC_TACTILE_FIELDS)
        ),
    )
    parser.add_argument(
        "--dmtac-tactile-compute-profile",
        choices=("dist-only", "legacy", "requested", "shear-depth"),
        default=None,
        help="SDK compute channels for DMTAC telemetry. Default follows backend config.",
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
        choices=["absolute", "delta_clutch"],
        default="absolute",
        help="Arm retargeting mode. Use delta_clutch with --keyboard-clutch for relative control.",
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
        "--impedance-mode",
        choices=["pose"],
        default="pose",
        help="Impedance controller reference mode. Pose is the default and matches the unified arm action.",
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
            "auto prefers auxiliary when present, then left_wrist, then right_wrist."
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
        default=0.1,
        help="Temporary dual-arm safety max per-frame TCP translation step in meters.",
    )
    parser.add_argument(
        "--dual-arm-temp-max-rot-step-deg",
        type=_positive_float,
        default=15.0,
        help="Temporary dual-arm safety max per-frame TCP rotation step in degrees.",
    )
    parser.add_argument(
        "--dual-arm-temp-max-joint-step-deg",
        type=_positive_float,
        default=15.0,
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
        help="Outer action/update rate. Defaults to 15 Hz for single-arm and bimanual teleop.",
    )
    parser.add_argument(
        "--arm-control-hz",
        type=_positive_float,
        default=DEFAULT_RM65_ARM_CONTROL_HZ,
        help="RM65 backend control/streaming rate after each outer action update. Defaults to 40 Hz.",
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
            "min(15, 0.9 * --hz) so a 15Hz loop is not flagged on normal scheduler jitter."
        ),
    )
    parser.add_argument(
        "--status-sec",
        type=_positive_float,
        default=1.0,
        help="Heartbeat interval for teleop status logs.",
    )
    parser.add_argument(
        "--debug-timing",
        "--debug-frozen",
        dest="debug_timing",
        action="store_true",
        help=(
            "Enable real-loop phase timing diagnostics. Slow frames print session, IK, "
            "render, episode-record, and status timing breakdowns."
        ),
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
        "--dual-arm-ik-hz",
        type=_positive_float,
        default=15.0,
        help="Target dual-arm MultiTask IK update frequency used for solver dt and default time budget.",
    )
    parser.add_argument(
        "--left-right-base-distance",
        type=float,
        default=ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
        help="Annotated dual-arm base distance used by the SAPIEN scene and default calibration.",
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
    parser.add_argument("--keyboard-clutch", action="store_true", help="Enable terminal clutch toggles: z/x/c/r, q to quit")
    parser.add_argument("--no-record", action="store_true", help="Disable JSONL recording for this session")
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
            "Optional base directory for debug JSONL session output. When set, session JSONL is "
            "written to <session-record-root>/<timestamp>_vive_rm65_real_teleop."
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
    parser.add_argument("--write-video", dest="write_video", action="store_true")
    parser.add_argument("--no-video", dest="write_video", action="store_false")
    parser.set_defaults(write_video=True)
    parser.add_argument(
        "--tactile-storage",
        choices=("external", "parquet"),
        default="external",
        help="How high-dimensional tactile frames are stored in the formal dataset.",
    )
    parser.add_argument("--external-tactile", dest="tactile_storage", action="store_const", const="external")
    parser.add_argument(
        "--writer-workers",
        type=_positive_int,
        default=4,
        help="Number of async formal dataset writer workers.",
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
        help="Disable episode trajectory recording support for this run.",
    )
    parser.add_argument(
        "--episode-root",
        type=str,
        default="",
        help=(
            "Optional directory used to store episode_XXXXXX subdirectories. "
            "Defaults to <session_record_root>/trajectories when session recording is enabled; "
            "overrides --dataset-root for episode trajectory files only."
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
        help="Optional calibration profile to load before runtime home calibration.",
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
        help="Optional RM65 IK tuning JSON. Defaults to teleop_runtime/configs/rm65_ik_tuning.json.",
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
            "If omitted, real teleop uses max(default, 1.1 * --arm-control-hz / --hz) so a "
            "15Hz action loop with a 50Hz arm loop is not capped as if too few inner cycles elapsed."
        ),
    )
    parser.add_argument(
        "--arm-step-dt-stall-reset-sec",
        type=_positive_float,
        default=0.25,
        help="Reset actual-dt scaling back to the nominal control dt after longer send stalls.",
    )
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
    raw_arg_set = set(raw_argv)
    _apply_real_loop_timing_defaults(args, raw_argv=raw_argv)
    args._dataset_name_explicit = _argv_has_option(raw_argv, "--dataset-name")
    try:
        _resolve_movement_multiple_args(args, raw_arg_set=raw_arg_set)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    _apply_formal_dataset_layout(args)
    args.gui_requested = bool(getattr(args, "open_browser_gui", True)) and not (
        "--no-gui" in raw_arg_set or "--headless" in raw_arg_set
    )
    if not bool(args.headless):
        args.skip_renderer = False
    if int(args.tactile_vis_size) <= 0:
        parser.error("--tactile-vis-size must be positive.")
    if bool(args.no_camera) and bool(args.vis_cam):
        parser.error("--vis_cam requires cameras; remove --no-camera.")
    args.requested_tcp_mode = str(getattr(args, "tcp_mode", TOOL_TCP_FRAME))
    args.tcp_mode = TOOL_TCP_FRAME
    active_sides = _active_sides_from_args(args)
    try:
        _hand_backend_mode(args)
        if _hand_backend_mode(args) == HAND_BACKEND_DMTAC_GRIPPER:
            _resolved_gripper_policy(args)
        _apply_default_arm_ips(args, active_sides=active_sides)
        _validate_required_arm_ips(args, active_sides)
        _save_explicit_arm_ips_latest(args, raw_argv=raw_argv)
        _run_gui_device_preflight(args, active_sides=active_sides, parser=parser)
        _apply_default_q6_ports(args, parser=parser)
    except ValueError as exc:
        parser.error(str(exc))
    if bool(args.visualize_tactile):
        if _hand_backend_mode(args) == HAND_BACKEND_DMTAC_GRIPPER:
            if getattr(args, "dmtac_tactile", None) is False:
                parser.error(
                    "--visualize-tactile with --hand-backend dmtac-gripper requires "
                    "DMTAC tactile reads; remove --dmtac-no-tactile."
                )
            args.dmtac_tactile = True
        else:
            tactile_regions = _parse_hand_tactile_regions(args.hand_tactile_regions)
            if not tactile_regions:
                parser.error(
                    "--visualize-tactile requires non-empty --hand-tactile-regions, "
                    "for example 'thumb_1,index_1,palm' or 'all'."
                )
            missing_hand_ip_sides = [
                side for side in active_sides if not str(getattr(args, f"{side}_hand_ip") or "")
            ]
            if missing_hand_ip_sides:
                parser.error(
                    "--visualize-tactile requires real InspireTac hand IPs for active sides; "
                    f"missing={missing_hand_ip_sides}."
                )
    return args


def _resolve_logical_side_assignments(args: argparse.Namespace) -> dict[str, tuple[str, str]]:
    return {
        "left_arm_ip": ("right" if args.swap_arm_sides else "left", args.left_arm_ip),
        "right_arm_ip": ("left" if args.swap_arm_sides else "right", args.right_arm_ip),
    }


def _opposite_side(side: str) -> str:
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    raise ValueError(f"side must be left or right, got {side!r}")


def _tracker_source_side_for_logical_side(args: argparse.Namespace, logical_side: str) -> str:
    side = str(logical_side)
    return _opposite_side(side) if bool(getattr(args, "swap_operator_sides", False)) else side


def _tracker_name_for_source_side(args: argparse.Namespace, source_side: str) -> str:
    if source_side == "left":
        return str(args.left_tracker_name)
    if source_side == "right":
        return str(args.right_tracker_name)
    raise ValueError(f"source_side must be left or right, got {source_side!r}")


def _tracker_role_for_source_side(source_side: str) -> str:
    if source_side == "left":
        return "left_wrist"
    if source_side == "right":
        return "right_wrist"
    raise ValueError(f"source_side must be left or right, got {source_side!r}")


def _tracker_legacy_index_for_source_side(source_side: str) -> int:
    if source_side == "left":
        return 0
    if source_side == "right":
        return 1
    raise ValueError(f"source_side must be left or right, got {source_side!r}")


def _build_vive_tracker_configs_for_args(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> list[ViveTrackerConfig]:
    tracker_configs: list[ViveTrackerConfig] = []
    if "left" in active_sides:
        source_side = _tracker_source_side_for_logical_side(args, "left")
        tracker_configs.append(
            ViveTrackerConfig(
                name=_tracker_name_for_source_side(args, source_side),
                side="left",
                role=_tracker_role_for_source_side(source_side),
                channel_name="left_wrist_pose",
                legacy_index=_tracker_legacy_index_for_source_side(source_side),
            )
        )
    if "right" in active_sides:
        source_side = _tracker_source_side_for_logical_side(args, "right")
        tracker_configs.append(
            ViveTrackerConfig(
                name=_tracker_name_for_source_side(args, source_side),
                side="right",
                role=_tracker_role_for_source_side(source_side),
                channel_name="right_wrist_pose",
                legacy_index=_tracker_legacy_index_for_source_side(source_side),
            )
        )
    if args.aux_tracker_name:
        tracker_configs.append(
            ViveTrackerConfig(
                name=args.aux_tracker_name,
                side=None,
                role="auxiliary",
                channel_name="auxiliary_tracker_pose",
                legacy_index=2,
                critical=False,
            )
        )
    return tracker_configs


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


def _pose_xyz_rpy_summary(
    channel: object | None,
    *,
    now: float,
    enabled: bool = True,
) -> dict[str, object]:
    if not enabled:
        return {
            "enabled": False,
            "valid": False,
            "xyz_m": None,
            "rpy_deg": None,
            "age_sec": None,
            "tracker_name": "",
            "tracker_role": "",
            "seq": None,
        }
    if channel is None:
        return {
            "enabled": True,
            "valid": False,
            "xyz_m": None,
            "rpy_deg": None,
            "age_sec": None,
            "tracker_name": "",
            "tracker_role": "",
            "seq": None,
        }
    metadata = getattr(channel, "metadata", None) or {}
    pose = as_pose_matrix(getattr(channel, "value"))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpy_deg = SciPyRotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
    except ValueError:
        rpy_deg = np.full(3, np.nan, dtype=float)
    rpy_deg = np.round(rpy_deg.astype(float), 1)
    rpy_values = [float(value) if np.isfinite(value) else None for value in rpy_deg]
    timestamp = metadata.get("receiver_timestamp")
    if timestamp is None:
        timestamp = getattr(channel, "timestamp", None)
    age_sec = max(0.0, now - float(timestamp)) if timestamp is not None else None
    return {
        "enabled": True,
        "valid": bool(getattr(channel, "valid", False)),
        "xyz_m": np.round(pose[:3, 3].astype(float), 4).tolist(),
        "rpy_deg": rpy_values,
        "age_sec": round(age_sec, 3) if age_sec is not None else None,
        "tracker_name": str(metadata.get("tracker_name") or ""),
        "tracker_role": str(metadata.get("tracker_role") or ""),
        "seq": metadata.get("seq"),
    }


def _raw_tracker_pose_state_from_channels(
    channels: dict[str, object],
    *,
    active_sides: tuple[str, ...],
    now: float,
) -> dict[str, object]:
    active = set(active_sides)
    return {
        "left": _pose_xyz_rpy_summary(
            channels.get("left_wrist_pose"),
            now=now,
            enabled="left" in active,
        ),
        "right": _pose_xyz_rpy_summary(
            channels.get("right_wrist_pose"),
            now=now,
            enabled="right" in active,
        ),
    }


def _publish_raw_tracker_pose_state(
    ui: CalibrationWebUi | None,
    channels: dict[str, object],
    *,
    active_sides: tuple[str, ...],
    now: float | None = None,
) -> None:
    if ui is None:
        return
    current_time = time.time() if now is None else float(now)
    ui.publish(
        raw_tracker_poses=_raw_tracker_pose_state_from_channels(
            channels,
            active_sides=active_sides,
            now=current_time,
        )
    )


def _vive_calibration_tracker_roles(active_sides: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{side}_wrist" for side in active_sides if side in {"left", "right"})


def _vive_calibration_profile_ready(
    calibration: TeleopCalibration | None,
    *,
    active_sides: tuple[str, ...],
) -> bool:
    if calibration is None:
        return False
    if calibration.world_to_desk_pose() is None:
        return False
    roles = _vive_calibration_tracker_roles(active_sides)
    return all(role in calibration.tracker_offsets for role in roles)


def _vive_calibration_profile_status_text(
    calibration: TeleopCalibration | None,
    *,
    active_sides: tuple[str, ...],
) -> str:
    if calibration is None:
        return "calibration: unavailable"
    desk_ready = calibration.world_to_desk_pose() is not None
    roles = _vive_calibration_tracker_roles(active_sides)
    missing_roles = [role for role in roles if role not in calibration.tracker_offsets]
    offsets_text = "ok" if not missing_roles else "missing " + ",".join(missing_roles)
    return f"desk={'ok' if desk_ready else 'missing'} tracker_offsets={offsets_text}"


def _vive_calibration_profile_ui_state(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    status: str = "",
    path: str | Path | None = None,
    loaded: bool | None = None,
) -> dict[str, object]:
    calibration = components.session.calibration
    profile_path = (
        str(path)
        if path is not None
        else str(getattr(args, "gui_last_vive_calibration_profile", "") or getattr(args, "load_profile", "") or "")
    )
    status_text = _vive_calibration_profile_status_text(
        calibration,
        active_sides=components.active_sides,
    )
    if status:
        status_text = f"{status}\n{status_text}"
    if loaded is None:
        loaded = bool(str(getattr(args, "load_profile", "") or "").strip())
    return {
        "vive_calibration_profile_path": profile_path,
        "vive_calibration_profile_status": status_text,
        "vive_calibration_profile_ready": _vive_calibration_profile_ready(
            calibration,
            active_sides=components.active_sides,
        ),
        "vive_calibration_profile_loaded": bool(loaded),
    }


def _latest_vive_calibration_profile_path() -> Path | None:
    if not DEFAULT_CALIBRATION_PROFILE_DIR.exists():
        return None
    candidates: list[Path] = []
    seen: set[Path] = set()
    for pattern in (
        "*_vive_rm65_real_teleop.json",
        "*_vive_rm65_virtual_desk.json",
    ):
        for path in DEFAULT_CALIBRATION_PROFILE_DIR.glob(pattern):
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


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
    for candidate in ("auxiliary_tracker_pose", "left_wrist_pose", "right_wrist_pose"):
        channel = channels.get(candidate)
        if channel is not None and bool(getattr(channel, "valid", False)):
            return candidate
    if str(getattr(args, "aux_tracker_name", "") or "").strip():
        return "auxiliary_tracker_pose"
    return "left_wrist_pose"


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
    ui.append_message(stage_name, "点击“开始采集”或按 c 开始定时采集；r 重置当前阶段，q 取消。")
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
            ui.append_message(stage_name, "当前阶段已重置。")
        if "c" in events or "p" in events:
            if not capturing and not preparing:
                poses = []
                preparing = True
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
        _publish_vive_capture_state(
            ui,
            args,
            components,
            channels,
            now=loop_start,
            stage=stage_name,
            active_controls=["c", "r", "q"] if not capture_active else ["r", "q"],
            control_hint=direction_hint,
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
                ui.append_message(stage_name, f"样本不足 {len(poses)}/{int(min_samples)}，请按 c 重新采集。")
                poses = []
                min_samples_reported = False
                waiting_for_samples_reported = False
                last_sample_key = None
            else:
                error = validator(poses) if validator is not None else None
                if error is None:
                    ui.append_message(stage_name, f"采集接受：samples={len(poses)} hz={effective_hz:.1f}。")
                    return poses
                ui.append_message(stage_name, f"采集质量不合格：{error}。请按 c 重新采集。")
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
    ui.append_message(stage_id, "点击“开始采集”或按 c 开始定时采集；这里只写 tracker offset，不保存 runtime Home。")
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
        if "c" in events or "p" in events:
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
            if right_valid:
                right_last_sample_key = _channel_sample_key(right_channel)
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
            active_controls=["c", "r", "q"] if not capture_active else ["r", "q"],
            control_hint="左右手中立姿态；该阶段只标定 tracker offset，不保存 runtime Home。",
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
                f"need>={min_samples} left_hz={left_hz:.1f} right_hz={right_hz:.1f}，请按 c 重采。",
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

    channels = source.get_latest_channels()
    desk_channel_name = _resolve_virtual_desk_channel_name(args, channels)
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
    profile.save_json(output_path)
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


def _start_camera_sources(camera_sources: Mapping[str, RealSenseRGBSource]) -> None:
    for camera_key, source in camera_sources.items():
        _log("Camera", f"starting camera {camera_key}")
        source.start()


def _close_camera_sources(camera_sources: Mapping[str, RealSenseRGBSource]) -> None:
    for source in camera_sources.values():
        try:
            source.close()
        except Exception:
            pass


def _parse_hand_tactile_regions(value: object) -> tuple[str, ...]:
    return parse_tactile_regions(value)


def _arm_control_mode_arg(args: argparse.Namespace) -> str:
    return str(getattr(args, "arm_control_mode", "impedance"))


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


def _expand_cli_path(path_text: str) -> Path:
    return Path(os.path.expandvars(str(path_text))).expanduser()


def _sanitize_dataset_name_for_path(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "task"


def _default_task_id_from_name(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "task"


def _load_task_registry(args: argparse.Namespace) -> list[dict[str, str]]:
    cached = getattr(args, "_task_registry", None)
    if isinstance(cached, list):
        return [dict(item) for item in cached if isinstance(item, Mapping)]
    path_text = str(getattr(args, "task_config", "") or "").strip()
    if not path_text:
        setattr(args, "_task_registry", [])
        return []
    path = _expand_cli_path(path_text)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        setattr(args, "_task_registry", [])
        return []
    if not isinstance(payload, Mapping):
        raise ValueError(f"Task config {path} must contain a JSON object.")
    raw_tasks = payload.get("tasks", [])
    if isinstance(raw_tasks, Mapping):
        iterable = []
        for key, value in raw_tasks.items():
            if isinstance(value, Mapping):
                item = dict(value)
                item.setdefault("key", str(key))
                iterable.append(item)
    elif isinstance(raw_tasks, Sequence) and not isinstance(raw_tasks, (str, bytes)):
        iterable = list(raw_tasks)
    else:
        raise ValueError(f"Task config {path} field 'tasks' must be an array or object.")
    tasks: list[dict[str, str]] = []
    for raw in iterable:
        if not isinstance(raw, Mapping):
            continue
        key = str(raw.get("key") or raw.get("task_id") or raw.get("gripper_task") or "").strip()
        english_name = str(raw.get("english_name") or raw.get("name_en") or key).strip()
        chinese_name = str(raw.get("chinese_name") or raw.get("name_zh") or "").strip()
        gripper_task = str(raw.get("gripper_task") or key).strip()
        if not key:
            continue
        tasks.append(
            {
                "key": key,
                "english_name": english_name or key,
                "chinese_name": chinese_name,
                "description": str(raw.get("description") or "").strip(),
                "gripper_task": gripper_task,
            }
        )
    setattr(args, "_task_registry", [dict(item) for item in tasks])
    return tasks


def _task_definition_for_key(args: argparse.Namespace, key: str) -> dict[str, str] | None:
    key_text = str(key or "").strip()
    if not key_text:
        return None
    for item in _load_task_registry(args):
        if key_text in {item.get("key", ""), item.get("gripper_task", "")}:
            return dict(item)
    return None


def _resolve_formal_dataset_layout(args: argparse.Namespace) -> tuple[Path, str, str, str, str, str]:
    configured_task_id = str(getattr(args, "collection_task_id", "") or "").strip()
    gripper_task = str(getattr(args, "gripper_task", "") or "").strip()
    task_def = _task_definition_for_key(args, configured_task_id or gripper_task)
    default_task_name = ""
    default_task_name_zh = ""
    default_task_description = ""
    default_task_id = gripper_task
    if task_def is not None:
        default_task_id = str(task_def.get("key") or default_task_id).strip()
        default_task_name = str(task_def.get("english_name") or "").strip()
        default_task_name_zh = str(task_def.get("chinese_name") or "").strip()
        default_task_description = (
            str(task_def.get("description") or "").strip()
            or default_task_name_zh
        )
    task_name = (
        str(getattr(args, "collection_task_name", "") or "").strip()
        or default_task_name
        or gripper_task
        or "vive_rm65_real_teleop"
    )
    task_id = (
        configured_task_id
        or default_task_id
        or _default_task_id_from_name(task_name)
    )
    task_name_zh = (
        str(getattr(args, "collection_task_name_zh", "") or "").strip()
        or default_task_name_zh
    )
    task_description = (
        str(getattr(args, "collection_task_description", "") or "").strip()
        or default_task_description
        or task_name_zh
        or task_name
    )
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    dataset_dir_name = f"{date_prefix}_{_sanitize_dataset_name_for_path(task_name)}"
    dataset_root = _expand_cli_path(str(args.dataset_root)) / dataset_dir_name
    dataset_name_text = str(getattr(args, "dataset_name", "") or "").strip()
    explicit_dataset_name = getattr(args, "_dataset_name_explicit", None)
    dataset_name_is_explicit = (
        bool(dataset_name_text)
        if explicit_dataset_name is None
        else bool(explicit_dataset_name)
    )
    dataset_name = dataset_name_text if dataset_name_is_explicit and dataset_name_text else dataset_dir_name
    return dataset_root, task_name, task_name_zh, task_id, task_description, dataset_name


def _apply_formal_dataset_layout(args: argparse.Namespace) -> None:
    if not hasattr(args, "_dataset_name_explicit"):
        args._dataset_name_explicit = bool(str(getattr(args, "dataset_name", "") or "").strip())
    dataset_base_root = _expand_cli_path(str(args.dataset_root))
    (
        dataset_root,
        task_name,
        task_name_zh,
        task_id,
        task_description,
        dataset_name,
    ) = _resolve_formal_dataset_layout(args)
    args.dataset_base_root = str(dataset_base_root)
    args.dataset_root = str(dataset_root)
    args.dataset_dir_name = dataset_root.name
    args.collection_task_name = task_name
    args.collection_task_name_zh = task_name_zh
    args.collection_task_id = task_id
    args.collection_task_description = task_description
    args.dataset_name = dataset_name


def _refresh_formal_dataset_layout(args: argparse.Namespace) -> None:
    base = str(getattr(args, "dataset_base_root", "") or "").strip()
    if base:
        args.dataset_root = base
    _apply_formal_dataset_layout(args)


def _collection_task_ui_state(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset_base_root": str(getattr(args, "dataset_base_root", "") or getattr(args, "dataset_root", "") or ""),
        "dataset_root": str(getattr(args, "dataset_root", "") or ""),
        "dataset_dir_name": str(getattr(args, "dataset_dir_name", "") or ""),
        "collection_task_id": str(getattr(args, "collection_task_id", "") or ""),
        "collection_task_name": str(getattr(args, "collection_task_name", "") or ""),
        "collection_task_name_zh": str(getattr(args, "collection_task_name_zh", "") or ""),
        "collection_task_description": str(getattr(args, "collection_task_description", "") or ""),
        "gripper_task": str(getattr(args, "gripper_task", "") or ""),
        "task_registry": _load_task_registry(args),
    }


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


def _session_recording_config(args: argparse.Namespace) -> RecordingConfig:
    return RecordingConfig(
        enabled=True,
        save_commands=not bool(getattr(args, "no_session_record_commands", False)),
        flush_every=max(1, int(getattr(args, "session_record_flush_every", 1) or 1)),
        compact_json=bool(getattr(args, "session_record_compact_json", False)),
    )


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


def _build_episode_metadata(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    start_reason: str,
) -> dict[str, object]:
    calibration = components.session.calibration
    assert calibration is not None
    session_record_root = (
        None if components.session.recorder is None else str(components.session.recorder.root_dir)
    )
    episode_root = _episode_root_dir(args, components)
    return {
        "source_entrypoint": "teleop_runtime.debug.run_vive_rm65_real_teleop",
        "start_reason": start_reason,
        "control_mode": args.control_mode,
        "arm_control_mode": _arm_control_mode_arg(args),
        "impedance_mode": _impedance_mode_arg(args),
        "strict_tool_profile": _strict_tool_profile_arg(args),
        "force_tare_sec": _force_tare_sec_arg(args),
        "arm_mode": "real",
        "active_sides": list(components.active_sides),
        "runtime_arm_state_source": "last_commanded_action",
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
        "movement_multiple": _movement_multiple_value(args),
        "translation_scale": float(args.translation_scale),
        "left_right_base_distance_m": float(args.left_right_base_distance),
        "collision_scene_profile": args.collision_scene_profile,
        "collision_avoidance_mode": args.collision_avoidance_mode,
        "collision_guard_requested": bool(components.collision_guard_requested),
        "collision_guard_enabled": bool(components.dual_arm_retargeter is not None),
        "dataset_root": str(getattr(args, "dataset_root", "") or "") or None,
        "session_record_root": session_record_root,
        "session_record_commands": not bool(getattr(args, "no_session_record_commands", False)),
        "session_record_flush_every": int(getattr(args, "session_record_flush_every", 1) or 1),
        "session_record_compact_json": bool(getattr(args, "session_record_compact_json", False)),
        "episode_root": None if episode_root is None else str(episode_root),
        "load_profile": args.load_profile or None,
        "load_initial_alignment": args.load_initial_alignment or None,
        "logical_arm_ip_mapping": dict(components.logical_arm_ip_mapping),
        "left_hand_ip": args.left_hand_ip or None,
        "right_hand_ip": args.right_hand_ip or None,
        "left_hand_port": getattr(args, "left_hand_port", "") or None,
        "right_hand_port": getattr(args, "right_hand_port", "") or None,
        "hand_backend": _hand_backend_mode(args),
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
            if _hand_backend_mode(args) == HAND_BACKEND_DMTAC_GRIPPER
            else {}
        ),
        "gripper_policy_by_side": (
            _resolved_gripper_policies_by_side(args, components.active_sides)
            if _hand_backend_mode(args) == HAND_BACKEND_DMTAC_GRIPPER
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


@dataclass
class RealDatasetFinalizeTask:
    episode_index: int
    frame_count: int
    reason: str
    started_at: float
    thread: threading.Thread | None = None
    done: bool = False
    success: bool = False
    error: str = ""
    completed_at: float | None = None

    def mark_done(self, *, success: bool, error: str = "") -> None:
        self.done = True
        self.success = bool(success)
        self.error = str(error or "")
        self.completed_at = time.time()


def _collection_backends_from_components(components: RealTeleopComponents) -> dict[str, Any]:
    backends: dict[str, Any] = {}
    left_arm = components.left_state_backend or components.left_command_backend
    right_arm = components.right_state_backend or components.right_command_backend
    if left_arm is not None:
        backends["left_arm"] = left_arm
    if right_arm is not None:
        backends["right_arm"] = right_arm
    if components.left_hand_backend is not None:
        backends["left_hand"] = components.left_hand_backend
    if components.right_hand_backend is not None:
        backends["right_hand"] = components.right_hand_backend
    return backends


def _collection_sources_from_components(
    components: RealTeleopComponents,
    camera_sources: Mapping[str, RealSenseRGBSource],
) -> dict[str, Any]:
    sources: dict[str, Any] = {components.keyboard_source.name: components.keyboard_source}
    for camera_key, source in camera_sources.items():
        sources[f"camera.{camera_key}"] = source
    return sources


def _collection_step_result(
    step_result: SessionStepResult,
    *,
    active_command: TeleopCommand | None,
    gripper_command: TeleopCommand | None,
) -> SessionStepResult:
    command = active_command
    if gripper_command is not None:
        command = (
            gripper_command
            if command is None
            else merge_teleop_commands((command, gripper_command))
        )
    return SessionStepResult(
        timestamp=float(step_result.timestamp),
        state=step_result.state,
        control_hz=float(step_result.control_hz),
        channels=dict(step_result.channels),
        observation=step_result.observation,
        command=command,
        command_dispatched=bool(step_result.command_dispatched or gripper_command is not None),
        action_record=None,
        faults=list(step_result.faults),
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
        self.writer = LeRobotV21Writer(
            LeRobotV21WriterConfig(
                root_dir=self.dataset_root,
                nominal_fps=float(args.hz),
                task_id=str(args.collection_task_id),
                task_description=str(args.collection_task_description),
                dataset_name=str(args.dataset_name),
                write_parquet=bool(args.write_parquet),
                write_video=bool(args.write_video and not args.no_camera),
                tactile_storage=str(args.tactile_storage),
                allow_missing_video_frames=int(args.storage_queue_size) > 0,
            )
        )
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
        self.collector_components = CollectorComponents(
            session=components.session,
            sources=_collection_sources_from_components(components, self.camera_sources),
            retargeters={},
            backends=_collection_backends_from_components(components),
            rig=components.session.rig,
            metadata={
                "source_entrypoint": "teleop_runtime.debug.run_vive_rm65_real_teleop",
                "active_sides": list(components.active_sides),
                "movement_multiple": _movement_multiple_value(args),
                "record_actual_wrist_camera_pose": bool(self.arm_state_poller is not None),
                "actual_wrist_camera_pose_hz": float(
                    getattr(args, "actual_wrist_camera_pose_hz", 30.0)
                ),
                "actual_wrist_camera_pose_max_age_sec": float(
                    getattr(args, "actual_wrist_camera_pose_max_age_sec", 0.2)
                ),
            },
        )
        self.sample_builder = CollectorSampleBuilder(
            CollectorSampleBuilderConfig(
                active_sides=components.active_sides,
                camera_buffers={
                    camera_key: source.frame_buffer()
                    for camera_key, source in self.camera_sources.items()
                },
                episode_index=self.writer.next_episode_index,
                tactile_array_snapshot_policy=(
                    "always"
                    if str(args.tactile_storage).strip().lower().replace("-", "_") == "parquet"
                    else "on_change"
                ),
            ),
            arm_state_metadata_provider=self._latest_actual_arm_state_metadata,
        )
        self.writer_observer = CollectionWriterObserver(
            components=self.collector_components,
            sample_builder=self.sample_builder,
            write_planner=lambda sample, _components: self.writer.plan_sample_jobs(sample),
            writer_pool=self.writer_pool,
            config=CollectionWriterObserverConfig(
                name="real_teleop_dataset_observer",
                storage_queue_size=int(args.storage_queue_size),
            ),
        )
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
            and str(self.writer.config.task_id) == str(self.args.collection_task_id)
            and str(self.writer.config.task_description) == str(self.args.collection_task_description)
            and str(self.writer.config.dataset_name) == str(self.args.dataset_name)
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
    def frame_count(self) -> int:
        return int(self.sample_builder.frame_index)

    @property
    def finalizing(self) -> bool:
        return any(not task.done for task in self.finalize_tasks)

    def start(self) -> None:
        self.writer_observer.start()
        if self.arm_state_poller is not None:
            self.arm_state_poller.start()

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
        episode_index = self.writer.start_episode(metadata=metadata)
        self.sample_builder.set_episode(episode_index)
        self.active_episode_index = int(episode_index)
        self.active_started_at = float(now)
        self._record_timestamps.clear()
        self._episode_start_monotonic = time.monotonic()
        return int(episode_index)

    def _note_recorded_sample(self) -> None:
        current_time = time.monotonic()
        self._record_timestamps.append(current_time)
        cutoff = current_time - self.frequency_window_sec
        while self._record_timestamps and self._record_timestamps[0] < cutoff:
            self._record_timestamps.popleft()

    def collection_frequency_status(self) -> dict[str, Any]:
        frame_count = int(self.sample_builder.frame_index)
        start_time = self._episode_start_monotonic
        current_time = time.monotonic()
        elapsed_sec = (
            max(0.0, current_time - float(start_time))
            if start_time is not None
            else 0.0
        )
        average_hz = (float(frame_count) / elapsed_sec) if elapsed_sec > 0.0 else None
        rolling_hz = None
        if len(self._record_timestamps) >= 2:
            span = float(self._record_timestamps[-1] - self._record_timestamps[0])
            if span > 0.0:
                rolling_hz = float(len(self._record_timestamps) - 1) / span
        sample_hz = rolling_hz if rolling_hz is not None else average_hz
        enough_data = (
            self.is_recording
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
            "frame_count": frame_count,
            "tactile_frame_counts": tactile_frame_counts,
            "tactile_write_hz": tactile_write_hz,
            "tactile_write_hz_mean": tactile_write_hz_mean,
            "window_sec": float(self.frequency_window_sec),
            "warning": warning,
        }

    def record_step(
        self,
        step_result: SessionStepResult,
        *,
        include_debug_metadata: bool = False,
        phase_recorder: Any | None = None,
        metadata_extra: Mapping[str, Any] | None = None,
    ) -> Any | None:
        if not self.is_recording:
            return None
        previous_include_debug_metadata = bool(self.sample_builder.include_debug_metadata)
        self.sample_builder.include_debug_metadata = bool(include_debug_metadata)
        try:
            result = self.writer_observer.handle_step(
                step_result,
                phase_recorder=phase_recorder,
                metadata_extra=metadata_extra,
            )
        finally:
            self.sample_builder.include_debug_metadata = previous_include_debug_metadata
        self._note_recorded_sample()
        return result

    def _writer_stats_payload(self) -> dict[str, Any]:
        try:
            return asdict(self.writer_pool.stats())
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _storage_queue_stats_payload(self) -> dict[str, Any]:
        try:
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
        started_at: float,
        frequency_status: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
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
            "background_finalize": True,
            "finalize_started_at": float(started_at),
        }

    def _run_finalize_task(
        self,
        task: RealDatasetFinalizeTask,
        *,
        summary: Mapping[str, Any],
    ) -> None:
        summary_payload = dict(summary)
        try:
            self.writer_observer.flush()
            summary_payload["writer_pool_stats_after_flush"] = self._writer_stats_payload()
            summary_payload["storage_queue_stats"] = self._storage_queue_stats_payload()
            self.writer.finalize_episode(task.episode_index, summary=summary_payload)
        except BaseException as exc:  # noqa: BLE001
            task.mark_done(success=False, error=f"{type(exc).__name__}: {exc}")
            return
        task.mark_done(success=True)

    def stop_episode(
        self,
        *,
        reason: str,
        step_index: int,
        last_state: str | None,
    ) -> RealDatasetFinalizeTask | None:
        if self.active_episode_index is None:
            return None
        episode_index = int(self.active_episode_index)
        frame_count = int(self.sample_builder.frame_index)
        frequency_status = self.collection_frequency_status()
        self.active_episode_index = None
        self.active_started_at = None
        self._episode_start_monotonic = None
        if frame_count <= 0:
            self.writer_observer.flush()
            self.writer.discard_episode(episode_index)
            self.discarded_episodes += 1
            self._record_timestamps.clear()
            return None
        task = RealDatasetFinalizeTask(
            episode_index=episode_index,
            frame_count=frame_count,
            reason=str(reason),
            started_at=time.time(),
        )
        summary = self._finalize_summary(
            reason=reason,
            step_index=step_index,
            last_state=last_state,
            frame_count=frame_count,
            started_at=task.started_at,
            frequency_status=frequency_status,
        )
        thread = threading.Thread(
            target=self._run_finalize_task,
            name=f"real_dataset_finalize_{episode_index}",
            args=(task,),
            kwargs={"summary": summary},
            daemon=True,
        )
        task.thread = thread
        self.finalize_tasks.append(task)
        thread.start()
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
            for task in list(self.finalize_tasks):
                if task.thread is not None:
                    task.thread.join(timeout=0.1)
            self.poll_finalize_tasks()

    def close(self) -> None:
        if self.arm_state_poller is not None:
            self.arm_state_poller.close()
        self.wait_finalize_tasks()
        self.writer_observer.close(flush=True)


def _dataset_storage_debug_text(dataset_recorder: RealDatasetRecorder) -> str:
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
    dataset_recorder: RealDatasetRecorder,
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


def _build_components(args: argparse.Namespace) -> RealTeleopComponents:
    _resolve_movement_multiple_args(args)
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
    action_dt = 1.0 / float(args.hz)
    arm_control_hz = float(getattr(args, "arm_control_hz", DEFAULT_RM65_ARM_CONTROL_HZ))
    arm_control_dt = 1.0 / arm_control_hz
    arm_io_threads = max(1, int(args.arm_io_threads))
    collision_guard_requested = (
        set(active_sides) == {"left", "right"}
        and not args.disable_dual_arm_collision_avoidance
    )
    temporary_safety_enabled = bool(
        getattr(args, "dual_arm_temporary_safety", True)
        and set(active_sides) == {"left", "right"}
    )
    temporary_safety_limiter = (
        TemporaryDualArmSafetyLimiter(
            TemporaryDualArmSafetyConfig(
                enabled=True,
                active_sides=tuple(active_sides),
                max_xyz_step_m=float(args.dual_arm_temp_max_xyz_step_m),
                max_rot_step_deg=float(args.dual_arm_temp_max_rot_step_deg),
                max_joint_step_deg=float(args.dual_arm_temp_max_joint_step_deg),
                command_scale=float(args.dual_arm_temp_command_scale),
            )
        )
        if temporary_safety_enabled
        else None
    )
    arm_max_angvel = np.asarray(
        getattr(args, "arm_max_angvel", (0.35, 0.35, 0.35, 0.5, 0.5, 0.5)),
        dtype=float,
    ).reshape(6)
    movement_multiple = _movement_multiple_value(args)
    realistic_joint_step = tuple((arm_max_angvel * action_dt).tolist())
    collision_scene_profile = build_dual_arm_scene_profile(args.collision_scene_profile)
    ik_tuning = load_ik_tuning_config(args.ik_tuning_json or None)
    target_pose_smoothing_config = load_default_pose_smoothing_config()
    target_pose_smoother = build_bimanual_pose_smoother(target_pose_smoothing_config)
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
    if _hand_backend_mode(args) == HAND_BACKEND_DMTAC_GRIPPER:
        _dmtac_backend_configs(args, active_sides)
    tracker_translation_offset = _vive_tracker_local_translation_offset(args)
    calibration.metadata.update(
        {
            "source_entrypoint": "teleop_runtime.debug.run_vive_rm65_real_teleop",
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
                if _hand_backend_mode(args) == HAND_BACKEND_DMTAC_GRIPPER
                else {}
            ),
            "gripper_policy_by_side": (
                _resolved_gripper_policies_by_side(args, active_sides)
                if _hand_backend_mode(args) == HAND_BACKEND_DMTAC_GRIPPER
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
            "dual_arm_collision_avoidance_requested": collision_guard_requested,
            "dual_arm_temporary_safety_enabled": bool(temporary_safety_limiter is not None),
            "dual_arm_temporary_safety": {
                "max_xyz_step_m": float(getattr(args, "dual_arm_temp_max_xyz_step_m", 0.015)),
                "max_rot_step_deg": float(getattr(args, "dual_arm_temp_max_rot_step_deg", 5.0)),
                "max_joint_step_deg": float(getattr(args, "dual_arm_temp_max_joint_step_deg", 4.0)),
                "command_scale": float(getattr(args, "dual_arm_temp_command_scale", 1.0)),
            },
            "target_pose_smoothing": target_pose_smoothing_config.to_metadata(),
            "collision_avoidance_mode": args.collision_avoidance_mode,
            "collision_scene_profile": args.collision_scene_profile,
            "ik_tuning_source": ik_tuning.source_path,
            "ik_objective_mode": ik_tuning.ik_objective_mode,
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
    default_clutch_pressed = args.control_mode == "delta_clutch" and not args.keyboard_clutch
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

    left_retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="left",
            input_channel_name="left_wrist_pose",
            effector_name="left_arm",
            control_mode=args.control_mode,
            translation_scale=movement_multiple,
            input_local_translation_offset=tracker_translation_offset,
            input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
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
        )
    )

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
            debug_ik_profile = bool(
                getattr(args, "debug_timing", False)
                or getattr(args, "debug_ik_timing", False)
            )
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
                            float(getattr(args, "dual_arm_ik_hz", 15.0))
                            if str(getattr(args, "arm_mode", "real")) == "real"
                            else float(args.hz)
                        ),
                        enforce_realistic_joint_step=bool(
                            str(getattr(args, "arm_mode", "real")) == "real"
                        ),
                        left_execution_max_joint_step_rad=realistic_joint_step,
                        right_execution_max_joint_step_rad=realistic_joint_step,
                        left_tool_tcp_pose_flange_m_rad=_tool_tcp_pose_flange_arg(args, "left"),
                        right_tool_tcp_pose_flange_m_rad=_tool_tcp_pose_flange_arg(args, "right"),
                        left_initial_qpos=left_seed_qpos,
                        right_initial_qpos=right_seed_qpos,
                        ik_tuning=ik_tuning,
                        debug_profile=debug_ik_profile,
                        debug_profile_print_every=(
                            int(getattr(args, "debug_ik_timing_print_every", 0))
                            if bool(getattr(args, "debug_ik_timing", False))
                            else 0
                        ),
                        solve_time_budget_ms=getattr(args, "ik_time_budget_ms", None),
                        output_command_type=(
                            "cartesian_pose"
                            if arm_control_mode == "impedance"
                            else "joint_vector"
                        ),
                        target_pose_smoother=target_pose_smoother,
                    ),
                    left_pose_retargeter=left_retargeter,
                    right_pose_retargeter=right_retargeter,
                )
        except (ModuleNotFoundError, RuntimeError, ValueError) as exc:
            collision_guard_setup_error = f"{type(exc).__name__}: {exc}"
        else:
            retargeters.append(dual_arm_retargeter)
    if dual_arm_retargeter is None:
        side_retargeters = {"left": left_retargeter, "right": right_retargeter}
        retargeters.extend(side_retargeters[side] for side in active_sides)

    calibration.metadata.update(
        {
            "dual_arm_collision_avoidance_enabled": dual_arm_retargeter is not None,
            "collision_avoidance_setup_error": collision_guard_setup_error,
        }
    )

    hand_backend_mode = _hand_backend_mode(args)
    gripper_policy = (
        _resolved_gripper_policy(args)
        if hand_backend_mode == HAND_BACKEND_DMTAC_GRIPPER
        else {}
    )
    dmtac_configs = (
        _dmtac_backend_configs(args, active_sides)
        if hand_backend_mode == HAND_BACKEND_DMTAC_GRIPPER
        else {}
    )
    left_hand_port = getattr(args, "left_hand_port", "") or ""
    right_hand_port = getattr(args, "right_hand_port", "") or ""
    hand_source = None
    if hand_backend_mode != HAND_BACKEND_NONE and (left_hand_port or right_hand_port):
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

    left_hand_retargeter = None
    right_hand_retargeter = None
    left_gripper_retargeter = None
    right_gripper_retargeter = None
    if hand_backend_mode == HAND_BACKEND_DMTAC_GRIPPER:
        left_dmtac_config = dmtac_configs.get("left")
        right_dmtac_config = dmtac_configs.get("right")
        if (
            "left" in active_sides
            and left_hand_port
            and left_dmtac_config is not None
            and bool(left_dmtac_config.connect_hardware)
        ):
            left_gripper_retargeter = ScalarToGripperRetargeter(
                ScalarToGripperRetargeterConfig(
                    side="left",
                    input_channel_name="left_hand_q6",
                    effector_name="left_hand",
                    source_dim=int(gripper_policy["source_dim"]),
                    output_dim=int(gripper_policy["output_dim"]),
                    invert=bool(gripper_policy["invert"]),
                    alpha=float(gripper_policy["alpha"]),
                    input_min=float(gripper_policy["input_min"]),
                    input_max=float(gripper_policy["input_max"]),
                    output_max=float(gripper_policy["max_action"]),
                )
            )
        if (
            "right" in active_sides
            and right_hand_port
            and right_dmtac_config is not None
            and bool(right_dmtac_config.connect_hardware)
        ):
            right_gripper_retargeter = ScalarToGripperRetargeter(
                ScalarToGripperRetargeterConfig(
                    side="right",
                    input_channel_name="right_hand_q6",
                    effector_name="right_hand",
                    source_dim=int(gripper_policy["source_dim"]),
                    output_dim=int(gripper_policy["output_dim"]),
                    invert=bool(gripper_policy["invert"]),
                    alpha=float(gripper_policy["alpha"]),
                    input_min=float(gripper_policy["input_min"]),
                    input_max=float(gripper_policy["input_max"]),
                    output_max=float(gripper_policy["max_action"]),
                )
            )
    elif hand_backend_mode == HAND_BACKEND_INSPIRE and "left" in active_sides and left_hand_port:
        left_hand_retargeter = JointsToDexHandRetargeter(
            JointsToDexHandRetargeterConfig(
                side="left",
                input_channel_name="left_hand_q6",
                effector_name="left_hand",
                joint_order=tuple(getattr(args, "hand_joint_order", DEFAULT_INSPIRE_HAND_JOINT_ORDER)),
                invert=bool(getattr(args, "hand_invert", True)),
                alpha=float(getattr(args, "hand_command_alpha", 1.0)),
            )
        )
    if hand_backend_mode == HAND_BACKEND_INSPIRE and "right" in active_sides and right_hand_port:
        right_hand_retargeter = JointsToDexHandRetargeter(
            JointsToDexHandRetargeterConfig(
                side="right",
                input_channel_name="right_hand_q6",
                effector_name="right_hand",
                joint_order=tuple(getattr(args, "hand_joint_order", DEFAULT_INSPIRE_HAND_JOINT_ORDER)),
                invert=bool(getattr(args, "hand_invert", True)),
                alpha=float(getattr(args, "hand_command_alpha", 1.0)),
            )
        )
    if left_hand_retargeter is not None:
        retargeters.append(left_hand_retargeter)
    if right_hand_retargeter is not None:
        retargeters.append(right_hand_retargeter)

    hand_tactile_regions = _parse_hand_tactile_regions(getattr(args, "hand_tactile_regions", "all"))
    hand_telemetry_hz = float(getattr(args, "hand_telemetry_hz", max(float(args.hz), 10.0)))

    left_hand_backend = None
    if hand_backend_mode == HAND_BACKEND_DMTAC_GRIPPER:
        left_dmtac_config = dmtac_configs.get("left")
        if "left" in active_sides and left_dmtac_config is not None:
            left_hand_backend = DMTacGripperBackend(left_dmtac_config)
    elif hand_backend_mode == HAND_BACKEND_INSPIRE and "left" in active_sides and (left_hand_port or args.left_hand_ip):
        left_hand_backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip=args.left_hand_ip or "0.0.0.0",
                side="left",
                connect_hardware=bool(args.left_hand_ip),
                max_fingertip_force=int(
                    getattr(args, "max_fingertip_force", DEFAULT_INSPIRE_HAND_MAX_FORCE)
                ),
                send_threshold=float(getattr(args, "hand_send_threshold", 0.002)),
                enable_telemetry=bool(args.left_hand_ip),
                telemetry_hz=hand_telemetry_hz,
                tactile_regions=hand_tactile_regions,
            )
        )
    right_hand_backend = None
    if hand_backend_mode == HAND_BACKEND_DMTAC_GRIPPER:
        right_dmtac_config = dmtac_configs.get("right")
        if "right" in active_sides and right_dmtac_config is not None:
            right_hand_backend = DMTacGripperBackend(right_dmtac_config)
    elif hand_backend_mode == HAND_BACKEND_INSPIRE and "right" in active_sides and (right_hand_port or args.right_hand_ip):
        right_hand_backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip=args.right_hand_ip or "0.0.0.0",
                side="right",
                connect_hardware=bool(args.right_hand_ip),
                max_fingertip_force=int(
                    getattr(args, "max_fingertip_force", DEFAULT_INSPIRE_HAND_MAX_FORCE)
                ),
                send_threshold=float(getattr(args, "hand_send_threshold", 0.002)),
                enable_telemetry=bool(args.right_hand_ip),
                telemetry_hz=hand_telemetry_hz,
                tactile_regions=hand_tactile_regions,
            )
        )

    session_sources = [
        VivePoseSource(
            tracker_configs=tracker_configs,
            host=args.host,
            port=args.port,
        ),
        keyboard_source,
    ]
    if hand_source is not None:
        session_sources.append(hand_source)

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
                    action_hz=arm_control_hz,
                    max_angvel=tuple(arm_max_angvel.tolist()),
                    actual_send_dt_alpha=float(args.arm_step_dt_alpha),
                    actual_send_dt_min_scale=1.0,
                    actual_send_dt_max_scale=float(args.arm_step_dt_max_scale),
                    actual_send_dt_stall_reset_sec=float(args.arm_step_dt_stall_reset_sec),
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
                    control_dt=arm_control_dt,
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
                    background_streaming_enabled=True,
                    background_streaming_hz=arm_control_hz,
                    tool_profile=_tool_profile_arg(args, logical_side),
                    ik_tuning=ik_tuning,
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
    )
    session_record_root = _session_record_root_dir(args)
    session_recorder = (
        SessionRecorder(
            root_dir=session_record_root,
            config=_session_recording_config(args),
        )
        if not args.no_record
        else None
    )
    session = TeleopSession(
        config=TeleopSessionConfig(
            mode="vive_rm65_real_teleop",
            tcp_mode=args.tcp_mode,
            record_enabled=not args.no_record,
            auto_recovery_enabled=True,
            recovery_healthy_frame_count=max(1, int(args.recovery_healthy_frames)),
            hold_motion_on_freeze=False,
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
            target_control_hz=float(args.hz),
        ),
        command_filter=(
            temporary_safety_limiter.filter_command
            if temporary_safety_limiter is not None
            else None
        ),
    )
    _log(
        "Safety",
        "safety_freeze_hold=off auto_recovery=on for RM65 real teleop; "
        "FROZEN stops new command dispatch until healthy frames enter RECOVERING. "
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
    _sync_dual_arm_retargeter_from_preview(components)
    return synced


def _reset_runtime_motion_state_from_real_qpos(
    components: RealTeleopComponents,
    *,
    required: bool,
) -> dict[str, bool]:
    synced = {side: False for side in components.active_sides}
    for side in components.active_sides:
        state_backend = components.left_state_backend if side == "left" else components.right_state_backend
        preview_backend = components.left_preview_backend if side == "left" else components.right_preview_backend
        command_backend = _command_backend_for_side(components, side)
        if state_backend is None:
            if required:
                raise RuntimeError(f"{side}_arm has no real state backend for runtime restart.")
            continue
        state = state_backend.refresh_state_from_hardware()
        qpos = state.get("current_qpos")
        if qpos is None:
            if required:
                raise RuntimeError(
                    f"{side}_arm runtime restart failed: no current_qpos available from hardware."
                )
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
        preview_backend.set_local_qpos(qpos_array)
        if command_backend is not None:
            command_backend.reset_motion_state(qpos_array)
        synced[side] = True
    _sync_dual_arm_retargeter_from_preview(components)
    return synced


def _sync_real_execution_step_limits(
    components: RealTeleopComponents,
    *,
    now: float | None = None,
) -> None:
    if components.dual_arm_retargeter is None:
        return
    if not components.dual_arm_retargeter.config.enforce_realistic_joint_step:
        return
    if components.left_command_backend is None or components.right_command_backend is None:
        return
    tracker = components.arm_loop_step_dt_tracker
    current_time = time.time() if now is None else float(now)
    command_config = components.left_command_backend.config
    base_dt = float(command_config.control_dt)
    smoothed_loop_dt = float(tracker.get("smoothed_loop_dt", base_dt) or base_dt)
    last_loop_time = tracker.get("last_loop_time")
    if last_loop_time is None:
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
    if components.dual_arm_retargeter.config.joint_velocity_profile is not None:
        components.dual_arm_retargeter.set_execution_dt_sec(effective_loop_dt)
        return
    components.dual_arm_retargeter.set_execution_max_joint_step(
        left_step_rad=np.asarray(components.left_command_backend.config.max_angvel, dtype=float)
        * effective_loop_dt,
        right_step_rad=np.asarray(components.right_command_backend.config.max_angvel, dtype=float)
        * effective_loop_dt,
    )


def _apply_active_arm_command_to_preview_real(
    components: RealTeleopComponents,
    active_command,
) -> None:
    if active_command is None:
        return
    left_command = active_command.commands.get("left_arm")
    if left_command is not None:
        components.left_preview_backend.send_command(left_command)
    right_command = active_command.commands.get("right_arm")
    if right_command is not None:
        components.right_preview_backend.send_command(right_command)


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
    if _arm_control_mode_arg(args) != "impedance":
        return
    backends: list[tuple[str, RM65ImpedanceArmBackend]] = []
    for side in components.active_sides:
        backend = _command_backend_for_side(components, side)
        if not isinstance(backend, RM65ImpedanceArmBackend):
            raise RuntimeError(f"{side}_arm command backend is not RM65ImpedanceArmBackend.")
        backends.append((side, backend))
    _log(
        "ForceControl",
        f"runtime home {'captured' if home_captured else 'skipped'}; "
        f"hold active tools still for tare_sec={_force_tare_sec_arg(args):.2f}s "
        f"before enabling pose impedance.",
    )

    def enable_one(item: tuple[str, RM65ImpedanceArmBackend]) -> tuple[str, dict[str, object]]:
        side, backend = item
        backend.enable_force_control()
        timeout_sec = max(1.0, 3.0 * float(backend.config.control_dt))
        return side, backend.wait_until_force_control_ready(timeout_sec=timeout_sec)

    if len(backends) > 1 and int(getattr(args, "arm_io_threads", 1)) > 1:
        with ThreadPoolExecutor(max_workers=len(backends), thread_name_prefix="rm65_tare") as executor:
            results = [future.result() for future in (executor.submit(enable_one, item) for item in backends)]
    else:
        results = [enable_one(item) for item in backends]
    for side, state in results:
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
    return f"send={code}:{_short_status_text(text)}" if text is not None else f"send={code}"


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
    if _hand_backend_mode(args) == HAND_BACKEND_DMTAC_GRIPPER:
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


def _json_safe_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False).reshape(-1).round(4).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value


def _safe_float_value(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _safe_int_value(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compact_runtime_float(value: object, *, digits: int = 3) -> str:
    numeric = _safe_float_value(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{int(digits)}f}"


def _compact_runtime_bool(value: object) -> str:
    return "1" if bool(value) else "0"


def _q6_channel_status(
    channel: object | None,
    *,
    side: str,
    now: float,
) -> dict[str, object]:
    channel_name = f"{side}_hand_q6"
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
            source_dim = int(getattr(config, "source_dim", source_dim))
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
        if 0 <= source_dim < q6_array.size:
            raw_source = _safe_float_value(q6_array[source_dim])
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
    channel = None if channels is None else channels.get(f"{side}_hand_q6")
    q6_status = _q6_channel_status(channel, side=side, now=now)
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
        "telemetry_elapsed_s": state.get("telemetry_elapsed_s"),
        "tactile_fields": _json_safe_value(state.get("tactile_fields") or ()),
        "tactile_devices": _json_safe_value(state.get("tactile_devices") or ()),
        "tactile_shapes": _json_safe_value(state.get("tactile_shapes") or {}),
        "tactile_timings_ms": _json_safe_value(state.get("tactile_timings_ms") or {}),
        "tactile_read_hz": state.get("tactile_read_hz"),
        "tactile_read_skipped": state.get("tactile_read_skipped"),
        "tactile_process_read": state.get("tactile_process_read"),
        "tactile_worker_alive": state.get("tactile_worker_alive"),
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
        try:
            command = retargeter.retarget(observation, components.session.calibration)
        except Exception as exc:  # noqa: BLE001
            finish("retarget_exception", error=f"{type(exc).__name__}: {exc}")
            _log("Gripper", f"{side} Q6 retarget failed: {exc}", now=now)
            continue
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
        if not sent_ok:
            fields = backend_send_fields(backend)
            backend_status = str(fields.get("backend_status") or "send_returned_false")
            finish(backend_status, **fields)
            continue
        finish("sent", sent=True, **backend_send_fields(backend))
        commands.append(command)
    if not commands:
        return None
    return merge_teleop_commands(commands)


def _hand_channel_status(
    components: RealTeleopComponents,
    channels: dict[str, object],
) -> tuple[str, ...]:
    statuses: list[str] = []
    for side in components.active_sides:
        channel = channels.get(f"{side}_hand_q6")
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
        channel = None if channels is None else channels.get(f"{side}_hand_q6")
        q6_status = _q6_channel_status(channel, side=side, now=current_time)
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
                q6_mode = "q6" if retargeter is not None else "manual"
                suffix = f":{error}" if error else ""
                prefix = f"{side}={connected}/{q6_mode}/cmd={commanded_text}/meas={measured_text}{suffix}"
        statuses.append(
            f"{prefix} q6={q6_valid} dim={q6_source_dim} src={q6_source} "
            f"target={q6_target} filt={q6_filtered} vals={q6_values_text} "
            f"send={cmd_text}/{sent_text} send_src={cmd_source} send_target={cmd_target} "
            f"send_ms={cmd_send_ms}"
            + (f" skip={cmd_skip}" if cmd_skip else "")
            + (f" q6_error={q6_error}" if q6_error else "")
        )
    return tuple(statuses)


_DEBUG_TIMING_RED = "\033[1;31;40m"
_DEBUG_TIMING_RESET = "\033[0m"


def _debug_timing_fault_text(result) -> str:
    if result is None or not result.faults:
        return "faults=none"
    codes = [
        f"{fault.code}:{fault.channel_name}" if fault.channel_name else str(fault.code)
        for fault in result.faults
    ]
    return "faults=" + ",".join(codes)


def _runtime_fault_label(fault: object) -> str:
    code = str(getattr(fault, "code", "unknown"))
    channel_name = getattr(fault, "channel_name", None)
    label = f"{code}:{channel_name}" if channel_name else code
    metadata = getattr(fault, "metadata", None)
    if not isinstance(metadata, Mapping):
        return label
    details: list[str] = []
    if "translation_jump_m" in metadata:
        details.append(f"jump_m={_compact_runtime_float(metadata.get('translation_jump_m'), digits=4)}")
    if "rotation_jump_deg" in metadata:
        details.append(f"rot_deg={_compact_runtime_float(metadata.get('rotation_jump_deg'), digits=2)}")
    return f"{label}({','.join(details)})" if details else label


def _debug_timing_arm_text(
    *,
    left_hw_state: dict[str, object],
    right_hw_state: dict[str, object],
    left_command_state: dict[str, object],
    right_command_state: dict[str, object],
    target_step_scale: float,
) -> str:
    def code_text(state: dict[str, object]) -> str:
        code = state.get("hardware_result_code")
        text = state.get("hardware_result_text")
        return f"{code}:{text}" if text is not None else str(code)

    return (
        "arms="
        f"left_hw={code_text(left_hw_state)} "
        f"right_hw={code_text(right_hw_state)} "
        f"left_cmd={_command_backend_status_text(left_command_state)} "
        f"right_cmd={_command_backend_status_text(right_command_state)} "
        f"left_step_scale={float(left_command_state.get('effective_step_scale', 1.0)):.2f} "
        f"right_step_scale={float(right_command_state.get('effective_step_scale', 1.0)):.2f} "
        f"target_step_scale={float(target_step_scale):.2f}"
    )


def _debug_timing_ik_text(active_command) -> str:
    if active_command is None:
        return "ik=none"
    metadata = getattr(active_command, "metadata", {}) or {}
    profile = metadata.get("ik_debug_profile")
    status = metadata.get("collision_avoidance_status")
    jump_gate = metadata.get("ik_jump_gate_status")
    branch = metadata.get("ik_branch_candidate")
    fallback = metadata.get("collision_avoidance_fallback_reason")
    prefix = f"ik=status={status} jump={jump_gate} branch={branch}"
    if fallback:
        prefix += f" fallback={fallback}"
    if not isinstance(profile, dict):
        return prefix
    min_phase = profile.get("min_distance_phase_ms")
    min_phase_text = ""
    if isinstance(min_phase, dict) and min_phase:
        ordered = sorted(
            ((str(name), float(value)) for name, value in min_phase.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        min_phase_text = " min_phase=" + ",".join(
            f"{name}:{value:.1f}ms" for name, value in ordered[:3]
        )
    return (
        f"{prefix} "
        f"total={float(profile.get('frame_total_ms', 0.0)):.1f}ms "
        f"solve={float(profile.get('solve_total_ms', 0.0)):.1f}ms "
        f"provider={float(profile.get('provider_total_ms', 0.0)):.1f}ms "
        f"other={float(profile.get('non_provider_ms', 0.0)):.1f}ms "
        f"fk={float(profile.get('fk_ms', 0.0)):.1f}ms "
        f"sing={float(profile.get('singularity_ms', 0.0)):.1f}ms "
        f"cand_eval={float(profile.get('candidate_eval_ms', 0.0)):.1f}ms "
        f"qp={float(profile.get('solve_qp_ms', 0.0)):.1f}ms "
        f"min_dist={float(profile.get('min_distance_total_ms', 0.0)):.1f}ms "
        f"branch={float(profile.get('branch_ms', 0.0)):.1f}ms "
        f"jump_gate={float(profile.get('jump_gate_ms', 0.0)):.1f}ms "
        f"hold_recovery={float(profile.get('hold_recovery_ms', 0.0)):.1f}ms "
        f"provider_calls={int(profile.get('provider_call_count', 0))} "
        f"candidate_evals={int(profile.get('candidate_eval_count', 0))} "
        f"solve_calls={int(profile.get('solve_calls', 0))} "
        f"min_calls={int(profile.get('min_distance_calls', 0))}"
        f"{min_phase_text}"
    )


def _float_or_none(*values: object) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(result):
            return result
    return None


def _int_or_zero(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _bool_or_false(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _mapping_or_empty(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


_IK_RED_WARNING_STATUS_CODES = {"failed", "hold_last_safe"}
_IK_RED_WARNING_SINGULARITY_GATE_CODES = {"hold_entry"}
_IK_RED_WARNING_LABELS = {
    "failed": "IK failed",
    "hold_last_safe": "hold last safe",
    "hold_entry": "singularity hold entry",
}


def _stripped_code(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _ik_warning_entries_from_metadata(
    metadata: Mapping[str, object] | None,
    *,
    source: str,
) -> list[dict[str, str]]:
    if not isinstance(metadata, Mapping):
        return []
    metadata_dict = dict(metadata)
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(code: str, field: str) -> None:
        normalized = code.strip()
        if not normalized:
            return
        key = (str(source), normalized)
        if key in seen:
            return
        seen.add(key)
        entries.append(
            {
                "source": str(source),
                "code": normalized,
                "field": str(field),
                "label": _IK_RED_WARNING_LABELS.get(normalized, normalized),
            }
        )

    for key in ("status", "ik_status"):
        code = _stripped_code(metadata_dict.get(key))
        if code in _IK_RED_WARNING_STATUS_CODES:
            add(code, key)
    singularity_gate_status = _stripped_code(metadata_dict.get("singularity_gate_status"))
    if singularity_gate_status in _IK_RED_WARNING_SINGULARITY_GATE_CODES:
        add(singularity_gate_status, "singularity_gate_status")
    return entries


def _ik_warning_payload_from_entries(entries: Sequence[Mapping[str, str]]) -> dict[str, object]:
    compact_entries: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        compact_entry = {
            "source": str(entry.get("source") or "-"),
            "code": str(entry.get("code") or "-"),
            "field": str(entry.get("field") or "-"),
            "label": str(entry.get("label") or entry.get("code") or "-"),
        }
        key = (compact_entry["source"], compact_entry["code"], compact_entry["field"])
        if key in seen:
            continue
        seen.add(key)
        compact_entries.append(compact_entry)
    if not compact_entries:
        return {
            "active": False,
            "level": "none",
            "entries": [],
            "codes": [],
            "sources": [],
            "message": "",
        }
    codes = list(dict.fromkeys(entry["code"] for entry in compact_entries))
    sources = list(dict.fromkeys(entry["source"] for entry in compact_entries))
    message = "; ".join(
        f"{entry['source']}:{entry['label']}"
        for entry in compact_entries
    )
    return {
        "active": True,
        "level": "red",
        "entries": compact_entries,
        "codes": codes,
        "sources": sources,
        "message": message,
    }


def _build_ik_warning_payload(
    *,
    active_command,
    left_command_state: Mapping[str, object] | None = None,
    right_command_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    entries: list[dict[str, str]] = []
    if active_command is not None:
        entries.extend(
            _ik_warning_entries_from_metadata(
                getattr(active_command, "metadata", {}) or {},
                source="retargeter_command",
            )
        )
        commands = getattr(active_command, "commands", {}) or {}
        if isinstance(commands, Mapping):
            for side in ("left", "right"):
                side_command = commands.get(f"{side}_arm")
                if side_command is None:
                    continue
                entries.extend(
                    _ik_warning_entries_from_metadata(
                        getattr(side_command, "metadata", {}) or {},
                        source=f"{side}_retargeter_command",
                    )
                )
    for side, state in (
        ("left", left_command_state),
        ("right", right_command_state),
    ):
        if not isinstance(state, Mapping):
            continue
        entries.extend(
            _ik_warning_entries_from_metadata(
                _mapping_or_empty(state.get("ik_metadata")),
                source=f"{side}_backend",
            )
        )
        entries.extend(
            _ik_warning_entries_from_metadata(
                _mapping_or_empty(state.get("singularity_metadata")),
                source=f"{side}_backend",
            )
        )
    return _ik_warning_payload_from_entries(entries)


def _ik_timing_summary_from_metadata(
    metadata: Mapping[str, object] | None,
    *,
    source: str,
) -> dict[str, object] | None:
    if metadata is None:
        return None
    metadata_dict = dict(metadata)
    profile = _mapping_or_empty(metadata_dict.get("ik_debug_profile"))
    frame_total_ms = _float_or_none(
        profile.get("frame_total_ms"),
        metadata_dict.get("ik_frame_total_ms"),
        metadata_dict.get("ik_backend_solve_ms"),
    )
    solve_total_ms = _float_or_none(
        profile.get("solve_total_ms"),
        metadata_dict.get("ik_solve_total_ms"),
        metadata_dict.get("ik_backend_solve_ms"),
    )
    if frame_total_ms is None and solve_total_ms is None:
        return None
    solve_qp_ms = _float_or_none(profile.get("solve_qp_ms"), metadata_dict.get("ik_solve_qp_ms"))
    min_distance_total_ms = _float_or_none(
        profile.get("min_distance_total_ms"),
        metadata_dict.get("ik_min_distance_total_ms"),
    )
    provider_elapsed_ms = _float_or_none(
        profile.get("provider_total_ms"),
        metadata_dict.get("ik_provider_total_ms"),
        metadata_dict.get("ik_provider_elapsed_ms"),
    )
    non_provider_ms = _float_or_none(
        profile.get("non_provider_ms"),
        metadata_dict.get("ik_non_provider_ms"),
    )
    fk_ms = _float_or_none(profile.get("fk_ms"), metadata_dict.get("ik_fk_ms"))
    singularity_ms = _float_or_none(
        profile.get("singularity_ms"),
        metadata_dict.get("ik_singularity_ms"),
    )
    collision_ms = _float_or_none(profile.get("collision_ms"), metadata_dict.get("ik_collision_ms"))
    candidate_eval_ms = _float_or_none(
        profile.get("candidate_eval_ms"),
        metadata_dict.get("ik_candidate_eval_ms"),
    )
    time_budget_ms = _float_or_none(
        profile.get("time_budget_ms"),
        metadata_dict.get("ik_time_budget_ms"),
    )
    time_budget_exceeded = _bool_or_false(
        profile.get("time_budget_exceeded", metadata_dict.get("ik_time_budget_exceeded", False))
    )
    status = str(
        metadata_dict.get("collision_avoidance_status")
        or metadata_dict.get("status")
        or metadata_dict.get("ik_status")
        or "-"
    )
    return {
        "available": True,
        "source": source,
        "status": status,
        "jump_gate_status": str(metadata_dict.get("ik_jump_gate_status") or "-"),
        "branch_candidate": str(
            metadata_dict.get("ik_branch_candidate")
            or metadata_dict.get("ik_branch_selection")
            or metadata_dict.get("candidate_source")
            or "-"
        ),
        "frame_total_ms": 0.0 if frame_total_ms is None else frame_total_ms,
        "solve_total_ms": 0.0 if solve_total_ms is None else solve_total_ms,
        "solve_qp_ms": 0.0 if solve_qp_ms is None else solve_qp_ms,
        "min_distance_total_ms": 0.0 if min_distance_total_ms is None else min_distance_total_ms,
        "provider_elapsed_ms": 0.0 if provider_elapsed_ms is None else provider_elapsed_ms,
        "non_provider_ms": non_provider_ms,
        "fk_ms": fk_ms,
        "singularity_ms": singularity_ms,
        "collision_ms": collision_ms,
        "candidate_eval_ms": candidate_eval_ms,
        "provider_call_count": _int_or_zero(profile.get("provider_call_count")),
        "candidate_eval_count": _int_or_zero(profile.get("candidate_eval_count")),
        "branch_ms": _float_or_none(profile.get("branch_ms")) or 0.0,
        "jump_gate_ms": _float_or_none(profile.get("jump_gate_ms")) or 0.0,
        "hold_recovery_ms": _float_or_none(profile.get("hold_recovery_ms")) or 0.0,
        "solve_calls": _int_or_zero(profile.get("solve_calls")),
        "min_distance_calls": _int_or_zero(profile.get("min_distance_calls")),
        "time_budget_ms": time_budget_ms,
        "time_budget_exceeded": time_budget_exceeded,
    }


def _ik_timing_summaries(
    *,
    active_command,
    left_command_state: Mapping[str, object] | None = None,
    right_command_state: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    if active_command is not None:
        summary = _ik_timing_summary_from_metadata(
            getattr(active_command, "metadata", {}) or {},
            source="retargeter_command",
        )
        if summary is not None:
            summaries.append(summary)
    for side, state in (
        ("left", left_command_state),
        ("right", right_command_state),
    ):
        if not isinstance(state, Mapping):
            continue
        summary = _ik_timing_summary_from_metadata(
            _mapping_or_empty(state.get("ik_metadata")),
            source=f"{side}_backend",
        )
        if summary is not None:
            summaries.append(summary)
    return summaries


def _build_ik_timing_payload(
    *,
    active_command,
    left_command_state: Mapping[str, object] | None = None,
    right_command_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    ik_warning = _build_ik_warning_payload(
        active_command=active_command,
        left_command_state=left_command_state,
        right_command_state=right_command_state,
    )
    summaries = _ik_timing_summaries(
        active_command=active_command,
        left_command_state=left_command_state,
        right_command_state=right_command_state,
    )
    if not summaries:
        return {
            "available": False,
            "message": "暂无 IK timing 元数据；retargeter 或后端尚未提供本帧耗时。",
            "ik_warning": ik_warning,
        }
    primary = next(
        (summary for summary in summaries if summary.get("source") == "retargeter_command"),
        max(summaries, key=lambda item: float(item.get("frame_total_ms", 0.0))),
    )
    payload = dict(primary)
    payload["per_source"] = [
        {
            "source": str(summary.get("source") or "-"),
            "frame_total_ms": float(summary.get("frame_total_ms", 0.0)),
            "time_budget_exceeded": bool(summary.get("time_budget_exceeded", False)),
        }
        for summary in summaries
    ]
    payload["message"] = "per_source=" + ",".join(
        f"{item['source']}:{float(item['frame_total_ms']):.1f}ms"
        for item in payload["per_source"]
    )
    payload["ik_warning"] = ik_warning
    return payload


def _ik_timing_status_text(payload: Mapping[str, object] | None) -> str:
    if not isinstance(payload, Mapping) or not bool(payload.get("available", False)):
        return "ik_time=unavailable"
    frame_total_ms = float(payload.get("frame_total_ms", 0.0))
    budget = _float_or_none(payload.get("time_budget_ms"))
    budget_text = "budget=off" if budget is None or budget <= 0.0 else f"budget={budget:.1f}ms"
    exceeded = " exceeded" if bool(payload.get("time_budget_exceeded", False)) else ""
    return (
        f"ik_time={frame_total_ms:.1f}ms {budget_text}{exceeded} "
        f"source={payload.get('source')}"
    )


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
    if _hand_backend_mode(args) == HAND_BACKEND_DMTAC_GRIPPER:
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


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _pose_matrix_debug(matrix: object) -> dict[str, object] | None:
    try:
        pose = as_pose_matrix(matrix)
    except Exception:
        return None
    try:
        rpy_deg = SciPyRotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
    except Exception:
        rpy_deg = np.full(3, np.nan)
    return {
        "matrix": pose.copy(),
        "xyz_m": pose[:3, 3].copy(),
        "rpy_deg": rpy_deg,
    }


def _qpos_debug(qpos: object) -> dict[str, object] | None:
    if qpos is None:
        return None
    try:
        values = np.asarray(qpos, dtype=float).reshape(6)
    except (TypeError, ValueError):
        return None
    return {
        "rad": values.copy(),
        "deg": np.rad2deg(values).copy(),
    }


def _compact_numeric_list(values: object, *, max_items: int = 12) -> list[float] | dict[str, object] | None:
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    if arr.size > int(max_items):
        return {
            "omitted": True,
            "reason": "compact_debug_log",
            "shape": list(np.asarray(values).shape),
            "size": int(arr.size),
        }
    return [float(value) for value in arr.tolist()]


def _compact_debug_value(value: object, *, max_items: int = 12, depth: int = 0) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _compact_numeric_list(value, max_items=max_items)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        if depth >= 3:
            return {
                "omitted": True,
                "reason": "compact_debug_log_depth",
                "type": "mapping",
                "size": len(value),
            }
        out: dict[str, object] = {}
        items = list(value.items())
        for raw_key, item in items[:max_items]:
            key = str(raw_key)
            if key == "ik_resolver_init":
                continue
            out[key] = _compact_debug_value(item, max_items=max_items, depth=depth + 1)
        if len(items) > max_items:
            out["_omitted_items"] = int(len(items) - max_items)
        return out
    if isinstance(value, (list, tuple)):
        if depth >= 3:
            return {
                "omitted": True,
                "reason": "compact_debug_log_depth",
                "type": "sequence",
                "size": len(value),
            }
        out = [
            _compact_debug_value(item, max_items=max_items, depth=depth + 1)
            for item in list(value)[:max_items]
        ]
        if len(value) > max_items:
            out.append({"_omitted_items": int(len(value) - max_items)})
        return out
    return str(value)


def _pose_compact_debug(matrix: object) -> dict[str, object] | None:
    try:
        pose = as_pose_matrix(matrix)
    except Exception:
        return None
    try:
        rpy_deg = SciPyRotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
    except Exception:
        rpy_deg = np.full(3, np.nan)
    return {
        "xyz_m": [float(value) for value in pose[:3, 3].tolist()],
        "rpy_deg": [float(value) for value in rpy_deg.tolist()],
    }


def _qpos_compact_debug(qpos: object) -> dict[str, object] | None:
    if qpos is None:
        return None
    try:
        values = np.asarray(qpos, dtype=float).reshape(6)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(values)):
        return None
    return {"deg": [float(value) for value in np.rad2deg(values).tolist()]}


def _pose_delta_compact_debug(
    current_pose: object,
    target_pose: object,
) -> dict[str, float] | None:
    try:
        current = as_pose_matrix(current_pose)
        target = as_pose_matrix(target_pose)
    except Exception:
        return None
    try:
        delta_rot = SciPyRotation.from_matrix(target[:3, :3]) * SciPyRotation.from_matrix(
            current[:3, :3]
        ).inv()
        angular_deg = float(np.rad2deg(delta_rot.magnitude()))
    except Exception:
        angular_deg = float("nan")
    return {
        "translation_m": float(np.linalg.norm(target[:3, 3] - current[:3, 3])),
        "rotation_deg": angular_deg,
    }


_IK_PROFILE_COMPACT_KEYS = (
    "target_already_reached",
    "frame_total_ms",
    "time_budget_ms",
    "time_budget_exceeded",
    "solve_total_ms",
    "solve_qp_ms",
    "solve_integrate_ms",
    "solve_calls",
    "provider_total_ms",
    "provider_call_count",
    "non_provider_ms",
    "fk_ms",
    "singularity_ms",
    "collision_ms",
    "candidate_eval_ms",
    "candidate_eval_count",
    "local_candidate_eval_count",
    "min_distance_total_ms",
    "min_distance_calls",
    "branch_ms",
    "jump_gate_ms",
    "hold_recovery_ms",
)


def _ik_profile_compact_debug(profile: object) -> dict[str, object]:
    if not isinstance(profile, Mapping):
        return {}
    out = {
        key: _compact_debug_value(profile.get(key))
        for key in _IK_PROFILE_COMPACT_KEYS
        if key in profile
    }
    min_phase = profile.get("min_distance_phase_ms")
    if isinstance(min_phase, Mapping):
        out["min_distance_phase_ms"] = {
            str(key): _compact_debug_value(value)
            for key, value in min_phase.items()
        }
    return out


def _execution_step_compact_debug(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        return {}
    keys = (
        "enabled",
        "step_limited",
        "weighted_delta_deg",
        "left_max_abs_delta_deg",
        "right_max_abs_delta_deg",
        "joint_velocity_limit_profile",
        "joint_velocity_limit_dt_sec",
        "joint_velocity_limit_max_dt_sec",
        "joint_velocity_limit_ratio",
        "left_max_step_deg",
        "right_max_step_deg",
    )
    return {
        key: _compact_debug_value(metadata.get(key))
        for key in keys
        if key in metadata
    }


def _command_metadata_compact_debug(metadata: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        return {}
    selected_keys = (
        "collision_avoidance_status",
        "collision_avoidance_fallback_reason",
        "collision_distance_m",
        "ik_branch_candidate",
        "ik_branch_distance_score_deg",
        "ik_solve_iterations",
        "ik_solve_pose_error",
        "ik_jump_gate_status",
        "ik_jump_gate_reason",
        "ik_jump_gate_release_ready",
        "ik_jump_gate_hold_frames",
        "ik_jump_gate_metrics",
        "joint_limit_margin",
        "ik_candidate_selection",
        "target_activity",
        "scene_profile",
        "collision_pair_count",
        "ik_frame_total_ms",
        "ik_time_budget_ms",
        "ik_time_budget_exceeded",
    )
    out = {
        key: _compact_debug_value(metadata.get(key))
        for key in selected_keys
        if key in metadata
    }
    if "execution_step_metadata" in metadata:
        out["execution_step_metadata"] = _execution_step_compact_debug(
            metadata.get("execution_step_metadata")
        )
    for key in (
        "target_pose_base",
        "target_pose_tcp_base",
        "target_pose_desk",
        "source_target_pose_base",
        "source_target_pose_desk",
        "source_target_pose_operator",
        "solver_output_endpoint_pose_base",
    ):
        if key in metadata:
            out[key] = _pose_compact_debug(metadata.get(key))
    if "solver_output_qpos" in metadata:
        out["solver_output_qpos"] = _qpos_compact_debug(metadata.get("solver_output_qpos"))
    if "ik_debug_profile" in metadata:
        out["ik_debug_profile"] = _ik_profile_compact_debug(metadata.get("ik_debug_profile"))
    return out


def _teleop_command_metadata_compact_debug(command: TeleopCommand | None) -> dict[str, object]:
    if command is None:
        return {}
    metadata = dict(getattr(command, "metadata", {}) or {})
    selected_keys = (
        "collision_avoidance_status",
        "collision_avoidance_fallback_reason",
        "collision_distance_m",
        "ik_branch_candidate",
        "ik_jump_gate_status",
        "ik_candidate_selection",
        "target_activity",
        "scene_profile",
        "collision_pair_count",
        "ik_frame_total_ms",
        "ik_time_budget_ms",
        "ik_time_budget_exceeded",
    )
    out = {
        key: _compact_debug_value(metadata.get(key))
        for key in selected_keys
        if key in metadata
    }
    if "ik_debug_profile" in metadata:
        out["ik_debug_profile"] = _ik_profile_compact_debug(metadata.get("ik_debug_profile"))
    if "temporary_dual_arm_safety" in metadata:
        out["temporary_dual_arm_safety"] = _compact_debug_value(
            metadata.get("temporary_dual_arm_safety"),
            max_items=24,
        )
    return out


def _command_compact_debug_payload(command) -> dict[str, object] | None:
    if command is None:
        return None
    value_pose = (
        _pose_compact_debug(getattr(command, "value", None))
        if getattr(command, "command_type", "") == "cartesian_pose"
        else None
    )
    value_qpos = (
        _qpos_compact_debug(getattr(command, "value", None))
        if getattr(command, "command_type", "") == "joint_vector"
        else None
    )
    metadata = dict(getattr(command, "metadata", {}) or {})
    return {
        "command_type": getattr(command, "command_type", ""),
        "frame_name": getattr(command, "frame_name", None),
        "value_pose": value_pose,
        "value_qpos": value_qpos,
        "solver_output_qpos": _qpos_compact_debug(metadata.get("solver_output_qpos")),
        "metadata": _command_metadata_compact_debug(metadata),
    }


def _impedance_record_compact_debug(record: object) -> dict[str, object] | None:
    if not isinstance(record, Mapping):
        return None
    keys = (
        "step",
        "dt_sec",
        "reference",
        "force_norm_n",
        "torque_norm_nm",
        "command_joint_step_deg",
        "allowed_command_joint_step_deg",
        "joint_rate_limited",
        "joint_velocity_limit_profile",
        "joint_velocity_limit_dt_sec",
        "joint_velocity_limit_ratio",
        "joint_safety",
        "ik_method",
        "ik_backoff_scale",
        "ik_reference_backoff_scale",
        "ik_command_step_backoff_scale",
        "ik_command_step_candidate_deg",
        "ik_command_step_allowed_deg",
        "ik_command_step_applied_deg",
        "ik_resolver_tolerance_fraction",
        "ik_timing",
        "singularity",
        "joint_limit_margin",
        "pose_tracking_linear_m",
        "pose_tracking_angular_rad",
    )
    out = {
        key: _compact_debug_value(record.get(key))
        for key in keys
        if key in record
    }
    if "ik_metadata" in record:
        out["ik_metadata"] = _backend_ik_metadata_compact_debug(record.get("ik_metadata"))
    return out


def _backend_ik_metadata_compact_debug(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, Mapping):
        return {}
    selected_keys = (
        "ik_status",
        "ik_jump_gate_status",
        "ik_jump_gate_reason",
        "ik_solve_total_ms",
        "ik_backend_solve_ms",
        "ik_provider_total_ms",
        "ik_provider_elapsed_ms",
        "ik_min_distance_total_ms",
        "ik_solve_qp_ms",
        "ik_non_provider_ms",
        "singularity_gate_status",
        "target_singularity_label",
        "current_singularity_label",
        "target_wrist_center_distance_m",
        "current_wrist_center_distance_m",
    )
    out = {
        key: _compact_debug_value(metadata.get(key))
        for key in selected_keys
        if key in metadata
    }
    if "ik_debug_profile" in metadata:
        out["ik_debug_profile"] = _ik_profile_compact_debug(metadata.get("ik_debug_profile"))
    return out


def _backend_compact_debug_payload(state: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(state, Mapping):
        return {}
    out: dict[str, object] = {
        "hardware_result_code": state.get("hardware_result_code"),
        "hardware_result_text": state.get("hardware_result_text"),
        "send_result_code": state.get("send_result_code"),
        "send_result_text": state.get("send_result_text"),
        "effective_control_dt_sec": state.get("effective_control_dt_sec"),
        "effective_step_scale": state.get("effective_step_scale"),
        "impedance_enabled": state.get("impedance_enabled"),
        "impedance_loop_state": state.get("impedance_loop_state"),
        "current_qpos": _qpos_compact_debug(state.get("current_qpos")),
        "last_target_qpos": _qpos_compact_debug(state.get("last_target_qpos")),
        "current_tcp_pose": _pose_compact_debug(state.get("current_tcp_pose")),
        "last_target_tcp_pose": _pose_compact_debug(state.get("last_target_tcp_pose")),
        "pose_tracking_error": _pose_delta_compact_debug(
            state.get("current_tcp_pose"),
            state.get("last_target_tcp_pose"),
        ),
        "ik_metadata": _backend_ik_metadata_compact_debug(state.get("ik_metadata")),
        "last_impedance_record": _impedance_record_compact_debug(
            state.get("last_impedance_record")
        ),
    }
    return {key: value for key, value in out.items() if value is not None}


def _observation_pose_compact_debug(step_result) -> dict[str, object]:
    channels = getattr(step_result, "channels", {}) or {}
    out: dict[str, object] = {}
    for channel_name in ("left_wrist_pose", "right_wrist_pose"):
        channel = channels.get(channel_name)
        if channel is None:
            continue
        out[channel_name] = {
            "valid": bool(getattr(channel, "valid", False)),
            "timestamp": float(getattr(channel, "timestamp", 0.0)),
            "frame_name": getattr(channel, "frame_name", None),
            "source_name": getattr(channel, "source_name", None),
            "pose": _pose_compact_debug(getattr(channel, "value", None)),
        }
    return out


def _command_debug_payload(command) -> dict[str, object] | None:
    if command is None:
        return None
    metadata = dict(getattr(command, "metadata", {}) or {})
    selected_metadata_keys = (
        "collision_avoidance_status",
        "collision_avoidance_fallback_reason",
        "collision_distance_m",
        "ik_branch_candidate",
        "ik_branch_distance_score_deg",
        "ik_solve_iterations",
        "ik_solve_pose_error",
        "ik_jump_gate_status",
        "ik_jump_gate_reason",
        "ik_jump_gate_release_ready",
        "ik_jump_gate_hold_frames",
        "ik_jump_gate_metrics",
        "joint_limit_margin",
        "ik_candidate_selection",
        "target_activity",
        "scene_profile",
        "collision_pair_count",
        "ik_frame_total_ms",
        "ik_time_budget_ms",
        "ik_time_budget_exceeded",
        "execution_step_metadata",
        "target_pose_base",
        "target_pose_tcp_base",
        "target_pose_desk",
        "source_target_pose_base",
        "source_target_pose_desk",
        "source_target_pose_operator",
        "solver_output_endpoint_pose_base",
        "solver_output_qpos",
        "ik_debug_profile",
    )
    selected_metadata = {
        key: metadata.get(key)
        for key in selected_metadata_keys
        if key in metadata
    }
    solver_qpos = _qpos_debug(metadata.get("solver_output_qpos"))
    value_pose = (
        _pose_matrix_debug(getattr(command, "value", None))
        if getattr(command, "command_type", "") == "cartesian_pose"
        else None
    )
    value_qpos = (
        _qpos_debug(getattr(command, "value", None))
        if getattr(command, "command_type", "") == "joint_vector"
        else None
    )
    return {
        "command_type": getattr(command, "command_type", ""),
        "frame_name": getattr(command, "frame_name", None),
        "value_pose": value_pose,
        "value_qpos": value_qpos,
        "solver_output_qpos": solver_qpos,
        "metadata": selected_metadata,
    }


def _teleop_command_metadata_debug(command: TeleopCommand | None) -> dict[str, object]:
    if command is None:
        return {}
    metadata = dict(getattr(command, "metadata", {}) or {})
    selected_keys = (
        "collision_avoidance_status",
        "collision_avoidance_fallback_reason",
        "collision_distance_m",
        "ik_branch_candidate",
        "ik_jump_gate_status",
        "ik_candidate_selection",
        "target_activity",
        "scene_profile",
        "collision_pair_count",
        "ik_frame_total_ms",
        "ik_time_budget_ms",
        "ik_time_budget_exceeded",
        "ik_debug_profile",
        "temporary_dual_arm_safety",
    )
    return {key: metadata.get(key) for key in selected_keys if key in metadata}


def _impedance_record_debug(record: object) -> dict[str, object] | None:
    if not isinstance(record, Mapping):
        return None
    keys = (
        "step",
        "dt_sec",
        "reference",
        "command_pose",
        "command_joint_deg",
        "measured_pose",
        "measured_joint_deg",
        "offset",
        "target_offset",
        "force_norm_n",
        "torque_norm_nm",
        "command_joint_step_deg",
        "allowed_command_joint_step_deg",
        "joint_rate_limited",
        "joint_velocity_limit_profile",
        "joint_velocity_limit_dt_sec",
        "joint_velocity_limit_requested_delta_deg",
        "joint_velocity_limit_delta_deg",
        "joint_velocity_limit_max_step_deg",
        "joint_velocity_limit_ratio",
        "joint_safety",
        "ik_method",
        "ik_backoff_scale",
        "ik_reference_backoff_scale",
        "ik_command_step_backoff_scale",
        "ik_command_step_candidate_deg",
        "ik_command_step_allowed_deg",
        "ik_command_step_applied_deg",
        "ik_resolver_tolerance_fraction",
        "ik_timing",
        "singularity",
        "joint_limit_margin",
        "ik_metadata",
        "pose_tracking_linear_m",
        "pose_tracking_angular_rad",
    )
    return {key: record.get(key) for key in keys if key in record}


def _backend_debug_payload(state: Mapping[str, object] | None) -> dict[str, object]:
    if not isinstance(state, Mapping):
        return {}
    return {
        "hardware_result_code": state.get("hardware_result_code"),
        "hardware_result_text": state.get("hardware_result_text"),
        "send_result_code": state.get("send_result_code"),
        "send_result_text": state.get("send_result_text"),
        "effective_control_dt_sec": state.get("effective_control_dt_sec"),
        "effective_step_scale": state.get("effective_step_scale"),
        "impedance_enabled": state.get("impedance_enabled"),
        "impedance_loop_state": state.get("impedance_loop_state"),
        "current_qpos": _qpos_debug(state.get("current_qpos")),
        "last_target_qpos": _qpos_debug(state.get("last_target_qpos")),
        "current_tcp_pose": _pose_matrix_debug(state.get("current_tcp_pose")),
        "last_target_tcp_pose": _pose_matrix_debug(state.get("last_target_tcp_pose")),
        "ik_metadata": state.get("ik_metadata"),
        "last_impedance_record": _impedance_record_debug(state.get("last_impedance_record")),
    }


def _observation_pose_debug(step_result) -> dict[str, object]:
    channels = getattr(step_result, "channels", {}) or {}
    out: dict[str, object] = {}
    for channel_name in ("left_wrist_pose", "right_wrist_pose"):
        channel = channels.get(channel_name)
        if channel is None:
            continue
        out[channel_name] = {
            "valid": bool(getattr(channel, "valid", False)),
            "timestamp": float(getattr(channel, "timestamp", 0.0)),
            "frame_name": getattr(channel, "frame_name", None),
            "source_name": getattr(channel, "source_name", None),
            "pose": _pose_matrix_debug(getattr(channel, "value", None)),
        }
    return out


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


class IkDebugLogRecorder:
    def __init__(self, *, root_dir: Path | str = "logs", max_queue_size: int = 256) -> None:
        self.root_dir = Path(root_dir)
        self.path: Path | None = None
        self._queue: queue.Queue[Mapping[str, object] | None] | None = None
        self._writer_thread: threading.Thread | None = None
        self._max_queue_size = max(16, int(max_queue_size))
        self.frame_count = 0
        self.dropped_frame_count = 0

    @property
    def active(self) -> bool:
        return self._queue is not None

    def start(self, *, now: float | None = None) -> Path:
        if self.active and self.path is not None:
            return self.path
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time() if now is None else now))
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.root_dir / f"ik_debug_{timestamp}.jsonl"
        log_file = self.path.open("a", encoding="utf-8")
        log_queue: queue.Queue[Mapping[str, object] | None] = queue.Queue(maxsize=self._max_queue_size)
        self._queue = log_queue
        self.frame_count = 0
        self.dropped_frame_count = 0
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            args=(log_queue, log_file, self.path),
            daemon=True,
            name="ik_debug_log_writer",
        )
        self._writer_thread.start()
        self.write_event("start", {"path": str(self.path)}, now=now)
        return self.path

    def _writer_loop(
        self,
        log_queue: queue.Queue[Mapping[str, object] | None],
        log_file,
        path: Path,
    ) -> None:
        pending_frames = 0
        try:
            while True:
                row = log_queue.get()
                if row is None:
                    break
                log_file.write(json.dumps(_json_safe(row), ensure_ascii=False, separators=(",", ":")) + "\n")
                if row.get("event") == "frame":
                    pending_frames += 1
                    if pending_frames % 20 == 0:
                        log_file.flush()
                else:
                    log_file.flush()
            log_file.flush()
        except Exception as exc:  # noqa: BLE001 - logging must not break teleop.
            print(f"WARNING: IK debug log writer failed for {path}: {type(exc).__name__}: {exc}")
        finally:
            try:
                log_file.close()
            except Exception:
                pass

    def _enqueue_row(self, row: Mapping[str, object], *, kind: str) -> bool:
        log_queue = self._queue
        if log_queue is None:
            return False
        try:
            log_queue.put_nowait(dict(row))
            return True
        except queue.Full:
            if kind == "frame":
                self.dropped_frame_count += 1
            return False

    def write_event(self, event: str, payload: Mapping[str, object] | None = None, *, now: float | None = None) -> None:
        if self._queue is None:
            return
        row = {
            "schema": "dexteleop_ik_debug_v1",
            "event": str(event),
            "timestamp": float(time.time() if now is None else now),
            "payload": dict(payload or {}),
        }
        try:
            self._queue.put(row, timeout=0.02)
        except queue.Full:
            self.dropped_frame_count += 1

    def record_frame(self, payload: Mapping[str, object]) -> None:
        if self._queue is None:
            return
        if self._enqueue_row(payload, kind="frame"):
            self.frame_count += 1

    def stop(self, *, reason: str, now: float | None = None) -> Path | None:
        log_queue = self._queue
        if log_queue is None:
            return self.path
        self.write_event(
            "stop",
            {
                "reason": str(reason),
                "frame_count": int(self.frame_count),
                "dropped_frame_count": int(self.dropped_frame_count),
            },
            now=now,
        )
        try:
            log_queue.put_nowait(None)
        except queue.Full:
            threading.Thread(
                target=lambda: log_queue.put(None),
                daemon=True,
                name="ik_debug_log_stop",
            ).start()
        self._queue = None
        return self.path

    def close(self) -> None:
        self.stop(reason="close")


class RealTeleopControlInput:
    def __init__(
        self,
        *,
        terminal: TerminalClutchInput,
        ui: CalibrationWebUi | None,
        args: argparse.Namespace,
        components: RealTeleopComponents,
    ) -> None:
        self.terminal = terminal
        self.ui = ui
        self.args = args
        self.components = components
        self.enabled = bool(terminal.enabled or ui is not None)
        self.should_quit = False
        self._calibration_requested = False
        self._calibration_request_from_hold = False
        self._last_calibration_started_from_hold = False
        self._hold_requested = False
        self._hold_active = False
        self._tracker_swap_locked = False
        self._teleop_started = False
        self._last_raw_pose_publish_at = 0.0
        self._raw_pose_publish_interval_sec = 0.2
        self._last_capture_progress_publish_at = 0.0
        self._capture_progress_publish_interval_sec = 0.1
        self._last_ik_timing_publish_at = 0.0
        self._ik_timing_publish_interval_sec = 0.2
        self._last_ik_warning_publish_at = 0.0
        self._ik_warning_publish_interval_sec = 0.2
        self._last_ik_warning_signature: tuple[object, ...] | None = None
        self._ik_debug_log_toggle_requested = False
        self._ik_debug_logging_active = False
        self._ik_debug_log_path = ""
        self._gripper_control_enabled = False
        self._hold_request_reason = "manual"
        self._q6_hold_recalibration_gesture = Q6HoldRecalibrationGesture(
            Q6HoldRecalibrationGestureConfig(
                enabled=bool(getattr(args, "q6_hold_recalibration_gesture", False)),
                hold_sec=float(getattr(args, "q6_gesture_hold_sec", 1.0)),
                close_threshold=float(getattr(args, "q6_gesture_close_threshold", 0.95)),
                open_threshold=float(getattr(args, "q6_gesture_open_threshold", 0.85)),
            )
        )
        self._last_gripper_status_publish_at = 0.0
        self._gripper_status_publish_interval_sec = 0.2
        self._episode_start_requested = False
        self._episode_stop_requested = False
        self._camera_visualization_toggle_requested = False
        self._tactile_visualization_toggle_requested = False
        self._camera_visualization_active = bool(getattr(args, "vis_cam", False))
        self._tactile_visualization_active = bool(getattr(args, "visualize_tactile", False))

    def _episode_recording_available(self) -> bool:
        return (
            not bool(getattr(self.args, "no_collection", False))
            or not bool(getattr(self.args, "no_episode_recording", False))
        )

    def _camera_visualization_available(self) -> bool:
        return bool(getattr(self.args, "camera_visualization_available", False))

    def _tactile_visualization_available(self) -> bool:
        return bool(getattr(self.args, "tactile_visualization_available", False))

    def _visualization_controls(self) -> list[str]:
        controls: list[str] = []
        if self._camera_visualization_available():
            controls.append("toggle_camera_vis")
        if self._tactile_visualization_available():
            controls.append("toggle_tactile_vis")
        return controls

    def _vive_calibration_controls(self, *, allow_load: bool) -> list[str]:
        if not allow_load:
            return [VIVE_CALIBRATION_GUI_EVENTS[1]]
        return [
            VIVE_CALIBRATION_GUI_EVENTS[0],
            VIVE_CALIBRATION_GUI_EVENTS[1],
            VIVE_CALIBRATION_GUI_EVENTS[2],
        ]

    def _movement_multiple_controls(self) -> list[str]:
        return list(MOVEMENT_MULTIPLE_GUI_EVENTS)

    def _teleop_controls(self) -> list[str]:
        controls = ["q", "d", "h"]
        if self._episode_recording_available():
            controls.extend(EPISODE_GUI_EVENTS)
        controls.extend(self._visualization_controls())
        controls.extend(self._vive_calibration_controls(allow_load=False))
        return controls

    def _single_arm_tracker_swap_allowed_in_hold(self) -> bool:
        return bool(
            self._hold_active and len(tuple(self.components.active_sides)) == 1
        )

    def _pre_home_controls(self) -> list[str]:
        controls = ["p", "w", "q"]
        controls.extend(self._visualization_controls())
        controls.extend(self._vive_calibration_controls(allow_load=True))
        controls.extend(self._movement_multiple_controls())
        return controls

    def _hold_controls(self) -> list[str]:
        controls = ["p", "c"]
        if len(tuple(self.components.active_sides)) == 1:
            controls.append("w")
        controls.append("q")
        controls.extend(self._visualization_controls())
        controls.extend(self._vive_calibration_controls(allow_load=True))
        controls.extend(self._movement_multiple_controls())
        return controls

    def start(self) -> None:
        self.terminal.start()
        if self.ui is None:
            self.enabled = bool(self.terminal.enabled)
            return
        self.ui.publish(
            stage="RM65 真机遥操准备",
            selected_side="",
            host=str(self.args.host),
            port=int(self.args.port),
            load_profile=str(getattr(self.args, "load_profile", "") or ""),
            active_controls=self._pre_home_controls(),
            control_hint=self._control_hint(),
            capture_stage="runtime_home",
            capture_phase="ready",
            teleop_running=False,
            debug_ik_timing=bool(getattr(self.args, "debug_ik_timing", False)),
            debug_ik_logging=False,
            ik_debug_log_path="",
            ik_timing={
                "available": False,
                "message": "等待遥操启动后接收 IK timing 元数据。",
            },
            ik_warning=_build_ik_warning_payload(active_command=None),
            gripper_control_available=_gripper_gui_available(self.components),
            gripper_q6_available=_has_q6_gripper_control(self.components),
            gripper_control_enabled=False,
            gripper_status=_build_gripper_status(self.components),
            episode_recording_available=self._episode_recording_available(),
            episode_recording=False,
            episode_dir="",
            episode_frame_count=None,
            episode_sample_hz=None,
            episode_target_hz=float(getattr(self.args, "collection_warning_hz", 15.0)),
            episode_frequency_warning={"active": False, "message": ""},
            **_collection_task_ui_state(self.args),
            camera_visualization_available=self._camera_visualization_available(),
            camera_visualization_active=self._camera_visualization_active,
            tactile_visualization_available=self._tactile_visualization_available(),
            tactile_visualization_active=self._tactile_visualization_active,
            visualization_status="",
            **_movement_multiple_ui_state(self.args),
            **_vive_calibration_profile_ui_state(self.args, self.components),
            capture_prep_remaining_sec=None,
            capture_prep_total_sec=3.0,
            capture_waiting_for_samples=False,
        )
        self.ui.start()
        ui_url = str(getattr(self.ui, "url", "") or "")
        _log(
            "WebGUI",
            f"ready url={ui_url or '<unknown>'} state_url={(ui_url.rstrip('/') + '/state.json') if ui_url else '<unknown>'} "
            f"debug_url={(ui_url.rstrip('/') + '/debug.json') if ui_url else '<unknown>'} "
            "capture_stage=runtime_home capture_phase=ready",
        )
        self.ui.append_message("WebGUI", f"后端状态已发布；如果页面不刷新，请打开 {ui_url or '当前 Web GUI URL'}。")
        self.publish_raw_tracker_pose_state(force=True)
        self.enabled = True

    def close(self) -> None:
        self.terminal.close()
        if self.ui is not None:
            self.ui.close()

    def poll(self) -> None:
        current_time = time.time()
        self.terminal.poll()
        if self.terminal.should_quit:
            self.should_quit = True
        if self.terminal.consume_calibration_request():
            self._calibration_requested = True
            self._calibration_request_from_hold = bool(self._hold_active)
        if self.ui is not None:
            for event in self.ui.consume_events():
                if event == "q":
                    self.should_quit = True
                    if self._teleop_started:
                        self.ui.publish(
                            stage="退出 / 停臂中",
                            active_controls=[],
                            control_hint="已收到网页退出；后端正在停止机械臂并关闭遥操。",
                            capture_stage="",
                            capture_phase="idle",
                            teleop_running=False,
                            capturing=False,
                            capture_waiting_for_samples=False,
                        )
                        self.ui.append_message("EmergencyStop", "网页退出已触发机械臂停臂请求。")
                elif event in {"p", "c"}:
                    self._calibration_requested = True
                    self._calibration_request_from_hold = bool(self._hold_active)
                elif event == "h":
                    self.request_hold()
                elif event == "w":
                    self._toggle_vive_tracker_swap()
                elif event == "d":
                    self._ik_debug_log_toggle_requested = True
                elif event == "episode_start" or event.startswith("episode_start:"):
                    if event.startswith("episode_start:"):
                        self._apply_episode_start_payload(event.split(":", 1)[1])
                    self._episode_start_requested = True
                elif event == "episode_stop":
                    self._episode_stop_requested = True
                elif event == "toggle_camera_vis":
                    self._camera_visualization_toggle_requested = True
                elif event == "toggle_tactile_vis":
                    self._tactile_visualization_toggle_requested = True
                elif event == "calibrate_vive_coordinate_profile":
                    self._calibrate_vive_coordinate_profile()
                elif event == "save_vive_calibration_profile":
                    self._save_vive_calibration_profile()
                elif event == "load_vive_calibration_profile":
                    self._load_vive_calibration_profile()
                elif event.startswith("movement_multiple_set:"):
                    self._set_movement_multiple_from_gui(event.split(":", 1)[1])
                elif event in MOVEMENT_MULTIPLE_GUI_EVENTS:
                    self._handle_movement_multiple_event(event)
        self._poll_q6_hold_recalibration_gesture(now=current_time)
        if self.ui is not None:
            self.publish_raw_tracker_pose_state(now=current_time)
            self.publish_gripper_status(now=current_time)

    def consume_calibration_request(self) -> bool:
        requested = bool(self._calibration_requested)
        request_from_hold = bool(self._calibration_request_from_hold or self._hold_active)
        self._calibration_requested = False
        self._calibration_request_from_hold = False
        self._last_calibration_started_from_hold = False
        if requested:
            if self._teleop_started and not self._hold_active:
                if self.ui is not None:
                    self.ui.append_message("Hold", "运行中重采前请先点击 Hold / 冻结。")
                    self.ui.publish(
                        active_controls=self._teleop_controls(),
                        control_hint="运行中重采前请先点击 Hold / 冻结；Hold 只停臂，不会自动重采。",
                        teleop_running=True,
                        gripper_control_enabled=self.gripper_control_enabled(),
                    )
                return False
            if request_from_hold:
                self._hold_active = False
                self._hold_request_reason = "manual"
                self._last_calibration_started_from_hold = True
            self._gripper_control_enabled = False
            self._tracker_swap_locked = True
            if self.ui is not None:
                self.ui.publish(
                    active_controls=["q"],
                    control_hint="已确认开始，VIVE tracker 左右映射已锁定；3 秒后采集 runtime home。",
                    teleop_running=False,
                    home_ready=False,
                    gripper_control_enabled=False,
                )
        return requested

    def _request_runtime_home_calibration_from_gui(self) -> None:
        if self.ui is None:
            return
        if self._teleop_started and not self._hold_active:
            message = "运行中重新标定前请先点击 Hold / 冻结。"
            self.ui.append_message("Calibration", message)
            self.ui.publish(
                active_controls=self._teleop_controls(),
                control_hint=message,
                teleop_running=True,
                gripper_control_enabled=self.gripper_control_enabled(),
            )
            return
        self._calibration_requested = True
        self._calibration_request_from_hold = bool(self._hold_active)
        self._gripper_control_enabled = False
        self.ui.publish(
            stage="准备重新标定 runtime home",
            active_controls=["q"],
            control_hint="已请求重新进入 Home 标定流程；后端将先倒计时，再采集当前 VIVE/机械臂 home。",
            capture_stage="runtime_home",
            capture_phase="ready",
            teleop_running=False,
            home_ready=False,
            gripper_control_enabled=False,
            capture_waiting_for_samples=False,
        )
        self.ui.append_message("Calibration", "已请求重新进入 runtime home 标定流程。")

    def _publish_movement_multiple(self, *, message: str = "") -> None:
        if self.ui is None:
            return
        self.ui.publish(**_movement_multiple_ui_state(self.args))
        if message:
            self.ui.append_message("MovementMultiple", message)

    def _set_movement_multiple_value(self, value: float, *, source: str) -> float | None:
        if self._teleop_started and not self._hold_active:
            message = "运行中不直接修改 movement_multiple；请先 Hold，避免目标位置因倍率变化发生跳变。"
            _log("MovementMultiple", f"{source}: rejected while teleop running")
            self._publish_movement_multiple(message=message)
            return None
        try:
            movement_multiple = _apply_movement_multiple(self.args, self.components, value)
        except (ValueError, argparse.ArgumentTypeError) as exc:
            message = f"移动倍率无效: {exc}"
            _log("MovementMultiple", message)
            self._publish_movement_multiple(message=message)
            return None
        message = (
            f"movement_multiple={movement_multiple:.3f}; "
            "只缩放 tracker 相对 runtime home 的平移，旋转保持原逻辑。"
        )
        _log("MovementMultiple", f"{source}: {message}")
        self._publish_movement_multiple(message=message)
        return movement_multiple

    def _set_movement_multiple_from_gui(self, text: str) -> None:
        try:
            value = float(str(text).strip())
        except ValueError:
            self._set_movement_multiple_value(float("nan"), source="gui_set")
            return
        self._set_movement_multiple_value(value, source="gui_set")

    def _handle_movement_multiple_event(self, event: str) -> None:
        current = _movement_multiple_value(self.args)
        step = MOVEMENT_MULTIPLE_GUI_STEP
        if event == "movement_multiple_down":
            value = max(MOVEMENT_MULTIPLE_GUI_MIN, current - step)
        elif event == "movement_multiple_up":
            value = current + step
        elif event == "movement_multiple_reset":
            value = MOVEMENT_MULTIPLE_DEFAULT
        elif event == "movement_multiple_set":
            value = current
        else:
            return
        self._set_movement_multiple_value(value, source=event)

    def _apply_episode_start_payload(self, text: str) -> None:
        try:
            payload = json.loads(str(text or "{}"))
        except json.JSONDecodeError as exc:
            if self.ui is not None:
                self.ui.append_message("Dataset", f"数采参数无效: {exc}")
            return
        if not isinstance(payload, Mapping):
            return
        dataset_base_root = str(payload.get("dataset_base_root") or "").strip()
        task_key = str(payload.get("task_key") or "").strip()
        english_name = str(payload.get("task_english_name") or "").strip()
        chinese_name = str(payload.get("task_chinese_name") or "").strip()
        task_def = _task_definition_for_key(self.args, task_key)
        if task_def is not None:
            if not english_name:
                english_name = str(task_def.get("english_name") or task_key)
            if not chinese_name:
                chinese_name = str(task_def.get("chinese_name") or "")
            gripper_task = str(task_def.get("gripper_task") or task_key)
        else:
            gripper_task = task_key
        if dataset_base_root:
            self.args.dataset_root = dataset_base_root
            self.args.dataset_base_root = str(_expand_cli_path(dataset_base_root))
        if gripper_task:
            self.args.gripper_task = gripper_task
        task_name = english_name or gripper_task or str(getattr(self.args, "collection_task_name", "") or "")
        if task_name:
            self.args.collection_task_name = task_name
        if chinese_name:
            self.args.collection_task_name_zh = chinese_name
        self.args.collection_task_id = task_key or _default_task_id_from_name(task_name or gripper_task)
        task_description = (
            str(task_def.get("description") or "").strip()
            if task_def is not None
            else ""
        )
        self.args.collection_task_description = task_description or chinese_name or task_name
        _refresh_formal_dataset_layout(self.args)
        self._apply_current_gripper_task_policy()
        if self.ui is not None:
            self.ui.publish(**_collection_task_ui_state(self.args))
            self.ui.append_message(
                "Dataset",
                (
                    f"数采任务已更新: {self.args.collection_task_name} "
                    f"({self.args.collection_task_id}) -> {self.args.dataset_root}"
                ),
            )

    def _apply_current_gripper_task_policy(self) -> None:
        _clear_gripper_policy_cache(self.args)
        policies = _resolved_gripper_policies_by_side(self.args, self.components.active_sides)
        self.components.gripper_policy.clear()
        self.components.gripper_policy.update(_resolved_gripper_policy(self.args))
        for side in self.components.active_sides:
            backend = _hand_backend_for_side(self.components, side)
            policy = policies.get(side)
            if not isinstance(backend, DMTacGripperBackend) or policy is None:
                continue
            backend.config.speed = float(policy["speed"])
            backend.config.torque = float(policy["torque"])
            backend.config.max_action = float(policy["max_action"])
            backend.config.send_threshold = float(policy["send_threshold"])
            backend.config.command_hz = float(policy["command_hz"])
            backend.config.continuous_grasp = bool(policy["continuous_grasp"])
            backend.config.telemetry_hz = float(policy["telemetry_hz"])
        if self.ui is not None:
            self.publish_gripper_status(force=True)

    def request_hold(self, *, reason: str = "manual") -> None:
        if not self._teleop_started:
            return
        self._hold_requested = True
        self._hold_active = True
        self._gripper_control_enabled = False
        self._hold_request_reason = str(reason or "manual")
        self._q6_hold_recalibration_gesture.reset_for_hold()
        is_q6_gesture = self._hold_request_reason == "q6_gesture"
        if self.ui is not None:
            self.ui.publish(
                stage="Hold / 停臂保持",
                active_controls=self._hold_controls(),
                control_hint=(
                    "机械臂保持当前位置；请松开双手 Q6 后三维到释放区间 1 秒，再同时握紧 1 秒以重新标定。"
                    if is_q6_gesture
                    else (
                        "机械臂保持当前位置；单臂模式可先点击“交换 Tracker”，"
                        "然后重新点击“开始遥操”或“采集 Home”。"
                        if len(tuple(self.components.active_sides)) == 1
                        else (
                            "机械臂保持当前位置；需要重新标定时，"
                            "摆好 VIVE/Q6 后点击“开始遥操”或“采集 Home”。"
                        )
                    )
                ),
                capture_stage="runtime_home",
                capture_phase="ready",
                teleop_running=False,
                capture_waiting_for_samples=False,
                gripper_control_enabled=False,
                **_movement_multiple_ui_state(self.args),
            )
            message = (
                "Q6 手势已请求冻结机械臂；松开双手后三维到释放区间 1 秒，再次握紧 1 秒会重采 runtime home。"
                if is_q6_gesture
                else "已请求冻结机械臂；不会自动重采 runtime home。"
            )
            self.ui.append_message("Hold", message)

    def consume_hold_request(self) -> bool:
        requested = bool(self._hold_requested)
        self._hold_requested = False
        return requested

    def hold_active(self) -> bool:
        return bool(self._hold_active)

    def hold_request_reason(self) -> str:
        return str(self._hold_request_reason or "manual")

    def consume_episode_start_request(self) -> bool:
        requested = bool(self._episode_start_requested)
        self._episode_start_requested = False
        return bool(self.terminal.consume_episode_start_request() or requested)

    def consume_episode_stop_request(self) -> bool:
        requested = bool(self._episode_stop_requested)
        self._episode_stop_requested = False
        return bool(self.terminal.consume_episode_stop_request() or requested)

    def consume_camera_visualization_toggle_request(self) -> bool:
        requested = bool(self._camera_visualization_toggle_requested)
        self._camera_visualization_toggle_requested = False
        return requested

    def consume_tactile_visualization_toggle_request(self) -> bool:
        requested = bool(self._tactile_visualization_toggle_requested)
        self._tactile_visualization_toggle_requested = False
        return requested

    def consume_ik_debug_log_toggle_request(self) -> bool:
        requested = bool(self._ik_debug_log_toggle_requested)
        self._ik_debug_log_toggle_requested = False
        return requested

    def ik_debug_timing_enabled(self) -> bool:
        return bool(getattr(self.args, "debug_ik_timing", False)) or bool(self._ik_debug_logging_active)

    def gripper_control_enabled(self) -> bool:
        return bool(self._gripper_control_enabled and _has_q6_gripper_control(self.components))

    def _poll_q6_hold_recalibration_gesture(self, *, now: float) -> None:
        gesture = self._q6_hold_recalibration_gesture
        if not bool(gesture.config.enabled):
            return
        if set(self.components.active_sides) != {"left", "right"}:
            gesture.reset_for_running()
            return
        source = _hand_q6_source_from_components(self.components)
        if source is None:
            gesture.update(
                left_q6=None,
                right_q6=None,
                teleop_started=self._teleop_started,
                hold_active=self._hold_active,
                now=float(now),
            )
            return
        channels = source.get_latest_channels()
        left_q6 = Q6HoldRecalibrationGesture.q6_values_from_channel(channels.get("left_hand_q6"))
        right_q6 = Q6HoldRecalibrationGesture.q6_values_from_channel(channels.get("right_hand_q6"))
        event = gesture.update(
            left_q6=left_q6,
            right_q6=right_q6,
            teleop_started=self._teleop_started,
            hold_active=self._hold_active,
            now=float(now),
        )
        if event == Q6HoldRecalibrationGesture.EVENT_HOLD:
            self.request_hold(reason="q6_gesture")
            _log(
                "Q6Gesture",
                "both hands q6 dims 4/5/6 closed for hold window; requested hold",
                now=now,
            )
        elif event == Q6HoldRecalibrationGesture.EVENT_REARMED:
            if self.ui is not None:
                self.ui.append_message("Q6Gesture", "已检测到双手后三维进入释放区间；再次同时握紧 1 秒将重采 runtime home。")
            _log(
                "Q6Gesture",
                "both hands q6 dims 4/5/6 released; waiting for second close to recalibrate",
                now=now,
            )
        elif event == Q6HoldRecalibrationGesture.EVENT_RESTART:
            self._calibration_requested = True
            self._calibration_request_from_hold = True
            self._gripper_control_enabled = False
            if self.ui is not None:
                self.ui.publish(
                    stage="准备重采 runtime home",
                    active_controls=["q", *self._visualization_controls()],
                    control_hint="Q6 手势已确认；即将重采 runtime home 并恢复遥操。",
                    teleop_running=False,
                    gripper_control_enabled=False,
                )
                self.ui.append_message("Q6Gesture", "已检测到第二次同时握紧；准备重采 runtime home。")
            _log(
                "Q6Gesture",
                "both hands q6 dims 4/5/6 closed again; requested runtime home recalibration",
                now=now,
            )

    def set_ik_debug_logging_state(self, *, active: bool, path: str = "") -> None:
        self._ik_debug_logging_active = bool(active)
        self._ik_debug_log_path = str(path or "")
        if self.ui is not None:
            self.ui.publish(
                debug_ik_timing=self.ik_debug_timing_enabled(),
                debug_ik_logging=self._ik_debug_logging_active,
                ik_debug_log_path=self._ik_debug_log_path,
            )

    def set_episode_recording_state(
        self,
        *,
        active: bool,
        episode_dir: str | Path | None = None,
        frame_count: int | None = None,
        frequency_status: Mapping[str, object] | None = None,
    ) -> None:
        if self.ui is None:
            return
        frequency_payload = dict(frequency_status or {})
        warning_payload = dict(frequency_payload.get("warning") or {})
        if not active:
            warning_payload = {"active": False, "message": ""}
        tactile_write_hz_payload = dict(frequency_payload.get("tactile_write_hz") or {})
        tactile_frame_counts_payload = dict(frequency_payload.get("tactile_frame_counts") or {})
        self.ui.publish(
            episode_recording_available=self._episode_recording_available(),
            episode_recording=bool(active),
            episode_dir="" if episode_dir is None else str(episode_dir),
            episode_frame_count=frame_count,
            episode_sample_hz=frequency_payload.get("sample_hz") if active else None,
            episode_target_hz=frequency_payload.get("target_hz") if active else None,
            episode_tactile_write_hz=tactile_write_hz_payload if active else {},
            episode_tactile_write_hz_mean=(
                frequency_payload.get("tactile_write_hz_mean") if active else None
            ),
            episode_tactile_frame_counts=tactile_frame_counts_payload if active else {},
            episode_frequency_warning=warning_payload,
            **_collection_task_ui_state(self.args),
        )

    def set_visualization_state(
        self,
        *,
        camera_active: bool | None = None,
        tactile_active: bool | None = None,
        status: str | None = None,
    ) -> None:
        if camera_active is not None:
            self._camera_visualization_active = bool(camera_active)
        if tactile_active is not None:
            self._tactile_visualization_active = bool(tactile_active)
        if self.ui is None:
            return
        self.ui.publish(
            camera_visualization_available=self._camera_visualization_available(),
            camera_visualization_active=self._camera_visualization_active,
            tactile_visualization_available=self._tactile_visualization_available(),
            tactile_visualization_active=self._tactile_visualization_active,
            visualization_status="" if status is None else str(status),
            active_controls=(
                self._teleop_controls()
                if self._teleop_started and not self._hold_active
                else [
                    *(["p", "c", "q"] if self._teleop_started else ["p", "w", "q"]),
                    *self._visualization_controls(),
                    *self._vive_calibration_controls(allow_load=True),
                    *self._movement_multiple_controls(),
                ]
            ),
            **_movement_multiple_ui_state(self.args),
        )

    def before_calibration_start_delay(
        self,
        delay_sec: float,
        *,
        arm_mode: str,
    ) -> bool:
        del arm_mode
        total = max(0.0, float(delay_sec))
        deadline = time.time() + total
        while True:
            remaining = max(0.0, deadline - time.time())
            if self.ui is not None:
                self.ui.publish(
                    stage="即将开始 runtime home",
                    active_controls=["q"],
                    control_hint=f"请松开按键并保持姿态，{remaining:.1f}s 后开始。",
                    capture_stage="runtime_home",
                    capture_phase="preparing",
                    teleop_running=False,
                    capture_prep_total_sec=total,
                    capture_prep_remaining_sec=remaining,
                    capture_waiting_for_samples=False,
                )
            self.publish_raw_tracker_pose_state(force=True)
            if remaining <= 0.0:
                break
            self.terminal.poll()
            if self.terminal.should_quit:
                self.should_quit = True
            if self.ui is not None:
                for event in self.ui.consume_events():
                    if event == "q":
                        self.should_quit = True
            if self.should_quit:
                return False
            time.sleep(min(0.05, remaining))
        if self.ui is not None:
            self.ui.publish(
                stage="采集 runtime home",
                active_controls=["q"],
                control_hint="正在采集当前 tracker 和机械臂 home，请保持静止。",
                capture_stage="runtime_home",
                capture_phase="capturing",
                teleop_running=False,
                capture_samples=0,
                capture_left_samples=0,
                capture_right_samples=0,
                capture_min_samples=int(getattr(self.args, "calibration_samples", 0)),
                capture_valid=False,
                capture_left_valid=False,
                capture_right_valid=False,
                capture_elapsed_sec=0.0,
                capture_duration_sec=float(getattr(self.args, "calibration_timeout", 0.0)),
                capture_prep_remaining_sec=0.0,
                capture_waiting_for_samples=False,
            )
        self.publish_raw_tracker_pose_state(force=True)
        return True

    def publish_teleop_running(self, *, home_ready: bool = True, message: str | None = None) -> None:
        was_hold_active = bool(self._hold_active)
        was_hold_restart = bool(self._last_calibration_started_from_hold)
        self._last_calibration_started_from_hold = False
        self._teleop_started = True
        self._hold_active = False
        self._gripper_control_enabled = _has_q6_gripper_control(self.components)
        if was_hold_active or was_hold_restart:
            self._q6_hold_recalibration_gesture.mark_runtime_restarted()
        else:
            self._q6_hold_recalibration_gesture.reset_for_running()
        if self.ui is None:
            return
        control_hint = "遥操已开始；Q6 夹爪控制会自动开启；Hold 只冻结机械臂，重采需再次点击开始。"
        if self._q6_hold_recalibration_gesture.config.enabled:
            control_hint = (
                "遥操已开始；Q6 手势模式已开启：双手后三维握紧 1 秒进入 Hold，"
                "松开到释放区间 1 秒后再次握紧 1 秒会重采 runtime home。"
            )
        self.ui.publish(
            stage="RM65 真机遥操运行中",
            active_controls=self._teleop_controls(),
            control_hint=control_hint,
            home_ready=bool(home_ready),
            capture_stage="teleop",
            capture_phase="teleop",
            teleop_running=True,
            capturing=False,
            capture_samples=None,
            capture_left_samples=None,
            capture_right_samples=None,
            capture_min_samples=None,
            capture_valid=None,
            capture_left_valid=None,
            capture_right_valid=None,
            capture_elapsed_sec=None,
            capture_duration_sec=None,
            capture_prep_remaining_sec=None,
            capture_prep_total_sec=None,
            capture_waiting_for_samples=False,
            debug_ik_timing=self.ik_debug_timing_enabled(),
            debug_ik_logging=self._ik_debug_logging_active,
            ik_debug_log_path=self._ik_debug_log_path,
            ik_timing={
                "available": False,
                "message": "等待第一帧 IK timing 元数据。",
            },
            ik_warning=_build_ik_warning_payload(active_command=None),
            gripper_control_available=_gripper_gui_available(self.components),
            gripper_q6_available=_has_q6_gripper_control(self.components),
            gripper_control_enabled=self.gripper_control_enabled(),
            gripper_status=_build_gripper_status(self.components),
            episode_recording_available=self._episode_recording_available(),
            **_collection_task_ui_state(self.args),
            camera_visualization_available=self._camera_visualization_available(),
            camera_visualization_active=self._camera_visualization_active,
            tactile_visualization_available=self._tactile_visualization_available(),
            tactile_visualization_active=self._tactile_visualization_active,
            **_movement_multiple_ui_state(self.args),
            **_vive_calibration_profile_ui_state(self.args, self.components),
        )
        _log(
            "WebGUI",
            f"teleop state published url={str(getattr(self.ui, 'url', '') or '<unknown>')} "
            f"gripper_control={'on' if self.gripper_control_enabled() else 'off'} "
            f"q6_sides={[side for side in self.components.active_sides if _gripper_retargeter_for_side(self.components, side) is not None]}",
        )
        self.ui.append_message("RealTeleop", message or "runtime home 完成，已进入真机遥操。")
        self.publish_raw_tracker_pose_state(force=True)
        self.publish_gripper_status(force=True)

    def publish_calibration_waiting(self, message: str = "") -> None:
        self._calibration_requested = False
        self._calibration_request_from_hold = False
        self._last_calibration_started_from_hold = False
        if self._teleop_started:
            self._hold_active = True
            self._gripper_control_enabled = False
            self._q6_hold_recalibration_gesture.reset_for_hold()
        self._tracker_swap_locked = bool(self._teleop_started)
        if self.ui is None:
            return
        if message:
            self.ui.append_message("Calibration", message)
        self.ui.publish(
            stage="RM65 真机遥操准备" if not self._teleop_started else "Hold / 停臂保持",
            active_controls=self._hold_controls()
            if self._teleop_started
            else self._pre_home_controls(),
            control_hint=(
                message
                or "请选择：标定/加载/保存 VIVE 坐标系，或采集 Home 后开始遥操。"
            ),
            capture_stage="runtime_home",
            capture_phase="ready",
            teleop_running=False,
            home_ready=False,
            capturing=False,
            capture_samples=None,
            capture_left_samples=None,
            capture_right_samples=None,
            capture_min_samples=None,
            capture_valid=None,
            capture_left_valid=None,
            capture_right_valid=None,
            capture_elapsed_sec=None,
            capture_duration_sec=None,
            capture_waiting_for_samples=False,
            gripper_control_enabled=False,
            **_collection_task_ui_state(self.args),
            **_movement_multiple_ui_state(self.args),
            **_vive_calibration_profile_ui_state(self.args, self.components),
        )

    def publish_runtime_home_progress(
        self,
        *,
        sample_counts: Mapping[str, int],
        valid_by_side: Mapping[str, bool],
        min_samples: int,
        elapsed_sec: float,
        timeout_sec: float,
        waiting_for_samples: bool,
        now: float | None = None,
    ) -> None:
        if self.ui is None:
            return
        current_time = time.time() if now is None else float(now)
        complete = all(int(sample_counts.get(side, 0)) >= int(min_samples) for side in self.components.active_sides)
        if (
            not complete
            and current_time - self._last_capture_progress_publish_at < self._capture_progress_publish_interval_sec
        ):
            return
        self._last_capture_progress_publish_at = current_time
        left_samples = int(sample_counts.get("left", 0)) if "left" in self.components.active_sides else None
        right_samples = int(sample_counts.get("right", 0)) if "right" in self.components.active_sides else None
        single_count = None
        if len(self.components.active_sides) == 1:
            single_count = int(sample_counts.get(self.components.active_sides[0], 0))
        self.ui.publish(
            stage="采集 runtime home",
            capture_stage="runtime_home",
            capture_phase="capturing",
            capturing=True,
            capture_samples=single_count,
            capture_left_samples=left_samples,
            capture_right_samples=right_samples,
            capture_min_samples=int(min_samples),
            capture_valid=complete,
            capture_left_valid=valid_by_side.get("left") if "left" in self.components.active_sides else None,
            capture_right_valid=valid_by_side.get("right") if "right" in self.components.active_sides else None,
            capture_elapsed_sec=float(elapsed_sec),
            capture_duration_sec=float(timeout_sec),
            capture_waiting_for_samples=bool(waiting_for_samples and not complete),
            control_hint=(
                "正在采集 runtime home 样本，请保持 VIVE 和机械臂静止。"
                if not complete
                else "runtime home 样本已采满，正在完成标定。"
            ),
        )

    def _calibrate_vive_coordinate_profile(self) -> None:
        if self.ui is None:
            return
        if self._teleop_started and not self._hold_active:
            message = "运行中不能重新标定 VIVE 坐标系；请先点击 Hold / 冻结。"
            self.ui.append_message("ViveCalibration", message)
            self.ui.publish(control_hint=message)
            return
        try:
            path = _calibrate_gui_vive_coordinate_profile(self.args, self.components, self.ui)
        except Exception as exc:  # noqa: BLE001
            message = f"标定 VIVE 坐标系失败: {type(exc).__name__}: {exc}"
            _log("ViveCalibration", message)
            self.ui.append_message("ViveCalibration", message)
            self.ui.publish(
                active_controls=[
                    "p",
                    "c",
                    "q",
                    *self._visualization_controls(),
                    *self._vive_calibration_controls(allow_load=True),
                    *self._movement_multiple_controls(),
                ],
                capture_stage="runtime_home",
                capture_phase="ready",
                capturing=False,
                capture_waiting_for_samples=False,
                **_movement_multiple_ui_state(self.args),
                **_vive_calibration_profile_ui_state(
                    self.args,
                    self.components,
                    status=message,
                    loaded=False,
                )
            )
            return
        message = f"VIVE 坐标系已重新标定并自动保存：{path}；请重新采集 runtime Home 后再遥操。"
        _log("ViveCalibration", message)
        self.ui.append_message("ViveCalibration", message)
        self.ui.publish(
            stage="VIVE 坐标系已保存",
            active_controls=[
                "p",
                "c",
                "q",
                *self._visualization_controls(),
                *self._vive_calibration_controls(allow_load=True),
                *self._movement_multiple_controls(),
            ],
            home_ready=False,
            capture_stage="runtime_home",
            capture_phase="ready",
            capturing=False,
            capture_waiting_for_samples=False,
            control_hint="坐标系已保存；请点击“开始遥操”或“采集 Home”重新采集 runtime Home。",
            **_movement_multiple_ui_state(self.args),
            **_vive_calibration_profile_ui_state(
                self.args,
                self.components,
                status="坐标系标定并保存成功",
                path=path,
                loaded=True,
            ),
        )

    def _save_vive_calibration_profile(self) -> None:
        if self.ui is None:
            return
        try:
            path = _save_gui_vive_calibration_profile(self.args, self.components)
        except Exception as exc:  # noqa: BLE001
            message = f"保存 VIVE 坐标系失败: {type(exc).__name__}: {exc}"
            _log("ViveCalibration", message)
            self.ui.append_message("ViveCalibration", message)
            self.ui.publish(
                **_vive_calibration_profile_ui_state(
                    self.args,
                    self.components,
                    status=message,
                    loaded=None,
                )
            )
            return
        message = f"已保存 VIVE 坐标系 profile：{path}"
        _log("ViveCalibration", message)
        self.ui.append_message("ViveCalibration", message)
        self.ui.publish(
            **_vive_calibration_profile_ui_state(
                self.args,
                self.components,
                status="保存成功",
                path=path,
                loaded=None,
            )
        )

    def _load_vive_calibration_profile(self) -> None:
        if self.ui is None:
            return
        if self._teleop_started and not self._hold_active:
            message = "运行中不能加载 VIVE 坐标系；请先点击 Hold / 冻结。"
            self.ui.append_message("ViveCalibration", message)
            self.ui.publish(
                control_hint=message,
                **_vive_calibration_profile_ui_state(self.args, self.components),
            )
            return
        try:
            path = _load_gui_vive_calibration_profile(self.args, self.components)
        except Exception as exc:  # noqa: BLE001
            message = f"加载 VIVE 坐标系失败: {type(exc).__name__}: {exc}"
            _log("ViveCalibration", message)
            self.ui.append_message("ViveCalibration", message)
            self.ui.publish(
                **_vive_calibration_profile_ui_state(
                    self.args,
                    self.components,
                    status=message,
                    loaded=False,
                )
            )
            return
        self._gripper_control_enabled = False
        message = f"已加载 VIVE 坐标系 profile：{path}；请重新采集 runtime home。"
        _log("ViveCalibration", message)
        self.ui.append_message("ViveCalibration", message)
        self.ui.publish(
            stage="已加载 VIVE 坐标系",
            active_controls=[
                "p",
                "c",
                "q",
                *self._visualization_controls(),
                *self._vive_calibration_controls(allow_load=True),
                *self._movement_multiple_controls(),
            ],
            control_hint="VIVE 坐标系已加载；请点击“开始遥操”或“采集 Home”重新采集 runtime home。",
            capture_stage="runtime_home",
            capture_phase="ready",
            teleop_running=False,
            home_ready=False,
            gripper_control_enabled=False,
            **_movement_multiple_ui_state(self.args),
            **_vive_calibration_profile_ui_state(
                self.args,
                self.components,
                status="加载成功，等待重采 runtime home",
                path=path,
                loaded=True,
            ),
        )

    def publish_ik_timing_state(
        self,
        payload: Mapping[str, object],
        *,
        now: float | None = None,
        force: bool = False,
    ) -> None:
        if self.ui is None or not self.ik_debug_timing_enabled():
            return
        current_time = time.time() if now is None else float(now)
        if (
            not force
            and current_time - self._last_ik_timing_publish_at < self._ik_timing_publish_interval_sec
        ):
            return
        self._last_ik_timing_publish_at = current_time
        self.ui.publish(debug_ik_timing=True, ik_timing=dict(payload))

    def publish_ik_warning_state(
        self,
        payload: Mapping[str, object],
        *,
        now: float | None = None,
        force: bool = False,
    ) -> None:
        if self.ui is None:
            return
        warning = dict(payload)
        current_time = time.time() if now is None else float(now)
        codes_value = warning.get("codes")
        sources_value = warning.get("sources")
        codes = (
            codes_value
            if isinstance(codes_value, Sequence) and not isinstance(codes_value, (str, bytes))
            else ()
        )
        sources = (
            sources_value
            if isinstance(sources_value, Sequence) and not isinstance(sources_value, (str, bytes))
            else ()
        )
        signature = (
            bool(warning.get("active", False)),
            tuple(str(code) for code in codes if code is not None),
            tuple(str(source) for source in sources if source is not None),
            str(warning.get("message") or ""),
        )
        changed = signature != self._last_ik_warning_signature
        if (
            not force
            and not changed
            and current_time - self._last_ik_warning_publish_at < self._ik_warning_publish_interval_sec
        ):
            return
        self._last_ik_warning_signature = signature
        self._last_ik_warning_publish_at = current_time
        self.ui.publish(ik_warning=warning)

    def publish_gripper_status(
        self,
        *,
        channels: dict[str, object] | None = None,
        command_status: Mapping[str, Mapping[str, object]] | None = None,
        now: float | None = None,
        force: bool = False,
    ) -> None:
        if self.ui is None or not _gripper_gui_available(self.components):
            return
        current_time = time.time() if now is None else float(now)
        if (
            not force
            and current_time - self._last_gripper_status_publish_at < self._gripper_status_publish_interval_sec
        ):
            return
        if channels is None:
            source = _hand_q6_source_from_components(self.components)
            channels = {} if source is None else source.get_latest_channels()
        self._last_gripper_status_publish_at = current_time
        self.ui.publish(
            gripper_control_available=_gripper_gui_available(self.components),
            gripper_q6_available=_has_q6_gripper_control(self.components),
            gripper_control_enabled=self.gripper_control_enabled(),
            gripper_status=_build_gripper_status(
                self.components,
                channels=channels,
                command_status=command_status,
                now=current_time,
            ),
        )

    def _toggle_vive_tracker_swap(self) -> None:
        hold_swap_allowed = self._single_arm_tracker_swap_allowed_in_hold()
        if self._tracker_swap_locked and not hold_swap_allowed:
            if self.ui is not None:
                self.ui.append_message(
                    "InputMapping",
                    "已开始遥操准备，运行中不能互换 VIVE tracker。",
                )
            return
        self.args.swap_operator_sides = not bool(
            getattr(self.args, "swap_operator_sides", False)
        )
        _apply_vive_tracker_mapping(self.args, self.components)
        message = self._control_hint()
        if hold_swap_allowed:
            message = (
                f"{message} 单臂 Hold 状态已切换 tracker；"
                "请重新点击“开始遥操”采集 runtime home。"
            )
        _log("InputMapping", message)
        if self.ui is not None:
            payload: dict[str, object] = {"control_hint": message}
            if hold_swap_allowed:
                payload.update(
                    {
                        "active_controls": self._hold_controls(),
                        "home_ready": False,
                        "teleop_running": False,
                    }
                )
            self.ui.publish(**payload)
            self.ui.append_message("InputMapping", message)
            self.publish_raw_tracker_pose_state(force=True)

    def publish_raw_tracker_pose_state(
        self,
        *,
        channels: dict[str, object] | None = None,
        now: float | None = None,
        force: bool = False,
    ) -> None:
        if self.ui is None:
            return
        current_time = time.time() if now is None else float(now)
        if not force and current_time - self._last_raw_pose_publish_at < self._raw_pose_publish_interval_sec:
            return
        if channels is None:
            source = _vive_pose_source_from_components(self.components)
            channels = {} if source is None else source.get_latest_channels()
        _publish_raw_tracker_pose_state(
            self.ui,
            channels,
            active_sides=self.components.active_sides,
            now=current_time,
        )
        self._last_raw_pose_publish_at = current_time

    def _control_hint(self) -> str:
        mapping = ", ".join(
            f"logical_{side}<-{_tracker_source_side_for_logical_side(self.args, side)}_tracker"
            for side in self.components.active_sides
        )
        return (
            f"VIVE tracker 映射: {mapping}. "
            "点击“交换 Tracker”只会交换 VIVE 输入；机械臂、手套串口、真手和 profile 都不变。"
        )


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


def _wait_for_real_teleop_start_request(
    components: RealTeleopComponents,
    clutch_input: RealTeleopControlInput,
    *,
    arm_mode: str,
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
        if clutch_input.should_quit:
            return None
        if clutch_input.consume_calibration_request():
            if clutch_input.before_calibration_start_delay(3.0, arm_mode=arm_mode):
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
    if home_captured:
        synced = _reset_runtime_motion_state_from_real_qpos(components, required=True)
        _log("Calibration", f"runtime restart motion state reset from live real qpos synced={synced}")
    components.session.set_calibration_ready()
    components.session.mark_debug_passed()
    _enable_impedance_force_control_after_runtime_home(
        args,
        components,
        home_captured=home_captured,
    )
    components.session.arm_hardware()
    clutch_input.publish_teleop_running(
        home_ready=home_captured,
        message=(
            "runtime home 完成，已进入真机遥操。"
            if home_captured
            else "runtime home 未就绪，未进入正常真机遥操。"
        ),
    )


def main() -> None:
    args = parse_args()
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
    clutch_input = RealTeleopControlInput(
        terminal=TerminalClutchInput(components.keyboard_source),
        ui=control_ui,
        args=args,
        components=components,
    )
    episode_recorder = EpisodeRecorder(
        root_dir=_episode_root_dir(args, components),
        config=EpisodeRecorderConfig(enabled=not args.no_episode_recording),
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
        _sync_dual_arm_retargeter_from_preview(components)
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
        if components.hand_backend_mode == HAND_BACKEND_DMTAC_GRIPPER:
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
            f"control_mode={_arm_control_mode_arg(args)} "
            f"impedance_mode={_impedance_mode_arg(args)} "
            f"strict_tool_profile={_strict_tool_profile_arg(args)} "
            f"force_tare_sec={_force_tare_sec_arg(args):.2f} "
            f"state_read_retries={args.rm65_hardware_read_retries} "
            f"retry_delay_sec={args.rm65_hardware_read_retry_delay_sec:.3f} "
            f"io_threads={max(1, int(args.arm_io_threads))} "
            f"streaming_hz={float(args.hz):.1f} "
            f"minimum_control_hz={args.minimum_control_hz:.1f} "
            f"arm_max_angvel_rad_s={np.round(np.asarray(args.arm_max_angvel, dtype=float), 3).tolist()} "
            f"actual_dt_scaling={'on' if args.arm_step_dt_scaling else 'off'} "
            f"dt_alpha={args.arm_step_dt_alpha:.2f} "
            f"dt_max_scale={args.arm_step_dt_max_scale:.2f} "
            f"dt_stall_reset_sec={args.arm_step_dt_stall_reset_sec:.3f}",
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
            _log("TempSafety", "temporary dual-arm output limiter disabled")
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
        startup_home_captured = False
        manual_startup_request = _manual_calibration_confirmation_required(args, components)
        while True:
            startup_request = "home"
            if manual_startup_request:
                requested = _wait_for_real_teleop_start_request(
                    components,
                    clutch_input,
                    arm_mode="real",
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
                "p/Enter=start calibration, t=start episode, s=stop episode, z=left clutch, x=right clutch, c=both, r=release, q=quit",
            )
        elif args.control_mode == "delta_clutch":
            _log("Info", "delta_clutch is running with both clutch buttons held true by default.")
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
                f"tactile_storage={args.tactile_storage} start_key='t' stop_key='s'",
            )

        status_gate = ChangeGate()
        orientation_gate = ChangeGate()
        tactile_sum_gate = ChangeGate()
        debug_timing_state = RealDebugTimingState(
            enabled=bool(getattr(args, "debug_timing", False)),
            threshold_sec=float(getattr(args, "debug_timing_threshold_sec", 0.025)),
            report_interval_sec=float(getattr(args, "debug_timing_report_interval_sec", 1.0)),
        )
        ik_debug_profile_base_enabled = bool(
            getattr(args, "debug_timing", False)
            or getattr(args, "debug_ik_timing", False)
        )
        ik_debug_log_recorder = IkDebugLogRecorder()
        if debug_timing_state.enabled:
            _log(
                "DebugTiming",
                "enabled "
                f"threshold={debug_timing_state.threshold_sec:.3f}s "
                f"report_interval={debug_timing_state.report_interval_sec:.3f}s",
            )
        if bool(getattr(args, "debug_ik_timing", False)):
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
        episode_last_state: str | None = None
        last_collection_frequency_warning_active = False
        last_collection_frequency_debug_at = 0.0
        tactile_preview_rate = PreviewRateLimiter(float(getattr(args, "tactile_vis_hz", 5.0)))
        camera_preview_rate = PreviewRateLimiter(float(getattr(args, "vis_cam_hz", 10.0)))

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
                        "storage_queue_size": int(args.storage_queue_size),
                    }
                )
                try:
                    episode_index = dataset_recorder.start_episode(metadata, now=now)
                except Exception as exc:  # noqa: BLE001
                    message = f"正式 dataset episode 启动失败: {type(exc).__name__}: {exc}"
                    _log("Dataset", message, now=now)
                    if control_ui is not None:
                        control_ui.append_message("Dataset", message)
                    return False
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
            )
            return True

        def stop_recording(
            now: float,
            *,
            end_reason: str,
            summary: Mapping[str, object] | None = None,
        ) -> None:
            nonlocal episode_last_state, last_collection_frequency_warning_active
            stopped_any = False
            frame_count = recording_frame_count()
            if dataset_recorder is not None and dataset_recorder.is_recording:
                task = dataset_recorder.stop_episode(
                    reason=end_reason,
                    step_index=step_index,
                    last_state=episode_last_state,
                )
                stopped_any = True
                if task is None:
                    message = "正式数采已停止：当前 episode 没有样本，已丢弃。"
                else:
                    message = (
                        f"正式数采已停止：episode {task.episode_index} "
                        f"{task.frame_count} 帧，正在后台收尾。"
                    )
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
                clutch_input.set_episode_recording_state(
                    active=False,
                    episode_dir=recording_path(),
                    frame_count=frame_count,
                )
                episode_last_state = None
                last_collection_frequency_warning_active = False

        def poll_dataset_finalize(now: float) -> None:
            if dataset_recorder is None:
                return
            for task in dataset_recorder.poll_finalize_tasks():
                elapsed = 0.0 if task.completed_at is None else task.completed_at - task.started_at
                if task.success:
                    message = (
                        f"正式数据集 episode {task.episode_index} 已保存 "
                        f"({task.frame_count} 帧，后台收尾 {elapsed:.2f}s)。"
                    )
                    _log("Dataset", message, now=now)
                    if control_ui is not None:
                        control_ui.append_message("Dataset", message)
                else:
                    message = f"正式数据集 episode {task.episode_index} 保存失败: {task.error}"
                    _log("Dataset", message, now=now)
                    if control_ui is not None:
                        control_ui.append_message("Dataset", message)

        if args.record_auto_start:
            start_recording(time.time(), start_reason="auto_start")

        emergency_stop_sent = False
        while True:
            loop_start = time.time()
            loop_phase_timer = (
                PhaseTimer()
                if (debug_timing_state.enabled or recording_active())
                else None
            )

            def mark_loop_phase(name: str) -> None:
                if loop_phase_timer is not None:
                    loop_phase_timer.mark(name)

            clutch_input.poll()
            poll_dataset_finalize(loop_start)
            if clutch_input.consume_ik_debug_log_toggle_request():
                if ik_debug_log_recorder.active:
                    stopped_path = ik_debug_log_recorder.stop(reason="gui_toggle_stop", now=loop_start)
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
                        now=loop_start,
                    )
                    if control_ui is not None:
                        control_ui.append_message(
                            "IKDebugLog",
                            f"已停止 IK 记录：{stopped_path}",
                        )
                else:
                    started_path = ik_debug_log_recorder.start(now=loop_start)
                    ik_debug_log_mode = (
                        "full" if bool(getattr(args, "ik_debug_log_full", False)) else "compact"
                    )
                    ik_debug_log_recorder.write_event(
                        "mode",
                        {"mode": ik_debug_log_mode},
                        now=loop_start,
                    )
                    if components.dual_arm_retargeter is not None and len(components.active_sides) > 1:
                        components.dual_arm_retargeter.config.debug_profile = True
                        components.dual_arm_retargeter.ik_solver.debug_profile = True
                    clutch_input.set_ik_debug_logging_state(active=True, path=str(started_path))
                    _log(
                        "IKDebugLog",
                        f"started path={started_path} mode={ik_debug_log_mode}",
                        now=loop_start,
                    )
                    if control_ui is not None:
                        control_ui.append_message(
                            "IKDebugLog",
                            f"已开始 IK 记录：{started_path} mode={ik_debug_log_mode}",
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
                hold_hardware_reason = (
                    "q6 gesture hold requested"
                    if hold_reason == "q6_gesture"
                    else "gui hold requested"
                )
                if recording_active():
                    stop_recording(
                        loop_start,
                        end_reason="hold_requested",
                        summary={
                            "last_step_index": step_index,
                            "last_state": episode_last_state,
                        },
                    )
                recovery_plan = None
                components.session.hold_hardware(hold_hardware_reason)
                _log(
                    "Hold",
                    f"{hold_hardware_reason}; waiting for explicit runtime home calibration",
                    now=loop_start,
                )
                if control_ui is not None:
                    message = (
                        "机械臂已进入保持状态；松开双手后三维到释放区间 1 秒，再次握紧 1 秒后会重新采集 runtime home。"
                        if hold_reason == "q6_gesture"
                        else "机械臂已进入保持状态；点击“开始遥操/开始采集”后才会重新采集 runtime home。"
                    )
                    control_ui.append_message("Hold", message)
                continue
            if clutch_input.consume_calibration_request():
                if recording_active():
                    stop_recording(
                        loop_start,
                        end_reason="runtime_recalibration_requested",
                        summary={
                            "last_step_index": step_index,
                            "last_state": episode_last_state,
                        },
                    )
                recovery_plan = None
                if components.session.state_machine.state != TeleopRunState.DEBUG_ONLY:
                    components.session.hold_hardware("runtime home recalibration requested")
                _log("Calibration", "explicit runtime home recalibration requested from hold", now=loop_start)
                if not clutch_input.before_calibration_start_delay(3.0, arm_mode="real"):
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
            if clutch_input.consume_episode_start_request():
                if not recording_active():
                    start_recording(loop_start, start_reason="manual_key_start")
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
                        gripper_control_enabled=False,
                        **_movement_multiple_ui_state(args),
                    )
                if loop_phase_timer is not None:
                    mark_loop_phase("loop.hold_idle")
                time.sleep(max(0.0, min(0.02, (1.0 / float(args.hz)) - (time.time() - loop_start))))
                continue

            _sync_dual_arm_retargeter_from_preview(components)
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
            clutch_input.publish_raw_tracker_pose_state(
                channels=step_result.channels,
                now=loop_start,
            )
            gripper_command = None
            gripper_command_status: dict[str, dict[str, object]] = {}
            if clutch_input.gripper_control_enabled():
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
            mark_loop_phase("loop.gripper_control")
            active_command = step_result.command
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
                    if active_command is not None:
                        if _guard_impedance_force_control(
                            args,
                            components,
                            context="recovery_start",
                            now=loop_start,
                        ):
                            assert components.session.rig is not None
                            components.session.rig.send_commands(active_command)
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
                    if active_command is not None:
                        if _guard_impedance_force_control(
                            args,
                            components,
                            context="recovery_advance",
                            now=loop_start,
                        ):
                            assert components.session.rig is not None
                            components.session.rig.send_commands(active_command)
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

            _apply_active_arm_command_to_preview_real(components, active_command)
            mark_loop_phase("loop.preview_apply")
            _sync_dual_arm_retargeter_from_preview(components)
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
            if not recording_active():
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
            clutch_channels = components.keyboard_source.get_latest_channels()
            left_clutch_channel = clutch_channels.get("left_clutch_button")
            right_clutch_channel = clutch_channels.get("right_clutch_button")
            left_clutch = bool(left_clutch_channel.value) if left_clutch_channel is not None else False
            right_clutch = bool(right_clutch_channel.value) if right_clutch_channel is not None else False
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
                if dataset_recorder is not None and dataset_recorder.is_recording:
                    collection_metadata_extra = None
                    if loop_phase_timer is not None:
                        collection_metadata_extra = {
                            "loop_phase_timings_ms": [
                                {"name": name, "elapsed_ms": elapsed * 1000.0}
                                for name, elapsed in loop_phase_timer.phases
                            ],
                            "loop_elapsed_before_record_s": loop_phase_timer.elapsed_sec(),
                        }
                    dataset_recorder.record_step(
                        _collection_step_result(
                            step_result,
                            active_command=active_command,
                            gripper_command=gripper_command,
                        ),
                        include_debug_metadata=bool(ik_debug_log_recorder.active),
                        phase_recorder=phase_recorder,
                        metadata_extra=collection_metadata_extra,
                    )
                frequency_status = recording_frequency_status()
                publish_collection_frequency_warning(frequency_status, now=loop_start)
                if episode_recorder.is_recording:
                    episode_recorder.record_frame(
                        _build_episode_frame(
                            components,
                            step_index=step_index,
                            step_result=step_result,
                            active_command=active_command,
                            hand_command=hand_command,
                            recovery_plan=recovery_plan,
                            left_clutch=left_clutch,
                            right_clutch=right_clutch,
                        ),
                        timestamp=loop_start,
                    )
                clutch_input.set_episode_recording_state(
                    active=True,
                    episode_dir=recording_path(),
                    frame_count=recording_frame_count(),
                    frequency_status=frequency_status,
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
            ik_warning_payload = _build_ik_warning_payload(
                active_command=active_command,
                left_command_state=left_command_state,
                right_command_state=right_command_state,
            )
            clutch_input.publish_ik_warning_state(
                ik_warning_payload,
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
                else 1.0 / float(args.hz)
            )
            target_step_scale = (
                float(components.arm_loop_step_dt_tracker.get("smoothed_loop_dt", base_control_dt))
                / base_control_dt
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
            step_scale_snapshot = (
                round(float(left_command_state.get("effective_step_scale", 1.0)), 3),
                round(float(right_command_state.get("effective_step_scale", 1.0)), 3),
                round(
                    target_step_scale,
                    3,
                ),
            )
            command_backend_snapshot = (
                _command_backend_status_text(left_command_state),
                _command_backend_status_text(right_command_state),
            )
            status_snapshot = (
                step_result.state.value,
                fault_codes,
                channel_valid,
                left_clutch,
                right_clutch,
                collision_status,
                collision_fallback,
                jump_gate_status,
                branch_candidate,
                temporary_safety_status,
                hardware_snapshot,
                command_backend_snapshot,
                step_scale_snapshot,
                hand_input_status,
                gripper_status,
                clutch_input.gripper_control_enabled(),
            )
            mark_loop_phase("loop.status_snapshot")
            if status_gate.should_emit(
                "real_teleop_status",
                status_snapshot,
                now=loop_start,
                heartbeat_sec=args.status_sec,
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
                    f"{collision_bits}"
                    f"{hand_bits}"
                    f"{gripper_bits}",
                    now=loop_start,
                )
            mark_loop_phase("loop.status_emit")
            if loop_phase_timer is not None:
                active_elapsed_sec = loop_phase_timer.elapsed_sec()
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

            step_index += 1
            if args.max_steps > 0 and step_index >= args.max_steps:
                break

            remaining = (1.0 / args.hz) - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        if "ik_debug_log_recorder" in locals():
            ik_debug_log_recorder.stop(reason="finally", now=time.time())
        if "recording_active" in locals() and recording_active():
            stop_recording(
                time.time(),
                end_reason="shutdown",
                summary={"reason": "finally"},
            )
        if dataset_recorder is not None:
            try:
                dataset_recorder.close()
            except Exception as exc:  # noqa: BLE001
                _log("Dataset", f"close failed: {type(exc).__name__}: {exc}")
        episode_recorder.close()
        clutch_input.close()
        components.session.close()
        if components.arm_state_read_executor is not None:
            components.arm_state_read_executor.shutdown(wait=True)
        if components.left_state_backend is not None:
            components.left_state_backend.close()
        if components.right_state_backend is not None:
            components.right_state_backend.close()
        if components.left_hand_backend is not None and not components.left_hand_control_via_rig:
            components.left_hand_backend.close()
        if components.right_hand_backend is not None and not components.right_hand_control_via_rig:
            components.right_hand_backend.close()
        if camera_previewer is not None:
            camera_previewer.close()
        _close_camera_sources(camera_sources)
        if tactile_visualizer is not None:
            tactile_visualizer.close()
        if original_stdout is not None:
            sys.stdout = original_stdout
        if silenced_stdout is not None:
            silenced_stdout.close()


if __name__ == "__main__":
    main()
