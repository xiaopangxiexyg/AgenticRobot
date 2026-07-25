"""Supported real-policy deployment composition runtime.

The canonical shell entrypoint is ``python -m teleop_runtime.deploy``.  Its
thin CLI wrapper delegates to :func:`main`; this module also remains a
supported direct/programmatic entrypoint for active callers such as the OpenPI
real launcher.  It has no announced end-of-life.  Any future retirement must
first migrate direct callers, publish a replacement and migration guide, then
run through an explicit deprecation period.
"""

from __future__ import annotations

import argparse
import json
import math
import select
import sys
import threading
import termios
import time
import tty
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ..configs.camera_paths import DEFAULT_REALSENSE_DEVICE_CONFIG_PATH
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.layout import episode_id_from_index
from dataset_contracts.policy_artifact import policy_artifact_from_metadata
from dataset_contracts.pose_frames import load_pose_frame_contract
from dataset_contracts.pose_training import POSE_TRAINING_ORIGIN_NATIVE_V23
from dataset_contracts.v22 import DEXTELEOP_DATASET_SCHEMA_V22, policy_io_layout_from_metadata
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from deployments.client import load_policy_artifact
from project_paths import repo_path
from robot_motion.rm65.assets import require_asset_path
from robot_motion.action import HandActionSpec, hand_action_specs_metadata
from robot_motion.deploy.actions import (
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    NormalizedAction,
)
from robot_motion.deploy.canonical_action import CanonicalDeployActionPipeline
from robot_motion.deploy.async_producer import (
    POLICY_ASYNC_PRODUCER_ELIGIBLE,
    RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
    RAW_RTC_FINAL_CHUNK_PROFILE_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_V1,
)
from robot_motion.deploy.observation import (
    DEFAULT_HAND_STATE_MAX_AGE_SEC,
    build_deployment_observation,
)
from robot_motion.deploy.observation_view import default_deploy_policy_observation_contract
from robot_motion.deploy.pose_adapter import convert_policy_absolute_vector_to_runtime_raw
from robot_motion.deploy.runtime import (
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    resolve_action_queue_policy,
)
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    ACTION_HAND_SLICES,
    ALL_SIDES,
    DEPLOY_ACTION_VECTOR_ACTION_SPACE,
    pose_matrix_to_xyzquat,
    to_serializable,
)
from robot_motion.deploy.session_contract import (
    POLICY_CONTRACT_V22_NATIVE,
    RuntimeEndEffectorCapability,
    resolve_deploy_session_contract,
    resolve_policy_action_pose_mode_from_artifact,
)
from robot_motion.replay.trajectory import QPOS_INIT_SOURCES
from robot_motion.rm65 import (
    DEFAULT_COLLISION_SCENE,
    DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M,
    DEFAULT_DUAL_ARM_COLLISION_MIN_DISTANCE_M,
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
    IKObjectiveWeights,
    LEGACY_POSITION_CONTROL_PROFILE_NAME,
    RM65_BIMANUAL_POSITION_V1,
    RM65_BIMANUAL_POSITION_V1_NAME,
    RM65_POSITION_CONTROL_PROFILE_CHOICES,
    RM65_POSITION_TRACKING_FOLLOW_ALPHA,
    RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
    RM65_POSITION_TRACKING_MAX_COMMAND_STEP_DEG,
    RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S,
    load_ik_tuning_config,
)
from robot_motion.position_control import (
    DEFAULT_POSITION_DEGRADED_SPEED_SCALE,
    DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
    DEFAULT_POSITION_VELOCITY_RAMP_SEC,
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
    POSITION_COMMAND_PLANNER_CHOICES,
)
from robot_motion.pose_frame_profiles import default_rm65_workspace_near_boundary_runtime_pose_frame_contract
from robot_motion.runtime import TeleopCommand
from robot_motion.runtime.causal_history import CausalSnapshotHistory
from robot_motion.runtime_gate import IKActionChunkGate

from ..backends.arms import (
    DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN,
    RM65ArmBackend,
    RM65ArmBackendConfig,
    RM65ImpedanceArmBackend,
    RM65ImpedanceArmBackendConfig,
)
from ..action.replay_proposal import (
    DatasetReplayProposalShadowAdapter,
    DatasetReplayProposalShadowConfig,
    replay_adapter_debug_payload,
)
from .rm65_legacy_worker_binding import RM65LegacyWorkerExecutionBinding
from .rtc_frontend import RtcPolicyShadowConfig, RtcPolicyShadowFrontend
from .rtc_debug import (
    RTC_DEBUG_LEVEL_OFF,
    RTC_DEBUG_LEVEL_SUMMARY,
    RTC_DEBUG_LEVEL_TRACE,
    RTC_DEBUG_LEVELS,
    RtcDebugEventObserver,
    normalize_rtc_debug_level,
)
from .rtc_step_fanout import (
    PolicyActionEmissionFanout,
    RtcFrontendStepFanout,
    RtcPolicyEmissionPublisher,
)
from deployments.client.rtc_frontend import build_rtc_model_shadow_frontend
from ..backends.grippers import (
    DEFAULT_DMTAC_GRIPPER_COMMAND_HZ,
    DEFAULT_DMTAC_GRIPPER_TELEMETRY_HZ,
    DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
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
from ..collection.camera_preview import CameraPreviewConfig, CameraPreviewer
from ..collection.hand_state_polling import (
    HandStateHistoryPoller,
    HandStateHistoryPollerConfig,
)
from ..configs.dmtac_hardware import (
    add_dmtac_assignment_arguments,
    dmtac_backend_configs_from_assignment,
    load_dmtac_hardware_assignment,
    manual_dmtac_backend_config_for_side,
    require_dmtac_sides_resolved,
)
from ..defaults import (
    DEFAULT_RM65_ARM_CONTROL_HZ,
    DEFAULT_RM65_POSITION_ARM_CONTROL_HZ,
    DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
    DEFAULT_TELEOP_SAMPLE_HZ,
)
from ..execution_mode_contract import validate_dataset_arm_control_mode
from ..hand_mapping import parse_tactile_regions
from ..interfaces import EffectorBackend
from ..position_control import (
    ARM_EXECUTION_PROFILE_CHOICES,
    POSITION_15_60_120HZ_PROFILE_NAME,
    PositionDeployExecutor,
    PositionDeployExecutorConfig,
    PositionDeployExecutorFault,
    apply_rm65_position_profile_defaults,
    build_position_collision_planner_from_started_backends,
    resolve_real_arm_execution_config,
)
from ..sources.camera import (
    RealSenseRGBCameraConfig,
    RealSenseRGBSource,
    load_realsense_camera_config_map,
)
from ..tactile_visualizer import (
    TactileUVVisualizer,
    TactileVisualizerConfig,
    tactile_by_side_from_hand_states,
)
from .dispatch import (
    dispatch_teleop_command,
    require_arm_backends_ready_for_position_dispatch,
)
from .arbiter_shadow import (
    ARBITER_SHADOW_MODE_SHADOW,
    ARBITER_SHADOW_MODES_M3,
    PassivePolicyArbiterShadow,
    PolicyArbiterShadowConfig,
)
from .fixed_init import DeployFixedInitController, load_deploy_fixed_init_plan
from deployments.client.policy import load_policy
from deployments.client.rtc_executor import (
    rtc_client_inference_bundle_debug_payload,
)
from .policy_qpos_init import (
    DeployPolicyQposInitController,
    DeployPolicyQposInitSettings,
    load_deploy_policy_qpos_init_plan,
)
from .replay import (
    DatasetReplayPolicy,
    infer_replay_action_hz,
    load_dataset_replay_policy,
    resolve_replay_parquet_path,
)
from .safety_shadow import evaluate_policy_legacy_aligned_safety
from .startup_init import StartupInitializer, resolve_startup_init_action


DEFAULT_CAMERA_CONFIG_PATH = (
    DEFAULT_REALSENSE_DEVICE_CONFIG_PATH
)
# Deployment runs the vision policy at roughly 15 Hz on the RM65 profile.  A
# 100 ms observation gate leaves only one half-frame of jitter budget at that
# rate, so keep a deployment-specific margin without changing collection's
# stricter sampling contract.
DEFAULT_DEPLOYMENT_CAMERA_FRAME_MAX_AGE_SEC = 0.2
DEFAULT_DEPLOYMENT_CAMERA_KEEP_LAST = 4
HAND_BACKEND_INSPIRE = "inspire"
HAND_BACKEND_DMTAC_GRIPPER = "dmtac-gripper"
HAND_BACKEND_CHOICES = (HAND_BACKEND_INSPIRE, HAND_BACKEND_DMTAC_GRIPPER)
GRIPPER_POLICY_DEFAULTS = {
    "output_dim": 0,
    "speed": 0.2,
    "torque": 0.2,
    "max_action": GRIPPER_ACTION_MAX,
    "send_threshold": 0.002,
    "command_hz": DEFAULT_DMTAC_GRIPPER_COMMAND_HZ,
    "continuous_grasp": True,
    "telemetry_hz": DEFAULT_DMTAC_GRIPPER_TELEMETRY_HZ,
    "telemetry_command_grace_s": DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
}
DEPLOY_RM65_ARM_MAX_ANGVEL = (0.7, 0.7, 0.9, 1.5, 1.5, 2.0)
DEPLOY_ARM_STEP_DT_ALPHA = 0.25
DEPLOY_ARM_STEP_DT_MAX_SCALE = 3.0
DEPLOY_ARM_STEP_DT_STALL_RESET_SEC = 0.25
DEPLOY_MAX_ARM_JOINT_STEP_RAD = float(np.deg2rad(6.0))
RM65_ARM_CONTROL_MODE_POSITION = "position"
RM65_ARM_CONTROL_MODE_IMPEDANCE = "impedance"
RM65_ARM_CONTROL_MODES = (RM65_ARM_CONTROL_MODE_POSITION, RM65_ARM_CONTROL_MODE_IMPEDANCE)


@dataclass(frozen=True)
class _PreparedActionExecution:
    accepted: bool
    mode: str
    dispatch_attempted: bool
    dispatch_completed: bool
    asynchronous: bool = False
    position_source_index: int | None = None


def normalize_rm65_arm_control_mode(value: object) -> str:
    text = str(value or RM65_ARM_CONTROL_MODE_POSITION).strip().lower().replace("-", "_")
    aliases = {
        "pos": RM65_ARM_CONTROL_MODE_POSITION,
        "movej": RM65_ARM_CONTROL_MODE_POSITION,
        "canfd": RM65_ARM_CONTROL_MODE_POSITION,
        "force": RM65_ARM_CONTROL_MODE_IMPEDANCE,
        "force_control": RM65_ARM_CONTROL_MODE_IMPEDANCE,
        "admittance": RM65_ARM_CONTROL_MODE_IMPEDANCE,
        "force_admittance": RM65_ARM_CONTROL_MODE_IMPEDANCE,
    }
    mode = aliases.get(text, text)
    if mode not in RM65_ARM_CONTROL_MODES:
        raise ValueError(f"Unsupported RM65 arm control mode {value!r}; expected one of {RM65_ARM_CONTROL_MODES}.")
    return mode


def effective_arm_control_hz(args: argparse.Namespace) -> float:
    value = getattr(args, "arm_control_hz", None)
    if value is not None:
        return float(value)
    if normalize_rm65_arm_control_mode(getattr(args, "arm_control_mode", None)) == RM65_ARM_CONTROL_MODE_IMPEDANCE:
        return float(DEFAULT_RM65_ARM_CONTROL_HZ)
    return float(DEFAULT_RM65_POSITION_ARM_CONTROL_HZ)


def effective_position_reference_hz(args: argparse.Namespace) -> float:
    return float(
        getattr(
            args,
            "position_reference_hz",
            DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
        )
    )


def effective_arm_canfd_follow(args: argparse.Namespace) -> bool:
    value = getattr(args, "arm_canfd_follow", None)
    if value is not None:
        return bool(value)
    return normalize_rm65_arm_control_mode(getattr(args, "arm_control_mode", None)) == RM65_ARM_CONTROL_MODE_POSITION


class NonBlockingKeyboard:
    def __init__(self):
        self._fd: int | None = None
        self._old_settings: list[Any] | None = None

    def __enter__(self) -> "NonBlockingKeyboard":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def poll(self) -> str | None:
        if self._fd is None:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return None
        return sys.stdin.read(1)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)


class LoopRate:
    def __init__(self):
        self._last: float | None = None
        self.hz = 0.0

    def tick(self, now: float) -> float:
        if self._last is not None:
            dt = max(float(now - self._last), 1e-9)
            instant = 1.0 / dt
            self.hz = instant if self.hz <= 0.0 else 0.2 * instant + 0.8 * self.hz
        self._last = float(now)
        return float(self.hz)


_DEBUG_REDACT_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "password",
    "secret",
}


def _debug_redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in _DEBUG_REDACT_KEYS:
                redacted[key_text] = "<redacted>"
            else:
                redacted[key_text] = _debug_redact(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_debug_redact(item) for item in value]
    return value


def _debug_args_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = dict(vars(args))
    policy_kwargs_text = payload.get("policy_kwargs_json")
    if isinstance(policy_kwargs_text, str):
        try:
            policy_kwargs = json.loads(policy_kwargs_text or "{}")
        except json.JSONDecodeError:
            policy_kwargs = policy_kwargs_text
        payload["policy_kwargs_json"] = _debug_redact(policy_kwargs)
    return _debug_redact(payload)


def _debug_vector(
    value: Any,
    *,
    length: int | None = None,
    require_finite: bool = True,
) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if length is not None and vector.shape != (int(length),):
        return None
    if require_finite and not np.all(np.isfinite(vector)):
        return None
    return vector.astype(np.float32, copy=True)


def _debug_xyzquat(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except Exception:
        return None
    if array.shape == (4, 4):
        pose = pose_matrix_to_xyzquat(array)
        return None if pose is None else pose.astype(np.float32, copy=True)
    vector = _debug_vector(array, length=7)
    if vector is None:
        return None
    quat = np.asarray(vector[3:7], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return None
    vector = vector.astype(np.float32, copy=True)
    vector[3:7] = (quat / norm).astype(np.float32)
    return vector


def _debug_action_vector(action: Any) -> np.ndarray | None:
    vector = getattr(action, "vector", action)
    return _debug_vector(vector, length=ACTION_DIM, require_finite=False)


def _debug_pose_by_side_from_vector(
    vector: np.ndarray | None,
    active_sides: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if vector is None:
        return {}
    poses: dict[str, np.ndarray] = {}
    for side in active_sides:
        target_slice = ACTION_ARM_POSE_SLICES[side]
        pose = _debug_xyzquat(vector[target_slice])
        if pose is not None:
            poses[side] = pose
    return poses


def _debug_hand_by_side_from_vector(
    vector: np.ndarray | None,
    active_sides: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if vector is None:
        return {}
    hands: dict[str, np.ndarray] = {}
    for side in active_sides:
        target_slice = ACTION_HAND_SLICES[side]
        target = _debug_vector(vector[target_slice])
        if target is not None:
            hands[side] = target
    return hands


def _debug_arm_joint_targets(action: Any, active_sides: tuple[str, ...]) -> dict[str, np.ndarray]:
    targets = getattr(action, "arm_joint_targets", {}) or {}
    if not isinstance(targets, Mapping):
        return {}
    resolved: dict[str, np.ndarray] = {}
    for side in active_sides:
        target = _debug_vector(targets.get(side), length=6)
        if target is not None:
            resolved[side] = target
    return resolved


def _debug_action_summary(action: Any, active_sides: tuple[str, ...]) -> dict[str, Any] | None:
    vector = _debug_action_vector(action)
    if vector is None:
        return None
    metadata = getattr(action, "metadata", {})
    return {
        "source": str(metadata.get("source", "")) if isinstance(metadata, Mapping) else "",
        "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
        "vector": vector,
        "arm_pose_by_side": _debug_pose_by_side_from_vector(vector, active_sides),
        "arm_joint_targets_by_side": _debug_arm_joint_targets(action, active_sides),
        "hand_by_side": _debug_hand_by_side_from_vector(vector, active_sides),
    }


def _debug_quat_delta_deg(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    ref_quat = np.asarray(reference[3:7], dtype=float)
    candidate_quat = np.asarray(candidate[3:7], dtype=float)
    ref_norm = float(np.linalg.norm(ref_quat))
    candidate_norm = float(np.linalg.norm(candidate_quat))
    if ref_norm <= 1e-12 or candidate_norm <= 1e-12:
        return None
    ref_quat = ref_quat / ref_norm
    candidate_quat = candidate_quat / candidate_norm
    dot = float(np.clip(abs(np.dot(ref_quat, candidate_quat)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _debug_pose_delta(reference: Any, candidate: Any) -> dict[str, Any] | None:
    ref_pose = _debug_xyzquat(reference)
    candidate_pose = _debug_xyzquat(candidate)
    if ref_pose is None or candidate_pose is None:
        return None
    xyz_delta = np.asarray(candidate_pose[:3] - ref_pose[:3], dtype=np.float32)
    return {
        "xyz_delta_m": xyz_delta,
        "xyz_norm_m": float(np.linalg.norm(xyz_delta)),
        "rot_delta_deg": _debug_quat_delta_deg(ref_pose, candidate_pose),
    }


def _debug_pose_delta_by_side(
    reference_by_side: Mapping[str, Any],
    candidate_by_side: Mapping[str, Any],
) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for side in ALL_SIDES:
        if side not in reference_by_side or side not in candidate_by_side:
            continue
        delta = _debug_pose_delta(reference_by_side[side], candidate_by_side[side])
        if delta is not None:
            deltas[side] = delta
    return deltas


def _debug_command_arm_pose_by_side(command: Any, active_sides: tuple[str, ...]) -> dict[str, np.ndarray]:
    if command is None:
        return {}
    commands = getattr(command, "commands", {}) or {}
    poses: dict[str, np.ndarray] = {}
    for side in active_sides:
        effector_command = commands.get(f"{side}_arm")
        if effector_command is None or getattr(effector_command, "command_type", "") != "cartesian_pose":
            continue
        metadata = getattr(effector_command, "metadata", {}) or {}
        pose = None
        if isinstance(metadata, Mapping):
            pose = _debug_xyzquat(metadata.get("target_xyzquat"))
            if pose is None:
                pose = _debug_xyzquat(metadata.get("target_pose_base"))
        if pose is None:
            pose = _debug_xyzquat(getattr(effector_command, "value", None))
        if pose is not None:
            poses[side] = pose
    return poses


def _debug_command_summary(command: Any, active_sides: tuple[str, ...]) -> dict[str, Any] | None:
    if command is None:
        return None
    commands = getattr(command, "commands", {}) or {}
    command_payload: dict[str, Any] = {}
    for effector_name, effector_command in commands.items():
        metadata = getattr(effector_command, "metadata", {}) or {}
        item: dict[str, Any] = {
            "command_type": getattr(effector_command, "command_type", ""),
            "side": getattr(effector_command, "side", None),
            "frame_name": getattr(effector_command, "frame_name", None),
            "value": getattr(effector_command, "value", None),
            "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
        }
        if item["command_type"] == "cartesian_pose":
            item["target_xyzquat"] = _debug_xyzquat(
                item["metadata"].get("target_xyzquat")
                if isinstance(item["metadata"], Mapping)
                else None
            )
            if item["target_xyzquat"] is None:
                item["target_xyzquat"] = _debug_xyzquat(getattr(effector_command, "value", None))
        command_payload[str(effector_name)] = item
    return {
        "timestamp": getattr(command, "timestamp", None),
        "metadata": dict(getattr(command, "metadata", {}) or {}),
        "commands": command_payload,
        "arm_pose_by_side": _debug_command_arm_pose_by_side(command, active_sides),
    }


def _debug_arm_state_summary(states: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any]:
    if not states:
        return {}
    keys = (
        "current_qpos",
        "current_tcp_pose",
        "hardware_qpos",
        "last_target_qpos",
        "last_command",
        "last_command_timestamp",
        "hardware_pose_6d",
        "hardware_refresh_time",
        "hardware_result_code",
        "hardware_result_text",
        "hardware_arm_err",
        "hardware_sys_err",
        "send_result_code",
        "send_result_text",
        "emergency_stopped",
        "effective_control_dt_sec",
        "effective_step_scale",
        "ik_metadata",
        "singularity_metadata",
        "tcp_pose_source",
        "state_error",
        "pose_error",
        "active_tool_name",
        "configured_tcp_enabled",
        "tool_tcp_source",
        "tool_tcp_endpoint_delta_m",
        "tool_tcp_endpoint_delta_deg",
    )
    payload: dict[str, Any] = {}
    for side, state in states.items():
        side_payload = {key: state.get(key) for key in keys if key in state}
        pose = _debug_xyzquat(side_payload.get("current_tcp_pose"))
        if pose is not None:
            side_payload["current_tcp_xyzquat"] = pose
        payload[str(side)] = side_payload
    return payload


def _debug_state_pose_by_side(states: Mapping[str, Mapping[str, Any]] | None) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    if not states:
        return poses
    for side, state in states.items():
        pose = _debug_xyzquat(state.get("current_tcp_pose"))
        if pose is not None:
            poses[str(side)] = pose
    return poses


def _debug_backend_arm_states(backends: Mapping[str, EffectorBackend]) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for side, backend in backends.items():
        try:
            states[side] = dict(backend.get_state())
        except Exception as exc:  # noqa: BLE001
            states[side] = {"state_error": f"{type(exc).__name__}: {exc}"}
    return states


def _debug_backend_limits(
    backends: Mapping[str, EffectorBackend],
    *,
    now: float,
) -> dict[str, Any]:
    limits: dict[str, Any] = {}
    for side, backend in backends.items():
        get_effective_step = getattr(backend, "get_effective_max_joint_step", None)
        try:
            effective_step = get_effective_step(now=now) if callable(get_effective_step) else None
        except Exception as exc:  # noqa: BLE001
            effective_step = None
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = ""
        config = backend.config
        limits[side] = {
            "control_dt_sec": float(config.control_dt),
            "max_angvel_rad_s": tuple(float(value) for value in config.max_angvel),
            "effective_max_joint_step_rad": effective_step,
            "background_streaming_enabled": bool(getattr(config, "background_streaming_enabled", False)),
            "background_streaming_hz": (
                None
                if getattr(config, "background_streaming_hz", None) is None
                else float(getattr(config, "background_streaming_hz"))
            ),
            "canfd_follow": getattr(config, "canfd_follow", None),
            "observation_only": bool(getattr(config, "observation_only", False)),
            "send_commands": getattr(config, "send_commands", None),
            "error": error,
        }
    return limits


class DeploymentDebugLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        root_dir: Path | None,
        active_sides: tuple[str, ...],
        flush_every_events: int = 1,
        flush_interval_sec: float = 0.0,
    ):
        self.enabled = bool(enabled)
        self.active_sides = tuple(active_sides)
        self.path: Path | None = None
        self._handle: Any = None
        self._lock = threading.RLock()
        self._write_failed = False
        self._flush_every_events = int(flush_every_events)
        self._flush_interval_sec = float(flush_interval_sec)
        if self._flush_every_events <= 0:
            raise ValueError("flush_every_events must be positive.")
        if (
            not math.isfinite(self._flush_interval_sec)
            or self._flush_interval_sec < 0.0
        ):
            raise ValueError("flush_interval_sec must be finite and non-negative.")
        self._pending_flush_events = 0
        self._last_flush_at = time.monotonic()
        if not self.enabled:
            return
        debug_root = root_dir if root_dir is not None else repo_path("teleop_records", "deployment_debug")
        debug_root = Path(debug_root).expanduser()
        if not debug_root.is_absolute():
            debug_root = repo_path(debug_root)
        debug_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int((time.time() % 1.0) * 1000.0)
        self.path = debug_root / f"deploy_debug_{stamp}_{millis:03d}.jsonl"
        self._handle = self.path.open("a", encoding="utf-8")
        print(f"[DeployDebug] writing action/pose debug JSONL to {self.path}", flush=True)

    def close(self) -> None:
        with self._lock:
            if self._handle is None:
                return
            self._handle.flush()
            self._handle.close()
            self._handle = None

    def log_event(self, event: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            if not self.enabled or self._handle is None:
                return
            record = {
                "event": str(event),
                "wall_time": time.time(),
                **dict(payload),
            }
            try:
                self._handle.write(
                    json.dumps(
                        to_serializable(record),
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                self._pending_flush_events += 1
                now = time.monotonic()
                if (
                    self._pending_flush_events >= self._flush_every_events
                    or (
                        self._flush_interval_sec > 0.0
                        and now - self._last_flush_at
                        >= self._flush_interval_sec
                    )
                ):
                    self._handle.flush()
                    self._pending_flush_events = 0
                    self._last_flush_at = now
            except Exception as exc:  # noqa: BLE001
                if not self._write_failed:
                    print(f"[DeployDebug] disabled after write failure: {type(exc).__name__}: {exc}", flush=True)
                self._write_failed = True
                self.enabled = False

    def log_startup(
        self,
        *,
        args: argparse.Namespace,
        policy: Any,
        active_sides: tuple[str, ...],
        hands_enabled: bool,
        effective_action_hz: float,
        action_hz_info: Mapping[str, Any],
        effective_policy_hz: float,
        policy_hz_source: str,
        backend_control_hz: float,
        action_queue_policy: str,
        send_actions: bool,
        arm_backends: Mapping[str, EffectorBackend],
    ) -> None:
        if not self.enabled:
            return
        self.log_event(
            "startup",
            {
                "args": _debug_args_payload(args),
                "policy_name": getattr(policy, "name", ""),
                "policy_metadata": _debug_redact(getattr(policy, "metadata", {})),
                "active_sides": tuple(active_sides),
                "arms_enabled": bool(args.arms_enabled),
                "hands_enabled": bool(hands_enabled),
                "send_actions": bool(send_actions),
                "effective_action_hz": float(effective_action_hz),
                "effective_policy_hz": float(effective_policy_hz),
                "action_hz_info": dict(action_hz_info),
                "policy_hz_source": str(policy_hz_source),
                "backend_control_hz": float(backend_control_hz),
                "action_queue_policy": str(action_queue_policy),
                "arm_execution": (
                    args._real_arm_execution_config.to_metadata()
                    if getattr(args, "_real_arm_execution_config", None) is not None
                    else None
                ),
                "safety_limits": {
                    "arm_workspace_min": tuple(float(value) for value in args.arm_workspace_min),
                    "arm_workspace_max": tuple(float(value) for value in args.arm_workspace_max),
                    "max_arm_xyz_step_m": float(args.max_arm_xyz_step_m),
                    "max_arm_rot_step_deg": float(args.max_arm_rot_step_deg),
                    "max_arm_joint_step_rad": float(args.max_arm_joint_step_rad),
                    "max_hand_step": float(args.max_hand_step),
                },
                "backend_limits_by_side": _debug_backend_limits(arm_backends, now=time.time()),
            },
        )

    def log_policy_chunk(
        self,
        *,
        timestamp: float,
        step_index: int,
        policy_name: str,
        policy_output: Any,
        actions: list[NormalizedAction],
        accepted: bool,
        discarded_by_ik: bool,
        queue_policy: str,
        queue_len_before: int,
        queue_len_after: int,
        faults: list[str],
        ik_recovery_statuses: Mapping[str, str] | None = None,
        ik_recovery_diagnostics: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.log_event(
            "policy_chunk",
            {
                "timestamp": float(timestamp),
                "step_index": int(step_index),
                "policy_name": str(policy_name),
                "policy_output_type": type(policy_output).__name__,
                "chunk_size": int(len(actions)),
                "accepted": bool(accepted),
                "discarded_by_ik": bool(discarded_by_ik),
                "queue_policy": str(queue_policy),
                "queue_len_before": int(queue_len_before),
                "queue_len_after": int(queue_len_after),
                "faults": list(faults),
                "ik_recovery_statuses": (
                    {} if ik_recovery_statuses is None else dict(ik_recovery_statuses)
                ),
                "ik_recovery_diagnostics": (
                    {}
                    if ik_recovery_diagnostics is None
                    else {str(side): dict(value) for side, value in ik_recovery_diagnostics.items()}
                ),
                "actions": [
                    _debug_action_summary(action, self.active_sides)
                    for action in actions
                ],
            },
        )

    def log_policy_error(
        self,
        *,
        timestamp: float,
        step_index: int,
        policy_name: str,
        exc: BaseException,
        queue_len_before: int,
    ) -> None:
        if not self.enabled:
            return
        self.log_event(
            "policy_error",
            {
                "timestamp": float(timestamp),
                "step_index": int(step_index),
                "policy_name": str(policy_name),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "queue_len_before": int(queue_len_before),
            },
        )

    def log_action_step(
        self,
        *,
        timestamp: float,
        step_index: int,
        policy_name: str,
        raw_action: NormalizedAction,
        canonical_action: Any,
        pre_arm_states: Mapping[str, Mapping[str, Any]],
        post_arm_states: Mapping[str, Mapping[str, Any]] | None,
        arm_backends: Mapping[str, EffectorBackend],
        send_actions: bool,
        queue_len_before_pop: int,
        queue_len_after_pop: int,
        faults: list[str],
        dispatch_error: BaseException | None,
        execution: _PreparedActionExecution | None = None,
    ) -> None:
        if not self.enabled:
            return
        raw_summary = _debug_action_summary(raw_action, self.active_sides)
        filtered_summary = _debug_action_summary(getattr(canonical_action, "action", None), self.active_sides)
        command_summary = _debug_command_summary(getattr(canonical_action, "command", None), self.active_sides)
        raw_pose = {} if raw_summary is None else raw_summary.get("arm_pose_by_side", {})
        filtered_pose = {} if filtered_summary is None else filtered_summary.get("arm_pose_by_side", {})
        command_pose = {} if command_summary is None else command_summary.get("arm_pose_by_side", {})
        pre_tcp_pose = _debug_state_pose_by_side(pre_arm_states)
        post_tcp_pose = _debug_state_pose_by_side(post_arm_states)
        command = getattr(canonical_action, "command", None)
        command_count = len(getattr(command, "commands", {}) or {}) if command is not None else 0
        dispatch_attempted = (
            bool(execution.dispatch_attempted)
            if execution is not None
            else bool(command_count and send_actions)
        )
        self.log_event(
            "action_step",
            {
                "timestamp": float(timestamp),
                "step_index": int(step_index),
                "policy_name": str(policy_name),
                "send_actions": bool(send_actions),
                "dispatch_attempted": bool(dispatch_attempted),
                "dispatch_completed": bool(
                    execution.dispatch_completed if execution is not None else dispatch_attempted
                ),
                "dispatch_asynchronous": bool(
                    execution.asynchronous if execution is not None else False
                ),
                "dispatch_mode": (
                    execution.mode
                    if execution is not None
                    else ("send_actions" if send_actions else "dry_run")
                ),
                "position_source_index": (
                    None if execution is None else execution.position_source_index
                ),
                "dispatch_error": (
                    None
                    if dispatch_error is None
                    else {"type": type(dispatch_error).__name__, "message": str(dispatch_error)}
                ),
                "queue_len_before_pop": int(queue_len_before_pop),
                "queue_len_after_pop": int(queue_len_after_pop),
                "faults": list(faults),
                "raw_action": raw_summary,
                "filtered_action": filtered_summary,
                "command": command_summary,
                "canonical_metadata": getattr(canonical_action, "metadata", {}),
                "raw_to_filtered_pose_delta": _debug_pose_delta_by_side(raw_pose, filtered_pose),
                "filtered_to_command_pose_delta": _debug_pose_delta_by_side(filtered_pose, command_pose),
                "command_to_pre_tcp_delta": _debug_pose_delta_by_side(command_pose, pre_tcp_pose),
                "command_to_post_tcp_delta": _debug_pose_delta_by_side(command_pose, post_tcp_pose),
                "pre_arm_state_by_side": _debug_arm_state_summary(pre_arm_states),
                "post_arm_state_by_side": _debug_arm_state_summary(post_arm_states),
                "backend_limits_by_side": _debug_backend_limits(arm_backends, now=time.time()),
            },
        )


class ArmStateReader:
    def __init__(
        self,
        backends: Mapping[str, EffectorBackend],
        *,
        refresh_hz: float,
        compute_tcp_pose: bool = True,
    ):
        self.backends = dict(backends)
        self.refresh_hz = float(refresh_hz)
        self.compute_tcp_pose = bool(compute_tcp_pose)
        self._last_refresh_at = {side: 0.0 for side in self.backends}
        self._history_by_side = {
            side: CausalSnapshotHistory(maxlen=256, name=f"deploy:arm:{side}")
            for side in self.backends
        }

    def snapshot(self, now: float) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        refresh_period = 0.0 if self.refresh_hz <= 0.0 else 1.0 / self.refresh_hz
        for side, backend in self.backends.items():
            refresh_error: str | None = None
            if (
                refresh_period > 0.0
                and backend.is_ready()
                and now - self._last_refresh_at.get(side, 0.0) >= refresh_period
            ):
                try:
                    refresh = getattr(backend, "refresh_state_from_hardware", None)
                    if callable(refresh):
                        refresh()
                except Exception as exc:  # noqa: BLE001
                    refresh_error = f"refresh_failed:{type(exc).__name__}:{exc}"
                self._last_refresh_at[side] = now
            state = dict(backend.get_state())
            if refresh_error is not None:
                state["state_error"] = refresh_error
            state["actual_tcp_pose"] = None
            state["actual_tcp_pose_source"] = None
            state["actual_tcp_pose_timestamp"] = None
            if self.compute_tcp_pose:
                try:
                    hardware_qpos = state.get("hardware_qpos")
                    compute_pose = getattr(backend, "compute_tcp_pose", None)
                    if not callable(compute_pose):
                        raise AttributeError("backend has no compute_tcp_pose")
                    if hardware_qpos is not None:
                        measured_pose = compute_pose(qpos=hardware_qpos)
                        state["current_tcp_pose"] = measured_pose
                        state["tcp_pose_source"] = "hardware_qpos_fk"
                        try:
                            hardware_qpos_vector = np.asarray(
                                hardware_qpos, dtype=float
                            ).reshape(-1)
                            measurement_timestamp = float(
                                state.get("hardware_measurement_timestamp")
                            )
                            measurement_sequence = state.get(
                                "hardware_measurement_sequence"
                            )
                            measurement_valid = state.get(
                                "hardware_measurement_valid"
                            )
                            read_success = state.get("hardware_read_success")
                            sequence_valid = (
                                measurement_sequence is None
                                or int(measurement_sequence) > 0
                            )
                            actual_valid = bool(
                                hardware_qpos_vector.shape == (6,)
                                and np.all(np.isfinite(hardware_qpos_vector))
                                and np.isfinite(measurement_timestamp)
                                and isinstance(measurement_valid, (bool, np.bool_))
                                and bool(measurement_valid)
                                and not (
                                    isinstance(read_success, (bool, np.bool_))
                                    and not bool(read_success)
                                )
                                and sequence_valid
                            )
                        except (TypeError, ValueError, OverflowError):
                            actual_valid = False
                            measurement_timestamp = None
                        if actual_valid:
                            state["actual_tcp_pose"] = np.asarray(
                                measured_pose, dtype=float
                            ).copy()
                            state["actual_tcp_pose_source"] = "hardware_qpos_fk"
                            state["actual_tcp_pose_timestamp"] = float(
                                measurement_timestamp
                            )
                    else:
                        state["current_tcp_pose"] = compute_pose()
                        state["tcp_pose_source"] = "current_qpos_fk"
                except Exception as exc:  # noqa: BLE001
                    state.setdefault("pose_error", f"compute_tcp_pose_failed:{type(exc).__name__}:{exc}")
            source_timestamp = state.get(
                "hardware_measurement_timestamp",
                state.get("hardware_refresh_time"),
            )
            self._history_by_side[side].publish(
                state,
                source_timestamp=source_timestamp,
                available_timestamp=None,
            )
            selected = self._history_by_side[side].latest_at_or_before(float(now))
            if selected is None:
                states[side] = {
                    "causal_snapshot_missing": True,
                    "causal_snapshot_cutoff_timestamp": float(now),
                    "causal_snapshot_contract": (
                        "source_timestamp<=T0_and_available_timestamp<=T0"
                    ),
                }
            else:
                selected_state, selection = selected
                selected_state.update(
                    {
                        "causal_snapshot_missing": False,
                        "causal_snapshot_sequence": selection.sequence,
                        "causal_snapshot_source_timestamp": selection.source_timestamp,
                        "causal_snapshot_available_timestamp": selection.available_timestamp,
                        "causal_snapshot_age_sec": selection.age_sec,
                        "causal_snapshot_cutoff_timestamp": float(now),
                        "causal_snapshot_contract": (
                            "source_timestamp<=T0_and_available_timestamp<=T0"
                        ),
                    }
                )
                states[side] = selected_state
        return states


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _unit_interval_float(text: str) -> float:
    value = float(text)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("value must be in (0, 1]")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _parse_dmtac_tactile_fields(text: str) -> tuple[str, ...]:
    try:
        return normalize_dmtac_tactile_fields(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_active_sides(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        parts = [
            part.strip().lower()
            for item in value
            for part in str(item).replace(",", " ").split()
            if part.strip()
        ]
    else:
        parts = [part.strip().lower() for part in str(value).replace(",", " ").split() if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("active-sides must be one of: left, right, both")
    if parts == ["both"]:
        return ("left", "right")
    invalid = sorted(set(parts) - set(ALL_SIDES))
    if invalid:
        raise argparse.ArgumentTypeError("active-sides must be one of: left, right, both")
    return tuple(side for side in ALL_SIDES if side in set(parts))


def _parse_end_effector_sides(
    value: str | list[str] | tuple[str, ...],
    *,
    active_sides: tuple[str, ...],
    hands_enabled: bool,
) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        parts = [
            part.strip().lower()
            for item in value
            for part in str(item).replace(",", " ").split()
            if part.strip()
        ]
    else:
        parts = [part.strip().lower() for part in str(value).replace(",", " ").split() if part.strip()]
    if not parts or parts == ["active"]:
        return tuple(active_sides) if hands_enabled else ()
    if parts == ["none"]:
        return ()
    if parts == ["both"]:
        parts = ["left", "right"]
    invalid = set(parts) - set(ALL_SIDES)
    if invalid:
        raise argparse.ArgumentTypeError(
            "end-effector-sides must be active, none, left, right, or both"
        )
    if not hands_enabled:
        raise ValueError("--end-effector-sides requires hands to be enabled.")
    requested = set(parts)
    unavailable = requested - set(active_sides)
    if unavailable:
        raise ValueError(
            "--end-effector-sides must be a subset of --active-sides, got "
            + ",".join(sorted(unavailable))
        )
    return tuple(side for side in ALL_SIDES if side in requested)


def _parse_float_tuple(text: str, *, length: int, name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != int(length):
        raise argparse.ArgumentTypeError(f"{name} must contain {length} comma-separated floats")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid {name}: {text}") from exc
    return values


def _non_negative_float(text: str) -> float:
    value = float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _parse_vector3(value: str | list[str] | tuple[object, ...]) -> tuple[float, float, float]:
    if isinstance(value, (list, tuple)):
        parts = [
            part.strip()
            for item in value
            for part in str(item).replace(",", " ").split()
            if part.strip()
        ]
        text = " ".join(parts)
    else:
        text = str(value)
        parts = [part.strip() for part in text.replace(",", " ").split() if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("vector3 must contain 3 floats")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid vector3: {text}") from exc
    return (values[0], values[1], values[2])


def _parse_qpos6(text: str) -> tuple[float, ...]:
    return _parse_float_tuple(text, length=6, name="qpos6")


def _parse_arm_max_angvel(text: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("arm max angvel must contain 1 or 6 floats")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid arm max angvel: {text}") from exc
    if len(values) == 1:
        if values[0] <= 0.0:
            raise argparse.ArgumentTypeError("arm max angvel must be positive")
        return (float(values[0]),) * 6
    if len(values) != 6 or any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("arm max angvel must contain 1 or 6 positive floats")
    return tuple(float(value) for value in values)


def _hand_backend_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "hand_backend", HAND_BACKEND_INSPIRE) or HAND_BACKEND_INSPIRE)
    if mode not in HAND_BACKEND_CHOICES:
        raise ValueError(f"Unsupported hand backend {mode!r}; expected one of {HAND_BACKEND_CHOICES}.")
    return mode


def _hand_backend_mode_for_side(args: argparse.Namespace, side: str) -> str:
    side = str(side).strip().lower()
    if side not in ALL_SIDES:
        raise ValueError(f"Unsupported hand backend side {side!r}.")
    override = getattr(args, f"{side}_hand_backend", None)
    mode = _hand_backend_mode(args) if override in (None, "") else str(override)
    if mode not in HAND_BACKEND_CHOICES:
        raise ValueError(
            f"Unsupported {side} hand backend {mode!r}; expected one of {HAND_BACKEND_CHOICES}."
        )
    return mode


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
        merged = dict(base_gripper) if isinstance(base_gripper, Mapping) else {}
        merged.update(dict(gripper) if isinstance(gripper, Mapping) else dict(selected))
        return merged
    if isinstance(base_gripper, Mapping):
        return dict(base_gripper)
    if "gripper" in payload:
        raise ValueError(f"Profile {path} field 'gripper' must be an object.")
    return dict(payload)


def _profile_value(profile: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in profile:
            return profile[key]
    return None


def _resolved_gripper_policy(args: argparse.Namespace) -> dict[str, Any]:
    cached = getattr(args, "_resolved_gripper_policy", None)
    if isinstance(cached, dict):
        return dict(cached)

    profile = _load_gripper_task_profile(args)
    policy: dict[str, Any] = dict(GRIPPER_POLICY_DEFAULTS)
    profile_keys = {
        "output_dim": ("output_dim", "gripper_slot"),
        "speed": ("speed",),
        "torque": ("torque", "force"),
        "max_action": ("max_action", "max_close"),
        "send_threshold": ("send_threshold",),
        "command_hz": ("command_hz", "max_command_hz"),
        "continuous_grasp": ("continuous_grasp", "continuous"),
        "telemetry_hz": ("telemetry_hz",),
        "telemetry_command_grace_s": ("telemetry_command_grace_s",),
        "tactile_fields": ("tactile_fields", "dmtac_tactile_fields"),
        "tactile_compute_profile": ("tactile_compute_profile", "dmtac_tactile_compute_profile"),
    }
    for key, aliases in profile_keys.items():
        value = _profile_value(profile, *aliases)
        if value is not None:
            policy[key] = value

    cli_overrides = {
        "output_dim": getattr(args, "gripper_output_dim", None),
        "speed": getattr(args, "gripper_speed", None),
        "torque": getattr(args, "gripper_torque", None),
        "max_action": getattr(args, "gripper_max_action", None),
        "send_threshold": getattr(args, "gripper_send_threshold", None),
        "command_hz": getattr(args, "gripper_command_hz", None),
        "continuous_grasp": getattr(args, "gripper_continuous_grasp", None),
        "telemetry_hz": getattr(args, "gripper_telemetry_hz", None),
        "telemetry_command_grace_s": getattr(
            args, "gripper_telemetry_command_grace_s", None
        ),
    }
    for key, value in cli_overrides.items():
        if value is not None:
            policy[key] = value

    policy["output_dim"] = int(policy["output_dim"])
    policy["speed"] = float(policy["speed"])
    policy["torque"] = float(policy["torque"])
    policy["max_action"] = float(policy["max_action"])
    policy["send_threshold"] = float(policy["send_threshold"])
    policy["command_hz"] = float(policy["command_hz"])
    policy["continuous_grasp"] = bool(policy["continuous_grasp"])
    policy["telemetry_hz"] = float(policy["telemetry_hz"])
    policy["telemetry_command_grace_s"] = float(
        policy["telemetry_command_grace_s"]
    )
    if policy.get("tactile_fields") is not None:
        policy["tactile_fields"] = normalize_dmtac_tactile_fields(policy["tactile_fields"])
    if policy.get("tactile_compute_profile") is not None:
        policy["tactile_compute_profile"] = str(policy["tactile_compute_profile"])
    if int(policy["output_dim"]) != 0:
        raise ValueError("dmtac-gripper output_dim/gripper_slot must be 0.")
    if float(policy["send_threshold"]) < 0.0:
        raise ValueError("gripper send_threshold must be >= 0.")
    if float(policy["command_hz"]) <= 0.0:
        raise ValueError("gripper command_hz must be > 0.")
    if float(policy["telemetry_hz"]) <= 0.0:
        raise ValueError("gripper telemetry_hz must be > 0.")
    if float(policy["telemetry_command_grace_s"]) < 0.0:
        raise ValueError("gripper telemetry_command_grace_s must be >= 0.")
    setattr(args, "_resolved_gripper_policy", dict(policy))
    return dict(policy)


def _dmtac_tactile_enabled(args: argparse.Namespace) -> bool:
    explicit = getattr(args, "dmtac_tactile", None)
    if explicit is not None:
        return bool(explicit)
    return True


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
    config.gripper_telemetry_command_grace_s = float(
        policy.get(
            "telemetry_command_grace_s",
            DMTAC_GRIPPER_TELEMETRY_COMMAND_GRACE_S,
        )
    )
    tactile_timeout_ms = getattr(args, "dmtac_tactile_timeout_ms", None)
    if tactile_timeout_ms is not None:
        config.tactile_timeout_ms = int(tactile_timeout_ms)
    tactile_wait_for_new = getattr(args, "dmtac_tactile_wait_for_new", None)
    if tactile_wait_for_new is not None:
        config.tactile_wait_for_new = bool(tactile_wait_for_new)
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
    if bool(getattr(args, "dmtac_tactile_sequential_read", False)):
        config.tactile_parallel_read = False
    if not _dmtac_tactile_enabled(args):
        config.connect_tactile = False
        config.tactile_ids = ()
    if bool(getattr(args, "rtc_real_shadow", False)):
        # Live raw-RTC shadow may read gripper/tactile observations, but its
        # startup must not clear faults, write speed/torque, open the gripper,
        # reset tactile calibration, or accept a command through a bypass API.
        config.observation_only = True
        config.send_commands = False
        config.clear_fault_on_start = False
        config.open_on_start = False
        config.tactile_reset_on_start = False
        config.tactile_warmup_frames = 0
    config.enable_telemetry = bool(config.connect_hardware or config.connect_tactile)
    return config


def _dmtac_backend_configs(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> dict[str, DMTacGripperBackendConfig]:
    cached = getattr(args, "_dmtac_backend_configs", None)
    if isinstance(cached, dict):
        return dict(cached)
    dmtac_sides = tuple(
        side
        for side in active_sides
        if _hand_backend_mode_for_side(args, side) == HAND_BACKEND_DMTAC_GRIPPER
    )
    if not dmtac_sides:
        setattr(args, "_dmtac_backend_configs", {})
        return {}

    policy = _resolved_gripper_policy(args)
    configs: dict[str, DMTacGripperBackendConfig] = {}
    if bool(getattr(args, "dmtac_default_config", True)):
        try:
            assignment = load_dmtac_hardware_assignment(
                str(getattr(args, "dmtac_config", "latest") or "latest"),
                validate_current_usb=bool(getattr(args, "dmtac_validate_current_usb", True)),
                allow_missing=bool(getattr(args, "dmtac_allow_missing", False)),
                assume_yes_for_extra=bool(getattr(args, "dmtac_assume_yes_for_extra", True)),
            )
            configs.update(
                dmtac_backend_configs_from_assignment(
                    assignment,
                    active_sides=dmtac_sides,
                    tactile_backend=str(getattr(args, "dmtac_tactile_backend", "cpu") or "cpu"),
                    speed=float(policy["speed"]),
                    torque=float(policy["torque"]),
                    max_action=float(policy["max_action"]),
                )
            )
        except Exception as exc:  # noqa: BLE001
            if bool(getattr(args, "dmtac_config_required", False)):
                raise RuntimeError(
                    f"Failed to load DMTAC hardware assignment {getattr(args, 'dmtac_config', 'latest')!r}: {exc}"
                ) from exc
            print(f"[DMTAC] WARNING: could not load DMTAC hardware assignment: {exc}", flush=True)

    for side in dmtac_sides:
        manual = manual_dmtac_backend_config_for_side(args, side, policy)
        if manual is not None:
            configs[side] = manual

    resolved = {
        side: _apply_gripper_policy_to_dmtac_config(config, args=args, policy=policy)
        for side, config in configs.items()
        if side in active_sides and (config.connect_hardware or config.connect_tactile)
    }
    if bool(getattr(args, "dmtac_config_required", False)):
        require_dmtac_sides_resolved(
            dmtac_sides,
            resolved,
            config=str(getattr(args, "dmtac_config", "latest") or "latest"),
        )
    setattr(args, "_dmtac_backend_configs", dict(resolved))
    setattr(args, "dmtac_config_loaded_sides", sorted(resolved))
    return dict(resolved)


def add_args(parser: argparse.ArgumentParser) -> None:
    policy_qpos_defaults = DeployPolicyQposInitSettings()
    parser.add_argument(
        "--policy",
        "--policy-spec",
        dest="policy_spec",
        type=str,
        default="hold",
        help="Policy loader spec: hold, package.module:Policy, or /path/policy.py:Policy.",
    )
    parser.add_argument("--policy-kwargs-json", type=str, default="{}")
    parser.add_argument(
        "--rtc-profile",
        choices=(
            "off",
            RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
            RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
        ),
        default="off",
        help=(
            "Experimental default-off RTC frontend profile. RTC-3c remains the builtin-hold "
            "virtual shadow; raw_rtc_final_chunk enables an explicit model adapter with "
            "the existing final-chunk model/server contract. Both force 30 Hz action/decision, "
            "15 Hz observation requests, disable position chunk lookahead, and forbid motion output."
        ),
    )
    parser.add_argument(
        "--rtc-real-shadow",
        action="store_true",
        help=(
            "Opt into the explicit live observation/transport no-action shadow. "
            "Requires the raw final-chunk RTC profile; hardware backends are "
            "configured for observation-only startup (no command, force, tool-frame, "
            "singularity, or background-stream writes). Default is off."
        ),
    )
    parser.add_argument(
        "--rtc-request-validity-steps",
        type=_positive_int,
        default=10,
        help=(
            "RTC request/plan validity in action-keyframe steps. The default 10 "
            "preserves the existing fail-closed profile; explicit model deployments "
            "may bind a larger tested window without changing model/server payloads."
        ),
    )
    parser.add_argument(
        "--rtc-worker-stop-timeout-sec",
        type=_non_negative_float,
        default=0.0,
        help=(
            "Bounded wait for an in-flight RTC worker during frontend close. "
            "0 preserves the existing non-waiting close behavior."
        ),
    )
    parser.add_argument(
        "--policy-artifact-json",
        type=Path,
        default=None,
        help=(
            "Validated dexteleop_policy_artifact_v1 sidecar. Enables strict v2.2 "
            "layout/dimension binding before backend construction."
        ),
    )
    parser.add_argument(
        "--policy-action-pose-mode",
        choices=("absolute", "relative"),
        default="absolute",
        help=(
            "Pose representation emitted by the policy. Relative actions are converted using "
            "the current measured TCP before canonical safety/IK; default is absolute."
        ),
    )
    parser.add_argument(
        "--policy-observation-firewall-mode",
        choices=("strict", "dual-validate"),
        default="strict",
        help=(
            "Model-input firewall mode. strict (default) makes the typed actor view the "
            "only prediction source; dual-validate is a dry-run compatibility check and "
            "is forbidden with --send-actions."
        ),
    )
    parser.add_argument(
        "--action-arbiter-mode",
        choices=ARBITER_SHADOW_MODES_M3,
        default="off",
        help=(
            "M3 passive ActionArbiter mode. off preserves the legacy path; shadow "
            "evaluates the same popped PolicyActionEmission without dispatch authority "
            "and is currently restricted to dry-run/replay."
        ),
    )
    parser.add_argument(
        "--runtime-pose-frame-contract",
        type=load_pose_frame_contract,
        default=None,
        metavar="JSON",
        help=(
            "Current robot runtime raw_absolute->canonical-desk pose contract. "
            "Native-v2.3-trained pose policies default to the calibrated dual-RM65 "
            "URDF workspace-near-boundary profile; use this option for another station. Policies "
            "declared as v2.2 raw identity compatibility may only use the matching "
            "same-source identity binding; omitting it selects that audited identity "
            "profile."
        ),
    )
    parser.add_argument(
        "--debug-replay-output-pose-mode",
        choices=("absolute", "relative"),
        default="absolute",
        help=(
            "Representation emitted by the v2.2 debug-replay bridge. Relative mode "
            "uses the measured TCP at each replay chunk start and must be paired with "
            "--policy-action-pose-mode relative."
        ),
    )
    parser.add_argument("--active-sides", nargs="+", default=("left", "right"))
    parser.add_argument(
        "--end-effector-sides",
        type=str,
        default="active",
        help=(
            "Configured hand/gripper sides: active (default), none, left, right, or both. "
            "This is independent from the two fixed arm-pose slots."
        ),
    )
    parser.add_argument(
        "--hz",
        type=_positive_float,
        default=DEFAULT_TELEOP_SAMPLE_HZ,
        help="Fallback action rate when --action-hz is 0 and no dataset/policy rate is inferred.",
    )
    parser.add_argument(
        "--arm-execution-profile",
        choices=ARM_EXECUTION_PROFILE_CHOICES,
        default=None,
        help=(
            "Canonical real-arm profile shared with the existing teleop runtime: "
            "position deploy defaults to position_15_60_120hz. position and "
            "position_45_150hz remain explicit compatibility profiles; per-stage "
            "rate flags remain supported as overrides."
        ),
    )
    parser.add_argument(
        "--arm-control-mode",
        choices=RM65_ARM_CONTROL_MODES,
        default=RM65_ARM_CONTROL_MODE_POSITION,
        help=(
            "RM65 deployment execution mode. position uses Movej_CANFD passthrough by default; "
            "impedance uses the force-admittance backend at 40Hz without CANFD follow by default."
        ),
    )
    parser.add_argument(
        "--position-control-profile",
        choices=RM65_POSITION_CONTROL_PROFILE_CHOICES,
        default=RM65_BIMANUAL_POSITION_V1_NAME,
        help=(
            "Versioned position action-execution profile. Policy/dataset main actions "
            "are already source-shaped, so the formal deploy path does not repeat "
            "collection-side inactive, tracker, or post-IK qref smoothing."
        ),
    )
    parser.add_argument(
        "--arm-control-hz",
        type=_positive_float,
        default=None,
        help=(
            "RM65 backend control/streaming rate after each action update. Position deploy "
            "defaults to 120Hz; explicit compatibility profile 'position' uses 150Hz."
        ),
    )
    parser.add_argument(
        "--position-reference-hz",
        type=_positive_float,
        default=DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
        help=(
            "Position-mode pose-reference/IK rate between the policy action rate and "
            "the RM65 command stream. Position deploy defaults to 60Hz; impedance mode ignores it."
        ),
    )
    parser.add_argument(
        "--position-chunk-lookahead",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the already-predicted next action inside one policy chunk for "
            "zero-period causal 15->reference interpolation (default: enabled). "
            "Disable to restore the compatibility one-action-period delayed path."
        ),
    )
    parser.add_argument(
        "--position-collision-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run coupled dual-arm collision-aware IK on every position reference. "
            "This is mandatory with --send-actions; --no-position-collision-check "
            "is accepted only for non-motion legacy dry-run inspection."
        ),
    )
    parser.add_argument(
        "--position-collision-scene-profile",
        type=str,
        default=DEFAULT_COLLISION_SCENE,
        help="Shared RM65 scene profile used by high-rate coupled collision IK.",
    )
    parser.add_argument(
        "--position-collision-min-distance-m",
        type=_non_negative_float,
        default=DEFAULT_DUAL_ARM_COLLISION_MIN_DISTANCE_M,
        help="Minimum configured collision distance for each accepted high-rate qpos endpoint.",
    )
    parser.add_argument(
        "--position-collision-distance-epsilon-m",
        type=_non_negative_float,
        default=DEFAULT_DUAL_ARM_COLLISION_DISTANCE_EPSILON_M,
        help="Numeric tolerance used by the high-rate endpoint collision check.",
    )
    parser.add_argument(
        "--position-collision-debug-profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable collision-solver timing profiling. Full q/pose/distance diagnostics "
            "are written by the existing --debug logger independently."
        ),
    )
    parser.add_argument(
        "--position-ik-solve-max-iterations",
        type=_positive_int,
        default=4,
        help=(
            "Position-only coupled IK iteration upper bound (formal default 4). "
            "The impedance path does not consume this option."
        ),
    )
    parser.add_argument(
        "--position-ik-solve-min-iterations",
        type=_positive_int,
        default=2,
        help=(
            "Position-only coupled IK hard minimum iterations (default 2)."
        ),
    )
    parser.add_argument(
        "--position-ik-damping",
        type=_positive_float,
        default=0.03,
        help="Position-only Pink QP damping (default 0.03).",
    )
    parser.add_argument(
        "--position-ik-orientation-weight",
        type=_non_negative_float,
        default=0.35,
        help="Position-only IK orientation objective weight; position weight is 1.0.",
    )
    parser.add_argument(
        "--position-ik-adaptive-iterations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After the minimum iterations, continue only while the shared "
            "planner's frame budget has room."
        ),
    )
    parser.add_argument(
        "--position-ik-iteration-reserve-ms",
        type=_non_negative_float,
        default=5.0,
        help="Time reserved after optional IK iterations for limits/collision selection.",
    )
    parser.add_argument(
        "--arm-canfd-follow",
        dest="arm_canfd_follow",
        action="store_true",
        help="Send RM65 CANFD commands with SDK follow=True. Defaults on only for position mode.",
    )
    parser.add_argument(
        "--no-arm-canfd-follow",
        dest="arm_canfd_follow",
        action="store_false",
        help="Send RM65 CANFD commands with SDK follow=False.",
    )
    parser.set_defaults(arm_canfd_follow=None)
    parser.add_argument(
        "--arm-canfd-trajectory-mode",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="RM65 Movej_CANFD trajectory mode for position deployment.",
    )
    parser.add_argument(
        "--arm-canfd-radio",
        type=int,
        choices=tuple(range(0, 101)),
        default=0,
        help="RM65 Movej_CANFD radio/smoothing value for position deployment.",
    )
    parser.add_argument(
        "--force-tare-sec",
        type=_positive_float,
        default=3.0,
        help="Tare duration before enabling impedance force control in deployment.",
    )
    parser.add_argument(
        "--strict-tool-profile",
        dest="strict_tool_profile",
        action="store_true",
        default=True,
        help="Require confirmed TCP, payload, and impedance profile for impedance deployment.",
    )
    parser.add_argument(
        "--no-strict-tool-profile",
        dest="strict_tool_profile",
        action="store_false",
        help="Allow incomplete RM65 tool profiles for lab debugging.",
    )
    parser.add_argument(
        "--action-hz",
        type=_non_negative_float,
        default=0.0,
        help=(
            "Action send/control rate. 0 keeps backward-compatible --hz unless a replay/reference "
            "dataset is provided, in which case the real dataset rate is inferred."
        ),
    )
    parser.add_argument("--action-hz-dataset", type=Path, default=None)
    parser.add_argument("--action-hz-parquet", type=Path, default=None)
    parser.add_argument("--action-hz-episode", type=_non_negative_int, default=None)
    parser.add_argument("--action-hz-start-index", type=_non_negative_int, default=None)
    parser.add_argument("--action-hz-end-index", type=int, default=None)
    parser.add_argument(
        "--policy-hz",
        type=_positive_float,
        default=None,
        help=(
            "Policy inference rate. Omit to match the effective action/control rate."
        ),
    )
    parser.add_argument("--min-action-queue", type=_non_negative_int, default=0)
    parser.add_argument(
        "--action-chunk-execution-steps",
        type=_positive_int,
        default=None,
        metavar="X",
        help=(
            "Execute only the first X actions from every received policy chunk, "
            "then request/consume the next chunk. Omit to keep the complete chunk."
        ),
    )
    parser.add_argument(
        "--action-queue-policy",
        choices=["auto", "replace", "append"],
        default="auto",
        help=(
            "How newly predicted chunks update the pending action queue. "
            "auto preserves replace for normal policies and uses append for debug replay."
        ),
    )
    parser.add_argument(
        "--append-action-chunks",
        action="store_true",
        help="Compatibility alias for --action-queue-policy append.",
    )
    parser.add_argument("--max-steps", type=_non_negative_int, default=0)
    parser.add_argument("--status-sec", type=_positive_float, default=1.0)
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Write deployment action/pose debug JSONL under "
            "teleop_records/deployment_debug by default."
        ),
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Directory for --debug JSONL logs. Relative paths resolve under the DexTeleop repo.",
    )
    parser.add_argument(
        "--rtc-debug-level",
        choices=RTC_DEBUG_LEVELS,
        default=None,
        help=(
            "RTC JSONL detail when --debug and an RTC profile are enabled. "
            "Default: summary. trace records every RTC control tick."
        ),
    )
    parser.add_argument(
        "--rtc-debug-queue-capacity",
        type=_positive_int,
        default=2048,
        help=(
            "Bounded non-blocking RTC debug event queue capacity. "
            "Full queues drop diagnostic events, never control work."
        ),
    )
    parser.add_argument(
        "--ik-recovery-stable-frames",
        type=_positive_int,
        default=3,
        help=(
            "After IK smoothing/recovery clears, wait this many stable control frames, "
            "then require a fresh policy inference before accepting action chunks."
        ),
    )
    parser.add_argument("--startup-warmup-sec", type=float, default=1.0)
    parser.add_argument("--debug-replay-dataset", type=Path, default=None)
    parser.add_argument("--debug-replay-parquet", type=Path, default=None)
    parser.add_argument("--debug-replay-episode", type=_non_negative_int, default=0)
    parser.add_argument("--debug-replay-action-column", type=str, default="action")
    parser.add_argument("--debug-replay-start-index", type=_non_negative_int, default=0)
    parser.add_argument("--debug-replay-end-index", type=int, default=None)
    parser.add_argument("--debug-replay-chunk-size", type=_non_negative_int, default=32)
    parser.add_argument("--debug-replay-loop", action="store_true")
    parser.add_argument(
        "--init-time-s",
        type=float,
        default=5.0,
        help="Seconds used to smoothly move to the selected initial target before execution.",
    )
    parser.add_argument(
        "--init-arm-position-tolerance-m",
        type=_positive_float,
        default=0.03,
        help="Startup init arm position tolerance in meters before policy execution is released.",
    )
    parser.add_argument(
        "--init-arm-rotation-tolerance-deg",
        type=_positive_float,
        default=20.0,
        help="Startup init arm orientation tolerance in degrees before policy execution is released.",
    )
    parser.add_argument(
        "--init-source",
        choices=["auto", "current", "dataset", "replay", "none"],
        default="auto",
        help=(
            "Initial target source. auto uses replay first frame in replay mode, "
            "an explicit init dataset when provided, otherwise current robot state."
        ),
    )
    parser.add_argument("--init-dataset", type=Path, default=None)
    parser.add_argument("--init-parquet", type=Path, default=None)
    parser.add_argument("--init-episode", type=_non_negative_int, default=0)
    parser.add_argument("--init-frame", type=_non_negative_int, default=0)
    parser.add_argument("--init-action-column", type=str, default="action")
    parser.add_argument("--deploy-fixed-init-dataset", type=Path, default=None)
    parser.add_argument("--deploy-fixed-init-parquet", type=Path, default=None)
    parser.add_argument("--deploy-fixed-init-episode", type=str, default="0:0:end")
    parser.add_argument("--deploy-fixed-init-action-column", type=str, default="action")
    parser.add_argument(
        "--deploy-fixed-init-arm-max-angvel",
        type=_parse_arm_max_angvel,
        default=(4.5, 4.5, 4.5, 6.0, 6.0, 8.0),
    )
    parser.add_argument("--deploy-fixed-init-qpos-tolerance", type=_positive_float, default=0.1)
    parser.add_argument(
        "--policy-qpos-init-dataset",
        type=Path,
        default=None,
        help=(
            "Dataset root used to move to a recorded qpos frame before policy inference. "
            "This uses the same qpos target selection/planning path as replay init."
        ),
    )
    parser.add_argument("--policy-qpos-init-parquet", type=Path, default=None)
    parser.add_argument("--policy-qpos-init-episode", type=_non_negative_int, default=0)
    parser.add_argument("--policy-qpos-init-frame", type=_non_negative_int, default=0)
    parser.add_argument("--policy-qpos-init-action-column", type=str, default="action")
    parser.add_argument(
        "--policy-qpos-init-source",
        choices=QPOS_INIT_SOURCES,
        default="auto",
        help="Recorded qpos source for dataset-frame initialization.",
    )
    parser.add_argument(
        "--policy-qpos-init-timeout-s",
        type=_positive_float,
        default=policy_qpos_defaults.move_to_start_timeout_s,
    )
    parser.add_argument(
        "--policy-qpos-init-dt-sec",
        type=_positive_float,
        default=policy_qpos_defaults.qpos_init_dt_sec,
    )
    parser.add_argument(
        "--policy-qpos-init-joint-velocity-profile",
        type=str,
        default=policy_qpos_defaults.qpos_init_joint_velocity_profile,
    )
    parser.add_argument(
        "--policy-qpos-init-arm-max-angvel",
        "--qpos-init-arm-max-angvel",
        dest="policy_qpos_init_arm_max_angvel",
        type=_parse_arm_max_angvel,
        default=None,
        help=(
            "Optional first-frame QPOS init joint velocity limit in rad/s. "
            "Accepts one value or six comma-separated values; overrides "
            "--policy-qpos-init-joint-velocity-profile for qpos init planning."
        ),
    )
    parser.add_argument(
        "--policy-qpos-init-max-steps-per-side",
        type=_positive_int,
        default=policy_qpos_defaults.qpos_init_max_steps_per_side,
    )
    parser.add_argument(
        "--policy-qpos-init-collision-check",
        dest="policy_qpos_init_collision_check",
        action="store_true",
        default=policy_qpos_defaults.qpos_init_collision_check,
    )
    parser.add_argument(
        "--policy-qpos-init-no-collision-check",
        dest="policy_qpos_init_collision_check",
        action="store_false",
    )
    parser.add_argument(
        "--policy-qpos-init-collision-scene-profile",
        type=str,
        default=policy_qpos_defaults.qpos_init_collision_scene_profile,
    )
    parser.add_argument(
        "--policy-qpos-init-branch-search",
        dest="policy_qpos_init_branch_search",
        action="store_true",
        default=policy_qpos_defaults.qpos_init_branch_search,
    )
    parser.add_argument(
        "--policy-qpos-init-no-branch-search",
        dest="policy_qpos_init_branch_search",
        action="store_false",
    )
    parser.add_argument(
        "--policy-qpos-init-branch-search-max-branches",
        type=_non_negative_int,
        default=policy_qpos_defaults.qpos_init_branch_search_max_branches,
    )
    parser.add_argument(
        "--policy-qpos-init-branch-neutral-qpos-path",
        type=Path,
        default=policy_qpos_defaults.qpos_init_branch_neutral_qpos_path,
    )
    parser.add_argument(
        "--policy-qpos-init-qpos-tolerance",
        type=_positive_float,
        default=policy_qpos_defaults.qpos_reached_tolerance_rad,
    )
    parser.add_argument(
        "--start-init_qpos",
        "--start-init-qpos",
        dest="start_init_qpos",
        action="store_true",
        help=(
            "Explicit safety override: allow real arm policy execution to start from the "
            "current qpos without a dataset-frame policy qpos init."
        ),
    )
    parser.add_argument("--require-ready-observation", action="store_true")
    parser.add_argument("--pause-on-policy-error", dest="pause_on_policy_error", action="store_true", default=True)
    parser.add_argument("--no-pause-on-policy-error", dest="pause_on_policy_error", action="store_false")
    parser.add_argument("--send-actions", action="store_true")
    parser.add_argument(
        "--allow-missing-v22-end-effectors",
        action="store_true",
        help=(
            "Explicit arm-only v2.2 hardware test mode: allow dataset layout end-effector "
            "slots to be omitted from runtime when --end-effector-sides none is set. "
            "No hand command is dispatched. Default remains fail-closed."
        ),
    )
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument(
        "--wait-for-ready-confirm",
        dest="wait_for_ready_confirm",
        action="store_true",
        default=None,
        help="After startup init reaches ready, wait for A/a before policy inference and action execution.",
    )
    parser.add_argument(
        "--no-wait-for-ready-confirm",
        dest="wait_for_ready_confirm",
        action="store_false",
        help="Start policy immediately after startup init completes.",
    )
    parser.add_argument("--allow-arm-connect-failure", action="store_true")
    parser.add_argument("--allow-hand-connect-failure", action="store_true")

    parser.add_argument("--enable-arms", dest="arms_enabled", action="store_true", default=True)
    parser.add_argument("--no-arms", dest="arms_enabled", action="store_false")
    parser.add_argument("--left-arm-ip", type=str, default="")
    parser.add_argument("--right-arm-ip", type=str, default="")
    parser.add_argument(
        "--left-tool-profile",
        type=str,
        default="",
        help=(
            "Optional explicit left RM65 tool profile name or JSON path. "
            "Use position_endpoint for endpoint-TCP position control without payload/impedance metadata."
        ),
    )
    parser.add_argument(
        "--right-tool-profile",
        type=str,
        default="",
        help=(
            "Optional explicit right RM65 tool profile name or JSON path. "
            "Use position_endpoint for endpoint-TCP position control without payload/impedance metadata."
        ),
    )
    parser.add_argument("--arm-urdf-path", type=str, default="")
    parser.add_argument(
        "--rm65-ik-framework",
        choices=["robot_motion"],
        default="robot_motion",
        help=(
            "RM65 IK implementation for position arm backends. "
            "robot_motion routes shared RM65 IK construction and solving through robot_motion.rm65."
        ),
    )
    parser.add_argument(
        "--arm-max-angvel",
        type=_parse_arm_max_angvel,
        default=DEPLOY_RM65_ARM_MAX_ANGVEL,
        help=(
            "Scalar or 6 comma-separated RM65 joint velocity limits in rad/s. "
            "Default matches run_vive_rm65_real_teleop."
        ),
    )
    parser.add_argument(
        "--position-qref-max-angvel",
        type=_parse_arm_max_angvel,
        default=None,
        help=(
            "Position-only 60 Hz accepted-qref velocity cap in rad/s. "
            "Omit to reuse --arm-max-angvel. This is independent from the "
            "120 Hz qcmd planner envelope."
        ),
    )
    parser.add_argument(
        "--position-command-planner",
        choices=POSITION_COMMAND_PLANNER_CHOICES,
        default=POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
        help="Position-only high-rate qcmd planner.",
    )
    parser.add_argument(
        "--position-velocity-ramp-sec",
        type=_positive_float,
        default=DEFAULT_POSITION_VELOCITY_RAMP_SEC,
    )
    parser.add_argument(
        "--position-target-velocity-filter-tau-sec",
        type=_non_negative_float,
        default=DEFAULT_POSITION_TARGET_VELOCITY_FILTER_TAU_SEC,
        help="Position-only target qd low-pass time constant; default 0.03 s.",
    )
    parser.add_argument(
        "--position-max-acceleration-rad-s2",
        type=_parse_arm_max_angvel,
        default=RM65_POSITION_TRACKING_MAX_ACCELERATION_RAD_S2,
    )
    parser.add_argument(
        "--position-max-command-step-deg",
        type=_positive_float,
        default=RM65_POSITION_TRACKING_MAX_COMMAND_STEP_DEG,
    )
    parser.add_argument(
        "--position-degraded-speed-scale",
        type=_unit_interval_float,
        default=DEFAULT_POSITION_DEGRADED_SPEED_SCALE,
    )
    parser.add_argument(
        "--arm-follow-qpos-alpha",
        type=_unit_interval_float,
        default=None,
        help="Position-only final qref-follow alpha applied by the 120 Hz backend.",
    )
    parser.add_argument(
        "--arm-step-dt-scaling",
        dest="arm_step_dt_scaling",
        action="store_true",
        default=True,
        help="Enable real-arm step scaling based on deploy action dt. Default matches real teleop.",
    )
    parser.add_argument(
        "--disable-arm-step-dt-scaling",
        dest="arm_step_dt_scaling",
        action="store_false",
        help="Disable deploy arm step scaling and use one backend control tick per policy action.",
    )
    parser.add_argument(
        "--arm-step-dt-alpha",
        type=_unit_interval_float,
        default=DEPLOY_ARM_STEP_DT_ALPHA,
        help="EMA factor for RM65 backend measured-send-dt scaling. Default matches real teleop.",
    )
    parser.add_argument(
        "--arm-step-dt-max-scale",
        type=_positive_float,
        default=None,
        help=(
            "Maximum multiplier for arm step limits when action dt is slower than backend control dt. "
            "Omit to use max(3.0, 1.1 * --arm-control-hz / effective_action_hz), matching real teleop."
        ),
    )
    parser.add_argument(
        "--arm-step-dt-stall-reset-sec",
        type=_positive_float,
        default=DEPLOY_ARM_STEP_DT_STALL_RESET_SEC,
        help="Stall threshold for RM65 backend measured-send-dt scaling. Default matches real teleop.",
    )
    parser.add_argument(
        "--singularity-qpos-init-slowdown",
        type=_unit_interval_float,
        default=DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN,
        help=(
            "Scale RM65 joint steps when first-frame QPOS init crosses singularity. "
            "Only applies to replay/deploy QPOS initialization sources."
        ),
    )
    parser.add_argument("--arm-state-refresh-hz", type=float, default=10.0)
    parser.add_argument("--rm65-hardware-read-retries", type=_non_negative_int, default=1)
    parser.add_argument("--rm65-hardware-read-retry-delay-sec", type=float, default=0.02)
    parser.add_argument("--left-initial-qpos", type=_parse_qpos6, default=None)
    parser.add_argument("--right-initial-qpos", type=_parse_qpos6, default=None)
    parser.add_argument(
        "--arm-workspace-min",
        nargs="+",
        default=(-0.60, -0.80, 0.00),
        help="Arm workspace minimum xyz. Accepts either three values or --arm-workspace-min=x,y,z.",
    )
    parser.add_argument(
        "--arm-workspace-max",
        nargs="+",
        default=(0.60, 0.80, 0.90),
        help="Arm workspace maximum xyz. Accepts either three values or --arm-workspace-max=x,y,z.",
    )
    parser.add_argument("--max-arm-xyz-step-m", type=_non_negative_float, default=0.1)
    parser.add_argument("--max-arm-rot-step-deg", type=_non_negative_float, default=15.0)
    parser.add_argument(
        "--max-arm-joint-step-rad",
        type=_non_negative_float,
        default=DEPLOY_MAX_ARM_JOINT_STEP_RAD,
        help=(
            "Per-action joint-vector safety limit in rad. "
            "Default is 6deg, matching real teleop temporary joint-step limit."
        ),
    )

    parser.add_argument("--enable-hands", dest="hands_enabled", action="store_true", default=None)
    parser.add_argument("--no-hands", dest="hands_enabled", action="store_false")
    parser.add_argument(
        "--hand-backend",
        choices=HAND_BACKEND_CHOICES,
        default=HAND_BACKEND_INSPIRE,
        help="Hand backend type. Use dmtac-gripper to deploy with saved DMTAC gripper/tactile assignment.",
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
    parser.add_argument("--left-hand-ip", type=str, default="")
    parser.add_argument("--right-hand-ip", type=str, default="")
    parser.add_argument("--hand-modbus-port", type=int, default=6000)
    parser.add_argument("--hand-send-threshold", type=float, default=0.002)
    parser.add_argument("--hand-telemetry-hz", type=_positive_float, default=20.0)
    parser.add_argument("--hand-tactile-regions", "--tactile-regions", "--tactile-region", dest="hand_tactile_regions", type=str, default="all")
    parser.add_argument(
        "--inspire-hand-max-force",
        "--inspire-tac-max-fingertip-force",
        "--max-force",
        "--max-fingertip-force",
        "--hand-max-force",
        dest="max_fingertip_force",
        type=_non_negative_int,
        default=DEFAULT_INSPIRE_HAND_MAX_FORCE,
        help=(
            "InspireTac startup max_fingertip_force written to all six fingers. "
            f"Default: {DEFAULT_INSPIRE_HAND_MAX_FORCE}."
        ),
    )
    parser.add_argument(
        "--max-hand-step",
        type=float,
        default=0.0,
        help=(
            "Reserved/no-op for future hand q6 rate limiting. "
            "Current deploy clips hand q6 to [0, 1] but does not rate-limit it."
        ),
    )
    add_dmtac_assignment_arguments(parser, assignment_includes_tactile=True)
    parser.add_argument("--dmtac-tactile-backend", type=str, default="cpu")
    parser.add_argument("--dmtac-tactile-timeout-ms", type=_positive_int, default=None)
    parser.add_argument("--dmtac-tactile-wait-for-new", action="store_true", default=None)
    parser.add_argument("--dmtac-tactile-sequential-read", action="store_true")
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
    )
    parser.add_argument("--dmtac-with-tactile", dest="dmtac_tactile", action="store_true", default=None)
    parser.add_argument("--dmtac-no-tactile", dest="dmtac_tactile", action="store_false")
    parser.add_argument(
        "--gripper-task-profile",
        type=str,
        default="",
        help="JSON profile with top-level gripper defaults and/or tasks.<name>.gripper overrides.",
    )
    parser.add_argument("--task-id", type=str, default="", help="Task id used to select gripper task profile entries.")
    parser.add_argument("--task-name", type=str, default="", help="Task name used to select gripper task profile entries.")
    parser.add_argument(
        "--gripper-task",
        type=str,
        default="",
        help="Explicit gripper task key; overrides --task-id/--task-name for --gripper-task-profile.",
    )
    parser.add_argument("--gripper-output-dim", type=int, default=None)
    parser.add_argument("--gripper-speed", type=float, default=None)
    parser.add_argument(
        "--gripper-torque",
        "--gripper-force",
        dest="gripper_torque",
        type=float,
        default=None,
    )
    parser.add_argument("--gripper-max-action", type=float, default=None)
    parser.add_argument("--gripper-send-threshold", type=float, default=None)
    parser.add_argument("--gripper-continuous", dest="gripper_continuous_grasp", action="store_true", default=None)
    parser.add_argument("--gripper-no-continuous", dest="gripper_continuous_grasp", action="store_false")
    parser.add_argument("--gripper-command-hz", type=_positive_float, default=None)
    parser.add_argument("--gripper-telemetry-hz", type=_positive_float, default=None)
    parser.add_argument(
        "--gripper-telemetry-command-grace-s", type=float, default=None
    )

    parser.add_argument("--camera-mode", choices=["none", "single", "multi"], default="none")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG_PATH)
    parser.add_argument("--camera-key", type=str, default="front")
    parser.add_argument("--camera-serial", "--camera-id", dest="camera_serial", type=str, default="")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--camera-keep-last",
        type=int,
        default=DEFAULT_DEPLOYMENT_CAMERA_KEEP_LAST,
        help=(
            "Minimum causal frame history retained by deployment camera sources. "
            "The value also applies when cameras are loaded from --camera-config."
        ),
    )
    parser.add_argument("--camera-color-space", choices=["rgb", "bgr"], default="rgb")
    parser.add_argument("--camera-wait-timeout-ms", type=int, default=1000)
    parser.add_argument("--camera-stale-after-sec", type=float, default=2.0)
    parser.add_argument(
        "--camera-frame-max-age-sec",
        type=_positive_float,
        default=DEFAULT_DEPLOYMENT_CAMERA_FRAME_MAX_AGE_SEC,
        help=(
            "Maximum causal camera source age accepted by the deployment observation "
            f"firewall (default: {DEFAULT_DEPLOYMENT_CAMERA_FRAME_MAX_AGE_SEC:.3f}s)."
        ),
    )
    parser.add_argument("--camera-auto-exposure-priority", type=int, default=None)
    parser.add_argument(
        "--camera-startup-hardware-reset",
        action="store_true",
        help=(
            "Reset each selected RealSense by serial before opening its stream. "
            "Default-off so existing deployment profiles and camera configs are unchanged."
        ),
    )
    parser.add_argument(
        "--camera-start-stagger-sec",
        type=float,
        default=0.0,
        help=(
            "Delay between sequential RealSense starts. Camera-config insertion order "
            "is preserved; the default 0 keeps the existing deploy behavior."
        ),
    )
    parser.add_argument("--require-camera-frame", action="store_true")
    parser.add_argument("--vis_cam", "--vis-cam", dest="vis_cam", action="store_true")
    parser.add_argument("--vis-cam-key", "--vis-camera-key", type=str, default="")
    parser.add_argument("--visualize-tactile", action="store_true")
    parser.add_argument("--tactile-vis-value-scale", type=float, default=0.05)
    parser.add_argument("--tactile-vis-size", type=int, default=640)


def parse_policy_kwargs(text: str) -> dict[str, Any]:
    payload = json.loads(str(text or "{}"))
    if not isinstance(payload, dict):
        raise ValueError("--policy-kwargs-json must decode to a JSON object.")
    return payload


def _hands_enabled(args: argparse.Namespace) -> bool:
    if args.hands_enabled is not None:
        return bool(args.hands_enabled)
    if any(
        _hand_backend_mode_for_side(args, side) == HAND_BACKEND_DMTAC_GRIPPER
        for side in ALL_SIDES
    ):
        return True
    return bool(
        args.left_hand_ip
        or args.right_hand_ip
        or args.visualize_tactile
    )


def _position_deploy_ik_tuning(args: argparse.Namespace):
    """Bind the formal action-to-qref objective without source-side shaping."""

    if (
        str(getattr(args, "position_control_profile_resolved", ""))
        == LEGACY_POSITION_CONTROL_PROFILE_NAME
    ):
        return None
    tuning = load_ik_tuning_config()
    mode_name = RM65_BIMANUAL_POSITION_V1_NAME
    position_weight = RM65_BIMANUAL_POSITION_V1.ik_execution.position_weight
    orientation_weight = float(args.position_ik_orientation_weight)
    objective_modes = dict(tuning.objective_modes)
    objective_modes[mode_name] = IKObjectiveWeights(
        single_arm_position_weight=position_weight,
        single_arm_orientation_weight=orientation_weight,
        dual_arm_position_cost=position_weight,
        dual_arm_orientation_cost=orientation_weight,
    )
    return replace(
        tuning,
        ik_objective_mode=mode_name,
        objective_modes=objective_modes,
    )


def validate_args(
    args: argparse.Namespace,
    *,
    raw_argv: tuple[str, ...] | list[str] = (),
) -> tuple[str, ...]:
    active_sides = _parse_active_sides(args.active_sides)
    args.active_sides = active_sides
    observation_firewall_mode = str(
        getattr(args, "policy_observation_firewall_mode", "strict")
    ).strip().lower()
    rtc_profile_id = str(getattr(args, "rtc_profile", "off")).strip()
    rtc_profile_enabled = rtc_profile_id != "off"
    rtc_real_shadow = bool(getattr(args, "rtc_real_shadow", False))
    requested_rtc_debug_level = getattr(args, "rtc_debug_level", None)
    if requested_rtc_debug_level is None:
        rtc_debug_level = (
            RTC_DEBUG_LEVEL_SUMMARY
            if bool(getattr(args, "debug", False)) and rtc_profile_enabled
            else RTC_DEBUG_LEVEL_OFF
        )
    else:
        rtc_debug_level = normalize_rtc_debug_level(requested_rtc_debug_level)
    if rtc_debug_level != RTC_DEBUG_LEVEL_OFF and not bool(
        getattr(args, "debug", False)
    ):
        raise ValueError("--rtc-debug-level summary/trace requires --debug.")
    if rtc_debug_level != RTC_DEBUG_LEVEL_OFF and not rtc_profile_enabled:
        raise ValueError("--rtc-debug-level summary/trace requires an RTC profile.")
    setattr(args, "_rtc_debug_level", rtc_debug_level)
    if rtc_real_shadow and rtc_profile_id != RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1:
        raise ValueError(
            "--rtc-real-shadow requires the explicit raw final-chunk RTC profile; "
            "it is not available for RTC-3c virtual policy or sync deployment."
        )
    if rtc_profile_enabled:
        if rtc_profile_id not in {
            RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
            RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
        }:
            raise ValueError(f"Unsupported RTC profile: {rtc_profile_id!r}.")
        if bool(args.send_actions):
            raise ValueError("RTC shadow profile forbids --send-actions.")
        if args.debug_replay_dataset or args.debug_replay_parquet:
            raise ValueError("RTC profiles do not modify or consume debug replay.")
        if args.action_hz_dataset or args.action_hz_parquet:
            raise ValueError("RTC profiles do not consume replay/reference rate inputs.")
        if bool(args.append_action_chunks) or str(args.action_queue_policy) == "append":
            raise ValueError("RTC profiles use timed mutable-future merge, not queue append.")
        if rtc_profile_id == RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1:
            if str(args.policy_spec or "hold").strip().lower() not in {
                "",
                "hold",
                "builtin:hold",
            }:
                raise ValueError("RTC-3c is restricted to the builtin hold virtual policy.")
            if args.policy_artifact_json is not None:
                raise ValueError("RTC-3c virtual shadow does not bind a model artifact yet.")
            if parse_policy_kwargs(args.policy_kwargs_json):
                raise ValueError("RTC-3c builtin hold policy accepts no policy kwargs.")
        elif args.policy_artifact_json is None:
            raise ValueError(
                "raw-RTC final-chunk requires an explicit --policy-artifact-json "
                "sidecar; model identity must not be inferred from a model name."
            )
        if (
            rtc_profile_id == RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1
            and not rtc_real_shadow
            and (
                bool(args.left_arm_ip)
                or bool(args.right_arm_ip)
                or bool(args.left_hand_ip)
                or bool(args.right_hand_ip)
                or (
                    bool(getattr(args, "hands_enabled", False))
                    and any(
                        _hand_backend_mode_for_side(args, side)
                        == HAND_BACKEND_DMTAC_GRIPPER
                        for side in ALL_SIDES
                    )
                )
            )
        ):
            raise ValueError(
                "raw-RTC with live hardware endpoints requires explicit "
                "--rtc-real-shadow; otherwise use synthetic/local inputs."
            )
        if float(args.action_hz) not in {
            0.0,
            RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.action_hz,
        }:
            raise ValueError("RTC shadow action rate is fixed at 30 Hz.")
        if args.policy_hz is not None and not math.isclose(
            float(args.policy_hz),
            RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.decision_hz,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("RTC shadow decision rate is fixed at 30 Hz.")
        args.position_chunk_lookahead = False
        args.action_hz = RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.action_hz
        args.policy_hz = RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.decision_hz
    elif (
        int(getattr(args, "rtc_request_validity_steps", 10)) != 10
        or float(getattr(args, "rtc_worker_stop_timeout_sec", 0.0)) != 0.0
    ):
        raise ValueError(
            "RTC validity/worker-stop overrides require an explicit --rtc-profile."
        )
    if rtc_real_shadow:
        # A live observation shadow has no init/qpos authority.  Keeping a
        # dataset-frame init controller alive would either block the model on a
        # target it cannot execute or accidentally reintroduce a dispatch path
        # if a future caller forgets to pass send_actions=False.
        if args.deploy_fixed_init_dataset or args.deploy_fixed_init_parquet:
            raise ValueError(
                "--rtc-real-shadow does not accept deploy-fixed-init dataset inputs."
            )
        if args.policy_qpos_init_dataset or args.policy_qpos_init_parquet:
            raise ValueError(
                "--rtc-real-shadow does not accept policy-qpos-init dataset inputs."
            )
        if bool(args.send_actions):
            raise ValueError("--rtc-real-shadow forbids --send-actions.")
    if bool(args.send_actions) and observation_firewall_mode != "strict":
        raise ValueError(
            "--send-actions requires --policy-observation-firewall-mode strict."
        )
    if (
        bool(args.send_actions)
        and str(getattr(args, "action_arbiter_mode", "off"))
        == ARBITER_SHADOW_MODE_SHADOW
    ):
        raise ValueError(
            "M3 --action-arbiter-mode shadow is dry-run/replay only; "
            "formal real shadow requires the M4B durable trace gate."
        )
    raw_tokens_list = [str(value) for value in raw_argv]
    if not any(
        token in ("--arm-canfd-follow", "--no-arm-canfd-follow")
        for token in raw_tokens_list
    ) and getattr(args, "arm_canfd_follow", None) is False:
        raw_tokens_list.append("--no-arm-canfd-follow")
    if not any(
        token == "--position-ik-solve-min-iterations"
        or token.startswith("--position-ik-solve-min-iterations=")
        for token in raw_tokens_list
    ) and int(args.position_ik_solve_min_iterations) != 2:
        raw_tokens_list.append("--position-ik-solve-min-iterations")
    if not any(
        token == "--position-ik-solve-max-iterations"
        or token.startswith("--position-ik-solve-max-iterations=")
        for token in raw_tokens_list
    ) and int(args.position_ik_solve_max_iterations) != 4:
        raw_tokens_list.append("--position-ik-solve-max-iterations")
    raw_tokens = tuple(raw_tokens_list)
    requested_profile = getattr(args, "arm_execution_profile", None)
    position_requested = bool(
        requested_profile != "impedance_40hz"
        and str(getattr(args, "arm_control_mode", "")).strip()
        == RM65_ARM_CONTROL_MODE_POSITION
    )
    if (
        requested_profile in (None, "")
        and position_requested
    ):
        # Deployment defaults to the clock contract already validated by the
        # current position teleop. Explicit rate flags remain profile
        # overrides through resolve_real_arm_execution_config().
        args.arm_execution_profile = POSITION_15_60_120HZ_PROFILE_NAME
    explicit_compatibility_arm_profile = (
        any(
            token == "--arm-execution-profile"
            or token.startswith("--arm-execution-profile=")
            for token in raw_tokens
        )
        and requested_profile not in (None, "", POSITION_15_60_120HZ_PROFILE_NAME)
    )
    if explicit_compatibility_arm_profile:
        args.position_control_profile_resolved = LEGACY_POSITION_CONTROL_PROFILE_NAME
        args.position_control_profile_metadata = {
            "name": LEGACY_POSITION_CONTROL_PROFILE_NAME,
            "compatibility_mode": True,
            "reason": f"explicit_arm_execution_profile:{requested_profile}",
        }
    else:
        if (
            str(args.position_control_profile)
            == LEGACY_POSITION_CONTROL_PROFILE_NAME
        ):
            if not any(
                token == "--position-ik-solve-max-iterations"
                or token.startswith("--position-ik-solve-max-iterations=")
                for token in raw_tokens
            ):
                args.position_ik_solve_max_iterations = 2
            if not any(
                token in (
                    "--position-ik-adaptive-iterations",
                    "--no-position-ik-adaptive-iterations",
                )
                for token in raw_tokens
            ):
                args.position_ik_adaptive_iterations = False
        apply_rm65_position_profile_defaults(
            args,
            raw_argv=raw_tokens,
            mode=(
                "impedance"
                if requested_profile == "impedance_40hz"
                else str(args.arm_control_mode)
            ),
            include_source_shaping=False,
        )
    current_position_profile = bool(
        args.arm_execution_profile == POSITION_15_60_120HZ_PROFILE_NAME
    )
    if current_position_profile and not any(
        token == "--arm-max-angvel" or token.startswith("--arm-max-angvel=")
        for token in raw_tokens
    ):
        args.arm_max_angvel = RM65_POSITION_TRACKING_MAX_VELOCITY_RAD_S
    if current_position_profile and not any(
        token == "--arm-follow-qpos-alpha"
        or token.startswith("--arm-follow-qpos-alpha=")
        for token in raw_tokens
    ):
        args.arm_follow_qpos_alpha = RM65_POSITION_TRACKING_FOLLOW_ALPHA
    execution_config = resolve_real_arm_execution_config(
        args,
        raw_argv=raw_tokens,
    )
    setattr(args, "_real_arm_execution_config", execution_config)
    args.arm_control_mode = execution_config.mode
    args.arm_canfd_follow = execution_config.arm_canfd_follow
    args.arm_control_hz = execution_config.arm_control_hz
    args.position_reference_hz = execution_config.position_reference_hz
    args.arm_follow_qpos_alpha = execution_config.arm_follow_qpos_alpha
    if rtc_profile_enabled and args.arm_control_mode == RM65_ARM_CONTROL_MODE_POSITION:
        rates = RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.position_rates
        if not math.isclose(
            float(args.position_reference_hz),
            rates.reference_hz,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(args.arm_control_hz),
            rates.command_hz,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "RTC-3c position cadence is fixed at 60 Hz reference / "
                "120 Hz command."
            )
    if (
        args.arm_control_mode == RM65_ARM_CONTROL_MODE_POSITION
        and float(args.arm_control_hz) < float(args.position_reference_hz)
    ):
        raise ValueError(
            "--arm-control-hz must be >= --position-reference-hz in position mode "
            "so every reference can be represented by the command stream."
        )
    position_collision_enabled = bool(
        getattr(args, "position_collision_check", True)
    )
    if (
        float(args.position_collision_distance_epsilon_m)
        > float(args.position_collision_min_distance_m)
    ):
        raise ValueError(
            "--position-collision-distance-epsilon-m must not exceed "
            "--position-collision-min-distance-m."
        )
    if int(args.position_ik_solve_min_iterations) > int(
        args.position_ik_solve_max_iterations
    ):
        raise ValueError(
            "--position-ik-solve-min-iterations must be <= "
            "--position-ik-solve-max-iterations."
        )
    args.arm_workspace_min = _parse_vector3(args.arm_workspace_min)
    args.arm_workspace_max = _parse_vector3(args.arm_workspace_max)
    if np.any(
        np.asarray(args.arm_workspace_min, dtype=float)
        > np.asarray(args.arm_workspace_max, dtype=float)
    ):
        raise ValueError("--arm-workspace-min must not exceed --arm-workspace-max.")
    configured_hands_enabled = _hands_enabled(args)
    end_effector_sides = _parse_end_effector_sides(
        args.end_effector_sides,
        active_sides=active_sides,
        hands_enabled=configured_hands_enabled,
    )
    args.end_effector_sides = end_effector_sides
    hands_enabled = bool(end_effector_sides)
    if bool(args.allow_missing_v22_end_effectors) and hands_enabled:
        raise ValueError(
            "--allow-missing-v22-end-effectors is only valid with "
            "--end-effector-sides none for an explicit arm-only run."
        )
    if not bool(args.arms_enabled) and not hands_enabled:
        raise ValueError("Enable at least one actuator family: arms or hands.")
    if int(args.debug_replay_chunk_size) <= 0:
        raise ValueError("--debug-replay-chunk-size must be positive.")
    if float(args.init_time_s) < 0.0:
        raise ValueError("--init-time-s must be non-negative.")
    if args.init_dataset and args.init_parquet:
        raise ValueError("Use only one of --init-dataset or --init-parquet.")
    if args.deploy_fixed_init_dataset and args.deploy_fixed_init_parquet:
        raise ValueError("Use only one of --deploy-fixed-init-dataset or --deploy-fixed-init-parquet.")
    if args.policy_qpos_init_dataset and args.policy_qpos_init_parquet:
        raise ValueError("Use only one of --policy-qpos-init-dataset or --policy-qpos-init-parquet.")
    if args.deploy_fixed_init_dataset or args.deploy_fixed_init_parquet:
        if args.policy_qpos_init_dataset or args.policy_qpos_init_parquet:
            raise ValueError("Use only one of --deploy-fixed-init-* or --policy-qpos-init-*.")
        if active_sides != ("right",):
            raise ValueError("--deploy-fixed-init-* is currently right-arm/right-hand only; use --active-sides right.")
        if not bool(args.arms_enabled):
            raise ValueError("--deploy-fixed-init-* requires arms to be enabled.")
    if args.policy_qpos_init_dataset or args.policy_qpos_init_parquet:
        if not bool(args.arms_enabled):
            raise ValueError("--policy-qpos-init-* requires arms to be enabled.")
    has_dataset_qpos_start = bool(
        args.policy_qpos_init_dataset
        or args.policy_qpos_init_parquet
        or args.deploy_fixed_init_dataset
        or args.deploy_fixed_init_parquet
    )
    if (
        bool(args.send_actions)
        and bool(args.arms_enabled)
        and not has_dataset_qpos_start
        and not bool(args.start_init_qpos)
    ):
        raise ValueError(
            "Real arm policy execution requires dataset-frame qpos initialization "
            "(--policy-qpos-init-dataset or --policy-qpos-init-parquet). "
            "To intentionally start from the current qpos, pass --start-init_qpos."
        )
    if (
        raw_argv
        and bool(args.send_actions)
        and bool(args.arms_enabled)
        and not any(
            token == "--arm-execution-profile"
            or token.startswith("--arm-execution-profile=")
            or token == "--arm-control-mode"
            or token.startswith("--arm-control-mode=")
            for token in raw_tokens
        )
    ):
        raise ValueError(
            "Real arm policy execution requires an explicit execution branch: "
            "pass --arm-execution-profile "
            "impedance_40hz|position_15_60_120hz|position "
            "or --arm-control-mode impedance|position."
        )
    if (
        bool(args.send_actions)
        and bool(args.arms_enabled)
        and args.arm_control_mode == RM65_ARM_CONTROL_MODE_POSITION
        and not position_collision_enabled
    ):
        raise ValueError(
            "--send-actions in position mode requires --position-collision-check; "
            "it may only be disabled for a non-motion dry run"
        )
    if (
        bool(args.send_actions)
        and bool(args.arms_enabled)
        and args.arm_control_mode == RM65_ARM_CONTROL_MODE_POSITION
        and active_sides != ("left", "right")
    ):
        raise ValueError(
            "High-rate coupled collision IK requires --active-sides left right (or both). "
            "A single active arm has no authoritative passive-arm qpos/TCP, so the "
            "dual-arm model must fail closed rather than guess it."
        )
    if args.debug_replay_end_index is not None and int(args.debug_replay_end_index) <= int(args.debug_replay_start_index):
        raise ValueError("--debug-replay-end-index must be greater than --debug-replay-start-index.")
    action_hz_start = args.debug_replay_start_index if args.action_hz_start_index is None else args.action_hz_start_index
    action_hz_end = args.debug_replay_end_index if args.action_hz_end_index is None else args.action_hz_end_index
    if action_hz_end is not None and int(action_hz_end) <= int(action_hz_start):
        raise ValueError("--action-hz-end-index must be greater than --action-hz-start-index.")
    if args.debug_replay_dataset and args.debug_replay_parquet:
        raise ValueError("Use only one of --debug-replay-dataset or --debug-replay-parquet.")
    if args.action_hz_dataset and args.action_hz_parquet:
        raise ValueError("Use only one of --action-hz-dataset or --action-hz-parquet.")
    if (args.debug_replay_dataset or args.debug_replay_parquet) and str(args.policy_spec or "hold").strip().lower() not in {"", "hold", "builtin:hold"}:
        raise ValueError("--debug-replay-* replaces --policy; omit --policy in replay mode.")
    parse_tactile_regions(args.hand_tactile_regions)
    if bool(args.send_actions) and bool(args.arms_enabled):
        for side in active_sides:
            if not getattr(args, f"{side}_arm_ip"):
                raise ValueError(f"--send-actions requires --{side}-arm-ip for active side '{side}'.")
    if bool(args.send_actions) and hands_enabled:
        for side in end_effector_sides:
            if _hand_backend_mode_for_side(args, side) == HAND_BACKEND_INSPIRE:
                if not getattr(args, f"{side}_hand_ip"):
                    raise ValueError(f"--send-actions with hands requires --{side}-hand-ip.")
    if bool(args.visualize_tactile):
        if any(
            _hand_backend_mode_for_side(args, side) == HAND_BACKEND_DMTAC_GRIPPER
            for side in end_effector_sides
        ):
            raise ValueError("--visualize-tactile currently supports InspireTac UV tactile, not DMTAC gripper tactile.")
        if not hands_enabled:
            raise ValueError("--visualize-tactile requires hands to be enabled.")
        for side in end_effector_sides:
            if not getattr(args, f"{side}_hand_ip"):
                raise ValueError(f"--visualize-tactile requires --{side}-hand-ip.")
    if args.camera_mode in {"single", "multi"}:
        if args.camera_mode == "multi" and args.camera_serial:
            raise ValueError("--camera-mode multi uses --camera-config; omit --camera-serial.")
        if not args.camera_serial and not Path(args.camera_config).exists():
            raise ValueError(f"--camera-mode {args.camera_mode} requires an existing --camera-config.")
        if int(args.camera_width) <= 0 or int(args.camera_height) <= 0:
            raise ValueError("--camera-width/--camera-height must be positive.")
        if int(args.camera_fps) <= 0:
            raise ValueError("--camera-fps must be positive.")
        if int(args.camera_keep_last) <= 0:
            raise ValueError("--camera-keep-last must be positive.")
        if float(args.camera_frame_max_age_sec) <= 0.0:
            raise ValueError("--camera-frame-max-age-sec must be positive.")
        if int(args.camera_wait_timeout_ms) <= 0:
            raise ValueError("--camera-wait-timeout-ms must be positive.")
        if float(args.camera_stale_after_sec) <= 0.0:
            raise ValueError("--camera-stale-after-sec must be positive.")
        if float(args.camera_start_stagger_sec) < 0.0:
            raise ValueError("--camera-start-stagger-sec must be >= 0.")
    elif bool(args.vis_cam):
        raise ValueError("--vis_cam requires --camera-mode single or --camera-mode multi.")
    return active_sides


def _runtime_action_keyframe_period_sec(args: argparse.Namespace) -> float:
    """Bind RTC runtime timing without changing the legacy sync default."""

    profile_id = str(getattr(args, "rtc_profile", "off")).strip()
    if profile_id == RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1:
        return 1.0 / RTC_3C_VIRTUAL_SHADOW_PROFILE_V1.action_hz
    if profile_id == RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1:
        return 1.0 / RAW_RTC_FINAL_CHUNK_PROFILE_V1.action_hz
    # Preserve the historical PolicyDeployRuntimeConfig default for sync_queue_v1.
    return 1.0 / 15.0


def _action_hz_reference(args: argparse.Namespace) -> tuple[Path | None, Path | None, int, int, int | None]:
    dataset = args.action_hz_dataset or args.debug_replay_dataset
    parquet = args.action_hz_parquet or args.debug_replay_parquet
    episode = (
        int(args.action_hz_episode)
        if args.action_hz_episode is not None
        else int(args.debug_replay_episode)
    )
    start_index = (
        int(args.action_hz_start_index)
        if args.action_hz_start_index is not None
        else int(args.debug_replay_start_index)
    )
    end_index = (
        int(args.action_hz_end_index)
        if args.action_hz_end_index is not None
        else args.debug_replay_end_index
    )
    return dataset, parquet, episode, start_index, end_index


def resolve_effective_action_hz(args: argparse.Namespace, policy: Any | None = None) -> tuple[float, dict[str, Any]]:
    explicit = float(args.action_hz)
    if explicit > 0.0:
        return explicit, {"source": "cli.action_hz", "hz": explicit}

    if policy is not None:
        estimate = getattr(policy, "metadata", {}).get("action_hz_estimate")
        if isinstance(estimate, Mapping):
            hz = float(estimate.get("hz", 0.0) or 0.0)
            if hz > 0.0:
                return hz, dict(estimate)

    dataset, parquet, episode, start_index, end_index = _action_hz_reference(args)
    if dataset or parquet:
        path, metadata = resolve_replay_parquet_path(
            dataset_root=dataset,
            parquet_path=parquet,
            episode_index=int(episode),
        )
        estimate = infer_replay_action_hz(
            parquet_path=path,
            dataset_root=dataset,
            episode_index=int(episode),
            start_index=int(start_index),
            end_index=end_index,
            resolved_metadata=metadata,
        )
        hz = float(estimate.get("hz", 0.0) or 0.0)
        if hz > 0.0:
            estimate = dict(estimate)
            estimate["source_path"] = str(path)
            return hz, estimate

    default_hz = float(args.hz)
    return default_hz, {"source": "cli.hz", "hz": default_hz}


def resolve_effective_policy_hz(args: argparse.Namespace, *, action_hz: float, action_hz_source: str) -> tuple[float, str]:
    if args.policy_hz is not None:
        return float(args.policy_hz), "cli.policy_hz"
    return float(action_hz), "action_hz"


def resolve_arm_step_dt_max_scale(
    args: argparse.Namespace,
    *,
    control_hz: float,
    action_hz: float | None = None,
) -> tuple[float, bool]:
    explicit = getattr(args, "arm_step_dt_max_scale", None)
    if explicit is not None:
        return float(explicit), False
    target_hz = float(action_hz or 0.0)
    if target_hz <= 0.0:
        target_hz = float(getattr(args, "hz", DEFAULT_TELEOP_SAMPLE_HZ) or DEFAULT_TELEOP_SAMPLE_HZ)
    target_hz = max(1e-6, target_hz)
    required_scale = 1.1 * max(1e-6, float(control_hz)) / target_hz
    return max(DEPLOY_ARM_STEP_DT_MAX_SCALE, required_scale), True


def apply_arm_step_dt_defaults(
    args: argparse.Namespace,
    *,
    control_hz: float,
    action_hz: float | None = None,
) -> None:
    max_scale, auto = resolve_arm_step_dt_max_scale(
        args,
        control_hz=float(control_hz),
        action_hz=action_hz,
    )
    setattr(args, "arm_step_dt_max_scale", float(max_scale))
    setattr(args, "arm_step_dt_max_scale_auto", bool(auto))


def deploy_arm_command_step_dt_sec(
    args: argparse.Namespace,
    *,
    control_hz: float,
    action_hz: float,
) -> float:
    base_dt = 1.0 / max(1e-6, float(control_hz))
    if not bool(getattr(args, "arm_step_dt_scaling", True)):
        return base_dt
    action_dt = 1.0 / max(1e-6, float(action_hz))
    max_scale = float(getattr(args, "arm_step_dt_max_scale", DEPLOY_ARM_STEP_DT_MAX_SCALE))
    return float(np.clip(action_dt, base_dt, base_dt * max_scale))


def deploy_arm_backend_control_dt_sec(
    args: argparse.Namespace,
    *,
    control_hz: float,
    action_hz: float | None,
) -> float:
    arm_control_mode = normalize_rm65_arm_control_mode(getattr(args, "arm_control_mode", None))
    if arm_control_mode != RM65_ARM_CONTROL_MODE_POSITION:
        return 1.0 / max(1e-6, float(control_hz))
    del action_hz
    return 1.0 / max(1e-6, effective_position_reference_hz(args))


def deploy_arm_command_max_joint_step(
    args: argparse.Namespace,
    *,
    control_hz: float,
    action_hz: float,
) -> np.ndarray:
    step_dt = deploy_arm_command_step_dt_sec(
        args,
        control_hz=float(control_hz),
        action_hz=float(action_hz),
    )
    return np.asarray(args.arm_max_angvel, dtype=float).reshape(6) * step_dt


def annotate_deploy_arm_command_limits(
    command: Any,
    *,
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
    control_hz: float,
    action_hz: float,
) -> None:
    if command is None:
        return
    commands = getattr(command, "commands", {}) or {}
    if not commands:
        return
    step_dt = deploy_arm_command_step_dt_sec(
        args,
        control_hz=float(control_hz),
        action_hz=float(action_hz),
    )
    max_joint_step = deploy_arm_command_max_joint_step(
        args,
        control_hz=float(control_hz),
        action_hz=float(action_hz),
    )
    backend_control_dt = deploy_arm_backend_control_dt_sec(
        args,
        control_hz=float(control_hz),
        action_hz=float(action_hz),
    )
    for side in active_sides:
        effector_command = commands.get(f"{side}_arm")
        if effector_command is None:
            continue
        effector_command.metadata["max_joint_step_rad"] = max_joint_step.copy()
        effector_command.metadata["deploy_max_joint_step_rad"] = max_joint_step.copy()
        effector_command.metadata["deploy_action_step_dt_sec"] = float(step_dt)
        effector_command.metadata["deploy_backend_control_dt_sec"] = float(backend_control_dt)
        effector_command.metadata["deploy_arm_step_dt_scaling"] = bool(
            getattr(args, "arm_step_dt_scaling", True)
        )
        effector_command.metadata["deploy_arm_step_dt_max_scale"] = float(
            getattr(args, "arm_step_dt_max_scale", DEPLOY_ARM_STEP_DT_MAX_SCALE)
        )


def command_effector_kind_subset(
    command: TeleopCommand | None,
    *,
    kind: str,
) -> TeleopCommand | None:
    """Copy only arm or hand commands while preserving command metadata."""

    if command is None:
        return None
    suffix = f"_{str(kind).strip()}"
    subset = TeleopCommand(
        timestamp=command.timestamp,
        metadata=dict(command.metadata),
    )
    for effector_name, effector_command in command.commands.items():
        if str(effector_name).endswith(suffix):
            subset.add_command(effector_command.copy())
    return subset if subset.commands else None


def _dispatch_position_discrete_command_checked(
    command: TeleopCommand,
    *,
    hand_backends: Mapping[str, EffectorBackend],
) -> Mapping[str, Any]:
    """Dispatch one reference-clock-aligned hand keyframe and preserve receipts.

    Hand backends historically used both exceptions and return values to
    report I/O failures.  Position deploy needs an explicit result so the
    asynchronous executor can distinguish a safe no-op/rate skip from a real
    rejection and hold the arm stream on the latter.
    """

    receipts: dict[str, dict[str, Any]] = {}
    accepted_all = True
    skipped_any = False
    for effector_name, effector_command in command.commands.items():
        side = str(effector_command.side or "")
        backend = hand_backends.get(side)
        if backend is None:
            raise KeyError(f"Deploy backend missing for discrete effector {effector_name!r}.")
        started = time.monotonic()
        try:
            result = backend.send_command(effector_command)
            state = (
                backend.get_state(
                    include_tactile_arrays=False,
                    include_tactile_vector=False,
                )
                if isinstance(backend, DMTacGripperBackend)
                else backend.get_state()
            )
        except Exception as exc:
            receipt = {
                "accepted": False,
                "skipped": False,
                "fatal": True,
                "status": "exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "duration_ms": (time.monotonic() - started) * 1000.0,
            }
            receipts[str(effector_name)] = receipt
            command.metadata["position_discrete_backend_receipts"] = dict(receipts)
            effector_command.metadata["position_discrete_backend_receipt"] = receipt
            raise RuntimeError(
                f"{effector_name} discrete dispatch failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        status = "accepted"
        accepted = result is not False
        skipped = False
        if isinstance(backend, DMTacGripperBackend):
            status = str(state.get("gripper_last_send_status") or "unknown")
            if status == "rate_limited":
                accepted = False
                skipped = True
            elif status == "skipped_threshold":
                # The requested value is already within the configured no-op
                # threshold; no additional physical packet is required.
                accepted = True
            elif status != "sent" and result is False:
                accepted = False
        elif isinstance(backend, InspireTacHandBackend):
            status = "sent_or_threshold_noop" if result is not False else "rejected"

        fatal = bool(not accepted and not skipped)
        receipt = {
            "accepted": bool(accepted),
            "skipped": bool(skipped),
            "fatal": bool(fatal),
            "status": status,
            "backend_type": type(backend).__name__,
            "backend_ready": bool(backend.is_ready()),
            "telemetry_error": state.get("telemetry_error"),
            "duration_ms": (time.monotonic() - started) * 1000.0,
        }
        receipts[str(effector_name)] = receipt
        effector_command.metadata["position_discrete_backend_receipt"] = receipt
        command.metadata["position_discrete_backend_receipts"] = dict(receipts)
        if fatal:
            raise RuntimeError(
                f"{effector_name} discrete dispatch rejected: "
                f"status={status} telemetry_error={state.get('telemetry_error')}"
            )
        accepted_all = accepted_all and bool(accepted)
        skipped_any = skipped_any or bool(skipped)
    return {
        "accepted": bool(accepted_all),
        "skipped": bool(skipped_any),
        "fatal": False,
        "effectors": receipts,
    }


def hold_deploy_arm_backends(
    arm_backends: Mapping[str, EffectorBackend],
    *,
    reason: str,
) -> None:
    """Request all configured real arm backends to hold before reporting errors."""

    errors: list[str] = []
    for side, backend in arm_backends.items():
        try:
            backend.hold_motion(str(reason))
        except Exception as exc:  # noqa: BLE001 - attempt every active side.
            errors.append(f"{side}: {type(exc).__name__}: {exc}")
    if errors:
        raise RuntimeError("deploy arm hold failed: " + "; ".join(errors))


def build_arm_backends(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
    *,
    control_hz: float | None,
    action_hz: float | None = None,
) -> dict[str, EffectorBackend]:
    if not bool(args.arms_enabled):
        return {}
    arm_urdf_path = str(args.arm_urdf_path or require_asset_path("rm65_arm.urdf"))
    control_hz = effective_arm_control_hz(args) if control_hz is None else float(control_hz)
    control_dt = 1.0 / float(control_hz)
    backend_control_dt = deploy_arm_backend_control_dt_sec(
        args,
        control_hz=float(control_hz),
        action_hz=action_hz,
    )
    arm_control_mode = normalize_rm65_arm_control_mode(getattr(args, "arm_control_mode", None))
    worker_binding = getattr(args, "_rm65_legacy_worker_binding", None)
    if worker_binding is not None:
        if not isinstance(worker_binding, RM65LegacyWorkerExecutionBinding):
            raise TypeError("_rm65_legacy_worker_binding has an unsupported type")
        if tuple(active_sides) != ("left", "right"):
            raise ValueError(
                "RM65 arms-only worker binding requires coupled left/right sides"
            )
        if arm_control_mode != RM65_ARM_CONTROL_MODE_POSITION:
            raise ValueError(
                "RM65 arms-only worker binding excludes impedance control"
            )
    command_input_hz = (
        effective_position_reference_hz(args)
        if arm_control_mode == RM65_ARM_CONTROL_MODE_POSITION
        else action_hz
    )
    apply_arm_step_dt_defaults(
        args,
        control_hz=float(control_hz),
        action_hz=command_input_hz,
    )
    arm_canfd_follow = effective_arm_canfd_follow(args)
    observation_only = bool(getattr(args, "rtc_real_shadow", False))
    if observation_only and bool(getattr(args, "send_actions", False)):
        raise ValueError("rtc real shadow backend construction requires send_actions=False")
    if observation_only:
        # A read-only shadow must not accidentally inherit a force/CANFD
        # execution setting from the checked-in lab profile.
        arm_canfd_follow = False
    backends: dict[str, EffectorBackend] = {}
    for side in active_sides:
        arm_ip = str(getattr(args, f"{side}_arm_ip") or "")
        if arm_control_mode == RM65_ARM_CONTROL_MODE_IMPEDANCE:
            backends[side] = RM65ImpedanceArmBackend(
                RM65ImpedanceArmBackendConfig(
                    arm_ip=arm_ip or "0.0.0.0",
                    side=side,
                    connect_hardware=bool(arm_ip),
                    observation_only=observation_only,
                    tool_profile=getattr(args, f"{side}_tool_profile", "") or None,
                    strict_tool_profile=bool(getattr(args, "strict_tool_profile", True)),
                    impedance_mode="pose",
                    force_tare_sec=float(getattr(args, "force_tare_sec", 3.0)),
                    control_dt=control_dt,
                    action_hz=(
                        None
                        if action_hz is None
                        else float(action_hz)
                    ),
                    max_angvel=tuple(float(value) for value in args.arm_max_angvel),
                    actual_send_dt_alpha=float(args.arm_step_dt_alpha),
                    actual_send_dt_min_scale=1.0,
                    actual_send_dt_max_scale=float(args.arm_step_dt_max_scale),
                    actual_send_dt_stall_reset_sec=float(args.arm_step_dt_stall_reset_sec),
                    debug_timing=bool(getattr(args, "debug", False)),
                    # Force-policy deployment follows the impedance-only
                    # safety envelope.  The position branch below keeps its
                    # established planner and timing behavior unchanged.
                    impedance_overrides={
                        "canfd_follow": bool(arm_canfd_follow),
                        "joint_rate_limit_use_nominal_dt": True,
                        "admittance_use_nominal_dt": True,
                        "joint_operating_margin_deg": 3.0,
                        "ik_hard_deadline_enabled": True,
                        # Deployment has a 15 Hz policy cadence but must keep
                        # force/admittance and safe output at 40 Hz.  The
                        # controller therefore owns full Pinocchio IK on a
                        # separate 15 Hz worker; its 40 Hz path is local-only
                        # and yields while a full solve is in flight.  This is
                        # intentionally an OpenPI deployment override, never a
                        # teleoperation default.
                        "deployment_dual_rate_ik_enabled": True,
                        "deployment_full_ik_hz": 15.0,
                        "deployment_full_ik_deadline_ms": 66.66666666666667,
                        "deployment_local_ik_deadline_ms": 6.0,
                        "deployment_full_ik_preempts_local": True,
                        "ik_resolver_candidate_provider_names": ["rm65_pinocchio"],
                        "ik_resolver_local_step_provider_names": ["rm65_pinocchio"],
                        "ik_resolver_local_step_max_position_worsen_m": 0.003,
                        "ik_resolver_local_step_max_orientation_worsen_deg": 0.5,
                        "ik_resolver_local_step_dynamic_worsen_enabled": False,
                        "direct_joint_hint_max_linear_offset_m": 0.0,
                        "direct_joint_hint_max_angular_offset_rad": 0.0,
                    },
                )
            )
            continue
        binding_kwargs = (
            {}
            if worker_binding is None
            else worker_binding.arm_backend_kwargs(
                side,
                connect_hardware=bool(arm_ip),
            )
        )
        backends[side] = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip=arm_ip or "0.0.0.0",
                side=side,
                arm_urdf_path=arm_urdf_path,
                connect_hardware=bool(arm_ip),
                observation_only=observation_only,
                control_dt=backend_control_dt,
                max_angvel=tuple(float(value) for value in args.arm_max_angvel),
                singularity_qpos_init_slowdown=float(args.singularity_qpos_init_slowdown),
                dry_run_fast=not bool(arm_ip),
                hardware_read_retries=int(args.rm65_hardware_read_retries),
                hardware_read_retry_delay_sec=float(args.rm65_hardware_read_retry_delay_sec),
                initial_qpos=getattr(args, f"{side}_initial_qpos"),
                tool_profile=getattr(args, f"{side}_tool_profile", "") or None,
                ik_framework=str(getattr(args, "rm65_ik_framework", "robot_motion")),
                use_actual_send_dt_scaling=bool(getattr(args, "arm_step_dt_scaling", True)),
                actual_send_dt_alpha=float(args.arm_step_dt_alpha),
                actual_send_dt_min_scale=1.0,
                actual_send_dt_max_scale=float(args.arm_step_dt_max_scale),
                actual_send_dt_stall_reset_sec=float(args.arm_step_dt_stall_reset_sec),
                background_streaming_enabled=bool(arm_ip) and bool(arm_canfd_follow),
                background_streaming_hz=float(control_hz),
                position_command_planner=str(args.position_command_planner),
                position_velocity_ramp_sec=float(args.position_velocity_ramp_sec),
                position_target_velocity_filter_tau_sec=float(
                    args.position_target_velocity_filter_tau_sec
                ),
                position_max_acceleration_rad_s2=tuple(
                    float(value)
                    for value in args.position_max_acceleration_rad_s2
                ),
                position_max_command_step_deg=float(
                    args.position_max_command_step_deg
                ),
                position_degraded_speed_scale=float(
                    args.position_degraded_speed_scale
                ),
                arm_follow_qpos_alpha=float(args.arm_follow_qpos_alpha),
                canfd_follow=bool(arm_canfd_follow),
                canfd_trajectory_mode=int(getattr(args, "arm_canfd_trajectory_mode", 0)),
                canfd_radio=int(getattr(args, "arm_canfd_radio", 0)),
            ),
            **binding_kwargs,
        )
    return backends


def build_hand_backends(args: argparse.Namespace, active_sides: tuple[str, ...]) -> dict[str, EffectorBackend]:
    if not active_sides or not _hands_enabled(args):
        return {}
    configs = _dmtac_backend_configs(args, active_sides)
    backends: dict[str, EffectorBackend] = {
            side: DMTacGripperBackend(config)
            for side, config in configs.items()
            if side in active_sides
            and _hand_backend_mode_for_side(args, side) == HAND_BACKEND_DMTAC_GRIPPER
    }

    tactile_regions = parse_tactile_regions(args.hand_tactile_regions)
    observation_only = bool(getattr(args, "rtc_real_shadow", False))
    for side in active_sides:
        if _hand_backend_mode_for_side(args, side) != HAND_BACKEND_INSPIRE:
            continue
        hand_ip = str(getattr(args, f"{side}_hand_ip") or "")
        backends[side] = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip=hand_ip or "0.0.0.0",
                port=int(args.hand_modbus_port),
                side=side,
                effector_name=f"{side}_hand",
                connect_hardware=bool(hand_ip),
                observation_only=observation_only,
                max_fingertip_force=int(args.max_fingertip_force),
                send_threshold=float(args.hand_send_threshold),
                enable_telemetry=bool(hand_ip),
                telemetry_hz=float(args.hand_telemetry_hz),
                tactile_regions=tuple(tactile_regions),
            )
        )
    return backends


def _argument_was_supplied(raw_argv: list[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in raw_argv)


def resolve_policy_camera_input_keys(
    args: argparse.Namespace,
    policy_artifact: Any | None,
    *,
    raw_argv: list[str],
) -> tuple[str, ...] | None:
    """Resolve artifact-bound logical cameras before any device is opened.

    ``None`` deliberately means a legacy artifact with no formal visual
    contract, preserving its existing runtime camera behavior.  A tuple means
    the exact complete set that may be captured and forwarded to the policy.
    """

    camera_input = None if policy_artifact is None else policy_artifact.camera_input
    if camera_input is None:
        return None
    selected_keys = camera_input.source_keys
    if not selected_keys:
        if args.camera_mode != "none":
            raise ValueError(
                "policy artifact declares no camera inputs; use --camera-mode none."
            )
        return selected_keys
    if args.camera_mode == "none":
        if camera_input.required:
            raise ValueError(
                "policy artifact requires camera inputs "
                f"{list(selected_keys)}; choose --camera-mode single or multi."
            )
        return selected_keys
    if args.camera_serial:
        if len(selected_keys) != 1:
            raise ValueError(
                "--camera-serial can satisfy only one artifact camera input; "
                f"artifact requires {list(selected_keys)}."
            )
        selected_key = selected_keys[0]
        if _argument_was_supplied(raw_argv, "--camera-key") and args.camera_key != selected_key:
            raise ValueError(
                "--camera-key disagrees with policy artifact camera input: "
                f"expected {selected_key!r}, got {args.camera_key!r}."
            )
        # The default CLI key is ``front``.  For a wrist-only artifact it is
        # safe to substitute the artifact's logical key before source creation.
        args.camera_key = selected_key
    elif args.camera_mode == "single" and len(selected_keys) != 1:
        raise ValueError(
            "--camera-mode single requires exactly one artifact camera input; "
            f"artifact requires {list(selected_keys)}."
        )
    return selected_keys


def build_camera_sources(
    args: argparse.Namespace,
    *,
    selected_camera_keys: tuple[str, ...] | None = None,
) -> dict[str, RealSenseRGBSource]:
    if args.camera_mode == "none":
        return {}
    if selected_camera_keys == ():
        raise ValueError("A policy artifact without camera inputs requires --camera-mode none.")
    if args.camera_serial:
        if args.camera_mode != "single":
            raise ValueError("--camera-serial can only be used with --camera-mode single.")
        config = RealSenseRGBCameraConfig(
            serial=str(args.camera_serial),
            width=int(args.camera_width),
            height=int(args.camera_height),
            fps=int(args.camera_fps),
            camera_key=str(args.camera_key),
            keep_last=int(args.camera_keep_last),
            color_space=str(args.camera_color_space),
            wait_timeout_ms=int(args.camera_wait_timeout_ms),
            stale_after_sec=float(args.camera_stale_after_sec),
            auto_exposure_priority=args.camera_auto_exposure_priority,
            startup_hardware_reset=bool(
                getattr(args, "camera_startup_hardware_reset", False)
            ),
            metadata={"source": "teleop_runtime_deploy_cli"},
        )
        if selected_camera_keys is not None and config.camera_key not in selected_camera_keys:
            raise ValueError(
                f"--camera-key {config.camera_key!r} is not selected by policy artifact "
                f"{list(selected_camera_keys)}."
            )
        return {config.camera_key: RealSenseRGBSource(config)}
    camera_configs = load_realsense_camera_config_map(Path(args.camera_config))
    if selected_camera_keys is not None:
        missing_keys = tuple(key for key in selected_camera_keys if key not in camera_configs)
        if missing_keys:
            raise ValueError(
                "--camera-config does not provide policy artifact camera inputs: "
                f"{list(missing_keys)}."
            )
        selected = set(selected_camera_keys)
        camera_configs = {
            key: config for key, config in camera_configs.items() if key in selected
        }
    # Config files historically omitted keep_last, which left deployment with
    # the source default of two frames even though --camera-keep-last already
    # defaulted to four for the single-camera CLI path.  Apply the deployment
    # setting to config-backed sources as a minimum while preserving a larger
    # value explicitly requested by a config file.
    requested_keep_last = int(
        getattr(args, "camera_keep_last", DEFAULT_DEPLOYMENT_CAMERA_KEEP_LAST)
    )
    for config in camera_configs.values():
        if isinstance(config, RealSenseRGBCameraConfig):
            config.keep_last = max(int(config.keep_last), requested_keep_last)
    if bool(getattr(args, "camera_startup_hardware_reset", False)):
        for config in camera_configs.values():
            config.startup_hardware_reset = True
    if args.camera_mode == "single" and len(camera_configs) != 1:
        raise ValueError(f"--camera-mode single expects exactly one camera config, got {len(camera_configs)}.")
    if args.camera_mode == "multi" and not camera_configs:
        raise ValueError("--camera-mode multi requires at least one camera config.")
    return {key: RealSenseRGBSource(config) for key, config in camera_configs.items()}


def start_camera_sources(
    camera_sources: Mapping[str, RealSenseRGBSource],
    *,
    stagger_sec: float = 0.0,
) -> None:
    """Start selected cameras serially in config order, with optional staggering."""

    delay_sec = float(stagger_sec)
    if delay_sec < 0.0:
        raise ValueError("camera start stagger_sec must be >= 0.")
    for index, (camera_key, camera_source) in enumerate(camera_sources.items()):
        if index > 0 and delay_sec > 0.0:
            print(
                f"[Camera] waiting {delay_sec:.2f}s before starting camera {camera_key}",
                flush=True,
            )
            time.sleep(delay_sec)
        config = camera_source.config
        print(
            "[Camera] starting "
            f"camera={camera_key} serial={config.serial} "
            f"startup_hardware_reset={bool(config.startup_hardware_reset)}",
            flush=True,
        )
        camera_source.start()


def select_preview_camera_sources(camera_sources: dict[str, RealSenseRGBSource], requested_key: str) -> dict[str, RealSenseRGBSource]:
    if not camera_sources:
        return {}
    key = str(requested_key or "").strip()
    if key:
        if key not in camera_sources:
            available = ", ".join(sorted(camera_sources))
            raise ValueError(f"--vis-cam-key {key!r} not found. Available camera keys: {available}")
        return {key: camera_sources[key]}
    selected_key = "front" if "front" in camera_sources else sorted(camera_sources)[0]
    return {selected_key: camera_sources[selected_key]}


def compact_log_value(value: object, *, max_chars: int = 90) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def camera_status_text(camera_sources: dict[str, RealSenseRGBSource], camera_previewer: CameraPreviewer | None) -> str:
    if not camera_sources:
        return "camera=off"
    bits: list[str] = []
    if camera_previewer is not None and camera_previewer.disabled_reason:
        bits.append("preview=disabled:" + compact_log_value(camera_previewer.disabled_reason, max_chars=80))
    for camera_key, camera_source in sorted(camera_sources.items()):
        try:
            info = camera_source.device_info()
        except Exception as exc:  # noqa: BLE001
            bits.append(f"{camera_key}=status_error:{type(exc).__name__}")
            continue
        frame_count = int(info.get("frame_count", 0) or 0)
        last_error = info.get("last_error")
        if last_error:
            bits.append(f"{camera_key}=frames:{frame_count},err:{compact_log_value(last_error, max_chars=70)}")
        else:
            bits.append(f"{camera_key}=frames:{frame_count}")
    return "camera=" + ";".join(bits)


def hand_state_snapshot(
    backends: Mapping[str, EffectorBackend],
    *,
    history_poller: HandStateHistoryPoller | None = None,
    timestamp: float | None = None,
) -> dict[str, dict[str, Any]]:
    if history_poller is not None and timestamp is not None:
        return history_poller.latest_by_side(float(timestamp))
    return {side: backend.get_state() for side, backend in backends.items()}


def hand_action_specs_for_backends(
    args: argparse.Namespace,
    backends: Mapping[str, EffectorBackend],
    active_sides: tuple[str, ...],
) -> dict[str, Any]:
    specs = {}
    for side in active_sides:
        if _hand_backend_mode_for_side(args, side) == HAND_BACKEND_DMTAC_GRIPPER:
            specs[side] = HandActionSpec.dmtac_gripper(side)
        else:
            specs[side] = HandActionSpec.inspire_tac(side)
    return hand_action_specs_metadata(specs)


def print_controls(
    *,
    send_actions: bool,
    auto_start: bool,
    wait_for_ready_confirm: bool,
    fixed_init_enabled: bool = False,
    policy_qpos_init_enabled: bool = False,
) -> None:
    print("\n" + "=" * 60)
    print("teleop_runtime policy deployment loop started")
    print(f"  send_actions={'on' if send_actions else 'off(dry-run)'}")
    if (fixed_init_enabled or policy_qpos_init_enabled) and auto_start:
        initial_policy_state = "paused(auto-start after init)"
    else:
        initial_policy_state = "active" if auto_start else "paused"
    print(f"  initial_policy_state={initial_policy_state}")
    if wait_for_ready_confirm:
        print("  ready_confirm=on(A/a required after startup init)")
    if fixed_init_enabled:
        print("  fixed_init=on(H first-frame qpos, H execute T, then startup init)")
        print("Controls: H=pause policy or fixed-init step, g=start after fixed-init, r=manual safety re-arm, A=confirm, q=quit")
    elif policy_qpos_init_enabled:
        print("  policy_qpos_init=on(g moves to dataset frame qpos, then startup init)")
        print("Controls: g=start qpos init / resume, r=manual safety re-arm, A=confirm ready, h=pause+clear queue, q=quit")
    else:
        print("Controls: g=start/resume, r=manual safety re-arm, A=confirm ready/start policy, h=pause+clear queue, q=quit")
    print("=" * 60 + "\n")


def _resolve_action_queue_policy(args: argparse.Namespace, *, debug_replay: bool) -> str:
    """Compatibility wrapper for the shared queue-policy resolver."""

    return resolve_action_queue_policy(
        action_queue_policy=getattr(args, "action_queue_policy", "auto"),
        append_action_chunks=bool(getattr(args, "append_action_chunks", False)),
        debug_replay=bool(debug_replay),
    )


def _resolve_runtime_pose_frame_contract(
    configured_contract: Any | None,
    *,
    policy_artifact: Any | None,
    policy_metadata: Mapping[str, Any],
) -> tuple[Any | None, str]:
    """Resolve the RM65 workspace-near-boundary default without changing v2.2 semantics."""

    if configured_contract is not None:
        return configured_contract, "explicit"

    pose_training_data = (
        None if policy_artifact is None else policy_artifact.pose_training_data
    )
    if (
        pose_training_data is not None
        and pose_training_data.origin == POSE_TRAINING_ORIGIN_NATIVE_V23
    ):
        return (
            default_rm65_workspace_near_boundary_runtime_pose_frame_contract(),
            "default_rm65_dual_arm_workspace_near_boundary_v1",
        )
    if policy_artifact is not None:
        # In particular, v2.2 raw-identity compatibility must stay identity.
        return None, "v22_identity_compatibility"

    dataset_info = policy_metadata.get("dataset_info")
    if (
        isinstance(dataset_info, Mapping)
        and str(dataset_info.get("dexteleop_dataset_schema") or "")
        == DEXTELEOP_DATASET_SCHEMA_V23
    ):
        return (
            default_rm65_workspace_near_boundary_runtime_pose_frame_contract(),
            "default_rm65_dual_arm_workspace_near_boundary_v1",
        )
    return None, "not_applicable"


def _send_legacy_deployment_action_commands(
    commands: Mapping[str, Any],
    *,
    arm_backends: Mapping[str, EffectorBackend],
    hand_backends: Mapping[str, EffectorBackend],
) -> None:
    """Archived direct DeploymentAction command dispatcher.

    The active deploy policy path now resolves NormalizedAction through
    RobotAction/TeleopCommand in teleop_runtime.deploy.canonical_action.
    """

    for side in ALL_SIDES:
        arm_command = commands.get(f"{side}_arm")
        arm_backend = arm_backends.get(side)
        if arm_command is not None and arm_backend is not None:
            arm_backend.send_command(arm_command)
        hand_command = commands.get(f"{side}_hand")
        hand_backend = hand_backends.get(side)
        if hand_command is not None and hand_backend is not None:
            hand_backend.send_command(hand_command)


def enable_impedance_force_control_backends(
    arm_backends: Mapping[str, EffectorBackend],
    *,
    ready_timeout_sec: float = 1.0,
) -> None:
    for side, backend in arm_backends.items():
        enable = getattr(backend, "enable_force_control", None)
        if not callable(enable):
            continue
        enable()
        wait_ready = getattr(backend, "wait_until_force_control_ready", None)
        if callable(wait_ready):
            wait_ready(timeout_sec=float(ready_timeout_sec))
        print(f"[Deploy] {side} impedance force control enabled.", flush=True)


def warm_start_impedance_force_control_backends(
    arm_backends: Mapping[str, EffectorBackend],
) -> tuple[str, ...]:
    """Prepare impedance controllers without starting their periodic output loops.

    The pose-impedance backend performs force-sensor setup and captures its
    initial safe reference in ``warm_start_force_control``.  Starting the
    40 Hz loop is deliberately a separate, explicit action: otherwise a
    deployment which is still waiting for ``g`` would already issue hold
    commands and continuously run IK.
    """

    warmed_sides: list[str] = []
    for side, backend in arm_backends.items():
        warm_start = getattr(backend, "warm_start_force_control", None)
        if not callable(warm_start):
            continue
        warm_start()
        warmed_sides.append(str(side))
    return tuple(warmed_sides)


def _compact_faults(faults: list[str], *, limit: int = 4) -> str:
    if not faults:
        return "none"
    shown = faults[:limit]
    suffix = "" if len(faults) <= limit else f",+{len(faults) - limit}"
    return ";".join(shown) + suffix


def _policy_status_text(policy: Any) -> str:
    wrapped_policy = getattr(policy, "policy", None)
    status = getattr(wrapped_policy, "status_text", None)
    if isinstance(status, str) and status:
        return status
    if callable(status):
        text = status()
        if text:
            return str(text)
    return ""


def _debug_replay_execution_exhausted(policy: Any, *, queued_actions: int) -> bool:
    """Stop non-loop replay after its already-decoded final chunk is consumed."""

    wrapped_policy = getattr(policy, "policy", policy)
    if not isinstance(wrapped_policy, DatasetReplayPolicy):
        return False
    return (
        not bool(wrapped_policy.loop)
        and bool(wrapped_policy.finished)
        and int(queued_actions) <= 0
    )


def _skip_startup_replay_frames(policy: Any, count: int = 1) -> int:
    wrapped_policy = getattr(policy, "policy", policy)
    skip = getattr(wrapped_policy, "skip_startup_frames", None)
    if not callable(skip):
        return 0
    return int(skip(count))


def _validate_debug_replay_arm_control_mode(
    policy: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    contract = validate_dataset_arm_control_mode(
        getattr(policy, "metadata", {}),
        selected_mode=args.arm_control_mode,
        hardware_output=bool(args.send_actions),
    )
    payload = contract.to_dict()
    metadata = getattr(policy, "metadata", None)
    if isinstance(metadata, dict):
        metadata["dataset_arm_control_mode_contract"] = dict(payload)
    if contract.warning:
        print(f"[ReplayMode] WARNING: {contract.warning}", flush=True)
    return payload


def _bind_policy_pose_mode_from_artifact(
    args: argparse.Namespace,
    policy_artifact: Any,
    *,
    raw_argv: list[str],
) -> None:
    """Compatibility wrapper around the shared artifact binding contract."""

    explicit = any(
        token == "--policy-action-pose-mode"
        or token.startswith("--policy-action-pose-mode=")
        for token in raw_argv
    )
    args.policy_action_pose_mode = resolve_policy_action_pose_mode_from_artifact(
        policy_artifact,
        configured_mode=str(args.policy_action_pose_mode),
        explicit=explicit,
    )


def _policy_observation_contract(
    *,
    policy: Any,
    policy_artifact: Any,
    deploy_layout: Any,
    pose_mode: str,
) -> Any:
    """Bind the strict actor view to this artifact/layout and its adapter needs."""

    effective_layout = deploy_layout
    if effective_layout is None:
        wrapped_policy = getattr(policy, "policy", None)
        policy_config = getattr(wrapped_policy, "config", None)
        effective_layout = getattr(policy_config, "policy_io_layout", None)
    hand_state_keys: set[str] = set()
    if effective_layout is not None:
        for side in ALL_SIDES:
            spec = effective_layout.end_effector(side)
            if spec.present:
                hand_state_keys.add(str(spec.state_source_field))
            channel = effective_layout.channel("state", side)
            for token in channel.tokens:
                if token.source_metadata_key:
                    hand_state_keys.add(str(token.source_metadata_key))

    if policy_artifact is not None:
        artifact_hash = canonical_content_hash(
            policy_artifact.to_dict(),
            domain="dexteleop.policy_observation_artifact_binding.v1",
        )
        binding_id = f"artifact={artifact_hash}"
    elif effective_layout is not None:
        binding_id = (
            f"layout={effective_layout.layout_hash};model=unversioned"
        )
    else:
        binding_id = f"policy={getattr(policy, 'name', 'unknown')}"
    return default_deploy_policy_observation_contract(
        binding_id=binding_id,
        last_action_pose_mode=str(pose_mode),
        compatibility_hand_state_keys=tuple(sorted(hand_state_keys)),
    )


def main(
    argv: list[str] | None = None,
    *,
    rtc_step_fanout: RtcFrontendStepFanout | None = None,
    rtc_emission_fanout: PolicyActionEmissionFanout | None = None,
    rtc_emission_publisher: RtcPolicyEmissionPublisher | None = None,
    rtc_model_adapter_factory: Callable[..., Any] | None = None,
) -> int:
    if rtc_step_fanout is not None and not isinstance(
        rtc_step_fanout,
        RtcFrontendStepFanout,
    ):
        raise TypeError("rtc_step_fanout must be an RtcFrontendStepFanout or None.")
    if rtc_emission_fanout is not None and not isinstance(
        rtc_emission_fanout,
        PolicyActionEmissionFanout,
    ):
        raise TypeError(
            "rtc_emission_fanout must be a PolicyActionEmissionFanout or None."
        )
    if rtc_emission_publisher is not None and not isinstance(
        rtc_emission_publisher,
        RtcPolicyEmissionPublisher,
    ):
        raise TypeError(
            "rtc_emission_publisher must be an "
            "RtcPolicyEmissionPublisher or None."
        )
    if (
        rtc_emission_fanout is not None
        and rtc_emission_publisher is not None
    ):
        raise ValueError(
            "Use rtc_emission_publisher or the compatibility "
            "rtc_emission_fanout, not both."
        )
    parser = argparse.ArgumentParser(
        description="Deploy a policy on teleop_runtime RM65 arms, optional InspireTac hands, and RealSense."
    )
    add_args(parser)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    debug_replay = bool(args.debug_replay_dataset or args.debug_replay_parquet)
    policy_kwargs = {} if debug_replay else parse_policy_kwargs(args.policy_kwargs_json)
    policy_artifact = (
        None
        if args.policy_artifact_json is None
        else load_policy_artifact(args.policy_artifact_json)
    )
    kwargs_artifact = policy_kwargs.get("policy_artifact")
    if kwargs_artifact is not None:
        kwargs_contract = policy_artifact_from_metadata(kwargs_artifact)
        if policy_artifact is not None and kwargs_contract.to_dict() != policy_artifact.to_dict():
            raise ValueError(
                "--policy-kwargs-json policy_artifact disagrees with --policy-artifact-json."
            )
        policy_artifact = kwargs_contract
    _bind_policy_pose_mode_from_artifact(
        args,
        policy_artifact,
        raw_argv=raw_argv,
    )
    if debug_replay and str(args.debug_replay_output_pose_mode) != str(args.policy_action_pose_mode):
        raise ValueError(
            "debug replay output pose mode must match policy action pose mode: "
            f"{args.debug_replay_output_pose_mode!r} != {args.policy_action_pose_mode!r}"
        )
    if policy_artifact is not None:
        layout_sides = tuple(
            side
            for side in ALL_SIDES
            if policy_artifact.policy_io_layout.end_effector(side).present
        )
        if not any(
            token == "--end-effector-sides" or token.startswith("--end-effector-sides=")
            for token in raw_argv
        ):
            args.end_effector_sides = "none" if not layout_sides else " ".join(layout_sides)
        if args.hands_enabled is None:
            args.hands_enabled = bool(layout_sides)
    active_sides = validate_args(args, raw_argv=raw_argv)
    if rtc_step_fanout is not None and str(args.rtc_profile) != (
        RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1
    ):
        raise ValueError("rtc_step_fanout requires the explicit RTC-3c profile.")
    if rtc_emission_fanout is not None and str(args.rtc_profile) == "off":
        raise ValueError("rtc_emission_fanout requires an explicit RTC profile.")
    if rtc_emission_publisher is not None and str(args.rtc_profile) == "off":
        raise ValueError("rtc_emission_publisher requires an explicit RTC profile.")
    if rtc_model_adapter_factory is not None and not callable(
        rtc_model_adapter_factory
    ):
        raise TypeError("rtc_model_adapter_factory must be callable or None.")
    selected_camera_keys = resolve_policy_camera_input_keys(
        args,
        policy_artifact,
        raw_argv=raw_argv,
    )
    end_effector_sides = tuple(args.end_effector_sides)
    hands_enabled = bool(end_effector_sides)
    if debug_replay:
        policy = load_dataset_replay_policy(
            dataset_root=args.debug_replay_dataset,
            parquet_path=args.debug_replay_parquet,
            episode_index=int(args.debug_replay_episode),
            action_column=str(args.debug_replay_action_column),
            start_index=int(args.debug_replay_start_index),
            end_index=args.debug_replay_end_index,
            chunk_size=int(args.debug_replay_chunk_size),
            loop=bool(args.debug_replay_loop),
            active_sides=active_sides,
            end_effector_sides=end_effector_sides,
            output_pose_mode=str(args.debug_replay_output_pose_mode),
            policy_io_layout=(
                None if policy_artifact is None else policy_artifact.policy_io_layout
            ),
        )
        _validate_debug_replay_arm_control_mode(policy, args)
    else:
        if policy_artifact is not None:
            policy_kwargs["policy_artifact"] = policy_artifact.to_dict()
        policy = load_policy(args.policy_spec, kwargs=policy_kwargs)
    deploy_layout = None if policy_artifact is None else policy_artifact.policy_io_layout
    if deploy_layout is None and debug_replay:
        dataset_info = policy.metadata.get("dataset_info")
        if (
            isinstance(dataset_info, Mapping)
            and str(dataset_info.get("dexteleop_dataset_schema") or "")
            in {DEXTELEOP_DATASET_SCHEMA_V22, DEXTELEOP_DATASET_SCHEMA_V23}
        ):
            deploy_layout = policy_io_layout_from_metadata(
                dataset_info,
                allow_legacy_v21=False,
            )
    runtime_capabilities = None
    if deploy_layout is not None:
        runtime_capabilities = {
            side: RuntimeEndEffectorCapability(
                side=side,
                adapter=deploy_layout.end_effector(side).adapter,
                backend=_hand_backend_mode_for_side(args, side),
            )
            for side in end_effector_sides
        }
    (
        runtime_pose_frame_contract,
        runtime_pose_frame_contract_source,
    ) = _resolve_runtime_pose_frame_contract(
        args.runtime_pose_frame_contract,
        policy_artifact=policy_artifact,
        policy_metadata=getattr(policy, "metadata", {}),
    )
    deploy_session_contract = resolve_deploy_session_contract(
        active_sides=active_sides,
        model_metadata=getattr(policy, "metadata", {}),
        policy_artifact=policy_artifact,
        policy_io_layout=deploy_layout,
        runtime_pose_frame_contract=runtime_pose_frame_contract,
        runtime_end_effectors=runtime_capabilities,
        hardware_output=bool(args.send_actions),
        allow_missing_end_effectors=bool(args.allow_missing_v22_end_effectors),
    )
    policy_observation_contract = _policy_observation_contract(
        policy=policy,
        policy_artifact=policy_artifact,
        deploy_layout=deploy_layout,
        pose_mode=str(args.policy_action_pose_mode),
    )
    policy_deploy_runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=active_sides,
            arms_enabled=bool(args.arms_enabled),
            hands_enabled=hands_enabled,
            policy_action_pose_mode=str(args.policy_action_pose_mode),
            policy_absolute_frame=str(deploy_session_contract.policy_absolute_frame),
            runtime_pose_frame_contract=deploy_session_contract.runtime_pose_frame_contract,
            policy_observation_contract=policy_observation_contract,
            observation_firewall_mode=str(
                args.policy_observation_firewall_mode
            ).replace("-", "_"),
            action_chunk_execution_steps=(
                None
                if getattr(args, "action_chunk_execution_steps", None) is None
                else int(args.action_chunk_execution_steps)
            ),
            action_keyframe_period_sec=_runtime_action_keyframe_period_sec(args),
        )
    )
    policy.metadata["resolved_deploy_session_contract"] = deploy_session_contract.to_dict()
    policy.metadata["policy_observation_firewall"] = (
        policy_deploy_runtime.observation_firewall_manifest()
    )
    policy.metadata["runtime_pose_frame_contract_source"] = (
        runtime_pose_frame_contract_source
    )
    policy.metadata["action_chunk_execution_steps"] = (
        policy_deploy_runtime.config.action_chunk_execution_steps
    )
    if runtime_pose_frame_contract_source == "default_rm65_dual_arm_workspace_near_boundary_v1":
        print(
            "[Deploy] runtime pose frame default: dual RM65 calibrated URDF "
            "workspace_near_boundary_frame.",
            flush=True,
        )
    if deploy_session_contract.policy_contract_mode == POLICY_CONTRACT_V22_NATIVE:
        raise RuntimeError(
            "v22_native typed RobotAction transport is available, but this run_policy loop "
            "does not yet have a configured native end-effector safety/dispatch adapter. "
            "Use an explicit legacy_q6_slot_indices bridge for this runtime."
        )
    action_queue_policy = _resolve_action_queue_policy(args, debug_replay=debug_replay)
    startup_init_action, startup_init_source = resolve_startup_init_action(args, policy)
    if startup_init_action is not None:
        startup_absolute_frame = deploy_session_contract.policy_absolute_frame
        if "policy_absolute_frame=runtime_raw" in startup_init_source:
            startup_absolute_frame = "runtime_raw"
        elif "policy_absolute_frame=canonical_desk" in startup_init_source:
            startup_absolute_frame = "canonical_desk"
        startup_init_action = convert_policy_absolute_vector_to_runtime_raw(
            startup_init_action,
            active_sides=active_sides,
            policy_absolute_frame=startup_absolute_frame,
            runtime_pose_frame_contract=(
                deploy_session_contract.runtime_pose_frame_contract
            ),
        )
    fixed_init_plan = load_deploy_fixed_init_plan(
        dataset_root=args.deploy_fixed_init_dataset,
        parquet_path=args.deploy_fixed_init_parquet,
        episode_spec=str(args.deploy_fixed_init_episode),
        action_column=str(args.deploy_fixed_init_action_column),
        active_sides=active_sides,
    )
    policy_qpos_init_plan = load_deploy_policy_qpos_init_plan(
        dataset_root=args.policy_qpos_init_dataset,
        parquet_path=args.policy_qpos_init_parquet,
        episode_index=int(args.policy_qpos_init_episode),
        frame_index=int(args.policy_qpos_init_frame),
        action_column=str(args.policy_qpos_init_action_column),
        active_sides=active_sides,
        qpos_init_source=str(args.policy_qpos_init_source),
    )

    effective_action_hz, action_hz_info = resolve_effective_action_hz(args, policy)
    effective_policy_hz, policy_hz_source = resolve_effective_policy_hz(
        args,
        action_hz=float(effective_action_hz),
        action_hz_source=str(action_hz_info.get("source", "")),
    )

    policy_arbiter_shadow: PassivePolicyArbiterShadow | None = None
    replay_proposal_shadow_adapter: DatasetReplayProposalShadowAdapter | None = None
    if str(args.action_arbiter_mode) == ARBITER_SHADOW_MODE_SHADOW:
        shadow_effectors = tuple(
            [
                f"{side}_arm"
                for side in active_sides
                if bool(args.arms_enabled)
            ]
            + [
                f"{side}_hand"
                for side in end_effector_sides
                if hands_enabled
            ]
        )
        runtime_session_id = f"deploy-shadow-{uuid.uuid4()}"
        policy_arbiter_shadow = PassivePolicyArbiterShadow(
            PolicyArbiterShadowConfig(
                runtime_session_id=runtime_session_id,
                control_domain_id="deploy-control-0",
                embodiment_instance_id="rm65-bimanual-0",
                producer_id=(
                    f"replay:{policy.name}"
                    if debug_replay
                    else f"policy:{policy.name}"
                ),
                active_effectors=shadow_effectors,
                action_space=DEPLOY_ACTION_VECTOR_ACTION_SPACE,
                # Event-limited: ordinary 15/30/40 Hz emissions are evaluated
                # immediately, while >60 Hz producers are coalesced by the
                # shared production decision-rate ceiling.
                decision_hz=60.0,
                policy_node_id="single_policy",
                clock_domain_id="local_monotonic",
                boot_id=runtime_session_id,
            )
        )
        print(
            "[ArbiterShadow] passive non-authoritative evaluation enabled "
            "with a 60 Hz event-limited gate; legacy execution remains owner.",
            flush=True,
        )
        if debug_replay:
            dataset_info = policy.metadata.get("dataset_info")
            dataset_info = dataset_info if isinstance(dataset_info, Mapping) else {}
            try:
                replay_proposal_shadow_adapter = DatasetReplayProposalShadowAdapter(
                    DatasetReplayProposalShadowConfig(
                        runtime_session_id=runtime_session_id,
                        control_domain_id="deploy-control-0",
                        embodiment_instance_id="rm65-bimanual-0",
                        producer_id=policy_arbiter_shadow.config.producer_id,
                        producer_epoch=policy_arbiter_shadow.config.producer_epoch,
                        component_schema_hash=(
                            policy_arbiter_shadow.component_catalog.semantic_content_hash
                        ),
                        component_coverage=shadow_effectors,
                        source_path=str(policy.metadata.get("source_path") or ""),
                        episode_id=str(
                            policy.metadata.get("episode_id")
                            or episode_id_from_index(int(args.debug_replay_episode))
                        ),
                        action_field=str(args.debug_replay_action_column),
                        source_action_space=DEPLOY_ACTION_VECTOR_ACTION_SPACE,
                        source_hz=float(policy.metadata["action_hz"]),
                        source_start_index=int(args.debug_replay_start_index),
                        dataset_metadata=dataset_info,
                        target_frame_contract_hash=(
                            None
                            if deploy_session_contract.runtime_pose_frame_contract is None
                            else deploy_session_contract.runtime_pose_frame_contract.contract_hash
                        ),
                    )
                )
                print(
                    "[ReplayProposalShadow] typed passive replay adapter enabled; "
                    "DatasetReplayPolicy remains the only cursor/queue owner.",
                    flush=True,
                )
            except (KeyError, TypeError, ValueError) as exc:
                # Legacy execution stays unchanged, but an untyped replay must
                # never be mislabeled as a policy proposal in the new facts.
                print(
                    "[ReplayProposalShadow][WARN] typed replay shadow disabled: "
                    f"{exc}",
                    file=sys.stderr,
                    flush=True,
                )

    backend_control_hz = float(args.arm_control_hz)
    arm_backends = build_arm_backends(
        args,
        active_sides,
        control_hz=backend_control_hz,
        action_hz=float(effective_action_hz),
    )
    hand_backends = build_hand_backends(args, end_effector_sides)
    hand_state_poller = (
        HandStateHistoryPoller(
            backends_by_side=hand_backends,
            config=HandStateHistoryPollerConfig(
                poll_hz=max(60.0, 2.0 * float(effective_action_hz)),
            ),
        )
        if hand_backends
        else None
    )
    hand_action_specs = (
        hand_action_specs_for_backends(args, hand_backends, end_effector_sides)
        if hands_enabled
        else {}
    )
    arm_reader = ArmStateReader(
        arm_backends,
        refresh_hz=float(args.arm_state_refresh_hz),
        compute_tcp_pose=bool(args.arms_enabled),
    )
    camera_sources = build_camera_sources(
        args,
        selected_camera_keys=selected_camera_keys,
    )
    preview_camera_sources = (
        select_preview_camera_sources(camera_sources, str(args.vis_cam_key))
        if bool(args.vis_cam) and camera_sources
        else {}
    )
    camera_previewer = CameraPreviewer(CameraPreviewConfig()) if bool(args.vis_cam) and preview_camera_sources else None
    visualizer = (
        TactileUVVisualizer(
            TactileVisualizerConfig(
                enabled_sides=end_effector_sides,
                image_size=(int(args.tactile_vis_size), int(args.tactile_vis_size)),
                value_scale=float(args.tactile_vis_value_scale),
            )
        )
        if args.visualize_tactile
        else None
    )
    safety_filter = ActionSafetyFilter(
        ActionSafetyFilterConfig(
            active_sides=active_sides,
            arms_enabled=bool(args.arms_enabled),
            hands_enabled=bool(hands_enabled),
            end_effector_sides=end_effector_sides,
            workspace_min=tuple(float(value) for value in args.arm_workspace_min),
            workspace_max=tuple(float(value) for value in args.arm_workspace_max),
            max_arm_xyz_step_m=float(args.max_arm_xyz_step_m),
            max_arm_rot_step_deg=float(args.max_arm_rot_step_deg),
            max_arm_joint_step_rad=float(args.max_arm_joint_step_rad),
            max_hand_step=float(args.max_hand_step),
            arm_joint_lower_rad=tuple(float(value) for value in RM65_ABSOLUTE_Q_MIN_RAD),
            arm_joint_upper_rad=tuple(float(value) for value in RM65_ABSOLUTE_Q_MAX_RAD),
        )
    )
    canonical_action_pipeline = CanonicalDeployActionPipeline(
        safety_filter=safety_filter,
        active_sides=active_sides,
        arms_enabled=bool(args.arms_enabled),
        hands_enabled=bool(hands_enabled),
    )
    rtc_frontend: RtcPolicyShadowFrontend | None = None
    rtc_debug_observer: RtcDebugEventObserver | None = None

    def observe_rtc_transport_exchange(bundle: Any) -> None:
        observer = rtc_debug_observer
        if observer is None or not observer.accepts(RTC_DEBUG_LEVEL_SUMMARY):
            return
        observer.offer(
            "rtc_transport_exchange",
            rtc_client_inference_bundle_debug_payload(bundle),
        )

    if str(args.rtc_profile) == RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1:
        rtc_session_id = f"deploy-rtc-shadow-{uuid.uuid4()}"
        runtime_frame_hash = getattr(
            deploy_session_contract.runtime_pose_frame_contract,
            "contract_hash",
            None,
        )
        if runtime_frame_hash is None:
            runtime_frame_hash = canonical_content_hash(
                {
                    "policy_absolute_frame": (
                        deploy_session_contract.policy_absolute_frame
                    ),
                    "runtime_pose_frame_contract": None,
                },
                domain="dexteleop.rtc_3c.runtime_frame_binding.v1",
            )
        rtc_frontend = RtcPolicyShadowFrontend(
            config=RtcPolicyShadowConfig(
                runtime_session_id=rtc_session_id,
                control_domain_id="deploy-control-0",
                producer_id=f"policy:{policy.name}:rtc-shadow",
                policy_node_id=policy_deploy_runtime.config.policy_node_id,
                artifact_hash=canonical_content_hash(
                    {
                        "policy_name": policy.name,
                        "policy_metadata": policy.metadata,
                    },
                    domain="dexteleop.rtc_3c.virtual_policy_artifact.v1",
                ),
                action_codec_hash=canonical_content_hash(
                    {
                        "action_space": DEPLOY_ACTION_VECTOR_ACTION_SPACE,
                        "pose_mode": str(args.policy_action_pose_mode),
                    },
                    domain="dexteleop.rtc_3c.action_codec.v1",
                ),
                frame_contract_hash=str(runtime_frame_hash),
                layout_hash=(
                    deploy_layout.layout_hash
                    if deploy_layout is not None
                    else canonical_content_hash(
                        {
                            "active_sides": active_sides,
                            "end_effector_sides": end_effector_sides,
                            "arms_enabled": bool(args.arms_enabled),
                            "hands_enabled": hands_enabled,
                        },
                        domain="dexteleop.rtc_3c.deploy_layout.v1",
                    )
                ),
                normalization_hash=canonical_content_hash(
                    {"normalization": "runtime_absolute_none_v1"},
                    domain="dexteleop.rtc_3c.normalization.v1",
                ),
                request_validity_steps=int(args.rtc_request_validity_steps),
                worker_stop_timeout_sec=float(args.rtc_worker_stop_timeout_sec),
            ),
            deploy_runtime=policy_deploy_runtime,
            policy=policy,
            created_at=time.monotonic(),
            emission_adapter=canonical_action_pipeline.action_adapter,
        )
        if rtc_step_fanout is not None:
            rtc_step_fanout.bind_frontend(rtc_frontend)
    elif str(args.rtc_profile) == RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1:
        if rtc_model_adapter_factory is None:
            raise ValueError(
                "raw-RTC final-chunk requires a model-package adapter factory; "
                "the generic deploy runtime will not infer one from policy names."
            )
        model_adapter = rtc_model_adapter_factory(
            policy=policy,
            policy_artifact=policy_artifact,
            deploy_session_contract=deploy_session_contract,
            deploy_runtime=policy_deploy_runtime,
            args=args,
            active_sides=active_sides,
        )
        rtc_frontend = build_rtc_model_shadow_frontend(
            model_adapter,
            runtime_session_id=f"deploy-raw-rtc-{uuid.uuid4()}",
            control_domain_id="deploy-control-0",
            producer_id=f"policy:{policy.name}:raw-rtc",
            policy_node_id=policy_deploy_runtime.config.policy_node_id,
            deploy_runtime=policy_deploy_runtime,
            policy_name=policy.name,
            preparer=canonical_action_pipeline,
            created_at=time.monotonic(),
            profile=RAW_RTC_FINAL_CHUNK_PROFILE_V1,
            request_validity_steps=int(args.rtc_request_validity_steps),
            worker_stop_timeout_sec=float(args.rtc_worker_stop_timeout_sec),
            observe_bundle=observe_rtc_transport_exchange,
        )
    if rtc_emission_fanout is not None:
        if rtc_frontend is None:
            raise RuntimeError("RTC emission fanout has no RTC frontend.")
        rtc_emission_fanout.bind_frontend(rtc_frontend)
    if rtc_emission_publisher is not None:
        if rtc_frontend is None:
            raise RuntimeError("RTC emission publisher has no RTC frontend.")
        rtc_emission_publisher.bind_frontend(rtc_frontend)
    ik_action_gate = IKActionChunkGate(stable_frames_required=int(args.ik_recovery_stable_frames))
    startup_initializer = StartupInitializer(
        duration_s=float(args.init_time_s),
        source=startup_init_source,
        target_action=startup_init_action,
        active_sides=active_sides,
        arms_enabled=bool(args.arms_enabled),
        hands_enabled=bool(hands_enabled),
        arm_position_tolerance_m=float(args.init_arm_position_tolerance_m),
        arm_rotation_tolerance_deg=float(args.init_arm_rotation_tolerance_deg),
    )
    startup_initializer.reset()
    fixed_init_controller = (
        None
        if fixed_init_plan is None
        else DeployFixedInitController(
            fixed_init_plan,
            arm_max_angvel=tuple(float(value) for value in args.deploy_fixed_init_arm_max_angvel),
            qpos_tolerance=float(args.deploy_fixed_init_qpos_tolerance),
        )
    )
    policy_qpos_init_controller = (
        None
        if policy_qpos_init_plan is None
        else DeployPolicyQposInitController(
            policy_qpos_init_plan,
            active_sides=active_sides,
            settings=DeployPolicyQposInitSettings(
                qpos_reached_tolerance_rad=float(args.policy_qpos_init_qpos_tolerance),
                move_to_start_timeout_s=float(args.policy_qpos_init_timeout_s),
                qpos_init_dt_sec=float(args.policy_qpos_init_dt_sec),
                qpos_init_joint_velocity_profile=str(args.policy_qpos_init_joint_velocity_profile),
                qpos_init_max_angvel_rad_s=args.policy_qpos_init_arm_max_angvel,
                qpos_init_max_steps_per_side=int(args.policy_qpos_init_max_steps_per_side),
                qpos_init_collision_check=bool(args.policy_qpos_init_collision_check),
                qpos_init_collision_scene_profile=str(args.policy_qpos_init_collision_scene_profile),
                qpos_init_branch_search=bool(args.policy_qpos_init_branch_search),
                qpos_init_branch_search_max_branches=int(args.policy_qpos_init_branch_search_max_branches),
                qpos_init_branch_neutral_qpos_path=args.policy_qpos_init_branch_neutral_qpos_path,
            ),
        )
    )

    policy_active = (
        bool(args.auto_start)
        and fixed_init_controller is None
        and policy_qpos_init_controller is None
    )
    send_actions = bool(args.send_actions)
    wait_for_ready_confirm = (
        bool(send_actions) and fixed_init_controller is None
        if args.wait_for_ready_confirm is None
        else bool(args.wait_for_ready_confirm)
    )
    ready_confirmed = not wait_for_ready_confirm
    last_action = np.full(ACTION_DIM, np.nan, dtype=np.float32)
    step_index = 0
    loop_rate = LoopRate()
    last_status_at = 0.0
    last_fault_text: str | None = None
    started_at = time.time()
    debug_logger = DeploymentDebugLogger(
        enabled=bool(args.debug),
        root_dir=args.debug_dir,
        active_sides=active_sides,
        flush_every_events=16,
        flush_interval_sec=0.25,
    )
    if rtc_frontend is not None:
        rtc_debug_observer = RtcDebugEventObserver(
            level=str(getattr(args, "_rtc_debug_level", RTC_DEBUG_LEVEL_OFF)),
            sink=debug_logger.log_event,
            queue_capacity=int(args.rtc_debug_queue_capacity),
        )
    if rtc_debug_observer is not None and rtc_debug_observer.enabled:
        source_binding = rtc_frontend.source_binding_candidate
        rtc_debug_observer.offer(
            "rtc_source_binding_candidate_prepared",
            {
                "schema": "dexteleop_rtc_source_binding_candidate_debug_v1",
                "source_binding_hash": source_binding.semantic_content_hash,
                "source_descriptor_hash": source_binding.descriptor_hash,
                "source_slot_id": source_binding.source_slot_id,
                "producer_id": source_binding.producer_id,
                "producer_epoch": source_binding.producer_epoch,
                "binding_generation": source_binding.binding_generation,
                "runtime_session_id": source_binding.runtime_session_id,
                "control_domain_id": source_binding.control_domain_id,
            },
        )
        if (
            rtc_emission_publisher is not None
            and rtc_emission_publisher.active_binding is not None
        ):
            rtc_debug_observer.offer(
                "rtc_source_binding_activated",
                {
                    "schema": (
                        "dexteleop_rtc_source_binding_activation_debug_v1"
                    ),
                    "source_binding_hash": (
                        rtc_emission_publisher.active_binding.semantic_content_hash
                    ),
                    "source_descriptor_hash": (
                        rtc_emission_publisher.active_binding.descriptor_hash
                    ),
                    "producer_id": (
                        rtc_emission_publisher.active_binding.producer_id
                    ),
                    "producer_epoch": (
                        rtc_emission_publisher.active_binding.producer_epoch
                    ),
                    "binding_generation": (
                        rtc_emission_publisher.active_binding.binding_generation
                    ),
                    "activation_owner": "application_publisher",
                },
            )
    position_deploy_executor: PositionDeployExecutor | None = None
    if (
        bool(args.arms_enabled)
        and normalize_rm65_arm_control_mode(args.arm_control_mode)
        == RM65_ARM_CONTROL_MODE_POSITION
    ):
        position_collision_required = bool(
            args.position_collision_check and active_sides == ("left", "right")
        )
        position_reference_hz = effective_position_reference_hz(args)
        position_command_hz = (
            float(backend_control_hz)
            if effective_arm_canfd_follow(args)
            else float(position_reference_hz)
        )
        arm_only_pipeline = CanonicalDeployActionPipeline(
            safety_filter=safety_filter,
            active_sides=active_sides,
            arms_enabled=True,
            hands_enabled=False,
        )

        def dispatch_position_reference(command: TeleopCommand) -> Mapping[str, Any]:
            annotate_deploy_arm_command_limits(
                command,
                args=args,
                active_sides=active_sides,
                control_hz=float(backend_control_hz),
                action_hz=float(position_reference_hz),
            )
            require_arm_backends_ready_for_position_dispatch(
                arm_backends,
                require_background_target=False,
                stage="pre_arm_reference_queue",
                required_sides=active_sides,
            )
            dispatch_result = dispatch_teleop_command(
                command,
                arm_backends=arm_backends,
                hand_backends={},
                collect_backend_evidence=True,
            )
            dispatch_result["arm_backend_health"] = (
                require_arm_backends_ready_for_position_dispatch(
                    arm_backends,
                    require_background_target=True,
                    stage="post_arm_reference_queue",
                    required_sides=active_sides,
                )
            )
            return dispatch_result

        def dispatch_position_discrete(command: TeleopCommand) -> Mapping[str, Any]:
            # Recheck after the arm target was queued and immediately before
            # any scheduled hand packet.  This catches a background stream fault
            # that appeared in the arm->hand worker gap and makes the executor
            # latch/hold instead of tail-sending the hand keyframe.
            require_arm_backends_ready_for_position_dispatch(
                arm_backends,
                require_background_target=True,
                stage="pre_discrete_hand_dispatch",
                required_sides=active_sides,
            )
            return _dispatch_position_discrete_command_checked(
                command,
                hand_backends=hand_backends,
            )

        position_worker_binding = getattr(args, "_rm65_legacy_worker_binding", None)
        if position_worker_binding is not None and not isinstance(
            position_worker_binding,
            RM65LegacyWorkerExecutionBinding,
        ):
            raise TypeError("_rm65_legacy_worker_binding has an unsupported type")
        position_deploy_executor = PositionDeployExecutor(
            config=PositionDeployExecutorConfig(
                sample_hz=float(effective_action_hz),
                reference_hz=float(position_reference_hz),
                command_hz=float(position_command_hz),
                stale_after_sec=max(0.25, 3.0 / float(effective_action_hz)),
                queue_size=max(32, int(args.min_action_queue) * 4),
                debug_events_enabled=bool(debug_logger.enabled),
                collision_required=position_collision_required,
                allow_non_authoritative_collision_reset_in_dry_run=not bool(
                    send_actions
                ),
                chunk_lookahead_enabled=bool(args.position_chunk_lookahead),
            ),
            canonical_pipeline=arm_only_pipeline,
            dispatch_callback=(dispatch_position_reference if send_actions else None),
            discrete_dispatch_callback=(
                dispatch_position_discrete if send_actions and hands_enabled else None
            ),
            hold_callback=(
                (
                    lambda: hold_deploy_arm_backends(
                        arm_backends,
                        reason="position deploy executor pause/fault",
                    )
                )
                if send_actions
                else None
            ),
            collision_planner_factory=(
                (
                    lambda: build_position_collision_planner_from_started_backends(
                        arm_backends,
                        reference_hz=float(position_reference_hz),
                        scene_profile=str(args.position_collision_scene_profile),
                        collision_min_distance_m=float(
                            args.position_collision_min_distance_m
                        ),
                        collision_distance_epsilon_m=float(
                            args.position_collision_distance_epsilon_m
                        ),
                        max_joint_velocity_rad_s=tuple(
                            float(value)
                            for value in (
                                args.arm_max_angvel
                                if args.position_qref_max_angvel is None
                                else args.position_qref_max_angvel
                            )
                        ),
                        max_joint_step_rad=(
                            None
                            if float(args.max_arm_joint_step_rad) <= 0.0
                            else float(args.max_arm_joint_step_rad)
                        ),
                        damping=float(args.position_ik_damping),
                        solve_max_iterations=int(
                            args.position_ik_solve_max_iterations
                        ),
                        solve_min_iterations=int(
                            args.position_ik_solve_min_iterations
                        ),
                        solve_adaptive_iterations_enabled=bool(
                            args.position_ik_adaptive_iterations
                        ),
                        solve_iteration_reserve_ms=float(
                            args.position_ik_iteration_reserve_ms
                        ),
                        debug_profile=bool(args.position_collision_debug_profile),
                        ik_tuning=_position_deploy_ik_tuning(args),
                    )
                )
                if position_collision_required
                else None
            ),
            **(
                {}
                if position_worker_binding is None
                else position_worker_binding.position_executor_kwargs()
            ),
        )

    def flush_position_executor_debug_events() -> None:
        if position_deploy_executor is None:
            return
        for raw_event in position_deploy_executor.drain_debug_events():
            event = dict(raw_event)
            event_name = str(event.pop("event", "unknown"))
            debug_logger.log_event(f"position_executor_{event_name}", event)

    def execute_prepared_action(
        canonical_action: Any,
        observation: Any,
        *,
        policy_name: str,
        lookahead_canonical_action: Any | None = None,
        lookahead_observation: Any | None = None,
    ) -> _PreparedActionExecution:
        command = getattr(canonical_action, "command", None)
        if command is None or not command.commands:
            return _PreparedActionExecution(
                accepted=False,
                mode="no_command",
                dispatch_attempted=False,
                dispatch_completed=False,
            )
        if position_deploy_executor is None:
            if send_actions:
                dispatch_teleop_command(
                    command,
                    arm_backends=arm_backends,
                    hand_backends=hand_backends,
                )
            return _PreparedActionExecution(
                accepted=True,
                mode="direct" if send_actions else "dry_run",
                dispatch_attempted=bool(send_actions),
                dispatch_completed=bool(send_actions),
            )

        arm_command = command_effector_kind_subset(command, kind="arm")
        hand_command = command_effector_kind_subset(command, kind="hand")
        lookahead_hand_command = None
        if lookahead_canonical_action is not None:
            lookahead_hand_command = command_effector_kind_subset(
                getattr(lookahead_canonical_action, "command", None),
                kind="hand",
            )
        if arm_command is not None:
            try:
                source_index = position_deploy_executor.submit(
                    canonical_action.filtered,
                    observation,
                    policy_name=str(policy_name),
                    discrete_command=hand_command,
                    lookahead_filtered=(
                        None
                        if lookahead_canonical_action is None
                        else lookahead_canonical_action.filtered
                    ),
                    lookahead_observation=lookahead_observation,
                    lookahead_discrete_command=lookahead_hand_command,
                )
            except PositionDeployExecutorFault:
                fault_stage = str(
                    position_deploy_executor.snapshot().get("fault_stage") or ""
                )
                if fault_stage in {
                    "safety_gate",
                    "15hz_keyframe_safety",
                    "reference_anchor_safety",
                }:
                    return _PreparedActionExecution(
                        accepted=False,
                        mode="position_safety_fault_latched",
                        dispatch_attempted=False,
                        dispatch_completed=False,
                    )
                raise
            return _PreparedActionExecution(
                accepted=True,
                mode=("position_async_queue" if send_actions else "position_dry_run_queue"),
                dispatch_attempted=False,
                dispatch_completed=False,
                asynchronous=True,
                position_source_index=int(source_index),
            )
        elif send_actions and hand_command is not None:
            # A hand-only command has no arm keyframe whose causal reference
            # clock could carry it. Mixed arm+hand actions are queued together
            # above so both become effective on the same logical reference tick.
            try:
                require_arm_backends_ready_for_position_dispatch(
                    arm_backends,
                    require_background_target=False,
                    stage="pre_direct_hand_only_dispatch",
                    required_sides=active_sides,
                )
                dispatch_teleop_command(
                    hand_command,
                    arm_backends={},
                    hand_backends=hand_backends,
                )
            except Exception:
                position_deploy_executor.clear()
                raise
            return _PreparedActionExecution(
                accepted=True,
                mode="position_hand_only_direct",
                dispatch_attempted=True,
                dispatch_completed=True,
            )
        return _PreparedActionExecution(
            accepted=bool(hand_command is not None),
            mode="position_hand_only_dry_run",
            dispatch_attempted=False,
            dispatch_completed=False,
        )

    def latch_canonical_safety_fault(
        canonical_action: Any,
        observation: Any,
        *,
        policy_name: str,
    ) -> None:
        """Turn a blocked 15 Hz canonical action into a real hold boundary."""

        canonical_faults = tuple(str(item) for item in getattr(canonical_action, "faults", ()))
        if not canonical_faults:
            return
        if position_deploy_executor is not None:
            position_deploy_executor.latch_safety_fault(
                canonical_faults,
                observation=observation,
                stage="15hz_keyframe_safety",
            )
            return
        if send_actions:
            hold_deploy_arm_backends(
                arm_backends,
                reason=(
                    f"15 Hz deploy safety fault ({policy_name}): "
                    + ", ".join(canonical_faults)
                ),
            )

    impedance_force_control_is_deferred = bool(
        send_actions
        and bool(args.arms_enabled)
        and normalize_rm65_arm_control_mode(args.arm_control_mode)
        == RM65_ARM_CONTROL_MODE_IMPEDANCE
    )
    impedance_force_control_enabled = False

    def enable_deferred_impedance_force_control(*, trigger: str) -> None:
        """Start impedance output only at an explicit policy/init start boundary."""

        nonlocal impedance_force_control_enabled
        if not impedance_force_control_is_deferred or impedance_force_control_enabled:
            return
        enable_impedance_force_control_backends(arm_backends, ready_timeout_sec=1.0)
        impedance_force_control_enabled = True
        print(
            "[Deploy] impedance periodic output enabled "
            f"after explicit start trigger={trigger}.",
            flush=True,
        )

    def hold_rtc_if_eligible(*, now: float, reason: str) -> None:
        if (
            rtc_frontend is not None
            and rtc_frontend.state.lifecycle_state
            == POLICY_ASYNC_PRODUCER_ELIGIBLE
        ):
            transition = rtc_frontend.hold(now=now, reason=reason)
            debug_logger.log_event(
                "rtc_lifecycle_transition",
                {
                    "target": "HOLD",
                    "reason": reason,
                    "producer_transition_hash": (
                        transition.producer_transition.receipt.semantic_content_hash
                    ),
                    "worker_stop_hash": (
                        transition.worker_stop_receipt.semantic_content_hash
                    ),
                },
            )

    def rearm_rtc_if_needed(*, now: float, reason: str) -> bool:
        if rtc_frontend is None:
            return True
        if (
            rtc_frontend.state.lifecycle_state
            == POLICY_ASYNC_PRODUCER_ELIGIBLE
        ):
            return True
        try:
            transition = rtc_frontend.rearm(now=now, reason=reason)
        except RuntimeError as exc:
            print(f"[RTC-3c] rearm deferred: {exc}", flush=True)
            return False
        debug_logger.log_event(
            "rtc_lifecycle_transition",
            {
                "target": "ELIGIBLE",
                "reason": reason,
                "producer_transition_hash": (
                    transition.receipt.semantic_content_hash
                ),
            },
        )
        return True

    try:
        print(
            "[Deploy] policy="
            f"{policy.name} active_sides={active_sides} "
            f"arms={bool(args.arms_enabled)} hands={hands_enabled} "
            "hand_backends="
            f"{ {side: _hand_backend_mode_for_side(args, side) for side in end_effector_sides} } "
            f"arm_profile={args._real_arm_execution_config.resolved_profile} "
            f"arm_control_mode={args.arm_control_mode} "
            f"arm_canfd_follow={bool(args.arm_canfd_follow)} "
            f"hz={float(effective_action_hz):.2f} "
            f"policy_hz={float(effective_policy_hz):.2f} "
            f"backend_hz={backend_control_hz:.2f} "
            f"queue_policy={action_queue_policy} "
            f"rtc_profile={args.rtc_profile}"
        )
        if action_hz_info.get("source") != "cli.hz":
            print(
                "[Deploy] action_hz="
                f"{float(effective_action_hz):.3f} source={action_hz_info.get('source')} "
                f"policy_hz_source={policy_hz_source}"
            )
        if position_deploy_executor is not None:
            executor_config = position_deploy_executor.config
            collision_reset_label = (
                ("fresh_dual_hardware" if send_actions else "dry_run_state_allowed")
                if executor_config.collision_required
                else "disabled"
            )
            print(
                "[PositionDeploy] chain="
                f"safe_{executor_config.sample_hz:g}hz"
                f"->causal_{executor_config.reference_hz:g}hz"
                f"->{executor_config.command_hz:g}hz "
                "delay=1_action_period interpolation=xyz_linear+shortest_slerp "
                "temporal_smoothing=off "
                f"collision_reference={'on' if executor_config.collision_required else 'off(dry-run)'} "
                "collision_scope=selected_scene_endpoint_only "
                "segment_reference_to_command_certified=false "
                "collision_reset="
                f"{collision_reset_label} "
                f"collision_reset_max_age_s={executor_config.collision_reset_measurement_max_age_sec:g} "
                f"collision_reset_pair_skew_s={executor_config.collision_reset_pair_max_skew_sec:g}"
            )
        if policy.metadata.get("debug_replay"):
            print(
                "[Replay] source="
                f"{policy.metadata.get('source_path')} frames={policy.metadata.get('frame_count')} "
                f"chunk_size={int(args.debug_replay_chunk_size)} loop={bool(args.debug_replay_loop)} "
                f"action_hz={policy.metadata.get('action_hz')}"
            )
        if fixed_init_plan is not None:
            print(
                "[FixedInit] source="
                f"{fixed_init_plan.source_path} "
                f"episode={fixed_init_plan.source_episode_index} "
                f"frames={fixed_init_plan.frame_count} "
                f"range=[{fixed_init_plan.start_index},{fixed_init_plan.end_index}) "
                f"hz={fixed_init_plan.action_hz:.3f} "
                f"after=T_then_startup_init:{startup_init_source}"
            )
        if policy_qpos_init_plan is not None:
            print(
                "[PolicyQposInit] source="
                f"{policy_qpos_init_plan.source_path} "
                f"episode={policy_qpos_init_plan.source_episode_index} "
                f"frame={policy_qpos_init_plan.frame_index} "
                f"qpos_source={policy_qpos_init_plan.metadata.get('qpos_init_source')} "
                f"targets={policy_qpos_init_plan.target_sources_by_side} "
                f"after=startup_init:{startup_init_source}"
            )
        elif bool(args.start_init_qpos) and bool(args.send_actions) and bool(args.arms_enabled):
            print(
                "[Deploy] start_init_qpos override enabled: "
                "real arm policy execution may start from current qpos without dataset-frame qpos init."
            )
        if not send_actions:
            print("[Deploy] dry-run: actions are filtered and logged but not sent to hardware.")
        if (
            send_actions
            and hands_enabled
        ):
            missing_hand_sides = [
                side
                for side in end_effector_sides
                if _hand_backend_mode_for_side(args, side) == HAND_BACKEND_DMTAC_GRIPPER
                and side not in hand_backends
            ]
            if missing_hand_sides:
                raise RuntimeError(
                    "DMTAC gripper backend config is missing for active side(s): "
                    + ",".join(missing_hand_sides)
                )
        debug_logger.log_startup(
            args=args,
            policy=policy,
            active_sides=active_sides,
            hands_enabled=hands_enabled,
            effective_action_hz=float(effective_action_hz),
            action_hz_info=action_hz_info,
            effective_policy_hz=float(effective_policy_hz),
            policy_hz_source=policy_hz_source,
            backend_control_hz=backend_control_hz,
            action_queue_policy=action_queue_policy,
            send_actions=send_actions,
            arm_backends=arm_backends,
        )

        for side, backend in arm_backends.items():
            backend.start()
            if not backend.is_ready() and not bool(args.allow_arm_connect_failure):
                raise RuntimeError(f"{side} arm backend is not ready: {backend.get_state()}")
        if position_deploy_executor is not None:
            position_deploy_executor.prepare_collision_planner()
            position_deploy_executor.start()
        if impedance_force_control_is_deferred:
            warmed_sides = warm_start_impedance_force_control_backends(arm_backends)
            if warmed_sides:
                print(
                    "[Deploy] impedance prepared without periodic output; "
                    "press g/H/A to enable it, or use --auto-start.",
                    flush=True,
                )
        for side, backend in hand_backends.items():
            backend.start()
            if not backend.is_ready() and not bool(args.allow_hand_connect_failure):
                raise RuntimeError(f"{side} hand backend is not ready: {backend.get_state()}")
        if hand_state_poller is not None:
            hand_state_poller.start()
        if camera_sources:
            print(
                "[Camera] configured "
                f"cameras={sorted(camera_sources)} mode={args.camera_mode} "
                f"vis_cam={bool(args.vis_cam)}"
            )
            if camera_previewer is not None:
                if camera_previewer.ensure_available():
                    print("[Camera] --vis_cam display preflight ok.")
                else:
                    print(f"[Camera] preview disabled: {camera_previewer.disabled_reason}")
        start_camera_sources(
            camera_sources,
            stagger_sec=float(args.camera_start_stagger_sec),
        )
        if rtc_frontend is not None:
            rtc_frontend.start()
            if str(args.rtc_profile) == RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1:
                print(
                    "[raw-RTC] OpenPI/model final-chunk shadow active: "
                    "send_actions=false, position_lookahead=false, "
                    "timed-plan admission enabled."
                    + (
                        " live observation-only hardware guard enabled."
                        if bool(getattr(args, "rtc_real_shadow", False))
                        else ""
                    ),
                    flush=True,
                )
            else:
                print(
                    "[RTC-3c] virtual-policy shadow active: send_actions=false, "
                    "position_lookahead=false, timed-plan admission enabled.",
                    flush=True,
                )
        print_controls(
            send_actions=send_actions,
            auto_start=bool(args.auto_start),
            wait_for_ready_confirm=wait_for_ready_confirm,
            fixed_init_enabled=fixed_init_controller is not None,
            policy_qpos_init_enabled=policy_qpos_init_controller is not None,
        )
        if policy_qpos_init_controller is not None and bool(args.auto_start):
            enable_deferred_impedance_force_control(trigger="auto_start_policy_qpos_init")
            message = policy_qpos_init_controller.begin(
                arm_backends=arm_backends,
                now=time.time(),
            )
            if message:
                print(message)
        elif bool(args.auto_start) and fixed_init_controller is None:
            enable_deferred_impedance_force_control(trigger="auto_start")

        with NonBlockingKeyboard() as keyboard:
            while True:
                loop_started_at = time.time()
                control_loop_started_at = time.monotonic()
                position_safety_fault_latched = False
                if position_deploy_executor is not None:
                    flush_position_executor_debug_events()
                    executor_fault = position_deploy_executor.snapshot()
                    if bool(executor_fault.get("fault_latched")):
                        if str(executor_fault.get("fault_stage") or "") in {
                            "safety_gate",
                            "15hz_keyframe_safety",
                            "reference_anchor_safety",
                        }:
                            position_safety_fault_latched = True
                        else:
                            position_deploy_executor.raise_if_fault()
                key = keyboard.poll()
                preview_keys: list[str] = []
                if camera_previewer is not None:
                    preview_keys = camera_previewer.update(preview_camera_sources)
                if preview_keys and key is None:
                    key = preview_keys[0]
                key = None if key is None else key.lower()
                rearm_requested = bool(key == "r")
                if rearm_requested and not position_safety_fault_latched:
                    print("[Deploy] r ignored: no manually re-armable safety fault is latched.")
                elif rearm_requested:
                    policy_active = False
                    policy_deploy_runtime.clear_action_queue()
                elif key == "g":
                    if position_safety_fault_latched:
                        policy_active = False
                        print(
                            "[Deploy] g ignored: a safety fault is latched; press r only after "
                            "fresh in-domain measured feedback is available."
                        )
                        continue
                    if position_deploy_executor is not None:
                        position_deploy_executor.clear()
                        last_action = np.full(ACTION_DIM, np.nan, dtype=np.float32)
                    if not rearm_rtc_if_needed(
                        now=control_loop_started_at,
                        reason="operator_g",
                    ):
                        policy_active = False
                        continue
                    policy.reset()
                    policy_deploy_runtime.clear_action_queue()
                    startup_initializer.reset()
                    policy_deploy_runtime.reset_prediction_schedule()
                    ready_confirmed = not wait_for_ready_confirm
                    if fixed_init_controller is not None and fixed_init_controller.blocks_policy:
                        print("[Deploy] ignore g: complete fixed-init H -> H before starting policy.")
                    elif policy_qpos_init_controller is not None and policy_qpos_init_controller.blocks_policy:
                        policy_active = False
                        enable_deferred_impedance_force_control(trigger="g_policy_qpos_init")
                        message = policy_qpos_init_controller.begin(
                            arm_backends=arm_backends,
                            now=loop_started_at,
                        )
                        if message:
                            print(message)
                    else:
                        enable_deferred_impedance_force_control(trigger="g_policy")
                        policy_active = True
                        print("[Deploy] policy active")
                elif key == "a":
                    if fixed_init_controller is not None and fixed_init_controller.blocks_policy:
                        print("[Deploy] ready confirm ignored: complete fixed-init H -> H first.")
                    elif policy_qpos_init_controller is not None and policy_qpos_init_controller.blocks_policy:
                        print("[Deploy] ready confirm ignored: complete policy qpos init first (press g).")
                    else:
                        if not rearm_rtc_if_needed(
                            now=control_loop_started_at,
                            reason="operator_ready_confirm",
                        ):
                            policy_active = False
                            continue
                        if not policy_active:
                            policy_active = True
                        enable_deferred_impedance_force_control(trigger="A_policy_or_ready_confirm")
                        if startup_initializer.enabled and not startup_initializer.completed:
                            print("[Deploy] ready confirm ignored: startup init is not complete yet.")
                        else:
                            ready_confirmed = True
                            policy_deploy_runtime.clear_action_queue()
                            policy_deploy_runtime.reset_prediction_schedule()
                            print("[Deploy] ready confirmed; policy inference/execution may start.")
                elif key == "h":
                    hold_rtc_if_eligible(
                        now=control_loop_started_at,
                        reason="operator_hold",
                    )
                    if position_deploy_executor is not None:
                        position_deploy_executor.clear()
                        last_action = np.full(ACTION_DIM, np.nan, dtype=np.float32)
                    policy_deploy_runtime.clear_action_queue()
                    ready_confirmed = not wait_for_ready_confirm
                    if policy_active:
                        policy_active = False
                        print("[Deploy] policy paused; queue cleared")
                    elif fixed_init_controller is not None:
                        enable_deferred_impedance_force_control(trigger="H_fixed_init")
                        message = fixed_init_controller.handle_h(
                            arm_backends=arm_backends,
                            now=loop_started_at,
                        )
                        if message:
                            print(message)
                    elif policy_qpos_init_controller is not None and policy_qpos_init_controller.running:
                        policy_active = False
                        print("[PolicyQposInit] ignore h while qpos init is running.")
                    else:
                        policy_active = False
                        print("[Deploy] policy paused")
                elif key == "x" and fixed_init_controller is not None and fixed_init_controller.running:
                    print("[FixedInit] ignore x while fixed trajectory is running.")
                elif key == "x" and policy_qpos_init_controller is not None and policy_qpos_init_controller.running:
                    print("[PolicyQposInit] ignore x while qpos init is running.")
                elif key == "q":
                    if position_deploy_executor is not None:
                        position_deploy_executor.clear()
                    print("\n[Deploy] quitting policy deployment loop\n")
                    break

                arm_states = arm_reader.snapshot(loop_started_at)
                ik_gate_event = ik_action_gate.update(arm_states)
                if ik_action_gate.blocks_action_output:
                    policy_deploy_runtime.clear_action_queue()
                    hold_rtc_if_eligible(
                        now=control_loop_started_at,
                        reason="ik_action_gate_blocked",
                    )
                if ik_gate_event:
                    policy_deploy_runtime.reset_prediction_schedule()
                    print(f"[Deploy] {ik_gate_event}; action queue cleared")
                hand_states = hand_state_snapshot(
                    hand_backends,
                    history_poller=hand_state_poller,
                    timestamp=loop_started_at,
                )
                control_hz = loop_rate.tick(loop_started_at)
                observation = build_deployment_observation(
                    active_sides=active_sides,
                    arms_enabled=bool(args.arms_enabled),
                    hands_enabled=hands_enabled,
                    end_effector_sides=end_effector_sides,
                    step_index=step_index,
                    arm_states=arm_states,
                    hand_states=hand_states,
                    camera_sources=camera_sources,
                    last_action=last_action,
                    timestamp=loop_started_at,
                    require_images=bool(args.require_camera_frame),
                    camera_frame_max_age_sec=float(args.camera_frame_max_age_sec),
                    hand_state_max_age_sec=DEFAULT_HAND_STATE_MAX_AGE_SEC,
                    metadata={
                        "control_hz": control_hz,
                        "causal_snapshot_contract": (
                            "source_timestamp<=T0_and_available_timestamp<=T0"
                        ),
                        "causal_snapshot_cutoff_timestamp": float(loop_started_at),
                        "policy_active": policy_active,
                        "send_actions": send_actions,
                        "policy_absolute_frame": deploy_session_contract.policy_absolute_frame,
                        "runtime_pose_frame_contract_hash": (
                            None
                            if deploy_session_contract.runtime_pose_frame_contract is None
                            else deploy_session_contract.runtime_pose_frame_contract.contract_hash
                        ),
                        "hand_action_specs_by_side": hand_action_specs,
                        "hand_active_mask_by_side": {
                            side: list(spec.get("active_mask", ()))
                            for side, spec in hand_action_specs.items()
                        },
                    },
                )
                faults = list(observation.faults)
                if position_safety_fault_latched:
                    policy_active = False
                    policy_deploy_runtime.clear_action_queue()
                    hold_rtc_if_eligible(
                        now=control_loop_started_at,
                        reason="position_safety_fault_latched",
                    )
                    faults.append("position_safety_fault_manual_rearm_required")
                    if rearm_requested and position_deploy_executor is not None:
                        try:
                            position_deploy_executor.manual_rearm(observation)
                        except Exception as exc:  # noqa: BLE001 - operator-facing re-arm gate.
                            print(
                                "[Deploy] manual re-arm rejected; re-home is required if feedback is "
                                f"unavailable/out of domain: {type(exc).__name__}: {exc}"
                            )
                        else:
                            last_action = np.full(ACTION_DIM, np.nan, dtype=np.float32)
                            policy.reset()
                            startup_initializer.reset()
                            policy_deploy_runtime.reset_prediction_schedule()
                            ready_confirmed = not wait_for_ready_confirm
                            faults.append("position_safety_rearmed_press_g_to_resume")
                            print("[Deploy] safety fault re-armed; press g to begin a fresh policy stream.")
                if loop_started_at - started_at < float(args.startup_warmup_sec):
                    faults.append("startup_warmup")

                if visualizer is not None:
                    visualizer.update(tactile_by_side_from_hand_states(active_sides, hand_states))

                fixed_completed_this_step = False
                if fixed_init_controller is not None:
                    fixed_result = fixed_init_controller.step(
                        now=loop_started_at,
                        send_actions=send_actions,
                        arm_backends=arm_backends,
                        hand_backends=hand_backends,
                    )
                    if fixed_result.last_action is not None:
                        last_action = fixed_result.last_action.astype(np.float32, copy=True)
                    if fixed_result.message:
                        print(fixed_result.message)
                    if fixed_result.completed:
                        policy.reset()
                        policy_deploy_runtime.clear_action_queue()
                        startup_initializer.reset()
                        policy_deploy_runtime.reset_prediction_schedule()
                        ready_confirmed = not wait_for_ready_confirm
                        if bool(args.auto_start):
                            policy_active = True
                            print("[Deploy] fixed-init complete; policy auto-start released.")
                        else:
                            policy_active = False
                            print("[Deploy] fixed-init complete; press g to run startup init/policy.")
                        fixed_completed_this_step = True

                policy_qpos_completed_this_step = False
                if policy_qpos_init_controller is not None:
                    policy_qpos_result = policy_qpos_init_controller.step(
                        now=loop_started_at,
                        send_actions=send_actions,
                        arm_backends=arm_backends,
                    )
                    if policy_qpos_result.last_action is not None:
                        last_action = policy_qpos_result.last_action.astype(np.float32, copy=True)
                    if policy_qpos_result.message:
                        print(policy_qpos_result.message)
                    if policy_qpos_result.completed:
                        policy.reset()
                        policy_deploy_runtime.clear_action_queue()
                        startup_initializer.reset()
                        policy_deploy_runtime.reset_prediction_schedule()
                        ready_confirmed = not wait_for_ready_confirm
                        policy_active = True
                        policy_qpos_completed_this_step = True
                        print("[Deploy] policy qpos init complete; startup init/policy released.")

                if policy_active:
                    position_action_submitted_this_loop = False
                    can_run_policy = not (bool(args.require_ready_observation) and faults)
                    if _debug_replay_execution_exhausted(
                        policy,
                        queued_actions=policy_deploy_runtime.action_queue_length(),
                    ):
                        can_run_policy = False
                    if fixed_completed_this_step:
                        can_run_policy = False
                    if policy_qpos_completed_this_step:
                        can_run_policy = False
                    if fixed_init_controller is not None and fixed_init_controller.blocks_policy:
                        can_run_policy = False
                    if policy_qpos_init_controller is not None and policy_qpos_init_controller.blocks_policy:
                        can_run_policy = False
                    if startup_initializer.enabled and not startup_initializer.completed:
                        if can_run_policy:
                            init_action, init_faults, init_ready = startup_initializer.next_action(
                                observation,
                                now=loop_started_at,
                            )
                            faults.extend(init_faults)
                            if init_action is not None:
                                canonical_action = canonical_action_pipeline.prepare(
                                    init_action,
                                    observation,
                                    policy_name="startup_init",
                                )
                                faults.extend(canonical_action.faults)
                                latch_canonical_safety_fault(
                                    canonical_action,
                                    observation,
                                    policy_name="startup_init",
                                )
                                if (
                                    startup_initializer.progress >= 1.0
                                    and not canonical_action.faults
                                ):
                                    startup_initializer.update_resolved_target_action(
                                        canonical_action.action.vector
                                    )
                                if canonical_action.command is not None and canonical_action.command.commands:
                                    execution = execute_prepared_action(
                                        canonical_action,
                                        observation,
                                        policy_name="startup_init",
                                    )
                                    if execution.accepted:
                                        position_action_submitted_this_loop = (
                                            position_deploy_executor is not None
                                        )
                                        last_action = canonical_action.action.vector.astype(np.float32, copy=True)
                                if init_ready and not canonical_action.faults:
                                    startup_initializer.mark_complete()
                                    if startup_initializer.source in {"replay:first_frame", "policy:first_frame"}:
                                        skipped = _skip_startup_replay_frames(policy, count=1)
                                        if skipped:
                                            print(
                                                "[StartupInit] replay startup target consumed; "
                                                f"skipped {skipped} replay frame(s)."
                                            )
                                    policy_deploy_runtime.clear_action_queue()
                                    policy_deploy_runtime.reset_prediction_schedule()
                                    if wait_for_ready_confirm and not ready_confirmed:
                                        print(
                                            "[StartupInit] ready; press A to start policy inference/execution."
                                        )
                        can_run_policy = False
                    if can_run_policy and wait_for_ready_confirm and not ready_confirmed:
                        can_run_policy = False
                    if (
                        rtc_frontend is None
                        and can_run_policy
                        and policy_deploy_runtime.should_predict(
                            now=control_loop_started_at,
                            policy_interval_s=1.0 / float(effective_policy_hz),
                            min_action_queue=int(args.min_action_queue),
                        )
                    ):
                        prediction = policy_deploy_runtime.predict_and_queue(
                            policy,
                            observation,
                            now=control_loop_started_at,
                            queue_policy=action_queue_policy,
                            discard_policy_output=ik_action_gate.discards_policy_output,
                            action_keyframe_period_sec=(
                                1.0 / float(effective_action_hz)
                            ),
                        )
                        queue_len_before_update = prediction.queue_len_before
                        if prediction.accepted:
                            ik_action_gate.mark_policy_output_accepted()
                            debug_logger.log_policy_chunk(
                                timestamp=loop_started_at,
                                step_index=step_index,
                                policy_name=policy.name,
                                policy_output=prediction.policy_output,
                                actions=list(prediction.actions),
                                accepted=True,
                                discarded_by_ik=False,
                                queue_policy=action_queue_policy,
                                queue_len_before=queue_len_before_update,
                                queue_len_after=prediction.queue_len_after,
                                faults=faults,
                                ik_recovery_statuses=ik_action_gate.last_statuses,
                                ik_recovery_diagnostics=ik_action_gate.last_diagnostics,
                            )
                        elif prediction.discarded_by_gate:
                            print("[Deploy] IK recovery active; discarded predicted action chunk.")
                            debug_logger.log_policy_chunk(
                                timestamp=loop_started_at,
                                step_index=step_index,
                                policy_name=policy.name,
                                policy_output=prediction.policy_output,
                                actions=list(prediction.actions),
                                accepted=False,
                                discarded_by_ik=True,
                                queue_policy=action_queue_policy,
                                queue_len_before=queue_len_before_update,
                                queue_len_after=prediction.queue_len_after,
                                faults=faults,
                                ik_recovery_statuses=ik_action_gate.last_statuses,
                                ik_recovery_diagnostics=ik_action_gate.last_diagnostics,
                            )
                        elif prediction.error is None:
                            faults.extend(prediction.faults)
                            debug_logger.log_policy_chunk(
                                timestamp=loop_started_at,
                                step_index=step_index,
                                policy_name=policy.name,
                                policy_output=prediction.policy_output,
                                actions=[],
                                accepted=False,
                                discarded_by_ik=False,
                                queue_policy=action_queue_policy,
                                queue_len_before=queue_len_before_update,
                                queue_len_after=prediction.queue_len_after,
                                faults=faults,
                                ik_recovery_statuses=ik_action_gate.last_statuses,
                                ik_recovery_diagnostics=ik_action_gate.last_diagnostics,
                            )
                        else:
                            faults.extend(prediction.faults)
                            if prediction.rollback_error is not None:
                                rollback_error = prediction.rollback_error
                                print(
                                    "[Deploy] policy prediction rollback error: "
                                    f"{type(rollback_error).__name__}: {rollback_error}"
                                )
                            exc = prediction.error
                            print(f"[Deploy] policy error: {type(exc).__name__}: {exc}")
                            faults.append(f"policy_error:{type(exc).__name__}")
                            debug_logger.log_policy_error(
                                timestamp=loop_started_at,
                                step_index=step_index,
                                policy_name=policy.name,
                                exc=exc,
                                queue_len_before=queue_len_before_update,
                            )
                            if bool(args.pause_on_policy_error):
                                policy_active = False
                                if position_deploy_executor is not None:
                                    position_deploy_executor.clear()
                                    last_action = observation.hold_action()

                    # An empty policy queue is intentionally a no-op.  The
                    # impedance backend already keeps its last safe command;
                    # synthesising the current measured TCP here would
                    # re-anchor the Cartesian reference on every gap.
                    allow_policy_output = (
                        can_run_policy and not ik_action_gate.blocks_action_output
                    )
                    if policy_arbiter_shadow is not None:
                        shadow_gate_status = (
                            "fault"
                            if position_safety_fault_latched
                            else "ready"
                            if allow_policy_output
                            else "hold"
                        )
                        policy_arbiter_shadow.update_execution_gate(
                            shadow_gate_status,
                            reason=(
                                "legacy_output_eligible"
                                if shadow_gate_status == "ready"
                                else "legacy_safety_fault_latched"
                                if shadow_gate_status == "fault"
                                else "legacy_output_not_eligible"
                            ),
                            produced_at=control_loop_started_at,
                        )
                    if rtc_frontend is None:
                        action_step = policy_deploy_runtime.prepare_next_action(
                            observation,
                            preparer=canonical_action_pipeline,
                            policy_name=policy.name,
                            allow_output=allow_policy_output,
                        )
                    else:
                        rtc_step = rtc_frontend.step(
                            observation,
                            now=control_loop_started_at,
                            allow_output=allow_policy_output,
                        )
                        if rtc_step_fanout is not None:
                            rtc_step_fanout.put_nowait(rtc_step)
                        emission = rtc_step.emission_result.emission
                        if (
                            rtc_emission_publisher is not None
                            and emission is not None
                        ):
                            publication_receipt = (
                                rtc_emission_publisher.publish_committed_emission(
                                    emission,
                                    rtc_step.emission_commit_receipt,
                                    now=control_loop_started_at,
                                )
                            )
                            if (
                                rtc_debug_observer is not None
                                and rtc_debug_observer.accepts(
                                    RTC_DEBUG_LEVEL_SUMMARY
                                )
                            ):
                                rtc_debug_observer.offer(
                                    "rtc_emission_publication",
                                    {
                                        "schema": (
                                            "dexteleop_rtc_emission_publication_debug_v1"
                                        ),
                                        **publication_receipt.to_payload(),
                                        "publication_receipt_hash": (
                                            publication_receipt.semantic_content_hash
                                        ),
                                    },
                                )
                            if not publication_receipt.accepted:
                                raise RuntimeError(
                                    "RTC committed emission publication was "
                                    "rejected: "
                                    f"{publication_receipt.reasons}"
                                )
                        if rtc_emission_fanout is not None and emission is not None:
                            commit = rtc_step.emission_commit_transition
                            if (
                                commit is None
                                or not commit.receipt.accepted
                                or commit.receipt.emission_id != emission.emission_id
                            ):
                                raise RuntimeError(
                                    "RTC emission publication requires exact "
                                    "accepted producer commit evidence."
                                )
                            rtc_emission_fanout.put_nowait(emission)
                        action_step = (
                            policy_deploy_runtime.prepare_emission_result(
                                rtc_step.emission_result,
                                observation,
                                preparer=canonical_action_pipeline,
                                policy_name=policy.name,
                            )
                        )
                        if rtc_debug_observer is not None:
                            summary_worthy = bool(
                                rtc_step.completion is not None
                                or rtc_step.result_transition is not None
                                or rtc_step.emission_result.emission is not None
                                or rtc_step.request_transition is not None
                                or rtc_step.lifecycle_transition is not None
                                or rtc_step.budget_exceeded
                            )
                            minimum_level = (
                                RTC_DEBUG_LEVEL_SUMMARY
                                if summary_worthy
                                else RTC_DEBUG_LEVEL_TRACE
                            )
                            if rtc_debug_observer.accepts(minimum_level):
                                rtc_debug_observer.offer(
                                    "rtc_frontend_step",
                                    rtc_step.to_debug_payload(),
                                    minimum_level=minimum_level,
                                )
                        if (
                            rtc_step.result_transition is not None
                            and not rtc_step.result_transition.receipt.accepted
                        ):
                            faults.extend(
                                "rtc_result_rejected:"
                                f"{reason}"
                                for reason in (
                                    rtc_step.result_transition.receipt.reasons
                                )
                            )
                        elif (
                            rtc_step.result_transition is not None
                            and rtc_step.result_transition.receipt.accepted
                        ):
                            ik_action_gate.mark_policy_output_accepted()
                        if rtc_step.budget_exceeded:
                            faults.append("rtc_control_budget_exceeded")
                        if (
                            rtc_step.state.lifecycle_state
                            != POLICY_ASYNC_PRODUCER_ELIGIBLE
                        ):
                            policy_active = False
                    arbiter_shadow_step = None
                    if policy_arbiter_shadow is not None:
                        if action_step.emission is not None:
                            if debug_replay:
                                if (
                                    replay_proposal_shadow_adapter is not None
                                    and action_step.raw_action is not None
                                ):
                                    try:
                                        replay_adapter_result = (
                                            replay_proposal_shadow_adapter.adapt_emission(
                                                action_step.emission,
                                                raw_action_metadata=(
                                                    action_step.raw_action.metadata
                                                ),
                                            )
                                        )
                                        debug_logger.log_event(
                                            "replay_proposal_adapter",
                                            replay_adapter_debug_payload(
                                                replay_adapter_result
                                            ),
                                        )
                                        if replay_adapter_result.proposal_draft is not None:
                                            arbiter_shadow_step = (
                                                policy_arbiter_shadow.evaluate_proposal_draft(
                                                    replay_adapter_result.proposal_draft,
                                                    source_event_id=(
                                                        replay_adapter_result
                                                        .source_adapter_receipt
                                                        .representation_trace_id
                                                    ),
                                                    emission=action_step.emission,
                                                )
                                            )
                                        else:
                                            arbiter_shadow_step = policy_arbiter_shadow.status(
                                                "replay_adapter_rejected",
                                                reason="replay_source_or_binding_invalid",
                                                emission_id=action_step.emission.emission_id,
                                                emission=action_step.emission,
                                                occurred_at=control_loop_started_at,
                                            )
                                    except (KeyError, TypeError, ValueError) as exc:
                                        arbiter_shadow_step = policy_arbiter_shadow.status(
                                            "replay_adapter_error",
                                            reason=type(exc).__name__,
                                            emission_id=action_step.emission.emission_id,
                                            emission=action_step.emission,
                                            occurred_at=control_loop_started_at,
                                        )
                                else:
                                    arbiter_shadow_step = policy_arbiter_shadow.status(
                                        "replay_adapter_unavailable",
                                        reason="typed_replay_contract_unavailable",
                                        emission_id=action_step.emission.emission_id,
                                        emission=action_step.emission,
                                        occurred_at=control_loop_started_at,
                                    )
                            else:
                                arbiter_shadow_step = (
                                    policy_arbiter_shadow.evaluate_emission(
                                        action_step.emission
                                    )
                                )
                        elif action_step.emission_error is not None:
                            arbiter_shadow_step = policy_arbiter_shadow.status(
                                "emission_error",
                                reason=type(action_step.emission_error).__name__,
                                occurred_at=control_loop_started_at,
                            )
                        if arbiter_shadow_step is not None:
                            debug_logger.log_event(
                                "arbitration_shadow_step",
                                arbiter_shadow_step.to_debug_payload(),
                            )
                    queue_len_before_pop = action_step.queue_len_before
                    raw_action = action_step.raw_action
                    raw_lookahead_action = action_step.raw_lookahead_action
                    queue_len_after_pop = action_step.queue_len_after
                    if raw_action is not None:
                        canonical_action = action_step.prepared_action
                        if (
                            not debug_replay
                            and action_step.emission is not None
                            and arbiter_shadow_step is not None
                            and arbiter_shadow_step.evaluation is not None
                        ):
                            try:
                                policy_safety_comparison = (
                                    evaluate_policy_legacy_aligned_safety(
                                        emission=action_step.emission,
                                        arbitration_evaluation_id=(
                                            arbiter_shadow_step.evaluation.evaluation_id
                                        ),
                                        observation=observation,
                                        legacy_result=canonical_action,
                                        safety_filter=(
                                            canonical_action_pipeline.safety_filter
                                        ),
                                    )
                                )
                                debug_logger.log_event(
                                    "policy_safety_shadow_comparison",
                                    policy_safety_comparison.to_debug_payload(),
                                )
                            except Exception as exc:  # noqa: BLE001
                                # This is a passive observer.  Its failure must
                                # not alter the legacy fault list, last_action,
                                # queue state, or execution result.
                                debug_logger.log_event(
                                    "policy_safety_shadow_error",
                                    {
                                        "non_authoritative": True,
                                        "emission_id": action_step.emission.emission_id,
                                        "arbitration_evaluation_id": (
                                            arbiter_shadow_step.evaluation.evaluation_id
                                        ),
                                        "error_type": type(exc).__name__,
                                    },
                                )
                        lookahead_canonical_action = None
                        lookahead_observation = None
                        if (
                            position_deploy_executor is not None
                            and bool(args.position_chunk_lookahead)
                            and raw_lookahead_action is not None
                            and not canonical_action.faults
                        ):
                            # The current safe action is the correct 15 Hz
                            # reference for filtering the next action in the
                            # same already-decoded chunk.  This is validation,
                            # not an extra temporal smoothing stage.
                            lookahead_observation = replace(
                                observation,
                                last_action=canonical_action.action.vector.astype(
                                    np.float32,
                                    copy=True,
                                ),
                            )
                            prepared_lookahead = canonical_action_pipeline.prepare(
                                raw_lookahead_action,
                                lookahead_observation,
                                policy_name=policy.name,
                            )
                            if not prepared_lookahead.faults:
                                lookahead_canonical_action = prepared_lookahead
                        faults.extend(canonical_action.faults)
                        latch_canonical_safety_fault(
                            canonical_action,
                            observation,
                            policy_name=policy.name,
                        )
                        if (
                            rtc_frontend is not None
                            and canonical_action.faults
                            and rtc_frontend.state.lifecycle_state
                            == POLICY_ASYNC_PRODUCER_ELIGIBLE
                        ):
                            rtc_fault = rtc_frontend.fault(
                                now=control_loop_started_at,
                                reason="canonical_safety_fault",
                            )
                            debug_logger.log_event(
                                "rtc_lifecycle_transition",
                                {
                                    "target": "FAULT",
                                    "reason": "canonical_safety_fault",
                                    "producer_transition_hash": (
                                        rtc_fault.producer_transition.receipt
                                        .semantic_content_hash
                                    ),
                                    "worker_stop_hash": (
                                        rtc_fault.worker_stop_receipt.semantic_content_hash
                                    ),
                                },
                            )
                            policy_active = False
                        dispatch_error: BaseException | None = None
                        execution: _PreparedActionExecution | None = None
                        post_arm_states: dict[str, dict[str, Any]] | None = None
                        if position_deploy_executor is None:
                            annotate_deploy_arm_command_limits(
                                canonical_action.command,
                                args=args,
                                active_sides=active_sides,
                                control_hz=backend_control_hz,
                                action_hz=float(effective_action_hz),
                            )
                        try:
                            if canonical_action.command is not None and canonical_action.command.commands:
                                execution = execute_prepared_action(
                                    canonical_action,
                                    observation,
                                    policy_name=policy.name,
                                    lookahead_canonical_action=(
                                        lookahead_canonical_action
                                    ),
                                    lookahead_observation=lookahead_observation,
                                )
                                if execution.accepted:
                                    position_action_submitted_this_loop = (
                                        position_deploy_executor is not None
                                    )
                                    last_action = canonical_action.action.vector.astype(np.float32, copy=True)
                            if debug_logger.enabled:
                                post_arm_states = _debug_backend_arm_states(arm_backends)
                        except Exception as exc:  # noqa: BLE001
                            dispatch_error = exc
                            if (
                                policy_arbiter_shadow is not None
                                and action_step.emission is not None
                            ):
                                legacy_shadow_event = (
                                    policy_arbiter_shadow.observe_legacy_execution(
                                        emission=action_step.emission,
                                        shadow_step=arbiter_shadow_step,
                                        canonical_action=canonical_action,
                                        execution=execution,
                                        dispatch_error=dispatch_error,
                                    )
                                )
                                debug_logger.log_event(
                                    "legacy_execution_observation",
                                    legacy_shadow_event.to_debug_payload(),
                                )
                            if debug_logger.enabled:
                                post_arm_states = _debug_backend_arm_states(arm_backends)
                                debug_logger.log_action_step(
                                    timestamp=loop_started_at,
                                    step_index=step_index,
                                    policy_name=policy.name,
                                    raw_action=raw_action,
                                    canonical_action=canonical_action,
                                    pre_arm_states=arm_states,
                                    post_arm_states=post_arm_states,
                                    arm_backends=arm_backends,
                                    send_actions=send_actions,
                                    queue_len_before_pop=queue_len_before_pop,
                                    queue_len_after_pop=queue_len_after_pop,
                                    faults=faults,
                                    dispatch_error=dispatch_error,
                                    execution=execution,
                                )
                            raise
                        if (
                            policy_arbiter_shadow is not None
                            and action_step.emission is not None
                        ):
                            legacy_shadow_event = (
                                policy_arbiter_shadow.observe_legacy_execution(
                                    emission=action_step.emission,
                                    shadow_step=arbiter_shadow_step,
                                    canonical_action=canonical_action,
                                    execution=execution,
                                    dispatch_error=dispatch_error,
                                )
                            )
                            debug_logger.log_event(
                                "legacy_execution_observation",
                                legacy_shadow_event.to_debug_payload(),
                            )
                        debug_logger.log_action_step(
                            timestamp=loop_started_at,
                            step_index=step_index,
                            policy_name=policy.name,
                            raw_action=raw_action,
                            canonical_action=canonical_action,
                            pre_arm_states=arm_states,
                            post_arm_states=post_arm_states,
                            arm_backends=arm_backends,
                            send_actions=send_actions,
                            queue_len_before_pop=queue_len_before_pop,
                            queue_len_after_pop=queue_len_after_pop,
                            faults=faults,
                            dispatch_error=dispatch_error,
                            execution=execution,
                        )

                    if (
                        position_deploy_executor is not None
                        and not position_action_submitted_this_loop
                        and (
                            ik_action_gate.blocks_action_output
                            or bool(faults)
                            or not can_run_policy
                            or (wait_for_ready_confirm and not ready_confirmed)
                        )
                        and bool(position_deploy_executor.snapshot().get("active"))
                    ):
                        position_deploy_executor.clear()
                        last_action = observation.hold_action()

                fault_text = _compact_faults(faults)
                if fault_text != last_fault_text:
                    print(f"[State] faults={fault_text}")
                    last_fault_text = fault_text

                if loop_started_at - last_status_at >= float(args.status_sec):
                    policy_status = _policy_status_text(policy)
                    pending_action_count = (
                        policy_deploy_runtime.action_queue_length()
                        if rtc_frontend is None
                        else len(rtc_frontend.state.plan.keyframes)
                    )
                    position_executor_status = ""
                    if position_deploy_executor is not None:
                        position_snapshot = position_deploy_executor.snapshot()
                        last_collision = dict(
                            position_snapshot.get("last_collision") or {}
                        )
                        last_collision_reset = dict(
                            position_snapshot.get("last_collision_reset") or {}
                        )
                        collision_reset_status = (
                            "pending"
                            if not last_collision_reset
                            else (
                                "authoritative"
                                if bool(last_collision_reset.get("authoritative"))
                                else "dry_run_fallback"
                            )
                        )
                        collision_distance = last_collision.get(
                            "position_collision_distance_m"
                        )
                        try:
                            collision_distance_text = (
                                "n/a"
                                if collision_distance is None
                                else f"{float(collision_distance):.4f}"
                            )
                        except (TypeError, ValueError):
                            collision_distance_text = "invalid"
                        position_executor_status = (
                            "position_refs="
                            f"{position_snapshot.get('reference_count', 0)} "
                            f"position_queue={position_snapshot.get('queue_depth', 0)} "
                            f"position_missed={position_snapshot.get('missed_reference_ticks', 0)} "
                            f"position_hand="
                            f"{position_snapshot.get('discrete_dispatch_count', 0)}/"
                            f"{position_snapshot.get('discrete_dispatch_attempt_count', 0)} "
                            f"position_hand_skip="
                            f"{position_snapshot.get('discrete_skipped_count', 0) + position_snapshot.get('discrete_backend_skipped_count', 0)} "
                            f"position_debug_drop="
                            f"{position_snapshot.get('debug_event_drop_count', 0)} "
                            f"position_collision="
                            f"{last_collision.get('position_collision_status', 'pending')} "
                            f"position_collision_d_m={collision_distance_text} "
                            f"position_collision_counts="
                            f"{position_snapshot.get('collision_tracking_count', 0)}/"
                            f"{position_snapshot.get('collision_limited_count', 0)}/"
                            f"{position_snapshot.get('collision_backoff_count', 0)}/"
                            f"{position_snapshot.get('collision_hold_count', 0)} "
                            f"position_collision_endpoint="
                            f"{int(bool(position_snapshot.get('endpoint_collision_certified')))} "
                            "position_collision_segment=0 "
                            f"position_collision_reset={collision_reset_status}"
                        )
                    status_parts = [
                        text
                        for text in (
                            fixed_init_controller.status_text if fixed_init_controller is not None else "",
                            (
                                policy_qpos_init_controller.status_text
                                if policy_qpos_init_controller is not None
                                else ""
                            ),
                            startup_initializer.status_text,
                            "ready=wait:A" if (
                                wait_for_ready_confirm
                                and policy_active
                                and startup_initializer.completed
                                and not ready_confirmed
                            ) else "",
                            policy_status,
                            ik_action_gate.status_text(),
                            position_executor_status,
                            (
                                "rtc="
                                f"{rtc_frontend.state.lifecycle_state.lower()} "
                                "rtc_worker="
                                f"{rtc_frontend.state.worker_session_state.lower()} "
                                "rtc_emissions="
                                f"{rtc_frontend.selected_emission_count}"
                                if rtc_frontend is not None
                                else ""
                            ),
                        )
                        if text
                    ]
                    suffix = f" {' '.join(status_parts)}" if status_parts else ""
                    print(
                        "[Status] "
                        f"step={step_index} hz={control_hz:.2f} "
                        f"policy={'active' if policy_active else 'paused'} "
                        f"send={'on' if send_actions else 'dry-run'} "
                        f"queue={pending_action_count} "
                        f"state_valid={int(np.sum(observation.state_valid))}/{observation.state_valid.size} "
                        f"{camera_status_text(camera_sources, camera_previewer)}"
                        f"{suffix}"
                    )
                    last_status_at = loop_started_at

                step_index += 1
                if int(args.max_steps) > 0 and step_index >= int(args.max_steps):
                    break
                loop_target_hz = float(effective_action_hz)
                if fixed_init_controller is not None and fixed_init_controller.running:
                    loop_target_hz = max(loop_target_hz, float(fixed_init_controller.plan.action_hz))
                if policy_qpos_init_controller is not None and policy_qpos_init_controller.running:
                    loop_target_hz = max(
                        loop_target_hz,
                        1.0 / max(0.001, float(args.policy_qpos_init_dt_sec)),
                    )
                sleep_time = (1.0 / loop_target_hz) - (time.time() - loop_started_at)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[Deploy] keyboard interrupt; stopping\n")
    finally:
        position_workers_stopped = True
        cleanup_errors: list[str] = []

        def close_resource(label: str, resource: Any) -> None:
            try:
                resource.close()
            except Exception as exc:  # noqa: BLE001 - close every independent resource.
                error = f"{label}: {type(exc).__name__}: {exc}"
                cleanup_errors.append(error)
                debug_logger.log_event(
                    "cleanup_error",
                    {"resource": str(label), "error_type": type(exc).__name__, "error": str(exc)},
                )
                print(f"[DeployCleanup] {error}", file=sys.stderr)

        if rtc_frontend is not None:
            try:
                rtc_stop = rtc_frontend.close(now=time.monotonic())
                if rtc_debug_observer is not None:
                    rtc_debug_observer.offer(
                        "rtc_frontend_close",
                        {
                            "worker_stop_hash": rtc_stop.semantic_content_hash,
                            "worker_session_state": rtc_stop.session_state,
                            "thread_exited": rtc_stop.thread_exited,
                            "execution_drain_evidence": (
                                rtc_stop.execution_drain_evidence
                            ),
                            "stop_status": rtc_stop.status,
                        },
                    )
            except Exception as exc:  # noqa: BLE001 - continue all cleanup.
                error = f"rtc_frontend: {type(exc).__name__}: {exc}"
                cleanup_errors.append(error)
                print(f"[DeployCleanup] {error}", file=sys.stderr)
        if rtc_debug_observer is not None:
            try:
                rtc_debug_was_enabled = rtc_debug_observer.enabled
                rtc_debug_status = rtc_debug_observer.close()
                if rtc_debug_was_enabled:
                    debug_logger.log_event(
                        "rtc_debug_observer_status",
                        rtc_debug_status.to_payload(),
                    )
            except Exception as exc:  # noqa: BLE001 - diagnostic shutdown only.
                error = f"rtc_debug_observer: {type(exc).__name__}: {exc}"
                cleanup_errors.append(error)
                print(f"[DeployCleanup] {error}", file=sys.stderr)
        if position_deploy_executor is not None:
            try:
                position_deploy_executor.close()
            except Exception as exc:  # noqa: BLE001 - inspect worker liveness before backend close.
                close_snapshot = position_deploy_executor.snapshot()
                position_workers_stopped = not bool(
                    close_snapshot.get("reference_worker_alive")
                    or close_snapshot.get("discrete_worker_alive")
                )
                print(
                    f"[PositionDeploy] close/hold error: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                cleanup_errors.append(
                    f"position_executor: {type(exc).__name__}: {exc}"
                )
                debug_logger.log_event(
                    "position_executor_close_error",
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "snapshot": close_snapshot,
                        "backend_close_skipped": bool(not position_workers_stopped),
                    },
                )
            flush_position_executor_debug_events()
            debug_logger.log_event(
                "position_executor_final_snapshot",
                position_deploy_executor.snapshot(),
            )
        if fixed_init_controller is not None:
            try:
                fixed_init_controller.restore_speed(arm_backends)
            except Exception as exc:  # noqa: BLE001
                error = f"fixed_init_restore_speed: {type(exc).__name__}: {exc}"
                cleanup_errors.append(error)
                debug_logger.log_event(
                    "cleanup_error",
                    {
                        "resource": "fixed_init_restore_speed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                print(f"[DeployCleanup] {error}", file=sys.stderr)
        if hand_state_poller is not None:
            close_resource("hand_state_poller", hand_state_poller)
        if position_workers_stopped:
            for side, backend in arm_backends.items():
                close_resource(f"{side}_arm_backend", backend)
            for side, backend in hand_backends.items():
                close_resource(f"{side}_hand_backend", backend)
        else:
            cleanup_errors.append(
                "position_backends: close skipped because a dispatch worker is still alive"
            )
            print(
                "[PositionDeploy] ERROR: a dispatch worker is still alive; "
                "arm/hand backend close was intentionally skipped to avoid concurrent I/O.",
                file=sys.stderr,
            )
        for camera_key, camera_source in camera_sources.items():
            close_resource(f"camera:{camera_key}", camera_source)
        if camera_previewer is not None:
            close_resource("camera_previewer", camera_previewer)
        if visualizer is not None:
            close_resource("tactile_visualizer", visualizer)
        try:
            debug_logger.close()
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(f"debug_logger: {type(exc).__name__}: {exc}")
            print(
                f"[DeployCleanup] debug_logger: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        if cleanup_errors:
            print(
                f"[DeployCleanup] completed with {len(cleanup_errors)} error(s).",
                file=sys.stderr,
            )
    return 1 if cleanup_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
