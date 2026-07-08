from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cameras.configs import DEFAULT_REALSENSE_DEVICE_CONFIG_PATH
from project_paths import repo_path
from rm65_inspire_tac.paths import require_asset_path
from robot_motion.runtime_gate import IKActionChunkGate

from ..backends.arms import (
    DEFAULT_SINGULARITY_QPOS_INIT_SLOWDOWN,
    RM65ArmBackend,
    RM65ArmBackendConfig,
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
from ..action import HandActionSpec, hand_action_specs_metadata
from ..collection.camera_preview import CameraPreviewConfig, CameraPreviewer
from ..configs.dmtac_hardware import (
    dmtac_backend_configs_from_assignment,
    load_dmtac_hardware_assignment,
)
from ..defaults import DEFAULT_RM65_ARM_CONTROL_HZ, DEFAULT_TELEOP_SAMPLE_HZ
from ..hand_mapping import parse_tactile_regions
from ..interfaces import EffectorBackend
from ..sources.camera import (
    RealSenseRGBCameraConfig,
    RealSenseRGBSource,
    load_realsense_camera_config_map,
)
from ..tactile_visualizer import (
    TactileUVVisualizer,
    TactileVisualizerConfig,
    tactile_by_side_from_backends,
)
from .actions import (
    ActionChunkBuffer,
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    NormalizedAction,
    normalize_policy_output,
)
from .canonical_action import CanonicalDeployActionPipeline
from .dispatch import dispatch_teleop_command
from .fixed_init import DeployFixedInitController, load_deploy_fixed_init_plan
from .observation import build_deployment_observation
from .policy import load_policy
from .policy_qpos_init import (
    DeployPolicyQposInitController,
    DeployPolicyQposInitSettings,
    load_deploy_policy_qpos_init_plan,
)
from .replay import infer_replay_action_hz, load_dataset_replay_policy, resolve_replay_parquet_path
from .schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    ACTION_HAND_SLICES,
    ALL_SIDES,
    pose_matrix_to_xyzquat,
    to_serializable,
)
from .startup_init import StartupInitializer, resolve_startup_init_action
from ..replay.trajectory import QPOS_INIT_SOURCES


DEFAULT_CAMERA_CONFIG_PATH = (
    DEFAULT_REALSENSE_DEVICE_CONFIG_PATH
)
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
    "telemetry_hz": 10.0,
}
DEPLOY_RM65_ARM_MAX_ANGVEL = (0.7, 0.7, 0.9, 1.5, 1.5, 2.0)
DEPLOY_ARM_STEP_DT_ALPHA = 0.25
DEPLOY_ARM_STEP_DT_MAX_SCALE = 3.0
DEPLOY_ARM_STEP_DT_STALL_RESET_SEC = 0.25
DEPLOY_MAX_ARM_JOINT_STEP_RAD = float(np.deg2rad(6.0))


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


def _debug_backend_arm_states(backends: Mapping[str, RM65ArmBackend]) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for side, backend in backends.items():
        try:
            states[side] = dict(backend.get_state())
        except Exception as exc:  # noqa: BLE001
            states[side] = {"state_error": f"{type(exc).__name__}: {exc}"}
    return states


def _debug_backend_limits(
    backends: Mapping[str, RM65ArmBackend],
    *,
    now: float,
) -> dict[str, Any]:
    limits: dict[str, Any] = {}
    for side, backend in backends.items():
        try:
            effective_step = backend.get_effective_max_joint_step(now=now)
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
            "background_streaming_enabled": bool(config.background_streaming_enabled),
            "background_streaming_hz": float(config.background_streaming_hz),
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
    ):
        self.enabled = bool(enabled)
        self.active_sides = tuple(active_sides)
        self.path: Path | None = None
        self._handle: Any = None
        self._write_failed = False
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
        if self._handle is None:
            return
        self._handle.flush()
        self._handle.close()
        self._handle = None

    def log_event(self, event: str, payload: Mapping[str, Any]) -> None:
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
            self._handle.flush()
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
        arm_backends: Mapping[str, RM65ArmBackend],
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
        arm_backends: Mapping[str, RM65ArmBackend],
        send_actions: bool,
        queue_len_before_pop: int,
        queue_len_after_pop: int,
        faults: list[str],
        dispatch_error: BaseException | None,
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
        dispatch_attempted = bool(command_count and send_actions)
        self.log_event(
            "action_step",
            {
                "timestamp": float(timestamp),
                "step_index": int(step_index),
                "policy_name": str(policy_name),
                "send_actions": bool(send_actions),
                "dispatch_attempted": bool(dispatch_attempted),
                "dispatch_mode": "send_actions" if send_actions else "dry_run",
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
        backends: Mapping[str, RM65ArmBackend],
        *,
        refresh_hz: float,
        compute_tcp_pose: bool = True,
    ):
        self.backends = dict(backends)
        self.refresh_hz = float(refresh_hz)
        self.compute_tcp_pose = bool(compute_tcp_pose)
        self._last_refresh_at = {side: 0.0 for side in self.backends}

    def snapshot(self, now: float) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        refresh_period = 0.0 if self.refresh_hz <= 0.0 else 1.0 / self.refresh_hz
        for side, backend in self.backends.items():
            if (
                refresh_period > 0.0
                and backend.is_ready()
                and now - self._last_refresh_at.get(side, 0.0) >= refresh_period
            ):
                try:
                    backend.refresh_state_from_hardware()
                except Exception as exc:  # noqa: BLE001
                    state = backend.get_state()
                    state["state_error"] = f"refresh_failed:{type(exc).__name__}:{exc}"
                    states[side] = state
                    continue
                self._last_refresh_at[side] = now
            state = backend.get_state()
            if self.compute_tcp_pose:
                try:
                    hardware_qpos = state.get("hardware_qpos")
                    if hardware_qpos is not None:
                        state["current_tcp_pose"] = backend.compute_tcp_pose(qpos=hardware_qpos)
                        state["tcp_pose_source"] = "hardware_qpos_fk"
                    else:
                        state["current_tcp_pose"] = backend.compute_tcp_pose()
                        state["tcp_pose_source"] = "current_qpos_fk"
                except Exception as exc:  # noqa: BLE001
                    state.setdefault("pose_error", f"compute_tcp_pose_failed:{type(exc).__name__}:{exc}")
            states[side] = state
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
    setattr(args, "_resolved_gripper_policy", dict(policy))
    return dict(policy)


def _dmtac_tactile_enabled(args: argparse.Namespace) -> bool:
    explicit = getattr(args, "dmtac_tactile", None)
    if explicit is not None:
        return bool(explicit)
    return True


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
                    active_sides=active_sides,
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

    for side in active_sides:
        manual = _manual_dmtac_config_for_side(args, side, policy)
        if manual is not None:
            configs[side] = manual

    resolved = {
        side: _apply_gripper_policy_to_dmtac_config(config, args=args, policy=policy)
        for side, config in configs.items()
        if side in active_sides and (config.connect_hardware or config.connect_tactile)
    }
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
    parser.add_argument("--active-sides", nargs="+", default=("left", "right"))
    parser.add_argument(
        "--hz",
        type=_positive_float,
        default=DEFAULT_TELEOP_SAMPLE_HZ,
        help="Fallback action rate when --action-hz is 0 and no dataset/policy rate is inferred.",
    )
    parser.add_argument(
        "--arm-control-hz",
        type=_positive_float,
        default=DEFAULT_RM65_ARM_CONTROL_HZ,
        help="RM65 backend streaming/control rate after each action update. Defaults to 40 Hz.",
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
    parser.add_argument("--deploy-fixed-init-qpos-tolerance", type=_positive_float, default=0.01)
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
    parser.add_argument("--max-arm-xyz-step-m", type=float, default=0.1)
    parser.add_argument("--max-arm-rot-step-deg", type=float, default=15.0)
    parser.add_argument(
        "--max-arm-joint-step-rad",
        type=float,
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
    parser.add_argument("--left-gripper-port", type=str, default="", help="Optional left 1D gripper serial port.")
    parser.add_argument("--right-gripper-port", type=str, default="", help="Optional right 1D gripper serial port.")
    parser.add_argument("--left-gripper-id", type=int, default=1, help="Left gripper Modbus id.")
    parser.add_argument("--right-gripper-id", type=int, default=1, help="Right gripper Modbus id.")
    parser.add_argument(
        "--dmtac-config",
        type=str,
        default="latest",
        help="DMTAC gripper/tactile assignment config used when --hand-backend dmtac-gripper.",
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

    parser.add_argument("--camera-mode", choices=["none", "single", "multi"], default="none")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG_PATH)
    parser.add_argument("--camera-key", type=str, default="front")
    parser.add_argument("--camera-serial", "--camera-id", dest="camera_serial", type=str, default="")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-keep-last", type=int, default=4)
    parser.add_argument("--camera-color-space", choices=["rgb", "bgr"], default="rgb")
    parser.add_argument("--camera-wait-timeout-ms", type=int, default=1000)
    parser.add_argument("--camera-stale-after-sec", type=float, default=2.0)
    parser.add_argument("--camera-auto-exposure-priority", type=int, default=None)
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
    if _hand_backend_mode(args) == HAND_BACKEND_DMTAC_GRIPPER:
        return True
    return bool(
        args.left_hand_ip
        or args.right_hand_ip
        or args.visualize_tactile
    )


def validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    active_sides = _parse_active_sides(args.active_sides)
    args.active_sides = active_sides
    args.arm_workspace_min = _parse_vector3(args.arm_workspace_min)
    args.arm_workspace_max = _parse_vector3(args.arm_workspace_max)
    hands_enabled = _hands_enabled(args)
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
        if _hand_backend_mode(args) == HAND_BACKEND_INSPIRE:
            for side in active_sides:
                if not getattr(args, f"{side}_hand_ip"):
                    raise ValueError(f"--send-actions with hands requires --{side}-hand-ip.")
    if bool(args.visualize_tactile):
        if _hand_backend_mode(args) == HAND_BACKEND_DMTAC_GRIPPER:
            raise ValueError("--visualize-tactile currently supports InspireTac UV tactile, not DMTAC gripper tactile.")
        if not hands_enabled:
            raise ValueError("--visualize-tactile requires hands to be enabled.")
        for side in active_sides:
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
        if int(args.camera_wait_timeout_ms) <= 0:
            raise ValueError("--camera-wait-timeout-ms must be positive.")
        if float(args.camera_stale_after_sec) <= 0.0:
            raise ValueError("--camera-stale-after-sec must be positive.")
    elif bool(args.vis_cam):
        raise ValueError("--vis_cam requires --camera-mode single or --camera-mode multi.")
    return active_sides


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
    base_dt = 1.0 / max(1e-6, float(control_hz))
    for side in active_sides:
        effector_command = commands.get(f"{side}_arm")
        if effector_command is None:
            continue
        effector_command.metadata["max_joint_step_rad"] = max_joint_step.copy()
        effector_command.metadata["deploy_max_joint_step_rad"] = max_joint_step.copy()
        effector_command.metadata["deploy_action_step_dt_sec"] = float(step_dt)
        effector_command.metadata["deploy_backend_control_dt_sec"] = float(base_dt)
        effector_command.metadata["deploy_arm_step_dt_scaling"] = bool(
            getattr(args, "arm_step_dt_scaling", True)
        )
        effector_command.metadata["deploy_arm_step_dt_max_scale"] = float(
            getattr(args, "arm_step_dt_max_scale", DEPLOY_ARM_STEP_DT_MAX_SCALE)
        )


def build_arm_backends(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
    *,
    control_hz: float,
    action_hz: float | None = None,
) -> dict[str, RM65ArmBackend]:
    if not bool(args.arms_enabled):
        return {}
    arm_urdf_path = str(args.arm_urdf_path or require_asset_path("rm65_arm.urdf"))
    control_dt = 1.0 / float(control_hz)
    apply_arm_step_dt_defaults(args, control_hz=float(control_hz), action_hz=action_hz)
    backends: dict[str, RM65ArmBackend] = {}
    for side in active_sides:
        arm_ip = str(getattr(args, f"{side}_arm_ip") or "")
        backends[side] = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip=arm_ip or "0.0.0.0",
                side=side,
                arm_urdf_path=arm_urdf_path,
                connect_hardware=bool(arm_ip),
                control_dt=control_dt,
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
                background_streaming_enabled=bool(arm_ip),
                background_streaming_hz=float(control_hz),
            )
        )
    return backends


def build_hand_backends(args: argparse.Namespace, active_sides: tuple[str, ...]) -> dict[str, EffectorBackend]:
    if not _hands_enabled(args):
        return {}
    if _hand_backend_mode(args) == HAND_BACKEND_DMTAC_GRIPPER:
        configs = _dmtac_backend_configs(args, active_sides)
        return {
            side: DMTacGripperBackend(config)
            for side, config in configs.items()
            if side in active_sides
        }

    tactile_regions = parse_tactile_regions(args.hand_tactile_regions)
    backends: dict[str, EffectorBackend] = {}
    for side in active_sides:
        hand_ip = str(getattr(args, f"{side}_hand_ip") or "")
        backends[side] = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip=hand_ip or "0.0.0.0",
                port=int(args.hand_modbus_port),
                side=side,
                effector_name=f"{side}_hand",
                connect_hardware=bool(hand_ip),
                max_fingertip_force=int(args.max_fingertip_force),
                send_threshold=float(args.hand_send_threshold),
                enable_telemetry=bool(hand_ip),
                telemetry_hz=float(args.hand_telemetry_hz),
                tactile_regions=tuple(tactile_regions),
            )
        )
    return backends


def build_camera_sources(args: argparse.Namespace) -> dict[str, RealSenseRGBSource]:
    if args.camera_mode == "none":
        return {}
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
            metadata={"source": "teleop_runtime_deploy_cli"},
        )
        return {config.camera_key: RealSenseRGBSource(config)}
    camera_configs = load_realsense_camera_config_map(Path(args.camera_config))
    if args.camera_mode == "single" and len(camera_configs) != 1:
        raise ValueError(f"--camera-mode single expects exactly one camera config, got {len(camera_configs)}.")
    if args.camera_mode == "multi" and not camera_configs:
        raise ValueError("--camera-mode multi requires at least one camera config.")
    return {key: RealSenseRGBSource(config) for key, config in camera_configs.items()}


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


def hand_state_snapshot(backends: Mapping[str, EffectorBackend]) -> dict[str, dict[str, Any]]:
    return {side: backend.get_state() for side, backend in backends.items()}


def hand_action_specs_for_backends(
    args: argparse.Namespace,
    backends: Mapping[str, EffectorBackend],
    active_sides: tuple[str, ...],
) -> dict[str, Any]:
    specs = {}
    hand_backend_mode = _hand_backend_mode(args)
    for side in active_sides:
        if hand_backend_mode == HAND_BACKEND_DMTAC_GRIPPER:
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
        print("Controls: H=pause policy or fixed-init step, g=start after fixed-init, A=confirm, q=quit")
    elif policy_qpos_init_enabled:
        print("  policy_qpos_init=on(g moves to dataset frame qpos, then startup init)")
        print("Controls: g=start qpos init / resume, A=confirm ready, h=pause+clear queue, q=quit")
    else:
        print("Controls: g=start/resume, A=confirm ready/start policy, h=pause+clear queue, q=quit")
    print("=" * 60 + "\n")


def _policy_should_run(
    *,
    now: float,
    last_policy_at: float | None,
    policy_interval_s: float,
    chunk_buffer: ActionChunkBuffer,
    min_action_queue: int,
) -> bool:
    if len(chunk_buffer) > int(min_action_queue):
        return False
    if last_policy_at is None:
        return True
    return now - float(last_policy_at) >= float(policy_interval_s)


def _resolve_action_queue_policy(args: argparse.Namespace, *, debug_replay: bool) -> str:
    if bool(getattr(args, "append_action_chunks", False)):
        return "append"
    policy = str(getattr(args, "action_queue_policy", "auto") or "auto").strip().lower()
    if policy == "auto":
        return "append" if bool(debug_replay) else "replace"
    if policy not in {"replace", "append"}:
        raise ValueError(f"Unsupported action queue policy: {policy!r}.")
    return policy


def _send_legacy_deployment_action_commands(
    commands: Mapping[str, Any],
    *,
    arm_backends: Mapping[str, RM65ArmBackend],
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


def _skip_startup_replay_frames(policy: Any, count: int = 1) -> int:
    wrapped_policy = getattr(policy, "policy", policy)
    skip = getattr(wrapped_policy, "skip_startup_frames", None)
    if not callable(skip):
        return 0
    return int(skip(count))


def _default_action(observation: Any) -> NormalizedAction:
    return NormalizedAction(vector=observation.hold_action())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deploy a policy on teleop_runtime RM65 arms, optional InspireTac hands, and RealSense."
    )
    add_args(parser)
    args = parser.parse_args(argv)
    active_sides = validate_args(args)
    hands_enabled = _hands_enabled(args)
    debug_replay = bool(args.debug_replay_dataset or args.debug_replay_parquet)
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
        )
    else:
        policy = load_policy(args.policy_spec, kwargs=parse_policy_kwargs(args.policy_kwargs_json))
    action_queue_policy = _resolve_action_queue_policy(args, debug_replay=debug_replay)
    startup_init_action, startup_init_source = resolve_startup_init_action(args, policy)
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

    backend_control_hz = float(args.arm_control_hz)
    arm_backends = build_arm_backends(
        args,
        active_sides,
        control_hz=backend_control_hz,
        action_hz=float(effective_action_hz),
    )
    hand_backends = build_hand_backends(args, active_sides)
    hand_action_specs = (
        hand_action_specs_for_backends(args, hand_backends, active_sides)
        if hands_enabled
        else {}
    )
    arm_reader = ArmStateReader(
        arm_backends,
        refresh_hz=float(args.arm_state_refresh_hz),
        compute_tcp_pose=bool(args.arms_enabled),
    )
    camera_sources = build_camera_sources(args)
    preview_camera_sources = (
        select_preview_camera_sources(camera_sources, str(args.vis_cam_key))
        if bool(args.vis_cam) and camera_sources
        else {}
    )
    camera_previewer = CameraPreviewer(CameraPreviewConfig()) if bool(args.vis_cam) and preview_camera_sources else None
    visualizer = (
        TactileUVVisualizer(
            TactileVisualizerConfig(
                enabled_sides=active_sides,
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
            workspace_min=tuple(float(value) for value in args.arm_workspace_min),
            workspace_max=tuple(float(value) for value in args.arm_workspace_max),
            max_arm_xyz_step_m=float(args.max_arm_xyz_step_m),
            max_arm_rot_step_deg=float(args.max_arm_rot_step_deg),
            max_arm_joint_step_rad=float(args.max_arm_joint_step_rad),
            max_hand_step=float(args.max_hand_step),
        )
    )
    canonical_action_pipeline = CanonicalDeployActionPipeline(
        safety_filter=safety_filter,
        active_sides=active_sides,
        arms_enabled=bool(args.arms_enabled),
        hands_enabled=bool(hands_enabled),
    )
    chunk_buffer = ActionChunkBuffer()
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
    last_policy_at: float | None = None
    step_index = 0
    loop_rate = LoopRate()
    last_status_at = 0.0
    last_fault_text: str | None = None
    started_at = time.time()
    debug_logger = DeploymentDebugLogger(
        enabled=bool(args.debug),
        root_dir=args.debug_dir,
        active_sides=active_sides,
    )

    try:
        print(
            "[Deploy] policy="
            f"{policy.name} active_sides={active_sides} "
            f"arms={bool(args.arms_enabled)} hands={hands_enabled} "
            f"hand_backend={_hand_backend_mode(args)} "
            f"hz={float(effective_action_hz):.2f} "
            f"policy_hz={float(effective_policy_hz):.2f} "
            f"backend_hz={backend_control_hz:.2f} "
            f"queue_policy={action_queue_policy}"
        )
        if action_hz_info.get("source") != "cli.hz":
            print(
                "[Deploy] action_hz="
                f"{float(effective_action_hz):.3f} source={action_hz_info.get('source')} "
                f"policy_hz_source={policy_hz_source}"
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
            and _hand_backend_mode(args) == HAND_BACKEND_DMTAC_GRIPPER
        ):
            missing_hand_sides = [side for side in active_sides if side not in hand_backends]
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
        for side, backend in hand_backends.items():
            backend.start()
            if not backend.is_ready() and not bool(args.allow_hand_connect_failure):
                raise RuntimeError(f"{side} hand backend is not ready: {backend.get_state()}")
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
        for camera_source in camera_sources.values():
            camera_source.start()
        print_controls(
            send_actions=send_actions,
            auto_start=bool(args.auto_start),
            wait_for_ready_confirm=wait_for_ready_confirm,
            fixed_init_enabled=fixed_init_controller is not None,
            policy_qpos_init_enabled=policy_qpos_init_controller is not None,
        )
        if policy_qpos_init_controller is not None and bool(args.auto_start):
            message = policy_qpos_init_controller.begin(
                arm_backends=arm_backends,
                now=time.time(),
            )
            if message:
                print(message)

        with NonBlockingKeyboard() as keyboard:
            while True:
                loop_started_at = time.time()
                key = keyboard.poll()
                preview_keys: list[str] = []
                if camera_previewer is not None:
                    preview_keys = camera_previewer.update(preview_camera_sources)
                if preview_keys and key is None:
                    key = preview_keys[0]
                key = None if key is None else key.lower()
                if key == "g":
                    policy.reset()
                    chunk_buffer.clear()
                    startup_initializer.reset()
                    last_policy_at = None
                    ready_confirmed = not wait_for_ready_confirm
                    if fixed_init_controller is not None and fixed_init_controller.blocks_policy:
                        print("[Deploy] ignore g: complete fixed-init H -> H before starting policy.")
                    elif policy_qpos_init_controller is not None and policy_qpos_init_controller.blocks_policy:
                        policy_active = False
                        message = policy_qpos_init_controller.begin(
                            arm_backends=arm_backends,
                            now=loop_started_at,
                        )
                        if message:
                            print(message)
                    else:
                        policy_active = True
                        print("[Deploy] policy active")
                elif key == "a":
                    if not policy_active:
                        policy_active = True
                    if fixed_init_controller is not None and fixed_init_controller.blocks_policy:
                        policy_active = False
                        print("[Deploy] ready confirm ignored: complete fixed-init H -> H first.")
                    elif policy_qpos_init_controller is not None and policy_qpos_init_controller.blocks_policy:
                        policy_active = False
                        print("[Deploy] ready confirm ignored: complete policy qpos init first (press g).")
                    elif startup_initializer.enabled and not startup_initializer.completed:
                        print("[Deploy] ready confirm ignored: startup init is not complete yet.")
                    else:
                        ready_confirmed = True
                        chunk_buffer.clear()
                        last_policy_at = None
                        print("[Deploy] ready confirmed; policy inference/execution may start.")
                elif key == "h":
                    chunk_buffer.clear()
                    ready_confirmed = not wait_for_ready_confirm
                    if policy_active:
                        policy_active = False
                        print("[Deploy] policy paused; queue cleared")
                    elif fixed_init_controller is not None:
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
                    print("\n[Deploy] quitting policy deployment loop\n")
                    break

                arm_states = arm_reader.snapshot(loop_started_at)
                ik_gate_event = ik_action_gate.update(arm_states)
                if ik_action_gate.blocks_action_output:
                    chunk_buffer.clear()
                if ik_gate_event:
                    last_policy_at = None
                    print(f"[Deploy] {ik_gate_event}; action queue cleared")
                hand_states = hand_state_snapshot(hand_backends)
                control_hz = loop_rate.tick(loop_started_at)
                observation = build_deployment_observation(
                    active_sides=active_sides,
                    arms_enabled=bool(args.arms_enabled),
                    hands_enabled=hands_enabled,
                    step_index=step_index,
                    arm_states=arm_states,
                    hand_states=hand_states,
                    camera_sources=camera_sources,
                    last_action=last_action,
                    timestamp=loop_started_at,
                    require_images=bool(args.require_camera_frame),
                    metadata={
                        "control_hz": control_hz,
                        "policy_active": policy_active,
                        "send_actions": send_actions,
                        "hand_action_specs_by_side": hand_action_specs,
                        "hand_active_mask_by_side": {
                            side: list(spec.get("active_mask", ()))
                            for side, spec in hand_action_specs.items()
                        },
                    },
                )
                faults = list(observation.faults)
                if loop_started_at - started_at < float(args.startup_warmup_sec):
                    faults.append("startup_warmup")

                if visualizer is not None:
                    visualizer.update(tactile_by_side_from_backends(active_sides, hand_backends.get))

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
                        chunk_buffer.clear()
                        startup_initializer.reset()
                        last_policy_at = None
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
                        chunk_buffer.clear()
                        startup_initializer.reset()
                        last_policy_at = None
                        ready_confirmed = not wait_for_ready_confirm
                        policy_active = True
                        policy_qpos_completed_this_step = True
                        print("[Deploy] policy qpos init complete; startup init/policy released.")

                if policy_active:
                    can_run_policy = not (bool(args.require_ready_observation) and faults)
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
                                if (
                                    startup_initializer.progress >= 1.0
                                    and not canonical_action.faults
                                ):
                                    startup_initializer.update_resolved_target_action(
                                        canonical_action.action.vector
                                    )
                                if canonical_action.command is not None and canonical_action.command.commands:
                                    if send_actions:
                                        dispatch_teleop_command(
                                            canonical_action.command,
                                            arm_backends=arm_backends,
                                            hand_backends=hand_backends,
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
                                    chunk_buffer.clear()
                                    last_policy_at = None
                                    if wait_for_ready_confirm and not ready_confirmed:
                                        print(
                                            "[StartupInit] ready; press A to start policy inference/execution."
                                        )
                        can_run_policy = False
                    if can_run_policy and wait_for_ready_confirm and not ready_confirmed:
                        can_run_policy = False
                    if can_run_policy and _policy_should_run(
                        now=loop_started_at,
                        last_policy_at=last_policy_at,
                        policy_interval_s=1.0 / float(effective_policy_hz),
                        chunk_buffer=chunk_buffer,
                        min_action_queue=int(args.min_action_queue),
                    ):
                        queue_len_before_update = len(chunk_buffer)
                        try:
                            policy_output = policy.predict(observation)
                            actions = normalize_policy_output(
                                policy_output,
                                observation,
                                active_sides=active_sides,
                                arms_enabled=bool(args.arms_enabled),
                                hands_enabled=hands_enabled,
                            )
                            if actions:
                                last_policy_at = loop_started_at
                                if ik_action_gate.discards_policy_output:
                                    chunk_buffer.clear()
                                    print("[Deploy] IK recovery active; discarded predicted action chunk.")
                                    debug_logger.log_policy_chunk(
                                        timestamp=loop_started_at,
                                        step_index=step_index,
                                        policy_name=policy.name,
                                        policy_output=policy_output,
                                        actions=actions,
                                        accepted=False,
                                        discarded_by_ik=True,
                                        queue_policy=action_queue_policy,
                                        queue_len_before=queue_len_before_update,
                                        queue_len_after=len(chunk_buffer),
                                        faults=faults,
                                    )
                                else:
                                    if action_queue_policy == "append":
                                        chunk_buffer.append(actions)
                                    else:
                                        chunk_buffer.replace(actions)
                                    ik_action_gate.mark_policy_output_accepted()
                                    debug_logger.log_policy_chunk(
                                        timestamp=loop_started_at,
                                        step_index=step_index,
                                        policy_name=policy.name,
                                        policy_output=policy_output,
                                        actions=actions,
                                        accepted=True,
                                        discarded_by_ik=False,
                                        queue_policy=action_queue_policy,
                                        queue_len_before=queue_len_before_update,
                                        queue_len_after=len(chunk_buffer),
                                        faults=faults,
                                    )
                            else:
                                faults.append("policy_empty_action")
                                debug_logger.log_policy_chunk(
                                    timestamp=loop_started_at,
                                    step_index=step_index,
                                    policy_name=policy.name,
                                    policy_output=policy_output,
                                    actions=[],
                                    accepted=False,
                                    discarded_by_ik=False,
                                    queue_policy=action_queue_policy,
                                    queue_len_before=queue_len_before_update,
                                    queue_len_after=len(chunk_buffer),
                                    faults=faults,
                                )
                        except Exception as exc:  # noqa: BLE001
                            print(f"[Deploy] policy error: {type(exc).__name__}: {exc}")
                            chunk_buffer.clear()
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

                    queue_len_before_pop = len(chunk_buffer)
                    raw_action = (
                        chunk_buffer.pop(default=_default_action(observation))
                        if can_run_policy and not ik_action_gate.blocks_action_output
                        else None
                    )
                    queue_len_after_pop = len(chunk_buffer)
                    if raw_action is not None:
                        canonical_action = canonical_action_pipeline.prepare(
                            raw_action,
                            observation,
                            policy_name=policy.name,
                        )
                        faults.extend(canonical_action.faults)
                        dispatch_error: BaseException | None = None
                        post_arm_states: dict[str, dict[str, Any]] | None = None
                        annotate_deploy_arm_command_limits(
                            canonical_action.command,
                            args=args,
                            active_sides=active_sides,
                            control_hz=backend_control_hz,
                            action_hz=float(effective_action_hz),
                        )
                        try:
                            if canonical_action.command is not None and canonical_action.command.commands:
                                if send_actions:
                                    dispatch_teleop_command(
                                        canonical_action.command,
                                        arm_backends=arm_backends,
                                        hand_backends=hand_backends,
                                    )
                                last_action = canonical_action.action.vector.astype(np.float32, copy=True)
                            if debug_logger.enabled:
                                post_arm_states = _debug_backend_arm_states(arm_backends)
                        except Exception as exc:  # noqa: BLE001
                            dispatch_error = exc
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
                                )
                            raise
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
                        )

                fault_text = _compact_faults(faults)
                if fault_text != last_fault_text:
                    print(f"[State] faults={fault_text}")
                    last_fault_text = fault_text

                if loop_started_at - last_status_at >= float(args.status_sec):
                    policy_status = _policy_status_text(policy)
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
                        )
                        if text
                    ]
                    suffix = f" {' '.join(status_parts)}" if status_parts else ""
                    print(
                        "[Status] "
                        f"step={step_index} hz={control_hz:.2f} "
                        f"policy={'active' if policy_active else 'paused'} "
                        f"send={'on' if send_actions else 'dry-run'} "
                        f"queue={len(chunk_buffer)} "
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
        if fixed_init_controller is not None:
            fixed_init_controller.restore_speed(arm_backends)
        for backend in arm_backends.values():
            backend.close()
        for backend in hand_backends.values():
            backend.close()
        for camera_source in camera_sources.values():
            camera_source.close()
        if camera_previewer is not None:
            camera_previewer.close()
        if visualizer is not None:
            visualizer.close()
        debug_logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
