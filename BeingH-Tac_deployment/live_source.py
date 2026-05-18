from __future__ import annotations

import argparse
from collections import deque
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from paths import ensure_import_paths
from profiles import DeploymentProfile
from tactile import TactileHistory


ensure_import_paths()

from inspire_data_collect.common import EffectorCommand as FR3EffectorCommand  # noqa: E402
from inspire_data_collect.arm.fr3_backend import FR3ArmBackend, FR3ArmBackendConfig  # noqa: E402
from inspire_data_collect.hand.hand_backend import (  # noqa: E402
    InspireTacHandBackend as FR3InspireTacHandBackend,
    InspireTacHandBackendConfig as FR3InspireTacHandBackendConfig,
)
from inspire_data_collect.deploy.actions import (  # noqa: E402
    ActionSafetyFilter as FR3ActionSafetyFilter,
    ActionSafetyFilterConfig as FR3ActionSafetyFilterConfig,
    normalize_policy_output as normalize_fr3_policy_output,
)
from inspire_data_collect.deploy.observation import (  # noqa: E402
    build_deployment_observation as build_fr3_deployment_observation,
)
from inspire_data_collect.utils.camera_preview import (  # noqa: E402
    CameraPreviewConfig as FR3CameraPreviewConfig,
    CameraPreviewer as FR3CameraPreviewer,
)
from inspire_data_collect.utils.realsense_rgb import (  # noqa: E402
    RealSenseRGBCameraConfig as FR3RealSenseRGBCameraConfig,
    RealSenseRGBSource as FR3RealSenseRGBSource,
    load_realsense_camera_config_map as load_fr3_realsense_camera_config_map,
)


ALL_SIDES = ("left", "right")
STATE_ARM_POSE_SLICES = {"left": slice(6, 13), "right": slice(25, 32)}
STATE_HAND_SLICES = {"left": slice(13, 19), "right": slice(32, 38)}
ActionSafetyFilter = None
ActionSafetyFilterConfig = None
CameraPreviewConfig = None
CameraPreviewer = None
build_arm_backends = None
build_camera_sources = None
build_deployment_observation = None
build_hand_backends = None
normalize_policy_output = None
select_preview_camera_sources = None
EffectorCommand = None


def _ensure_rm65_imports() -> None:
    global ActionSafetyFilter
    global ActionSafetyFilterConfig
    global CameraPreviewConfig
    global CameraPreviewer
    global EffectorCommand
    global build_arm_backends
    global build_camera_sources
    global build_deployment_observation
    global build_hand_backends
    global normalize_policy_output
    global select_preview_camera_sources
    if build_arm_backends is not None:
        return
    from teleop_runtime.collection.camera_preview import (  # noqa: E402
        CameraPreviewConfig as _CameraPreviewConfig,
        CameraPreviewer as _CameraPreviewer,
    )
    from teleop_runtime.deploy.actions import (  # noqa: E402
        ActionSafetyFilter as _ActionSafetyFilter,
        ActionSafetyFilterConfig as _ActionSafetyFilterConfig,
        normalize_policy_output as _normalize_policy_output,
    )
    from teleop_runtime.deploy.observation import build_deployment_observation as _build_deployment_observation  # noqa: E402
    from teleop_runtime.deploy.run_policy import (  # noqa: E402
        build_arm_backends as _build_arm_backends,
        build_camera_sources as _build_camera_sources,
        build_hand_backends as _build_hand_backends,
        select_preview_camera_sources as _select_preview_camera_sources,
    )
    from teleop_runtime.schemas import EffectorCommand as _EffectorCommand  # noqa: E402

    ActionSafetyFilter = _ActionSafetyFilter
    ActionSafetyFilterConfig = _ActionSafetyFilterConfig
    CameraPreviewConfig = _CameraPreviewConfig
    CameraPreviewer = _CameraPreviewer
    EffectorCommand = _EffectorCommand
    build_arm_backends = _build_arm_backends
    build_camera_sources = _build_camera_sources
    build_deployment_observation = _build_deployment_observation
    build_hand_backends = _build_hand_backends
    normalize_policy_output = _normalize_policy_output
    select_preview_camera_sources = _select_preview_camera_sources


def add_live_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--active-sides", choices=["left", "right", "both"], default="both")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--policy-hz", type=float, default=10.0)
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--send-actions", action="store_true")
    parser.add_argument("--allow-arm-connect-failure", action="store_true")
    parser.add_argument("--allow-hand-connect-failure", action="store_true")
    parser.add_argument("--left-arm-ip", type=str, default="")
    parser.add_argument("--right-arm-ip", type=str, default="")
    parser.add_argument("--arm-urdf-path", type=str, default="")
    parser.add_argument("--arm-max-angvel", type=str, default="0.35,0.35,0.35,0.5,0.5,0.5")
    parser.add_argument("--arm-state-refresh-hz", type=float, default=10.0)
    parser.add_argument("--rm65-hardware-read-retries", type=int, default=1)
    parser.add_argument("--rm65-hardware-read-retry-delay-sec", type=float, default=0.02)
    parser.add_argument("--left-initial-qpos", type=str, default=None)
    parser.add_argument("--right-initial-qpos", type=str, default=None)
    parser.add_argument("--enable-hands", dest="hands_enabled", action="store_true", default=None)
    parser.add_argument("--no-hands", dest="hands_enabled", action="store_false")
    parser.add_argument("--left-hand-ip", type=str, default="")
    parser.add_argument("--right-hand-ip", type=str, default="")
    parser.add_argument("--hand-modbus-port", type=int, default=6000)
    parser.add_argument("--hand-send-threshold", type=float, default=0.002)
    parser.add_argument("--hand-telemetry-hz", type=float, default=20.0)
    parser.add_argument("--hand-tactile-regions", "--tactile-regions", dest="hand_tactile_regions", type=str, default="all")
    parser.add_argument(
        "--inspire-hand-max-force",
        "--max-force",
        "--max-fingertip-force",
        dest="max_fingertip_force",
        type=int,
        default=800,
    )
    parser.add_argument("--camera-mode", choices=["none", "single", "multi"], default="none")
    parser.add_argument(
        "--camera-config",
        type=str,
        default="/home/ai/Desktop/DexTeleop/teleop_runtime/configs/devices/realsense.json",
    )
    parser.add_argument("--camera-key", type=str, default="front")
    parser.add_argument("--camera-serial", type=str, default="")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-keep-last", type=int, default=4)
    parser.add_argument("--camera-color-space", choices=["rgb", "bgr"], default="rgb")
    parser.add_argument("--camera-wait-timeout-ms", type=int, default=1000)
    parser.add_argument("--camera-stale-after-sec", type=float, default=2.0)
    parser.add_argument("--camera-start-retry-attempts", type=int, default=8)
    parser.add_argument("--camera-start-retry-delay-sec", type=float, default=0.5)
    parser.add_argument("--camera-auto-exposure-priority", type=int, default=None)
    parser.add_argument("--live-camera-preview", dest="live_camera_preview", action="store_true", default=True)
    parser.add_argument("--no-live-camera-preview", dest="live_camera_preview", action="store_false")
    parser.add_argument("--vis-cam-key", "--vis_camera_key", dest="vis_cam_key", type=str, default="")
    parser.add_argument("--arm-workspace-min", type=str, default="-0.60,-0.80,0.00")
    parser.add_argument("--arm-workspace-max", type=str, default="0.60,0.80,0.90")
    parser.add_argument("--max-arm-xyz-step-m", type=float, default=0.03)
    parser.add_argument("--max-arm-rot-step-deg", type=float, default=8.0)
    parser.add_argument("--max-arm-joint-step-rad", type=float, default=0.08)
    parser.add_argument("--max-hand-step", type=float, default=0.08)
    parser.add_argument("--live-ready-timeout-s", type=float, default=5.0)
    parser.add_argument("--live-ready-poll-s", type=float, default=0.05)
    parser.add_argument("--action-hz", type=float, default=12.0)
    parser.add_argument("--action-queue-policy", choices=["replace", "append"], default="replace")
    parser.add_argument("--action-underrun-log-sec", type=float, default=2.0)


def _parse_sides(value: str) -> tuple[str, ...]:
    if value == "both":
        return ("left", "right")
    return (value,)


def _parse_float_tuple(value: str | None, length: int) -> tuple[float, ...] | None:
    if value in (None, ""):
        return None
    parts = [float(x.strip()) for x in str(value).split(",") if x.strip()]
    if len(parts) == 1 and length == 6:
        parts = parts * 6
    if len(parts) != length:
        raise ValueError(f"Expected {length} comma-separated floats, got {value!r}.")
    return tuple(parts)


def _hands_enabled(args: argparse.Namespace) -> bool:
    if args.hands_enabled is not None:
        return bool(args.hands_enabled)
    return bool(args.left_hand_ip or args.right_hand_ip)


def _finite_vector(value: Any, width: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] != width or not np.all(np.isfinite(arr)):
        print(f"WARNING: live {name} invalid or unavailable; using zeros with shape ({width},).")
        return np.zeros(width, dtype=np.float32)
    return arr.astype(np.float32, copy=True)


def _normalize_camera_serial(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {
        "",
        "auto",
        "none",
        "null",
        "your_realsense_serial",
        "your-realsense-serial",
        "<your_realsense_serial>",
    }:
        return ""
    return text


def _normalize_endpoint_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {
        "",
        "none",
        "null",
        "auto",
        "$left_arm_ip",
        "$right_arm_ip",
        "$left_hand_ip",
        "$right_hand_ip",
        "${left_arm_ip}",
        "${right_arm_ip}",
        "${left_hand_ip}",
        "${right_hand_ip}",
    }:
        return ""
    return text


def _coerce_builder_args(args: argparse.Namespace) -> argparse.Namespace:
    payload = vars(args).copy()
    payload["active_sides"] = _parse_sides(str(args.active_sides))
    payload["arm_max_angvel"] = _parse_float_tuple(args.arm_max_angvel, 6)
    payload["left_initial_qpos"] = _parse_float_tuple(args.left_initial_qpos, 6)
    payload["right_initial_qpos"] = _parse_float_tuple(args.right_initial_qpos, 6)
    payload["arm_workspace_min"] = _parse_float_tuple(args.arm_workspace_min, 3)
    payload["arm_workspace_max"] = _parse_float_tuple(args.arm_workspace_max, 3)
    payload["camera_serial"] = _normalize_camera_serial(args.camera_serial)
    for key in ("left_arm_ip", "right_arm_ip", "left_hand_ip", "right_hand_ip"):
        env_key = key.upper()
        payload[key] = _normalize_endpoint_text(payload.get(key)) or _normalize_endpoint_text(os.environ.get(env_key))
    payload["arms_enabled"] = True
    if payload.get("hands_enabled") is None:
        payload["hands_enabled"] = bool(args.left_hand_ip or args.right_hand_ip)
    return argparse.Namespace(**payload)


def _required_missing_endpoints(profile: DeploymentProfile, args: argparse.Namespace) -> tuple[list[str], list[str]]:
    missing_arms = [
        side
        for side in profile.required_arms
        if not _normalize_endpoint_text(getattr(args, f"{side}_arm_ip", ""))
    ]
    missing_hands = [
        side
        for side in profile.required_hands
        if not _normalize_endpoint_text(getattr(args, f"{side}_hand_ip", ""))
    ]
    return missing_arms, missing_hands


def _is_fr3_profile(profile: DeploymentProfile) -> bool:
    return str(getattr(profile, "arm_backend", "rm65")).lower() == "fr3"


def _fr3_host_for_side(args: argparse.Namespace, side: str) -> str:
    return _normalize_endpoint_text(getattr(args, f"{side}_arm_ip", ""))


class GenericArmStateReader:
    def __init__(
        self,
        backends: Mapping[str, Any],
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
            state: dict[str, Any]
            if isinstance(backend, FR3ArmBackend):
                raw = backend.get_state()
                qpos = raw.get("qpos")
                raw_state = raw.get("raw_state") or {}
                current_qpos = None if qpos is None else np.asarray(qpos, dtype=np.float32).reshape(-1)
                state = {
                    "qpos": None
                    if current_qpos is None or current_qpos.shape != (7,)
                    else current_qpos.astype(np.float32, copy=True),
                    "current_qpos": None
                    if current_qpos is None or current_qpos.shape != (7,)
                    else current_qpos.astype(np.float32, copy=True),
                    "current_tcp_pose": None,
                    "last_command": None,
                    "state_error": raw.get("state_error"),
                    "hardware_result_code": 0 if backend.is_ready() else -1,
                    "hardware_result_text": "ready" if backend.is_ready() else str(raw.get("state_error") or "not_ready"),
                    "raw_state": dict(raw_state) if isinstance(raw_state, Mapping) else {},
                    "startup_alignment_enabled": raw.get("startup_alignment_enabled"),
                    "startup_alignment_active": raw.get("startup_alignment_active"),
                    "startup_alignment_complete": raw.get("startup_alignment_complete"),
                    "startup_alignment_progress": raw.get("startup_alignment_progress"),
                    "controller_timestamp": raw.get("controller_timestamp"),
                    "state_receive_timestamp": raw.get("state_receive_timestamp"),
                    "state_age_sec": raw.get("state_age_sec"),
                    "server_age_s": raw.get("server_age_s"),
                }
                ee_pos = raw_state.get("ee_pos")
                ee_quat = raw_state.get("ee_quat")
                if ee_pos is not None and ee_quat is not None:
                    pose = np.concatenate(
                        [
                            np.asarray(ee_pos, dtype=np.float32).reshape(-1)[:3],
                            np.asarray(ee_quat, dtype=np.float32).reshape(-1)[:4],
                        ],
                        axis=0,
                    )
                    if pose.shape == (7,) and np.all(np.isfinite(pose)):
                        state["current_tcp_pose"] = np.asarray(pose, dtype=np.float32).copy()
                states[side] = state
                continue

            if (
                refresh_period > 0.0
                and hasattr(backend, "is_ready")
                and backend.is_ready()
                and now - self._last_refresh_at.get(side, 0.0) >= refresh_period
            ):
                try:
                    backend.refresh_state_from_hardware()
                except Exception as exc:
                    state = backend.get_state()
                    state["state_error"] = f"refresh_failed:{type(exc).__name__}:{exc}"
                    states[side] = state
                    continue
                self._last_refresh_at[side] = now
            state = backend.get_state()
            if self.compute_tcp_pose:
                try:
                    state["current_tcp_pose"] = backend.compute_tcp_pose()
                except Exception as exc:
                    state.setdefault("pose_error", f"compute_tcp_pose_failed:{type(exc).__name__}:{exc}")
            states[side] = state
        return states


def build_profile_arm_backends(
    profile: DeploymentProfile,
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
    *,
    control_hz: float,
) -> dict[str, Any]:
    if _is_fr3_profile(profile):
        backends: dict[str, Any] = {}
        for side in active_sides:
            host = _fr3_host_for_side(args, side)
            backends[side] = FR3ArmBackend(
                FR3ArmBackendConfig(
                    host=host or "0.0.0.0",
                    port=5558,
                    timeout_s=5.0,
                    side=side,
                    effector_name="arm",
                    connect_hardware=bool(host),
                    state_poll_hz=float(getattr(args, "arm_state_refresh_hz", 20.0)),
                    max_state_age_s=0.5,
                    wait_state_timeout_s=max(0.1, float(getattr(args, "live_ready_timeout_s", 5.0))),
                    startup_align_enabled=False,
                    silent_status=True,
                )
            )
        return backends
    _ensure_rm65_imports()
    return build_arm_backends(args, active_sides, control_hz=float(control_hz))


def build_profile_hand_backends(
    profile: DeploymentProfile,
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> dict[str, Any]:
    if _is_fr3_profile(profile):
        tactile_regions = tuple(str(x).strip() for x in str(args.hand_tactile_regions).split(",") if str(x).strip())
        if not tactile_regions:
            tactile_regions = ("all",)
        backends: dict[str, Any] = {}
        for side in active_sides:
            hand_ip = str(getattr(args, f"{side}_hand_ip") or "")
            backends[side] = FR3InspireTacHandBackend(
                FR3InspireTacHandBackendConfig(
                    hand_ip=hand_ip or "0.0.0.0",
                    side=side,
                    effector_name=f"{side}_hand",
                    port=int(args.hand_modbus_port),
                    connect_hardware=bool(hand_ip),
                    max_fingertip_force=int(args.max_fingertip_force),
                    send_threshold=float(args.hand_send_threshold),
                    enable_telemetry=bool(hand_ip),
                    telemetry_hz=float(args.hand_telemetry_hz),
                    tactile_regions=tactile_regions,
                )
            )
        return backends
    _ensure_rm65_imports()
    return build_hand_backends(args, active_sides)


def build_profile_camera_sources(profile: DeploymentProfile, args: argparse.Namespace) -> dict[str, Any]:
    if _is_fr3_profile(profile):
        if args.camera_mode == "none":
            return {}
        if args.camera_serial:
            if args.camera_mode != "single":
                raise ValueError("--camera-serial can only be used with --camera-mode single.")
            config = FR3RealSenseRGBCameraConfig(
                serial=str(args.camera_serial),
                width=int(args.camera_width),
                height=int(args.camera_height),
                fps=int(args.camera_fps),
                camera_key=str(args.camera_key),
                keep_last=int(args.camera_keep_last),
                color_space=str(args.camera_color_space),
                wait_timeout_ms=int(args.camera_wait_timeout_ms),
                stale_after_sec=float(args.camera_stale_after_sec),
                start_retry_attempts=int(args.camera_start_retry_attempts),
                start_retry_delay_sec=float(args.camera_start_retry_delay_sec),
                auto_exposure_priority=args.camera_auto_exposure_priority,
                metadata={"source": "beingh_tac_fr3_live_client"},
            )
            return {config.camera_key: FR3RealSenseRGBSource(config)}
        camera_configs = load_fr3_realsense_camera_config_map(args.camera_config)
        if args.camera_mode == "single" and len(camera_configs) != 1:
            raise ValueError(f"--camera-mode single expects exactly one camera config, got {len(camera_configs)}.")
        if args.camera_mode == "multi" and not camera_configs:
            raise ValueError("--camera-mode multi requires at least one camera config.")
        return {key: FR3RealSenseRGBSource(config) for key, config in camera_configs.items()}
    _ensure_rm65_imports()
    return build_camera_sources(args)


def select_profile_preview_camera_sources(profile: DeploymentProfile, camera_sources: dict[str, Any], requested_key: str) -> dict[str, Any]:
    if not camera_sources:
        return {}
    if not _is_fr3_profile(profile):
        _ensure_rm65_imports()
        return select_preview_camera_sources(camera_sources, requested_key)
    key = str(requested_key or "").strip()
    if key:
        if key not in camera_sources:
            available = ", ".join(sorted(camera_sources))
            raise ValueError(f"--vis-cam-key {key!r} not found. Available camera keys: {available}")
        return {key: camera_sources[key]}
    selected_key = "front" if "front" in camera_sources else sorted(camera_sources)[0]
    return {selected_key: camera_sources[selected_key]}


def build_profile_observation(
    profile: DeploymentProfile,
    *,
    active_sides: tuple[str, ...],
    arms_enabled: bool,
    hands_enabled: bool,
    step_index: int,
    arm_states: Mapping[str, Mapping[str, Any] | None],
    hand_states: Mapping[str, Mapping[str, Any] | None],
    camera_sources: Mapping[str, Any] | None = None,
    last_action: np.ndarray | None = None,
    timestamp: float | None = None,
    require_images: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    if _is_fr3_profile(profile):
        active = tuple(side for side in active_sides if side in ALL_SIDES)
        arm_side = active[0] if active else "right"
        return build_fr3_deployment_observation(
            active_sides=active,
            arm_enabled=bool(arms_enabled),
            step_index=step_index,
            arm_state=arm_states.get(arm_side),
            hand_states=hand_states,
            camera_sources=camera_sources,
            last_action=last_action,
            timestamp=timestamp,
            require_images=require_images,
            metadata=metadata,
        )
    _ensure_rm65_imports()
    return build_deployment_observation(
        active_sides=active_sides,
        arms_enabled=bool(arms_enabled),
        hands_enabled=bool(hands_enabled),
        step_index=step_index,
        arm_states=arm_states,
        hand_states=hand_states,
        camera_sources=camera_sources,
        last_action=last_action,
        timestamp=timestamp,
        require_images=require_images,
        metadata=metadata,
    )


def hand_state_snapshot_for_backends(hand_backends: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for side, backend in hand_backends.items():
        if hasattr(backend, "get_state"):
            states[side] = dict(backend.get_state())
    return states


@dataclass
class LiveSnapshot:
    payload_base: dict[str, Any]
    observation: Any


def server_result_for_deploy(result: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    aliases = {
        "action.left_arm.pose": "left_arm_pose",
        "action.right_arm.pose": "right_arm_pose",
        "action.left_hand.joint_target": "left_hand_joint_target",
        "action.right_hand.joint_target": "right_hand_joint_target",
    }
    for src, dst in aliases.items():
        if src in result:
            output[dst] = result[src]
    return output


def fr3_server_result_for_deploy(result: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    aliases = {
        "action.arm.joint_target": "arm_joint_target",
        "action.replay.arm_joint_target": "arm_joint_target",
        "action.right_hand.joint_target": "right_hand_joint_target",
        "action.left_hand.joint_target": "left_hand_joint_target",
    }
    for src, dst in aliases.items():
        if src in result:
            output[dst] = result[src]
    return output


class _FR3PoseIK:
    def __init__(self) -> None:
        try:
            import pinocchio as pin
        except Exception as exc:
            raise RuntimeError("Converting Franka EEF pose actions to joints requires pinocchio.") from exc

        self.pin = pin
        urdf = Path(
            os.environ.get(
                "FR3_IK_URDF",
                "/home/user/zc/fr3_handeye/assets/franka_description/fr3.urdf",
            )
        ).expanduser()
        if not urdf.exists():
            raise FileNotFoundError(f"FR3 IK URDF not found: {urdf}")
        ee_link = os.environ.get("FR3_IK_EE_LINK", "fr3_link8")
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(ee_link)
        self.arm_dof = 7
        self.q_min = np.asarray(self.model.lowerPositionLimit[: self.arm_dof], dtype=np.float64)
        self.q_max = np.asarray(self.model.upperPositionLimit[: self.arm_dof], dtype=np.float64)

    def solve_pose(
        self,
        pose: Any,
        q_seed: Any,
        *,
        max_iter: int = 100,
        tol: float = 1e-4,
        damping: float = 1e-4,
        max_step_rad: float = 0.03,
    ) -> np.ndarray:
        pin = self.pin
        target = np.asarray(pose, dtype=np.float64).reshape(-1)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError(f"FR3 IK target pose must be finite pose7, got {target.shape}.")
        quat = target[3:7].copy()
        quat_norm = float(np.linalg.norm(quat))
        if quat_norm <= 1e-9:
            raise ValueError("FR3 IK target quaternion has zero norm.")
        quat /= quat_norm
        o_m_des = pin.SE3(pin.Quaternion(quat[3], quat[0], quat[1], quat[2]).matrix(), target[:3])

        q_seed_arr = np.asarray(q_seed, dtype=np.float64).reshape(-1)
        if q_seed_arr.shape[0] < self.arm_dof or not np.all(np.isfinite(q_seed_arr[: self.arm_dof])):
            q_arm = np.zeros((self.arm_dof,), dtype=np.float64)
        else:
            q_arm = np.clip(q_seed_arr[: self.arm_dof], self.q_min, self.q_max)
        q = np.concatenate([q_arm, np.zeros(max(0, self.model.nq - self.arm_dof), dtype=np.float64)])

        for _ in range(max(1, int(max_iter))):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[self.ee_frame_id]
            err = pin.log(o_m_des.actInv(current)).vector
            if float(np.linalg.norm(err)) < float(tol):
                break
            jac = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.ee_frame_id,
                pin.ReferenceFrame.LOCAL,
            )[:, : self.arm_dof]
            delta = -jac.T.dot(np.linalg.solve(jac.dot(jac.T) + float(damping) * np.eye(6), err))
            delta = np.clip(delta, -float(max_step_rad), float(max_step_rad))
            q[: self.arm_dof] = np.clip(q[: self.arm_dof] + delta, self.q_min, self.q_max)
        return q[: self.arm_dof].astype(np.float32, copy=True)


_FR3_POSE_IK: _FR3PoseIK | None = None


def _fr3_pose_ik() -> _FR3PoseIK:
    global _FR3_POSE_IK
    if _FR3_POSE_IK is None:
        _FR3_POSE_IK = _FR3PoseIK()
    return _FR3_POSE_IK


def _fr3_pose_chunk_to_joint_chunk(value: Any, observation: Any) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float32)
    one_dim = poses.ndim == 1
    if one_dim:
        poses = poses.reshape(1, -1)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"FR3 arm pose action must have shape (7,) or (N,7), got {poses.shape}.")
    solver = _fr3_pose_ik()
    seed = np.asarray(getattr(observation, "state", np.zeros(19, dtype=np.float32))[:7], dtype=np.float32)
    outputs = []
    for pose in poses:
        seed = solver.solve_pose(pose, seed)
        outputs.append(seed.astype(np.float32, copy=True))
    chunk = np.stack(outputs, axis=0).astype(np.float32)
    return chunk[0] if one_dim else chunk


def _fr3_expand_compact_13_action(arr: np.ndarray) -> np.ndarray:
    """Expand Franka compact [arm(7), right_hand(6)] action to full 19-D layout."""
    if arr.ndim == 1:
        full = np.full((19,), np.nan, dtype=np.float32)
        full[:7] = arr[:7]
        full[7:13] = arr[7:13]
        return full
    full = np.full((arr.shape[0], 19), np.nan, dtype=np.float32)
    full[:, :7] = arr[:, :7]
    full[:, 7:13] = arr[:, 7:13]
    return full


def _coerce_fr3_result_action(result: Mapping[str, Any], observation: Any) -> Mapping[str, Any] | dict[str, Any]:
    arm_pose = result.get("action.arm.pose")
    action_raw_format = str(result.get("action_raw_format", "")).lower()
    raw_arr = None
    if "action_raw" in result:
        raw_arr = np.asarray(result["action_raw"], dtype=np.float32)
    if arm_pose is None and raw_arr is not None and raw_arr.ndim in (1, 2):
        raw_width = raw_arr.shape[0] if raw_arr.ndim == 1 else raw_arr.shape[1]
        if raw_width == 13 and "arm_pose" in action_raw_format:
            arm_pose = raw_arr[:7] if raw_arr.ndim == 1 else raw_arr[:, :7]
    if arm_pose is not None:
        output: dict[str, Any] = {
            "arm_joint_target": _fr3_pose_chunk_to_joint_chunk(arm_pose, observation),
        }
        if "action.right_hand.joint_target" in result:
            output["right_hand_joint_target"] = result["action.right_hand.joint_target"]
        elif raw_arr is not None:
            raw_width = raw_arr.shape[0] if raw_arr.ndim == 1 else raw_arr.shape[1]
            if raw_width >= 13:
                output["right_hand_joint_target"] = raw_arr[7:13] if raw_arr.ndim == 1 else raw_arr[:, 7:13]
        return output

    output = fr3_server_result_for_deploy(result)
    if output:
        return output
    for key in ("action", "actions", "action_chunk"):
        if key in result:
            return {key: result[key]}
    for key in ("action_raw", "action_unified"):
        if key not in result:
            continue
        arr = np.asarray(result[key], dtype=np.float32)
        if arr.ndim == 1:
            width = int(arr.shape[0])
        elif arr.ndim == 2:
            width = int(arr.shape[1])
        else:
            continue
        if width == 19:
            return {"action": arr}
        if width == 13:
            return {"action": _fr3_expand_compact_13_action(arr)}
        if width == 7:
            return {"arm_joint_target": arr}
        if width == 6:
            return {"right_hand_joint_target": arr}
    return {}


def _format_action_vector(value: Any, *, precision: int = 4) -> str:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return np.array2string(arr, precision=precision, separator=", ", suppress_small=False)


def _format_fr3_split(prefix: str, vector: Any) -> str:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 19:
        padded = np.full((19,), np.nan, dtype=np.float32)
        padded[: min(arr.shape[0], 19)] = arr[:19]
        arr = padded
    return (
        f"{prefix} arm={_format_action_vector(arr[:7])} "
        f"right_hand={_format_action_vector(arr[7:13])} "
        f"left_hand={_format_action_vector(arr[13:19])}"
    )


def _first_action_row(value: Any) -> np.ndarray | None:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(-1)
    if arr.ndim == 2 and arr.shape[0] > 0:
        return arr[0].reshape(-1)
    return None


def _action_rows(value: Any) -> np.ndarray | None:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2 and arr.shape[0] > 0:
        return arr
    return None


def _format_minmax(label: str, value: Any) -> str:
    arr = np.asarray(value, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return f"{label}=all_nan"
    return f"{label}=[{float(finite.min()):.4f}, {float(finite.max()):.4f}]"


def format_fr3_policy_debug(result: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in ("action_unified", "action_raw", "action", "actions", "action_chunk"):
        if key not in result:
            continue
        arr = np.asarray(result[key], dtype=np.float32)
        first = _first_action_row(arr)
        rows = _action_rows(arr)
        if first is None:
            lines.append(f"[ActionDebug] policy {key}: shape={arr.shape}")
            continue
        preview_width = min(first.shape[0], 32)
        summary = f"[ActionDebug] policy {key}: shape={arr.shape} first{preview_width}={_format_action_vector(first[:preview_width])}"
        if rows is not None and rows.shape[1] >= 13:
            summary += (
                f" {_format_minmax('arm0', rows[:, 0])}"
                f" {_format_minmax('arm6', rows[:, 6])}"
                f" {_format_minmax('hand0', rows[:, 7])}"
                f" {_format_minmax('hand5', rows[:, 12])}"
            )
        lines.append(summary)
    return lines


def format_fr3_dataset_gt_debug(gt_action: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(gt_action, Mapping):
        return []
    source = str(gt_action.get("source", "unknown"))
    full = gt_action.get("full")
    if full is None:
        return [f"[ActionDebug] dataset_gt source={source} missing"]
    return [f"[ActionDebug] dataset_gt source={source} {_format_fr3_split('gt', full)}"]


def _send_commands(
    commands: Mapping[str, Any],
    *,
    arm_backends: Mapping[str, Any],
    hand_backends: Mapping[str, Any],
) -> None:
    arm_command = commands.get("arm")
    if arm_command is not None:
        side = str(getattr(arm_command, "side", "") or "")
        backend = arm_backends.get(side)
        if backend is not None:
            backend.send_command(arm_command)
    for side in ALL_SIDES:
        arm_command = commands.get(f"{side}_arm")
        if arm_command is not None and side in arm_backends:
            arm_backends[side].send_command(arm_command)
        hand_command = commands.get(f"{side}_hand")
        if hand_command is not None and side in hand_backends:
            hand_backends[side].send_command(hand_command)


def _send_direct_joint_targets(
    *,
    arm_backends: Mapping[str, Any],
    hand_backends: Mapping[str, Any],
    backend_kind: str,
    arm_qpos: Mapping[str, Any] | None = None,
    hand_qpos: Mapping[str, Any] | None = None,
) -> list[str]:
    statuses: list[str] = []
    arm_qpos = {} if arm_qpos is None else dict(arm_qpos)
    hand_qpos = {} if hand_qpos is None else dict(hand_qpos)

    for side, value in arm_qpos.items():
        if side not in arm_backends:
            statuses.append(f"{side}_arm_unavailable")
            continue
        target = np.asarray(value, dtype=np.float32).reshape(-1)
        expected_width = 7 if backend_kind == "fr3" else 6
        if target.shape != (expected_width,) or not np.all(np.isfinite(target)):
            statuses.append(f"{side}_arm_direct_qpos_invalid")
            continue
        backend = arm_backends[side]
        arm = getattr(backend, "arm", None)
        if arm is not None:
            arm.Movej_CANFD(np.asarray(target, dtype=float) / np.pi * 180.0, False)
            if hasattr(backend, "set_local_qpos"):
                backend.set_local_qpos(target)
            statuses.append(f"{side}_arm_direct_sent")
        else:
            if backend_kind == "fr3":
                command = FR3EffectorCommand(
                    effector_name="arm",
                    side=side,
                    value=target,
                    timestamp=time.time(),
                    metadata={"source": "startup_direct_init"},
                )
            else:
                _ensure_rm65_imports()
                command = EffectorCommand(
                    effector_name=f"{side}_arm",
                    side=side,
                    value=target,
                    command_type="joint_vector",
                    frame_name=None,
                    metadata={"source": "startup_direct_init"},
                )
            backend.send_command(command)
            statuses.append(f"{side}_arm_direct_dry_run")

    for side, value in hand_qpos.items():
        if side not in hand_backends:
            statuses.append(f"{side}_hand_unavailable")
            continue
        target = np.asarray(value, dtype=np.float32).reshape(-1)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            statuses.append(f"{side}_hand_direct_qpos_invalid")
            continue
        clipped = np.clip(target, 0.0, 1.0)
        if backend_kind == "fr3":
            command = FR3EffectorCommand(
                effector_name=f"{side}_hand",
                side=side,
                value=clipped,
                timestamp=time.time(),
                metadata={"source": "startup_direct_init"},
            )
        else:
            _ensure_rm65_imports()
            command = EffectorCommand(
                effector_name=f"{side}_hand",
                side=side,
                value=clipped,
                command_type="joint_vector",
                frame_name=None,
                metadata={"source": "startup_direct_init"},
            )
        hand_backends[side].send_command(command)
        statuses.append(f"{side}_hand_direct_sent")
    return statuses


class TimedActionDispatcher:
    """Consume model action chunks at a fixed robot-control frequency."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        active_sides: tuple[str, ...],
        arms_enabled: bool,
        hands_enabled: bool,
        arm_backends: Mapping[str, Any],
        hand_backends: Mapping[str, Any],
        safety_filter: ActionSafetyFilter,
        get_observation: Callable[[], Any],
        get_last_action: Callable[[], np.ndarray],
        set_last_action: Callable[[np.ndarray], None],
        policy_name: str,
        send_actions: bool,
        backend_kind: str = "rm65",
    ):
        self.args = args
        self.active_sides = active_sides
        self.arms_enabled = bool(arms_enabled)
        self.hands_enabled = bool(hands_enabled)
        self.arm_backends = dict(arm_backends)
        self.hand_backends = dict(hand_backends)
        self.safety_filter = safety_filter
        self.get_observation = get_observation
        self.get_last_action = get_last_action
        self.set_last_action = set_last_action
        self.policy_name = str(policy_name)
        self.send_actions = bool(send_actions)
        self.backend_kind = str(backend_kind)
        self.action_hz = max(1e-6, float(getattr(args, "action_hz", 12.0)))
        self.queue_policy = str(getattr(args, "action_queue_policy", "replace"))
        self.underrun_log_sec = max(0.0, float(getattr(args, "action_underrun_log_sec", 2.0)))
        self._queue: deque[Any] = deque()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_underrun_log_at = 0.0
        self._underrun_count = 0
        self._submitted_chunks = 0
        self._sent_actions = 0
        self._started_at = 0.0
        self._debug_submitted_chunks = 0
        self._debug_sent_actions = 0

    def _debug_enabled(self) -> bool:
        return bool(getattr(self.args, "debug_fr3_actions", False)) and self.backend_kind == "fr3"

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._started_at = time.monotonic()
        print(
            f"[Action] dispatcher started action_hz={self.action_hz:.3g} "
            f"queue_policy={self.queue_policy} send_actions={self.send_actions}"
        )
        self._thread = threading.Thread(
            target=self._loop,
            name=f"{self.policy_name}_action_dispatcher",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._lock:
            self._queue.clear()

    def submit_result(self, result: Mapping[str, Any]) -> list[str]:
        observation = self.get_observation()
        if self.backend_kind == "fr3":
            output = _coerce_fr3_result_action(result, observation)
            if not output:
                return ["missing_action"]
            actions = normalize_fr3_policy_output(
                output,
                observation,
                active_sides=self.active_sides,
                arm_enabled=self.arms_enabled,
            )
        else:
            _ensure_rm65_imports()
            output: Mapping[str, Any] | dict[str, np.ndarray] = server_result_for_deploy(result)
            if not output:
                if any(key in result for key in ("action", "actions", "action_chunk")):
                    output = result
                elif "action_raw" in result:
                    output = {"action": np.asarray(result["action_raw"], dtype=np.float32)}
                elif "action_unified" in result:
                    output = {"action": np.asarray(result["action_unified"], dtype=np.float32)}
                else:
                    return ["missing_action"]
            actions = normalize_policy_output(
                output,
                observation,
                active_sides=self.active_sides,
                arms_enabled=self.arms_enabled,
                hands_enabled=self.hands_enabled,
            )
        if not actions:
            return ["empty_action_chunk"]

        if self._debug_enabled():
            first = np.asarray(actions[0], dtype=np.float32).reshape(-1)
            rows = np.asarray(actions, dtype=np.float32).reshape(len(actions), -1)
            print(
                f"[ActionDebug] normalized_chunk index={self._debug_submitted_chunks} "
                f"len={len(actions)} {_format_fr3_split('normalized0', first)} "
                f"{_format_minmax('arm0', rows[:, 0])} {_format_minmax('arm6', rows[:, 6])}"
            )
            self._debug_submitted_chunks += 1

        with self._lock:
            remaining = len(self._queue)
            if self.queue_policy == "append":
                self._queue.extend(actions)
                queued = len(self._queue)
                replaced = 0
            else:
                self._queue.clear()
                self._queue.extend(actions)
                queued = len(self._queue)
                replaced = remaining
            self._submitted_chunks += 1

        status = [f"queued={queued}", f"chunk={len(actions)}"]
        if replaced:
            status.append(f"discarded_old={replaced}")
        return status

    def _pop_action(self) -> Any | None:
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    def _queue_len(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def sent_actions(self) -> int:
        return int(self._sent_actions)

    def wait_until_idle(self, timeout_s: float | None = None, poll_s: float = 0.01) -> bool:
        deadline = None if timeout_s is None else time.monotonic() + max(0.0, float(timeout_s))
        while True:
            if self._queue_len() == 0:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(max(1e-3, float(poll_s)))

    def wait_until_sent(
        self,
        target_sent: int,
        timeout_s: float | None = None,
        poll_s: float = 0.01,
        progress: Callable[[int, int], None] | None = None,
    ) -> bool:
        target = max(0, int(target_sent))
        deadline = None if timeout_s is None else time.monotonic() + max(0.0, float(timeout_s))
        last_reported = -1
        while True:
            sent = self.sent_actions
            if progress is not None and sent != last_reported:
                progress(min(sent, target), target)
                last_reported = sent
            if sent >= target:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(max(1e-3, float(poll_s)))

    def _loop(self) -> None:
        period_s = 1.0 / self.action_hz
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_tick:
                self._stop_event.wait(next_tick - now)
                continue
            next_tick += period_s

            action = self._pop_action()
            if action is None:
                self._handle_underrun()
                continue

            try:
                observation = self.get_observation()
                if self.backend_kind == "fr3":
                    deployment_action = self.safety_filter.filter(
                        action,
                        observation,
                        timestamp=time.time(),
                        policy_name=self.policy_name,
                    )
                else:
                    deployment_action = self.safety_filter.filter(
                        action,
                        observation,
                        policy_name=self.policy_name,
                    )
                if self.send_actions:
                    _send_commands(
                        deployment_action.commands,
                        arm_backends=self.arm_backends,
                        hand_backends=self.hand_backends,
                    )
                if self._debug_enabled():
                    command_parts: list[str] = []
                    arm_command = deployment_action.commands.get("arm")
                    if arm_command is not None:
                        raw_arm = None
                        safety = None
                        metadata = getattr(arm_command, "metadata", None)
                        if isinstance(metadata, Mapping):
                            raw_arm = metadata.get("raw_action")
                            safety = metadata.get("safety")
                        command_parts.append(f"cmd_arm={_format_action_vector(arm_command.value)}")
                        if raw_arm is not None:
                            command_parts.append(f"raw_arm={_format_action_vector(raw_arm)}")
                        if isinstance(safety, Mapping):
                            command_parts.append(
                                "arm_safety="
                                f"clip={bool(safety.get('safe_clip', False))},"
                                f"rate_limit={bool(safety.get('rate_limited', False))},"
                                f"max_step={float(safety.get('max_step_rad', 0.0)):.4f}"
                            )
                    for name in ("right_hand", "left_hand"):
                        command = deployment_action.commands.get(name)
                        if command is not None:
                            command_parts.append(f"cmd_{name}={_format_action_vector(command.value)}")
                    if self.active_sides:
                        side = self.active_sides[0]
                        backend = self.arm_backends.get(side)
                        if backend is not None and hasattr(backend, "get_state"):
                            try:
                                state = backend.get_state()
                            except Exception:
                                state = {}
                            if isinstance(state, Mapping):
                                measured_qpos = state.get("qpos")
                                last_sent_command = state.get("last_sent_command")
                                if measured_qpos is not None:
                                    command_parts.append(f"measured_qpos={_format_action_vector(measured_qpos)}")
                                if last_sent_command is not None:
                                    command_parts.append(f"backend_last_sent={_format_action_vector(last_sent_command)}")
                    commands_text = " ".join(command_parts) if command_parts else "commands=none"
                    print(
                        f"[ActionDebug] dispatched index={self._debug_sent_actions} "
                        f"{_format_fr3_split('filtered', deployment_action.vector)} {commands_text}"
                    )
                    self._debug_sent_actions += 1
                self.set_last_action(deployment_action.vector.astype(np.float32, copy=True))
                self._sent_actions += 1
                if deployment_action.faults:
                    print(
                        "[Action] dispatch faults/status: "
                        f"{';'.join(deployment_action.faults[:6])} queue={self._queue_len()}"
                    )
            except Exception as exc:
                print(f"WARNING: action dispatch failed: {type(exc).__name__}: {exc}")

    def _handle_underrun(self) -> None:
        self._underrun_count += 1
        if self.underrun_log_sec <= 0.0:
            return
        now = time.monotonic()
        if now - self._started_at < self.underrun_log_sec:
            return
        if now - self._last_underrun_log_at >= self.underrun_log_sec:
            self._last_underrun_log_at = now
            print(
                "[Action] queue underrun; holding last target until a new chunk arrives "
                f"(underruns={self._underrun_count}, sent={self._sent_actions})"
            )


class LiveObservationSource:
    def __init__(self, profile: DeploymentProfile, args: argparse.Namespace, history_size: int, history_stride: int):
        self.profile = profile
        self.backend_kind = "fr3" if _is_fr3_profile(profile) else "rm65"
        self.args = _coerce_builder_args(args)
        self.active_sides = tuple(side for side in ALL_SIDES if side in set(self.args.active_sides))
        required = set(profile.required_arms) | set(profile.required_hands)
        missing = sorted(required - set(self.active_sides))
        if missing:
            raise ValueError(f"{profile.data_config_name} live mode requires active sides {sorted(required)}, missing {missing}.")
        self.hands_enabled = bool(_hands_enabled(self.args) or profile.required_hands)
        self.args.hands_enabled = self.hands_enabled
        missing_arms, missing_hands = _required_missing_endpoints(profile, self.args)
        if missing_arms and not bool(self.args.allow_arm_connect_failure):
            flags = ", ".join(f"--{side}-arm-ip" for side in missing_arms)
            raise ValueError(
                f"{profile.data_config_name} live mode requires hardware arm IP for: {', '.join(missing_arms)}. "
                f"Set {flags}, or pass --allow-arm-connect-failure for dry-run debugging."
            )
        if missing_hands and not bool(self.args.allow_hand_connect_failure):
            flags = ", ".join(f"--{side}-hand-ip" for side in missing_hands)
            raise ValueError(
                f"{profile.data_config_name} live mode requires hardware hand IP for: {', '.join(missing_hands)}. "
                f"Set {flags}, or pass --allow-hand-connect-failure for dry-run debugging."
            )
        self.arm_backends = build_profile_arm_backends(
            profile,
            self.args,
            self.active_sides,
            control_hz=float(self.args.action_hz),
        )
        self.hand_backends = build_profile_hand_backends(profile, self.args, self.active_sides)
        self.arm_reader = GenericArmStateReader(
            self.arm_backends,
            refresh_hz=float(self.args.arm_state_refresh_hz),
            compute_tcp_pose=True,
        )
        self.camera_sources = build_profile_camera_sources(profile, self.args)
        self.preview_camera_sources = (
            select_profile_preview_camera_sources(profile, self.camera_sources, str(getattr(self.args, "vis_cam_key", "")))
            if bool(getattr(self.args, "live_camera_preview", True)) and self.camera_sources
            else {}
        )
        self.camera_previewer = (
            (FR3CameraPreviewer(FR3CameraPreviewConfig(window_prefix="beingh_live")) if self.backend_kind == "fr3" else CameraPreviewer(CameraPreviewConfig(window_prefix="beingh_live")))
            if self.preview_camera_sources
            else None
        )
        self.history = TactileHistory(history_size, history_stride)
        self.last_action = np.full(19 if self.backend_kind == "fr3" else 26, np.nan, dtype=np.float32)
        self.step_index = 0
        self.target_image_height = int(getattr(args, "fallback_image_height", 480))
        self.target_image_width = int(getattr(args, "fallback_image_width", 640))
        if self.backend_kind == "fr3":
            self.safety_filter = FR3ActionSafetyFilter(
                FR3ActionSafetyFilterConfig(
                    active_sides=self.active_sides,
                    arm_enabled=True,
                    max_arm_step_rad=float(self.args.max_arm_joint_step_rad),
                    max_hand_step=float(self.args.max_hand_step),
                )
            )
        else:
            _ensure_rm65_imports()
            self.safety_filter = ActionSafetyFilter(
                ActionSafetyFilterConfig(
                    active_sides=self.active_sides,
                    arms_enabled=True,
                    hands_enabled=self.hands_enabled,
                    workspace_min=self.args.arm_workspace_min,
                    workspace_max=self.args.arm_workspace_max,
                    max_arm_xyz_step_m=float(self.args.max_arm_xyz_step_m),
                    max_arm_rot_step_deg=float(self.args.max_arm_rot_step_deg),
                    max_arm_joint_step_rad=float(self.args.max_arm_joint_step_rad),
                    max_hand_step=float(self.args.max_hand_step),
                )
            )
        self.dispatcher = TimedActionDispatcher(
            args=self.args,
            active_sides=self.active_sides,
            arms_enabled=True,
            hands_enabled=self.hands_enabled,
            arm_backends=self.arm_backends,
            hand_backends=self.hand_backends,
            safety_filter=self.safety_filter,
            get_observation=lambda: self._build_observation(time.time()),
            get_last_action=lambda: self.last_action.copy(),
            set_last_action=self._set_last_action,
            policy_name="BeingH-Tac-live",
            send_actions=bool(self.args.send_actions),
            backend_kind=self.backend_kind,
        )

    def __enter__(self):
        try:
            for side, backend in self.arm_backends.items():
                backend.start()
                if not backend.is_ready() and not bool(self.args.allow_arm_connect_failure):
                    raise RuntimeError(f"{side} arm backend is not ready: {backend.get_state()}")
            for side, backend in self.hand_backends.items():
                backend.start()
                if not backend.is_ready() and not bool(self.args.allow_hand_connect_failure):
                    raise RuntimeError(f"{side} hand backend is not ready: {backend.get_state()}")
            for source in self.camera_sources.values():
                source.start()
            for camera_key, source in sorted(self.camera_sources.items()):
                info = {}
                device_info = getattr(source, "device_info", None)
                if callable(device_info):
                    try:
                        info = dict(device_info())
                    except Exception:
                        info = {}
                requested = str(info.get("requested_serial") or getattr(getattr(source, "config", None), "serial", "") or "")
                actual = str(info.get("serial_number") or info.get("serial") or "")
                serial_match = info.get("serial_match")
                print(
                    f"[Camera] {camera_key}: requested_serial={requested} actual_serial={actual} "
                    f"match={serial_match}"
                )
                if requested and actual and actual != requested:
                    raise RuntimeError(
                        f"Camera {camera_key!r} opened wrong RealSense device: "
                        f"requested serial {requested}, actual serial {actual}."
                    )
            if self.camera_previewer is not None:
                if self.camera_previewer.ensure_available():
                    print(f"[Camera] live preview active for cameras={sorted(self.preview_camera_sources)}")
                else:
                    print(f"WARNING: live camera preview disabled: {self.camera_previewer.disabled_reason}")
            self._wait_until_ready()
            self.dispatcher.start()
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        self.dispatcher.stop()
        if self.camera_previewer is not None:
            self.camera_previewer.close()
        for backend in self.arm_backends.values():
            backend.close()
        for backend in self.hand_backends.values():
            backend.close()
        for source in self.camera_sources.values():
            source.close()

    def _front_image(self, images: Mapping[str, np.ndarray]) -> np.ndarray:
        if not images:
            raise RuntimeError("Live camera frame is not ready; refusing to send zero image to server.")
        key = "front" if "front" in images else sorted(images)[0]
        image = np.asarray(images[key], dtype=np.uint8)
        if image.shape[:2] != (self.target_image_height, self.target_image_width):
            try:
                import cv2

                image = cv2.resize(
                    image,
                    (self.target_image_width, self.target_image_height),
                    interpolation=cv2.INTER_AREA,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to resize live camera frame from "
                    f"{image.shape[:2]} to {(self.target_image_height, self.target_image_width)}"
                ) from exc
        return image.astype(np.uint8, copy=False)

    def _build_observation(self, now: float) -> Any:
        arm_states = self.arm_reader.snapshot(now)
        hand_states = hand_state_snapshot_for_backends(self.hand_backends)
        return build_profile_observation(
            self.profile,
            active_sides=self.active_sides,
            arms_enabled=True,
            hands_enabled=self.hands_enabled,
            step_index=self.step_index,
            arm_states=arm_states,
            hand_states=hand_states,
            camera_sources=self.camera_sources,
            last_action=self.last_action,
            timestamp=now,
            require_images=False,
            metadata={"source": "beingh_tac_live_client"},
        )

    def _is_camera_ready(self, observation: Any) -> bool:
        if not self.camera_sources:
            return True
        return bool(observation.images)

    def _is_required_state_ready(self, observation: Any) -> bool:
        if not self._is_camera_ready(observation):
            return False
        if self.profile.data_config_name == "franka":
            return (
                np.all(np.isfinite(observation.state[:7]))
                and np.all(np.isfinite(observation.state[7:13]))
            )
        if self.profile.data_config_name == "realman20":
            return (
                np.all(np.isfinite(observation.state[STATE_ARM_POSE_SLICES["right"]]))
                and np.all(np.isfinite(observation.state[STATE_HAND_SLICES["right"]]))
            )
        if self.profile.data_config_name == "realman21":
            return (
                np.all(np.isfinite(observation.state[STATE_ARM_POSE_SLICES["right"]]))
                and np.all(np.isfinite(observation.state[STATE_HAND_SLICES["right"]]))
                and np.all(np.isfinite(observation.state[STATE_HAND_SLICES["left"]]))
            )
        if self.profile.data_config_name == "realman22":
            return (
                np.all(np.isfinite(observation.state[STATE_ARM_POSE_SLICES["left"]]))
                and np.all(np.isfinite(observation.state[STATE_ARM_POSE_SLICES["right"]]))
                and np.all(np.isfinite(observation.state[STATE_HAND_SLICES["left"]]))
                and np.all(np.isfinite(observation.state[STATE_HAND_SLICES["right"]]))
            )
        return False

    def _wait_until_ready(self) -> None:
        timeout_s = max(0.0, float(self.args.live_ready_timeout_s))
        deadline = time.time() + timeout_s
        last_faults: list[str] = []
        last_observation: Any | None = None
        while True:
            observation = self._build_observation(time.time())
            last_observation = observation
            last_faults = list(observation.faults)
            if self._is_required_state_ready(observation):
                return
            if time.time() >= deadline:
                break
            time.sleep(max(0.001, float(self.args.live_ready_poll_s)))
        if bool(self.args.allow_arm_connect_failure) or bool(self.args.allow_hand_connect_failure):
            print(
                "WARNING: live observation did not become fully ready before timeout; "
                f"continuing because connect-failure fallback is allowed. faults={last_faults[:6]}"
            )
            return
        detail = ""
        if last_observation is not None:
            arm_summary = {
                side: {
                    "hardware_result_code": state.get("hardware_result_code"),
                    "hardware_result_text": state.get("hardware_result_text"),
                    "has_current_qpos": state.get("current_qpos") is not None,
                    "has_current_tcp_pose": state.get("current_tcp_pose") is not None,
                    "pose_error": state.get("pose_error"),
                    "state_error": state.get("state_error"),
                }
                for side, state in last_observation.arm_states.items()
            }
            hand_summary = {
                side: {
                    "is_connected": state.get("is_connected"),
                    "has_joint_pos": state.get("joint_pos") is not None,
                    "has_tactile": state.get("tactile") is not None,
                    "telemetry_error": state.get("telemetry_error"),
                    "tactile_regions": state.get("tactile_regions"),
                }
                for side, state in last_observation.hand_states.items()
            }
            camera_summary = {
                key: {
                    "has_image": key in last_observation.images,
                    "timestamp": last_observation.image_timestamps.get(key),
                    "source_serial": (
                        (lambda info: info.get("serial_number") or info.get("serial"))(
                            dict(self.camera_sources[key].device_info()) if hasattr(self.camera_sources[key], "device_info") else {}
                        )
                        if key in self.camera_sources
                        else None
                    ),
                }
                for key in self.camera_sources
            }
            detail = f" arm_states={arm_summary} hand_states={hand_summary} camera_states={camera_summary}."
        raise RuntimeError(
            "Live observation is not ready. "
            f"faults={last_faults[:6]}. "
            "Check arm/hand IPs, hardware power/network, and hand telemetry."
            f"{detail}"
        )

    def _wait_for_next_ready_observation(self) -> Any:
        timeout_s = max(0.0, float(self.args.live_ready_timeout_s))
        deadline = time.time() + timeout_s
        last_observation: Any | None = None
        while True:
            observation = self._build_observation(time.time())
            last_observation = observation
            if self._is_required_state_ready(observation):
                return observation
            if time.time() >= deadline:
                break
            time.sleep(max(0.001, float(self.args.live_ready_poll_s)))
        raise RuntimeError(
            "Live observation lost readiness before request build. "
            f"faults={list(last_observation.faults[:6]) if last_observation is not None else []}"
        )

    def _state_for_profile(self, observation: Any) -> dict[str, np.ndarray]:
        state = observation.state
        if self.profile.data_config_name == "franka":
            arm_state = observation.arm_state
            raw_state = arm_state.get("raw_state") if isinstance(arm_state, Mapping) else None
            if isinstance(raw_state, Mapping) and raw_state.get("ee_pos") is not None and raw_state.get("ee_quat") is not None:
                eef_pose = np.concatenate(
                    [
                        _finite_vector(raw_state.get("ee_pos"), 3, "eef_pos"),
                        _finite_vector(raw_state.get("ee_quat"), 4, "eef_quat"),
                    ],
                    axis=0,
                ).astype(np.float32)
            else:
                eef_pose = np.zeros(7, dtype=np.float32)
            return {
                "eef_pose": eef_pose,
                "arm_qpos": _finite_vector(state[:7], 7, "arm_qpos"),
                "right_hand_qpos": _finite_vector(state[7:13], 6, "right_hand_qpos"),
            }
        if self.profile.data_config_name == "realman20":
            return {
                "right_arm_pose": _finite_vector(state[STATE_ARM_POSE_SLICES["right"]], 7, "right_arm_pose"),
                "right_hand_qpos": _finite_vector(state[STATE_HAND_SLICES["right"]], 6, "right_hand_qpos"),
            }
        if self.profile.data_config_name == "realman21":
            return {
                "right_arm_pose": _finite_vector(state[STATE_ARM_POSE_SLICES["right"]], 7, "right_arm_pose"),
                "right_hand_qpos": _finite_vector(state[STATE_HAND_SLICES["right"]], 6, "right_hand_qpos"),
                "left_hand_qpos": _finite_vector(state[STATE_HAND_SLICES["left"]], 6, "left_hand_qpos"),
            }
        if self.profile.data_config_name == "realman22":
            return {
                "left_arm_pose": _finite_vector(state[STATE_ARM_POSE_SLICES["left"]], 7, "left_arm_pose"),
                "right_arm_pose": _finite_vector(state[STATE_ARM_POSE_SLICES["right"]], 7, "right_arm_pose"),
                "left_hand_qpos": _finite_vector(state[STATE_HAND_SLICES["left"]], 6, "left_hand_qpos"),
                "right_hand_qpos": _finite_vector(state[STATE_HAND_SLICES["right"]], 6, "right_hand_qpos"),
            }
        raise ValueError(f"Unsupported live profile {self.profile.data_config_name}.")

    def _set_last_action(self, action: np.ndarray) -> None:
        width = 19 if self.backend_kind == "fr3" else 26
        self.last_action = np.asarray(action, dtype=np.float32).reshape(width)

    def next_payload_base(self) -> LiveSnapshot:
        observation = self._wait_for_next_ready_observation()
        if self.camera_previewer is not None:
            self.camera_previewer.update(self.preview_camera_sources)
        for side in ("left", "right"):
            tactile = observation.tactile_by_side.get(side)
            if tactile is not None and np.asarray(tactile).size:
                self.history.append(side, tactile)
        payload = {
            "image": self._front_image(observation.images),
            "state": self._state_for_profile(observation),
            "tactile": {
                "right": self.history.stack("right"),
                "left": self.history.stack("left"),
            },
        }
        self.step_index += 1
        return LiveSnapshot(payload_base=payload, observation=observation)

    def _server_result_for_deploy(self, result: Mapping[str, Any]) -> dict[str, Any]:
        return server_result_for_deploy(result)

    def consume_server_result(self, result: Mapping[str, Any]) -> list[str]:
        if not bool(self.args.auto_start):
            return ["policy_paused"]
        return self.dispatcher.submit_result(result)

    def send_direct_joint_targets(
        self,
        *,
        arm_qpos: Mapping[str, Any] | None = None,
        hand_qpos: Mapping[str, Any] | None = None,
    ) -> list[str]:
        return _send_direct_joint_targets(
            arm_backends=self.arm_backends,
            hand_backends=self.hand_backends,
            backend_kind=self.backend_kind,
            arm_qpos=arm_qpos,
            hand_qpos=hand_qpos,
        )

    def remember_observation(self, observation: Any) -> None:
        self._last_observation = observation

    @property
    def sent_actions(self) -> int:
        return self.dispatcher.sent_actions

    def wait_until_sent(
        self,
        target_sent: int,
        timeout_s: float | None = None,
        poll_s: float = 0.01,
        progress: Callable[[int, int], None] | None = None,
    ) -> bool:
        return self.dispatcher.wait_until_sent(
            target_sent=target_sent,
            timeout_s=timeout_s,
            poll_s=poll_s,
            progress=progress,
        )


class HardwareActionExecutor:
    """Send server actions to live hardware while observations come from a dataset."""

    def __init__(self, profile: DeploymentProfile, args: argparse.Namespace):
        self.profile = profile
        self.backend_kind = "fr3" if _is_fr3_profile(profile) else "rm65"
        self.args = _coerce_builder_args(args)
        required_sides = set(profile.required_arms) | set(profile.required_hands)
        self.active_sides = tuple(side for side in ALL_SIDES if side in required_sides)
        self.hands_enabled = bool(profile.required_hands)
        self.args.active_sides = self.active_sides
        self.args.hands_enabled = self.hands_enabled

        missing_arms, missing_hands = _required_missing_endpoints(profile, self.args)
        if missing_arms and not bool(self.args.allow_arm_connect_failure):
            flags = ", ".join(f"--{side}-arm-ip" for side in missing_arms)
            raise ValueError(
                f"--send-actions requires hardware arm IP for: {', '.join(missing_arms)}. "
                f"Set {flags} or export {', '.join(f'{side.upper()}_ARM_IP' for side in missing_arms)}."
            )
        if missing_hands and not bool(self.args.allow_hand_connect_failure):
            flags = ", ".join(f"--{side}-hand-ip" for side in missing_hands)
            raise ValueError(
                f"--send-actions requires hardware hand IP for: {', '.join(missing_hands)}. "
                f"Set {flags} or export {', '.join(f'{side.upper()}_HAND_IP' for side in missing_hands)}."
            )

        self.arm_backends = build_profile_arm_backends(
            profile,
            self.args,
            self.active_sides,
            control_hz=float(self.args.action_hz),
        )
        self.hand_backends = build_profile_hand_backends(profile, self.args, self.active_sides)
        self.arm_reader = GenericArmStateReader(
            self.arm_backends,
            refresh_hz=float(self.args.arm_state_refresh_hz),
            compute_tcp_pose=True,
        )
        self.last_action = np.full(19 if self.backend_kind == "fr3" else 26, np.nan, dtype=np.float32)
        self.step_index = 0
        self.start_dispatcher = True
        if self.backend_kind == "fr3":
            self.safety_filter = FR3ActionSafetyFilter(
                FR3ActionSafetyFilterConfig(
                    active_sides=self.active_sides,
                    arm_enabled=True,
                    max_arm_step_rad=float(self.args.max_arm_joint_step_rad),
                    max_hand_step=float(self.args.max_hand_step),
                )
            )
        else:
            _ensure_rm65_imports()
            self.safety_filter = ActionSafetyFilter(
                ActionSafetyFilterConfig(
                    active_sides=self.active_sides,
                    arms_enabled=True,
                    hands_enabled=self.hands_enabled,
                    workspace_min=self.args.arm_workspace_min,
                    workspace_max=self.args.arm_workspace_max,
                    max_arm_xyz_step_m=float(self.args.max_arm_xyz_step_m),
                    max_arm_rot_step_deg=float(self.args.max_arm_rot_step_deg),
                    max_arm_joint_step_rad=float(self.args.max_arm_joint_step_rad),
                    max_hand_step=float(self.args.max_hand_step),
                )
            )
        self.dispatcher = TimedActionDispatcher(
            args=self.args,
            active_sides=self.active_sides,
            arms_enabled=True,
            hands_enabled=self.hands_enabled,
            arm_backends=self.arm_backends,
            hand_backends=self.hand_backends,
            safety_filter=self.safety_filter,
            get_observation=self._build_observation,
            get_last_action=lambda: self.last_action.copy(),
            set_last_action=self._set_last_action,
            policy_name="BeingH-Tac-dataset",
            send_actions=True,
            backend_kind=self.backend_kind,
        )

    def __enter__(self) -> "HardwareActionExecutor":
        for side, backend in self.arm_backends.items():
            backend.start()
            if not backend.is_ready() and not bool(self.args.allow_arm_connect_failure):
                raise RuntimeError(f"{side} arm backend is not ready: {backend.get_state()}")
        for side, backend in self.hand_backends.items():
            backend.start()
            if not backend.is_ready() and not bool(self.args.allow_hand_connect_failure):
                raise RuntimeError(f"{side} hand backend is not ready: {backend.get_state()}")
        self._wait_until_ready()
        if self.start_dispatcher:
            self.dispatcher.start()
            print(f"[Action] dataset --send-actions enabled for sides={self.active_sides}")
        else:
            print(f"[Action] direct joint target mode enabled for sides={self.active_sides}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.dispatcher.stop()
        for backend in self.arm_backends.values():
            backend.close()
        for backend in self.hand_backends.values():
            backend.close()

    def _build_observation(self) -> Any:
        now = time.time()
        arm_states = self.arm_reader.snapshot(now)
        hand_states = hand_state_snapshot_for_backends(self.hand_backends)
        observation = build_profile_observation(
            self.profile,
            active_sides=self.active_sides,
            arms_enabled=True,
            hands_enabled=self.hands_enabled,
            step_index=self.step_index,
            arm_states=arm_states,
            hand_states=hand_states,
            camera_sources={},
            last_action=self.last_action,
            timestamp=now,
            require_images=False,
            metadata={"source": "beingh_tac_dataset_action_executor"},
        )
        self.step_index += 1
        return observation

    def _is_ready(self, observation: Any) -> bool:
        if self.backend_kind == "fr3":
            return bool(np.all(np.isfinite(observation.state[:13])))
        ready = True
        for side in self.profile.required_arms:
            ready = ready and np.all(np.isfinite(observation.state[STATE_ARM_POSE_SLICES[side]]))
        for side in self.profile.required_hands:
            ready = ready and np.all(np.isfinite(observation.state[STATE_HAND_SLICES[side]]))
        return bool(ready)

    def _wait_until_ready(self) -> None:
        timeout_s = max(0.0, float(self.args.live_ready_timeout_s))
        deadline = time.time() + timeout_s
        last_observation: Any | None = None
        while True:
            observation = self._build_observation()
            last_observation = observation
            if self._is_ready(observation):
                return
            if time.time() >= deadline:
                break
            time.sleep(max(0.001, float(self.args.live_ready_poll_s)))
        if bool(self.args.allow_arm_connect_failure) or bool(self.args.allow_hand_connect_failure):
            print(
                "WARNING: action executor hardware state did not become fully ready before timeout; "
                "continuing because connect-failure fallback is allowed."
            )
            return
        faults = [] if last_observation is None else list(last_observation.faults[:6])
        raise RuntimeError(f"Action executor hardware state is not ready. faults={faults}")

    def _set_last_action(self, action: np.ndarray) -> None:
        width = 19 if self.backend_kind == "fr3" else 26
        self.last_action = np.asarray(action, dtype=np.float32).reshape(width)

    def consume_server_result(self, result: Mapping[str, Any]) -> list[str]:
        return self.dispatcher.submit_result(result)

    def send_direct_joint_targets(
        self,
        *,
        arm_qpos: Mapping[str, Any] | None = None,
        hand_qpos: Mapping[str, Any] | None = None,
    ) -> list[str]:
        return _send_direct_joint_targets(
            arm_backends=self.arm_backends,
            hand_backends=self.hand_backends,
            backend_kind=self.backend_kind,
            arm_qpos=arm_qpos,
            hand_qpos=hand_qpos,
        )

    def wait_until_idle(self, timeout_s: float | None = None, poll_s: float = 0.01) -> bool:
        return self.dispatcher.wait_until_idle(timeout_s=timeout_s, poll_s=poll_s)

    @property
    def sent_actions(self) -> int:
        return self.dispatcher.sent_actions

    def wait_until_sent(
        self,
        target_sent: int,
        timeout_s: float | None = None,
        poll_s: float = 0.01,
        progress: Callable[[int, int], None] | None = None,
    ) -> bool:
        return self.dispatcher.wait_until_sent(
            target_sent=target_sent,
            timeout_s=timeout_s,
            poll_s=poll_s,
            progress=progress,
        )
