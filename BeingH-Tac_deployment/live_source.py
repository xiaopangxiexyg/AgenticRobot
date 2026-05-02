from __future__ import annotations

import argparse
from collections import deque
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from paths import ensure_import_paths
from profiles import DeploymentProfile
from tactile import TactileHistory


ensure_import_paths()

from teleop_runtime.deploy.actions import (  # noqa: E402
    ActionChunkBuffer,
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    normalize_policy_output,
)
from teleop_runtime.deploy.observation import build_deployment_observation  # noqa: E402
from teleop_runtime.deploy.run_policy import (  # noqa: E402
    ArmStateReader,
    build_arm_backends,
    build_camera_sources,
    build_hand_backends,
    hand_state_snapshot,
)
from teleop_runtime.deploy.schema import (  # noqa: E402
    ACTION_HAND_SLICES,
    ACTION_ARM_POSE_SLICES,
    ALL_SIDES,
    STATE_ARM_POSE_SLICES,
    STATE_HAND_SLICES,
)


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
    parser.add_argument("--inspire-hand-max-force", "--max-force", dest="max_fingertip_force", type=int, default=800)
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
    parser.add_argument("--camera-auto-exposure-priority", type=int, default=None)
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


def _send_commands(
    commands: Mapping[str, Any],
    *,
    arm_backends: Mapping[str, Any],
    hand_backends: Mapping[str, Any],
) -> None:
    for side in ALL_SIDES:
        arm_command = commands.get(f"{side}_arm")
        if arm_command is not None and side in arm_backends:
            arm_backends[side].send_command(arm_command)
        hand_command = commands.get(f"{side}_hand")
        if hand_command is not None and side in hand_backends:
            hand_backends[side].send_command(hand_command)


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
        output: Mapping[str, Any] | dict[str, np.ndarray] = server_result_for_deploy(result)
        if not output:
            if "action_raw" not in result:
                return ["missing_action"]
            output = {"action": np.asarray(result["action_raw"], dtype=np.float32)}

        observation = self.get_observation()
        actions = normalize_policy_output(
            output,
            observation,
            active_sides=self.active_sides,
            arms_enabled=self.arms_enabled,
            hands_enabled=self.hands_enabled,
        )
        if not actions:
            return ["empty_action_chunk"]

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
        if not profile.supports_live_rm65:
            raise ValueError(f"{profile.data_config_name} live mode is not supported by DexTeleop RM65 backends.")
        self.profile = profile
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
        self.arm_backends = build_arm_backends(self.args, self.active_sides)
        self.hand_backends = build_hand_backends(self.args, self.active_sides)
        self.arm_reader = ArmStateReader(
            self.arm_backends,
            refresh_hz=float(self.args.arm_state_refresh_hz),
            compute_tcp_pose=True,
        )
        self.camera_sources = build_camera_sources(self.args)
        self.history = TactileHistory(history_size, history_stride)
        self.last_action = np.full(26, np.nan, dtype=np.float32)
        self.step_index = 0
        self.target_image_height = int(getattr(args, "fallback_image_height", 480))
        self.target_image_width = int(getattr(args, "fallback_image_width", 640))
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
        )

    def __enter__(self):
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
        self._wait_until_ready()
        self.dispatcher.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.dispatcher.stop()
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
        hand_states = hand_state_snapshot(self.hand_backends)
        return build_deployment_observation(
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
        self.last_action = np.asarray(action, dtype=np.float32).reshape(26)

    def next_payload_base(self) -> LiveSnapshot:
        observation = self._wait_for_next_ready_observation()
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

    def remember_observation(self, observation: Any) -> None:
        self._last_observation = observation


class HardwareActionExecutor:
    """Send server actions to live hardware while observations come from a dataset."""

    def __init__(self, profile: DeploymentProfile, args: argparse.Namespace):
        if not profile.supports_live_rm65:
            raise ValueError(f"{profile.data_config_name} action execution is not supported by DexTeleop RM65 backends.")
        self.profile = profile
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

        self.arm_backends = build_arm_backends(self.args, self.active_sides)
        self.hand_backends = build_hand_backends(self.args, self.active_sides)
        self.arm_reader = ArmStateReader(
            self.arm_backends,
            refresh_hz=float(self.args.arm_state_refresh_hz),
            compute_tcp_pose=True,
        )
        self.last_action = np.full(26, np.nan, dtype=np.float32)
        self.step_index = 0
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
        self.dispatcher.start()
        print(f"[Action] dataset --send-actions enabled for sides={self.active_sides}")
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
        hand_states = hand_state_snapshot(self.hand_backends)
        observation = build_deployment_observation(
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
        self.last_action = np.asarray(action, dtype=np.float32).reshape(26)

    def consume_server_result(self, result: Mapping[str, Any]) -> list[str]:
        return self.dispatcher.submit_result(result)
