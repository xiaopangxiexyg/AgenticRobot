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

from rm65_inspire_tac.paths import require_asset_path

from ..backends.arms import RM65ArmBackend, RM65ArmBackendConfig
from ..backends.hands import (
    DEFAULT_INSPIRE_HAND_MAX_FORCE,
    InspireTacHandBackend,
    InspireTacHandBackendConfig,
)
from ..collection.camera_preview import CameraPreviewConfig, CameraPreviewer
from ..hand_mapping import parse_tactile_regions
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
from .observation import build_deployment_observation
from .policy import load_policy
from .replay import load_dataset_replay_policy
from .schema import ACTION_DIM, ALL_SIDES


DEFAULT_CAMERA_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "devices" / "realsense.json"
)


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
                    state["current_tcp_pose"] = backend.compute_tcp_pose()
                except Exception as exc:  # noqa: BLE001
                    state.setdefault("pose_error", f"compute_tcp_pose_failed:{type(exc).__name__}:{exc}")
            states[side] = state
        return states


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def _parse_active_sides(text: str) -> tuple[str, ...]:
    value = str(text).strip().lower()
    if value == "both":
        return ("left", "right")
    if value in set(ALL_SIDES):
        return (value,)
    raise argparse.ArgumentTypeError("active-sides must be one of: left, right, both")


def _parse_float_tuple(text: str, *, length: int, name: str) -> tuple[float, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != int(length):
        raise argparse.ArgumentTypeError(f"{name} must contain {length} comma-separated floats")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid {name}: {text}") from exc
    return values


def _parse_vector3(text: str) -> tuple[float, float, float]:
    values = _parse_float_tuple(text, length=3, name="vector3")
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


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--policy",
        "--policy-spec",
        dest="policy_spec",
        type=str,
        default="hold",
        help="Policy loader spec: hold, package.module:Policy, or /path/policy.py:Policy.",
    )
    parser.add_argument("--policy-kwargs-json", type=str, default="{}")
    parser.add_argument("--active-sides", type=_parse_active_sides, default=("left", "right"))
    parser.add_argument("--hz", type=_positive_float, default=30.0)
    parser.add_argument("--policy-hz", type=_positive_float, default=10.0)
    parser.add_argument("--min-action-queue", type=_non_negative_int, default=0)
    parser.add_argument("--append-action-chunks", action="store_true")
    parser.add_argument("--max-steps", type=_non_negative_int, default=0)
    parser.add_argument("--status-sec", type=_positive_float, default=1.0)
    parser.add_argument("--startup-warmup-sec", type=float, default=1.0)
    parser.add_argument("--debug-replay-dataset", type=Path, default=None)
    parser.add_argument("--debug-replay-parquet", type=Path, default=None)
    parser.add_argument("--debug-replay-episode", type=_non_negative_int, default=0)
    parser.add_argument("--debug-replay-action-column", type=str, default="action")
    parser.add_argument("--debug-replay-start-index", type=_non_negative_int, default=0)
    parser.add_argument("--debug-replay-end-index", type=int, default=None)
    parser.add_argument("--debug-replay-chunk-size", type=_non_negative_int, default=32)
    parser.add_argument("--debug-replay-loop", action="store_true")
    parser.add_argument("--require-ready-observation", action="store_true")
    parser.add_argument("--pause-on-policy-error", dest="pause_on_policy_error", action="store_true", default=True)
    parser.add_argument("--no-pause-on-policy-error", dest="pause_on_policy_error", action="store_false")
    parser.add_argument("--send-actions", action="store_true")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--allow-arm-connect-failure", action="store_true")
    parser.add_argument("--allow-hand-connect-failure", action="store_true")

    parser.add_argument("--enable-arms", dest="arms_enabled", action="store_true", default=True)
    parser.add_argument("--no-arms", dest="arms_enabled", action="store_false")
    parser.add_argument("--left-arm-ip", type=str, default="")
    parser.add_argument("--right-arm-ip", type=str, default="")
    parser.add_argument("--arm-urdf-path", type=str, default="")
    parser.add_argument("--arm-max-angvel", type=_parse_arm_max_angvel, default=(0.35, 0.35, 0.35, 0.5, 0.5, 0.5))
    parser.add_argument("--arm-state-refresh-hz", type=float, default=10.0)
    parser.add_argument("--rm65-hardware-read-retries", type=_non_negative_int, default=1)
    parser.add_argument("--rm65-hardware-read-retry-delay-sec", type=float, default=0.02)
    parser.add_argument("--left-initial-qpos", type=_parse_qpos6, default=None)
    parser.add_argument("--right-initial-qpos", type=_parse_qpos6, default=None)
    parser.add_argument("--arm-workspace-min", type=_parse_vector3, default=(-0.60, -0.80, 0.00))
    parser.add_argument("--arm-workspace-max", type=_parse_vector3, default=(0.60, 0.80, 0.90))
    parser.add_argument("--max-arm-xyz-step-m", type=float, default=0.03)
    parser.add_argument("--max-arm-rot-step-deg", type=float, default=8.0)
    parser.add_argument("--max-arm-joint-step-rad", type=float, default=0.08)

    parser.add_argument("--enable-hands", dest="hands_enabled", action="store_true", default=None)
    parser.add_argument("--no-hands", dest="hands_enabled", action="store_false")
    parser.add_argument("--left-hand-ip", type=str, default="")
    parser.add_argument("--right-hand-ip", type=str, default="")
    parser.add_argument("--hand-modbus-port", type=int, default=6000)
    parser.add_argument("--hand-send-threshold", type=float, default=0.002)
    parser.add_argument("--hand-telemetry-hz", type=_positive_float, default=20.0)
    parser.add_argument("--hand-tactile-regions", "--tactile-regions", "--tactile-region", dest="hand_tactile_regions", type=str, default="all")
    parser.add_argument(
        "--inspire-hand-max-force",
        "--max-force",
        "--max-fingertip-force",
        dest="max_fingertip_force",
        type=_non_negative_int,
        default=DEFAULT_INSPIRE_HAND_MAX_FORCE,
    )
    parser.add_argument("--max-hand-step", type=float, default=0.08)

    parser.add_argument("--camera-mode", choices=["none", "single", "multi"], default="none")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG_PATH)
    parser.add_argument("--camera-key", type=str, default="front")
    parser.add_argument("--camera-serial", type=str, default="")
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
    return bool(
        args.left_hand_ip
        or args.right_hand_ip
        or args.visualize_tactile
    )


def validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    active_sides = tuple(args.active_sides)
    hands_enabled = _hands_enabled(args)
    if not bool(args.arms_enabled) and not hands_enabled:
        raise ValueError("Enable at least one actuator family: arms or hands.")
    if int(args.debug_replay_chunk_size) <= 0:
        raise ValueError("--debug-replay-chunk-size must be positive.")
    if args.debug_replay_end_index is not None and int(args.debug_replay_end_index) <= int(args.debug_replay_start_index):
        raise ValueError("--debug-replay-end-index must be greater than --debug-replay-start-index.")
    if args.debug_replay_dataset and args.debug_replay_parquet:
        raise ValueError("Use only one of --debug-replay-dataset or --debug-replay-parquet.")
    if (args.debug_replay_dataset or args.debug_replay_parquet) and str(args.policy_spec or "hold").strip().lower() not in {"", "hold", "builtin:hold"}:
        raise ValueError("--debug-replay-* replaces --policy; omit --policy in replay mode.")
    parse_tactile_regions(args.hand_tactile_regions)
    if bool(args.send_actions) and bool(args.arms_enabled):
        for side in active_sides:
            if not getattr(args, f"{side}_arm_ip"):
                raise ValueError(f"--send-actions requires --{side}-arm-ip for active side '{side}'.")
    if bool(args.send_actions) and hands_enabled:
        for side in active_sides:
            if not getattr(args, f"{side}_hand_ip"):
                raise ValueError(f"--send-actions with hands requires --{side}-hand-ip.")
    if bool(args.visualize_tactile):
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


def build_arm_backends(args: argparse.Namespace, active_sides: tuple[str, ...]) -> dict[str, RM65ArmBackend]:
    if not bool(args.arms_enabled):
        return {}
    arm_urdf_path = str(args.arm_urdf_path or require_asset_path("rm65_arm.urdf"))
    control_dt = 1.0 / float(args.hz)
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
                dry_run_fast=not bool(arm_ip),
                hardware_read_retries=int(args.rm65_hardware_read_retries),
                hardware_read_retry_delay_sec=float(args.rm65_hardware_read_retry_delay_sec),
                initial_qpos=getattr(args, f"{side}_initial_qpos"),
                background_streaming_enabled=bool(arm_ip),
                background_streaming_hz=float(args.hz),
            )
        )
    return backends


def build_hand_backends(args: argparse.Namespace, active_sides: tuple[str, ...]) -> dict[str, InspireTacHandBackend]:
    if not _hands_enabled(args):
        return {}
    tactile_regions = parse_tactile_regions(args.hand_tactile_regions)
    backends: dict[str, InspireTacHandBackend] = {}
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


def hand_state_snapshot(backends: Mapping[str, InspireTacHandBackend]) -> dict[str, dict[str, Any]]:
    return {side: backend.get_state() for side, backend in backends.items()}


def print_controls(*, send_actions: bool, auto_start: bool) -> None:
    print("\n" + "=" * 60)
    print("teleop_runtime policy deployment loop started")
    print(f"  send_actions={'on' if send_actions else 'off(dry-run)'}")
    print(f"  initial_policy_state={'active' if auto_start else 'paused'}")
    print("Controls: g=start/resume, h=pause+clear queue, q=quit")
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


def _send_commands(
    commands: Mapping[str, Any],
    *,
    arm_backends: Mapping[str, RM65ArmBackend],
    hand_backends: Mapping[str, InspireTacHandBackend],
) -> None:
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
    if args.debug_replay_dataset or args.debug_replay_parquet:
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

    arm_backends = build_arm_backends(args, active_sides)
    hand_backends = build_hand_backends(args, active_sides)
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
    chunk_buffer = ActionChunkBuffer()

    policy_active = bool(args.auto_start)
    send_actions = bool(args.send_actions)
    last_action = np.full(ACTION_DIM, np.nan, dtype=np.float32)
    last_policy_at: float | None = None
    step_index = 0
    loop_rate = LoopRate()
    last_status_at = 0.0
    last_fault_text: str | None = None
    started_at = time.time()

    try:
        print(
            "[Deploy] policy="
            f"{policy.name} active_sides={active_sides} "
            f"arms={bool(args.arms_enabled)} hands={hands_enabled} "
            f"hz={float(args.hz):.2f} policy_hz={float(args.policy_hz):.2f}"
        )
        if policy.metadata.get("debug_replay"):
            print(
                "[Replay] source="
                f"{policy.metadata.get('source_path')} frames={policy.metadata.get('frame_count')} "
                f"chunk_size={int(args.debug_replay_chunk_size)} loop={bool(args.debug_replay_loop)}"
            )
        if not send_actions:
            print("[Deploy] dry-run: actions are filtered and logged but not sent to hardware.")

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
        print_controls(send_actions=send_actions, auto_start=bool(args.auto_start))

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
                    last_policy_at = None
                    policy_active = True
                    print("[Deploy] policy active")
                elif key == "h":
                    chunk_buffer.clear()
                    policy_active = False
                    print("[Deploy] policy paused")
                elif key == "q":
                    print("\n[Deploy] quitting policy deployment loop\n")
                    break

                arm_states = arm_reader.snapshot(loop_started_at)
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
                    },
                )
                faults = list(observation.faults)
                if loop_started_at - started_at < float(args.startup_warmup_sec):
                    faults.append("startup_warmup")

                if visualizer is not None:
                    visualizer.update(tactile_by_side_from_backends(active_sides, hand_backends.get))

                if policy_active:
                    can_run_policy = not (bool(args.require_ready_observation) and faults)
                    if can_run_policy and _policy_should_run(
                        now=loop_started_at,
                        last_policy_at=last_policy_at,
                        policy_interval_s=1.0 / float(args.policy_hz),
                        chunk_buffer=chunk_buffer,
                        min_action_queue=int(args.min_action_queue),
                    ):
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
                                if args.append_action_chunks:
                                    chunk_buffer.append(actions)
                                else:
                                    chunk_buffer.replace(actions)
                                last_policy_at = loop_started_at
                            else:
                                faults.append("policy_empty_action")
                        except Exception as exc:  # noqa: BLE001
                            print(f"[Deploy] policy error: {type(exc).__name__}: {exc}")
                            chunk_buffer.clear()
                            faults.append(f"policy_error:{type(exc).__name__}")
                            if bool(args.pause_on_policy_error):
                                policy_active = False

                    raw_action = chunk_buffer.pop(default=_default_action(observation)) if can_run_policy else None
                    if raw_action is not None:
                        deployment_action = safety_filter.filter(
                            raw_action,
                            observation,
                            policy_name=policy.name,
                        )
                        faults.extend(deployment_action.faults)
                        if deployment_action.commands:
                            if send_actions:
                                _send_commands(
                                    deployment_action.commands,
                                    arm_backends=arm_backends,
                                    hand_backends=hand_backends,
                                )
                            last_action = deployment_action.vector.astype(np.float32, copy=True)

                fault_text = _compact_faults(faults)
                if fault_text != last_fault_text:
                    print(f"[State] faults={fault_text}")
                    last_fault_text = fault_text

                if loop_started_at - last_status_at >= float(args.status_sec):
                    policy_status = _policy_status_text(policy)
                    suffix = f" {policy_status}" if policy_status else ""
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
                sleep_time = (1.0 / float(args.hz)) - (time.time() - loop_started_at)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[Deploy] keyboard interrupt; stopping\n")
    finally:
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
