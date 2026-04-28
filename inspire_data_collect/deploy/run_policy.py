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

from ..common import ALL_SIDES, parse_active_sides
from ..hand.hand_backend import (
    DEFAULT_INSPIRE_HAND_MAX_FORCE,
    InspireTacHandBackend,
    InspireTacHandBackendConfig,
)
from ..hand.hand_mapping import parse_tactile_regions
from ..utils.camera_preview import CameraPreviewConfig, CameraPreviewer
from ..utils.realsense_rgb import (
    RealSenseRGBCameraConfig,
    RealSenseRGBSource,
    load_realsense_camera_config_map,
)
from ..utils.tactile_visualizer import (
    TactileUVVisualizer,
    TactileVisualizerConfig,
    tactile_by_side_from_backends,
)
from ..utils.writer import ACTION_DIM, HAND_VECTOR_ORDER
from .actions import (
    ActionChunkBuffer,
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    normalize_policy_output,
)
from .observation import build_deployment_observation
from .policy import load_policy
from .replay import load_dataset_replay_policy


DEFAULT_CAMERA_CONFIG_PATH = Path(__file__).resolve().parents[1] / "realsense.json"


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


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--policy",
        "--policy-spec",
        dest="policy_spec",
        type=str,
        default="hold",
        help=(
            "Policy loader spec: hold, package.module:Policy, "
            "package.module:create_policy, or /path/policy.py:Policy."
        ),
    )
    parser.add_argument(
        "--policy-kwargs-json",
        type=str,
        default="{}",
        help="JSON object passed to Policy/create_policy constructors.",
    )
    parser.add_argument("--active-sides", type=parse_active_sides, default=ALL_SIDES)
    parser.add_argument("--hz", type=float, default=30.0, help="Control loop rate.")
    parser.add_argument("--policy-hz", type=float, default=10.0, help="Max synchronous policy query rate.")
    parser.add_argument(
        "--min-action-queue",
        type=int,
        default=0,
        help="Query policy when queued chunk length is <= this value.",
    )
    parser.add_argument(
        "--append-action-chunks",
        action="store_true",
        help="Append new chunks instead of replacing the pending queue.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="0 means run until stopped.")
    parser.add_argument("--status-sec", type=float, default=1.0)
    parser.add_argument("--startup-warmup-sec", type=float, default=1.0)
    parser.add_argument(
        "--debug-replay-dataset",
        type=Path,
        default=None,
        help="Debug mode: replay actions from a collected dataset root.",
    )
    parser.add_argument(
        "--debug-replay-parquet",
        type=Path,
        default=None,
        help="Debug mode: replay actions from a collected episode parquet directly.",
    )
    parser.add_argument(
        "--debug-replay-episode",
        type=int,
        default=0,
        help="Episode index to replay when --debug-replay-dataset is used.",
    )
    parser.add_argument(
        "--debug-replay-action-column",
        type=str,
        default="action",
        help="Parquet column containing the full 19-D action vector.",
    )
    parser.add_argument(
        "--debug-replay-start-index",
        type=int,
        default=0,
        help="First frame index from the episode parquet to replay.",
    )
    parser.add_argument(
        "--debug-replay-end-index",
        type=int,
        default=None,
        help="Exclusive end frame index from the episode parquet. Default replays to the end.",
    )
    parser.add_argument(
        "--debug-replay-chunk-size",
        type=int,
        default=32,
        help="Number of replay actions queued per policy query.",
    )
    parser.add_argument(
        "--debug-replay-loop",
        action="store_true",
        help="Loop the replay trajectory instead of holding the final action at the end.",
    )
    parser.add_argument(
        "--require-ready-observation",
        action="store_true",
        help="Skip policy and sending while active state/image observation faults exist.",
    )
    parser.add_argument(
        "--pause-on-policy-error",
        dest="pause_on_policy_error",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-pause-on-policy-error",
        dest="pause_on_policy_error",
        action="store_false",
    )

    parser.add_argument(
        "--send-actions",
        action="store_true",
        help="Actually send filtered policy actions to hardware. Default is dry-run.",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start policy execution immediately. Otherwise press 'g' after startup.",
    )
    parser.add_argument("--allow-hand-connect-failure", action="store_true")
    parser.add_argument("--allow-arm-connect-failure", action="store_true")

    parser.add_argument("--left-hand-ip", type=str, default="")
    parser.add_argument("--right-hand-ip", type=str, default="")
    parser.add_argument("--hand-modbus-port", type=int, default=6000)
    parser.add_argument("--hand-send-threshold", type=float, default=0.002)
    parser.add_argument("--hand-telemetry-hz", type=float, default=20.0)
    parser.add_argument(
        "--hand-tactile-regions",
        "--tactile-regions",
        "--tactile-region",
        dest="hand_tactile_regions",
        type=str,
        default="all",
    )
    parser.add_argument(
        "--inspire-hand-max-force",
        "--max-force",
        "--max-fingertip-force",
        dest="max_fingertip_force",
        type=int,
        default=DEFAULT_INSPIRE_HAND_MAX_FORCE,
        help="Max force value written to every InspireTac finger on startup.",
    )

    parser.add_argument("--enable-arm", action="store_true")
    parser.add_argument("--arm-side", choices=["right", "left"], default="right")
    parser.add_argument("--fr3-host", type=str, default="192.168.20.172")
    parser.add_argument("--fr3-port", type=int, default=5558)
    parser.add_argument("--fr3-timeout-s", type=float, default=5.0)
    parser.add_argument("--fr3-state-poll-hz", type=float, default=20.0)
    parser.add_argument("--fr3-max-state-age-s", type=float, default=0.5)
    parser.add_argument("--fr3-state-ready-timeout-s", type=float, default=2.0)
    parser.add_argument(
        "--disable-arm-startup-align",
        action="store_true",
        help="Disable initial FR3 ramp from current qpos to first policy target.",
    )
    parser.add_argument("--arm-startup-align-duration-s", type=float, default=5.0)
    parser.add_argument("--arm-startup-align-threshold-rad", type=float, default=0.02)

    parser.add_argument("--arm-shrink-deg", type=float, default=10.0)
    parser.add_argument(
        "--max-arm-step-rad",
        type=float,
        default=0.05,
        help="Per-control-tick arm action rate limit. <=0 disables.",
    )
    parser.add_argument(
        "--max-hand-step",
        type=float,
        default=0.08,
        help="Per-control-tick hand q6 action rate limit. <=0 disables.",
    )

    parser.add_argument(
        "--camera-mode",
        choices=["none", "single", "multi"],
        default="none",
    )
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
    parser.add_argument(
        "--require-camera-frame",
        action="store_true",
        help="Treat missing camera frames as observation faults.",
    )
    parser.add_argument("--vis_cam", "--vis-cam", dest="vis_cam", action="store_true")
    parser.add_argument("--vis-cam-key", "--vis-camera-key", type=str, default="")

    parser.add_argument("--visualize-tactile", action="store_true")
    parser.add_argument("--tactile-vis-value-scale", type=float, default=0.02)
    parser.add_argument("--tactile-vis-size", type=int, default=640)
    parser.add_argument(
        "--tactile-vis-window-mode",
        choices=["resizable", "autosize"],
        default="resizable",
    )


def parse_policy_kwargs(text: str) -> dict[str, Any]:
    payload = json.loads(str(text or "{}"))
    if not isinstance(payload, dict):
        raise ValueError("--policy-kwargs-json must decode to a JSON object.")
    return payload


def validate_args(args: argparse.Namespace) -> tuple[str, ...]:
    active_sides = tuple(args.active_sides)
    if float(args.hz) <= 0.0:
        raise ValueError("--hz must be positive.")
    if float(args.policy_hz) <= 0.0:
        raise ValueError("--policy-hz must be positive.")
    if float(args.status_sec) <= 0.0:
        raise ValueError("--status-sec must be positive.")
    if float(args.hand_telemetry_hz) <= 0.0:
        raise ValueError("--hand-telemetry-hz must be positive.")
    if int(args.max_fingertip_force) < 0:
        raise ValueError("--inspire-hand-max-force/--max-force must be non-negative.")
    if int(args.min_action_queue) < 0:
        raise ValueError("--min-action-queue must be non-negative.")
    replay_dataset = getattr(args, "debug_replay_dataset", None)
    replay_parquet = getattr(args, "debug_replay_parquet", None)
    if replay_dataset and replay_parquet:
        raise ValueError("Use only one of --debug-replay-dataset or --debug-replay-parquet.")
    replay_enabled = bool(replay_dataset or replay_parquet)
    if replay_enabled:
        if str(args.policy_spec or "hold").strip().lower() not in {"", "hold", "builtin:hold"}:
            raise ValueError("--debug-replay-* replaces --policy; omit --policy in replay mode.")
        if int(args.debug_replay_episode) < 0:
            raise ValueError("--debug-replay-episode must be non-negative.")
        if int(args.debug_replay_start_index) < 0:
            raise ValueError("--debug-replay-start-index must be non-negative.")
        if args.debug_replay_end_index is not None and int(args.debug_replay_end_index) < 0:
            raise ValueError("--debug-replay-end-index must be non-negative.")
        if (
            args.debug_replay_end_index is not None
            and int(args.debug_replay_end_index) <= int(args.debug_replay_start_index)
        ):
            raise ValueError("--debug-replay-end-index must be greater than --debug-replay-start-index.")
        if int(args.debug_replay_chunk_size) <= 0:
            raise ValueError("--debug-replay-chunk-size must be positive.")
    parse_tactile_regions(args.hand_tactile_regions)
    if args.send_actions:
        for side in active_sides:
            if not getattr(args, f"{side}_hand_ip"):
                raise ValueError(
                    f"--send-actions requires --{side}-hand-ip for active side '{side}'."
                )
    if args.visualize_tactile:
        for side in active_sides:
            if not getattr(args, f"{side}_hand_ip"):
                raise ValueError(
                    f"--visualize-tactile requires --{side}-hand-ip for active side '{side}'."
                )
    if args.enable_arm:
        if int(args.fr3_port) <= 0:
            raise ValueError("--fr3-port must be positive.")
        if float(args.fr3_timeout_s) <= 0.0:
            raise ValueError("--fr3-timeout-s must be positive.")
        if float(args.fr3_state_poll_hz) <= 0.0:
            raise ValueError("--fr3-state-poll-hz must be positive.")
        if float(args.fr3_max_state_age_s) <= 0.0:
            raise ValueError("--fr3-max-state-age-s must be positive.")
        if float(args.fr3_state_ready_timeout_s) <= 0.0:
            raise ValueError("--fr3-state-ready-timeout-s must be positive.")
    if args.camera_mode in {"single", "multi"}:
        if args.camera_mode == "multi" and args.camera_serial:
            raise ValueError("--camera-mode multi uses --camera-config; omit --camera-serial.")
        if not args.camera_serial and not Path(args.camera_config).exists():
            raise ValueError(
                f"--camera-mode {args.camera_mode} requires --camera-serial "
                "or an existing --camera-config."
            )
        if int(args.camera_width) <= 0 or int(args.camera_height) <= 0:
            raise ValueError("--camera-width/--camera-height must be positive.")
        if int(args.camera_fps) <= 0:
            raise ValueError("--camera-fps must be positive.")
        if int(args.camera_keep_last) <= 0:
            raise ValueError("--camera-keep-last must be positive.")
    elif bool(args.vis_cam):
        raise ValueError("--vis_cam requires --camera-mode single or --camera-mode multi.")
    return active_sides


def build_hand_backends(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> dict[str, InspireTacHandBackend]:
    tactile_regions = parse_tactile_regions(args.hand_tactile_regions)
    backends: dict[str, InspireTacHandBackend] = {}
    for side in active_sides:
        hand_ip = str(getattr(args, f"{side}_hand_ip") or "")
        backends[side] = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                side=side,
                effector_name=f"{side}_hand",
                hand_ip=hand_ip or "0.0.0.0",
                port=int(args.hand_modbus_port),
                connect_hardware=bool(hand_ip),
                max_fingertip_force=int(args.max_fingertip_force),
                send_threshold=float(args.hand_send_threshold),
                enable_telemetry=bool(hand_ip),
                telemetry_hz=float(args.hand_telemetry_hz),
                tactile_regions=tuple(tactile_regions),
            )
        )
    return backends


def build_arm_backend(args: argparse.Namespace) -> Any | None:
    if not args.enable_arm:
        return None
    from ..arm.fr3_backend import FR3ArmBackend, FR3ArmBackendConfig

    return FR3ArmBackend(
        FR3ArmBackendConfig(
            host=str(args.fr3_host),
            port=int(args.fr3_port),
            timeout_s=float(args.fr3_timeout_s),
            side=str(args.arm_side),
            effector_name="arm",
            state_poll_hz=float(args.fr3_state_poll_hz),
            max_state_age_s=float(args.fr3_max_state_age_s),
            wait_state_timeout_s=float(args.fr3_state_ready_timeout_s),
            startup_align_enabled=not bool(args.disable_arm_startup_align),
            startup_align_duration_s=float(args.arm_startup_align_duration_s),
            startup_align_threshold_rad=float(args.arm_startup_align_threshold_rad),
            silent_status=True,
        )
    )


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
            metadata={"source": "deploy_cli"},
        )
        return {config.camera_key: RealSenseRGBSource(config)}
    camera_configs = load_realsense_camera_config_map(Path(args.camera_config))
    if args.camera_mode == "single" and len(camera_configs) != 1:
        raise ValueError(
            f"--camera-mode single expects exactly one camera config, got {len(camera_configs)}."
        )
    if args.camera_mode == "multi" and not camera_configs:
        raise ValueError("--camera-mode multi requires at least one camera config.")
    return {key: RealSenseRGBSource(config) for key, config in camera_configs.items()}


def select_preview_camera_sources(
    camera_sources: dict[str, RealSenseRGBSource],
    requested_key: str,
) -> dict[str, RealSenseRGBSource]:
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


def camera_status_text(
    camera_sources: dict[str, RealSenseRGBSource],
    camera_previewer: CameraPreviewer | None,
) -> str:
    if not camera_sources:
        return "camera=off"
    bits: list[str] = []
    if camera_previewer is not None and camera_previewer.disabled_reason:
        bits.append(
            "preview=disabled:"
            + compact_log_value(camera_previewer.disabled_reason, max_chars=80)
        )
    for camera_key, camera_source in sorted(camera_sources.items()):
        try:
            info = camera_source.device_info()
        except Exception as exc:  # noqa: BLE001
            bits.append(f"{camera_key}=status_error:{type(exc).__name__}")
            continue
        frame_count = int(info.get("frame_count", 0) or 0)
        last_error = info.get("last_error")
        if last_error:
            bits.append(
                f"{camera_key}=frames:{frame_count},err:"
                f"{compact_log_value(last_error, max_chars=70)}"
            )
        else:
            bits.append(f"{camera_key}=frames:{frame_count}")
    return "camera=" + ";".join(bits)


def camera_state_signature_text(
    camera_sources: dict[str, RealSenseRGBSource],
    camera_previewer: CameraPreviewer | None,
) -> tuple[tuple[str, ...], str]:
    if not camera_sources:
        return (), "camera=off"
    signature_parts: list[str] = []
    text_parts: list[str] = []
    if camera_previewer is not None:
        disabled_reason = camera_previewer.disabled_reason
        if disabled_reason:
            reason_text = compact_log_value(disabled_reason, max_chars=80)
            signature_parts.append(f"preview:disabled:{reason_text}")
            text_parts.append(f"preview=disabled:{reason_text}")
        else:
            signature_parts.append("preview:active")
            text_parts.append("preview=active")
    for camera_key, camera_source in sorted(camera_sources.items()):
        try:
            info = camera_source.device_info()
        except Exception as exc:  # noqa: BLE001
            state = f"status_error:{type(exc).__name__}"
            signature_parts.append(f"{camera_key}:{state}")
            text_parts.append(f"{camera_key}={state}")
            continue
        frame_count = int(info.get("frame_count", 0) or 0)
        last_error = info.get("last_error")
        if last_error:
            error_text = compact_log_value(last_error, max_chars=70)
            signature_parts.append(f"{camera_key}:error:{error_text}")
            text_parts.append(f"{camera_key}=error:{error_text}")
        elif frame_count > 0:
            signature_parts.append(f"{camera_key}:streaming")
            text_parts.append(f"{camera_key}=streaming")
        else:
            signature_parts.append(f"{camera_key}:waiting")
            text_parts.append(f"{camera_key}=waiting")
    return tuple(signature_parts), "camera=" + ";".join(text_parts)


def hand_state_snapshot(backends: Mapping[str, InspireTacHandBackend]) -> dict[str, dict[str, Any]]:
    return {side: backend.get_state() for side, backend in backends.items()}


def arm_state_snapshot(backend: Any | None) -> dict[str, Any]:
    if backend is None:
        return {}
    return backend.get_state()


def print_controls(*, send_actions: bool, auto_start: bool) -> None:
    print("\n" + "=" * 60)
    print("策略部署循环已启动 / Policy deployment loop started")
    print(f"  send_actions={'on' if send_actions else 'off(dry-run)'}")
    print(f"  initial_policy_state={'active' if auto_start else 'paused'}")
    print("Controls:")
    print("  'G' = start/resume policy execution")
    print("  'H' = pause policy execution and clear queued chunk")
    print("  'Q' = quit")
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
    hand_backends: Mapping[str, InspireTacHandBackend],
    arm_backend: Any | None,
) -> None:
    arm_command = commands.get("arm")
    if arm_command is not None and arm_backend is not None:
        arm_backend.send_command(arm_command)
    for side in HAND_VECTOR_ORDER:
        command = commands.get(f"{side}_hand")
        backend = hand_backends.get(side)
        if command is not None and backend is not None:
            backend.send_command(command)


def _compact_faults(faults: list[str], *, limit: int = 3) -> str:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deploy a policy on InspireTac hands, optional FR3 arm, tactile, and RealSense."
    )
    add_args(parser)
    args = parser.parse_args(argv)
    active_sides = validate_args(args)
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
        policy_kwargs = parse_policy_kwargs(args.policy_kwargs_json)
        policy = load_policy(args.policy_spec, kwargs=policy_kwargs)

    hand_backends = build_hand_backends(args, active_sides)
    arm_backend = build_arm_backend(args)
    camera_sources = build_camera_sources(args)
    preview_camera_sources = (
        select_preview_camera_sources(camera_sources, str(args.vis_cam_key))
        if bool(args.vis_cam) and camera_sources
        else {}
    )
    camera_previewer = (
        CameraPreviewer(CameraPreviewConfig())
        if bool(args.vis_cam) and preview_camera_sources
        else None
    )
    visualizer = (
        TactileUVVisualizer(
            TactileVisualizerConfig(
                enabled_sides=active_sides,
                image_size=(int(args.tactile_vis_size), int(args.tactile_vis_size)),
                value_scale=float(args.tactile_vis_value_scale),
                resizable_window=str(args.tactile_vis_window_mode) == "resizable",
            )
        )
        if args.visualize_tactile
        else None
    )
    safety_filter = ActionSafetyFilter(
        ActionSafetyFilterConfig(
            active_sides=active_sides,
            arm_enabled=bool(args.enable_arm),
            arm_shrink_deg=float(args.arm_shrink_deg),
            max_arm_step_rad=float(args.max_arm_step_rad),
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
    last_camera_signature: tuple[str, ...] | None = None
    last_fault_text: str | None = None
    started_at = time.time()

    try:
        print(
            "[Deploy] policy="
            f"{policy.name} active_sides={active_sides} arm={bool(args.enable_arm)} "
            f"hz={float(args.hz):.2f} policy_hz={float(args.policy_hz):.2f}"
        )
        if policy.metadata.get("debug_replay"):
            print(
                "[Replay] source="
                f"{policy.metadata.get('source_path')} frames={policy.metadata.get('frame_count')} "
                f"chunk_size={int(args.debug_replay_chunk_size)} "
                f"loop={bool(args.debug_replay_loop)} "
                f"episode={policy.metadata.get('episode_index', 'direct')}"
            )
        if not send_actions:
            print("[Deploy] dry-run: actions are filtered and logged but not sent to hardware.")
        for side, backend in hand_backends.items():
            backend.start()
            if not backend.is_ready() and not bool(args.allow_hand_connect_failure):
                raise RuntimeError(f"{side} hand backend is not ready: {backend.get_state()}")
        if arm_backend is not None:
            arm_backend.start()
            if not arm_backend.is_ready() and not bool(args.allow_arm_connect_failure):
                raise RuntimeError(f"FR3 arm backend is not ready: {arm_backend.device_info()}")
        if camera_sources:
            print(
                "[Camera] configured "
                f"cameras={sorted(camera_sources)} mode={args.camera_mode} "
                f"vis_cam={bool(args.vis_cam)} "
                f"vis_camera_keys={sorted(preview_camera_sources)}"
            )
            if camera_previewer is not None:
                if camera_previewer.ensure_available():
                    print("[Camera] --vis_cam display preflight ok.")
                else:
                    print(
                        "[Camera] --vis_cam requested but preview is disabled: "
                        f"{camera_previewer.disabled_reason}"
                    )
        for camera_source in camera_sources.values():
            camera_source.start()
        print_controls(send_actions=send_actions, auto_start=bool(args.auto_start))

        with NonBlockingKeyboard() as keyboard:
            while True:
                loop_started_at = time.time()
                key = keyboard.poll()
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

                hand_states = hand_state_snapshot(hand_backends)
                arm_state = arm_state_snapshot(arm_backend)
                control_hz = loop_rate.tick(loop_started_at)
                observation = build_deployment_observation(
                    active_sides=active_sides,
                    arm_enabled=bool(args.enable_arm),
                    step_index=step_index,
                    hand_states=hand_states,
                    arm_state=arm_state,
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
                observation_faults = list(observation.faults)
                warmup_active = loop_started_at - started_at < float(args.startup_warmup_sec)
                if warmup_active:
                    observation_faults.append("startup_warmup")

                if visualizer is not None:
                    visualizer.update(
                        tactile_by_side_from_backends(active_sides, hand_backends.get)
                    )
                if camera_previewer is not None:
                    camera_previewer.update(preview_camera_sources)
                if camera_sources:
                    camera_signature, camera_text = camera_state_signature_text(
                        camera_sources,
                        camera_previewer,
                    )
                    if camera_signature != last_camera_signature:
                        print(f"[Camera] state changed: {camera_text}")
                        last_camera_signature = camera_signature

                if policy_active:
                    can_run_policy = not (
                        bool(args.require_ready_observation) and observation_faults
                    )
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
                                arm_enabled=bool(args.enable_arm),
                            )
                            if actions:
                                if args.append_action_chunks:
                                    chunk_buffer.append(actions)
                                else:
                                    chunk_buffer.replace(actions)
                                last_policy_at = loop_started_at
                            else:
                                observation_faults.append("policy_empty_action")
                        except Exception as exc:  # noqa: BLE001
                            print(f"[Deploy] policy error: {type(exc).__name__}: {exc}")
                            chunk_buffer.clear()
                            observation_faults.append(f"policy_error:{type(exc).__name__}")
                            if bool(args.pause_on_policy_error):
                                policy_active = False

                    raw_action = None
                    if can_run_policy:
                        raw_action = chunk_buffer.pop(default=observation.hold_action())
                    if raw_action is not None:
                        deployment_action = safety_filter.filter(
                            raw_action,
                            observation,
                            timestamp=loop_started_at,
                            policy_name=policy.name,
                        )
                        observation_faults.extend(deployment_action.faults)
                        if deployment_action.commands:
                            if send_actions:
                                _send_commands(
                                    deployment_action.commands,
                                    hand_backends=hand_backends,
                                    arm_backend=arm_backend,
                                )
                            last_action = deployment_action.vector.astype(np.float32, copy=True)

                fault_text = _compact_faults(observation_faults)
                if fault_text != last_fault_text:
                    print(f"[State] faults={fault_text}")
                    last_fault_text = fault_text

                if loop_started_at - last_status_at >= float(args.status_sec):
                    policy_status = _policy_status_text(policy)
                    policy_status_suffix = f" {policy_status}" if policy_status else ""
                    print(
                        "[Status] "
                        f"step={step_index} hz={control_hz:.2f} "
                        f"policy={'active' if policy_active else 'paused'} "
                        f"send={'on' if send_actions else 'dry-run'} "
                        f"queue={len(chunk_buffer)} "
                        f"state_valid={int(np.sum(observation.state_valid))}/{ACTION_DIM} "
                        f"tactile={','.join(k for k, v in observation.tactile_valid_by_side.items() if v) or 'none'} "
                        f"{camera_status_text(camera_sources, camera_previewer)}"
                        f"{policy_status_suffix}"
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
        if arm_backend is not None:
            arm_backend.close()
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
