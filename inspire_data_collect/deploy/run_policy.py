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
from ..hand.dexhand_q6_backend import (
    DEXHAND_TACTILE_AXIS_ORDER,
    DEXHAND_TACTILE_DIM,
    DEXHAND_TACTILE_FINGER_ORDER,
    DexHandQ6Backend,
    DexHandQ6BackendConfig,
)
from ..hand.hand_mapping import parse_tactile_regions
from ..utils.camera_preview import CameraPreviewConfig, CameraPreviewer
from ..utils.realsense_rgb import (
    RealSenseRGBCameraConfig,
    RealSenseRGBSource,
    load_realsense_camera_config_map,
)
from ..utils.tactile_visualizer import (
    DexHandForceVisualizer,
    DexHandForceVisualizerConfig,
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
from .replay import infer_replay_action_hz, load_dataset_replay_policy, resolve_replay_parquet_path
from .startup_init import StartupInitializer, resolve_startup_init_action


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


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _non_negative_float(text: str) -> float:
    value = float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


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
    parser.add_argument(
        "--hand-mode",
        choices=["inspire_hand", "dexhand_q6"],
        default="inspire_hand",
        help=(
            "Hardware backend used for hand commands. 'dexhand_q6' keeps the dataset/deploy "
            "action slot on the right hand while controlling one left DexHand hardware hand."
        ),
    )
    parser.add_argument("--hz", type=_positive_float, default=13.0, help="Fallback control loop rate.")
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
        help="Max synchronous policy query rate. Omit to match the effective action/control rate.",
    )
    parser.add_argument(
        "--min-action-queue",
        type=int,
        default=0,
        help="Query policy when queued chunk length is <= this value.",
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
        "--init-time-s",
        type=float,
        default=5.0,
        help="Seconds used to smoothly move to the selected initial target before execution.",
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
    parser.add_argument("--init-episode", type=int, default=0)
    parser.add_argument("--init-frame", type=int, default=0)
    parser.add_argument("--init-action-column", type=str, default="action")
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
    parser.add_argument(
        "--dexhand-hardware-hand",
        choices=["left", "right"],
        default="left",
        help="Physical DexHand side connected to the controller. Current supported dataset mapping expects left hardware.",
    )
    parser.add_argument("--dexhand-device-type", choices=["auto", "config", "100U", "200U"], default="auto")
    parser.add_argument("--dexhand-device-index", type=int, default=0)
    parser.add_argument("--dexhand-tactile-period-ms", type=int, default=20)
    parser.add_argument("--dexhand-send-threshold", type=float, default=0.002)
    parser.add_argument("--dexhand-mock", action="store_true")
    parser.add_argument("--no-dexhand-clear-errors-on-start", dest="dexhand_clear_errors_on_start", action="store_false", default=True)
    parser.add_argument("--no-dexhand-clear-errors-on-fault", dest="dexhand_clear_errors_on_fault", action="store_false", default=True)

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
        help="Compatibility option; hand q6 targets are clipped to [0, 1] and sent directly.",
    )

    parser.add_argument(
        "--camera-mode",
        choices=["none", "single", "multi"],
        default="none",
    )
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
    parser.add_argument(
        "--require-camera-frame",
        action="store_true",
        help="Treat missing camera frames as observation faults.",
    )
    parser.add_argument("--vis_cam", "--vis-cam", dest="vis_cam", action="store_true")
    parser.add_argument("--vis-cam-key", "--vis-camera-key", type=str, default="")
    parser.add_argument(
        "--vis-cam-window-mode",
        choices=["resizable", "autosize"],
        default="resizable",
    )
    parser.add_argument("--vis-cam-render-scale", type=float, default=2.0)

    parser.add_argument("--visualize-tactile", action="store_true")
    parser.add_argument("--tactile-vis-value-scale", type=float, default=0.02)
    parser.add_argument("--tactile-vis-size", type=int, default=640)
    parser.add_argument("--tactile-vis-render-scale", type=float, default=2.0)
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
    if float(args.action_hz) < 0.0:
        raise ValueError("--action-hz must be non-negative.")
    if args.policy_hz is not None and float(args.policy_hz) <= 0.0:
        raise ValueError("--policy-hz must be positive.")
    if float(args.status_sec) <= 0.0:
        raise ValueError("--status-sec must be positive.")
    if float(args.hand_telemetry_hz) <= 0.0:
        raise ValueError("--hand-telemetry-hz must be positive.")
    if int(args.max_fingertip_force) < 0:
        raise ValueError("--inspire-hand-max-force/--max-force must be non-negative.")
    if int(args.min_action_queue) < 0:
        raise ValueError("--min-action-queue must be non-negative.")
    if float(args.init_time_s) < 0.0:
        raise ValueError("--init-time-s must be non-negative.")
    if args.init_dataset and args.init_parquet:
        raise ValueError("Use only one of --init-dataset or --init-parquet.")
    if int(args.init_episode) < 0:
        raise ValueError("--init-episode must be non-negative.")
    if int(args.init_frame) < 0:
        raise ValueError("--init-frame must be non-negative.")
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
    action_hz_start = args.debug_replay_start_index if args.action_hz_start_index is None else args.action_hz_start_index
    action_hz_end = args.debug_replay_end_index if args.action_hz_end_index is None else args.action_hz_end_index
    if action_hz_end is not None and int(action_hz_end) < 0:
        raise ValueError("--action-hz-end-index must be non-negative.")
    if action_hz_end is not None and int(action_hz_end) <= int(action_hz_start):
        raise ValueError("--action-hz-end-index must be greater than --action-hz-start-index.")
    if args.action_hz_dataset and args.action_hz_parquet:
        raise ValueError("Use only one of --action-hz-dataset or --action-hz-parquet.")
    parse_tactile_regions(args.hand_tactile_regions)
    if str(args.hand_mode) == "inspire_hand":
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
    else:
        if active_sides != ("right",):
            raise ValueError("--hand-mode dexhand_q6 currently requires --active-sides right.")
        if str(args.dexhand_hardware_hand) != "left":
            raise ValueError("--hand-mode dexhand_q6 currently only supports --dexhand-hardware-hand left.")
        if int(args.dexhand_tactile_period_ms) < 0:
            raise ValueError("--dexhand-tactile-period-ms must be non-negative.")
        if float(args.dexhand_send_threshold) < 0.0:
            raise ValueError("--dexhand-send-threshold must be non-negative.")
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
        if float(args.vis_cam_render_scale) <= 0.0:
            raise ValueError("--vis-cam-render-scale must be positive.")
    elif bool(args.vis_cam):
        raise ValueError("--vis_cam requires --camera-mode single or --camera-mode multi.")
    if float(args.tactile_vis_render_scale) <= 0.0:
        raise ValueError("--tactile-vis-render-scale must be positive.")
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
        try:
            estimate = infer_replay_action_hz(
                parquet_path=path,
                dataset_root=dataset,
                episode_index=int(episode),
                start_index=int(start_index),
                end_index=end_index,
                resolved_metadata=metadata,
            )
        except ValueError as exc:
            estimate = {
                "hz": float(args.hz),
                "source": "cli.hz_after_inference_failed",
                "warning": str(exc),
            }
        hz = float(estimate.get("hz", 0.0) or 0.0)
        if hz > 0.0:
            estimate = dict(estimate)
            estimate["source_path"] = str(path)
            return hz, estimate

    fallback_hz = float(args.hz)
    return fallback_hz, {"source": "cli.hz", "hz": fallback_hz}


def resolve_effective_policy_hz(args: argparse.Namespace, *, action_hz: float) -> tuple[float, str]:
    if args.policy_hz is not None:
        return float(args.policy_hz), "cli.policy_hz"
    return float(action_hz), "action_hz"


def build_hand_backends(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> dict[str, Any]:
    if str(args.hand_mode) == "dexhand_q6":
        return {
            "right": DexHandQ6Backend(
                DexHandQ6BackendConfig(
                    side="right",
                    hardware_hand=str(args.dexhand_hardware_hand),
                    effector_name="right_hand",
                    connect_hardware=bool(args.send_actions),
                    send_commands=bool(args.send_actions),
                    send_threshold=float(args.dexhand_send_threshold),
                    enable_telemetry=True,
                    telemetry_hz=float(args.hand_telemetry_hz),
                    device_type=str(args.dexhand_device_type),
                    device_index=int(args.dexhand_device_index),
                    tactile_period_ms=int(args.dexhand_tactile_period_ms),
                    mock=bool(args.dexhand_mock),
                    clear_errors_on_start=bool(args.dexhand_clear_errors_on_start),
                    clear_errors_on_fault=bool(args.dexhand_clear_errors_on_fault),
                )
            )
        }

    tactile_regions = parse_tactile_regions(args.hand_tactile_regions)
    backends: dict[str, Any] = {}
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


def _resolve_action_queue_policy(args: argparse.Namespace, *, debug_replay: bool) -> str:
    if bool(getattr(args, "append_action_chunks", False)):
        return "append"
    policy = str(getattr(args, "action_queue_policy", "auto") or "auto").strip().lower()
    if policy == "auto":
        return "append" if bool(debug_replay) else "replace"
    if policy not in {"replace", "append"}:
        raise ValueError(f"Unsupported action queue policy: {policy!r}.")
    return policy


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


def _skip_startup_replay_frames(policy: Any, count: int = 1) -> int:
    wrapped_policy = getattr(policy, "policy", policy)
    skip = getattr(wrapped_policy, "skip_startup_frames", None)
    if not callable(skip):
        return 0
    return int(skip(count))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deploy a policy on InspireTac hands, optional FR3 arm, tactile, and RealSense."
    )
    add_args(parser)
    args = parser.parse_args(argv)
    active_sides = validate_args(args)
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
        policy_kwargs = parse_policy_kwargs(args.policy_kwargs_json)
        policy = load_policy(args.policy_spec, kwargs=policy_kwargs)
    action_queue_policy = _resolve_action_queue_policy(args, debug_replay=debug_replay)
    startup_init_action, startup_init_source = resolve_startup_init_action(args, policy)
    effective_action_hz, action_hz_info = resolve_effective_action_hz(args, policy)
    effective_policy_hz, policy_hz_source = resolve_effective_policy_hz(
        args,
        action_hz=float(effective_action_hz),
    )

    hand_backends = build_hand_backends(args, active_sides)
    arm_backend = build_arm_backend(args)
    camera_sources = build_camera_sources(args)
    preview_camera_sources = (
        select_preview_camera_sources(camera_sources, str(args.vis_cam_key))
        if bool(args.vis_cam) and camera_sources
        else {}
    )
    camera_previewer = (
        CameraPreviewer(
            CameraPreviewConfig(
                resizable_window=str(args.vis_cam_window_mode) == "resizable",
                render_scale=float(args.vis_cam_render_scale),
            )
        )
        if bool(args.vis_cam) and preview_camera_sources
        else None
    )
    visualizer = None
    if args.visualize_tactile:
        if str(args.hand_mode) == "dexhand_q6":
            visualizer = DexHandForceVisualizer(
                DexHandForceVisualizerConfig(
                    enabled_sides=active_sides,
                    image_size=(int(args.tactile_vis_size), int(args.tactile_vis_size)),
                    value_scale=float(args.tactile_vis_value_scale),
                    resizable_window=str(args.tactile_vis_window_mode) == "resizable",
                    render_scale=float(args.tactile_vis_render_scale),
                )
            )
        else:
            visualizer = TactileUVVisualizer(
                TactileVisualizerConfig(
                    enabled_sides=active_sides,
                    image_size=(int(args.tactile_vis_size), int(args.tactile_vis_size)),
                    value_scale=float(args.tactile_vis_value_scale),
                    resizable_window=str(args.tactile_vis_window_mode) == "resizable",
                    render_scale=float(args.tactile_vis_render_scale),
                )
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
    startup_initializer = StartupInitializer(
        duration_s=float(args.init_time_s),
        source=startup_init_source,
        target_action=startup_init_action,
        active_sides=active_sides,
        arm_enabled=bool(args.enable_arm),
    )
    startup_initializer.reset()

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
            f"{policy.name} active_sides={active_sides} hand_mode={args.hand_mode} arm={bool(args.enable_arm)} "
            f"hz={float(effective_action_hz):.2f} policy_hz={float(effective_policy_hz):.2f} "
            f"queue_policy={action_queue_policy}"
        )
        if str(args.hand_mode) == "dexhand_q6":
            print(
                "[DexHand] deployment active: dataset_slot=right "
                f"hardware_hand={args.dexhand_hardware_hand} tactile_dim={int(DEXHAND_TACTILE_DIM)} "
                f"tactile_layout={list(DEXHAND_TACTILE_FINGER_ORDER)}x{list(DEXHAND_TACTILE_AXIS_ORDER)}"
            )
        if action_hz_info.get("source") != "cli.hz":
            print(
                "[Deploy] action_hz="
                f"{float(effective_action_hz):.3f} source={action_hz_info.get('source')} "
                f"policy_hz_source={policy_hz_source}"
            )
        if startup_initializer.enabled:
            print(
                "[StartupInit] enabled "
                f"source={startup_init_source} duration={float(args.init_time_s):.2f}s"
            )
        else:
            print("[StartupInit] disabled")
        if policy.metadata.get("debug_replay"):
            print(
                "[Replay] source="
                f"{policy.metadata.get('source_path')} frames={policy.metadata.get('frame_count')} "
                f"chunk_size={int(args.debug_replay_chunk_size)} "
                f"loop={bool(args.debug_replay_loop)} "
                f"episode={policy.metadata.get('episode_index', 'direct')} "
                f"action_hz={policy.metadata.get('action_hz')}"
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
                    startup_initializer.reset()
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
                    if startup_initializer.enabled and not startup_initializer.completed:
                        if can_run_policy:
                            init_action, init_faults, init_ready = startup_initializer.next_action(
                                observation,
                                now=loop_started_at,
                            )
                            observation_faults.extend(init_faults)
                            if init_action is not None:
                                deployment_action = safety_filter.filter(
                                    init_action,
                                    observation,
                                    timestamp=loop_started_at,
                                    policy_name="startup_init",
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
                                if init_ready and not deployment_action.faults:
                                    startup_initializer.mark_complete()
                                    if startup_initializer.source == "replay:first_frame":
                                        skipped = _skip_startup_replay_frames(policy, count=1)
                                        if skipped:
                                            print(
                                                "[StartupInit] replay startup target consumed; "
                                                f"skipped {skipped} replay frame(s)."
                                            )
                                    chunk_buffer.clear()
                                    last_policy_at = None
                        can_run_policy = False
                    if can_run_policy and _policy_should_run(
                        now=loop_started_at,
                        last_policy_at=last_policy_at,
                        policy_interval_s=1.0 / float(effective_policy_hz),
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
                                if action_queue_policy == "append":
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
                    status_parts = [
                        text
                        for text in (startup_initializer.status_text, policy_status)
                        if text
                    ]
                    policy_status_suffix = f" {' '.join(status_parts)}" if status_parts else ""
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
                sleep_time = (1.0 / float(effective_action_hz)) - (time.time() - loop_started_at)
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
