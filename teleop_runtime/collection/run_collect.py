from __future__ import annotations

import argparse
import os
import re
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from datetime import datetime
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
from ..calibration import TeleopCalibration
from ..configs import build_annotated_dual_rm65_calibration
from ..hand_mapping import DEFAULT_INSPIRE_HAND_JOINT_ORDER, parse_tactile_regions
from ..hand_tcp_alignment import (
    DEFAULT_VIVE_TRACKER_HEIGHT_CORRECTION_M,
    tracker_hand_to_rm65_tcp_rotation,
    vive_tracker_height_correction_offset,
)
from ..LeftArmFixed import (
    FixedArmRetargeter,
    FixedArmRetargeterConfig,
    SmoothJitterArmRetargeter,
    SmoothJitterArmRetargeterConfig,
)
from ..retargeters import (
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
    JointsToDexHandRetargeter,
    JointsToDexHandRetargeterConfig,
    PoseToArmRetargeter,
    PoseToArmRetargeterConfig,
)
from ..diagnostics import TeleopRunState
from ..rigs import BimanualRig
from ..safety import TeleopSafetyConfig
from ..schemas import (
    ObservationChannel,
    TeleopCommand,
    TeleopObservation,
    merge_teleop_commands,
)
from ..session import TeleopSession, TeleopSessionConfig
from ..sources.camera import RealSenseRGBSource, load_realsense_camera_config_map
from ..sources.hand import HandQ6Source
from ..sources.misc import KeyboardButtonSource
from ..sources.pose import VivePoseSource, ViveTrackerConfig
from ..tactile_visualizer import (
    TactileUVVisualizer,
    TactileVisualizerConfig,
    tactile_by_side_from_backends,
)
from ..transforms import as_pose_matrix, interpolate_pose
from .async_writer import AsyncWriterConfig, AsyncWriterPool
from .build_collect_components import CollectorComponents
from .camera_preview import CameraPreviewConfig, CameraPreviewer
from .runtime import CollectionRuntime, CollectionRuntimeConfig
from .sample_assembler import CollectorSampleBuilder, CollectorSampleBuilderConfig
from .writer_v21 import LeRobotV21Writer, LeRobotV21WriterConfig


DEFAULT_CAMERA_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "devices" / "realsense.json"
)


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value


def _alpha_float(text: str) -> float:
    value = float(text)
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError("alpha must be in (0, 1]")
    return value


def _parse_joint_order(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("joint order must contain 6 comma-separated indices")
    try:
        order = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid joint order: {text}") from exc
    if sorted(order) != [0, 1, 2, 3, 4, 5]:
        raise argparse.ArgumentTypeError("joint order must be a permutation of 0,1,2,3,4,5")
    return order


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
    if len(values) != 6:
        raise argparse.ArgumentTypeError("arm max angvel must contain 1 or 6 floats")
    if any(value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("arm max angvel must be positive")
    return tuple(float(value) for value in values)


def _parse_hand_tactile_regions(value: object) -> tuple[str, ...]:
    return parse_tactile_regions(value)


def _parse_active_sides(text: str) -> tuple[str, ...]:
    value = str(text).strip().lower()
    if value == "both":
        return ("left", "right")
    if value in {"left", "right"}:
        return (value,)
    raise argparse.ArgumentTypeError("active-sides must be one of: left, right, both")


def _sanitize_task_name_for_path(task_name: str) -> str:
    text = str(task_name).strip()
    if not text:
        return "task"
    text = re.sub(r"[\\/]+", "-", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("._-")
    return text or "task"


def _default_task_id_from_name(task_name: str) -> str:
    slug = _sanitize_task_name_for_path(task_name).lower()
    slug = re.sub(r"[^0-9a-z_-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "task"


def _resolve_collection_layout(args: argparse.Namespace) -> tuple[Path, str, str, str, str]:
    legacy_task = str(getattr(args, "task", "")).strip()
    task_name = str(getattr(args, "task_name", "")).strip() or legacy_task or "default"
    task_id = (
        str(getattr(args, "task_id", "")).strip()
        or legacy_task
        or _default_task_id_from_name(task_name)
    )
    task_description = str(getattr(args, "task_description", "")).strip() or task_name
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    dataset_dir_name = f"{date_prefix}_{_sanitize_task_name_for_path(task_name)}"
    dataset_root = Path(args.dataset_root).expanduser() / dataset_dir_name
    dataset_name = str(getattr(args, "dataset_name", "")).strip() or dataset_dir_name
    return dataset_root, task_name, task_id, task_description, dataset_name


def _apply_collection_layout(args: argparse.Namespace) -> None:
    dataset_base_root = Path(args.dataset_root).expanduser()
    dataset_root, task_name, task_id, task_description, dataset_name = _resolve_collection_layout(args)
    args.dataset_base_root = str(dataset_base_root)
    args.dataset_root = str(dataset_root)
    args.dataset_dir_name = dataset_root.name
    args.task_name = task_name
    args.task_id = task_id
    args.task = task_id
    args.task_description = task_description
    args.dataset_name = dataset_name


def _active_side_value(args: argparse.Namespace, side: str, suffix: str) -> str:
    return str(getattr(args, f"{side}_{suffix}"))


def _assert_side_config_consistency(args: argparse.Namespace, active_sides: tuple[str, ...]) -> None:
    for side in ("left", "right"):
        if side in active_sides:
            if not _active_side_value(args, side, "arm_ip"):
                raise ValueError(f"--{side}-arm-ip is required when active-sides includes '{side}'.")
            continue
        unexpected_values = {
            "arm_ip": _active_side_value(args, side, "arm_ip"),
            "hand_ip": _active_side_value(args, side, "hand_ip"),
            "hand_port": _active_side_value(args, side, "hand_port"),
        }
        present = [flag for flag, value in unexpected_values.items() if value]
        if present:
            raise ValueError(
                f"Inactive side '{side}' received configuration for {present}; "
                "either remove those flags or include the side in --active-sides."
            )


def _assert_tactile_visualization_config(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> None:
    if not bool(args.visualize_tactile):
        return
    if not bool(args.with_tactile):
        raise ValueError(
            "--visualize-tactile requires tactile polling; remove --without-tactile."
        )
    tactile_regions = _parse_hand_tactile_regions(args.hand_tactile_regions)
    if not tactile_regions:
        raise ValueError(
            "--visualize-tactile requires non-empty --hand-tactile-regions, for example "
            "'thumb_1,index_1,palm' or 'all'."
        )
    missing_ip_sides = [
        side for side in active_sides if not str(getattr(args, f"{side}_hand_ip") or "")
    ]
    if missing_ip_sides:
        raise ValueError(
            "--visualize-tactile requires real InspireTac hand IPs for active sides; "
            f"missing={missing_ip_sides}."
        )


def _fixed_arm_sides(args: argparse.Namespace, active_sides: tuple[str, ...]) -> tuple[str, ...]:
    if not bool(getattr(args, "left_arm_fixed", False)):
        return ()
    if bool(getattr(args, "left_arm_jitter", False)):
        raise ValueError("--left-arm-fixed and --left-arm-jitter are mutually exclusive.")
    if set(active_sides) != {"left", "right"}:
        raise ValueError("--left-arm-fixed requires --active-sides both.")
    return ("left",)


def _jitter_arm_sides(args: argparse.Namespace, active_sides: tuple[str, ...]) -> tuple[str, ...]:
    if not bool(getattr(args, "left_arm_jitter", False)):
        return ()
    if bool(getattr(args, "left_arm_fixed", False)):
        raise ValueError("--left-arm-fixed and --left-arm-jitter are mutually exclusive.")
    if set(active_sides) != {"left", "right"}:
        raise ValueError("--left-arm-jitter requires --active-sides both.")
    if bool(getattr(args, "disable_dual_arm_collision_avoidance", False)):
        raise ValueError("--left-arm-jitter requires dual-arm IK; remove --disable-dual-arm-collision-avoidance.")
    return ("left",)


def _moving_arm_sides(args: argparse.Namespace, active_sides: tuple[str, ...]) -> tuple[str, ...]:
    non_operator = set(_fixed_arm_sides(args, active_sides)) | set(
        _jitter_arm_sides(args, active_sides)
    )
    return tuple(side for side in active_sides if side not in non_operator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect LeRobot-style teleop episodes from real RM65/Inspire hardware."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Base output directory. Actual dataset path is <dataset-root>/<YYYY-MM-DD>_<task-name>.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="",
        help="Task name used for dataset folder naming and metadata. Defaults to --task, then 'default'.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Legacy task id/name. Used as task id and as task name when --task-name is omitted.",
    )
    parser.add_argument("--task-id", type=str, default="", help="Optional task id; defaults to a slug from task name.")
    parser.add_argument("--task-description", type=str, default="", help="Optional task description.")
    parser.add_argument("--dataset-name", type=str, default="", help="Optional dataset name written into meta/info.json.")
    parser.add_argument(
        "--active-sides",
        type=_parse_active_sides,
        default=("left", "right"),
        help="One of: left, right, both.",
    )
    parser.add_argument("--load-profile", type=str, default="", help="Path to a TeleopCalibration JSON profile.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="UDP bind host for the VIVE receiver.")
    parser.add_argument("--port", type=int, default=9999, help="UDP bind port for the VIVE receiver.")
    parser.add_argument("--left-tracker-name", type=str, default="tracker_1", help="Left wrist tracker name.")
    parser.add_argument("--right-tracker-name", type=str, default="tracker_2", help="Right wrist tracker name.")
    parser.add_argument("--left-arm-ip", type=str, default="", help="Left RM65 IP.")
    parser.add_argument("--right-arm-ip", type=str, default="", help="Right RM65 IP.")
    parser.add_argument(
        "--left-arm-fixed",
        action="store_true",
        help=(
            "Keep the left arm fixed at the hardware position captured by p; "
            "right arm and both hands keep normal teleop, and bimanual dataset fields remain populated."
        ),
    )
    parser.add_argument(
        "--left-arm-jitter",
        action="store_true",
        help=(
            "Keep the left arm near the hardware position captured by p, but feed a smooth random "
            "small cartesian jitter target into the dual-arm IK retargeter."
        ),
    )
    parser.add_argument(
        "--left-arm-jitter-position-m",
        type=float,
        default=0.01,
        help="Maximum left TCP jitter translation radius in desk-frame meters.",
    )
    parser.add_argument(
        "--left-arm-jitter-rotation-deg",
        type=float,
        default=3.0,
        help="Maximum left TCP jitter rotation angle in degrees.",
    )
    parser.add_argument(
        "--left-arm-jitter-refresh-sec",
        type=_positive_float,
        default=1.2,
        help="Seconds between new random left-arm jitter targets.",
    )
    parser.add_argument(
        "--left-arm-jitter-smoothing-sec",
        type=float,
        default=0.45,
        help="First-order smoothing time constant for left-arm jitter; 0 jumps directly to each random target.",
    )
    parser.add_argument(
        "--left-arm-jitter-seed",
        type=int,
        default=-1,
        help="Random seed for left-arm jitter; negative uses OS entropy.",
    )
    parser.add_argument("--left-hand-ip", type=str, default="", help="Optional left InspireTac IP.")
    parser.add_argument("--right-hand-ip", type=str, default="", help="Optional right InspireTac IP.")
    parser.add_argument("--left-hand-port", type=str, default="", help="Optional left q6 serial port.")
    parser.add_argument("--right-hand-port", type=str, default="", help="Optional right q6 serial port.")
    parser.add_argument("--hand-reader-baudrate", type=int, default=115200, help="Hand q6 baudrate.")
    parser.add_argument(
        "--hand-reader-timeout",
        type=float,
        default=0.05,
        help="Hand q6 serial timeout in seconds.",
    )
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
    parser.add_argument("--show-hand-raw-lines", action="store_true", help="Print raw hand q6 lines.")
    parser.add_argument(
        "--hand-joint-order",
        type=_parse_joint_order,
        default=DEFAULT_INSPIRE_HAND_JOINT_ORDER,
        help="Permutation applied to the incoming q6 hand order.",
    )
    parser.add_argument("--hand-invert", dest="hand_invert", action="store_true")
    parser.add_argument("--hand-no-invert", dest="hand_invert", action="store_false")
    parser.set_defaults(hand_invert=True)
    parser.add_argument("--hand-command-alpha", type=_alpha_float, default=1.0, help="EMA for hand command mapping.")
    parser.add_argument(
        "--hand-send-threshold",
        type=float,
        default=0.002,
        help="Only send hand commands when max delta exceeds this threshold.",
    )
    parser.add_argument("--hand-telemetry-hz", type=_positive_float, default=30.0, help="Hand telemetry polling rate.")
    parser.add_argument(
        "--hand-tactile-regions",
        type=str,
        default="",
        help="Comma-separated tactile regions or 'all'.",
    )
    parser.add_argument("--with-tactile", dest="with_tactile", action="store_true")
    parser.add_argument("--without-tactile", dest="with_tactile", action="store_false")
    parser.set_defaults(with_tactile=True)
    parser.add_argument(
        "--visualize-tactile",
        action="store_true",
        help="Show the InspireTac tactile UV OpenCV window while collecting.",
    )
    parser.add_argument(
        "--tactile-vis-value-scale",
        type=float,
        default=0.05,
        help="Scale used by the tactile visualizer response curve.",
    )
    parser.add_argument(
        "--tactile-vis-size",
        type=_positive_int,
        default=640,
        help="Per-hand tactile visualizer panel size in pixels.",
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
        "--control-mode",
        choices=["absolute", "delta_clutch"],
        default="absolute",
        help="Arm retargeting mode.",
    )
    parser.add_argument("--translation-scale", type=float, default=1.0, help="Desk-frame translation scale.")
    parser.add_argument(
        "--vive-tracker-height-correction",
        dest="vive_tracker_height_correction",
        action="store_true",
        help="Shift wrist tracker input by -Z in the tracker local frame before arm IK. Default.",
    )
    parser.add_argument(
        "--no-vive-tracker-height-correction",
        dest="vive_tracker_height_correction",
        action="store_false",
        help="Disable the 4cm VIVE tracker local -Z position correction.",
    )
    parser.add_argument(
        "--vive-tracker-height-correction-m",
        type=float,
        default=DEFAULT_VIVE_TRACKER_HEIGHT_CORRECTION_M,
        help="VIVE tracker height correction distance in meters; applied as local [0, 0, -value].",
    )
    parser.set_defaults(vive_tracker_height_correction=True)
    parser.add_argument("--hz", type=_positive_float, default=30.0, help="Target control rate.")
    parser.add_argument(
        "--arm-max-angvel",
        type=_parse_arm_max_angvel,
        default=(0.35, 0.35, 0.35, 0.5, 0.5, 0.5),
        help="Scalar or 6 comma-separated arm velocity limits in rad/s.",
    )
    parser.add_argument(
        "--minimum-control-hz",
        type=_positive_float,
        default=15.0,
        help="Freeze when control rate drops below this threshold.",
    )
    parser.add_argument(
        "--invalid-frame-limit",
        type=_positive_int,
        default=3,
        help="Freeze after this many consecutive invalid critical tracker frames.",
    )
    parser.add_argument(
        "--recovery-healthy-frames",
        type=_positive_int,
        default=10,
        help="Number of consecutive fault-free control frames required before auto recovery starts.",
    )
    parser.add_argument(
        "--recovery-ramp-sec",
        type=_positive_float,
        default=1.0,
        help="Absolute-control recovery ramp duration.",
    )
    parser.add_argument(
        "--disable-dual-arm-collision-avoidance",
        action="store_true",
        help="Disable the joint-space dual-arm collision-aware retargeter.",
    )
    parser.add_argument(
        "--collision-avoidance-mode",
        choices=["boundary_slide", "freeze_on_contact"],
        default="boundary_slide",
        help="Dual-arm collision response mode.",
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
        help="Dual-arm collision scene profile.",
    )
    parser.add_argument(
        "--camera-config",
        type=str,
        default=str(DEFAULT_CAMERA_CONFIG_PATH),
        help="JSON config for RealSense RGB cameras.",
    )
    parser.add_argument("--no-camera", action="store_true", help="Disable camera startup and video export.")
    parser.add_argument(
        "--vis_cam",
        "--vis-cam",
        dest="vis_cam",
        action="store_true",
        help="Show live RealSense RGB frames at the same resolution used for collection.",
    )
    parser.add_argument(
        "--vis-cam-key",
        "--vis-camera-key",
        type=str,
        default="",
        help=(
            "Camera key to visualize when --vis_cam is set. "
            "Defaults to 'front' when present, otherwise the first configured camera."
        ),
    )
    parser.add_argument("--write-video", dest="write_video", action="store_true")
    parser.add_argument("--no-video", dest="write_video", action="store_false")
    parser.set_defaults(write_video=True)
    parser.add_argument("--write-parquet", dest="write_parquet", action="store_true")
    parser.add_argument("--no-parquet", dest="write_parquet", action="store_false")
    parser.set_defaults(write_parquet=True)
    parser.add_argument("--nominal-fps", type=_positive_float, default=30.0, help="Dataset nominal fps metadata.")
    parser.add_argument("--writer-workers", type=int, default=2, help="Async writer worker count.")
    parser.add_argument("--writer-queue-size", type=int, default=256, help="Async writer queue size.")
    parser.add_argument(
        "--calibration-timeout",
        type=_positive_float,
        default=10.0,
        help="Timeout for runtime home calibration after startup.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=_positive_int,
        default=20,
        help="Number of valid wrist frames to average per moving arm side during runtime home calibration.",
    )
    parser.add_argument("--max-steps", type=_non_negative_int, default=0, help="Stop after N control steps, 0 means infinite.")
    parser.add_argument("--status-sec", type=_positive_float, default=2.0, help="Status print interval in seconds.")
    parser.add_argument(
        "--silent",
        "--quiet",
        dest="silent",
        action="store_true",
        default=True,
        help="Suppress periodic status lines; key events and warnings are still printed. Default.",
    )
    parser.add_argument(
        "--no-silent",
        "--no-quiet",
        "--verbose-status",
        dest="silent",
        action="store_false",
        help="Print periodic status lines every --status-sec seconds.",
    )
    args = parser.parse_args()
    if bool(args.no_camera):
        if bool(args.vis_cam):
            parser.error("--vis_cam requires cameras; remove --no-camera.")
        args.write_video = False
    _apply_collection_layout(args)
    return args


@dataclass(frozen=True)
class CollectApp:
    active_sides: tuple[str, ...]
    components: CollectorComponents
    calibration: TeleopCalibration
    camera_sources: dict[str, RealSenseRGBSource]
    preview_camera_sources: dict[str, RealSenseRGBSource]
    pose_retargeters: dict[str, PoseToArmRetargeter]
    fixed_arm_retargeters: dict[str, FixedArmRetargeter]
    jitter_arm_retargeters: dict[str, SmoothJitterArmRetargeter]
    hand_retargeters: dict[str, JointsToDexHandRetargeter]
    dual_arm_retargeter: DualArmCollisionAwareRetargeter | None
    unmanaged_backends: dict[str, Any]
    writer: LeRobotV21Writer
    sample_builder: CollectorSampleBuilder
    runtime: CollectionRuntime
    control_state: "CollectControlState"
    tactile_visualizer: TactileUVVisualizer | None
    camera_previewer: CameraPreviewer | None


@dataclass
class CollectControlState:
    recording_active: bool = False
    active_episode_index: int | None = None
    active_episode_started_at: float | None = None
    saved_episodes: int = 0
    discarded_episodes: int = 0


@dataclass
class RecoveryPlan:
    mode: str
    started_at: float
    duration_sec: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)


class TerminalKeyController:
    def __init__(self) -> None:
        self.enabled = False
        self.input_source = "disabled"
        self.disabled_reason: str | None = None
        self._fd: int | None = None
        self._owns_fd = False
        self._old_attrs: list[int] | None = None

    def start(self) -> None:
        self.close()
        try:
            if sys.stdin.isatty():
                self._fd = sys.stdin.fileno()
                self._owns_fd = False
                self.input_source = "stdin"
            else:
                self._fd = os.open("/dev/tty", os.O_RDONLY)
                self._owns_fd = True
                self.input_source = "/dev/tty"
            self._old_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self.enabled = True
            self.disabled_reason = None
        except (OSError, termios.error) as exc:
            self.enabled = False
            self.disabled_reason = f"{type(exc).__name__}: {exc}"
            self.input_source = "disabled"
            self._old_attrs = None
            if self._owns_fd and self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
            self._fd = None
            self._owns_fd = False

    def close(self) -> None:
        if self._fd is not None and self._old_attrs is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            except (OSError, termios.error):
                pass
        if self._owns_fd and self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self.enabled = False
        self._fd = None
        self._owns_fd = False
        self._old_attrs = None

    def poll(self) -> list[str]:
        if not self.enabled or self._fd is None:
            return []
        keys: list[str] = []
        while True:
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.0)
            except OSError as exc:
                self.enabled = False
                self.disabled_reason = f"{type(exc).__name__}: {exc}"
                self._old_attrs = None
                return keys
            if not ready:
                return keys
            try:
                chunk = os.read(self._fd, 32)
            except OSError as exc:
                self.enabled = False
                self.disabled_reason = f"{type(exc).__name__}: {exc}"
                return keys
            if not chunk:
                return keys
            text = chunk.decode(errors="ignore")
            for char in text:
                if char == "\x03":
                    raise KeyboardInterrupt
                keys.append(char.lower())


def _load_calibration(args: argparse.Namespace) -> TeleopCalibration:
    if args.load_profile:
        calibration = TeleopCalibration.load_json(args.load_profile)
    else:
        calibration = build_annotated_dual_rm65_calibration()
    calibration.metadata["left_tracker_name"] = str(args.left_tracker_name)
    calibration.metadata["right_tracker_name"] = str(args.right_tracker_name)
    offset = _vive_tracker_local_translation_offset(args)
    calibration.metadata["vive_tracker_height_correction_enabled"] = offset is not None
    calibration.metadata["vive_tracker_height_correction_m"] = float(
        getattr(
            args,
            "vive_tracker_height_correction_m",
            DEFAULT_VIVE_TRACKER_HEIGHT_CORRECTION_M,
        )
    )
    calibration.metadata["vive_tracker_local_translation_offset_m"] = (
        None if offset is None else offset.tolist()
    )
    return calibration


def _vive_tracker_local_translation_offset(args: argparse.Namespace) -> np.ndarray | None:
    if not bool(getattr(args, "vive_tracker_height_correction", True)):
        return None
    height_m = float(
        getattr(
            args,
            "vive_tracker_height_correction_m",
            DEFAULT_VIVE_TRACKER_HEIGHT_CORRECTION_M,
        )
    )
    if height_m < 0.0:
        raise ValueError("--vive-tracker-height-correction-m must be >= 0.")
    return vive_tracker_height_correction_offset(height_m)


def _build_vive_source(args: argparse.Namespace, active_sides: tuple[str, ...]) -> VivePoseSource:
    moving_sides = _moving_arm_sides(args, active_sides)
    non_operator_sides = tuple(side for side in active_sides if side not in moving_sides)
    tracker_configs: list[ViveTrackerConfig] = []
    if "left" in moving_sides:
        tracker_configs.append(
            ViveTrackerConfig(
                name=args.left_tracker_name,
                side="left",
                role="left_wrist",
                channel_name="left_wrist_pose",
            )
        )
    if "left" in non_operator_sides:
        tracker_configs.append(
            ViveTrackerConfig(
                name=args.left_tracker_name,
                side="left",
                role="left_wrist",
                channel_name="left_wrist_pose",
                critical=False,
            )
        )
    if "right" in moving_sides:
        tracker_configs.append(
            ViveTrackerConfig(
                name=args.right_tracker_name,
                side="right",
                role="right_wrist",
                channel_name="right_wrist_pose",
            )
        )
    return VivePoseSource(
        tracker_configs=tracker_configs,
        host=args.host,
        port=int(args.port),
        name="vive_pose_collect",
    )


def _build_camera_sources(args: argparse.Namespace) -> dict[str, RealSenseRGBSource]:
    if args.no_camera:
        return {}
    config_path = Path(args.camera_config)
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
    if key:
        if key not in camera_sources:
            available = ", ".join(sorted(camera_sources))
            raise ValueError(
                f"--vis-cam-key {key!r} not found. Available camera keys: {available}"
            )
        return {key: camera_sources[key]}
    selected_key = "front" if "front" in camera_sources else sorted(camera_sources)[0]
    return {selected_key: camera_sources[selected_key]}


def _build_hand_source(args: argparse.Namespace, active_sides: tuple[str, ...]) -> HandQ6Source | None:
    left_port = args.left_hand_port if "left" in active_sides and args.left_hand_port else None
    right_port = args.right_hand_port if "right" in active_sides and args.right_hand_port else None
    if left_port is None and right_port is None:
        return None
    return HandQ6Source.from_bimanual(
        left_port=left_port,
        right_port=right_port,
        baudrate=int(args.hand_reader_baudrate),
        timeout=float(args.hand_reader_timeout),
        show_raw_lines=bool(args.show_hand_raw_lines),
        exo_hand_calibration=bool(args.exo_hand_calibration),
        calibration_path=str(args.exo_hand_calibration_path or "") or None,
        critical=False,
        strict_expected=False,
    )


def _build_arm_retargeters(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> tuple[
    list[object],
    dict[str, PoseToArmRetargeter],
    dict[str, FixedArmRetargeter],
    dict[str, SmoothJitterArmRetargeter],
    DualArmCollisionAwareRetargeter | None,
]:
    fixed_sides = _fixed_arm_sides(args, active_sides)
    jitter_sides = _jitter_arm_sides(args, active_sides)
    moving_sides = _moving_arm_sides(args, active_sides)
    fixed_retargeters = {
        side: FixedArmRetargeter(
            FixedArmRetargeterConfig(
                side=side,
                effector_name=f"{side}_arm",
            )
        )
        for side in fixed_sides
    }
    jitter_seed = None if int(args.left_arm_jitter_seed) < 0 else int(args.left_arm_jitter_seed)
    jitter_retargeters = {
        side: SmoothJitterArmRetargeter(
            SmoothJitterArmRetargeterConfig(
                side=side,
                effector_name=f"{side}_arm",
                max_translation_m=float(args.left_arm_jitter_position_m),
                max_rotation_deg=float(args.left_arm_jitter_rotation_deg),
                target_refresh_sec=float(args.left_arm_jitter_refresh_sec),
                smoothing_time_constant_sec=float(args.left_arm_jitter_smoothing_sec),
                seed=jitter_seed,
            )
        )
        for side in jitter_sides
    }
    pose_retargeters: dict[str, PoseToArmRetargeter] = {}
    tracker_translation_offset = _vive_tracker_local_translation_offset(args)
    for side in moving_sides:
        pose_retargeters[side] = PoseToArmRetargeter(
            PoseToArmRetargeterConfig(
                side=side,
                input_channel_name=f"{side}_wrist_pose",
                effector_name=f"{side}_arm",
                control_mode=str(args.control_mode),
                translation_scale=float(args.translation_scale),
                input_local_translation_offset=tracker_translation_offset,
                input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
            )
        )

    if (
        not fixed_sides
        and set(active_sides) == {"left", "right"}
        and not bool(args.disable_dual_arm_collision_avoidance)
    ):
        left_pose_retargeter = (
            jitter_retargeters["left"]
            if "left" in jitter_retargeters
            else pose_retargeters["left"]
        )
        dual = DualArmCollisionAwareRetargeter(
            DualArmCollisionAwareRetargeterConfig(
                control_mode=str(args.control_mode),
                translation_scale=float(args.translation_scale),
                scene_profile=str(args.collision_scene_profile),
                contact_mode=str(args.collision_avoidance_mode),
                solver_frequency=float(args.hz),
            ),
            left_pose_retargeter=left_pose_retargeter,
            right_pose_retargeter=pose_retargeters["right"],
        )
        return [dual], pose_retargeters, fixed_retargeters, jitter_retargeters, dual
    return (
        [fixed_retargeters[side] for side in fixed_sides]
        + [pose_retargeters[side] for side in moving_sides],
        pose_retargeters,
        fixed_retargeters,
        jitter_retargeters,
        None,
    )


def _build_hand_retargeters(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> dict[str, JointsToDexHandRetargeter]:
    retargeters: dict[str, JointsToDexHandRetargeter] = {}
    for side in active_sides:
        hand_port = getattr(args, f"{side}_hand_port")
        if not hand_port:
            continue
        retargeters[side] = JointsToDexHandRetargeter(
            JointsToDexHandRetargeterConfig(
                side=side,
                input_channel_name=f"{side}_hand_q6",
                effector_name=f"{side}_hand",
                joint_order=tuple(args.hand_joint_order),
                invert=bool(args.hand_invert),
                alpha=float(args.hand_command_alpha),
            )
        )
    return retargeters


def _build_backends(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    arm_urdf_path = require_asset_path("rm65_arm.urdf")
    control_dt = 1.0 / float(args.hz)
    all_backends: dict[str, Any] = {}
    rig_backends: dict[str, Any] = {}
    tactile_regions = _parse_hand_tactile_regions(args.hand_tactile_regions) if args.with_tactile else ()

    for side in active_sides:
        arm_backend = RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip=str(getattr(args, f"{side}_arm_ip")),
                side=side,
                arm_urdf_path=arm_urdf_path,
                connect_hardware=True,
                control_dt=control_dt,
                max_angvel=tuple(float(value) for value in args.arm_max_angvel),
                dry_run_fast=False,
                background_streaming_enabled=True,
                background_streaming_hz=float(args.hz),
            )
        )
        all_backends[f"{side}_arm"] = arm_backend
        rig_backends[f"{side}_arm"] = arm_backend

        hand_ip = str(getattr(args, f"{side}_hand_ip"))
        hand_port = str(getattr(args, f"{side}_hand_port"))
        if not hand_ip and not hand_port:
            continue
        hand_backend = InspireTacHandBackend(
            InspireTacHandBackendConfig(
                hand_ip=hand_ip or "0.0.0.0",
                side=side,
                connect_hardware=bool(hand_ip),
                max_fingertip_force=int(args.max_fingertip_force),
                send_threshold=float(args.hand_send_threshold),
                enable_telemetry=bool(hand_ip),
                telemetry_hz=float(args.hand_telemetry_hz),
                tactile_regions=tactile_regions,
            )
        )
        all_backends[f"{side}_hand"] = hand_backend
    return all_backends, rig_backends


def _build_tactile_visualizer(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
) -> TactileUVVisualizer | None:
    if not bool(args.visualize_tactile):
        return None
    return TactileUVVisualizer(
        TactileVisualizerConfig(
            enabled_sides=active_sides,
            image_size=(int(args.tactile_vis_size), int(args.tactile_vis_size)),
            value_scale=float(args.tactile_vis_value_scale),
        )
    )


def _source_entrypoint_for_modes(
    *,
    fixed_sides: tuple[str, ...],
    jitter_sides: tuple[str, ...],
) -> str:
    if jitter_sides:
        return "teleop_runtime.LeftArmFixed.run_collect_jitter"
    if fixed_sides:
        return "teleop_runtime.LeftArmFixed.run_collect"
    return "teleop_runtime.collection.run_collect"


def _episode_metadata(
    args: argparse.Namespace,
    active_sides: tuple[str, ...],
    calibration: TeleopCalibration,
    camera_sources: dict[str, RealSenseRGBSource],
    preview_camera_sources: dict[str, RealSenseRGBSource],
) -> dict[str, Any]:
    fixed_sides = _fixed_arm_sides(args, active_sides)
    jitter_sides = _jitter_arm_sides(args, active_sides)
    moving_sides = _moving_arm_sides(args, active_sides)
    return {
        "source_entrypoint": _source_entrypoint_for_modes(
            fixed_sides=fixed_sides,
            jitter_sides=jitter_sides,
        ),
        "task_name": str(args.task_name),
        "task_id": str(args.task_id),
        "task_description": str(args.task_description),
        "dataset_name": str(args.dataset_name),
        "dataset_base_root": str(getattr(args, "dataset_base_root", "")),
        "dataset_dir_name": str(getattr(args, "dataset_dir_name", "")),
        "active_sides": list(active_sides),
        "moving_arm_sides": list(moving_sides),
        "fixed_arm_sides": list(fixed_sides),
        "jitter_arm_sides": list(jitter_sides),
        "left_arm_fixed": bool(fixed_sides),
        "left_arm_jitter": bool(jitter_sides),
        "left_arm_jitter_position_m": float(args.left_arm_jitter_position_m),
        "left_arm_jitter_rotation_deg": float(args.left_arm_jitter_rotation_deg),
        "left_arm_jitter_refresh_sec": float(args.left_arm_jitter_refresh_sec),
        "left_arm_jitter_smoothing_sec": float(args.left_arm_jitter_smoothing_sec),
        "left_arm_jitter_seed": None
        if int(args.left_arm_jitter_seed) < 0
        else int(args.left_arm_jitter_seed),
        "runtime_arm_state_source": "last_commanded_action",
        "control_mode": str(args.control_mode),
        "hz": float(args.hz),
        "nominal_fps": float(args.nominal_fps),
        "left_arm_ip": args.left_arm_ip or None,
        "right_arm_ip": args.right_arm_ip or None,
        "left_hand_ip": args.left_hand_ip or None,
        "right_hand_ip": args.right_hand_ip or None,
        "left_hand_port": args.left_hand_port or None,
        "right_hand_port": args.right_hand_port or None,
        "max_fingertip_force": int(args.max_fingertip_force),
        "hand_tactile_regions": list(_parse_hand_tactile_regions(args.hand_tactile_regions))
        if args.with_tactile
        else [],
        "exo_hand_calibration": bool(args.exo_hand_calibration),
        "exo_hand_calibration_path": args.exo_hand_calibration_path or None,
        "visualize_tactile": bool(args.visualize_tactile),
        "vis_cam": bool(args.vis_cam),
        "vis_camera_keys": sorted(preview_camera_sources),
        "camera_keys": sorted(camera_sources),
        "camera_config_path": None if args.no_camera else str(args.camera_config),
        "calibration_path": args.load_profile or None,
        "calibration_snapshot": calibration.to_dict(),
    }


def _log(tag: str, message: str) -> None:
    print(f"[{tag}] {message}", flush=True)


def _print_controls() -> None:
    print("\n" + "=" * 60)
    print("数据采集循环已启动 / Starting data collection loop...")
    print("控制说明 / Controls:")
    print("  'P' = 标定 home 并开始随动 / Calibrate home and enable following")
    print("  'Z' = 开始录制 / Start episode")
    print("  'C' = 保存当前片段 / Save current episode")
    print("  'D' = 丢弃当前片段 / Discard current episode")
    print("  'H' = 暂停随动并清除 home / Hold and clear home")
    print("  'Q' = 退出程序 / Quit program")
    print("按键焦点 / Key focus: terminal or OpenCV preview window")
    print("=" * 60 + "\n")


def _print_standby(app: CollectApp) -> None:
    home = "ready" if app.calibration.metadata.get("home_ready") else "unset, press P"
    print("Standby, Ready to start data collection...")
    print(f"  home: {home}")
    print(f"  下一个片段编号 Next episode ID: {app.writer.next_episode_index}\n")
    print("=" * 100)


def _compact_log_value(value: object, *, max_chars: int = 90) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _qpos_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        qpos = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
        return None
    return qpos.copy()


def _mean_pose(poses: list[np.ndarray]) -> np.ndarray:
    if not poses:
        raise ValueError("Cannot average an empty pose list.")
    mean_pose = np.eye(4, dtype=float)
    mean_pose[:3, 3] = np.mean([pose[:3, 3] for pose in poses], axis=0)
    rot_sum = np.sum([pose[:3, :3] for pose in poses], axis=0)
    u, _, vt = np.linalg.svd(rot_sum)
    mean_pose[:3, :3] = u @ vt
    if np.linalg.det(mean_pose[:3, :3]) < 0.0:
        u[:, -1] *= -1.0
        mean_pose[:3, :3] = u @ vt
    return mean_pose


def _channel_pose_in_desk(channel: ObservationChannel, calibration: TeleopCalibration) -> np.ndarray:
    pose = as_pose_matrix(channel.value)
    frame_name = channel.frame_name or ""
    metadata = channel.metadata or {}
    tracker_role = metadata.get("tracker_role")

    if frame_name in {"vive_world", "world"}:
        return calibration.world_to_desk(
            pose,
            tracker_role=str(tracker_role) if tracker_role else None,
        )
    if frame_name.endswith("_operator") or frame_name == "operator":
        return calibration.operator_to_desk(pose)
    if frame_name in {calibration.desk_frame.name, "desk"}:
        return pose
    raise ValueError(f"Unsupported calibration input frame: {frame_name or '<none>'}")


def _arm_backend(app: CollectApp, side: str) -> RM65ArmBackend | None:
    backend = app.components.backends.get(f"{side}_arm")
    return backend if isinstance(backend, RM65ArmBackend) else None


def _hand_backend(app: CollectApp, side: str) -> InspireTacHandBackend | None:
    backend = app.components.backends.get(f"{side}_hand")
    return backend if isinstance(backend, InspireTacHandBackend) else None


def _seed_dual_arm_retargeter_from_backends(app: CollectApp) -> None:
    if app.dual_arm_retargeter is None:
        return
    left_backend = _arm_backend(app, "left")
    right_backend = _arm_backend(app, "right")
    if left_backend is None or right_backend is None:
        return
    left_qpos = _qpos_or_none(left_backend.get_state().get("current_qpos"))
    right_qpos = _qpos_or_none(right_backend.get_state().get("current_qpos"))
    if left_qpos is None or right_qpos is None:
        return
    app.dual_arm_retargeter.set_current_qpos_pair(
        left_qpos=left_qpos,
        right_qpos=right_qpos,
    )


def _sync_arm_backends_from_hardware(
    app: CollectApp,
    *,
    required: bool,
) -> dict[str, bool]:
    synced = {side: False for side in app.active_sides}
    for side in app.active_sides:
        backend = _arm_backend(app, side)
        if backend is None:
            continue
        state = backend.refresh_state_from_hardware()
        qpos = _qpos_or_none(state.get("current_qpos"))
        if qpos is None:
            if required:
                raise RuntimeError(
                    f"{side}_arm startup sync failed: no current_qpos available from hardware "
                    f"(code={state.get('hardware_result_code')}, text={state.get('hardware_result_text')!r})."
                )
            continue
        backend.set_local_qpos(qpos)
        synced[side] = True
    _seed_dual_arm_retargeter_from_backends(app)
    return synced


def _set_fixed_arm_state_from_backend(
    app: CollectApp,
    *,
    side: str,
    captured_at: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    backend = _arm_backend(app, side)
    if backend is None:
        raise KeyError(f"Missing arm backend for fixed side '{side}'.")
    qpos = _qpos_or_none(backend.get_state().get("current_qpos"))
    if qpos is None:
        raise RuntimeError(f"Cannot fix {side}_arm without a valid current_qpos.")
    pose_base = backend.compute_tcp_pose()
    arm_calibration = app.calibration.get_arm_for_side(side)
    pose_desk = None if arm_calibration.T_D_B is None else arm_calibration.T_D_B @ pose_base
    app.fixed_arm_retargeters[side].set_fixed_state(
        qpos=qpos,
        target_pose_base=pose_base,
        target_pose_desk=pose_desk,
        captured_at=captured_at,
    )
    return qpos, pose_base, pose_desk


def _set_jitter_arm_anchor_from_backend(
    app: CollectApp,
    *,
    side: str,
    captured_at: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    backend = _arm_backend(app, side)
    if backend is None:
        raise KeyError(f"Missing arm backend for jitter side '{side}'.")
    qpos = _qpos_or_none(backend.get_state().get("current_qpos"))
    if qpos is None:
        raise RuntimeError(f"Cannot set jitter anchor for {side}_arm without a valid current_qpos.")
    pose_base = backend.compute_tcp_pose()
    arm_calibration = app.calibration.get_arm_for_side(side)
    pose_desk = None if arm_calibration.T_D_B is None else arm_calibration.T_D_B @ pose_base
    app.jitter_arm_retargeters[side].set_anchor_state(
        qpos=qpos,
        target_pose_base=pose_base,
        target_pose_desk=pose_desk,
        captured_at=captured_at,
    )
    return qpos, pose_base, pose_desk


def _set_jitter_randomization_enabled(app: CollectApp, enabled: bool) -> None:
    for retargeter in app.jitter_arm_retargeters.values():
        retargeter.set_jitter_enabled(bool(enabled))


def _freeze_jitter_arms_at_current_pose(app: CollectApp) -> None:
    if not app.jitter_arm_retargeters:
        return
    for side, retargeter in app.jitter_arm_retargeters.items():
        backend = _arm_backend(app, side)
        if backend is None:
            retargeter.set_jitter_enabled(False)
            continue
        try:
            state = backend.refresh_state_from_hardware()
        except Exception:
            state = backend.get_state()
        qpos = _qpos_or_none(state.get("current_qpos"))
        if qpos is not None:
            backend.set_local_qpos(qpos)
        try:
            pose_base = backend.compute_tcp_pose()
        except Exception:
            pose_base = None
        pose_desk = None
        if pose_base is not None:
            arm_calibration = app.calibration.get_arm_for_side(side)
            pose_desk = (
                None
                if arm_calibration.T_D_B is None
                else arm_calibration.T_D_B @ pose_base
            )
        retargeter.freeze_at_pose(
            qpos=qpos,
            target_pose_base=pose_base,
            target_pose_desk=pose_desk,
            calibration=app.calibration,
        )
    _seed_dual_arm_retargeter_from_backends(app)


def _build_runtime_fixed_arm_command(
    app: CollectApp,
    observation: TeleopObservation | None,
) -> TeleopCommand | None:
    if observation is None or not app.fixed_arm_retargeters:
        return None
    commands = [
        retargeter.retarget(observation, app.calibration)
        for retargeter in app.fixed_arm_retargeters.values()
    ]
    merged = merge_teleop_commands(tuple(commands))
    if not merged.commands:
        return None
    return merged


def _build_runtime_hand_command_for_recovery(
    app: CollectApp,
    observation: TeleopObservation | None,
) -> TeleopCommand | None:
    if observation is None or not app.hand_retargeters:
        return None
    commands = [
        retargeter.retarget(observation, app.calibration)
        for retargeter in app.hand_retargeters.values()
    ]
    merged = merge_teleop_commands(tuple(commands))
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


def _start_recovery_plan(
    app: CollectApp,
    observation: TeleopObservation,
    *,
    now: float,
    control_mode: str,
    ramp_sec: float,
) -> tuple[RecoveryPlan | None, TeleopCommand | None, bool]:
    _seed_dual_arm_retargeter_from_backends(app)
    moving_sides = tuple(
        side
        for side in app.active_sides
        if side not in app.fixed_arm_retargeters and side not in app.jitter_arm_retargeters
    )

    if control_mode == "delta_clutch":
        for side in moving_sides:
            app.pose_retargeters[side].reanchor_to_current_input(
                observation,
                app.calibration,
                robot_pose_desk=_backend_tcp_pose_desk(app, side=side),
            )
        _seed_dual_arm_retargeter_from_backends(app)
        return None, app.components.session.build_command(observation), True

    if app.dual_arm_retargeter is not None:
        context = app.dual_arm_retargeter.build_target_context(
            observation,
            app.calibration,
        )
        return (
            RecoveryPlan(
                mode="absolute_ramp_dual",
                started_at=now,
                duration_sec=float(ramp_sec),
                payload={
                    "context": context,
                    "start_targets": app.dual_arm_retargeter.current_world_targets(),
                    "end_targets": {
                        "left": context.left_target.copy(),
                        "right": context.right_target.copy(),
                    },
                },
            ),
            None,
            False,
        )

    end_resolutions: dict[str, dict[str, object]] = {}
    for side in moving_sides:
        resolution = app.pose_retargeters[side].resolve_absolute_target_from_observation(
            observation,
            app.calibration,
        )
        if resolution is None:
            raise RuntimeError(
                f"Absolute recovery requires active wrist channel '{side}_wrist_pose' to be valid."
            )
        end_resolutions[side] = resolution
    return (
        RecoveryPlan(
            mode="absolute_ramp_independent",
            started_at=now,
            duration_sec=float(ramp_sec),
            payload={
                "start_targets": {
                    side: _backend_tcp_pose_desk(app, side=side)
                    for side in moving_sides
                },
                "end_resolutions": end_resolutions,
            },
        ),
        None,
        False,
    )


def _advance_recovery_plan(
    app: CollectApp,
    observation: TeleopObservation,
    plan: RecoveryPlan,
    *,
    now: float,
) -> tuple[TeleopCommand, bool]:
    alpha = 1.0
    if plan.duration_sec > 1e-6:
        alpha = float(np.clip((now - plan.started_at) / plan.duration_sec, 0.0, 1.0))

    if plan.mode == "absolute_ramp_dual":
        if app.dual_arm_retargeter is None:
            raise RuntimeError(
                "Dual-arm recovery plan is active but dual retargeter is unavailable."
            )
        _seed_dual_arm_retargeter_from_backends(app)
        context = plan.payload["context"]
        start_targets = plan.payload["start_targets"]
        end_targets = plan.payload["end_targets"]
        command = app.dual_arm_retargeter.build_command_for_context(
            context,
            targets=app.dual_arm_retargeter.interpolate_world_targets(
                start_targets,
                end_targets,
                alpha,
            ),
        )
        command.timestamp = observation.timestamp
        return command, alpha >= 1.0

    if plan.mode == "absolute_ramp_independent":
        start_targets = plan.payload["start_targets"]
        end_resolutions = plan.payload["end_resolutions"]
        commands: list[TeleopCommand] = []
        for side, resolution in end_resolutions.items():
            target_pose = interpolate_pose(
                start_targets[side],
                resolution["target_pose_desk"],
                alpha,
            )
            commands.append(
                app.pose_retargeters[side].build_command_from_target_pose_desk(
                    target_pose_desk=target_pose,
                    observation=observation,
                    calibration=app.calibration,
                    pose_operator=resolution.get("pose_operator"),
                )
            )
        command = merge_teleop_commands(tuple(commands))
        command.timestamp = observation.timestamp
        return command, alpha >= 1.0

    raise ValueError(f"Unsupported recovery plan mode: {plan.mode}")


def _drive_recovery_step(
    app: CollectApp,
    args: argparse.Namespace,
    result: Any,
    recovery_plan: RecoveryPlan | None,
    *,
    now: float,
) -> RecoveryPlan | None:
    if result.state != TeleopRunState.RECOVERING:
        return None

    observation = result.observation
    fixed_command = _build_runtime_fixed_arm_command(app, observation)
    hand_command = _build_runtime_hand_command_for_recovery(app, observation)
    active_command: TeleopCommand | None = None
    recovery_done = False

    if observation is None:
        return recovery_plan

    if recovery_plan is None:
        recovery_plan, active_command, recovery_done = _start_recovery_plan(
            app,
            observation,
            now=now,
            control_mode=str(args.control_mode),
            ramp_sec=float(args.recovery_ramp_sec),
        )
        if recovery_plan is None:
            _log("Recovery", f"direct mode control_mode={args.control_mode}")
        else:
            _log(
                "Recovery",
                f"start mode={recovery_plan.mode} duration={recovery_plan.duration_sec:.2f}s",
            )
    else:
        active_command, recovery_done = _advance_recovery_plan(
            app,
            observation,
            recovery_plan,
            now=now,
        )

    if fixed_command is not None:
        active_command = (
            fixed_command
            if active_command is None
            else merge_teleop_commands((active_command, fixed_command))
        )
    active_command = _merge_runtime_hand_command(active_command, hand_command)
    if active_command is not None:
        if app.components.session.rig is None:
            raise RuntimeError("Recovery requires an active rig.")
        app.components.session.rig.send_commands(active_command)

    if recovery_done:
        app.components.session.finish_recovery("recovery completed")
        _log("Recovery", "completed")
        return None
    return recovery_plan


def _backend_tcp_pose_desk(
    app: CollectApp,
    *,
    side: str,
) -> np.ndarray:
    backend = _arm_backend(app, side)
    if backend is None:
        raise KeyError(f"Missing arm backend for side '{side}'.")
    pose_base = backend.compute_tcp_pose()
    arm_calibration = app.calibration.get_arm_for_side(side)
    if arm_calibration.T_D_B is None:
        raise RuntimeError(f"Arm calibration for side '{side}' is missing T_D_B.")
    return arm_calibration.T_D_B @ pose_base


def _resolve_robot_home_for_side(
    app: CollectApp,
    *,
    side: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    backend = _arm_backend(app, side)
    if backend is None:
        raise KeyError(f"Missing arm backend for side '{side}'.")
    qpos = _qpos_or_none(backend.get_state().get("current_qpos"))
    return _backend_tcp_pose_desk(app, side=side), qpos


def _start_unmanaged_backends(backends: Mapping[str, Any]) -> None:
    for backend in backends.values():
        if hasattr(backend, "start"):
            backend.start()


def _close_unmanaged_backends(backends: Mapping[str, Any]) -> None:
    for name, backend in reversed(list(backends.items())):
        if not hasattr(backend, "close"):
            continue
        try:
            backend.close()
        except Exception as exc:
            _log("CollectClose", f"backend={name} close_failed={exc}")


def _runtime_home_calibration(
    app: CollectApp,
    *,
    timeout_sec: float,
    num_samples: int,
) -> None:
    if not app.active_sides:
        return
    calibration = app.calibration
    fixed_sides = tuple(app.fixed_arm_retargeters)
    jitter_sides = tuple(app.jitter_arm_retargeters)
    moving_sides = tuple(
        side
        for side in app.active_sides
        if side not in app.fixed_arm_retargeters and side not in app.jitter_arm_retargeters
    )
    _log(
        "Calibration",
        "capturing runtime home for "
        f"moving_arm_sides={list(moving_sides)} with {num_samples} valid tracker frames per side; "
        f"fixed_arm_sides={list(fixed_sides)} jitter_arm_sides={list(jitter_sides)}; "
        "keep wrist trackers steady.",
    )
    start_time = time.time()
    samples = {side: [] for side in moving_sides}
    observed_channels: set[str] = set()
    progress_markers = sorted({1, max(1, num_samples // 2), num_samples})
    reported_markers = {side: set() for side in moving_sides}

    while any(len(samples[side]) < num_samples for side in moving_sides):
        now = time.time()
        if now - start_time > timeout_sec:
            counts = {side: len(samples[side]) for side in moving_sides}
            raise TimeoutError(
                "Timed out waiting for runtime home calibration frames. "
                f"sample_counts={counts}. "
                "Verify VIVE sender tracker_role values and receiver tracker-name arguments."
            )
        channels, _ = app.components.session.poll_sources(now)
        for side in moving_sides:
            channel_name = f"{side}_wrist_pose"
            channel = channels.get(channel_name)
            if channel is None:
                continue
            if channel_name not in observed_channels:
                metadata = channel.metadata or {}
                _log(
                    "Calibration",
                    "observed "
                    f"{channel_name} <- {metadata.get('tracker_name')}:{metadata.get('tracker_role')} "
                    f"frame={channel.frame_name}",
                )
                observed_channels.add(channel_name)
            if not channel.valid or len(samples[side]) >= num_samples:
                continue
            try:
                pose_desk = _channel_pose_in_desk(channel, calibration)
            except Exception:
                continue
            samples[side].append(pose_desk)
            for marker in progress_markers:
                if len(samples[side]) >= marker and marker not in reported_markers[side]:
                    reported_markers[side].add(marker)
                    _log("Calibration", f"{side} progress {len(samples[side])}/{num_samples}")
        if any(len(samples[side]) < num_samples for side in moving_sides):
            time.sleep(0.01)

    synced = _sync_arm_backends_from_hardware(app, required=True)
    _log("Calibration", f"locked robot home from hardware synced={synced}")

    capture_summary: dict[str, Any] = {
        "mode": "runtime_collect_auto",
        "arm_state_source": "last_commanded_action",
        "timestamp": time.time(),
        "num_samples": int(num_samples),
        "sides": list(app.active_sides),
        "moving_arm_sides": list(moving_sides),
        "fixed_arm_sides": list(fixed_sides),
        "jitter_arm_sides": list(jitter_sides),
        "sample_counts": {side: len(samples[side]) for side in moving_sides},
    }
    for side in app.active_sides:
        robot_home_desk, qpos = _resolve_robot_home_for_side(app, side=side)
        calibration.capture_home_pose(f"{side}_arm", robot_home_desk)
        if qpos is not None:
            calibration.capture_home_qpos(f"{side}_arm", qpos)
        if side in app.fixed_arm_retargeters:
            fixed_qpos, fixed_pose_base, _ = _set_fixed_arm_state_from_backend(
                app,
                side=side,
                captured_at=float(capture_summary["timestamp"]),
            )
            qpos = fixed_qpos
            fixed_mode = True
            jitter_mode = False
        elif side in app.jitter_arm_retargeters:
            jitter_qpos, fixed_pose_base, _ = _set_jitter_arm_anchor_from_backend(
                app,
                side=side,
                captured_at=float(capture_summary["timestamp"]),
            )
            qpos = jitter_qpos
            fixed_mode = False
            jitter_mode = True
        else:
            human_home = _mean_pose(samples[side])
            app.pose_retargeters[side].set_home_anchors(
                human_pose_desk=human_home,
                robot_pose_desk=robot_home_desk,
            )
            fixed_pose_base = None
            fixed_mode = False
            jitter_mode = False
        capture_summary[side] = {
            "fixed_arm": fixed_mode,
            "jitter_arm": jitter_mode,
            "robot_home_xyz_m": np.round(robot_home_desk[:3, 3], 4).tolist(),
            "home_qpos_rad": None if qpos is None else np.round(qpos, 4).tolist(),
            "fixed_pose_base_xyz_m": None
            if fixed_pose_base is None
            else np.round(fixed_pose_base[:3, 3], 4).tolist(),
        }
        _log(
            "Calibration",
            f"{side} fixed={fixed_mode} jitter={jitter_mode} "
            f"robot_home_xyz={capture_summary[side]['robot_home_xyz_m']} "
            f"qpos={capture_summary[side]['home_qpos_rad']}",
        )
    calibration.metadata["home_capture"] = capture_summary
    calibration.metadata["home_ready"] = True
    _seed_dual_arm_retargeter_from_backends(app)


def _clear_runtime_home(app: CollectApp) -> None:
    app.calibration.clear_home_state()
    for retargeter in app.pose_retargeters.values():
        retargeter.clear_home_anchors()
    for retargeter in app.fixed_arm_retargeters.values():
        retargeter.clear_fixed_state()
    for retargeter in app.jitter_arm_retargeters.values():
        retargeter.clear_anchor_state()


def _run_keyed_home_calibration(app: CollectApp, args: argparse.Namespace) -> None:
    if app.control_state.recording_active:
        _log("Keys", "ignore p: recording is active; press c to save or d to discard first.")
        return
    _log("Keys", "p: calibration requested; holding hardware and syncing arms.")
    app.components.session.hold_hardware("manual calibration requested")
    _clear_runtime_home(app)
    synced = _sync_arm_backends_from_hardware(app, required=True)
    _log("ArmSync", f"pre-calibration hardware sync complete synced={synced}")
    _runtime_home_calibration(
        app,
        timeout_sec=float(args.calibration_timeout),
        num_samples=int(args.calibration_samples),
    )
    app.writer.write_calibration_snapshot("teleop_profile", app.calibration.to_dict())
    app.components.session.set_calibration_ready()
    app.components.session.mark_debug_passed()
    app.components.session.arm_hardware()
    _log("Keys", "p: runtime home captured; arm following is enabled.")
    _print_standby(app)


def _start_recording_episode(app: CollectApp, args: argparse.Namespace) -> None:
    state = app.control_state
    if state.recording_active:
        _log("Episode", f"ignore z: episode {state.active_episode_index} is already recording.")
        return
    if not bool(app.calibration.metadata.get("home_ready")):
        _log("Episode", "ignore z: press p to calibrate home before recording.")
        return
    episode_index = app.writer.start_episode(
        metadata=_episode_metadata(
            args,
            app.active_sides,
            app.calibration,
            app.camera_sources,
            app.preview_camera_sources,
        )
    )
    app.sample_builder.set_episode(episode_index)
    _set_jitter_randomization_enabled(app, True)
    state.recording_active = True
    state.active_episode_index = int(episode_index)
    state.active_episode_started_at = time.time()
    print(f"\n开始录制片段 Start recording episode: {episode_index}", flush=True)


def _reset_recording_state(state: CollectControlState) -> None:
    state.recording_active = False
    state.active_episode_index = None
    state.active_episode_started_at = None


def _save_active_episode(
    app: CollectApp,
    *,
    step_count: int,
    reason: str,
    show_standby: bool = True,
) -> bool:
    state = app.control_state
    if not state.recording_active or state.active_episode_index is None:
        _log("Episode", "ignore c: no active recording.")
        return False
    episode_index = int(state.active_episode_index)
    frame_count = int(app.sample_builder.frame_index)
    print(f"\n录制完成 Recording finished for episode {episode_index}")
    print(f"  数据帧数 Data frames: {frame_count}")
    _freeze_jitter_arms_at_current_pose(app)
    if frame_count <= 0:
        app.runtime.flush()
        app.writer.discard_episode(episode_index)
        state.discarded_episodes += 1
        _reset_recording_state(state)
        print(f"  丢弃片段 Discarding episode {episode_index}: no samples")
        if show_standby:
            _print_standby(app)
        return False
    print(f"  正在保存片段 Saving episode {episode_index}...")
    for job in app.writer.plan_finalize_episode(
        episode_index,
        summary={
            "stop_reason": reason,
            "step_count": int(step_count),
            "submitted_samples_total": int(app.runtime.submitted_samples),
            "episode_frame_count": frame_count,
        },
    ):
        app.runtime.writer_pool.submit(job)
    app.runtime.flush()
    state.saved_episodes += 1
    _reset_recording_state(state)
    print(f"  已保存片段 Episode saved {episode_index}")
    if show_standby:
        _print_standby(app)
    return True


def _discard_active_episode(
    app: CollectApp,
    *,
    reason: str,
    show_standby: bool = True,
) -> bool:
    state = app.control_state
    if not state.recording_active or state.active_episode_index is None:
        _log("Episode", "ignore d: no active recording.")
        return False
    episode_index = int(state.active_episode_index)
    frame_count = int(app.sample_builder.frame_index)
    print(f"\n录制完成 Recording finished for episode {episode_index}")
    print(f"  数据帧数 Data frames: {frame_count}")
    _freeze_jitter_arms_at_current_pose(app)
    app.runtime.flush()
    discarded = app.writer.discard_episode(episode_index)
    state.discarded_episodes += 1
    _reset_recording_state(state)
    print(f"  丢弃片段 Discarding episode {episode_index} discarded={discarded}")
    if show_standby:
        _print_standby(app)
    return True


def _hold_and_clear_home(app: CollectApp) -> None:
    if app.control_state.recording_active:
        _discard_active_episode(app, reason="h/manual_hold", show_standby=False)
    app.components.session.hold_hardware("manual hold; runtime home cleared")
    _clear_runtime_home(app)
    _log("Keys", "h: arm following locked; previous p home calibration cleared.")
    _print_standby(app)


def _handle_key(
    key: str,
    *,
    app: CollectApp,
    args: argparse.Namespace,
    step_count: int,
) -> bool:
    if key in {"", "\n", "\r"}:
        return False
    if key == "p":
        _run_keyed_home_calibration(app, args)
        return False
    if key == "z":
        _start_recording_episode(app, args)
        return False
    if key == "c":
        _save_active_episode(app, step_count=step_count, reason="c/save")
        return False
    if key == "d":
        _discard_active_episode(app, reason="d/discard")
        return False
    if key == "h":
        _hold_and_clear_home(app)
        return False
    if key == "q":
        print("\n正在退出数据采集 Quitting the data collection...\n", flush=True)
        return True
    return False


def build_collect_app(args: argparse.Namespace) -> CollectApp:
    active_sides = tuple(args.active_sides)
    _assert_side_config_consistency(args, active_sides)
    _assert_tactile_visualization_config(args, active_sides)
    fixed_sides = _fixed_arm_sides(args, active_sides)
    jitter_sides = _jitter_arm_sides(args, active_sides)
    moving_sides = _moving_arm_sides(args, active_sides)
    calibration = _load_calibration(args)
    calibration.clear_home_state()
    camera_sources = _build_camera_sources(args)
    preview_camera_sources = (
        _select_preview_camera_sources(camera_sources, str(args.vis_cam_key))
        if bool(args.vis_cam) and camera_sources
        else {}
    )
    vive_source = _build_vive_source(args, active_sides)
    keyboard_source = KeyboardButtonSource(name="collection_clutch")
    default_clutch_pressed = args.control_mode == "delta_clutch"
    for side in ("left", "right"):
        keyboard_source.set_button(
            f"{side}_clutch_button",
            default_clutch_pressed,
            side=side,
            critical=False,
        )

    session_sources: list[Any] = [vive_source, keyboard_source]
    hand_source = _build_hand_source(args, active_sides)
    if hand_source is not None:
        session_sources.append(hand_source)
    session_sources.extend(camera_sources.values())

    (
        arm_retargeters,
        pose_retargeters,
        fixed_arm_retargeters,
        jitter_arm_retargeters,
        dual_arm_retargeter,
    ) = _build_arm_retargeters(
        args,
        active_sides,
    )
    hand_retargeters = _build_hand_retargeters(args, active_sides)
    retargeters = list(arm_retargeters) + list(hand_retargeters.values())

    all_backends, rig_backends = _build_backends(args, active_sides)
    tactile_visualizer = _build_tactile_visualizer(args, active_sides)
    camera_previewer = (
        CameraPreviewer(CameraPreviewConfig())
        if bool(args.vis_cam) and preview_camera_sources
        else None
    )
    rig_backend_names = set(rig_backends)
    for side in hand_retargeters:
        hand_backend = all_backends.get(f"{side}_hand")
        if hand_backend is not None:
            rig_backends[f"{side}_hand"] = hand_backend
            rig_backend_names.add(f"{side}_hand")

    rig = BimanualRig(
        backends=rig_backends,
        parallel_dispatch=len(rig_backends) > 1,
    )
    session = TeleopSession(
        config=TeleopSessionConfig(
            mode="collect_real",
            tcp_mode="hand_base",
            record_enabled=False,
            recovery_healthy_frame_count=max(1, int(args.recovery_healthy_frames)),
        ),
        sources=session_sources,
        retargeters=retargeters,
        rig=rig,
        calibration=calibration,
        safety_config=TeleopSafetyConfig(
            freeze_on_fault=True,
            required_channels=tuple(f"{side}_wrist_pose" for side in moving_sides),
            invalid_frame_limit=max(1, int(args.invalid_frame_limit)),
            minimum_control_hz=float(args.minimum_control_hz),
            target_control_hz=float(args.hz),
        ),
    )
    components = CollectorComponents.from_iterables(
        session=session,
        sources=session_sources,
        retargeters=retargeters,
        backends=all_backends,
        rig=rig,
        metadata={
            "source_entrypoint": _source_entrypoint_for_modes(
                fixed_sides=fixed_sides,
                jitter_sides=jitter_sides,
            ),
            "task_name": str(args.task_name),
            "task_id": str(args.task_id),
            "task_description": str(args.task_description),
            "dataset_name": str(args.dataset_name),
            "active_sides": list(active_sides),
            "moving_arm_sides": list(moving_sides),
            "fixed_arm_sides": list(fixed_sides),
            "jitter_arm_sides": list(jitter_sides),
            "runtime_arm_state_source": "last_commanded_action",
        },
    )

    writer = LeRobotV21Writer(
        LeRobotV21WriterConfig(
            root_dir=args.dataset_root,
            nominal_fps=float(args.nominal_fps),
            task_id=str(args.task_id),
            task_description=str(args.task_description),
            dataset_name=str(args.dataset_name),
            write_parquet=bool(args.write_parquet),
            write_video=bool(args.write_video and not args.no_camera),
        )
    )
    control_state = CollectControlState()
    sample_builder = CollectorSampleBuilder(
        CollectorSampleBuilderConfig(
            active_sides=active_sides,
            camera_buffers={
                camera_key: source.frame_buffer()
                for camera_key, source in camera_sources.items()
            },
            episode_index=writer.next_episode_index,
        )
    )
    writer_pool = AsyncWriterPool(
        writer.handle_job,
        config=AsyncWriterConfig(
            worker_count=int(args.writer_workers),
            max_queue_size=int(args.writer_queue_size),
        ),
        name="collect_writer",
    )
    runtime = CollectionRuntime(
        components=components,
        sample_builder=lambda result, built_components: (
            None
            if (
                not control_state.recording_active
                or result.command is None
                or str(result.state.value).lower() not in {"armed", "running"}
            )
            else sample_builder(result, built_components)
        ),
        write_planner=lambda sample, _components: writer.plan_sample_jobs(sample),
        writer_pool=writer_pool,
        config=CollectionRuntimeConfig(
            auto_start_session=False,
            auto_start_writer_pool=True,
        ),
    )
    return CollectApp(
        active_sides=active_sides,
        components=components,
        calibration=calibration,
        camera_sources=camera_sources,
        preview_camera_sources=preview_camera_sources,
        pose_retargeters=pose_retargeters,
        fixed_arm_retargeters=fixed_arm_retargeters,
        jitter_arm_retargeters=jitter_arm_retargeters,
        hand_retargeters=hand_retargeters,
        dual_arm_retargeter=dual_arm_retargeter,
        unmanaged_backends={
            name: backend
            for name, backend in all_backends.items()
            if name not in rig_backend_names
        },
        writer=writer,
        sample_builder=sample_builder,
        runtime=runtime,
        control_state=control_state,
        tactile_visualizer=tactile_visualizer,
        camera_previewer=camera_previewer,
    )


def _camera_status_text(app: CollectApp) -> str:
    if not app.camera_sources:
        return " camera=off"
    bits: list[str] = []
    if app.camera_previewer is not None and app.camera_previewer.disabled_reason:
        bits.append(
            "preview=disabled:"
            + _compact_log_value(app.camera_previewer.disabled_reason, max_chars=80)
        )
    for camera_key, source in sorted(app.camera_sources.items()):
        try:
            info = source.device_info()
        except Exception as exc:  # noqa: BLE001
            bits.append(f"{camera_key}=status_error:{type(exc).__name__}")
            continue
        frame_count = int(info.get("frame_count", 0) or 0)
        last_error = info.get("last_error")
        if last_error:
            bits.append(
                f"{camera_key}=frames:{frame_count},err:"
                f"{_compact_log_value(last_error, max_chars=70)}"
            )
        else:
            bits.append(f"{camera_key}=frames:{frame_count}")
    return "" if not bits else f" camera={';'.join(bits)}"


def _camera_state_signature_text(app: CollectApp) -> tuple[tuple[str, ...], str]:
    if not app.camera_sources:
        return (), "camera=off"
    signature_parts: list[str] = []
    text_parts: list[str] = []
    if app.camera_previewer is not None:
        disabled_reason = app.camera_previewer.disabled_reason
        if disabled_reason:
            reason_text = _compact_log_value(disabled_reason, max_chars=80)
            signature_parts.append(f"preview:disabled:{reason_text}")
            text_parts.append(f"preview=disabled:{reason_text}")
        else:
            signature_parts.append("preview:active")
            text_parts.append("preview=active")
    for camera_key, source in sorted(app.camera_sources.items()):
        try:
            info = source.device_info()
        except Exception as exc:  # noqa: BLE001
            state = f"status_error:{type(exc).__name__}"
            signature_parts.append(f"{camera_key}:{state}")
            text_parts.append(f"{camera_key}={state}")
            continue
        frame_count = int(info.get("frame_count", 0) or 0)
        last_error = info.get("last_error")
        if last_error:
            error_text = _compact_log_value(last_error, max_chars=70)
            signature_parts.append(f"{camera_key}:error:{error_text}")
            text_parts.append(f"{camera_key}=error:{error_text}")
        elif frame_count > 0:
            signature_parts.append(f"{camera_key}:streaming")
            text_parts.append(f"{camera_key}=streaming")
        else:
            signature_parts.append(f"{camera_key}:waiting")
            text_parts.append(f"{camera_key}=waiting")
    return tuple(signature_parts), "camera=" + ";".join(text_parts)


def _fault_signature(faults: list[Any]) -> tuple[str, ...]:
    parts: list[str] = []
    for fault in faults:
        code = str(getattr(fault, "code", fault))
        channel_name = getattr(fault, "channel_name", None)
        parts.append(f"{code}:{channel_name}" if channel_name else code)
    return tuple(sorted(parts))


def _fault_status_text(faults: list[Any]) -> str:
    bits: list[str] = []
    for fault in faults:
        code = str(getattr(fault, "code", fault))
        channel_name = getattr(fault, "channel_name", None)
        message = str(getattr(fault, "message", "") or "")
        if channel_name:
            code = f"{code}:{channel_name}"
        bits.append(code if not message else f"{code}({_compact_log_value(message, max_chars=70)})")
    return "; ".join(bits)


def _status_line(step_count: int, runtime: CollectionRuntime, result: Any | None, app: CollectApp) -> str:
    state_value = (
        result.state.value
        if result is not None
        else app.components.session.state_machine.state.value
    )
    control = app.control_state
    recording = "on" if control.recording_active else "off"
    home = "ready" if app.calibration.metadata.get("home_ready") else "unset"
    episode = control.active_episode_index
    control_hz = 0.0 if result is None else float(result.control_hz)
    hand_bits: list[str] = []
    if result is not None:
        for side in app.active_sides:
            channel = result.channels.get(f"{side}_hand_q6")
            if channel is None:
                continue
            if channel.valid:
                hand_bits.append(f"{side}=ok")
            else:
                metadata = channel.metadata or {}
                read_error = str(metadata.get("read_error") or "invalid")
                hand_bits.append(f"{side}=invalid:{read_error}")
    hand_status = "" if not hand_bits else f" hand_q6={','.join(hand_bits)}"
    fixed_status = (
        ""
        if not app.fixed_arm_retargeters
        else f" fixed_arm={','.join(sorted(app.fixed_arm_retargeters))}"
    )
    jitter_status = (
        ""
        if not app.jitter_arm_retargeters
        else f" jitter_arm={','.join(sorted(app.jitter_arm_retargeters))}"
    )
    return (
        f"step={step_count} state={state_value} home={home} recording={recording} "
        f"episode={episode} control_hz={control_hz:.2f} submitted={runtime.submitted_samples}"
        f"{fixed_status}{jitter_status}{hand_status}{_camera_status_text(app)}"
    )


def main() -> int:
    args = parse_args()
    app = build_collect_app(args)
    runtime = app.runtime
    key_controller = TerminalKeyController()
    step_count = 0
    last_status_at = 0.0
    last_fault_signature: tuple[str, ...] | None = None
    last_camera_state_signature: tuple[str, ...] | None = None
    loop_period_sec = 1.0 / float(args.hz)
    should_quit = False
    recovery_plan: RecoveryPlan | None = None

    try:
        if app.camera_sources:
            _log(
                "Camera",
                f"configured cameras={sorted(app.camera_sources)} "
                f"config={args.camera_config} vis_cam={bool(args.vis_cam)} "
                f"vis_camera_keys={sorted(app.preview_camera_sources)}",
            )
            if app.camera_previewer is not None:
                if app.camera_previewer.ensure_available():
                    _log(
                        "Camera",
                        "--vis_cam display preflight ok; preview windows will open when the main loop starts.",
                    )
                else:
                    _log(
                        "Camera",
                        "--vis_cam requested but preview is disabled: "
                        f"{app.camera_previewer.disabled_reason}",
                    )
        runtime.start()
        _start_unmanaged_backends(app.unmanaged_backends)
        app.components.session.start()
        key_controller.start()
        synced = _sync_arm_backends_from_hardware(app, required=True)
        _log("ArmSync", f"startup hardware sync complete synced={synced}")
        if app.camera_sources:
            app.writer.write_device_config(
                "realsense",
                {
                    camera_key: source.device_info()
                    for camera_key, source in app.camera_sources.items()
                },
            )
            if app.camera_previewer is not None and app.camera_previewer.disabled_reason is None:
                _log("Camera", f"--vis_cam active for cameras={sorted(app.preview_camera_sources)}")
        elif args.no_camera:
            _log("Camera", "--no-camera active; no RealSense config was loaded and no camera sources were started.")

        if app.fixed_arm_retargeters:
            _log(
                "LeftArmFixed",
                "enabled; left_arm will hold the qpos captured by p, right_wrist_pose drives right_arm.",
            )
        if app.jitter_arm_retargeters:
            _log(
                "LeftArmJitter",
                "enabled; left_arm jitter target is solved together with right_arm by dual-arm IK.",
            )
        print(f"数据集目录 / Dataset directory: {args.dataset_root}")
        print(f"任务名称 / Task name: {args.task_name}")
        print(f"任务 ID / Task ID: {args.task_id}")
        print(f"活动侧 / Active sides: {','.join(app.active_sides)}")
        if app.fixed_arm_retargeters:
            print(f"左臂固定 / Left arm fixed: {','.join(sorted(app.fixed_arm_retargeters))}")
        if app.jitter_arm_retargeters:
            print(f"左臂抖动 / Left arm jitter: {','.join(sorted(app.jitter_arm_retargeters))}")
        _print_controls()
        if not key_controller.enabled:
            _log(
                "Keys",
                "keyboard input is disabled; "
                f"reason={key_controller.disabled_reason or 'no interactive terminal'}",
            )
        else:
            _log("Keys", f"keyboard input active source={key_controller.input_source}")
        _print_standby(app)
        while not should_quit:
            if int(args.max_steps) > 0 and step_count >= int(args.max_steps):
                break
            loop_started_at = time.time()
            for key in key_controller.poll():
                should_quit = _handle_key(
                    key,
                    app=app,
                    args=args,
                    step_count=step_count,
                )
                if should_quit:
                    break
            if should_quit:
                break

            result = None
            if app.calibration.metadata.get("home_ready"):
                result = runtime.step(now=loop_started_at)
                step_count += 1
                recovery_plan = _drive_recovery_step(
                    app,
                    args,
                    result,
                    recovery_plan,
                    now=loop_started_at,
                )
                current_fault_signature = _fault_signature(result.faults)
                if current_fault_signature != last_fault_signature:
                    if current_fault_signature:
                        _log("State", "faults active: " + _fault_status_text(result.faults))
                    elif last_fault_signature is not None:
                        _log("State", "faults cleared")
                    last_fault_signature = current_fault_signature
            if app.tactile_visualizer is not None:
                app.tactile_visualizer.update(
                    tactile_by_side_from_backends(
                        app.active_sides,
                        lambda side: _hand_backend(app, side),
                    )
                )
            if app.camera_previewer is not None:
                for key in app.camera_previewer.update(app.preview_camera_sources):
                    should_quit = _handle_key(
                        key,
                        app=app,
                        args=args,
                        step_count=step_count,
                    )
                    if should_quit:
                        break
            if should_quit:
                break
            if app.camera_sources:
                camera_signature, camera_text = _camera_state_signature_text(app)
                if camera_signature != last_camera_state_signature:
                    _log("Camera", f"state changed: {camera_text}")
                    last_camera_state_signature = camera_signature
            if (
                not bool(args.silent)
                and (loop_started_at - last_status_at) >= float(args.status_sec)
            ):
                print(_status_line(step_count, runtime, result, app))
                last_status_at = loop_started_at
            elapsed = time.time() - loop_started_at
            remaining = loop_period_sec - elapsed
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n键盘中断，停止采集 Keyboard interrupt, stopping collection\n", flush=True)
    finally:
        try:
            if app.control_state.recording_active:
                _save_active_episode(
                    app,
                    step_count=step_count,
                    reason="shutdown/save_active",
                    show_standby=False,
                )
        finally:
            try:
                key_controller.close()
            finally:
                try:
                    if app.tactile_visualizer is not None:
                        app.tactile_visualizer.close()
                finally:
                    try:
                        if app.camera_previewer is not None:
                            app.camera_previewer.close()
                    finally:
                        runtime.close()
                        _close_unmanaged_backends(app.unmanaged_backends)

    print(
        "采集结束 / Collection done: "
        f"saved_episodes={app.control_state.saved_episodes} "
        f"discarded_episodes={app.control_state.discarded_episodes} "
        f"samples={runtime.submitted_samples} root={args.dataset_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
