from __future__ import annotations

import argparse
import json
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
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from project_paths import calibration_profiles_dir, default_dataset_root
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
from ..deploy.actions import ActionSafetyFilter, ActionSafetyFilterConfig, NormalizedAction
from ..deploy.observation import build_deployment_observation
from ..deploy.replay import (
    infer_replay_action_hz,
    load_replay_arm_joint_targets_from_parquet,
    load_replay_actions_from_parquet,
    resolve_replay_parquet_path,
)
from ..deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM, ACTION_HAND_SLICES
from ..deploy.startup_init import StartupInitializer
from ..rigs import BimanualRig
from ..safety import TeleopSafetyConfig
from ..schemas import (
    EffectorCommand,
    ObservationChannel,
    TeleopCommand,
    TeleopObservation,
    merge_teleop_commands,
)
from ..session import SessionStepResult, TeleopSession, TeleopSessionConfig
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
DEFAULT_DATASET_ROOT = default_dataset_root()
DEFAULT_CALIBRATION_PROFILE_DIR = calibration_profiles_dir()
_CALIBRATION_PROFILE_TIMESTAMP_RE = re.compile(r"^(\d{8})_(\d{6})")


def _calibration_profile_timestamp(path: Path) -> float | None:
    match = _CALIBRATION_PROFILE_TIMESTAMP_RE.match(path.name)
    if match is None:
        return None
    try:
        return datetime.strptime("_".join(match.groups()), "%Y%m%d_%H%M%S").timestamp()
    except ValueError:
        return None


def _latest_calibration_profile_path(
    profile_dir: Path = DEFAULT_CALIBRATION_PROFILE_DIR,
) -> Path | None:
    if not profile_dir.exists():
        return None
    candidates = [path for path in profile_dir.glob("*.json") if path.is_file()]
    if not candidates:
        return None

    def sort_key(path: Path) -> tuple[int, float, float, str]:
        name_timestamp = _calibration_profile_timestamp(path)
        try:
            modified_at = path.stat().st_mtime
        except OSError:
            modified_at = 0.0
        if name_timestamp is not None:
            return (1, float(name_timestamp), float(modified_at), path.name)
        return (0, float(modified_at), float(modified_at), path.name)

    return max(candidates, key=sort_key)


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


def _non_negative_float(text: str) -> float:
    value = float(text)
    if value < 0.0:
        raise argparse.ArgumentTypeError("value must be >= 0")
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


def _parse_qpos6(text: str) -> tuple[float, ...]:
    cleaned = str(text).strip().strip("[]()")
    parts = [part for part in re.split(r"[\s,]+", cleaned) if part]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("qpos must contain exactly 6 floats")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid qpos: {text}") from exc
    if not np.all(np.isfinite(np.asarray(values, dtype=float))):
        raise argparse.ArgumentTypeError("qpos values must be finite")
    return values


def _looks_like_single_token_qpos6(text: str) -> bool:
    token = str(text).strip()
    if not token or token.startswith("--"):
        return False
    parts = [part for part in token.split(",") if part.strip()]
    if len(parts) != 6:
        return False
    try:
        values = np.asarray([float(part) for part in parts], dtype=float)
    except ValueError:
        return False
    return bool(np.all(np.isfinite(values)))


def _normalize_qpos_option_argv(argv: Sequence[str]) -> list[str]:
    qpos_flags = {
        "--return-left-qpos",
        "--return-right-qpos",
        "--return-left-qpos-deg",
        "--return-right-qpos-deg",
    }
    normalized: list[str] = []
    index = 0
    tokens = list(argv)
    while index < len(tokens):
        token = str(tokens[index])
        if token in qpos_flags and (index + 1) < len(tokens):
            next_token = str(tokens[index + 1])
            if _looks_like_single_token_qpos6(next_token):
                normalized.append(f"{token}={next_token}")
                index += 2
                continue
        normalized.append(token)
        index += 1
    return normalized


def _qpos6_deg_to_rad(values: Sequence[float]) -> tuple[float, ...]:
    qpos_deg = np.asarray(values, dtype=float).reshape(6)
    return tuple(float(value) for value in np.deg2rad(qpos_deg))


def _qpos6_radians_look_suspicious(values: Sequence[float]) -> bool:
    qpos = np.asarray(values, dtype=float).reshape(6)
    return bool(np.max(np.abs(qpos)) > (2.0 * np.pi + 0.25))


def _parse_hand_tactile_regions(value: object) -> tuple[str, ...]:
    return parse_tactile_regions(value)


def _parse_active_sides(text: str) -> tuple[str, ...]:
    value = str(text).strip().lower()
    if value == "both":
        return ("left", "right")
    if value in {"left", "right"}:
        return (value,)
    raise argparse.ArgumentTypeError("active-sides must be one of: left, right, both")


def _parse_replay_group_key(text: str) -> str:
    key = str(text).strip().lower()
    if len(key) != 1:
        raise ValueError("replay group key must be a single character.")
    if key in {"p", "z", "c", "h", "q", "x", "r"}:
        raise ValueError(
            f"replay group key '{key}' conflicts with an existing control key."
        )
    return key


def _parse_replay_index_token(
    text: str,
    *,
    total_frames: int,
    default: int,
) -> int:
    token = str(text).strip().lower()
    if not token:
        return int(default)
    if token == "start":
        return 0
    if token in {"end", "last"}:
        return int(total_frames)
    if token in {"mid", "half"}:
        return int(total_frames // 2)
    value = int(token)
    if value < 0:
        value = int(total_frames) + value
    return int(value)


def _parse_episode_range_token(text: str, *, available_indices: Sequence[int]) -> tuple[int, ...]:
    token = str(text).strip().lower()
    available = tuple(sorted({int(index) for index in available_indices}))
    if not available:
        raise ValueError("No replay source episodes are available.")
    if not token or token in {"all", "*"}:
        return available
    separators = ("-", "..")
    for separator in separators:
        if separator in token:
            left, right = [part.strip() for part in token.split(separator, 1)]
            if not left or not right:
                raise ValueError(f"Invalid replay episode range: {text!r}")
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Replay episode range end must be >= start: {text!r}")
            selected = tuple(index for index in available if start <= index <= end)
            if not selected:
                raise ValueError(
                    f"Replay episode range {text!r} selected no available episodes; "
                    f"available=[{available[0]}..{available[-1]}]."
                )
            return selected
    selected_values = tuple(
        int(part.strip())
        for part in token.split(",")
        if part.strip()
    )
    if not selected_values:
        raise ValueError(f"Invalid replay episode range: {text!r}")
    available_set = set(available)
    missing = sorted(set(selected_values) - available_set)
    if missing:
        raise ValueError(
            f"Replay episode list {text!r} contains unavailable episode(s): {missing}."
        )
    return tuple(index for index in available if index in set(selected_values))


@dataclass(frozen=True)
class ReplayGroupSpec:
    key: str
    start_index: int
    end_index: int
    episode_range: str = "all"


@dataclass(frozen=True)
class ReplayInitEpisodeSpec:
    episode_index: int
    start_index: int
    end_index: int


def _parse_replay_group_spec(
    spec: str,
    *,
    available_episode_indices: Sequence[int],
) -> ReplayGroupSpec:
    parts = [part.strip() for part in str(spec).split(":")]
    if len(parts) != 2:
        raise ValueError(
            "replay group spec must be 'key:episode_range', "
            "for example 'a:0-49' or 'd:50-99'."
        )
    key = _parse_replay_group_key(parts[0])
    episode_range = parts[1] if parts[1] else "all"
    _parse_episode_range_token(episode_range, available_indices=available_episode_indices)
    return ReplayGroupSpec(
        key=key,
        start_index=0,
        end_index=-1,
        episode_range=str(episode_range),
    )


def _parse_replay_init_episode_spec(
    spec: str,
    *,
    default_total_frames: int,
) -> ReplayInitEpisodeSpec:
    text = str(spec).strip()
    if not text:
        raise ValueError("replay init episode spec is empty.")
    parts = [part.strip() for part in text.split(":")]
    if len(parts) not in {1, 3}:
        raise ValueError(
            "replay init episode spec must be 'episode' or 'episode:start:end', "
            "for example '0' or '3:0:end'."
        )
    episode_index = int(parts[0])
    start = 0
    end = int(default_total_frames)
    if len(parts) == 3:
        start = _parse_replay_index_token(
            parts[1],
            total_frames=default_total_frames,
            default=0,
        )
        end = _parse_replay_index_token(
            parts[2],
            total_frames=default_total_frames,
            default=default_total_frames,
        )
    if start < 0 or start >= default_total_frames:
        raise ValueError(
            f"replay init episode start_index={start} is outside [0, {default_total_frames})."
        )
    if end <= start or end > default_total_frames:
        raise ValueError(
            f"replay init episode end_index={end} must satisfy start < end <= {default_total_frames}."
        )
    return ReplayInitEpisodeSpec(
        episode_index=int(episode_index),
        start_index=int(start),
        end_index=int(end),
    )


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
        default=str(DEFAULT_DATASET_ROOT),
        help=(
            "Base output directory. Actual dataset path is "
            f"<dataset-root>/<YYYY-MM-DD>_<task-name>. Default: {DEFAULT_DATASET_ROOT}."
        ),
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
    parser.add_argument(
        "--load-profile",
        type=str,
        default="",
        help=(
            "Path to a TeleopCalibration JSON profile. If omitted, the newest "
            f"profile in {DEFAULT_CALIBRATION_PROFILE_DIR} is loaded when available."
        ),
    )
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
        default="all",
        help="Comma-separated tactile regions or 'all'. Defaults to all; pass an empty string to disable tactile reads.",
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
        "--return-left-qpos",
        type=_parse_qpos6,
        default=None,
        help="Optional fixed left-arm 6D joint target in radians. Press R to return to it.",
    )
    parser.add_argument(
        "--return-right-qpos",
        type=_parse_qpos6,
        default=None,
        help="Optional fixed right-arm 6D joint target in radians. Press R to return to it.",
    )
    parser.add_argument(
        "--return-left-qpos-deg",
        type=_parse_qpos6,
        default=None,
        help="Optional fixed left-arm 6D joint target in degrees. Converted to radians internally.",
    )
    parser.add_argument(
        "--return-right-qpos-deg",
        type=_parse_qpos6,
        default=None,
        help="Optional fixed right-arm 6D joint target in degrees. Converted to radians internally.",
    )
    parser.add_argument(
        "--return-home-angvel",
        type=_parse_arm_max_angvel,
        default=(0.18, 0.18, 0.18, 0.25, 0.25, 0.25),
        help="Scalar or 6 comma-separated joint velocity limits in rad/s used by R return-home.",
    )
    parser.add_argument(
        "--return-home-tolerance",
        type=_positive_float,
        default=0.01,
        help="Per-joint radian tolerance for completing R return-home.",
    )
    parser.add_argument(
        "--return-home-timeout",
        type=_positive_float,
        default=30.0,
        help="Maximum seconds allowed for an R return-home move.",
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
    parser.add_argument(
        "--replay-dataset",
        type=str,
        default="",
        help=(
            "Enable trajectory replay collection from a source dataset root. "
            "Press replay group keys to start automatic episodes."
        ),
    )
    parser.add_argument(
        "--replay-parquet",
        type=str,
        default="",
        help="Enable trajectory replay collection from a single source parquet file.",
    )
    parser.add_argument(
        "--replay-episode",
        type=int,
        default=0,
        help=(
            "Source episode index used with --replay-dataset when balanced replay is disabled, "
            "and with --replay-parquet metadata."
        ),
    )
    parser.add_argument(
        "--replay-all-episodes",
        dest="replay_all_episodes",
        action="store_true",
        help=(
            "With --replay-dataset, build each replay key from all dataset episodes and "
            "cycle through them evenly. Default."
        ),
    )
    parser.add_argument(
        "--replay-single-episode",
        dest="replay_all_episodes",
        action="store_false",
        help="Use only --replay-episode, preserving the older fixed-source behavior.",
    )
    parser.set_defaults(replay_all_episodes=True)
    parser.add_argument(
        "--replay-action-column",
        type=str,
        default="action",
        help="Source parquet full action column name; falls back to component action columns.",
    )
    parser.add_argument(
        "--replay-group",
        action="append",
        default=[],
        help=(
            "Replay key/source-episode mapping 'key:episode_range'. "
            "Episode ranges support 'all', '0-49', '2,4,8'. "
            "Each selected source episode is replayed from start to end. Defaults to 'a:all'."
        ),
    )
    parser.add_argument(
        "--replay-init-time",
        "--default-init-time",
        dest="replay_init_time",
        type=float,
        default=1.0,
        help="Seconds used to smoothly move to the first replay pose; this phase is recorded.",
    )
    parser.add_argument(
        "--replay-max-arm-xyz-step-m",
        type=float,
        default=0.0,
        help="Optional replay arm Cartesian per-step limiter; 0 disables extra limiting.",
    )
    parser.add_argument(
        "--replay-max-arm-rot-step-deg",
        type=float,
        default=0.0,
        help="Optional replay arm rotation per-step limiter; 0 disables extra limiting.",
    )
    parser.add_argument(
        "--replay-arm-max-angvel",
        type=_parse_arm_max_angvel,
        default=(3.0, 3.0, 3.0, 4.0, 4.0, 4.0),
        help=(
            "Temporary RM65 joint velocity limits in rad/s used while executing replay/init "
            "fixed trajectories. Higher values let hardware follow the source action Hz."
        ),
    )
    parser.add_argument(
        "--replay-randomize",
        action="store_true",
        help="Enable bounded smooth random offsets for replay arm poses.",
    )
    parser.add_argument(
        "--replay-random-seed",
        type=int,
        default=-1,
        help="Seed for replay randomization; negative uses OS entropy.",
    )
    parser.add_argument(
        "--replay-random-mode",
        choices=["episode_offset", "smooth_noise", "both"],
        default="both",
        help="Replay randomization mode.",
    )
    parser.add_argument(
        "--replay-random-xyz-m",
        type=_non_negative_float,
        default=0.01,
        help="Max absolute replay arm position offset in meters.",
    )
    parser.add_argument(
        "--replay-random-rot-deg",
        type=_non_negative_float,
        default=2.0,
        help="Max replay arm rotation offset in degrees.",
    )
    parser.add_argument(
        "--replay-random-noise-xyz-m",
        type=float,
        default=-1.0,
        help=(
            "Max smooth-noise xyz offset in meters; negative uses half of --replay-random-xyz-m."
        ),
    )
    parser.add_argument(
        "--replay-random-noise-rot-deg",
        type=float,
        default=-1.0,
        help=(
            "Max smooth-noise rotation offset in degrees; negative uses half of --replay-random-rot-deg."
        ),
    )
    parser.add_argument(
        "--replay-random-refresh-sec",
        type=_positive_float,
        default=0.8,
        help="Seconds between smooth random replay offset targets.",
    )
    parser.add_argument(
        "--replay-random-smoothing-sec",
        type=_non_negative_float,
        default=0.3,
        help="Seconds used to interpolate toward each smooth random offset target.",
    )
    parser.add_argument(
        "--replay-random-independent-sides",
        action="store_true",
        help="Use independent random offsets per arm side instead of one shared task-space offset.",
    )
    parser.add_argument(
        "--replay-init-episode",
        "--init-episode",
        dest="replay_init_episode",
        type=str,
        default="",
        help=(
            "Optional unrecorded replay pre-initialization trajectory T. "
            "Format 'episode[:start:end]' with start/end/mid tokens, for example '0' or '3:0:end'. "
            "In replay mode, press H once to move safely to T[0], then H again to execute T without recording."
        ),
    )
    parser.add_argument(
        "--replay-init-dataset",
        type=str,
        default="",
        help=(
            "Optional dataset root used only for --replay-init-episode. "
            "Defaults to --replay-dataset when omitted."
        ),
    )
    parser.add_argument(
        "--replay-init-parquet",
        type=str,
        default="",
        help=(
            "Optional single parquet used only for --replay-init-episode. "
            "Use only one of --replay-init-dataset or --replay-init-parquet."
        ),
    )
    parser.add_argument(
        "--replay-init-episode-move-time",
        type=_positive_float,
        default=3.0,
        help=(
            "Legacy first-H move-time value. Current first H uses strict qpos targets from "
            "action.replay.<side>_arm_joint_target instead of end-effector interpolation; not recorded."
        ),
    )
    parser.add_argument(
        "--replay-init-episode-to-record-time",
        type=_non_negative_float,
        default=1.0,
        help=(
            "Seconds used to move from init-episode T end pose to the selected A/D replay start pose; "
            "not recorded in init-episode mode."
        ),
    )
    parser.add_argument(
        "--replay-init-episode-require-ready",
        dest="replay_init_episode_require_ready",
        action="store_true",
        help="Require the H init-episode sequence to complete before replay group keys can start recording. Default.",
    )
    parser.add_argument(
        "--no-replay-init-episode-require-ready",
        dest="replay_init_episode_require_ready",
        action="store_false",
        help="Allow replay group keys to start recording even if the H init-episode sequence has not completed.",
    )
    parser.set_defaults(replay_init_episode_require_ready=True)
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
    args = parser.parse_args(_normalize_qpos_option_argv(sys.argv[1:]))
    for side in ("left", "right"):
        qpos_rad = getattr(args, f"return_{side}_qpos", None)
        qpos_deg = getattr(args, f"return_{side}_qpos_deg", None)
        if qpos_rad is not None and qpos_deg is not None:
            parser.error(
                f"Use only one of --return-{side}-qpos or --return-{side}-qpos-deg."
            )
        if qpos_deg is not None:
            setattr(args, f"return_{side}_qpos", _qpos6_deg_to_rad(qpos_deg))
        if qpos_rad is not None and _qpos6_radians_look_suspicious(qpos_rad):
            parser.error(
                f"--return-{side}-qpos looks like degrees; use --return-{side}-qpos-deg instead."
            )
    if bool(args.no_camera):
        if bool(args.vis_cam):
            parser.error("--vis_cam requires cameras; remove --no-camera.")
        args.write_video = False
    replay_requested = bool(str(args.replay_dataset).strip() or str(args.replay_parquet).strip())
    if replay_requested:
        if str(args.replay_dataset).strip() and str(args.replay_parquet).strip():
            parser.error("Use only one of --replay-dataset or --replay-parquet.")
        if float(args.replay_init_time) < 0.0:
            parser.error("--replay-init-time must be >= 0.")
        if float(args.replay_random_noise_xyz_m) < 0.0 and float(args.replay_random_noise_xyz_m) != -1.0:
            parser.error("--replay-random-noise-xyz-m must be >= 0, or -1 for auto.")
        if float(args.replay_random_noise_rot_deg) < 0.0 and float(args.replay_random_noise_rot_deg) != -1.0:
            parser.error("--replay-random-noise-rot-deg must be >= 0, or -1 for auto.")
    elif str(args.replay_init_episode).strip():
        parser.error("--replay-init-episode requires --replay-dataset or --replay-parquet.")
    if str(args.replay_init_dataset).strip() and str(args.replay_init_parquet).strip():
        parser.error("Use only one of --replay-init-dataset or --replay-init-parquet.")
    if str(args.replay_init_dataset).strip() and not str(args.replay_init_episode).strip():
        parser.error("--replay-init-dataset requires --replay-init-episode.")
    if str(args.replay_init_parquet).strip() and not str(args.replay_init_episode).strip():
        parser.error("--replay-init-parquet requires --replay-init-episode.")
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
    replay_groups: dict[str, "ReplayTrajectoryPool"]
    replay_action_filter: ActionSafetyFilter | None
    replay_init_plan: "ReplayInitEpisodePlan | None"


@dataclass
class CollectControlState:
    recording_active: bool = False
    active_episode_index: int | None = None
    active_episode_started_at: float | None = None
    saved_episodes: int = 0
    discarded_episodes: int = 0
    submitted_samples: int = 0
    active_replay_run: "ActiveReplayRun | None" = None
    active_unrecorded_replay_run: "ActiveUnrecordedReplayRun | None" = None
    replay_init_at_start: bool = False
    replay_init_ready: bool = False
    return_home_plan: "ReturnHomePlan | None" = None


@dataclass
class RecoveryPlan:
    mode: str
    started_at: float
    duration_sec: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReturnHomePlan:
    started_at: float
    target_qpos_by_side: dict[str, np.ndarray]
    original_max_angvel_by_side: dict[str, tuple[float, ...]]
    tolerance: float
    timeout_sec: float


@dataclass(frozen=True)
class ReplayTrajectoryGroup:
    key: str
    label: str
    source_path: str
    source_episode_index: int
    start_index: int
    end_index: int
    frame_count: int
    action_hz: float
    actions: np.ndarray
    arm_joint_targets: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayTrajectoryPool:
    key: str
    label: str
    groups: list[ReplayTrajectoryGroup]
    episode_indices: tuple[int, ...] = ()
    episode_range_spec: str = "all"
    cursor: int = 0
    started_count: int = 0

    def __post_init__(self) -> None:
        self.groups = list(self.groups)
        if not self.groups:
            raise ValueError(f"ReplayTrajectoryPool '{self.key}' requires at least one group.")

    @property
    def size(self) -> int:
        return int(len(self.groups))

    def next_group(self) -> ReplayTrajectoryGroup:
        index = int(self.cursor) % len(self.groups)
        group = self.groups[index]
        self.cursor = (index + 1) % len(self.groups)
        self.started_count += 1
        return group


@dataclass
class ActiveReplayRun:
    group: ReplayTrajectoryGroup
    startup_initializer: StartupInitializer
    episode_index: int
    started_at: float
    randomizer: "ReplayTrajectoryRandomizer | None" = None
    original_max_angvel_by_side: dict[str, tuple[float, ...]] = field(default_factory=dict)
    replay_index: int = 0
    last_action: np.ndarray = field(
        default_factory=lambda: np.full(ACTION_DIM, np.nan, dtype=np.float32)
    )


@dataclass(frozen=True)
class ReplayInitEpisodePlan:
    group: ReplayTrajectoryGroup
    move_time_s: float
    to_record_time_s: float


@dataclass
class ActiveUnrecordedReplayRun:
    group: ReplayTrajectoryGroup
    startup_initializer: StartupInitializer | None
    started_at: float
    purpose: str
    startup_target_qpos_by_side: dict[str, np.ndarray] = field(default_factory=dict)
    pending_pool_key: str = ""
    pending_pool_size: int = 0
    pending_pool_started_count: int = 0
    pending_pool_episode_range: str = ""
    pending_pool_episode_indices: tuple[int, ...] = ()
    pending_randomizer: "ReplayTrajectoryRandomizer | None" = None
    original_max_angvel_by_side: dict[str, tuple[float, ...]] = field(default_factory=dict)
    replay_index: int = 0
    last_action: np.ndarray = field(
        default_factory=lambda: np.full(ACTION_DIM, np.nan, dtype=np.float32)
    )


def _random_unit_rotvec(rng: np.random.Generator, max_angle_rad: float) -> np.ndarray:
    if float(max_angle_rad) <= 0.0:
        return np.zeros(3, dtype=np.float32)
    direction = rng.normal(size=3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        direction = direction / norm
    angle = rng.uniform(-float(max_angle_rad), float(max_angle_rad))
    return (direction * angle).astype(np.float32)


def _random_xyz_offset(rng: np.random.Generator, max_xyz_m: float) -> np.ndarray:
    if float(max_xyz_m) <= 0.0:
        return np.zeros(3, dtype=np.float32)
    return rng.uniform(
        -float(max_xyz_m),
        float(max_xyz_m),
        size=3,
    ).astype(np.float32)


def _smoothstep_unit(alpha: float) -> float:
    x = float(np.clip(alpha, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _replay_active_arm_poses_are_finite(
    action: np.ndarray,
    active_sides: tuple[str, ...],
) -> bool:
    vector = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
    for side in active_sides:
        pose = vector[ACTION_ARM_POSE_SLICES[side]]
        if not np.all(np.isfinite(pose)):
            return False
    return True


@dataclass
class ReplayPoseOffset:
    xyz_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    rotvec_rad: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def copy(self) -> "ReplayPoseOffset":
        return ReplayPoseOffset(
            xyz_m=np.asarray(self.xyz_m, dtype=np.float32).reshape(3).copy(),
            rotvec_rad=np.asarray(self.rotvec_rad, dtype=np.float32).reshape(3).copy(),
        )

    def to_metadata(self) -> dict[str, Any]:
        rotvec = np.asarray(self.rotvec_rad, dtype=np.float32).reshape(3)
        return {
            "xyz_m": np.round(np.asarray(self.xyz_m, dtype=np.float32).reshape(3), 6).tolist(),
            "rotvec_rad": np.round(rotvec, 6).tolist(),
            "rot_angle_deg": float(np.rad2deg(np.linalg.norm(rotvec))),
        }

    @staticmethod
    def interpolate(
        start: "ReplayPoseOffset",
        target: "ReplayPoseOffset",
        alpha: float,
    ) -> "ReplayPoseOffset":
        t = float(alpha)
        return ReplayPoseOffset(
            xyz_m=(
                np.asarray(start.xyz_m, dtype=np.float32)
                + (np.asarray(target.xyz_m, dtype=np.float32) - np.asarray(start.xyz_m, dtype=np.float32)) * t
            ).astype(np.float32),
            rotvec_rad=(
                np.asarray(start.rotvec_rad, dtype=np.float32)
                + (
                    np.asarray(target.rotvec_rad, dtype=np.float32)
                    - np.asarray(start.rotvec_rad, dtype=np.float32)
                )
                * t
            ).astype(np.float32),
        )


class ReplayTrajectoryRandomizer:
    def __init__(
        self,
        *,
        enabled: bool,
        mode: str,
        seed: int | None,
        active_sides: tuple[str, ...],
        max_xyz_m: float,
        max_rot_deg: float,
        noise_xyz_m: float,
        noise_rot_deg: float,
        refresh_sec: float,
        smoothing_sec: float,
        independent_sides: bool,
    ):
        self.enabled = bool(enabled)
        self.mode = str(mode)
        self.seed = seed
        self.active_sides = tuple(side for side in ("left", "right") if side in set(active_sides))
        self.max_xyz_m = max(0.0, float(max_xyz_m))
        self.max_rot_rad = np.deg2rad(max(0.0, float(max_rot_deg)))
        self.noise_xyz_m = max(0.0, float(noise_xyz_m))
        self.noise_rot_rad = np.deg2rad(max(0.0, float(noise_rot_deg)))
        self.refresh_sec = max(1e-6, float(refresh_sec))
        self.smoothing_sec = max(0.0, float(smoothing_sec))
        self.independent_sides = bool(independent_sides)
        self._rng = np.random.default_rng(seed)
        self._episode_offsets = self._sample_offsets(
            max_xyz_m=self.max_xyz_m,
            max_rot_rad=self.max_rot_rad,
        )
        self._current_noise = self._zero_offsets()
        self._start_noise = self._zero_offsets()
        self._target_noise = self._zero_offsets()
        self._target_started_at: float | None = None
        self._next_target_at: float | None = None

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        active_sides: tuple[str, ...],
        episode_index: int,
        pool_started_count: int,
        source_episode_index: int,
    ) -> "ReplayTrajectoryRandomizer | None":
        if not bool(getattr(args, "replay_randomize", False)):
            return None
        base_seed = int(getattr(args, "replay_random_seed", -1))
        seed = None
        if base_seed >= 0:
            seed = (
                int(base_seed)
                + int(episode_index) * 1000003
                + int(source_episode_index) * 9176
                + int(pool_started_count) * 131
            ) % (2**32)
        max_xyz = float(getattr(args, "replay_random_xyz_m", 0.0))
        max_rot = float(getattr(args, "replay_random_rot_deg", 0.0))
        noise_xyz = float(getattr(args, "replay_random_noise_xyz_m", -1.0))
        noise_rot = float(getattr(args, "replay_random_noise_rot_deg", -1.0))
        if noise_xyz < 0.0:
            noise_xyz = max_xyz * 0.5
        if noise_rot < 0.0:
            noise_rot = max_rot * 0.5
        return cls(
            enabled=True,
            mode=str(getattr(args, "replay_random_mode", "both")),
            seed=seed,
            active_sides=active_sides,
            max_xyz_m=max_xyz,
            max_rot_deg=max_rot,
            noise_xyz_m=noise_xyz,
            noise_rot_deg=noise_rot,
            refresh_sec=float(getattr(args, "replay_random_refresh_sec", 0.8)),
            smoothing_sec=float(getattr(args, "replay_random_smoothing_sec", 0.3)),
            independent_sides=bool(getattr(args, "replay_random_independent_sides", False)),
        )

    def _zero_offsets(self) -> dict[str, ReplayPoseOffset]:
        return {side: ReplayPoseOffset() for side in self.active_sides}

    def _sample_one_offset(self, *, max_xyz_m: float, max_rot_rad: float) -> ReplayPoseOffset:
        return ReplayPoseOffset(
            xyz_m=_random_xyz_offset(self._rng, max_xyz_m),
            rotvec_rad=_random_unit_rotvec(self._rng, max_rot_rad),
        )

    def _sample_offsets(self, *, max_xyz_m: float, max_rot_rad: float) -> dict[str, ReplayPoseOffset]:
        if not self.independent_sides:
            shared = self._sample_one_offset(max_xyz_m=max_xyz_m, max_rot_rad=max_rot_rad)
            return {side: shared.copy() for side in self.active_sides}
        return {
            side: self._sample_one_offset(max_xyz_m=max_xyz_m, max_rot_rad=max_rot_rad)
            for side in self.active_sides
        }

    def _episode_offset_for(self, side: str) -> ReplayPoseOffset:
        if self.mode not in {"episode_offset", "both"}:
            return ReplayPoseOffset()
        return self._episode_offsets.get(side, ReplayPoseOffset())

    def _noise_offset_for(self, side: str, now: float | None) -> ReplayPoseOffset:
        if self.mode not in {"smooth_noise", "both"}:
            return ReplayPoseOffset()
        if now is None:
            return self._current_noise.get(side, ReplayPoseOffset())
        self._advance_noise(float(now))
        return self._current_noise.get(side, ReplayPoseOffset())

    def _advance_noise(self, now: float) -> None:
        if self._target_started_at is None:
            self._target_started_at = float(now)
            self._next_target_at = float(now)
        if self._next_target_at is not None and now >= self._next_target_at:
            self._start_noise = {side: offset.copy() for side, offset in self._current_noise.items()}
            self._target_noise = self._sample_offsets(
                max_xyz_m=self.noise_xyz_m,
                max_rot_rad=self.noise_rot_rad,
            )
            self._target_started_at = float(now)
            self._next_target_at = float(now) + self.refresh_sec

        assert self._target_started_at is not None
        if self.smoothing_sec <= 1e-9:
            alpha = 1.0
        else:
            alpha = _smoothstep_unit((float(now) - self._target_started_at) / self.smoothing_sec)
        self._current_noise = {
            side: ReplayPoseOffset.interpolate(
                self._start_noise.get(side, ReplayPoseOffset()),
                self._target_noise.get(side, ReplayPoseOffset()),
                alpha,
            )
            for side in self.active_sides
        }

    @staticmethod
    def _compose_offsets(first: ReplayPoseOffset, second: ReplayPoseOffset) -> ReplayPoseOffset:
        return ReplayPoseOffset(
            xyz_m=(
                np.asarray(first.xyz_m, dtype=np.float32)
                + np.asarray(second.xyz_m, dtype=np.float32)
            ).astype(np.float32),
            rotvec_rad=(
                np.asarray(first.rotvec_rad, dtype=np.float32)
                + np.asarray(second.rotvec_rad, dtype=np.float32)
            ).astype(np.float32),
        )

    def offset_for(self, side: str, now: float | None) -> ReplayPoseOffset:
        return self._compose_offsets(
            self._episode_offset_for(side),
            self._noise_offset_for(side, now),
        )

    def apply(self, action: np.ndarray, *, now: float | None = None) -> np.ndarray:
        vector = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM).copy()
        if not self.enabled:
            return vector
        for side in self.active_sides:
            target_slice = ACTION_ARM_POSE_SLICES[side]
            pose = vector[target_slice].astype(np.float32, copy=True)
            if pose.shape != (7,) or not np.all(np.isfinite(pose)):
                continue
            offset = self.offset_for(side, now)
            pose[:3] = pose[:3] + np.asarray(offset.xyz_m, dtype=np.float32).reshape(3)
            quat = np.asarray(pose[3:7], dtype=float).reshape(4)
            quat_norm = float(np.linalg.norm(quat))
            if quat_norm <= 1e-12:
                continue
            base_rot = R.from_quat(quat / quat_norm)
            delta_rot = R.from_rotvec(np.asarray(offset.rotvec_rad, dtype=float).reshape(3))
            pose[3:7] = (delta_rot * base_rot).as_quat().astype(np.float32)
            vector[target_slice] = pose
        return vector

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "seed": self.seed,
            "active_sides": list(self.active_sides),
            "independent_sides": bool(self.independent_sides),
            "max_xyz_m": float(self.max_xyz_m),
            "max_rot_deg": float(np.rad2deg(self.max_rot_rad)),
            "noise_xyz_m": float(self.noise_xyz_m),
            "noise_rot_deg": float(np.rad2deg(self.noise_rot_rad)),
            "refresh_sec": float(self.refresh_sec),
            "smoothing_sec": float(self.smoothing_sec),
            "episode_offsets": {
                side: offset.to_metadata()
                for side, offset in sorted(self._episode_offsets.items())
            },
        }

    def current_offsets_metadata(self) -> dict[str, Any]:
        return {
            side: self.offset_for(side, None).to_metadata()
            for side in self.active_sides
        }


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
    requested_profile = str(args.load_profile or "").strip()
    profile_path: Path | None = Path(requested_profile) if requested_profile else None
    auto_selected_profile = False
    if profile_path is None:
        profile_path = _latest_calibration_profile_path()
        if profile_path is not None:
            auto_selected_profile = True
            args.load_profile = str(profile_path)

    if profile_path is not None:
        calibration = TeleopCalibration.load_json(profile_path)
        if auto_selected_profile:
            _log("Calibration", f"auto-loaded latest profile {profile_path}")
        else:
            _log("Calibration", f"loaded profile {profile_path}")
    else:
        calibration = build_annotated_dual_rm65_calibration()
        _log(
            "Calibration",
            f"no --load-profile and no JSON profile found in {DEFAULT_CALIBRATION_PROFILE_DIR}; using preset calibration",
        )
    calibration.metadata["calibration_profile_path"] = (
        None if profile_path is None else str(profile_path)
    )
    calibration.metadata["calibration_profile_auto_selected"] = bool(auto_selected_profile)
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
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fixed_sides = _fixed_arm_sides(args, active_sides)
    jitter_sides = _jitter_arm_sides(args, active_sides)
    moving_sides = _moving_arm_sides(args, active_sides)
    metadata = {
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
        "return_home": {
            "left_qpos_rad": None
            if args.return_left_qpos is None
            else [float(value) for value in args.return_left_qpos],
            "right_qpos_rad": None
            if args.return_right_qpos is None
            else [float(value) for value in args.return_right_qpos],
            "left_qpos_deg": None
            if args.return_left_qpos is None
            else np.rad2deg(np.asarray(args.return_left_qpos, dtype=float)).round(4).tolist(),
            "right_qpos_deg": None
            if args.return_right_qpos is None
            else np.rad2deg(np.asarray(args.return_right_qpos, dtype=float)).round(4).tolist(),
            "max_angvel_rad_s": [float(value) for value in args.return_home_angvel],
            "tolerance_rad": float(args.return_home_tolerance),
            "timeout_sec": float(args.return_home_timeout),
            "key": "r",
        },
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
    if extra:
        metadata.update(dict(extra))
    return metadata


def _log(tag: str, message: str) -> None:
    print(f"[{tag}] {message}", flush=True)


def _print_controls() -> None:
    print("\n" + "=" * 60)
    print("数据采集循环已启动 / Starting data collection loop...")
    print("控制说明 / Controls:")
    print("  'P' = 标定 home 并开始随动 / Calibrate home and enable following")
    print("  'R' = 缓慢移动到固定 qpos / Return arm(s) to configured qpos")
    print("  'Z' = 开始录制 / Start episode")
    print("  'C' = 保存当前片段 / Save current episode")
    print("  'D' = 丢弃当前片段 / Discard current episode")
    print("  'H' = 暂停随动并清除 home / Hold and clear home")
    print("  'Q' = 退出程序 / Quit program")
    print("按键焦点 / Key focus: terminal or OpenCV preview window")
    print("=" * 60 + "\n")


def _print_controls_for_app(app: CollectApp) -> None:
    _print_controls()
    if not app.replay_groups:
        return
    print("轨迹回放采集 / Replay trajectory collection:")
    if app.replay_init_plan is not None:
        group = app.replay_init_plan.group
        print(
            "  'H' = init_episode 两阶段预备 / "
            f"first H moves to T start, second H executes T without recording "
            f"(source_episode={group.source_episode_index}, frames [{group.start_index}, {group.end_index}))"
        )
    for key, pool in sorted(app.replay_groups.items()):
        example = pool.groups[0]
        print(
            f"  '{key.upper()}' = 执行轨迹组 {pool.label} / "
            f"Replay group {pool.label}, balanced over {pool.size} source episode(s), "
            f"episode_range={pool.episode_range_spec}, "
            f"example frames [{example.start_index}, {example.end_index})"
        )
    print("  'X' = 中止当前自动轨迹 / Abort active replay or init episode")
    print("  注意 / Note: replay 模式下 'D' 若作为轨迹键会优先启动轨迹组。")
    print("=" * 60 + "\n")


def _print_standby(app: CollectApp) -> None:
    if app.replay_groups:
        if app.replay_init_plan is None:
            home = "not required in replay mode"
        elif app.control_state.replay_init_ready:
            home = "init_episode ready, press replay key"
        elif app.control_state.replay_init_at_start:
            home = "at init_episode start, press H to execute T"
        else:
            home = "press H to move to init_episode start"
    else:
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


def _collect_arm_states_for_deployment(app: CollectApp) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        backend = _arm_backend(app, side)
        if backend is None:
            states[side] = {}
            continue
        try:
            state = dict(backend.refresh_state_from_hardware())
        except Exception:
            state = dict(backend.get_state())
        try:
            state["current_tcp_pose"] = backend.compute_tcp_pose()
        except Exception:
            state["current_tcp_pose"] = None
        states[side] = state
    return states


def _collect_hand_states_for_deployment(app: CollectApp) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        backend = _hand_backend(app, side)
        if backend is None:
            states[side] = {}
            continue
        try:
            states[side] = dict(backend.get_state())
        except Exception:
            states[side] = {}
    return states


def _deployment_command_to_teleop_command(
    commands: Mapping[str, EffectorCommand],
    *,
    timestamp: float,
    metadata: Mapping[str, Any] | None = None,
) -> TeleopCommand:
    command = TeleopCommand(
        timestamp=float(timestamp),
        metadata={} if metadata is None else dict(metadata),
    )
    for effector_command in commands.values():
        command.add_command(effector_command.copy())
    return command


def _submit_replay_sample(
    app: CollectApp,
    *,
    timestamp: float,
    command: TeleopCommand,
    control_hz: float,
    metadata: Mapping[str, Any],
) -> None:
    channels, faults = app.components.session.poll_sources(float(timestamp))
    result = SessionStepResult(
        timestamp=float(timestamp),
        state=TeleopRunState.RUNNING,
        control_hz=float(control_hz),
        channels=channels,
        observation=TeleopObservation(
            channels={name: channel.copy() for name, channel in channels.items()},
            timestamp=float(timestamp),
            metadata={"mode": "trajectory_replay"},
        ),
        command=command,
        faults=faults,
    )
    sample = app.sample_builder(result, app.components)
    if sample is None:
        return
    sample.metadata.update(dict(metadata))
    for job in app.writer.plan_sample_jobs(sample):
        app.runtime.writer_pool.submit(job)
    app.control_state.submitted_samples += 1


def _build_replay_deployment_observation(
    app: CollectApp,
    *,
    now: float,
    step_count: int,
    last_action: np.ndarray,
    collection_mode: str,
    replay_group: str,
    replay_index: int,
) -> Any:
    arm_states = _collect_arm_states_for_deployment(app)
    hand_states = _collect_hand_states_for_deployment(app)
    return build_deployment_observation(
        active_sides=app.active_sides,
        arms_enabled=True,
        hands_enabled=any(f"{side}_hand" in app.components.backends for side in app.active_sides),
        step_index=step_count,
        arm_states=arm_states,
        hand_states=hand_states,
        camera_sources=app.camera_sources,
        last_action=last_action,
        timestamp=float(now),
        require_images=False,
        metadata={
            "collection_mode": collection_mode,
            "replay_group": replay_group,
            "replay_index": int(replay_index),
        },
    )


def _build_joint_target_teleop_command(
    qpos_by_side: Mapping[str, np.ndarray],
    *,
    hand_action: np.ndarray | None = None,
    hand_sides: Sequence[str] = (),
    now: float,
) -> TeleopCommand:
    command = TeleopCommand(
        timestamp=float(now),
        metadata={"mode": "trajectory_replay_qpos_init"},
    )
    for side, qpos in qpos_by_side.items():
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=np.asarray(qpos, dtype=float).reshape(6),
                frame_name=None,
                metadata={"source": "replay_qpos_init"},
            )
        )
    if hand_action is not None:
        hand_vector = np.asarray(hand_action, dtype=np.float32).reshape(ACTION_DIM)
        for side in hand_sides:
            target = hand_vector[ACTION_HAND_SLICES[side]]
            if not np.all(np.isfinite(target)):
                continue
            command.add_command(
                EffectorCommand(
                    effector_name=f"{side}_hand",
                    side=side,
                    command_type="joint_vector",
                    value=np.clip(target, 0.0, 1.0),
                    frame_name=None,
                    metadata={"source": "replay_qpos_init"},
                )
            )
    return command


def _qpos_targets_reached(
    app: CollectApp,
    qpos_by_side: Mapping[str, np.ndarray],
    *,
    tolerance: float = 0.01,
) -> bool:
    for side, target in qpos_by_side.items():
        backend = _arm_backend(app, str(side))
        if backend is None:
            return False
        current = _refresh_return_home_local_state(app, str(side))
        if current is None:
            return False
        if float(np.max(np.abs(np.asarray(target, dtype=float).reshape(6) - current))) > float(tolerance):
            return False
    return True


def _hand_target_summary(action: np.ndarray, active_sides: Sequence[str]) -> str:
    bits: list[str] = []
    vector = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
    for side in active_sides:
        target = vector[ACTION_HAND_SLICES[side]]
        if not np.all(np.isfinite(target)):
            bits.append(f"{side}=invalid")
            continue
        bits.append(
            f"{side}=min:{float(np.min(target)):.3f},max:{float(np.max(target)):.3f}"
        )
    return " ".join(bits) if bits else "hands=none"


def _read_replay_dataset_episode_indices(dataset_root: str | Path) -> list[int]:
    root = Path(dataset_root).expanduser().resolve()
    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(
            f"Balanced replay requires dataset metadata: {episodes_path}"
        )
    indices: list[int] = []
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        try:
            indices.append(int(row["episode_index"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid replay episode row in {episodes_path}: {row!r}") from exc
    if not indices:
        raise ValueError(f"Replay dataset has no episodes in {episodes_path}.")
    return sorted(set(indices))


def _resolve_replay_episode_indices(args: argparse.Namespace) -> list[int]:
    dataset = str(args.replay_dataset).strip()
    if not dataset:
        return [int(args.replay_episode)]
    if not bool(getattr(args, "replay_all_episodes", True)):
        return [int(args.replay_episode)]
    return _read_replay_dataset_episode_indices(dataset)


def _build_replay_group_for_episode(
    args: argparse.Namespace,
    *,
    episode_index: int,
    group_spec: ReplayGroupSpec,
    balanced_episode_count: int,
    require_arm_joint_targets: bool = False,
) -> ReplayTrajectoryGroup:
    source_path, resolved_metadata = resolve_replay_parquet_path(
        dataset_root=str(args.replay_dataset).strip() or None,
        parquet_path=str(args.replay_parquet).strip() or None,
        episode_index=int(episode_index),
    )
    all_actions = load_replay_actions_from_parquet(
        source_path,
        action_column=str(args.replay_action_column or "action"),
        start_index=0,
        end_index=None,
    )
    total_frames = int(all_actions.shape[0])
    start_index = int(group_spec.start_index)
    end_index = int(group_spec.end_index)
    if end_index < 0:
        end_index = total_frames
    if start_index < 0 or start_index >= total_frames:
        raise ValueError(
            f"replay group '{group_spec.key}' start_index={start_index} "
            f"is outside source episode {episode_index} frame range [0, {total_frames})."
        )
    if end_index <= start_index or end_index > total_frames:
        raise ValueError(
            f"replay group '{group_spec.key}' end_index={end_index} must satisfy "
            f"start < end <= {total_frames} for source episode {episode_index}."
        )
    action_hz_info = infer_replay_action_hz(
        parquet_path=source_path,
        dataset_root=str(args.replay_dataset).strip() or None,
        episode_index=int(episode_index),
        start_index=start_index,
        end_index=end_index,
        resolved_metadata=resolved_metadata,
    )
    actions = all_actions[start_index:end_index].astype(np.float32, copy=True)
    arm_joint_targets = load_replay_arm_joint_targets_from_parquet(
        source_path,
        active_sides=args.active_sides,
        start_index=start_index,
        end_index=end_index,
        required=bool(require_arm_joint_targets),
    )
    source_episode_index = int(resolved_metadata.get("episode_index", int(episode_index)))
    return ReplayTrajectoryGroup(
        key=group_spec.key,
        label=f"episodes_{group_spec.episode_range}",
        source_path=str(source_path),
        source_episode_index=source_episode_index,
        start_index=int(start_index),
        end_index=int(end_index),
        frame_count=int(actions.shape[0]),
        action_hz=float(action_hz_info["hz"]),
        actions=actions,
        arm_joint_targets=arm_joint_targets,
        metadata={
            "action_hz_estimate": dict(action_hz_info),
            "source_metadata": dict(resolved_metadata),
            "action_column": str(args.replay_action_column or "action"),
            "balanced_replay": bool(str(args.replay_dataset).strip())
            and bool(getattr(args, "replay_all_episodes", True)),
            "balanced_episode_count": int(balanced_episode_count),
            "source_episode_index": source_episode_index,
            "episode_range_spec": str(group_spec.episode_range),
        },
    )


def _build_replay_groups(args: argparse.Namespace) -> dict[str, ReplayTrajectoryPool]:
    if not (str(args.replay_dataset).strip() or str(args.replay_parquet).strip()):
        return {}
    group_specs = list(args.replay_group or [])
    if not group_specs:
        group_specs = ["a:all"]
    available_episode_indices = _resolve_replay_episode_indices(args)
    pools: dict[str, ReplayTrajectoryPool] = {}
    for spec in group_specs:
        parsed_spec = _parse_replay_group_spec(
            spec,
            available_episode_indices=available_episode_indices,
        )
        if str(args.replay_parquet).strip() and parsed_spec.episode_range not in {"all", "*"}:
            raise ValueError(
                "With --replay-parquet, --replay-group only supports 'key:all' because "
                "there is only one source parquet. Use --replay-dataset to select episode index ranges."
            )
        selected_episode_indices = (
            tuple(available_episode_indices)
            if str(args.replay_parquet).strip()
            else _parse_episode_range_token(
                parsed_spec.episode_range,
                available_indices=available_episode_indices,
            )
        )
        candidates = [
            _build_replay_group_for_episode(
                args,
                episode_index=episode_index,
                group_spec=parsed_spec,
                balanced_episode_count=len(selected_episode_indices),
            )
            for episode_index in selected_episode_indices
        ]
        key = candidates[0].key
        label = candidates[0].label
        if key in pools:
            raise ValueError(f"Duplicate replay group key '{key}'.")
        if any(candidate.key != key for candidate in candidates):
            raise ValueError(f"Replay group spec {spec!r} produced inconsistent keys.")
        pools[key] = ReplayTrajectoryPool(
            key=key,
            label=label,
            groups=candidates,
            episode_indices=tuple(int(index) for index in selected_episode_indices),
            episode_range_spec=str(parsed_spec.episode_range),
        )
    return pools


def _build_replay_init_plan(args: argparse.Namespace) -> ReplayInitEpisodePlan | None:
    spec_text = str(getattr(args, "replay_init_episode", "") or "").strip()
    if not spec_text:
        return None
    init_dataset_root = str(getattr(args, "replay_init_dataset", "") or "").strip()
    init_parquet_path = str(getattr(args, "replay_init_parquet", "") or "").strip()
    source_dataset_root = init_dataset_root or str(args.replay_dataset).strip()
    source_parquet_path = init_parquet_path or str(args.replay_parquet).strip()
    episode_token = spec_text.split(":", 1)[0].strip()
    if not episode_token:
        raise ValueError("replay init episode spec must start with an episode index.")
    episode_index = int(episode_token)
    init_episode_path, _ = resolve_replay_parquet_path(
        dataset_root=source_dataset_root or None,
        parquet_path=source_parquet_path or None,
        episode_index=episode_index,
    )
    total_frames = int(
        load_replay_actions_from_parquet(
            init_episode_path,
            action_column=str(args.replay_action_column or "action"),
            start_index=0,
            end_index=None,
        ).shape[0]
    )
    parsed = _parse_replay_init_episode_spec(
        spec_text,
        default_total_frames=total_frames,
    )
    group_spec = ReplayGroupSpec(
        key="h",
        start_index=int(parsed.start_index),
        end_index=int(parsed.end_index),
        episode_range=str(parsed.episode_index),
    )
    group_args = argparse.Namespace(**vars(args))
    group_args.replay_dataset = source_dataset_root
    group_args.replay_parquet = source_parquet_path
    group = _build_replay_group_for_episode(
        group_args,
        episode_index=int(parsed.episode_index),
        group_spec=group_spec,
        balanced_episode_count=1,
        require_arm_joint_targets=True,
    )
    return ReplayInitEpisodePlan(
        group=group,
        move_time_s=float(args.replay_init_episode_move_time),
        to_record_time_s=float(args.replay_init_episode_to_record_time),
    )


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


def _build_runtime_hand_command_from_sources(
    app: CollectApp,
    *,
    now: float,
) -> TeleopCommand | None:
    if not app.hand_retargeters:
        return None
    channels, _ = app.components.session.poll_sources(float(now))
    observation = TeleopObservation(
        channels={name: channel.copy() for name, channel in channels.items()},
        timestamp=float(now),
        metadata={"mode": "return_home"},
    )
    return _build_runtime_hand_command_for_recovery(app, observation)


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


def _configured_return_home_targets(
    app: CollectApp,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {}
    for side in app.active_sides:
        raw_qpos = getattr(args, f"return_{side}_qpos", None)
        if raw_qpos is None:
            continue
        targets[side] = np.asarray(raw_qpos, dtype=float).reshape(6)
    return targets


def _return_home_targets_summary(targets: Mapping[str, np.ndarray]) -> str:
    parts = []
    for side, qpos in sorted(targets.items()):
        parts.append(f"{side}={np.round(qpos, 4).tolist()}")
    return " ".join(parts)


def _refresh_return_home_local_state(app: CollectApp, side: str) -> np.ndarray | None:
    backend = _arm_backend(app, side)
    if backend is None:
        return None
    try:
        state = backend.refresh_state_from_hardware()
    except Exception:
        state = backend.get_state()
    qpos = _qpos_or_none(state.get("current_qpos"))
    if qpos is not None:
        backend.set_local_qpos(qpos)
    return qpos


def _set_backend_max_angvel(backend: RM65ArmBackend, max_angvel: tuple[float, ...]) -> None:
    if len(max_angvel) != 6 or any(float(value) <= 0.0 for value in max_angvel):
        raise ValueError("max_angvel must contain 6 positive values.")
    backend.config.max_angvel = tuple(float(value) for value in max_angvel)


def _begin_replay_arm_speed_override(
    app: CollectApp,
    max_angvel: Sequence[float],
) -> dict[str, tuple[float, ...]]:
    replay_max_angvel = tuple(float(value) for value in max_angvel)
    original: dict[str, tuple[float, ...]] = {}
    for side in app.active_sides:
        backend = _arm_backend(app, side)
        if backend is None:
            continue
        original[side] = tuple(float(value) for value in backend.config.max_angvel)
        _set_backend_max_angvel(backend, replay_max_angvel)
    return original


def _restore_replay_arm_speed_override(
    app: CollectApp,
    original: Mapping[str, Sequence[float]],
) -> None:
    for side, max_angvel in original.items():
        backend = _arm_backend(app, str(side))
        if backend is None:
            continue
        _set_backend_max_angvel(backend, tuple(float(value) for value in max_angvel))


def _build_return_home_command(
    target_qpos_by_side: Mapping[str, np.ndarray],
    *,
    now: float,
) -> TeleopCommand:
    command = TeleopCommand(
        timestamp=float(now),
        metadata={"mode": "return_home", "key": "r"},
    )
    for side, qpos in target_qpos_by_side.items():
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=np.asarray(qpos, dtype=float).reshape(6),
                frame_name="joint",
                metadata={"source": "return_home_key"},
            )
        )
    return command


def _sync_teleop_home_after_return(app: CollectApp, *, now: float) -> None:
    if not bool(app.calibration.metadata.get("home_ready")):
        return
    channels, _ = app.components.session.poll_sources(float(now))
    observation = TeleopObservation(
        channels={name: channel.copy() for name, channel in channels.items()},
        timestamp=float(now),
        metadata={"mode": "return_home_resync"},
    )
    for side, retargeter in app.pose_retargeters.items():
        if side not in app.active_sides:
            continue
        try:
            robot_pose_desk = _backend_tcp_pose_desk(app, side=side)
            control_mode = str(getattr(retargeter.config, "control_mode", ""))
            if control_mode == "absolute":
                channel = observation.get_channel(retargeter.config.input_channel_name)
                if channel is None or not channel.valid:
                    raise RuntimeError(
                        f"missing valid '{retargeter.config.input_channel_name}' channel"
                    )
                retargeter.set_home_anchors(
                    human_pose_desk=_channel_pose_in_desk(channel, app.calibration),
                    robot_pose_desk=robot_pose_desk,
                )
            else:
                retargeter.reanchor_to_current_input(
                    observation,
                    app.calibration,
                    robot_pose_desk=robot_pose_desk,
                )
        except Exception as exc:
            _log("ReturnHome", f"{side} teleop resync skipped: {exc}")
    for side in app.fixed_arm_retargeters:
        try:
            _set_fixed_arm_state_from_backend(app, side=side, captured_at=float(now))
        except Exception as exc:
            _log("ReturnHome", f"{side} fixed-arm resync skipped: {exc}")
    for side in app.jitter_arm_retargeters:
        try:
            _set_jitter_arm_anchor_from_backend(app, side=side, captured_at=float(now))
        except Exception as exc:
            _log("ReturnHome", f"{side} jitter anchor resync skipped: {exc}")
    _seed_dual_arm_retargeter_from_backends(app)


def _start_return_home(app: CollectApp, args: argparse.Namespace) -> None:
    state = app.control_state
    if state.active_replay_run is not None:
        _log("ReturnHome", "ignore r: trajectory replay episode is active; press x to abort first.")
        return
    if state.return_home_plan is not None:
        _log("ReturnHome", "ignore r: return-home move is already active.")
        return
    targets = _configured_return_home_targets(app, args)
    if not targets:
        _log(
            "ReturnHome",
            "ignore r: no target configured; pass --return-left-qpos[-deg] and/or --return-right-qpos[-deg].",
        )
        return
    if app.components.session.rig is None:
        raise RuntimeError("Return-home requires an active rig.")
    if state.recording_active:
        _discard_active_episode(app, reason="r/return_home", show_standby=False)
    original_max_angvel_by_side: dict[str, tuple[float, ...]] = {}
    return_max_angvel = tuple(float(value) for value in args.return_home_angvel)
    for side in targets:
        backend = _arm_backend(app, side)
        if backend is None:
            raise KeyError(f"Missing arm backend for return-home side '{side}'.")
        original_max_angvel_by_side[side] = tuple(float(value) for value in backend.config.max_angvel)
        _set_backend_max_angvel(backend, return_max_angvel)
        _refresh_return_home_local_state(app, side)
    app.components.session.hold_hardware("manual return-home")
    _set_jitter_randomization_enabled(app, False)
    state.return_home_plan = ReturnHomePlan(
        started_at=time.time(),
        target_qpos_by_side={side: qpos.copy() for side, qpos in targets.items()},
        original_max_angvel_by_side=original_max_angvel_by_side,
        tolerance=float(args.return_home_tolerance),
        timeout_sec=float(args.return_home_timeout),
    )
    _log(
        "ReturnHome",
        "r: moving to configured qpos "
        f"targets={_return_home_targets_summary(targets)} "
        f"max_angvel={np.round(np.asarray(return_max_angvel, dtype=float), 4).tolist()}",
    )


def _advance_return_home(
    app: CollectApp,
    *,
    now: float,
) -> bool:
    plan = app.control_state.return_home_plan
    if plan is None:
        return False
    if app.components.session.rig is None:
        raise RuntimeError("Return-home requires an active rig.")
    command = _build_return_home_command(plan.target_qpos_by_side, now=now)
    hand_command = _build_runtime_hand_command_from_sources(app, now=now)
    if hand_command is not None:
        command = merge_teleop_commands((command, hand_command))
    app.components.session.rig.send_commands(command)

    errors: dict[str, float] = {}
    for side, target in plan.target_qpos_by_side.items():
        backend = _arm_backend(app, side)
        if backend is None:
            errors[side] = float("inf")
            continue
        current = _qpos_or_none(backend.get_state().get("current_qpos"))
        if current is None:
            errors[side] = float("inf")
            continue
        errors[side] = float(np.max(np.abs(target - current)))

    done = bool(errors) and all(error <= plan.tolerance for error in errors.values())
    timed_out = (now - plan.started_at) > plan.timeout_sec
    if not done and not timed_out:
        return True

    for side, original_max_angvel in plan.original_max_angvel_by_side.items():
        backend = _arm_backend(app, side)
        if backend is not None:
            _set_backend_max_angvel(backend, original_max_angvel)
    for side, target in plan.target_qpos_by_side.items():
        backend = _arm_backend(app, side)
        if backend is not None:
            latest_qpos = _refresh_return_home_local_state(app, side)
            backend.set_local_qpos(target if latest_qpos is None else latest_qpos)
    _sync_teleop_home_after_return(app, now=now)
    app.components.session.set_calibration_ready()
    app.components.session.mark_debug_passed()
    app.components.session.arm_hardware()
    app.control_state.return_home_plan = None
    error_summary = {side: round(error, 5) for side, error in errors.items()}
    if timed_out and not done:
        _log(
            "ReturnHome",
            f"timed out after {plan.timeout_sec:.1f}s; teleop re-armed at latest target, errors={error_summary}",
        )
    else:
        _log("ReturnHome", f"completed; teleop re-armed, errors={error_summary}")
    _print_standby(app)
    return False


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
    if state.active_replay_run is not None:
        _log("Episode", "ignore z: trajectory replay episode is active.")
        return
    if not app.replay_groups and not bool(app.calibration.metadata.get("home_ready")):
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
    state.active_replay_run = None


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
    active_run = state.active_replay_run
    if active_run is not None:
        _restore_replay_arm_speed_override(app, active_run.original_max_angvel_by_side)
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
    active_run = state.active_replay_run
    if active_run is not None:
        _restore_replay_arm_speed_override(app, active_run.original_max_angvel_by_side)
    app.runtime.flush()
    discarded = app.writer.discard_episode(episode_index)
    state.discarded_episodes += 1
    _reset_recording_state(state)
    print(f"  丢弃片段 Discarding episode {episode_index} discarded={discarded}")
    if show_standby:
        _print_standby(app)
    return True


def _start_replay_episode(
    app: CollectApp,
    args: argparse.Namespace,
    *,
    key: str,
) -> None:
    state = app.control_state
    pool = app.replay_groups.get(str(key).lower())
    if pool is None:
        return
    if state.recording_active or state.active_replay_run is not None or state.active_unrecorded_replay_run is not None:
        _log(
            "Replay",
            f"ignore {key}: episode {state.active_episode_index} is already active.",
        )
        return
    if app.components.session.rig is None:
        raise RuntimeError("Trajectory replay requires an active rig.")
    if app.replay_action_filter is None:
        raise RuntimeError("Trajectory replay action filter is unavailable.")
    if app.replay_init_plan is not None and bool(getattr(args, "replay_init_episode_require_ready", True)):
        if not state.replay_init_ready:
            _log(
                "Replay",
                f"ignore {key}: press H to complete the unrecorded init trajectory before recording replay episodes.",
            )
            return
    group = pool.next_group()
    first_action = group.actions[0].astype(np.float32, copy=True)
    episode_index = app.writer.next_episode_index
    randomizer = ReplayTrajectoryRandomizer.from_args(
        args,
        active_sides=app.active_sides,
        episode_index=episode_index,
        pool_started_count=pool.started_count,
        source_episode_index=group.source_episode_index,
    )
    if randomizer is not None:
        first_action = randomizer.apply(first_action, now=None)

    if app.replay_init_plan is not None:
        _start_unrecorded_replay_run(
            app,
            args,
            group=group,
            purpose="move_to_record_start",
            startup_duration_s=float(app.replay_init_plan.to_record_time_s),
            startup_source="replay:init_episode_to_record",
            target_action=first_action,
            pending_pool=pool,
            pending_randomizer=randomizer,
        )
        return

    startup_duration = float(args.replay_init_time)
    startup_source = "replay:first_frame" if startup_duration > 0.0 else "disabled"

    episode_index = app.writer.start_episode(
        metadata=_episode_metadata(
            args,
            app.active_sides,
            app.calibration,
            app.camera_sources,
            app.preview_camera_sources,
            extra={
                "collection_mode": "trajectory_replay",
                "replay_group_key": group.key,
                "replay_group_label": group.label,
                "replay_source_path": group.source_path,
                "replay_source_episode_index": int(group.source_episode_index),
                "replay_start_index": int(group.start_index),
                "replay_end_index": int(group.end_index),
                "replay_frame_count": int(group.frame_count),
                "replay_action_hz": float(group.action_hz),
                "replay_init_time": float(startup_duration),
                "replay_pool_size": int(pool.size),
                "replay_pool_started_count": int(pool.started_count),
                "replay_pool_episode_range": str(pool.episode_range_spec),
                "replay_pool_episode_indices": [int(index) for index in pool.episode_indices],
                "replay_init_episode_ready": bool(state.replay_init_ready),
                "replay_init_episode": None
                if app.replay_init_plan is None
                else {
                    "source_episode_index": int(app.replay_init_plan.group.source_episode_index),
                    "start_index": int(app.replay_init_plan.group.start_index),
                    "end_index": int(app.replay_init_plan.group.end_index),
                    "move_time_s": float(app.replay_init_plan.move_time_s),
                    "to_record_time_s": float(app.replay_init_plan.to_record_time_s),
                },
                "replay_randomization": {}
                if randomizer is None
                else randomizer.metadata(),
                "replay_metadata": dict(group.metadata),
            },
        )
    )
    app.sample_builder.set_episode(episode_index)
    startup_initializer = StartupInitializer(
        duration_s=float(startup_duration),
        source=startup_source,
        target_action=first_action,
        active_sides=app.active_sides,
        arms_enabled=True,
        hands_enabled=any(f"{side}_hand" in app.components.backends for side in app.active_sides),
    )
    startup_initializer.reset()
    state.recording_active = True
    state.active_episode_index = int(episode_index)
    state.active_episode_started_at = time.time()
    state.active_replay_run = ActiveReplayRun(
        group=group,
        startup_initializer=startup_initializer,
        episode_index=int(episode_index),
        started_at=time.time(),
        randomizer=randomizer,
        original_max_angvel_by_side=_begin_replay_arm_speed_override(
            app,
            args.replay_arm_max_angvel,
        ),
    )
    app.components.session.set_calibration_ready()
    app.components.session.mark_debug_passed()
    app.components.session.arm_hardware()
    _log(
        "Replay",
        f"started key={group.key} label={group.label} episode={episode_index} "
        f"source_episode={group.source_episode_index} pool={pool.started_count}/{pool.size} "
        f"frames={group.frame_count} hz={group.action_hz:.2f} init={float(startup_duration):.2f}s "
        f"randomize={randomizer is not None}",
    )
    _log("ReplayHand", f"first_action { _hand_target_summary(first_action, app.active_sides) }")


def _start_recorded_replay_run_from_group(
    app: CollectApp,
    args: argparse.Namespace,
    *,
    group: ReplayTrajectoryGroup,
    pool_size: int,
    pool_started_count: int,
    pool_episode_range: str,
    pool_episode_indices: Sequence[int],
    randomizer: ReplayTrajectoryRandomizer | None,
    startup_duration: float,
    startup_source: str,
    startup_target_action: np.ndarray | None,
) -> None:
    state = app.control_state
    if state.recording_active or state.active_replay_run is not None:
        _log("Replay", "ignore recorded start: another recorded replay is active.")
        return
    episode_index = app.writer.start_episode(
        metadata=_episode_metadata(
            args,
            app.active_sides,
            app.calibration,
            app.camera_sources,
            app.preview_camera_sources,
            extra={
                "collection_mode": "trajectory_replay",
                "replay_group_key": group.key,
                "replay_group_label": group.label,
                "replay_source_path": group.source_path,
                "replay_source_episode_index": int(group.source_episode_index),
                "replay_start_index": int(group.start_index),
                "replay_end_index": int(group.end_index),
                "replay_frame_count": int(group.frame_count),
                "replay_action_hz": float(group.action_hz),
                "replay_init_time": float(startup_duration),
                "replay_pool_size": int(pool_size),
                "replay_pool_started_count": int(pool_started_count),
                "replay_pool_episode_range": str(pool_episode_range),
                "replay_pool_episode_indices": [int(index) for index in pool_episode_indices],
                "replay_init_episode_ready": bool(state.replay_init_ready),
                "replay_init_episode": None
                if app.replay_init_plan is None
                else {
                    "source_episode_index": int(app.replay_init_plan.group.source_episode_index),
                    "start_index": int(app.replay_init_plan.group.start_index),
                    "end_index": int(app.replay_init_plan.group.end_index),
                    "move_time_s": float(app.replay_init_plan.move_time_s),
                    "to_record_time_s": float(app.replay_init_plan.to_record_time_s),
                },
                "replay_randomization": {}
                if randomizer is None
                else randomizer.metadata(),
                "replay_metadata": dict(group.metadata),
            },
        )
    )
    app.sample_builder.set_episode(episode_index)
    startup_initializer = StartupInitializer(
        duration_s=float(startup_duration),
        source=str(startup_source) if float(startup_duration) > 0.0 else "disabled",
        target_action=startup_target_action,
        active_sides=app.active_sides,
        arms_enabled=True,
        hands_enabled=any(f"{side}_hand" in app.components.backends for side in app.active_sides),
    )
    startup_initializer.reset()
    state.recording_active = True
    state.active_episode_index = int(episode_index)
    state.active_episode_started_at = time.time()
    state.active_replay_run = ActiveReplayRun(
        group=group,
        startup_initializer=startup_initializer,
        episode_index=int(episode_index),
        started_at=time.time(),
        randomizer=randomizer,
        original_max_angvel_by_side=_begin_replay_arm_speed_override(
            app,
            args.replay_arm_max_angvel,
        ),
    )
    app.components.session.set_calibration_ready()
    app.components.session.mark_debug_passed()
    app.components.session.arm_hardware()
    _log(
        "Replay",
        f"started key={group.key} label={group.label} episode={episode_index} "
        f"source_episode={group.source_episode_index} pool={pool_started_count}/{pool_size} "
        f"frames={group.frame_count} hz={group.action_hz:.2f} init={float(startup_duration):.2f}s "
        f"randomize={randomizer is not None}",
    )
    if startup_target_action is not None:
        _log("ReplayHand", f"startup_target { _hand_target_summary(startup_target_action, app.active_sides) }")


def _abort_replay_episode(app: CollectApp) -> bool:
    if app.control_state.active_replay_run is None:
        _log("Replay", "ignore x: no active replay episode.")
        return False
    return _discard_active_episode(
        app,
        reason="x/replay_abort",
        show_standby=True,
    )


def _start_unrecorded_replay_run(
    app: CollectApp,
    args: argparse.Namespace,
    *,
    group: ReplayTrajectoryGroup,
    purpose: str,
    startup_duration_s: float,
    startup_source: str,
    target_action: np.ndarray | None,
    startup_target_qpos_by_side: Mapping[str, np.ndarray] | None = None,
    pending_pool: ReplayTrajectoryPool | None = None,
    pending_randomizer: ReplayTrajectoryRandomizer | None = None,
) -> None:
    state = app.control_state
    if app.components.session.rig is None:
        raise RuntimeError("Trajectory replay requires an active rig.")
    if app.replay_action_filter is None:
        raise RuntimeError("Trajectory replay action filter is unavailable.")
    if state.recording_active or state.active_replay_run is not None or state.active_unrecorded_replay_run is not None:
        _log("ReplayInit", f"ignore start purpose={purpose}: another run is already active.")
        return
    startup_initializer: StartupInitializer | None = None
    if float(startup_duration_s) > 0.0:
        startup_initializer = StartupInitializer(
            duration_s=float(startup_duration_s),
            source=str(startup_source),
            target_action=None if target_action is None else np.asarray(target_action, dtype=np.float32).reshape(ACTION_DIM),
            active_sides=app.active_sides,
            arms_enabled=True,
            hands_enabled=any(f"{side}_hand" in app.components.backends for side in app.active_sides),
        )
        startup_initializer.reset()
    state.active_unrecorded_replay_run = ActiveUnrecordedReplayRun(
        group=group,
        startup_initializer=startup_initializer,
        started_at=time.time(),
        purpose=str(purpose),
        startup_target_qpos_by_side={}
        if startup_target_qpos_by_side is None
        else {
            str(side): np.asarray(qpos, dtype=np.float32).reshape(6).copy()
            for side, qpos in startup_target_qpos_by_side.items()
        },
        pending_pool_key="" if pending_pool is None else str(pending_pool.key),
        pending_pool_size=0 if pending_pool is None else int(pending_pool.size),
        pending_pool_started_count=0 if pending_pool is None else int(pending_pool.started_count),
        pending_pool_episode_range="" if pending_pool is None else str(pending_pool.episode_range_spec),
        pending_pool_episode_indices=()
        if pending_pool is None
        else tuple(int(index) for index in pending_pool.episode_indices),
        pending_randomizer=pending_randomizer,
        original_max_angvel_by_side=_begin_replay_arm_speed_override(
            app,
            args.replay_arm_max_angvel,
        ),
    )
    app.components.session.set_calibration_ready()
    app.components.session.mark_debug_passed()
    app.components.session.arm_hardware()
    _log(
        "ReplayInit",
        f"started purpose={purpose} source_episode={group.source_episode_index} "
        f"frames={group.frame_count} startup={float(startup_duration_s):.2f}s",
    )
    preview_action = (
        np.asarray(target_action, dtype=np.float32).reshape(ACTION_DIM)
        if target_action is not None
        else group.actions[0].astype(np.float32, copy=True)
    )
    _log("ReplayHand", f"{purpose} first_action { _hand_target_summary(preview_action, app.active_sides) }")


def _start_replay_init_episode_cycle(app: CollectApp, args: argparse.Namespace) -> None:
    plan = app.replay_init_plan
    state = app.control_state
    if plan is None:
        _log("ReplayInit", "ignore h: no --replay-init-episode is configured.")
        return
    if state.recording_active or state.active_replay_run is not None or state.active_unrecorded_replay_run is not None:
        _log("ReplayInit", "ignore h: another recording/replay run is active.")
        return
    if not state.replay_init_at_start:
        state.replay_init_ready = False
        first_action = plan.group.actions[0].astype(np.float32, copy=True)
        target_qpos_by_side = {
            side: targets[0].astype(np.float32, copy=True)
            for side, targets in plan.group.arm_joint_targets.items()
            if side in set(app.active_sides)
        }
        missing_sides = sorted(set(app.active_sides) - set(target_qpos_by_side))
        if missing_sides:
            raise RuntimeError(
                "Cannot start init_episode qpos move: missing replay arm joint targets for "
                f"{missing_sides}. Ensure the source dataset contains action.replay.<side>_arm_joint_target columns."
            )
        _start_unrecorded_replay_run(
            app,
            args,
            group=plan.group,
            purpose="move_to_init_start",
            startup_duration_s=0.0,
            startup_source="disabled",
            target_action=first_action,
            startup_target_qpos_by_side=target_qpos_by_side,
        )
        return
    _start_unrecorded_replay_run(
        app,
        args,
        group=plan.group,
        purpose="execute_init_episode",
        startup_duration_s=0.0,
        startup_source="disabled",
        target_action=None,
    )


def _rollback_pending_replay_pool_selection(app: CollectApp, run: ActiveUnrecordedReplayRun) -> None:
    if run.purpose != "move_to_record_start" or not run.pending_pool_key:
        return
    pool = app.replay_groups.get(str(run.pending_pool_key))
    if pool is None or pool.size <= 0:
        return
    pool.cursor = (int(pool.cursor) - 1) % int(pool.size)
    pool.started_count = max(0, int(pool.started_count) - 1)


def _advance_unrecorded_replay_run(
    app: CollectApp,
    args: argparse.Namespace,
    *,
    now: float,
    step_count: int,
) -> bool:
    run = app.control_state.active_unrecorded_replay_run
    if run is None:
        return False
    assert app.replay_action_filter is not None
    observation = _build_replay_deployment_observation(
        app,
        now=float(now),
        step_count=step_count,
        last_action=run.last_action,
        collection_mode="trajectory_replay_unrecorded",
        replay_group=run.group.label,
        replay_index=run.replay_index,
    )

    raw_action: NormalizedAction | None = None
    if run.purpose == "move_to_init_start" and run.startup_target_qpos_by_side:
        command = _build_joint_target_teleop_command(
            run.startup_target_qpos_by_side,
            hand_action=run.group.actions[0],
            hand_sides=[
                side
                for side in app.active_sides
                if f"{side}_hand" in getattr(app.components.rig, "backends", {})
            ],
            now=float(now),
        )
        if command.commands:
            app.components.session.rig.send_commands(command)
        if _qpos_targets_reached(app, run.startup_target_qpos_by_side):
            _restore_replay_arm_speed_override(app, run.original_max_angvel_by_side)
            app.control_state.active_unrecorded_replay_run = None
            app.control_state.replay_init_at_start = True
            app.control_state.replay_init_ready = False
            _log("ReplayInit", "init-episode qpos start reached; press H again to execute T without recording.")
            return True
        return False
    if run.startup_initializer is not None and run.startup_initializer.enabled and not run.startup_initializer.completed:
        init_action, init_faults, init_ready = run.startup_initializer.next_action(
            observation,
            now=float(now),
        )
        if init_faults:
            raise RuntimeError("Unrecorded replay startup init failed: " + ",".join(init_faults))
        if init_action is not None:
            raw_action = init_action
        if init_ready:
            run.startup_initializer.mark_complete()
    else:
        if run.purpose == "move_to_init_start":
            _restore_replay_arm_speed_override(app, run.original_max_angvel_by_side)
            app.control_state.active_unrecorded_replay_run = None
            app.control_state.replay_init_at_start = True
            app.control_state.replay_init_ready = False
            _log("ReplayInit", "init-episode start pose reached; press H again to execute T without recording.")
            return True
        if run.purpose == "move_to_record_start":
            _restore_replay_arm_speed_override(app, run.original_max_angvel_by_side)
            app.control_state.active_unrecorded_replay_run = None
            _start_recorded_replay_run_from_group(
                app,
                args,
                group=run.group,
                pool_size=int(run.pending_pool_size),
                pool_started_count=int(run.pending_pool_started_count),
                pool_episode_range=str(run.pending_pool_episode_range),
                pool_episode_indices=tuple(run.pending_pool_episode_indices),
                randomizer=run.pending_randomizer,
                startup_duration=0.0,
                startup_source="disabled",
                startup_target_action=None,
            )
            return True
        if run.replay_index >= run.group.frame_count:
            purpose = str(run.purpose)
            _restore_replay_arm_speed_override(app, run.original_max_angvel_by_side)
            app.control_state.active_unrecorded_replay_run = None
            if purpose == "execute_init_episode":
                app.control_state.replay_init_at_start = False
                app.control_state.replay_init_ready = True
                _log("ReplayInit", "init-episode T finished; press A/D to start recorded replay collection.")
                _print_standby(app)
            else:
                _log("ReplayInit", f"completed purpose={purpose}.")
            return True
        action_vector = run.group.actions[run.replay_index].astype(np.float32, copy=True)
        if not _replay_active_arm_poses_are_finite(action_vector, app.active_sides):
            raise RuntimeError("Unrecorded replay action contains invalid active arm pose values.")
        raw_action = NormalizedAction(
            vector=action_vector,
            metadata={
                "safety_overrides": {
                    "max_arm_xyz_step_m": 0.0,
                    "max_arm_rot_step_deg": 0.0,
                    "max_arm_joint_step_rad": 0.0,
                    "max_hand_step": 0.0,
                }
            },
        )
        run.replay_index += 1

    if raw_action is None:
        return False
    deployment_action = app.replay_action_filter.filter(
        raw_action,
        observation,
        policy_name="trajectory_replay_unrecorded",
    )
    if deployment_action.faults:
        raise RuntimeError("Unrecorded replay action filter faults: " + ",".join(deployment_action.faults))
    command = _deployment_command_to_teleop_command(
        deployment_action.commands,
        timestamp=float(now),
        metadata={
            "policy": "trajectory_replay_unrecorded",
            "purpose": run.purpose,
            "replay_group_key": run.group.key,
            "replay_group_label": run.group.label,
            "replay_index": int(run.replay_index),
        },
    )
    if command.commands:
        app.components.session.rig.send_commands(command)
    run.last_action = deployment_action.vector.astype(np.float32, copy=True)
    return False


def _advance_replay_episode(
    app: CollectApp,
    args: argparse.Namespace,
    *,
    now: float,
    step_count: int,
) -> bool:
    run = app.control_state.active_replay_run
    if run is None:
        return False
    assert app.replay_action_filter is not None
    observation = _build_replay_deployment_observation(
        app,
        now=float(now),
        step_count=step_count,
        last_action=run.last_action,
        collection_mode="trajectory_replay",
        replay_group=run.group.label,
        replay_index=run.replay_index,
    )

    phase = "replay"
    source_index: int | None = None
    raw_action: NormalizedAction | None = None
    if run.startup_initializer.enabled and not run.startup_initializer.completed:
        init_action, init_faults, init_ready = run.startup_initializer.next_action(
            observation,
            now=float(now),
        )
        if init_faults:
            raise RuntimeError("Replay startup init failed: " + ",".join(init_faults))
        if init_action is not None:
            raw_action = init_action
            phase = "init"
        if init_ready or float(run.startup_initializer.progress) >= 1.0:
            run.startup_initializer.mark_complete()
    else:
        if run.replay_index >= run.group.frame_count:
            _save_active_episode(
                app,
                step_count=step_count,
                reason="trajectory_replay/end",
                show_standby=True,
            )
            return True
        source_index = int(run.group.start_index + run.replay_index)
        action_vector = run.group.actions[run.replay_index].astype(np.float32, copy=True)
        if run.randomizer is not None:
            action_vector = run.randomizer.apply(action_vector, now=float(now))
        if not _replay_active_arm_poses_are_finite(action_vector, app.active_sides):
            raise RuntimeError("Replay action randomization produced invalid active arm pose values.")
        raw_action = NormalizedAction(
            vector=action_vector,
            metadata={
                "safety_overrides": {
                    "max_arm_xyz_step_m": 0.0,
                    "max_arm_rot_step_deg": 0.0,
                    "max_arm_joint_step_rad": 0.0,
                    "max_hand_step": 0.0,
                }
            },
        )
        run.replay_index += 1

    if raw_action is None:
        return False
    deployment_action = app.replay_action_filter.filter(
        raw_action,
        observation,
        policy_name="trajectory_replay",
    )
    if deployment_action.faults:
        raise RuntimeError("Replay action filter faults: " + ",".join(deployment_action.faults))
    command = _deployment_command_to_teleop_command(
        deployment_action.commands,
        timestamp=float(now),
        metadata={
            "policy": "trajectory_replay",
            "phase": phase,
            "replay_group_key": run.group.key,
            "replay_group_label": run.group.label,
            "replay_index": int(run.replay_index),
            "source_frame_index": source_index,
        },
    )
    if command.commands:
        app.components.session.rig.send_commands(command)
    run.last_action = deployment_action.vector.astype(np.float32, copy=True)
    control_hz = (
        float(args.hz)
        if phase == "init" and float(args.hz) > 0.0
        else float(run.group.action_hz)
    )
    _submit_replay_sample(
        app,
        timestamp=float(now),
        command=command,
        control_hz=control_hz,
        metadata={
            "collection_mode": "trajectory_replay",
            "replay_phase": phase,
            "replay_group_key": run.group.key,
            "replay_group_label": run.group.label,
            "replay_index": int(run.replay_index),
            "source_frame_index": source_index,
            "startup_progress": float(run.startup_initializer.progress),
            "replay_randomized": run.randomizer is not None,
            "replay_random_offsets": {}
            if run.randomizer is None
            else run.randomizer.current_offsets_metadata(),
        },
    )
    if run.replay_index >= run.group.frame_count and (
        not run.startup_initializer.enabled or run.startup_initializer.completed
    ):
        _save_active_episode(
            app,
            step_count=step_count,
            reason="trajectory_replay/end",
            show_standby=True,
        )
        return True
    return False


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
    if key == "r":
        _start_return_home(app, args)
        return False
    if key in app.replay_groups:
        _start_replay_episode(app, args, key=key)
        return False
    if key == "x" and app.replay_groups:
        if app.control_state.active_unrecorded_replay_run is not None:
            run = app.control_state.active_unrecorded_replay_run
            purpose = run.purpose
            _restore_replay_arm_speed_override(app, run.original_max_angvel_by_side)
            _rollback_pending_replay_pool_selection(app, run)
            app.control_state.active_unrecorded_replay_run = None
            app.control_state.replay_init_at_start = False
            app.control_state.replay_init_ready = False
            app.components.session.hold_hardware("abort unrecorded replay init")
            _log("ReplayInit", f"aborted purpose={purpose}.")
            _print_standby(app)
        else:
            _abort_replay_episode(app)
        return False
    if key == "p":
        if app.replay_groups:
            _log("Keys", "ignore p: trajectory replay mode does not require VIVE home calibration.")
            return False
        _run_keyed_home_calibration(app, args)
        return False
    if key == "z":
        if app.replay_groups:
            _log("Keys", "ignore z: press a replay group key to start an automatic episode.")
            return False
        _start_recording_episode(app, args)
        return False
    if key == "c":
        if app.control_state.active_replay_run is not None:
            _log("Keys", "ignore c: trajectory replay auto-saves at the end; press x to abort.")
            return False
        _save_active_episode(app, step_count=step_count, reason="c/save")
        return False
    if key == "d" and app.replay_groups:
        _log("Keys", "ignore d: no replay group bound to d.")
        return False
    if key == "d":
        _discard_active_episode(app, reason="d/discard")
        return False
    if key == "h":
        if app.replay_groups and app.replay_init_plan is not None:
            _start_replay_init_episode_cycle(app, args)
            return False
        if app.replay_groups:
            app.components.session.hold_hardware("manual hold in replay mode")
            app.control_state.replay_init_at_start = False
            app.control_state.replay_init_ready = False
            _log("Keys", "h: replay hardware hold; no init episode configured.")
            _print_standby(app)
            return False
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
    replay_groups = _build_replay_groups(args)
    replay_init_plan = _build_replay_init_plan(args)
    calibration = _load_calibration(args)
    calibration.clear_home_state()
    camera_sources = _build_camera_sources(args)
    preview_camera_sources = (
        _select_preview_camera_sources(camera_sources, str(args.vis_cam_key))
        if bool(args.vis_cam) and camera_sources
        else {}
    )
    vive_source = None if replay_groups else _build_vive_source(args, active_sides)
    keyboard_source = KeyboardButtonSource(name="collection_clutch")
    default_clutch_pressed = args.control_mode == "delta_clutch"
    for side in ("left", "right"):
        keyboard_source.set_button(
            f"{side}_clutch_button",
            default_clutch_pressed,
            side=side,
            critical=False,
        )

    session_sources: list[Any] = [keyboard_source] if vive_source is None else [vive_source, keyboard_source]
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
    hand_rig_sides = (
        tuple(
            side
            for side in active_sides
            if f"{side}_hand" in all_backends
        )
        if replay_groups
        else tuple(hand_retargeters)
    )
    for side in hand_rig_sides:
        hand_backend = all_backends.get(f"{side}_hand")
        if hand_backend is not None:
            rig_backends[f"{side}_hand"] = hand_backend
            rig_backend_names.add(f"{side}_hand")

    rig = BimanualRig(
        backends=rig_backends,
        parallel_dispatch=len(rig_backends) > 1,
    )
    replay_hands_enabled = any(f"{side}_hand" in rig_backends for side in active_sides)
    replay_action_filter = (
        ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=active_sides,
                arms_enabled=True,
                hands_enabled=bool(replay_hands_enabled),
                workspace_min=(-10.0, -10.0, -10.0),
                workspace_max=(10.0, 10.0, 10.0),
                max_arm_xyz_step_m=float(args.replay_max_arm_xyz_step_m),
                max_arm_rot_step_deg=float(args.replay_max_arm_rot_step_deg),
                max_arm_joint_step_rad=0.0,
                max_hand_step=0.0,
            )
        )
        if replay_groups
        else None
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
            required_channels=()
            if replay_groups
            else tuple(f"{side}_wrist_pose" for side in moving_sides),
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
        replay_groups=replay_groups,
        replay_action_filter=replay_action_filter,
        replay_init_plan=replay_init_plan,
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
    if result is None and control.active_replay_run is not None:
        control_hz = float(control.active_replay_run.group.action_hz)
    else:
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
    submitted = int(runtime.submitted_samples) + int(control.submitted_samples)
    replay = (
        ""
        if control.active_replay_run is None
        else (
            f" replay={control.active_replay_run.group.label}:"
            f"{control.active_replay_run.replay_index}/{control.active_replay_run.group.frame_count}"
        )
    )
    replay_init = ""
    if control.active_unrecorded_replay_run is not None:
        run = control.active_unrecorded_replay_run
        replay_init = (
            f" init_replay={run.purpose}:"
            f"{run.replay_index}/{run.group.frame_count}"
        )
    elif app.replay_init_plan is not None:
        replay_init = (
            f" init_ready={control.replay_init_ready}"
            f" init_at_start={control.replay_init_at_start}"
        )
    return_home = "" if control.return_home_plan is None else " return_home=active"
    return (
        f"step={step_count} state={state_value} home={home} recording={recording} "
        f"episode={episode} control_hz={control_hz:.2f} submitted={submitted}"
        f"{replay}{replay_init}{return_home}{fixed_status}{jitter_status}{hand_status}{_camera_status_text(app)}"
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
        return_home_targets = _configured_return_home_targets(app, args)
        if return_home_targets:
            print(
                "R 归位 / R return qpos: "
                f"{_return_home_targets_summary(return_home_targets)} "
                f"max_angvel={np.round(np.asarray(args.return_home_angvel, dtype=float), 4).tolist()}"
            )
        if app.replay_groups:
            print("采集模式 / Collection mode: trajectory replay")
            rig_backend_names = set(getattr(app.components.rig, "backends", {}) or {})
            replay_hand_sides = [
                side for side in app.active_sides if f"{side}_hand" in rig_backend_names
            ]
            print(
                "  replay hands: "
                f"enabled_sides={replay_hand_sides} "
                "(requires --left/right-hand-ip or hand backend config)"
            )
            if not replay_hand_sides:
                _log(
                    "ReplayHand",
                    "no hand backend is in the replay rig; pass --left-hand-ip/--right-hand-ip "
                    "for active sides if the fixed trajectory should move fingers.",
                )
            if app.replay_init_plan is not None:
                group = app.replay_init_plan.group
                print(
                    "  init_episode: "
                    f"source_episode={group.source_episode_index} "
                    f"frames=[{group.start_index},{group.end_index}) "
                    f"move_time={app.replay_init_plan.move_time_s:.2f}s "
                    f"to_record_time={app.replay_init_plan.to_record_time_s:.2f}s"
                )
            for key, pool in sorted(app.replay_groups.items()):
                example = pool.groups[0]
                print(
                    f"  replay key={key} label={pool.label} "
                    f"balanced_episodes={pool.size} "
                    f"episode_range={pool.episode_range_spec} "
                    f"example_frames=[{example.start_index},{example.end_index}) "
                    f"first_source={example.source_path}"
                )
        _print_controls_for_app(app)
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
            if app.control_state.active_unrecorded_replay_run is not None:
                _advance_unrecorded_replay_run(
                    app,
                    args,
                    now=loop_started_at,
                    step_count=step_count,
                )
                step_count += 1
            elif app.control_state.active_replay_run is not None:
                _advance_replay_episode(
                    app,
                    args,
                    now=loop_started_at,
                    step_count=step_count,
                )
                step_count += 1
            elif app.control_state.return_home_plan is not None:
                _advance_return_home(app, now=loop_started_at)
                step_count += 1
            elif app.calibration.metadata.get("home_ready"):
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
            active_replay = app.control_state.active_replay_run
            active_unrecorded = app.control_state.active_unrecorded_replay_run
            active_period_sec = loop_period_sec
            if active_replay is not None and (
                not active_replay.startup_initializer.enabled
                or active_replay.startup_initializer.completed
            ):
                active_period_sec = 1.0 / float(active_replay.group.action_hz)
            elif active_unrecorded is not None:
                initializer = active_unrecorded.startup_initializer
                if initializer is None or not initializer.enabled or initializer.completed:
                    active_period_sec = 1.0 / float(active_unrecorded.group.action_hz)
            remaining = active_period_sec - elapsed
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n键盘中断，停止采集 Keyboard interrupt, stopping collection\n", flush=True)
    finally:
        try:
            if app.control_state.active_unrecorded_replay_run is not None:
                run = app.control_state.active_unrecorded_replay_run
                purpose = run.purpose
                _restore_replay_arm_speed_override(app, run.original_max_angvel_by_side)
                _rollback_pending_replay_pool_selection(app, run)
                app.control_state.active_unrecorded_replay_run = None
                app.components.session.hold_hardware("shutdown during unrecorded replay init")
                _log("ReplayInit", f"shutdown stopped unrecorded run purpose={purpose}.")
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
