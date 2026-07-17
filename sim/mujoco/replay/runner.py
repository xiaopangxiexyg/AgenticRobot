from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim.contracts import SceneSpec

from robot_motion.replay.control_schedule import (
    DEFAULT_POSITION_QPOS_CONTROL_HZ,
    DEFAULT_REPLAY_REFERENCE_HZ,
    ReplayControlAction,
    ReplayControlTick,
    iter_position_qpos_control_ticks,
    validate_replay_rate,
)

from ..config import default_camera_registry_path
from ..controllers import GripperController, QposController
from ..recorder import Real2SimRecorder
from ..scene import MujocoCameraRenderer, MujocoScene
from ..transforms import pose_to_xyz_quat_xyzw
from ..dataset_v22 import MujocoV22DatasetWriter, MujocoV22DatasetWriterConfig
from .dataset import fallback_hz, parquet_column_names, read_parquet_rows, resolve_episode
from .dataset_writer import SimReplayDatasetWriter, SimReplayDatasetWriterConfig
from .rgb_distortion import (
    DEFAULT_DISTORTION_SOURCE_CAMERA_MODEL,
    RealRawRgbDistorter,
    SOURCE_CAMERA_MODEL_CHOICES,
    render_intrinsics_for_rgb_distortion,
)
from .trajectory import (
    GripperReplaySample,
    GripperSource,
    QposSample,
    QposSource,
    extract_gripper_sample,
    extract_qpos_sample,
    gripper_read_columns,
    qpos_for_side,
    qpos_read_columns,
)


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay DexTeleop dataset QPOS in MuJoCo.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="LeRobot-style dataset root.")
    parser.add_argument(
        "--episode",
        required=True,
        help="Episode index/id such as 0 or episode_000000, or an explicit parquet path.",
    )
    parser.add_argument("--urdf", required=True, type=Path, help="Explicit URDF path to load.")
    parser.add_argument(
        "--mode",
        choices=("dual", "right"),
        default="dual",
        help="dual replays both arms; right replays the right arm and fixes the left arm at frame 0.",
    )
    parser.add_argument(
        "--qpos-source",
        choices=("auto", "actual", "current", "action"),
        default="auto",
        help="Dataset qpos source. auto tries actual, current, then action/replay targets.",
    )
    parser.add_argument(
        "--gripper-source",
        choices=("auto", "action", "state", "measured", "commanded", "hand_action", "hand_observation", "teleop"),
        default="auto",
        help=(
            "Dataset gripper source. auto tries gripper action/state/measured/commanded, "
            "then hand observation vectors."
        ),
    )
    parser.add_argument("--disable-gripper-replay", action="store_true", help="Do not replay dataset gripper values.")
    parser.add_argument("--require-gripper-replay", action="store_true", help="Fail if a gripper value is missing.")
    parser.add_argument(
        "--gripper-side",
        choices=("auto", "left", "right"),
        default="auto",
        help="Dataset side used for a single un-sided MuJoCo gripper. auto uses right in right mode.",
    )
    parser.add_argument("--gripper-name", default="", help="Optional MuJoCo gripper spec name to replay.")
    parser.add_argument(
        "--gripper-command-dim",
        type=int,
        default=1,
        help="Hand q6 dimension used when replaying gripper from vector columns.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Camera key or URDF link to render. Repeatable. Default: all discovered cameras.",
    )
    parser.add_argument(
        "--camera-registry",
        type=Path,
        default=default_camera_registry_path(),
        help="Camera registry JSON used to map logical camera keys to URDF links.",
    )
    parser.add_argument("--width", type=int, default=640, help="Render width.")
    parser.add_argument("--height", type=int, default=480, help="Render height.")
    parser.add_argument("--fovy", type=float, default=60.0, help="Fallback camera vertical FOV in degrees.")
    parser.add_argument(
        "--action-hz",
        type=float,
        default=0.0,
        help="Main replay action rate. 0 uses the source episode nominal hz.",
    )
    parser.add_argument(
        "--reference-hz",
        type=float,
        default=DEFAULT_REPLAY_REFERENCE_HZ,
        help="Intermediate position-reference rate between dataset actions and qpos commands.",
    )
    parser.add_argument(
        "--control-hz",
        "--command-hz",
        dest="control_hz",
        type=float,
        default=DEFAULT_POSITION_QPOS_CONTROL_HZ,
        help=(
            "Final position-command rate. --command-hz is an alias; --control-hz "
            "remains for compatibility."
        ),
    )
    parser.add_argument(
        "--no-final-control-settle",
        action="store_true",
        help=(
            "Stop at the first command-grid tick at/after the last reference "
            "instead of allowing one final reference-period segment to settle."
        ),
    )
    parser.add_argument("--dt", type=float, default=0.0, help="MuJoCo timestep. 0 uses 1 / --control-hz.")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Zero-based sorted parquet row offset to start from.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to replay. 0 means until the episode ends.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--no-sleep", action="store_true", help="Run as fast as possible.")
    parser.add_argument("--record", action="store_true", help="Save frames.jsonl and rendered PNG images.")
    parser.add_argument("--record-dir", type=Path, default=None, help="Record output base directory.")
    parser.add_argument(
        "--export-dataset",
        type=Path,
        default=None,
        help="Legacy replay export. Prefer --export-dataset-v22 for new sim datasets.",
    )
    parser.add_argument(
        "--export-dataset-v22",
        type=Path,
        default=None,
        help=(
            "Export a native PolicyIOLayout v2.2 MuJoCo dataset. This does not use the "
            "legacy replay dataset writer."
        ),
    )
    parser.add_argument(
        "--export-v22-policy-io-layout",
        type=Path,
        default=None,
        help=(
            "Canonical RM65 bimanual DM-Tac PolicyIOLayout JSON for --export-dataset-v22. "
            "Default: the real-collection bimanual DM-Tac preset."
        ),
    )
    parser.add_argument(
        "--export-dataset-name",
        default="",
        help="Optional dataset_name written to exported meta/info.json.",
    )
    parser.add_argument(
        "--export-episode-index",
        type=int,
        default=None,
        help="Episode index for --export-dataset. Default: append next index.",
    )
    parser.add_argument(
        "--export-fps",
        type=float,
        default=0.0,
        help="Video fps for --export-dataset. 0 uses --action-hz/source nominal hz.",
    )
    parser.add_argument("--export-video-codec", default="mp4v", help="OpenCV fourcc for exported MP4 videos.")
    parser.add_argument("--export-parquet-compression", default="zstd", help="Parquet compression.")
    parser.add_argument(
        "--distort-rgb-to-real-raw",
        action="store_true",
        help=(
            "Postprocess rendered sim RGB frames into the source RealSense raw color distortion model. "
            "This only affects RGB images; depth export remains MuJoCo metric depth."
        ),
    )
    parser.add_argument(
        "--rgb-distortion-source-k",
        choices=SOURCE_CAMERA_MODEL_CHOICES,
        default=DEFAULT_DISTORTION_SOURCE_CAMERA_MODEL,
        help=(
            "Source pinhole camera matrix assumed for --distort-rgb-to-real-raw. "
            "mujoco uses the current MJCF fovy; undistort-alpha0/1 use OpenCV optimal "
            "undistorted matrices from the source RealSense intrinsics."
        ),
    )
    parser.add_argument(
        "--export-wrist-depth",
        action="store_true",
        help="Export both compact wrist depth MP4 streams and range masks for --export-dataset.",
    )
    parser.add_argument(
        "--export-wrist-depth-raw",
        action="store_true",
        help="Export sim wrist depth as compact MP4 uint8 linear-depth streams for --export-dataset.",
    )
    parser.add_argument(
        "--export-wrist-depth-mask",
        action="store_true",
        help="Export sim wrist depth range masks for --export-dataset.",
    )
    parser.add_argument(
        "--wrist-depth-camera",
        action="append",
        default=None,
        help="Camera key to write as wrist depth. Repeatable. Default: left_wrist and right_wrist.",
    )
    parser.add_argument(
        "--wrist-depth-scale",
        type=float,
        default=0.0,
        help="Fallback z16 scale in meters/unit. 0 uses source RealSense metadata or 0.001.",
    )
    parser.add_argument(
        "--wrist-depth-min-m",
        type=float,
        default=0.0,
        help="Depth-mask minimum range in meters. 0 uses source real dataset config or 0.05.",
    )
    parser.add_argument(
        "--wrist-depth-max-m",
        type=float,
        default=0.0,
        help="Depth-mask maximum range in meters. 0 uses source real dataset config or 0.5.",
    )
    parser.add_argument(
        "--wrist-depth-video-min-m",
        type=float,
        default=0.05,
        help="Depth MP4 visible range minimum in meters; only affects video encoding.",
    )
    parser.add_argument(
        "--wrist-depth-video-max-m",
        type=float,
        default=1.0,
        help="Depth MP4 visible range maximum in meters; only affects video encoding.",
    )
    parser.add_argument(
        "--export-pointcloud",
        action="store_true",
        help="Export per-frame sim pointcloud NPZ files from MuJoCo metric depth for --export-dataset.",
    )
    parser.add_argument(
        "--pointcloud-camera",
        action="append",
        default=None,
        help="Camera key to export pointclouds for. Repeatable. Default: left_wrist and right_wrist.",
    )
    parser.add_argument(
        "--pointcloud-depth-min-m",
        type=float,
        default=0.05,
        help="Pointcloud minimum metric depth in meters.",
    )
    parser.add_argument(
        "--pointcloud-depth-max-m",
        type=float,
        default=1.0,
        help="Pointcloud maximum metric depth in meters.",
    )
    parser.add_argument(
        "--pointcloud-pixel-stride",
        type=int,
        default=1,
        help="Pixel stride for pointcloud sampling. 1 keeps every valid depth pixel.",
    )
    parser.add_argument(
        "--pointcloud-frame-stride",
        type=int,
        default=1,
        help="Export pointcloud every N exported action frames.",
    )
    parser.add_argument("--no-render", action="store_true", help="Skip camera rendering.")
    parser.add_argument("--viewer", action="store_true", help="Open a passive MuJoCo GUI viewer.")
    parser.add_argument(
        "--viewer-duration",
        type=float,
        default=0.0,
        help="Seconds to keep the GUI viewer open. 0 means until replay ends or the window is closed.",
    )
    parser.add_argument("--hide-left-ui", action="store_true", help="Hide MuJoCo viewer left UI panel.")
    parser.add_argument("--hide-right-ui", action="store_true", help="Hide MuJoCo viewer right UI panel.")
    parser.add_argument(
        "--hold-viewer-at-end",
        action="store_true",
        help="Keep the MuJoCo GUI open after the selected replay window finishes.",
    )
    parser.add_argument("--mjcf-out", type=Path, default=None, help="Optional path to write generated MJCF.")
    parser.add_argument(
        "--debug-proxy-visuals",
        action="store_true",
        help="Highlight URDF proxy geoms and place them in MuJoCo geom group 4.",
    )
    parser.add_argument("--print-camera-poses", action="store_true", help="Print camera world poses as JSON.")
    parser.add_argument("--print-every", type=int, default=30, help="Progress print period in frames.")
    return parser


def _active_sides(mode: str) -> tuple[str, ...]:
    return ("right",) if mode == "right" else ("left", "right")


def _read_rows(
    *,
    parquet_path: Path,
    active_sides: Sequence[str],
    qpos_source: QposSource,
    gripper_sides: Sequence[str] = (),
    gripper_source: GripperSource = "auto",
) -> list[dict[str, Any]]:
    available = set(parquet_column_names(parquet_path))
    requested = qpos_read_columns(active_sides, qpos_source)
    if gripper_sides:
        requested = tuple(
            dict.fromkeys(
                [
                    *requested,
                    *gripper_read_columns(gripper_sides, gripper_source),
                ]
            )
        )
    columns = [column for column in requested if column in available]
    if not columns:
        return read_parquet_rows(parquet_path)
    return read_parquet_rows(parquet_path, columns=columns)


def _slice_rows(rows: Sequence[dict[str, Any]], *, start_frame: int, max_frames: int) -> list[dict[str, Any]]:
    if start_frame < 0:
        raise ValueError("--start-frame must be non-negative.")
    end = None if max_frames <= 0 else start_frame + max_frames
    sliced = list(rows[start_frame:end])
    if not sliced:
        raise ValueError(
            f"No rows selected: start_frame={start_frame}, max_frames={max_frames}, total_rows={len(rows)}."
        )
    return sliced


def _episode_first_timestamp(rows: Sequence[dict[str, Any]]) -> float | None:
    for row in rows:
        value = row.get("timestamp")
        if value is None:
            continue
        try:
            timestamp = float(value)
        except Exception:
            continue
        if np.isfinite(timestamp):
            return timestamp
    return None


def _numeric_intrinsics_from_device_info(device_info: Mapping[str, Any]) -> dict[str, object] | None:
    intrinsics = device_info.get("intrinsics")
    if not isinstance(intrinsics, Mapping):
        return None
    out: dict[str, object] = {}
    for key in ("width", "height", "fx", "fy", "ppx", "ppy"):
        value = intrinsics.get(key, device_info.get(key))
        if value is None:
            continue
        try:
            out[key] = float(value)
        except Exception:
            continue
    coeffs = intrinsics.get("coeffs")
    if isinstance(coeffs, (list, tuple)):
        try:
            out["coeffs"] = [float(value) for value in coeffs]
        except Exception:
            pass
    if "width" in out:
        out["width"] = int(out["width"])
    if "height" in out:
        out["height"] = int(out["height"])
    return out if {"width", "height", "fy"} <= set(out) else None


def _source_camera_intrinsics(dataset_root: Path) -> dict[str, dict[str, object]]:
    path = Path(dataset_root) / "meta" / "devices" / "realsense.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}
    intrinsics_by_name: dict[str, dict[str, object]] = {}
    for camera_key, device_info in payload.items():
        if not isinstance(device_info, Mapping):
            continue
        intrinsics = _numeric_intrinsics_from_device_info(device_info)
        if intrinsics is not None:
            intrinsics_by_name[str(camera_key)] = intrinsics
    return intrinsics_by_name


def _source_camera_depth_metadata(dataset_root: Path) -> dict[str, dict[str, object]]:
    path = Path(dataset_root) / "meta" / "devices" / "realsense.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {}
    out: dict[str, dict[str, object]] = {}
    for camera_key, device_info in payload.items():
        if not isinstance(device_info, Mapping):
            continue
        item: dict[str, object] = {}
        for key in (
            "depth_enabled",
            "depth_width",
            "depth_height",
            "depth_fps",
            "align_depth_to_color",
            "depth_scale_m_per_unit",
        ):
            if key in device_info:
                item[key] = device_info[key]
        depth_intrinsics = device_info.get("depth_intrinsics")
        if isinstance(depth_intrinsics, Mapping):
            item["depth_intrinsics"] = dict(depth_intrinsics)
        if item:
            out[str(camera_key)] = item
    return out


def _fallback_wrist_depth_scale(
    depth_metadata_by_camera: Mapping[str, Mapping[str, object]],
    *,
    explicit_scale: float,
) -> float:
    if float(explicit_scale) > 0.0:
        return float(explicit_scale)
    for metadata in depth_metadata_by_camera.values():
        try:
            scale = float(metadata.get("depth_scale_m_per_unit"))
        except Exception:
            continue
        if np.isfinite(scale) and scale > 0.0:
            return float(scale)
    return 0.001


def _source_wrist_depth_mask_range(dataset_root: Path) -> tuple[float, float]:
    default_range = (0.05, 0.5)
    info_path = Path(dataset_root) / "meta" / "info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            info = {}
        config = info.get("wrist_depth_mask_config") if isinstance(info, Mapping) else None
        if isinstance(config, Mapping):
            try:
                return float(config["depth_min_m"]), float(config["depth_max_m"])
            except Exception:
                pass
    episodes_path = Path(dataset_root) / "meta" / "episodes.jsonl"
    if episodes_path.exists():
        try:
            lines = episodes_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            lines = []
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            metadata = record.get("metadata") if isinstance(record, Mapping) else None
            if not isinstance(metadata, Mapping):
                continue
            try:
                return float(metadata["wrist_depth_min_m"]), float(metadata["wrist_depth_max_m"])
            except Exception:
                continue
    return default_range


def _resolve_wrist_depth_mask_range(args: argparse.Namespace, dataset_root: Path) -> tuple[float, float]:
    source_min, source_max = _source_wrist_depth_mask_range(dataset_root)
    min_m = float(args.wrist_depth_min_m) if float(args.wrist_depth_min_m) > 0.0 else source_min
    max_m = float(args.wrist_depth_max_m) if float(args.wrist_depth_max_m) > 0.0 else source_max
    if max_m <= min_m:
        raise SystemExit("--wrist-depth-max-m/source mask max must be greater than min.")
    return float(min_m), float(max_m)


def _initialize_inactive_sides(
    *,
    scene: MujocoScene,
    rows: Sequence[dict[str, Any]],
    inactive_sides: Sequence[str],
    qpos_source: QposSource,
) -> dict[str, str]:
    sources: dict[str, str] = {}
    if not rows:
        return sources
    first_row = rows[0]
    for side in inactive_sides:
        if side not in scene.arm_specs:
            continue
        try:
            qpos, column = qpos_for_side(first_row, side, qpos_source)
        except KeyError:
            continue
        scene.set_arm_qpos(side, qpos)
        sources[side] = column
    return sources


def _auto_gripper_side(args: argparse.Namespace, active_sides: Sequence[str]) -> str:
    if str(args.gripper_side) != "auto":
        return str(args.gripper_side)
    if str(args.mode) == "right":
        return "right"
    if "right" in set(active_sides):
        return "right"
    return str(next(iter(active_sides), "right"))


def _resolve_gripper_side_by_name(
    *,
    args: argparse.Namespace,
    scene: MujocoScene,
    active_sides: Sequence[str],
) -> dict[str, str]:
    if bool(args.disable_gripper_replay) or not scene.gripper_specs:
        return {}
    requested_name = str(args.gripper_name or "").strip()
    if requested_name:
        if requested_name not in scene.gripper_specs:
            raise KeyError(f"--gripper-name {requested_name!r} is not available: {sorted(scene.gripper_specs)}")
        names = (requested_name,)
    else:
        names = tuple(sorted(scene.gripper_specs))
    default_side = _auto_gripper_side(args, active_sides)
    out: dict[str, str] = {}
    for name in names:
        spec = scene.gripper_specs[name]
        out[name] = str(spec.side or default_side)
    return out


def _create_viewer(args: argparse.Namespace, scene: MujocoScene):
    if not args.viewer:
        return None
    import mujoco.viewer

    return mujoco.viewer.launch_passive(
        scene.model,
        scene.data,
        show_left_ui=not bool(args.hide_left_ui),
        show_right_ui=not bool(args.hide_right_ui),
    )


def _viewer_sim_alive(viewer_handle) -> bool:
    sim_ref = getattr(viewer_handle, "_sim", None)
    if callable(sim_ref):
        try:
            return sim_ref() is not None
        except Exception:
            return False
    try:
        return bool(viewer_handle.is_running())
    except Exception:
        return False


def _close_viewer(viewer_handle, *, timeout_sec: float = 3.0) -> None:
    try:
        viewer_handle.close()
    except Exception:
        return
    deadline = time.time() + max(float(timeout_sec), 0.0)
    while time.time() < deadline:
        if not _viewer_sim_alive(viewer_handle):
            break
        time.sleep(0.02)
    # MuJoCo's passive viewer runs its GUI loop on a daemon thread on Linux.
    # Give that thread a short chance to finish C++/GLFW teardown before the
    # Python process starts interpreter shutdown.
    time.sleep(0.05)


def _wait_for_frame(
    *,
    replay_started_at: float,
    playback_time_sec: float,
    speed: float,
    no_sleep: bool,
) -> None:
    if no_sleep:
        return
    target = replay_started_at + playback_time_sec / max(float(speed), 1e-6)
    remaining = target - time.time()
    if remaining > 0.0:
        time.sleep(remaining)


def _timestamp_for_playback(first_timestamp: float | None, playback_time_sec: float) -> float | None:
    if first_timestamp is None:
        return None
    return float(first_timestamp) + float(playback_time_sec)


def _qpos_sample_for_control_tick(
    tick: ReplayControlTick,
    *,
    action_sample: QposSample,
    first_timestamp: float | None,
) -> QposSample:
    return QposSample(
        frame_index=int(action_sample.frame_index),
        timestamp=_timestamp_for_playback(first_timestamp, tick.relative_time_sec),
        relative_time_sec=float(tick.relative_time_sec),
        qpos_by_side={side: qpos.astype(float, copy=True) for side, qpos in tick.qpos_by_side.items()},
        source_by_side=dict(action_sample.source_by_side),
    )


def _gripper_sample_for_control_tick(
    tick: ReplayControlTick,
    *,
    gripper_sample: GripperReplaySample | None,
    first_timestamp: float | None,
) -> GripperReplaySample | None:
    if gripper_sample is None:
        return None
    return GripperReplaySample(
        frame_index=int(gripper_sample.frame_index),
        timestamp=_timestamp_for_playback(first_timestamp, tick.relative_time_sec),
        relative_time_sec=float(tick.relative_time_sec),
        close_by_name=dict(gripper_sample.close_by_name),
        close_by_side=dict(gripper_sample.close_by_side),
        source_by_name=dict(gripper_sample.source_by_name),
        source_by_side=dict(gripper_sample.source_by_side),
    )


def _write_replay_metadata(
    recorder: Real2SimRecorder,
    *,
    args: argparse.Namespace,
    episode_path: Path,
    episode_id: str,
    total_rows: int,
    selected_rows: int,
    planned_reference_frames: int,
    planned_control_ticks: int,
    action_hz: float,
    reference_hz: float,
    control_hz: float,
    sim_timestep_sec: float,
    inactive_sources: dict[str, str],
) -> None:
    metadata = {
        "dataset_root": args.dataset_root,
        "episode": args.episode,
        "episode_id": episode_id,
        "parquet_path": episode_path,
        "urdf_path": args.urdf,
        "mode": args.mode,
        "qpos_source": args.qpos_source,
        "total_rows": total_rows,
        "selected_rows": selected_rows,
        "planned_reference_frames": planned_reference_frames,
        "planned_control_ticks": planned_control_ticks,
        "start_frame": args.start_frame,
        "max_frames": args.max_frames,
        "action_hz": action_hz,
        "reference_hz": reference_hz,
        "control_hz": control_hz,
        "command_hz": control_hz,
        "sim_timestep_sec": sim_timestep_sec,
        "position_qpos_passthrough": {
            "main_action_hz": action_hz,
            "reference_hz": reference_hz,
            "control_hz": control_hz,
            "command_hz": control_hz,
            "pipeline": "timestamped_action_to_reference_to_command",
            "final_control_settle": not bool(args.no_final_control_settle),
            "reference_final_time_rule": "append_exact_unaligned_action_endpoint",
            "command_final_time_rule": (
                "fixed_rate_first_tick_at_or_after_final_settle_deadline"
            ),
            "reference_command_tie_break": "accept_reference_before_command_emit",
        },
        "speed": args.speed,
        "inactive_sources": inactive_sources,
    }
    (recorder.root_dir / "replay_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    if float(args.speed) <= 0.0:
        raise SystemExit("--speed must be positive.")
    if args.export_dataset is not None and args.export_dataset_v22 is not None:
        raise SystemExit("Choose exactly one of --export-dataset and --export-dataset-v22.")
    export_requested = args.export_dataset is not None or args.export_dataset_v22 is not None
    export_v22_requested = args.export_dataset_v22 is not None
    if export_requested and bool(args.no_render):
        raise SystemExit("Dataset export records sim camera videos, so it cannot be used with --no-render.")
    if export_v22_requested and str(args.mode) != "dual":
        raise SystemExit("--export-dataset-v22 currently requires --mode dual for the fixed bimanual v2.2 arm core.")
    export_wrist_depth_raw = bool(args.export_wrist_depth or args.export_wrist_depth_raw)
    export_wrist_depth_mask = bool(args.export_wrist_depth or args.export_wrist_depth_mask)
    if (export_wrist_depth_raw or export_wrist_depth_mask) and not export_requested:
        raise SystemExit("--export-wrist-depth* requires --export-dataset.")
    if bool(args.export_pointcloud) and not export_requested:
        raise SystemExit("--export-pointcloud requires --export-dataset.")
    if export_v22_requested and (export_wrist_depth_raw or export_wrist_depth_mask or bool(args.export_pointcloud)):
        raise SystemExit(
            "--export-dataset-v22 currently exports the canonical RGB + low-dimensional v2.2 contract only; "
            "depth/pointcloud are intentionally unavailable rather than silently written through the legacy writer."
        )
    if float(args.wrist_depth_video_max_m) <= float(args.wrist_depth_video_min_m):
        raise SystemExit("--wrist-depth-video-max-m must be greater than --wrist-depth-video-min-m.")
    if float(args.pointcloud_depth_max_m) <= float(args.pointcloud_depth_min_m):
        raise SystemExit("--pointcloud-depth-max-m must be greater than --pointcloud-depth-min-m.")
    if int(args.pointcloud_pixel_stride) <= 0:
        raise SystemExit("--pointcloud-pixel-stride must be positive.")
    if int(args.pointcloud_frame_stride) <= 0:
        raise SystemExit("--pointcloud-frame-stride must be positive.")

    active_sides = _active_sides(args.mode)
    read_sides = ("left", "right") if args.mode == "right" else active_sides
    episode = resolve_episode(args.dataset_root, args.episode)
    source_nominal_hz = fallback_hz(episode)
    action_hz = (
        validate_replay_rate(float(args.action_hz), name="action_hz")
        if float(args.action_hz) > 0.0
        else validate_replay_rate(source_nominal_hz, name="source_nominal_hz")
    )
    reference_hz = validate_replay_rate(float(args.reference_hz), name="reference_hz")
    control_hz = validate_replay_rate(float(args.control_hz), name="control_hz")
    sim_timestep_sec = float(args.dt) if float(args.dt) > 0.0 else 1.0 / control_hz
    if not np.isfinite(sim_timestep_sec) or sim_timestep_sec <= 0.0:
        raise SystemExit("--dt must be finite and positive, or 0 to use 1 / --control-hz.")
    rows = _read_rows(
        parquet_path=episode.parquet_path,
        active_sides=read_sides,
        qpos_source=args.qpos_source,
        gripper_sides=("left", "right") if not bool(args.disable_gripper_replay) else (),
        gripper_source=args.gripper_source,
    )
    selected_rows = _slice_rows(
        rows,
        start_frame=int(args.start_frame),
        max_frames=int(args.max_frames),
    )
    first_timestamp = _episode_first_timestamp(rows)
    camera_intrinsics = _source_camera_intrinsics(episode.dataset_root)
    render_camera_intrinsics = (
        render_intrinsics_for_rgb_distortion(
            camera_intrinsics,
            source_camera_model=str(args.rgb_distortion_source_k),
        )
        if bool(args.distort_rgb_to_real_raw)
        else camera_intrinsics
    )
    camera_depth_metadata = _source_camera_depth_metadata(episode.dataset_root)
    wrist_depth_mask_min_m, wrist_depth_mask_max_m = _resolve_wrist_depth_mask_range(
        args,
        episode.dataset_root,
    )

    camera_names = tuple(args.camera) if args.camera else ("all",)
    kinematic_scene_spec = SceneSpec.kinematic_replay()
    scene = MujocoScene.from_urdf(
        args.urdf,
        camera_names=camera_names,
        camera_registry_path=args.camera_registry,
        camera_width=int(args.width),
        camera_height=int(args.height),
        default_camera_fovy_deg=float(args.fovy),
        camera_intrinsics_by_name=render_camera_intrinsics,
        timestep=sim_timestep_sec,
        gravity=kinematic_scene_spec.gravity_mps2,
        mjcf_path=args.mjcf_out,
        debug_proxy_visuals=bool(args.debug_proxy_visuals),
        scene_spec=kinematic_scene_spec,
    )
    missing = sorted(set(active_sides) - set(scene.arm_specs))
    if missing:
        raise SystemExit(f"URDF/MJCF is missing arm specs for active side(s): {missing}")

    inactive_sides = tuple(side for side in scene.arm_specs if side not in active_sides)
    inactive_sources = _initialize_inactive_sides(
        scene=scene,
        rows=rows,
        inactive_sides=inactive_sides,
        qpos_source=args.qpos_source,
    )
    # Dataset replay must remain faithful to recorded qpos. Collision checks
    # are audit-only here; projecting would silently alter the source data.
    controller = QposController(scene, collision_mode="audit")
    gripper_controller = GripperController(scene, collision_mode="audit")
    gripper_side_by_name = _resolve_gripper_side_by_name(
        args=args,
        scene=scene,
        active_sides=active_sides,
    )
    rgb_distorter: RealRawRgbDistorter | None = None
    rgb_postprocess_metadata: dict[str, object] = {}
    if bool(args.distort_rgb_to_real_raw):
        if not camera_intrinsics:
            raise SystemExit("--distort-rgb-to-real-raw requires source RealSense intrinsics in the dataset.")
        rgb_distorter = RealRawRgbDistorter(
            camera_intrinsics_by_name=camera_intrinsics,
            camera_metadata_by_name=scene.camera_metadata(),
            source_camera_model=str(args.rgb_distortion_source_k),
        )
        for camera_name in scene.camera_names:
            rgb_distorter.map_for(
                camera_name,
                width=int(args.width),
                height=int(args.height),
            )
        rgb_postprocess_metadata = rgb_distorter.metadata()

    action_samples: list[QposSample] = []
    gripper_samples: list[GripperReplaySample | None] = []
    control_actions: list[ReplayControlAction] = []
    first_sample_time: float | None = None
    first_sample_timestamp: float | None = None
    for action_index, (row_offset, row) in enumerate(
        zip(range(int(args.start_frame), int(args.start_frame) + len(selected_rows)), selected_rows)
    ):
        sample = extract_qpos_sample(
            row,
            active_sides=active_sides,
            source=args.qpos_source,
            first_timestamp=first_timestamp,
            fallback_hz=action_hz,
            row_offset=row_offset,
        )
        if first_sample_time is None:
            first_sample_time = float(sample.relative_time_sec)
        playback_time = max(0.0, float(sample.relative_time_sec) - float(first_sample_time))
        if first_sample_timestamp is None and sample.timestamp is not None:
            first_sample_timestamp = float(sample.timestamp)
        action_sample = QposSample(
            frame_index=int(sample.frame_index),
            timestamp=sample.timestamp,
            relative_time_sec=playback_time,
            qpos_by_side={side: qpos.astype(float, copy=True) for side, qpos in sample.qpos_by_side.items()},
            source_by_side=dict(sample.source_by_side),
        )
        action_samples.append(action_sample)
        control_actions.append(
            ReplayControlAction(
                action_index=int(action_index),
                frame_index=int(action_sample.frame_index),
                relative_time_sec=playback_time,
                qpos_by_side=action_sample.qpos_by_side,
                metadata={
                    "source_frame_index": int(action_sample.frame_index),
                    "source_timestamp": action_sample.timestamp,
                    "qpos_source_by_side": dict(action_sample.source_by_side),
                },
            )
        )

        gripper_sample = None
        if gripper_side_by_name:
            try:
                raw_gripper_sample = extract_gripper_sample(
                    row,
                    gripper_side_by_name=gripper_side_by_name,
                    source=args.gripper_source,
                    command_dim=int(args.gripper_command_dim),
                    first_timestamp=first_timestamp,
                    fallback_hz=action_hz,
                    row_offset=row_offset,
                )
            except KeyError:
                if bool(args.require_gripper_replay):
                    raise
            else:
                gripper_sample = GripperReplaySample(
                    frame_index=int(raw_gripper_sample.frame_index),
                    timestamp=raw_gripper_sample.timestamp,
                    relative_time_sec=playback_time,
                    close_by_name=dict(raw_gripper_sample.close_by_name),
                    close_by_side=dict(raw_gripper_sample.close_by_side),
                    source_by_name=dict(raw_gripper_sample.source_by_name),
                    source_by_side=dict(raw_gripper_sample.source_by_side),
                )
        gripper_samples.append(gripper_sample)

    control_ticks = iter_position_qpos_control_ticks(
        control_actions,
        reference_hz=reference_hz,
        control_hz=control_hz,
        include_final_settle=not bool(args.no_final_control_settle),
    )
    # Reference indices are contiguous. Some references can be accepted and
    # preempted between command ticks when users choose reference_hz >
    # control_hz, so counting only indices visible on emitted ticks would
    # under-report the stage-one stream.
    planned_reference_frames = 1 + max(
        int(tick.reference_index) for tick in control_ticks
    )

    print(
        json.dumps(
            {
                "dataset_root": str(episode.dataset_root),
                "episode_id": episode.episode_id,
                "parquet_path": str(episode.parquet_path),
                "total_rows": len(rows),
                "selected_rows": len(selected_rows),
                "action_hz": action_hz,
                "reference_hz": reference_hz,
                "control_hz": control_hz,
                "command_hz": control_hz,
                "sim_timestep_sec": sim_timestep_sec,
                "planned_reference_frames": planned_reference_frames,
                "planned_control_ticks": len(control_ticks),
                "mode": args.mode,
                "active_sides": list(active_sides),
                "inactive_sources": inactive_sources,
                "urdf": str(args.urdf),
                "cameras": list(scene.camera_names),
                "camera_intrinsics_from_dataset": sorted(camera_intrinsics),
                "camera_intrinsics_for_render": sorted(render_camera_intrinsics),
                "camera_depth_metadata_from_dataset": sorted(camera_depth_metadata),
                "grippers": scene.gripper_metadata(),
                "gripper_replay_side_by_name": gripper_side_by_name,
                "gripper_source": args.gripper_source,
                "export_wrist_depth_raw": export_wrist_depth_raw,
                "export_wrist_depth_mask": export_wrist_depth_mask,
                "export_pointcloud": bool(args.export_pointcloud),
                "pointcloud_cameras": list(args.pointcloud_camera or ("left_wrist", "right_wrist")),
                "pointcloud_depth_range_m": [
                    float(args.pointcloud_depth_min_m),
                    float(args.pointcloud_depth_max_m),
                ],
                "pointcloud_pixel_stride": int(args.pointcloud_pixel_stride),
                "pointcloud_frame_stride": int(args.pointcloud_frame_stride),
                "wrist_depth_mask_range_m": [wrist_depth_mask_min_m, wrist_depth_mask_max_m],
                "wrist_depth_video_visible_range_m": [
                    float(args.wrist_depth_video_min_m),
                    float(args.wrist_depth_video_max_m),
                ],
                "rgb_postprocess": rgb_postprocess_metadata,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.print_camera_poses:
        print(json.dumps(scene.camera_metadata(), indent=2, ensure_ascii=False))

    renderer = None
    recorder = None
    export_writer = None
    export_dataset_root: Path | None = None
    viewer_handle = None
    replay_started_at = time.time()
    processed = 0
    try:
        if not args.no_render:
            try:
                renderer = MujocoCameraRenderer(scene, width=int(args.width), height=int(args.height))
            except Exception as exc:  # noqa: BLE001 - give a practical MuJoCo GL hint.
                raise RuntimeError(
                    "Failed to create MuJoCo renderer. In headless sessions try "
                    "`MUJOCO_GL=egl` or `MUJOCO_GL=osmesa`."
                ) from exc
        if args.record:
            recorder = Real2SimRecorder.create_timestamped(
                scene,
                root_dir=args.record_dir,
                save_images=not bool(args.no_render),
            )
            _write_replay_metadata(
                recorder,
                args=args,
                episode_path=episode.parquet_path,
                episode_id=episode.episode_id,
                total_rows=len(rows),
                selected_rows=len(selected_rows),
                planned_reference_frames=planned_reference_frames,
                planned_control_ticks=len(control_ticks),
                action_hz=action_hz,
                reference_hz=reference_hz,
                control_hz=control_hz,
                sim_timestep_sec=sim_timestep_sec,
                inactive_sources=inactive_sources,
            )
            print(f"record_dir={recorder.root_dir}")
        if args.export_dataset is not None:
            if not scene.camera_names:
                raise RuntimeError("No MuJoCo cameras were discovered; cannot export a camera dataset.")
            export_fps = float(args.export_fps) if float(args.export_fps) > 0.0 else float(action_hz)
            export_writer = SimReplayDatasetWriter(
                scene,
                SimReplayDatasetWriterConfig(
                    root_dir=args.export_dataset,
                    dataset_name=str(args.export_dataset_name or Path(args.export_dataset).name),
                    task_id="mujoco_real2sim_replay",
                    task_description="MuJoCo real2sim replay",
                    nominal_fps=export_fps,
                    episode_index=args.export_episode_index,
                    video_codec=str(args.export_video_codec),
                    parquet_compression=str(args.export_parquet_compression),
                    source_dataset_root=episode.dataset_root,
                    source_episode_id=episode.episode_id,
                    source_parquet_path=episode.parquet_path,
                    urdf_path=args.urdf,
                    mode=args.mode,
                    qpos_source=args.qpos_source,
                    gripper_command_dim=int(args.gripper_command_dim),
                    write_wrist_depth_raw=export_wrist_depth_raw,
                    write_wrist_depth_mask=export_wrist_depth_mask,
                    wrist_depth_camera_names=tuple(args.wrist_depth_camera or ("left_wrist", "right_wrist")),
                    wrist_depth_scale_m_per_unit=_fallback_wrist_depth_scale(
                        camera_depth_metadata,
                        explicit_scale=float(args.wrist_depth_scale),
                    ),
                    wrist_depth_min_m=wrist_depth_mask_min_m,
                    wrist_depth_max_m=wrist_depth_mask_max_m,
                    wrist_depth_video_min_m=float(args.wrist_depth_video_min_m),
                    wrist_depth_video_max_m=float(args.wrist_depth_video_max_m),
                    wrist_depth_metadata_by_camera=camera_depth_metadata,
                    rgb_postprocess=rgb_postprocess_metadata,
                    write_pointcloud=bool(args.export_pointcloud),
                    pointcloud_camera_names=tuple(args.pointcloud_camera or ("left_wrist", "right_wrist")),
                    pointcloud_depth_min_m=float(args.pointcloud_depth_min_m),
                    pointcloud_depth_max_m=float(args.pointcloud_depth_max_m),
                    pointcloud_pixel_stride=int(args.pointcloud_pixel_stride),
                    pointcloud_frame_stride=int(args.pointcloud_frame_stride),
                ),
                camera_names=scene.camera_names,
            )
            export_dataset_root = export_writer.root_dir
            if export_writer.wrist_depth_enabled and not export_writer.wrist_depth_camera_names:
                raise RuntimeError(
                    "Wrist depth export requested, but none of the requested wrist depth cameras "
                    f"are active in the scene cameras: {tuple(args.wrist_depth_camera or ('left_wrist', 'right_wrist'))}"
                )
            if export_writer.pointcloud_enabled and not export_writer.pointcloud_camera_names:
                raise RuntimeError(
                    "Pointcloud export requested, but none of the requested pointcloud cameras "
                    f"are active in the scene cameras: {tuple(args.pointcloud_camera or ('left_wrist', 'right_wrist'))}"
                )
            print(f"export_dataset={export_dataset_root}")
        elif args.export_dataset_v22 is not None:
            if not scene.camera_names:
                raise RuntimeError("No MuJoCo cameras were discovered; cannot export a v2.2 camera dataset.")
            if set(gripper_side_by_name.values()) != {"left", "right"}:
                raise RuntimeError(
                    "Native RM65 v2.2 export requires one resolved MuJoCo DM-Tac gripper for each side; "
                    f"got {gripper_side_by_name}."
                )
            if any(sample is None for sample in gripper_samples):
                raise RuntimeError(
                    "Native RM65 v2.2 export requires finite 1D gripper commands for both sides in every source frame."
                )
            export_fps = float(args.export_fps) if float(args.export_fps) > 0.0 else float(action_hz)
            export_writer = MujocoV22DatasetWriter(
                scene,
                MujocoV22DatasetWriterConfig(
                    root_dir=args.export_dataset_v22,
                    policy_io_layout_path=args.export_v22_policy_io_layout,
                    dataset_name=str(args.export_dataset_name or Path(args.export_dataset_v22).name),
                    nominal_fps=export_fps,
                    episode_index=args.export_episode_index,
                    video_codec=str(args.export_video_codec),
                    parquet_compression=str(args.export_parquet_compression),
                    source_dataset_root=episode.dataset_root,
                    source_episode_id=episode.episode_id,
                    source_parquet_path=episode.parquet_path,
                    urdf_path=args.urdf,
                    mode=args.mode,
                    qpos_source=args.qpos_source,
                    gripper_source=args.gripper_source,
                    control_hz=control_hz,
                    reference_hz=reference_hz,
                    gripper_name_by_side={side: name for name, side in gripper_side_by_name.items()},
                ),
                camera_names=scene.camera_names,
            )
            export_dataset_root = export_writer.root_dir
            print(f"export_dataset_v22={export_dataset_root}")
        viewer_handle = _create_viewer(args, scene)
        viewer_started_at = time.time()
        last_export_action_index: int | None = None
        exported_camera_frames = 0

        for tick in control_ticks:
            action_sample = action_samples[int(tick.action_index)]
            sample = _qpos_sample_for_control_tick(
                tick,
                action_sample=action_sample,
                first_timestamp=first_sample_timestamp,
            )
            playback_time = float(tick.relative_time_sec)
            _wait_for_frame(
                replay_started_at=replay_started_at,
                playback_time_sec=playback_time,
                speed=float(args.speed),
                no_sleep=bool(args.no_sleep),
            )

            qpos_result = controller.apply(sample.qpos_by_side)
            gripper_sample = _gripper_sample_for_control_tick(
                tick,
                gripper_sample=gripper_samples[int(tick.action_index)],
                first_timestamp=first_sample_timestamp,
            )
            gripper_result: dict[str, float] = {}
            if gripper_sample is not None:
                gripper_result = gripper_controller.apply(gripper_sample.close_by_name)
            if viewer_handle is not None:
                if not viewer_handle.is_running():
                    break
                viewer_handle.sync()
                if float(args.viewer_duration) > 0.0:
                    if time.time() - viewer_started_at >= float(args.viewer_duration):
                        break
            export_this_tick = bool(
                export_writer is not None
                and last_export_action_index != int(tick.action_index)
            )
            save_record_images = bool(recorder is not None and recorder.config.save_images)
            render_this_tick = bool(
                renderer is not None
                and (
                    export_this_tick
                    or save_record_images
                    or (recorder is None and export_writer is None)
                )
            )
            images = renderer.render_all() if render_this_tick and renderer is not None else {}
            if rgb_distorter is not None and images:
                images = rgb_distorter.distort_images(images)
            depth_camera_names: tuple[str, ...] = ()
            if export_this_tick and export_writer is not None and not export_v22_requested:
                depth_camera_names = tuple(
                    dict.fromkeys(
                        [
                            *(
                                export_writer.wrist_depth_camera_names
                                if export_writer.wrist_depth_enabled
                                else ()
                            ),
                            *(
                                export_writer.pointcloud_camera_names
                                if export_writer.pointcloud_enabled
                                else ()
                            ),
                        ]
                    )
                )
            depth_images = (
                renderer.render_all_depth(depth_camera_names)
                if renderer is not None and depth_camera_names
                else {}
            )
            if export_writer is not None and export_this_tick:
                if export_v22_requested:
                    assert gripper_sample is not None
                    export_writer.record_frame(
                        sample=sample,
                        qpos_result=qpos_result,
                        gripper_sample=gripper_sample,
                        gripper_result=gripper_result,
                        images=images,
                        playback_time_sec=playback_time,
                        active_sides=active_sides,
                    )
                else:
                    export_writer.record_frame(
                        sample=sample,
                        qpos_result=qpos_result,
                        gripper_sample=gripper_sample,
                        gripper_result=gripper_result,
                        images=images,
                        depth_images=depth_images,
                        playback_time_sec=playback_time,
                        active_sides=active_sides,
                    )
                last_export_action_index = int(tick.action_index)
                exported_camera_frames += 1
            if recorder is not None:
                recorder.record_step(
                    images=images,
                    extra=_frame_extra(
                        args=args,
                        sample=sample,
                        control_tick=tick,
                        qpos_result=qpos_result,
                        gripper_sample=gripper_sample,
                        gripper_result=gripper_result,
                        playback_time_sec=playback_time,
                        action_hz=action_hz,
                        reference_hz=reference_hz,
                        control_hz=control_hz,
                    ),
                )
            processed += 1
            if int(args.print_every) > 0 and (processed == 1 or processed % int(args.print_every) == 0):
                print(
                    f"replay control_tick={processed}/{len(control_ticks)} "
                    f"reference={tick.reference_index + 1}/{planned_reference_frames} "
                    f"action={tick.action_index + 1}/{len(action_samples)} "
                    f"dataset_frame={sample.frame_index} t={sample.relative_time_sec:.3f}s "
                    f"alpha={tick.interpolation_alpha:.3f}"
                )
        if not args.record and renderer is not None:
            print(f"rendered cameras={list(scene.camera_names)}")
        if viewer_handle is not None and bool(args.hold_viewer_at_end) and float(args.viewer_duration) <= 0.0:
            print("viewer_hold=close the MuJoCo window to exit")
            while viewer_handle.is_running():
                viewer_handle.sync()
                time.sleep(0.05)
    finally:
        if viewer_handle is not None:
            _close_viewer(viewer_handle)
            viewer_handle = None
        if export_writer is not None:
            export_writer.close(
                summary={
                    "processed_frames": exported_camera_frames,
                    "processed_camera_frames": exported_camera_frames,
                    "processed_control_ticks": processed,
                    "processed_reference_frames": planned_reference_frames,
                    "processed_action_frames": len(action_samples),
                    "source_episode_id": episode.episode_id,
                    "source_parquet_path": episode.parquet_path,
                    "start_frame": args.start_frame,
                    "max_frames": args.max_frames,
                    "action_hz": action_hz,
                    "reference_hz": reference_hz,
                    "control_hz": control_hz,
                    "command_hz": control_hz,
                    "gripper_replay_side_by_name": gripper_side_by_name,
                    "gripper_source": args.gripper_source,
                    "collision_validation": (
                        "audit_only"
                        if scene.collision_safety_available
                        else "unavailable_collision_disabled"
                    ),
                }
            )
        if recorder is not None:
            recorder.close()
        if renderer is not None:
            renderer.close()

    final = {
        "processed_frames": processed,
        "processed_control_ticks": processed,
        "processed_reference_frames": planned_reference_frames,
        "processed_action_frames": len(action_samples),
        "exported_camera_frames": exported_camera_frames if export_writer is not None else None,
        "action_hz": action_hz,
        "reference_hz": reference_hz,
        "control_hz": control_hz,
        "command_hz": control_hz,
        "export_dataset": str(export_dataset_root) if export_dataset_root is not None else None,
        "export_dataset_schema": "dexteleop_dataset_v2.2" if export_v22_requested else None,
        "qpos_by_side": scene.arm_qpos_by_side(),
        "gripper_close_by_name": scene.gripper_close_by_name(),
        "gripper_qpos_by_name": scene.gripper_qpos_by_name(),
        "collision_validation": (
            "audit_only"
            if scene.collision_safety_available
            else "unavailable_collision_disabled"
        ),
        "controlled_body_pose_world_xyz_quat_xyzw": {
            side: pose_to_xyz_quat_xyzw(scene.body_pose(spec.controlled_body))
            for side, spec in sorted(scene.arm_specs.items())
        },
    }
    print(json.dumps(final, indent=2, ensure_ascii=False))


def _frame_extra(
    *,
    args: argparse.Namespace,
    sample: QposSample,
    control_tick: ReplayControlTick,
    qpos_result: dict[str, list[float]],
    gripper_sample: GripperReplaySample | None,
    gripper_result: dict[str, float],
    playback_time_sec: float,
    action_hz: float,
    reference_hz: float,
    control_hz: float,
) -> dict[str, object]:
    return {
        "replay": {
            "mode": args.mode,
            "qpos_source": args.qpos_source,
            "action_hz": float(action_hz),
            "reference_hz": float(reference_hz),
            "control_hz": float(control_hz),
            "command_hz": float(control_hz),
            "position_control_pipeline": "timestamped_action_to_reference_to_command",
            "control_tick_index": int(control_tick.control_index),
            "action_index": int(control_tick.action_index),
            "action_frame_index": int(control_tick.action_frame_index),
            "action_relative_time_sec": float(control_tick.action_relative_time_sec),
            "reference_index": int(control_tick.reference_index),
            "reference_relative_time_sec": float(
                control_tick.reference_relative_time_sec
            ),
            "reference_action_interpolation_alpha": float(
                control_tick.reference_action_interpolation_alpha
            ),
            "control_segment_elapsed_sec": float(control_tick.segment_elapsed_sec),
            "control_segment_duration_sec": float(control_tick.segment_duration_sec),
            "control_interpolation_alpha": float(control_tick.interpolation_alpha),
            "control_tick_metadata": control_tick.metadata,
            "reference_qpos_by_side": control_tick.reference_qpos_by_side,
            "dataset_frame_index": sample.frame_index,
            "dataset_timestamp": sample.timestamp,
            "relative_time_sec": sample.relative_time_sec,
            "playback_time_sec": playback_time_sec,
            "qpos_source_by_side": sample.source_by_side,
            "target_qpos_by_side": sample.qpos_by_side,
            "result_qpos_by_side": qpos_result,
            "gripper_source_by_name": {} if gripper_sample is None else gripper_sample.source_by_name,
            "gripper_source_by_side": {} if gripper_sample is None else gripper_sample.source_by_side,
            "target_gripper_close_by_name": {} if gripper_sample is None else gripper_sample.close_by_name,
            "target_gripper_close_by_side": {} if gripper_sample is None else gripper_sample.close_by_side,
            "result_gripper_close_by_name": gripper_result,
        }
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
