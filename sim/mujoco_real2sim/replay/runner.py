from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import default_camera_registry_path
from ..controllers import GripperController, QposController
from ..recorder import Real2SimRecorder
from ..scene import MujocoCameraRenderer, MujocoScene
from ..transforms import pose_to_xyz_quat_xyzw
from .dataset import fallback_hz, parquet_column_names, read_parquet_rows, resolve_episode
from .dataset_writer import SimReplayDatasetWriter, SimReplayDatasetWriterConfig
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
    parser.add_argument("--dt", type=float, default=1.0 / 30.0, help="MuJoCo timestep.")
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
        help="Export replayed sim cameras and qpos to a LeRobot-style dataset root.",
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
        help="Video fps for --export-dataset. 0 uses the source episode nominal hz.",
    )
    parser.add_argument("--export-video-codec", default="mp4v", help="OpenCV fourcc for exported MP4 videos.")
    parser.add_argument("--export-parquet-compression", default="zstd", help="Parquet compression.")
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


def _write_replay_metadata(
    recorder: Real2SimRecorder,
    *,
    args: argparse.Namespace,
    episode_path: Path,
    episode_id: str,
    total_rows: int,
    selected_rows: int,
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
        "start_frame": args.start_frame,
        "max_frames": args.max_frames,
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
    if args.export_dataset is not None and bool(args.no_render):
        raise SystemExit("--export-dataset records sim camera videos, so it cannot be used with --no-render.")

    active_sides = _active_sides(args.mode)
    read_sides = ("left", "right") if args.mode == "right" else active_sides
    episode = resolve_episode(args.dataset_root, args.episode)
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
    nominal_hz = fallback_hz(episode)
    first_timestamp = _episode_first_timestamp(rows)
    camera_intrinsics = _source_camera_intrinsics(episode.dataset_root)

    camera_names = tuple(args.camera) if args.camera else ("all",)
    scene = MujocoScene.from_urdf(
        args.urdf,
        camera_names=camera_names,
        camera_registry_path=args.camera_registry,
        camera_width=int(args.width),
        camera_height=int(args.height),
        default_camera_fovy_deg=float(args.fovy),
        camera_intrinsics_by_name=camera_intrinsics,
        timestep=float(args.dt),
        gravity=(0.0, 0.0, 0.0),
        mjcf_path=args.mjcf_out,
        debug_proxy_visuals=bool(args.debug_proxy_visuals),
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
    controller = QposController(scene)
    gripper_controller = GripperController(scene)
    gripper_side_by_name = _resolve_gripper_side_by_name(
        args=args,
        scene=scene,
        active_sides=active_sides,
    )

    print(
        json.dumps(
            {
                "dataset_root": str(episode.dataset_root),
                "episode_id": episode.episode_id,
                "parquet_path": str(episode.parquet_path),
                "total_rows": len(rows),
                "selected_rows": len(selected_rows),
                "mode": args.mode,
                "active_sides": list(active_sides),
                "inactive_sources": inactive_sources,
                "urdf": str(args.urdf),
                "cameras": list(scene.camera_names),
                "camera_intrinsics_from_dataset": sorted(camera_intrinsics),
                "grippers": scene.gripper_metadata(),
                "gripper_replay_side_by_name": gripper_side_by_name,
                "gripper_source": args.gripper_source,
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
    first_sample_time: float | None = None
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
                inactive_sources=inactive_sources,
            )
            print(f"record_dir={recorder.root_dir}")
        if args.export_dataset is not None:
            if not scene.camera_names:
                raise RuntimeError("No MuJoCo cameras were discovered; cannot export a camera dataset.")
            export_fps = float(args.export_fps) if float(args.export_fps) > 0.0 else float(nominal_hz)
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
                ),
                camera_names=scene.camera_names,
            )
            export_dataset_root = export_writer.root_dir
            print(f"export_dataset={export_dataset_root}")
        viewer_handle = _create_viewer(args, scene)
        viewer_started_at = time.time()

        for row_offset, row in enumerate(selected_rows, start=int(args.start_frame)):
            sample = extract_qpos_sample(
                row,
                active_sides=active_sides,
                source=args.qpos_source,
                first_timestamp=first_timestamp,
                fallback_hz=nominal_hz,
                row_offset=row_offset,
            )
            if first_sample_time is None:
                first_sample_time = sample.relative_time_sec
            playback_time = max(0.0, sample.relative_time_sec - first_sample_time)
            _wait_for_frame(
                replay_started_at=replay_started_at,
                playback_time_sec=playback_time,
                speed=float(args.speed),
                no_sleep=bool(args.no_sleep),
            )

            qpos_result = controller.apply(sample.qpos_by_side)
            gripper_sample = None
            gripper_result: dict[str, float] = {}
            if gripper_side_by_name:
                try:
                    gripper_sample = extract_gripper_sample(
                        row,
                        gripper_side_by_name=gripper_side_by_name,
                        source=args.gripper_source,
                        command_dim=int(args.gripper_command_dim),
                        first_timestamp=first_timestamp,
                        fallback_hz=nominal_hz,
                        row_offset=row_offset,
                    )
                except KeyError:
                    if bool(args.require_gripper_replay):
                        raise
                else:
                    gripper_result = gripper_controller.apply(gripper_sample.close_by_name)
            if viewer_handle is not None:
                if not viewer_handle.is_running():
                    break
                viewer_handle.sync()
                if float(args.viewer_duration) > 0.0:
                    if time.time() - viewer_started_at >= float(args.viewer_duration):
                        break
            images = renderer.render_all() if renderer is not None else {}
            if export_writer is not None:
                export_writer.record_frame(
                    sample=sample,
                    qpos_result=qpos_result,
                    gripper_sample=gripper_sample,
                    gripper_result=gripper_result,
                    images=images,
                    playback_time_sec=playback_time,
                    active_sides=active_sides,
                )
            if recorder is not None:
                recorder.record_step(
                    images=images,
                    extra=_frame_extra(
                        args=args,
                        sample=sample,
                        qpos_result=qpos_result,
                        gripper_sample=gripper_sample,
                        gripper_result=gripper_result,
                        playback_time_sec=playback_time,
                    ),
                )
            processed += 1
            if int(args.print_every) > 0 and (processed == 1 or processed % int(args.print_every) == 0):
                print(
                    f"replay frame={processed}/{len(selected_rows)} "
                    f"dataset_frame={sample.frame_index} t={sample.relative_time_sec:.3f}s"
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
                    "processed_frames": processed,
                    "source_episode_id": episode.episode_id,
                    "source_parquet_path": episode.parquet_path,
                    "start_frame": args.start_frame,
                    "max_frames": args.max_frames,
                    "gripper_replay_side_by_name": gripper_side_by_name,
                    "gripper_source": args.gripper_source,
                }
            )
        if recorder is not None:
            recorder.close()
        if renderer is not None:
            renderer.close()

    final = {
        "processed_frames": processed,
        "export_dataset": str(export_dataset_root) if export_dataset_root is not None else None,
        "qpos_by_side": scene.arm_qpos_by_side(),
        "gripper_close_by_name": scene.gripper_close_by_name(),
        "gripper_qpos_by_name": scene.gripper_qpos_by_name(),
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
    qpos_result: dict[str, list[float]],
    gripper_sample: GripperReplaySample | None,
    gripper_result: dict[str, float],
    playback_time_sec: float,
) -> dict[str, object]:
    return {
        "replay": {
            "mode": args.mode,
            "qpos_source": args.qpos_source,
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
