"""Record an immutable grasp trajectory on its original timestamp lattice.

This is an offline MuJoCo visualization.  It teleports the simulated arm to
each already-planned qpos sample, so the resulting video shows the planner's
exact timing without introducing another IK, controller, or execution owner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from project_paths import resolve_path
from sim.contracts import SimulationContract
from sim.mujoco.actuation import (
    cad_bimanual_dmtac_gripper_profile,
    rm65_bimanual_udp_feedback_profile,
)
from sim.mujoco.scene import MujocoCameraRenderer, MujocoScene
from teleop_runtime.skill_deploy.grasp_trajectory_record import (
    load_grasp_trajectory_record,
)

DEFAULT_BLUE_CUBE_CONTRACT = Path(
    "experiments/sim_point_grasp/configs/rm65_bimanual_blue_cube_v1.json"
)
REPORT_SCHEMA = "agenticrobot.mujoco.timestamped_grasp_video.v1"


def infer_uniform_sample_hz(
    timestamps_sec: np.ndarray,
    *,
    absolute_tolerance_sec: float = 1e-9,
) -> float:
    """Return the source sample rate or reject a non-uniform timestamp lattice."""

    timestamps = np.asarray(timestamps_sec, dtype=float).reshape(-1)
    if timestamps.size < 2 or not np.all(np.isfinite(timestamps)):
        raise ValueError("Trajectory timestamps must contain finite samples.")
    intervals = np.diff(timestamps)
    if np.any(intervals <= 0.0):
        raise ValueError("Trajectory timestamps must be strictly increasing.")
    interval = float(np.median(intervals))
    if not np.allclose(intervals, interval, atol=absolute_tolerance_sec, rtol=0.0):
        raise ValueError(
            "Constant-frame-rate video requires a uniform trajectory timestamp lattice."
        )
    return 1.0 / interval


def timestamped_video_samples(
    timestamps_sec: np.ndarray, video_fps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hold source qpos on a video clock; never interpolate a new robot path."""

    timestamps = np.asarray(timestamps_sec, dtype=float)
    if (timestamps.ndim != 1 or timestamps.size < 2
            or not np.isfinite(timestamps).all() or timestamps[0] != 0.0
            or np.any(np.diff(timestamps) <= 0.0)):
        raise ValueError("Video source timestamps must start at zero and increase.")
    if isinstance(video_fps, bool) or int(video_fps) != video_fps or video_fps <= 0:
        raise ValueError("video_fps must be a positive integer.")
    # The final source state appears no later than one video frame after its
    # real timestamp. Repeated frames are deliberate holds, not faster motion.
    clock = np.arange(math.ceil(timestamps[-1] * video_fps) + 1) / video_fps
    if clock[-1] < timestamps[-1]:
        clock = np.append(clock, len(clock) / video_fps)
    indices = np.searchsorted(timestamps, clock, side="right") - 1
    return clock, np.minimum(indices, timestamps.size - 1)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _annotate_frame(
    rgb: np.ndarray,
    *,
    elapsed_sec: float,
    duration_sec: float,
    sample_index: int,
    sample_count: int,
) -> np.ndarray:
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    text = (
        f"production qpos  t={elapsed_sec:07.3f}/{duration_sec:07.3f}s  "
        f"sample={sample_index + 1}/{sample_count}"
    )
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(
        bgr,
        text,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return bgr


def record_timestamped_trajectory(
    *,
    trajectory_record: Path,
    output_dir: Path,
    scene_contract_path: Path,
    camera_name: str,
    width: int,
    height: int,
    left_gripper_close: float,
    right_gripper_close: float,
    video_fps: int | None = None,
) -> dict[str, object]:
    """Replay source qpos at real time, optionally holding it on a video clock."""

    source = trajectory_record.expanduser().resolve(strict=True)
    destination = output_dir.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"Output directory already exists: {destination}")
    if width <= 0 or height <= 0:
        raise ValueError("Video width and height must be positive.")
    for name, value in (
        ("left_gripper_close", left_gripper_close),
        ("right_gripper_close", right_gripper_close),
    ):
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1].")

    trajectory, metadata = load_grasp_trajectory_record(source)
    if video_fps is None:
        source_hz = infer_uniform_sample_hz(trajectory.time_from_start_sec)
        rounded_hz = round(source_hz)
        if not math.isclose(source_hz, rounded_hz, abs_tol=1e-9, rel_tol=0.0):
            raise ValueError(
                "Trajectory sample rate must be an integer for MP4 timestamp preservation."
            )
        fps = int(rounded_hz)
        video_times = trajectory.time_from_start_sec
        video_indices = np.arange(video_times.size)
    else:
        source_hz = None
        fps = video_fps
        video_times, video_indices = timestamped_video_samples(
            trajectory.time_from_start_sec, fps
        )
    contract_path = resolve_path(scene_contract_path)
    contract = SimulationContract.load(contract_path)
    destination.mkdir(parents=True)
    video_path = destination / "production_trajectory_original_timestamps.mp4"

    scene = MujocoScene.from_contract(
        contract,
        mjcf_path=destination / "scene.mjcf",
        position_actuation_profile=rm65_bimanual_udp_feedback_profile(),
        gripper_actuation_profile=cad_bimanual_dmtac_gripper_profile(),
    )
    renderer = MujocoCameraRenderer(scene, width=width, height=height)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        renderer.close()
        raise RuntimeError(f"Could not open timestamped trajectory video: {video_path}")

    gripper_profile = cad_bimanual_dmtac_gripper_profile()
    gripper_close_by_side = {
        "left": float(left_gripper_close),
        "right": float(right_gripper_close),
    }
    try:
        for gripper_name, side in gripper_profile.side_by_gripper.items():
            scene.set_gripper_close(gripper_name, gripper_close_by_side[side])
        sample_count = int(trajectory.time_from_start_sec.size)
        duration_sec = float(trajectory.time_from_start_sec[-1])
        for elapsed_sec, index in zip(video_times, video_indices):
            scene.set_qpos_by_side(
                {
                    side: trajectory.qpos_by_side[side][index]
                    for side in ("left", "right")
                }
            )
            scene.data.qvel[:] = 0.0
            scene.forward()
            writer.write(
                _annotate_frame(
                    renderer.render(camera_name),
                    elapsed_sec=float(elapsed_sec),
                    duration_sec=duration_sec,
                    sample_index=index,
                    sample_count=sample_count,
                )
            )
    finally:
        writer.release()
        renderer.close()

    encoded_duration_sec = len(video_times) / float(fps)
    report: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "recorded",
        "non_authoritative": True,
        "execution_mode": "kinematic_exact_planned_qpos_visualization",
        "trajectory_record": str(source),
        "trajectory_hash": trajectory.semantic_content_hash,
        "trajectory_parameterization": metadata.get("parameterization"),
        "source_sample_count": sample_count,
        "source_sample_hz": source_hz,
        "source_duration_sec": duration_sec,
        "video_fps": fps,
        "video_frame_count": len(video_times),
        "video_encoded_duration_sec": encoded_duration_sec,
        "video_final_frame_timestamp_sec": float(video_times[-1]),
        "video_sampling": "all_source_samples" if video_fps is None else "previous_source_sample_hold",
        "maximum_source_hold_age_sec": float(np.max(
            video_times - trajectory.time_from_start_sec[video_indices]
        )),
        "video_terminal_frame_hold_sec": 1.0 / fps,
        "playback_rate": 1.0,
        "camera_name": camera_name,
        "resolution": [int(width), int(height)],
        "scene_contract": str(contract_path),
        "scene_contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "video": str(video_path),
        "gripper_close_by_side": gripper_close_by_side,
    }
    _write_json(destination / "timing_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record a saved production grasp trajectory in MuJoCo using the "
            "trajectory's original uniform timestamp lattice. Never connects to hardware."
        )
    )
    parser.add_argument("--trajectory-record", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scene-contract",
        type=Path,
        default=DEFAULT_BLUE_CUBE_CONTRACT,
    )
    parser.add_argument("--camera", default="front")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--left-gripper-close", type=float, default=0.2)
    parser.add_argument("--right-gripper-close", type=float, default=0.3)
    parser.add_argument("--video-fps", type=int,
                        help="Real-time video clock for nonuniform source timestamps; holds exact source qpos")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = record_timestamped_trajectory(
            trajectory_record=args.trajectory_record,
            output_dir=args.output_dir,
            scene_contract_path=args.scene_contract,
            camera_name=args.camera,
            width=args.width,
            height=args.height,
            left_gripper_close=args.left_gripper_close,
            right_gripper_close=args.right_gripper_close,
            video_fps=args.video_fps,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"Timestamped MuJoCo recording failed: {exc}") from exc
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
