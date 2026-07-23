from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import sys
import time
from typing import Iterable, Sequence

import numpy as np

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    __package__ = "utils.portable_vis.sxr_ego_head_pose"

from utils.head_eye.transforms import as_transform, invert_transform, make_transform


HEAD_POSE_TOPIC = "head_pose"
HEAD_POSE_MESSAGE_SIZE = 36
CAMERA_GROUPS = ("rgb", "tracking", "ctrl")
CAMERA_OPTICAL_TO_CALIBRATION_ROTATION = np.array(
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class HeadPoseSample:
    timestamp_ns: int
    position: np.ndarray
    quaternion: np.ndarray


@dataclass(frozen=True)
class CameraExtrinsic:
    name: str
    T_optical_camera_to_imu: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay an SXR Ego MCAP head pose at recorded speed and visualize the "
            "dynamic IMU/head frame plus calibrated camera frames."
        )
    )
    parser.add_argument(
        "episode",
        help="Episode directory, or a sensor.mcap path when --calibration is supplied.",
    )
    parser.add_argument(
        "--calibration",
        default="",
        help="Calibration JSON. Defaults to <episode>/calibration.json.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--max-fps", type=float, default=60.0, help="Maximum display refresh rate.")
    parser.add_argument(
        "--frame-mode",
        choices=("first", "world"),
        default="first",
        help="'first' displays poses relative to the first head pose; 'world' keeps recorded coordinates.",
    )
    parser.add_argument(
        "--quaternion-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Interpretation of the four recorded quaternion floats.",
    )
    parser.add_argument(
        "--pose-direction",
        choices=("world_from_imu", "imu_from_world"),
        default="world_from_imu",
        help="Direction represented by each recorded head pose.",
    )
    parser.add_argument(
        "--camera-groups",
        nargs="*",
        choices=CAMERA_GROUPS,
        default=list(CAMERA_GROUPS),
        help="Camera groups to display. Pass the option without values to show no cameras.",
    )
    parser.add_argument("--head-axis-size", type=float, default=0.08)
    parser.add_argument("--camera-axis-size", type=float, default=0.045)
    parser.add_argument(
        "--trail-seconds",
        type=float,
        default=8.0,
        help="Visible head-position trail duration; <=0 keeps the complete trail.",
    )
    parser.add_argument("--loop", action="store_true", help="Loop playback until the window closes.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Decode and validate inputs, print the resolved setup, and do not open a window.",
    )
    return parser


def decode_head_pose_message(data: bytes, *, quaternion_order: str = "xyzw") -> HeadPoseSample:
    if len(data) != HEAD_POSE_MESSAGE_SIZE:
        raise ValueError(
            f"head_pose payload must be {HEAD_POSE_MESSAGE_SIZE} bytes, got {len(data)}."
        )
    timestamp_ns, px, py, pz, q0, q1, q2, q3 = struct.unpack("<q7f", data)
    quaternion = np.array([q0, q1, q2, q3], dtype=np.float64)
    if quaternion_order == "wxyz":
        quaternion = quaternion[[1, 2, 3, 0]]
    elif quaternion_order != "xyzw":
        raise ValueError(f"unsupported quaternion order: {quaternion_order!r}")
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("head_pose quaternion is invalid.")
    return HeadPoseSample(
        timestamp_ns=int(timestamp_ns),
        position=np.array([px, py, pz], dtype=np.float64),
        quaternion=quaternion / norm,
    )


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm([x, y, z, w]))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("quaternion must have a finite non-zero norm.")
    x, y, z, w = np.array([x, y, z, w], dtype=np.float64) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def sample_to_transform(sample: HeadPoseSample, *, pose_direction: str) -> np.ndarray:
    transform = make_transform(
        quaternion_xyzw_to_matrix(sample.quaternion),
        sample.position,
    )
    if pose_direction == "world_from_imu":
        return transform
    if pose_direction == "imu_from_world":
        return invert_transform(transform)
    raise ValueError(f"unsupported pose direction: {pose_direction!r}")


def resolve_input_paths(episode: str, calibration: str = "") -> tuple[Path, Path]:
    episode_path = Path(episode).expanduser().resolve()
    if episode_path.is_dir():
        mcap_path = episode_path / "sensor.mcap"
        calibration_path = (
            Path(calibration).expanduser().resolve()
            if calibration
            else episode_path / "calibration.json"
        )
    else:
        mcap_path = episode_path
        if not calibration:
            raise ValueError("--calibration is required when episode is an MCAP file.")
        calibration_path = Path(calibration).expanduser().resolve()
    if not mcap_path.is_file():
        raise FileNotFoundError(f"MCAP file not found: {mcap_path}")
    if not calibration_path.is_file():
        raise FileNotFoundError(f"calibration file not found: {calibration_path}")
    return mcap_path, calibration_path


def load_head_poses(
    mcap_path: Path,
    *,
    quaternion_order: str,
) -> list[HeadPoseSample]:
    try:
        from mcap.reader import NonSeekingReader
    except ImportError as exc:
        raise RuntimeError(
            "The 'mcap' package is required. Run this utility in the EgoData environment "
            "or install the Python MCAP reader."
        ) from exc

    samples: list[HeadPoseSample] = []
    with mcap_path.open("rb") as stream:
        reader = NonSeekingReader(stream, validate_crcs=True)
        for _, channel, message in reader.iter_messages(
            topics=[HEAD_POSE_TOPIC],
            log_time_order=False,
        ):
            sample = decode_head_pose_message(
                message.data,
                quaternion_order=quaternion_order,
            )
            if sample.timestamp_ns != message.log_time:
                raise ValueError(
                    "head_pose payload timestamp does not match the MCAP log timestamp "
                    f"at sequence {message.sequence}."
                )
            samples.append(sample)
    if not samples:
        raise ValueError(f"no {HEAD_POSE_TOPIC!r} messages found in {mcap_path}.")
    timestamps = np.array([sample.timestamp_ns for sample in samples], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("head_pose timestamps are not strictly increasing.")
    return samples


def load_camera_extrinsics(
    calibration_path: Path,
    *,
    camera_groups: Iterable[str],
) -> list[CameraExtrinsic]:
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    images = payload.get("observation", {}).get("images", {})
    T_optical_to_calibration_camera = make_transform(
        CAMERA_OPTICAL_TO_CALIBRATION_ROTATION,
        [0.0, 0.0, 0.0],
    )
    resolved: list[CameraExtrinsic] = []
    for group in camera_groups:
        if group not in images:
            raise ValueError(f"camera group {group!r} is absent from {calibration_path}.")
        extrinsics = images[group].get("extrinsics", {})
        for camera_name in ("cam0", "cam1"):
            key = f"T_ic_{camera_name}_to_imu0"
            if key not in extrinsics:
                raise ValueError(f"missing observation.images.{group}.extrinsics.{key}")
            resolved.append(
                CameraExtrinsic(
                    name=f"{group}/{camera_name}",
                    T_optical_camera_to_imu=(
                        as_transform(extrinsics[key], name=key)
                        @ T_optical_to_calibration_camera
                    ),
                )
            )
    return resolved


def display_transforms(
    samples: Sequence[HeadPoseSample],
    *,
    pose_direction: str,
    frame_mode: str,
) -> list[np.ndarray]:
    world_from_imu = [
        sample_to_transform(sample, pose_direction=pose_direction) for sample in samples
    ]
    if frame_mode == "world":
        return world_from_imu
    if frame_mode == "first":
        first_from_world = invert_transform(world_from_imu[0])
        return [first_from_world @ transform for transform in world_from_imu]
    raise ValueError(f"unsupported frame mode: {frame_mode!r}")


def _frame_segments(transform: np.ndarray, axis_size: float) -> list[np.ndarray]:
    origin = transform[:3, 3]
    return [
        np.vstack((origin, origin + transform[:3, axis] * axis_size))
        for axis in range(3)
    ]


def _set_line_3d(line, points: np.ndarray) -> None:
    line.set_data(points[:, 0], points[:, 1])
    line.set_3d_properties(points[:, 2])


def _equal_limits(points: np.ndarray, padding: float) -> tuple[np.ndarray, float]:
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    center = (low + high) * 0.5
    half_extent = max(float(np.max(high - low)) * 0.55, float(padding), 1e-3)
    return center, half_extent


def run_visualization(
    samples: Sequence[HeadPoseSample],
    transforms: Sequence[np.ndarray],
    cameras: Sequence[CameraExtrinsic],
    *,
    speed: float,
    max_fps: float,
    head_axis_size: float,
    camera_axis_size: float,
    trail_seconds: float,
    loop: bool,
    frame_mode: str,
) -> None:
    if speed <= 0:
        raise ValueError("--speed must be greater than zero.")
    if max_fps <= 0:
        raise ValueError("--max-fps must be greater than zero.")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for the interactive viewer. "
            "Run this utility in the EgoData environment."
        ) from exc

    timestamps = np.array([sample.timestamp_ns for sample in samples], dtype=np.int64)
    elapsed = (timestamps - timestamps[0]).astype(np.float64) / 1e9
    positions = np.stack([transform[:3, 3] for transform in transforms])
    camera_positions = [
        transform @ camera.T_optical_camera_to_imu
        for transform in (transforms[0], transforms[-1])
        for camera in cameras
    ]
    limit_points = positions
    if camera_positions:
        limit_points = np.vstack(
            (limit_points, np.stack([transform[:3, 3] for transform in camera_positions]))
        )
    center, half_extent = _equal_limits(
        limit_points,
        padding=max(head_axis_size, camera_axis_size) * 1.5,
    )

    plt.ion()
    fig = plt.figure("SXR Ego dynamic head/camera extrinsics", figsize=(10.5, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
    ax.set_ylim(center[1] - half_extent, center[1] + half_extent)
    ax.set_zlim(center[2] - half_extent, center[2] + half_extent)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True, alpha=0.3)

    axis_colors = ("#e53935", "#43a047", "#1e88e5")
    axis_names = ("X", "Y", "Z")
    head_lines = [
        ax.plot([], [], [], color=color, linewidth=3.0, label=f"head {name}")[0]
        for color, name in zip(axis_colors, axis_names)
    ]
    trail_line = ax.plot([], [], [], color="#222222", linewidth=1.5, alpha=0.75, label="head trail")[0]
    camera_lines: list[tuple[CameraExtrinsic, list[object]]] = []
    for camera in cameras:
        lines = [
            ax.plot(
                [],
                [],
                [],
                color=color,
                linewidth=1.2,
                alpha=0.65,
                label=camera.name if axis == 2 else None,
            )[0]
            for axis, color in enumerate(axis_colors)
        ]
        camera_lines.append((camera, lines))

    if frame_mode == "first":
        for axis, segment in enumerate(_frame_segments(np.eye(4), head_axis_size * 1.15)):
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                segment[:, 2],
                color=axis_colors[axis],
                linewidth=1.0,
                linestyle="--",
                alpha=0.35,
            )
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    paused = False
    quit_requested = False

    def on_key(event) -> None:
        nonlocal paused, quit_requested
        if event.key in ("q", "escape"):
            quit_requested = True
            plt.close(fig)
        elif event.key == " ":
            paused = not paused

    fig.canvas.mpl_connect("key_press_event", on_key)
    print("Controls: Space pauses/resumes; q or Esc closes the viewer.")

    while plt.fignum_exists(fig.number) and not quit_requested:
        playback_start = time.monotonic()
        paused_at: float | None = None
        index = 0
        while index < len(samples) and plt.fignum_exists(fig.number) and not quit_requested:
            if paused:
                if paused_at is None:
                    paused_at = time.monotonic()
                plt.pause(0.02)
                continue
            if paused_at is not None:
                playback_start += time.monotonic() - paused_at
                paused_at = None

            playback_time = (time.monotonic() - playback_start) * speed
            next_index = int(np.searchsorted(elapsed, playback_time, side="right") - 1)
            index = max(index, min(next_index, len(samples) - 1))
            transform = transforms[index]
            for line, segment in zip(head_lines, _frame_segments(transform, head_axis_size)):
                _set_line_3d(line, segment)

            trail_start = 0
            if trail_seconds > 0:
                trail_start = int(
                    np.searchsorted(elapsed, elapsed[index] - trail_seconds, side="left")
                )
            _set_line_3d(trail_line, positions[trail_start : index + 1])

            for camera, lines in camera_lines:
                display_from_camera = transform @ camera.T_optical_camera_to_imu
                for line, segment in zip(
                    lines,
                    _frame_segments(display_from_camera, camera_axis_size),
                ):
                    _set_line_3d(line, segment)

            position = transform[:3, 3]
            ax.set_title(
                f"SXR Ego head/camera extrinsics  t={elapsed[index]:.3f}s  "
                f"sample={index + 1}/{len(samples)}\\n"
                f"head position=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f})"
            )
            fig.canvas.draw_idle()
            plt.pause(max(0.001, 1.0 / max_fps))
            if index == len(samples) - 1:
                break

        if not loop:
            break
    if plt.fignum_exists(fig.number):
        print("Playback complete. Close the window, or press q/Esc.")
        plt.ioff()
        plt.show()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mcap_path, calibration_path = resolve_input_paths(args.episode, args.calibration)
    samples = load_head_poses(mcap_path, quaternion_order=args.quaternion_order)
    cameras = load_camera_extrinsics(
        calibration_path,
        camera_groups=args.camera_groups,
    )
    transforms = display_transforms(
        samples,
        pose_direction=args.pose_direction,
        frame_mode=args.frame_mode,
    )
    duration_s = (samples[-1].timestamp_ns - samples[0].timestamp_ns) / 1e9
    print(f"MCAP: {mcap_path}")
    print(f"Calibration: {calibration_path}")
    print(f"Head poses: {len(samples)} samples over {duration_s:.6f} s")
    print(
        f"Interpretation: quaternion={args.quaternion_order}, "
        f"pose={args.pose_direction}, frame={args.frame_mode}"
    )
    print("Camera frames: " + (", ".join(camera.name for camera in cameras) or "(disabled)"))
    if args.dry_run:
        return 0
    run_visualization(
        samples,
        transforms,
        cameras,
        speed=args.speed,
        max_fps=args.max_fps,
        head_axis_size=args.head_axis_size,
        camera_axis_size=args.camera_axis_size,
        trail_seconds=args.trail_seconds,
        loop=args.loop,
        frame_mode=args.frame_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
