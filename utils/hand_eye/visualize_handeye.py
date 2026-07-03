from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from utils.hand_eye.calibration import load_result
from utils.hand_eye.camera import RealSenseCamera, RealSenseConfig
from utils.hand_eye.robot import RM65PoseReader


LEFT_ARM_IP = "192.168.101.21"
RIGHT_ARM_IP = "192.168.1.20"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize a wrist-mounted RealSense point cloud in RM65 base frame."
    )
    parser.add_argument("--calibration", required=True, help="Eye-in-hand JSON/NPZ/Npy file.")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--arm-ip", default="", help="Override the IP selected by --arm.")
    parser.add_argument("--expected-tool", default="", help="Optional active RM65 tool-frame name check.")
    parser.add_argument("--robot-retry", type=int, default=2)
    parser.add_argument("--camera", "--camera-index", dest="camera_index", type=int, default=None)
    parser.add_argument("--camera-serial", default=None)
    parser.add_argument("--resolution", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--depth-trunc-m", type=float, default=2.5)
    parser.add_argument("--max-points", type=int, default=25000)
    parser.add_argument("--projection-size-px", type=int, default=420)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = load_result(args.calibration)
    print("Loaded T_camera_to_gripper:")
    print(np.array2string(result.T_camera_to_gripper, precision=8, suppress_small=False))
    print("Loaded T_gripper_to_camera:")
    print(np.array2string(result.T_gripper_to_camera, precision=8, suppress_small=False))

    camera_config = RealSenseConfig(
        serial=args.camera_serial,
        camera_index=args.camera_index,
        resolution=tuple(args.resolution),
        fps=args.fps,
        enable_depth=True,
    )
    arm_ip = _selected_arm_ip(args)
    with RealSenseCamera(camera_config) as camera, RM65PoseReader(
        arm_ip,
        retry=args.robot_retry,
        expected_tool=args.expected_tool,
    ) as robot:
        tool_text = f"; tool={robot.current_tool}" if robot.current_tool else ""
        print(f"Connected RM65 arm={args.arm} ip={arm_ip}{tool_text}; camera serial={camera.serial or 'auto'}")
        _run_cv2_viewer(args=args, camera=camera, robot=robot, T_camera_to_gripper=result.T_camera_to_gripper)
    return 0


def _run_cv2_viewer(
    *,
    args: argparse.Namespace,
    camera: RealSenseCamera,
    robot: RM65PoseReader,
    T_camera_to_gripper: np.ndarray,
) -> None:
    cv2 = _require_cv2()
    window_name = "RM65 wrist camera point cloud"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("OpenCV viewer: q/Esc closes the window.")
    last_cloud_time = 0.0
    cached_points = np.empty((0, 3), dtype=np.float64)
    cached_colors = np.empty((0, 3), dtype=np.float64)

    while True:
        frame = camera.read()
        if frame.depth_z16 is not None:
            points, colors = _depth_to_numpy_cloud(
                frame.color_bgr,
                frame.depth_z16,
                intrinsics_matrix=frame.intrinsics_matrix,
                depth_scale_m=float(frame.depth_scale_m or 0.001),
                depth_trunc_m=float(args.depth_trunc_m),
                max_points=int(args.max_points),
            )
            if len(points) > 0:
                cached_points = points
                cached_colors = colors
                last_cloud_time = time.monotonic()

        robot_sample = robot.read()
        T_camera_to_base = robot_sample.T_gripper_to_base @ T_camera_to_gripper
        if len(cached_points) > 0:
            points_base = _transform_points(T_camera_to_base, cached_points)
            image = _draw_projection_views(
                cv2,
                points_base,
                cached_colors,
                T_camera_to_base=T_camera_to_base,
                panel_size=max(160, int(args.projection_size_px)),
            )
            _draw_status(cv2, image, len(points_base), last_cloud_time)
        else:
            image = np.full((260, 720, 3), 24, dtype=np.uint8)
            cv2.putText(
                image,
                "waiting for non-empty depth point cloud",
                (24, 136),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
    cv2.destroyWindow(window_name)


def _depth_to_numpy_cloud(
    color_bgr: np.ndarray,
    depth_z16: np.ndarray,
    *,
    intrinsics_matrix: np.ndarray,
    depth_scale_m: float,
    depth_trunc_m: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    depth_m = np.asarray(depth_z16, dtype=np.float64) * float(depth_scale_m)
    valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= float(depth_trunc_m))
    valid_indices = np.flatnonzero(valid.reshape(-1))
    if valid_indices.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    max_points = max(1, int(max_points))
    if valid_indices.size > max_points:
        pick = np.linspace(0, valid_indices.size - 1, max_points).astype(np.int64)
        valid_indices = valid_indices[pick]

    height, width = depth_m.shape
    rows = valid_indices // width
    cols = valid_indices % width
    z = depth_m.reshape(-1)[valid_indices]
    fx = float(intrinsics_matrix[0, 0])
    fy = float(intrinsics_matrix[1, 1])
    cx = float(intrinsics_matrix[0, 2])
    cy = float(intrinsics_matrix[1, 2])
    x = (cols.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    points = np.column_stack((x, y, z))
    color_rgb = np.asarray(color_bgr[:, :, ::-1], dtype=np.float64).reshape(height * width, 3)
    colors = color_rgb[valid_indices] / 255.0
    return points, colors


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _draw_projection_views(
    cv2,
    points_base: np.ndarray,
    colors_rgb: np.ndarray,
    *,
    T_camera_to_base: np.ndarray,
    panel_size: int,
) -> np.ndarray:
    bounds = _projection_bounds(points_base)
    panels = [
        _draw_projection_panel(
            cv2,
            points_base,
            colors_rgb,
            dims=(0, 1),
            labels=("X", "Y"),
            title="base XY",
            bounds=bounds,
            panel_size=panel_size,
            T_camera_to_base=T_camera_to_base,
        ),
        _draw_projection_panel(
            cv2,
            points_base,
            colors_rgb,
            dims=(0, 2),
            labels=("X", "Z"),
            title="base XZ",
            bounds=bounds,
            panel_size=panel_size,
            T_camera_to_base=T_camera_to_base,
        ),
        _draw_projection_panel(
            cv2,
            points_base,
            colors_rgb,
            dims=(1, 2),
            labels=("Y", "Z"),
            title="base YZ",
            bounds=bounds,
            panel_size=panel_size,
            T_camera_to_base=T_camera_to_base,
        ),
    ]
    return np.concatenate(panels, axis=1)


def _projection_bounds(points: np.ndarray) -> tuple[np.ndarray, float]:
    mins = np.percentile(points, 2.0, axis=0)
    maxs = np.percentile(points, 98.0, axis=0)
    center = (mins + maxs) * 0.5
    radius = max(0.1, float(np.max(maxs - mins)) * 0.6)
    return center, radius


def _draw_projection_panel(
    cv2,
    points: np.ndarray,
    colors_rgb: np.ndarray,
    *,
    dims: tuple[int, int],
    labels: tuple[str, str],
    title: str,
    bounds: tuple[np.ndarray, float],
    panel_size: int,
    T_camera_to_base: np.ndarray,
) -> np.ndarray:
    margin = 36
    image = np.full((panel_size, panel_size, 3), 24, dtype=np.uint8)
    center, radius = bounds
    xy = _project_points_to_panel(
        points,
        dims=dims,
        center=center,
        radius=radius,
        panel_size=panel_size,
        margin=margin,
    )
    colors_bgr = np.clip(colors_rgb[:, ::-1] * 255.0, 0, 255).astype(np.uint8)
    valid = (
        (xy[:, 0] >= margin)
        & (xy[:, 0] < panel_size - margin)
        & (xy[:, 1] >= margin)
        & (xy[:, 1] < panel_size - margin)
    )
    if np.any(valid):
        image[xy[valid, 1], xy[valid, 0]] = colors_bgr[valid]

    cv2.rectangle(image, (margin, margin), (panel_size - margin, panel_size - margin), (85, 85, 85), 1)
    _draw_frame_projection(cv2, image, np.eye(4), dims=dims, center=center, radius=radius, panel_size=panel_size, margin=margin, name="base")
    _draw_frame_projection(cv2, image, T_camera_to_base, dims=dims, center=center, radius=radius, panel_size=panel_size, margin=margin, name="cam")
    cv2.putText(image, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(
        image,
        f"{labels[0]} / {labels[1]}  {len(points)} pts",
        (12, panel_size - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return image


def _project_points_to_panel(
    points: np.ndarray,
    *,
    dims: tuple[int, int],
    center: np.ndarray,
    radius: float,
    panel_size: int,
    margin: int,
) -> np.ndarray:
    scale = (panel_size - 2 * margin) / (2.0 * float(radius))
    px = margin + (points[:, dims[0]] - center[dims[0]] + radius) * scale
    py = panel_size - margin - (points[:, dims[1]] - center[dims[1]] + radius) * scale
    return np.column_stack((np.rint(px), np.rint(py))).astype(np.int32)


def _draw_frame_projection(
    cv2,
    image: np.ndarray,
    transform: np.ndarray,
    *,
    dims: tuple[int, int],
    center: np.ndarray,
    radius: float,
    panel_size: int,
    margin: int,
    name: str,
) -> None:
    origin = transform[:3, 3]
    axes = transform[:3, :3] * 0.12
    colors = ((0, 0, 255), (0, 220, 0), (255, 80, 40))
    labels = ("x", "y", "z")
    origin_px = _project_points_to_panel(
        origin.reshape(1, 3),
        dims=dims,
        center=center,
        radius=radius,
        panel_size=panel_size,
        margin=margin,
    )[0]
    for index, color in enumerate(colors):
        end = origin + axes[:, index]
        end_px = _project_points_to_panel(
            end.reshape(1, 3),
            dims=dims,
            center=center,
            radius=radius,
            panel_size=panel_size,
            margin=margin,
        )[0]
        cv2.line(image, tuple(origin_px), tuple(end_px), color, 2, cv2.LINE_AA)
        cv2.putText(image, f"{name}.{labels[index]}", tuple(end_px), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)


def _draw_status(cv2, image: np.ndarray, point_count: int, last_cloud_time: float) -> None:
    age = max(0.0, time.monotonic() - last_cloud_time)
    text = f"{point_count} pts   cloud age {age:.2f}s   q quit"
    cv2.rectangle(image, (8, 8), (420, 42), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def _selected_arm_ip(args: argparse.Namespace) -> str:
    if args.arm_ip:
        return str(args.arm_ip)
    return LEFT_ARM_IP if args.arm == "left" else RIGHT_ARM_IP


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on calibration env
        raise ImportError("opencv-python is required for the hand-eye viewer.") from exc
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
