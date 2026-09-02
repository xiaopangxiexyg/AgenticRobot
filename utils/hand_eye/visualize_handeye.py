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
    __package__ = "utils.hand_eye"

from utils.calibration_common.visualization import (
    depth_to_numpy_cloud as _depth_to_numpy_cloud,
    draw_frame_projection as _draw_frame_projection,
    draw_projection_panel as _draw_projection_panel,
    draw_projection_views as _draw_projection_views,
    project_points_to_panel as _project_points_to_panel,
    projection_bounds as _projection_bounds,
    transform_points as _transform_points,
)

from .calibration import load_result
from .camera import RealSenseCamera, RealSenseConfig
from .robot import RM65PoseReader


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
