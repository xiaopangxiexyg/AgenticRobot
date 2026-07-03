from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from utils.handeye.calibration import load_result
from utils.handeye.camera import RealSenseCamera, RealSenseConfig
from utils.handeye.dual_arm import (
    DEFAULT_DUAL_ARM_URDF,
    derive_camera_to_arm_base,
    load_dual_arm_base_transforms,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize RealSense point cloud in RM65 base frame.")
    parser.add_argument("--calibration", required=True, help="Calibration JSON/NPZ/Npy file.")
    parser.add_argument("--calibration-arm", choices=("left", "right"), default="")
    parser.add_argument("--view-arm", choices=("left", "right"), default="")
    parser.add_argument("--dual-arm-urdf", default=DEFAULT_DUAL_ARM_URDF)
    parser.add_argument("--camera", "--camera-index", dest="camera_index", type=int, default=None)
    parser.add_argument("--camera-serial", default=None)
    parser.add_argument("--resolution", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--viewer", choices=("cv2", "matplotlib"), default="cv2")
    parser.add_argument("--depth-trunc-m", type=float, default=2.5)
    parser.add_argument("--initial-cloud-timeout-sec", type=float, default=5.0)
    parser.add_argument("--max-points", type=int, default=25000)
    parser.add_argument("--update-interval-sec", type=float, default=0.2)
    parser.add_argument("--projection-size-px", type=int, default=420)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = load_result(args.calibration)
    T_camera_to_base = result.T_camera_to_base
    if args.view_arm:
        if not args.calibration_arm:
            raise ValueError("--calibration-arm is required when --view-arm is set.")
        dual_arm = load_dual_arm_base_transforms(args.dual_arm_urdf)
        T_camera_to_base = derive_camera_to_arm_base(
            T_camera_to_base,
            source_arm=args.calibration_arm,
            target_arm=args.view_arm,
            dual_arm=dual_arm,
        )
        print(
            "Using dual-arm URDF transform: "
            f"{args.calibration_arm}_base -> {args.view_arm}_base from {dual_arm.urdf_path}"
        )
    print("Loaded T_camera_to_base:")
    print(np.array2string(T_camera_to_base, precision=8, suppress_small=False))

    camera_config = RealSenseConfig(
        serial=args.camera_serial,
        camera_index=args.camera_index,
        resolution=tuple(args.resolution),
        fps=args.fps,
        enable_depth=True,
    )
    with RealSenseCamera(camera_config) as camera:
        frame = camera.read()
        viewer_kwargs = dict(
            args=args,
            camera=camera,
            T_camera_to_base=T_camera_to_base,
            intrinsics_matrix=frame.intrinsics_matrix,
            depth_scale_m=float(frame.depth_scale_m or 0.001),
            initial_frame=frame,
        )
        if args.viewer == "matplotlib":
            _run_matplotlib_viewer(**viewer_kwargs)
        else:
            _run_cv2_viewer(**viewer_kwargs)
    return 0


def _run_cv2_viewer(
    args: argparse.Namespace,
    camera: RealSenseCamera,
    T_camera_to_base: np.ndarray,
    intrinsics_matrix: np.ndarray,
    *,
    depth_scale_m: float,
    initial_frame=None,
) -> None:
    cv2 = _require_cv2()
    points, colors = _read_non_empty_numpy_cloud(
        camera,
        intrinsics_matrix,
        depth_scale_m=depth_scale_m,
        depth_trunc_m=float(args.depth_trunc_m),
        max_points=int(args.max_points),
        timeout_sec=float(args.initial_cloud_timeout_sec),
        initial_frame=initial_frame,
    )
    window_name = "RM65 hand-eye projections"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("OpenCV viewer: q/Esc closes the window.")
    last_update = 0.0
    while True:
        now = time.monotonic()
        if now - last_update >= max(0.01, float(args.update_interval_sec)):
            points_base = _transform_points(T_camera_to_base, points)
            image = _draw_projection_views(
                cv2,
                points_base,
                colors,
                T_camera_to_base=T_camera_to_base,
                panel_size=max(160, int(args.projection_size_px)),
            )
            cv2.imshow(window_name, image)
            last_update = now

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

        frame = camera.read()
        if frame.depth_z16 is None:
            continue
        next_points, next_colors = _depth_to_numpy_cloud(
            frame.color_bgr,
            frame.depth_z16,
            intrinsics_matrix=intrinsics_matrix,
            depth_scale_m=depth_scale_m,
            depth_trunc_m=float(args.depth_trunc_m),
            max_points=int(args.max_points),
        )
        if len(next_points) > 0:
            points, colors = next_points, next_colors
    cv2.destroyWindow(window_name)


def _run_matplotlib_viewer(
    args: argparse.Namespace,
    camera: RealSenseCamera,
    T_camera_to_base: np.ndarray,
    intrinsics_matrix: np.ndarray,
    *,
    depth_scale_m: float,
    initial_frame=None,
) -> None:
    plt = _require_matplotlib()
    points, colors = _read_non_empty_numpy_cloud(
        camera,
        intrinsics_matrix,
        depth_scale_m=depth_scale_m,
        depth_trunc_m=float(args.depth_trunc_m),
        max_points=int(args.max_points),
        timeout_sec=float(args.initial_cloud_timeout_sec),
        initial_frame=initial_frame,
    )

    plt.ion()
    fig = plt.figure("RM65 hand-eye point cloud", figsize=(9.6, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    print("Close the Matplotlib window or press Ctrl-C to quit.")
    while plt.fignum_exists(fig.number):
        points_base = _transform_points(T_camera_to_base, points)
        ax.cla()
        ax.scatter(
            points_base[:, 0],
            points_base[:, 1],
            points_base[:, 2],
            c=colors,
            s=1,
            depthshade=False,
        )
        _plot_frame_axes(ax, np.eye(4), size=0.18, label="base")
        _plot_frame_axes(ax, T_camera_to_base, size=0.12, label="camera")
        _set_axes_from_points(ax, points_base)
        ax.set_xlabel("base X (m)")
        ax.set_ylabel("base Y (m)")
        ax.set_zlabel("base Z (m)")
        ax.set_title(f"RM65 hand-eye point cloud ({len(points_base)} points)")
        fig.canvas.draw_idle()
        plt.pause(max(0.01, float(args.update_interval_sec)))

        frame = camera.read()
        if frame.depth_z16 is None:
            continue
        next_points, next_colors = _depth_to_numpy_cloud(
            frame.color_bgr,
            frame.depth_z16,
            intrinsics_matrix=intrinsics_matrix,
            depth_scale_m=depth_scale_m,
            depth_trunc_m=float(args.depth_trunc_m),
            max_points=int(args.max_points),
        )
        if len(next_points) > 0:
            points, colors = next_points, next_colors


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


def _read_non_empty_numpy_cloud(
    camera: RealSenseCamera,
    intrinsics_matrix: np.ndarray,
    *,
    depth_scale_m: float,
    depth_trunc_m: float,
    max_points: int,
    timeout_sec: float,
    initial_frame=None,
) -> tuple[np.ndarray, np.ndarray]:
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    frame = initial_frame
    print("Waiting for a non-empty depth point cloud...")
    while time.monotonic() < deadline:
        if frame is None:
            frame = camera.read()
        if frame.depth_z16 is not None:
            points, colors = _depth_to_numpy_cloud(
                frame.color_bgr,
                frame.depth_z16,
                intrinsics_matrix=intrinsics_matrix,
                depth_scale_m=depth_scale_m,
                depth_trunc_m=depth_trunc_m,
                max_points=max_points,
            )
            if len(points) > 0:
                print(f"Initial cloud points: {len(points)}")
                return points, colors
        frame = None
    raise RuntimeError(
        "no non-empty RealSense depth point cloud was available. "
        "Check depth stream, scene distance, USB bandwidth, or increase --depth-trunc-m."
    )


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _plot_frame_axes(ax, transform: np.ndarray, *, size: float, label: str) -> None:
    origin = transform[:3, 3]
    axes = transform[:3, :3] * float(size)
    colors = ("r", "g", "b")
    names = ("x", "y", "z")
    for index, color in enumerate(colors):
        end = origin + axes[:, index]
        ax.plot(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            color=color,
            linewidth=2,
        )
        ax.text(end[0], end[1], end[2], f"{label}.{names[index]}", color=color)


def _set_axes_from_points(ax, points: np.ndarray) -> None:
    mins = np.percentile(points, 2.0, axis=0)
    maxs = np.percentile(points, 98.0, axis=0)
    center = (mins + maxs) * 0.5
    radius = max(0.1, float(np.max(maxs - mins)) * 0.6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


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
    xy = _project_points_to_panel(points, dims=dims, center=center, radius=radius, panel_size=panel_size, margin=margin)
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
    origin_px = _project_points_to_panel(origin.reshape(1, 3), dims=dims, center=center, radius=radius, panel_size=panel_size, margin=margin)[0]
    for index, color in enumerate(colors):
        end = origin + axes[:, index]
        end_px = _project_points_to_panel(end.reshape(1, 3), dims=dims, center=center, radius=radius, panel_size=panel_size, margin=margin)[0]
        cv2.line(image, tuple(origin_px), tuple(end_px), color, 2, cv2.LINE_AA)
        cv2.putText(image, f"{name}.{labels[index]}", tuple(end_px), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)


def _require_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on calibration env
        raise ImportError("matplotlib is required for hand-eye point-cloud visualization.") from exc
    return plt


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on calibration env
        raise ImportError("opencv-python is required for the default hand-eye viewer.") from exc
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
