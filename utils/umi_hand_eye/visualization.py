from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from utils.hand_eye.transforms import as_transform, invert_transform


@dataclass(frozen=True)
class UmiLiveCheckPose:
    T_tracker_to_desk: np.ndarray
    T_camera_to_desk: np.ndarray
    T_desk_to_camera: np.ndarray
    T_tracker_to_camera: np.ndarray
    tracker_xyz_m: np.ndarray
    tracker_rpy_deg: np.ndarray
    camera_xyz_m: np.ndarray
    camera_rpy_deg: np.ndarray


def compute_live_check_pose(
    result: Any,
    T_tracker_to_vive_world: object,
) -> UmiLiveCheckPose:
    tracker_to_vive = as_transform(
        T_tracker_to_vive_world,
        name="T_tracker_to_vive_world",
    )
    tracker_to_desk = result.T_vive_world_to_desk @ tracker_to_vive
    camera_to_desk = tracker_to_desk @ result.T_camera_to_tracker
    tracker_to_camera = result.T_tracker_to_camera
    return UmiLiveCheckPose(
        T_tracker_to_desk=as_transform(tracker_to_desk, name="T_tracker_to_desk"),
        T_camera_to_desk=as_transform(camera_to_desk, name="T_camera_to_desk"),
        T_desk_to_camera=invert_transform(camera_to_desk),
        T_tracker_to_camera=as_transform(
            tracker_to_camera,
            name="T_tracker_to_camera",
        ),
        tracker_xyz_m=np.asarray(tracker_to_desk[:3, 3], dtype=np.float64).copy(),
        tracker_rpy_deg=Rotation.from_matrix(tracker_to_desk[:3, :3]).as_euler(
            "xyz",
            degrees=True,
        ),
        camera_xyz_m=np.asarray(camera_to_desk[:3, 3], dtype=np.float64).copy(),
        camera_rpy_deg=Rotation.from_matrix(camera_to_desk[:3, :3]).as_euler(
            "xyz",
            degrees=True,
        ),
    )


def draw_live_check_overlay(
    cv2: Any,
    color_bgr: np.ndarray,
    *,
    intrinsics_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    pose: UmiLiveCheckPose,
    tracker_name: str,
    axis_length_m: float = 0.10,
    board_residual: tuple[float, float] | None = None,
) -> np.ndarray:
    image = np.asarray(color_bgr).copy()
    desk_visible = _draw_camera_frame_axes(
        cv2,
        image,
        intrinsics_matrix,
        distortion_coeffs,
        pose.T_desk_to_camera,
        axis_length_m=axis_length_m,
    )
    tracker_visible = _draw_camera_frame_axes(
        cv2,
        image,
        intrinsics_matrix,
        distortion_coeffs,
        pose.T_tracker_to_camera,
        axis_length_m=axis_length_m * 0.6,
    )
    _put_text_lines(
        cv2,
        image,
        [
            "LIVE CALIBRATION CHECK (q/Esc quit)",
            f"desk axes in image: {'visible' if desk_visible else 'behind camera'}",
            f"tracker axes in image: {'visible' if tracker_visible else 'outside/behind camera'}",
        ],
        origin=(12, 26),
    )
    panel = _draw_tracker_desk_panel(
        cv2,
        height=image.shape[0],
        pose=pose,
        tracker_name=tracker_name,
        board_residual=board_residual,
    )
    return np.concatenate((image, panel), axis=1)


def _draw_camera_frame_axes(
    cv2: Any,
    image: np.ndarray,
    intrinsics_matrix: np.ndarray,
    distortion_coeffs: np.ndarray,
    T_frame_to_camera: np.ndarray,
    *,
    axis_length_m: float,
) -> bool:
    if float(T_frame_to_camera[2, 3]) <= 1e-6:
        return False
    rvec, _ = cv2.Rodrigues(np.asarray(T_frame_to_camera[:3, :3], dtype=np.float64))
    cv2.drawFrameAxes(
        image,
        np.asarray(intrinsics_matrix, dtype=np.float64),
        np.asarray(distortion_coeffs, dtype=np.float64),
        rvec.reshape(3, 1),
        np.asarray(T_frame_to_camera[:3, 3], dtype=np.float64).reshape(3, 1),
        float(axis_length_m),
        2,
    )
    return True


def _draw_tracker_desk_panel(
    cv2: Any,
    *,
    height: int,
    pose: UmiLiveCheckPose,
    tracker_name: str,
    board_residual: tuple[float, float] | None,
) -> np.ndarray:
    width = 360
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    xyz = pose.tracker_xyz_m
    rpy = pose.tracker_rpy_deg
    camera_xyz = pose.camera_xyz_m
    camera_rpy = pose.camera_rpy_deg
    lines = [
        f"tracker: {tracker_name}",
        "pose frame: desk",
        f"tracker xyz m: {xyz[0]:+.4f} {xyz[1]:+.4f} {xyz[2]:+.4f}",
        f"tracker rpy deg: {rpy[0]:+.1f} {rpy[1]:+.1f} {rpy[2]:+.1f}",
        f"camera xyz m: {camera_xyz[0]:+.4f} {camera_xyz[1]:+.4f} {camera_xyz[2]:+.4f}",
        f"camera rpy deg: {camera_rpy[0]:+.1f} {camera_rpy[1]:+.1f} {camera_rpy[2]:+.1f}",
    ]
    if board_residual is None:
        lines.append("board residual: target not detected")
    else:
        lines.append(
            f"board residual: {board_residual[0] * 1000.0:.1f} mm / "
            f"{board_residual[1]:.2f} deg"
        )
    _put_text_lines(cv2, panel, lines, origin=(14, 28))

    plot_top = min(220, max(190, height // 3))
    plot_bottom = height - 34
    if plot_bottom <= plot_top + 80:
        return panel
    center = np.array([width // 2, (plot_top + plot_bottom) // 2], dtype=float)
    half_extent_px = min(width * 0.42, (plot_bottom - plot_top) * 0.42)
    radius_m = max(
        0.25,
        float(np.max(np.abs(np.concatenate((xyz[:2], camera_xyz[:2]))))) * 1.35,
    )
    scale = half_extent_px / radius_m

    cv2.rectangle(
        panel,
        (12, plot_top),
        (width - 12, plot_bottom),
        (75, 75, 75),
        1,
    )
    origin = tuple(np.rint(center).astype(int))
    x_end = (int(center[0] + half_extent_px), int(center[1]))
    y_end = (int(center[0]), int(center[1] - half_extent_px))
    cv2.arrowedLine(panel, origin, x_end, (0, 0, 255), 2, tipLength=0.08)
    cv2.arrowedLine(panel, origin, y_end, (0, 220, 0), 2, tipLength=0.08)
    cv2.putText(panel, "desk.x", x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)
    cv2.putText(panel, "desk.y", y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1)

    tracker_px = np.array(
        [center[0] + xyz[0] * scale, center[1] - xyz[1] * scale],
        dtype=float,
    )
    tracker_origin = tuple(np.rint(tracker_px).astype(int))
    cv2.circle(panel, tracker_origin, 6, (255, 220, 40), -1, cv2.LINE_AA)
    tracker_axis_length = min(55.0, half_extent_px * 0.35)
    for axis_index, color in ((0, (0, 0, 255)), (1, (0, 220, 0))):
        direction = pose.T_tracker_to_desk[:2, axis_index]
        end = tracker_px + np.array(
            [direction[0], -direction[1]],
            dtype=float,
        ) * tracker_axis_length
        cv2.arrowedLine(
            panel,
            tracker_origin,
            tuple(np.rint(end).astype(int)),
            color,
            2,
            tipLength=0.16,
        )
    camera_px = np.array(
        [center[0] + camera_xyz[0] * scale, center[1] - camera_xyz[1] * scale],
        dtype=float,
    )
    camera_origin = tuple(np.rint(camera_px).astype(int))
    cv2.circle(panel, camera_origin, 6, (255, 80, 255), -1, cv2.LINE_AA)
    for axis_index, color in ((0, (0, 0, 255)), (1, (0, 220, 0))):
        direction = pose.T_camera_to_desk[:2, axis_index]
        end = camera_px + np.array(
            [direction[0], -direction[1]],
            dtype=float,
        ) * tracker_axis_length
        cv2.arrowedLine(
            panel,
            camera_origin,
            tuple(np.rint(end).astype(int)),
            color,
            2,
            tipLength=0.16,
        )
    cv2.putText(
        panel,
        "C",
        (camera_origin[0] + 8, camera_origin[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 80, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "T",
        (tracker_origin[0] + 8, tracker_origin[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 220, 40),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "desk XY: T=tracker C=camera",
        (14, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return panel


def _put_text_lines(
    cv2: Any,
    image: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int],
) -> None:
    x, first_y = origin
    line_height = 24
    box_width = min(image.shape[1] - x - 4, max(280, max(len(line) for line in lines) * 9))
    box_height = len(lines) * line_height + 8
    cv2.rectangle(
        image,
        (x - 6, first_y - 20),
        (x + box_width, first_y - 20 + box_height),
        (0, 0, 0),
        -1,
    )
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x, first_y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
