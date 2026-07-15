from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from utils.dataset_postprocess.real_undistort.intrinsics import CameraIntrinsics, require_cv2
from utils.dataset_postprocess.real_undistort.video import SizePolicy

from ..geometry import (
    SceneGeometry,
    SideGeometry,
    pose7_or_none,
    vector_or_none,
)
from .mask import (
    GEOMETRY_BACKGROUND_MASK_NAME,
    GEOMETRY_KEEP_MASK_NAME,
    GEOMETRY_MASK_FAMILY,
    STATIC_BODY_MASK_NAME,
    TABLE_GEOMETRY_MASK_NAME,
    RayGrid,
    background_mask_from_keep,
    build_ray_grid,
    combine_keep_mask,
    load_binary_mask,
    mask_stats,
    render_mask_output,
    table_keep_mask,
    table_keep_mask_polygon,
    table_perimeter_points,
)


POSE_FRAME_CHOICES = ("auto", "arm_end", "tool_tcp", "camera")
POSE_SOURCE_CHOICES = ("auto", "pose", "qpos")
OUTPUT_KIND_CHOICES = ("masked", "overlay", "background-mask", "keep-mask")
TABLE_MASK_MODE_CHOICES = ("sampled-raycast", "raycast", "polygon")


def _valid_fps(value: float) -> float:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return 30.0
    return fps if 0.0 < fps < 1000.0 else 30.0


def _open_writer(path: Path, *, codec: str, fps: float, width: int, height: int):
    cv2 = require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*str(codec)),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    return writer


@dataclass
class _PhaseTiming:
    count: int = 0
    total_sec: float = 0.0
    min_sec: float = float("inf")
    max_sec: float = 0.0

    def add(self, elapsed_sec: float) -> None:
        value = max(0.0, float(elapsed_sec))
        self.count += 1
        self.total_sec += value
        self.min_sec = min(self.min_sec, value)
        self.max_sec = max(self.max_sec, value)

    def to_dict(self, *, total_elapsed_sec: float, frame_count: int) -> dict[str, Any]:
        avg_sec = 0.0 if self.count <= 0 else self.total_sec / float(self.count)
        per_frame_sec = 0.0 if frame_count <= 0 else self.total_sec / float(frame_count)
        return {
            "count": int(self.count),
            "total_sec": float(self.total_sec),
            "avg_ms": float(avg_sec * 1000.0),
            "per_frame_ms": float(per_frame_sec * 1000.0),
            "min_ms": 0.0 if self.count <= 0 else float(self.min_sec * 1000.0),
            "max_ms": float(self.max_sec * 1000.0),
            "elapsed_fraction": 0.0 if total_elapsed_sec <= 0.0 else float(self.total_sec / total_elapsed_sec),
        }


class _TimingAccumulator:
    def __init__(self) -> None:
        self._phases: dict[str, _PhaseTiming] = {}

    def add(self, phase: str, elapsed_sec: float) -> None:
        self._phases.setdefault(str(phase), _PhaseTiming()).add(elapsed_sec)

    def to_dict(self, *, elapsed_sec: float, frame_count: int) -> dict[str, Any]:
        phases = {
            key: value.to_dict(total_elapsed_sec=elapsed_sec, frame_count=frame_count)
            for key, value in sorted(
                self._phases.items(),
                key=lambda item: item[1].total_sec,
                reverse=True,
            )
        }
        inclusive_phases = {"frame_total_tracked"}
        tracked_sec = sum(
            item.total_sec
            for key, item in self._phases.items()
            if key not in inclusive_phases
        )
        return {
            "enabled": True,
            "frame_count": int(frame_count),
            "elapsed_sec": float(elapsed_sec),
            "elapsed_per_frame_ms": 0.0 if frame_count <= 0 else float(elapsed_sec / float(frame_count) * 1000.0),
            "tracked_sec": float(tracked_sec),
            "untracked_sec": float(max(0.0, elapsed_sec - tracked_sec)),
            "inclusive_phases": sorted(inclusive_phases & set(self._phases)),
            "phases": phases,
        }


PoseCandidate = tuple[str, str | None]


def _pose_candidate_columns(side: str) -> tuple[PoseCandidate, ...]:
    return (
        (f"observation.arm.{side}.actual_tcp_pose", None),
        (f"observation.arm.{side}.current_tcp_pose", None),
        (f"observation.{side}_arm.pose", None),
        # Current real datasets name this wrist-camera pose, but the collector
        # writes the TCP pose into this field. Keep it as a late fallback and
        # interpret it through pose_frame instead of assuming camera frame.
        (f"observation.wrist_camera.{side}.pose_base", None),
    )


def _qpos_candidate_columns(side: str) -> tuple[str, ...]:
    return (
        f"observation.arm.{side}.actual_qpos",
        f"observation.arm.{side}.actual_hardware_qpos",
        f"observation.arm.{side}.current_qpos",
        f"observation.{side}_arm.qpos",
        f"action.replay.{side}_arm_joint_target",
    )


def _format_side_column(column: str | None, side: str) -> str | None:
    if column is None:
        return None
    try:
        return str(column).format(side=side)
    except Exception as exc:
        raise ValueError(f"failed to format column {column!r} with side={side!r}") from exc


def _first_pose(
    row: Mapping[str, Any],
    side: str,
    explicit_column: str | None,
) -> tuple[np.ndarray, str, str | None] | None:
    formatted_column = _format_side_column(explicit_column, side)
    columns: tuple[PoseCandidate, ...] = ((formatted_column, None),) if formatted_column else _pose_candidate_columns(side)
    for column, frame_override in columns:
        if not column:
            continue
        pose = pose7_or_none(row.get(column))
        if pose is not None:
            return pose, column, frame_override
    return None


def _first_qpos(row: Mapping[str, Any], side: str, explicit_column: str | None) -> tuple[np.ndarray, str] | None:
    formatted_column = _format_side_column(explicit_column, side)
    columns = (formatted_column,) if formatted_column else _qpos_candidate_columns(side)
    for column in columns:
        if not column:
            continue
        qpos = vector_or_none(row.get(column), length=6)
        if qpos is not None:
            return qpos, column
    return None


def _joint_values(side_geometry: SideGeometry, qpos: Sequence[float]) -> dict[str, float]:
    values = np.asarray(qpos, dtype=float).reshape(6)
    return {name: float(value) for name, value in zip(side_geometry.joint_names, values)}


@dataclass(frozen=True)
class FrameMask:
    mask_family: str
    keep_mask: np.ndarray
    table_geometry_mask: np.ndarray
    static_body_mask: np.ndarray
    background_mask: np.ndarray
    source_column: str
    source_frame: str
    T_world_camera: np.ndarray

    @property
    def table_mask(self) -> np.ndarray:
        return self.table_geometry_mask

    @property
    def body_mask(self) -> np.ndarray:
        return self.static_body_mask


@dataclass(frozen=True)
class MaskBuildConfig:
    scene: SceneGeometry
    side: str
    intrinsics: CameraIntrinsics
    body_mask_path: Path | None = None
    pose_frame: str = "auto"
    pose_source: str = "auto"
    pose_column: str | None = None
    qpos_column: str | None = None
    T_end_tool: np.ndarray | None = None
    table_margin_m: float = 0.0
    table_height_offset_m: float = 0.0
    table_mask_mode: str = "sampled-raycast"
    table_raycast_stride: int = 2
    table_edge_samples: int = 16
    table_min_depth_m: float = 0.03
    table_dilate_px: int = 8
    table_boundary_extend_px: int | None = None
    body_dilate_px: int = 0
    keep_dilate_px: int = 0

    def __post_init__(self) -> None:
        if self.side not in self.scene.sides:
            raise ValueError(f"side must be one of {tuple(self.scene.sides)}, got {self.side!r}")
        if self.pose_frame not in POSE_FRAME_CHOICES:
            raise ValueError(f"pose_frame must be one of {POSE_FRAME_CHOICES}, got {self.pose_frame!r}")
        if self.pose_source not in POSE_SOURCE_CHOICES:
            raise ValueError(f"pose_source must be one of {POSE_SOURCE_CHOICES}, got {self.pose_source!r}")
        if self.table_mask_mode not in TABLE_MASK_MODE_CHOICES:
            raise ValueError(
                f"table_mask_mode must be one of {TABLE_MASK_MODE_CHOICES}, got {self.table_mask_mode!r}"
            )
        if int(self.table_edge_samples) < 2:
            raise ValueError("table_edge_samples must be at least 2")
        if int(self.table_raycast_stride) < 1:
            raise ValueError("table_raycast_stride must be at least 1")


class WristMaskBuilder:
    def __init__(self, config: MaskBuildConfig):
        self.config = config
        self.side_geometry = config.scene.sides[config.side]
        self.ray_grid: RayGrid | None = None
        self.table_points_world = (
            table_perimeter_points(
                table=config.scene.table,
                table_margin_m=config.table_margin_m,
                table_height_offset_m=config.table_height_offset_m,
                edge_samples=config.table_edge_samples,
            )
            if config.table_mask_mode == "polygon"
            else None
        )
        self.body_mask = load_binary_mask(
            config.body_mask_path,
            width=config.intrinsics.width,
            height=config.intrinsics.height,
            label=f"{config.side} body",
        )

    def _ray_grid(self) -> RayGrid:
        if self.ray_grid is None:
            stride = 1 if self.config.table_mask_mode == "raycast" else int(self.config.table_raycast_stride)
            self.ray_grid = build_ray_grid(self.config.intrinsics, stride=stride)
        return self.ray_grid

    def _full_size_mask(self, mask: np.ndarray) -> np.ndarray:
        height = int(self.config.intrinsics.height)
        width = int(self.config.intrinsics.width)
        if mask.shape[:2] == (height, width):
            return mask
        cv2 = require_cv2()
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    def _pose_frame(self) -> str:
        frame = self.config.pose_frame
        if frame != "auto":
            return frame
        if self.config.T_end_tool is not None:
            return "tool_tcp"
        return "arm_end"

    def _resolved_pose_frame(self, frame_override: str | None) -> str:
        return frame_override or self._pose_frame()

    def _camera_pose_from_pose(self, T_base_frame: np.ndarray, *, frame: str) -> np.ndarray:
        if frame == "camera":
            T_base_camera = T_base_frame
        elif frame == "tool_tcp":
            if self.config.T_end_tool is None:
                raise RuntimeError("pose_frame=tool_tcp requires T_end_tool from a tool profile")
            T_base_camera = self.side_geometry.camera_from_base_tool_pose(
                T_base_frame,
                self.config.T_end_tool,
            )
        elif frame == "arm_end":
            T_base_camera = self.side_geometry.camera_from_base_end_pose(T_base_frame)
        else:
            raise ValueError(f"unsupported pose frame: {frame!r}")
        return self.side_geometry.world_from_base_pose(T_base_camera)

    def _camera_pose_from_qpos(self, qpos: Sequence[float]) -> np.ndarray:
        values = _joint_values(self.side_geometry, qpos)
        T_base_end = self.config.scene.urdf.transform_between(
            self.side_geometry.base_link,
            self.side_geometry.end_link,
            joint_values=values,
        )
        return self.side_geometry.world_from_base_pose(
            self.side_geometry.camera_from_base_end_pose(T_base_end)
        )

    def camera_pose_for_row(self, row: Mapping[str, Any]) -> tuple[np.ndarray, str, str]:
        source = self.config.pose_source
        if source in {"auto", "pose"}:
            pose = _first_pose(row, self.config.side, self.config.pose_column)
            if pose is not None:
                source_frame = self._resolved_pose_frame(pose[2])
                return self._camera_pose_from_pose(pose[0], frame=source_frame), pose[1], source_frame
            if source == "pose":
                raise KeyError(f"no finite pose found for {self.config.side}")
        qpos = _first_qpos(row, self.config.side, self.config.qpos_column)
        if qpos is None:
            raise KeyError(f"no finite pose or qpos found for {self.config.side}")
        return self._camera_pose_from_qpos(qpos[0]), qpos[1], "qpos_fk"

    def build(self, row: Mapping[str, Any]) -> FrameMask:
        T_world_camera, source_column, source_frame = self.camera_pose_for_row(row)
        table_mask = None
        if self.config.table_mask_mode == "polygon" and self.table_points_world is not None:
            table_mask = table_keep_mask_polygon(
                table_points_world=self.table_points_world,
                T_world_camera=T_world_camera,
                intrinsics=self.config.intrinsics,
                min_depth_m=self.config.table_min_depth_m,
            )
        if table_mask is None:
            table_mask = table_keep_mask(
                table=self.config.scene.table,
                T_world_camera=T_world_camera,
                ray_grid=self._ray_grid(),
                table_margin_m=self.config.table_margin_m,
                table_height_offset_m=self.config.table_height_offset_m,
            )
        table_mask = self._full_size_mask(table_mask)
        table_dilate_px = (
            self.config.table_dilate_px
            if self.config.table_boundary_extend_px is None
            else int(self.config.table_boundary_extend_px)
        )
        keep_mask = combine_keep_mask(
            body_mask=self.body_mask,
            table_mask=table_mask,
            body_dilate_px=self.config.body_dilate_px,
            table_dilate_px=table_dilate_px,
            keep_dilate_px=self.config.keep_dilate_px,
        )
        return FrameMask(
            mask_family=GEOMETRY_MASK_FAMILY,
            keep_mask=keep_mask,
            table_geometry_mask=table_mask,
            static_body_mask=self.body_mask,
            background_mask=background_mask_from_keep(keep_mask),
            source_column=source_column,
            source_frame=source_frame,
            T_world_camera=T_world_camera,
        )


@dataclass(frozen=True)
class VideoAdjustResult:
    input_path: Path
    output_path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    elapsed_sec: float
    output_kind: str
    pose_source_counts: dict[str, int] = field(default_factory=dict)
    source_frame_counts: dict[str, int] = field(default_factory=dict)
    keep_fraction_mean: float = 0.0
    background_fraction_mean: float = 0.0
    timing_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def processing_fps(self) -> float:
        return 0.0 if self.elapsed_sec <= 0.0 else float(self.frame_count) / float(self.elapsed_sec)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "mask_family": GEOMETRY_MASK_FAMILY,
            "mask_names": {
                "table": TABLE_GEOMETRY_MASK_NAME,
                "body": STATIC_BODY_MASK_NAME,
                "keep": GEOMETRY_KEEP_MASK_NAME,
                "background": GEOMETRY_BACKGROUND_MASK_NAME,
            },
            "frame_count": int(self.frame_count),
            "fps": float(self.fps),
            "width": int(self.width),
            "height": int(self.height),
            "elapsed_sec": float(self.elapsed_sec),
            "processing_fps": float(self.processing_fps),
            "output_kind": self.output_kind,
            "pose_source_counts": dict(self.pose_source_counts),
            "source_frame_counts": dict(self.source_frame_counts),
            "keep_fraction_mean": float(self.keep_fraction_mean),
            "background_fraction_mean": float(self.background_fraction_mean),
        }
        if self.timing_summary:
            payload["timing_summary"] = self.timing_summary
        return payload


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    mask_builder: WristMaskBuilder,
    camera_key: str,
    output_kind: str = "masked",
    codec: str = "mp4v",
    size_policy: SizePolicy = "strict",
    max_frames: int = 0,
    overwrite: bool = False,
    mask_blur_px: int = 0,
    mask_blur_sigma: float = 0.0,
    progress: Callable[[int], None] | None = None,
    debug_timing: bool = False,
) -> VideoAdjustResult:
    if output_kind not in OUTPUT_KIND_CHOICES:
        raise ValueError(f"output_kind must be one of {OUTPUT_KIND_CHOICES}, got {output_kind!r}")

    cv2 = require_cv2()
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input video does not exist: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output video already exists: {output_path}")

    row_by_video_index: dict[int, Mapping[str, Any]] = {}
    video_index_column = f"observation.images.{camera_key}.video_frame_index"
    for row_index, row in enumerate(rows):
        try:
            video_index = int(row.get(video_index_column, row_index))
        except Exception:
            video_index = row_index
        row_by_video_index[video_index] = row

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open input video: {input_path}")
    writer = None
    started = time.perf_counter()
    frame_count = 0
    pose_source_counts: dict[str, int] = {}
    source_frame_counts: dict[str, int] = {}
    keep_fractions: list[float] = []
    background_fractions: list[float] = []
    timing = _TimingAccumulator() if bool(debug_timing) else None
    try:
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = _valid_fps(capture.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0:
            ok, first = capture.read()
            if not ok:
                raise RuntimeError(f"input video has no readable frames: {input_path}")
            height, width = first.shape[:2]
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        expected = (mask_builder.config.intrinsics.width, mask_builder.config.intrinsics.height)
        if (width, height) != expected:
            if size_policy != "scale":
                raise ValueError(
                    f"video size {width}x{height} does not match intrinsics/mask {expected[0]}x{expected[1]}; "
                    "use --size-policy scale only if the video was uniformly resized."
                )
            raise NotImplementedError("size-policy scale is not implemented for wrist_cam_adjust yet")
        writer = _open_writer(output_path, codec=codec, fps=fps, width=width, height=height)

        while True:
            if max_frames > 0 and frame_count >= int(max_frames):
                break
            if timing is None:
                ok, frame = capture.read()
            else:
                phase_started = time.perf_counter()
                ok, frame = capture.read()
                timing.add("read_frame", time.perf_counter() - phase_started)
            if not ok:
                break
            frame_started = time.perf_counter() if timing is not None else 0.0
            row = row_by_video_index.get(frame_count)
            if row is None:
                if frame_count < len(rows):
                    row = rows[frame_count]
                else:
                    raise IndexError(f"no dataset row for video frame {frame_count}")
            if timing is None:
                frame_mask = mask_builder.build(row)
                output = render_mask_output(
                    frame,
                    keep_mask=frame_mask.keep_mask,
                    output_kind=output_kind,
                    blur_px=mask_blur_px,
                    blur_sigma=mask_blur_sigma,
                )
                writer.write(output)
            else:
                phase_started = time.perf_counter()
                frame_mask = mask_builder.build(row)
                timing.add("build_mask", time.perf_counter() - phase_started)
                phase_started = time.perf_counter()
                output = render_mask_output(
                    frame,
                    keep_mask=frame_mask.keep_mask,
                    output_kind=output_kind,
                    blur_px=mask_blur_px,
                    blur_sigma=mask_blur_sigma,
                )
                timing.add("render_output", time.perf_counter() - phase_started)
                phase_started = time.perf_counter()
                writer.write(output)
                timing.add("write_frame", time.perf_counter() - phase_started)
            pose_source_counts[frame_mask.source_column] = pose_source_counts.get(frame_mask.source_column, 0) + 1
            source_frame_counts[frame_mask.source_frame] = source_frame_counts.get(frame_mask.source_frame, 0) + 1
            if timing is None:
                keep_fractions.append(float(mask_stats(frame_mask.keep_mask)["fraction"]))
                background_fractions.append(float(mask_stats(frame_mask.background_mask)["fraction"]))
            else:
                phase_started = time.perf_counter()
                keep_fractions.append(float(mask_stats(frame_mask.keep_mask)["fraction"]))
                background_fractions.append(float(mask_stats(frame_mask.background_mask)["fraction"]))
                timing.add("collect_stats", time.perf_counter() - phase_started)
            frame_count += 1
            if progress is not None:
                if timing is None:
                    progress(frame_count)
                else:
                    phase_started = time.perf_counter()
                    progress(frame_count)
                    timing.add("progress", time.perf_counter() - phase_started)
            if timing is not None:
                timing.add("frame_total_tracked", time.perf_counter() - frame_started)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    elapsed = time.perf_counter() - started
    timing_summary = timing.to_dict(elapsed_sec=elapsed, frame_count=frame_count) if timing is not None else {}
    return VideoAdjustResult(
        input_path=input_path,
        output_path=output_path,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        elapsed_sec=elapsed,
        output_kind=output_kind,
        pose_source_counts=pose_source_counts,
        source_frame_counts=source_frame_counts,
        keep_fraction_mean=float(np.mean(keep_fractions)) if keep_fractions else 0.0,
        background_fraction_mean=float(np.mean(background_fractions)) if background_fractions else 0.0,
        timing_summary=timing_summary,
    )
