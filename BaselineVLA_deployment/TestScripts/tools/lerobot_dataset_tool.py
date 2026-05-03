#!/usr/bin/env python3
"""Inspect and visualize local LeRobot dataset folders."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pyarrow.parquet as pq

from Datasets.video_utils import load_video_frames_ffmpeg_uint8


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def discover_dataset_roots(search_root: Path) -> list[Path]:
    return sorted({path.parent.parent.resolve() for path in search_root.glob("**/meta/info.json")})


@dataclass
class EpisodeData:
    episode_index: int
    tasks: list[str]
    state: np.ndarray
    action: np.ndarray
    timestamp: np.ndarray
    frame_index: np.ndarray
    sample_index: np.ndarray
    task_index: np.ndarray

    @property
    def length(self) -> int:
        return int(self.timestamp.shape[0])

    @property
    def duration_seconds(self) -> float:
        if self.length == 0:
            return 0.0
        return float(self.timestamp[-1] - self.timestamp[0])


@dataclass
class LeRobotDataset:
    root: Path
    info: dict[str, Any]
    stats: dict[str, Any]
    episodes: list[dict[str, Any]]
    tasks: list[dict[str, Any]]

    @classmethod
    def open(cls, root: Path | str) -> "LeRobotDataset":
        dataset_root = Path(root).expanduser().resolve()
        meta_dir = dataset_root / "meta"
        info_path = meta_dir / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing info.json under {meta_dir}")
        return cls(
            root=dataset_root,
            info=load_json(info_path),
            stats=load_json(meta_dir / "stats.json") if (meta_dir / "stats.json").exists() else {},
            episodes=load_jsonl(meta_dir / "episodes.jsonl"),
            tasks=load_jsonl(meta_dir / "tasks.jsonl"),
        )

    @property
    def dataset_name(self) -> str:
        parent_name = self.root.parent.name
        if parent_name and parent_name != self.root.name:
            return f"{parent_name}/{self.root.name}"
        return self.root.name

    @property
    def robot_type(self) -> str:
        return str(self.info.get("robot_type", "unknown"))

    @property
    def fps(self) -> int:
        return int(self.info.get("fps", 0) or 0)

    @property
    def total_episodes(self) -> int:
        return int(self.info.get("total_episodes", 0) or 0)

    @property
    def total_frames(self) -> int:
        return int(self.info.get("total_frames", 0) or 0)

    @property
    def chunk_size(self) -> int:
        return int(self.info.get("chunks_size", 1000) or 1000)

    @property
    def state_dim(self) -> int:
        return int(self.info["features"]["observation.state"]["shape"][0])

    @property
    def action_dim(self) -> int:
        return int(self.info["features"]["action"]["shape"][0])

    @property
    def state_names(self) -> list[str]:
        return list(self.info["features"]["observation.state"].get("names") or [])

    @property
    def action_names(self) -> list[str]:
        return list(self.info["features"]["action"].get("names") or [])

    @property
    def video_keys(self) -> list[str]:
        keys: list[str] = []
        for key, spec in self.info.get("features", {}).items():
            if spec.get("dtype") == "video":
                keys.append(key)
        return keys

    @property
    def split_spec(self) -> dict[str, str]:
        return dict(self.info.get("splits") or {})

    @property
    def task_lookup(self) -> dict[int, str]:
        return {int(row["task_index"]): str(row["task"]) for row in self.tasks}

    def chunk_for_episode(self, episode_index: int) -> int:
        return episode_index // self.chunk_size

    def data_path_for_episode(self, episode_index: int) -> Path:
        template = self.info["data_path"]
        return self.root / template.format(
            episode_chunk=self.chunk_for_episode(episode_index),
            episode_index=episode_index,
        )

    def video_path_for_episode(self, episode_index: int, video_key: str) -> Path:
        template = self.info["video_path"]
        return self.root / template.format(
            episode_chunk=self.chunk_for_episode(episode_index),
            episode_index=episode_index,
            video_key=video_key,
        )

    def episode_record(self, episode_index: int) -> dict[str, Any]:
        for record in self.episodes:
            if int(record["episode_index"]) == int(episode_index):
                return record
        raise KeyError(f"Episode {episode_index} not found in {self.root}")

    def episode_indices(self) -> list[int]:
        return [int(record["episode_index"]) for record in self.episodes]

    def episode_length_stats(self) -> dict[str, float]:
        if not self.episodes:
            return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
        lengths = np.asarray([int(record["length"]) for record in self.episodes], dtype=np.float32)
        return {
            "count": int(lengths.shape[0]),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "mean": round(float(np.mean(lengths)), 2),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.root),
            "dataset_name": self.dataset_name,
            "robot_type": self.robot_type,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "estimated_duration_minutes": round(self.total_frames / max(self.fps, 1) / 60.0, 2),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "state_names": self.state_names,
            "action_names": self.action_names,
            "video_keys": self.video_keys,
            "splits": self.split_spec,
            "tasks": self.tasks,
            "episode_length": self.episode_length_stats(),
            "feature_shapes": {
                key: spec.get("shape")
                for key, spec in self.info.get("features", {}).items()
            },
        }

    def load_episode(self, episode_index: int) -> EpisodeData:
        parquet_path = self.data_path_for_episode(episode_index)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing parquet file: {parquet_path}")

        table = pq.ParquetFile(parquet_path).read()
        columns = {name: table.column(name).to_pylist() for name in table.column_names}

        record = self.episode_record(episode_index)
        task_lookup = self.task_lookup
        task_indices = np.asarray(columns.get("task_index", []), dtype=np.int64)
        resolved_tasks: list[str] = []
        if record.get("tasks"):
            resolved_tasks.extend(str(task) for task in record["tasks"])
        elif task_indices.size > 0:
            for task_index in np.unique(task_indices):
                if int(task_index) in task_lookup:
                    resolved_tasks.append(task_lookup[int(task_index)])

        return EpisodeData(
            episode_index=int(episode_index),
            tasks=resolved_tasks,
            state=np.asarray(columns["observation.state"], dtype=np.float32),
            action=np.asarray(columns["action"], dtype=np.float32),
            timestamp=np.asarray(columns["timestamp"], dtype=np.float32),
            frame_index=np.asarray(columns["frame_index"], dtype=np.int64),
            sample_index=np.asarray(columns["index"], dtype=np.int64),
            task_index=task_indices,
        )

    def read_video_frames(
        self,
        episode_index: int,
        frame_numbers: Sequence[int],
        video_key: str,
    ) -> list[np.ndarray]:
        if not frame_numbers:
            return []
        video_path = self.video_path_for_episode(episode_index, video_key)
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video file: {video_path}")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()

        if frame_width > 0 and frame_height > 0:
            try:
                frames = load_video_frames_ffmpeg_uint8(
                    video_path=video_path,
                    frame_indices=frame_numbers,
                    resize_height=frame_height,
                    resize_width=frame_width,
                    color_order="bgr",
                )
                return [frame.copy() for frame in frames]
            except Exception as error:
                print(
                    f"[warn] ffmpeg clip decode failed for {video_path}: {error}. "
                    "Falling back to OpenCV frame seeks.",
                    file=sys.stderr,
                )
        return self._read_video_frames_opencv(
            video_path,
            frame_numbers,
            frame_height=frame_height,
            frame_width=frame_width,
        )

    def _read_video_frames_opencv(
        self,
        video_path: Path,
        frame_numbers: Sequence[int],
        frame_height: int,
        frame_width: int,
    ) -> list[np.ndarray]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")

        frames: list[np.ndarray] = []
        try:
            for frame_number in frame_numbers:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
                ok, frame = capture.read()
                if not ok or frame is None:
                    blank_height = max(1, int(frame_height))
                    blank_width = max(1, int(frame_width))
                    blank = np.zeros((blank_height, blank_width, 3), dtype=np.uint8)
                    cv2.putText(
                        blank,
                        "frame missing",
                        (24, min(blank_height - 12, max(24, blank_height // 2))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    frames.append(blank)
                else:
                    frames.append(frame)
        finally:
            capture.release()
        return frames


def format_compact_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def pad_rect(x: int, y: int, width: int, height: int, padding: int) -> tuple[int, int, int, int]:
    return x + padding, y + padding, width - 2 * padding, height - 2 * padding


def draw_panel(canvas: np.ndarray, x: int, y: int, width: int, height: int, title: str) -> tuple[int, int, int, int]:
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (78, 86, 94), 1)
    cv2.putText(canvas, title, (x + 18, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (236, 240, 245), 2, cv2.LINE_AA)
    return x + 18, y + 46, width - 36, height - 64


def draw_text_lines(
    canvas: np.ndarray,
    lines: Iterable[str],
    x: int,
    y: int,
    line_height: int = 26,
    color: tuple[int, int, int] = (220, 224, 230),
    font_scale: float = 0.62,
    thickness: int = 1,
) -> int:
    baseline_y = y
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (x, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        baseline_y += line_height
    return baseline_y


def downsample_columns(values: np.ndarray, target_columns: int) -> np.ndarray:
    if values.shape[1] <= target_columns:
        return values
    indices = np.linspace(0, values.shape[1] - 1, num=target_columns)
    indices = np.round(indices).astype(np.int64)
    return values[:, indices]


def make_heatmap_image(values: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, float]:
    matrix = np.asarray(values, dtype=np.float32).T
    matrix = downsample_columns(matrix, max(width, 1))
    value_min = float(np.min(matrix)) if matrix.size else 0.0
    value_max = float(np.max(matrix)) if matrix.size else 0.0
    if math.isclose(value_min, value_max):
        normalized = np.full(matrix.shape, 127, dtype=np.uint8)
    else:
        normalized = ((matrix - value_min) / (value_max - value_min) * 255.0).clip(0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_NEAREST)
    return heatmap, value_min, value_max


def draw_heatmap_section(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    values: np.ndarray,
    row_labels: Sequence[str],
) -> None:
    inner_x, inner_y, inner_w, inner_h = draw_panel(canvas, x, y, width, height, title)
    label_width = min(150, max(96, inner_w // 7))
    legend_height = 26
    heatmap_x = inner_x + label_width
    heatmap_y = inner_y
    heatmap_w = inner_w - label_width
    heatmap_h = inner_h - legend_height

    heatmap, value_min, value_max = make_heatmap_image(values, heatmap_w, heatmap_h)
    canvas[heatmap_y : heatmap_y + heatmap_h, heatmap_x : heatmap_x + heatmap_w] = heatmap
    cv2.rectangle(canvas, (heatmap_x, heatmap_y), (heatmap_x + heatmap_w, heatmap_y + heatmap_h), (95, 100, 106), 1)

    label_count = max(len(row_labels), 1)
    for row_index, label in enumerate(row_labels):
        row_center = int(heatmap_y + (row_index + 0.5) * heatmap_h / label_count)
        short_label = label.replace("observation.", "").replace("action_", "a").replace("state_", "s")
        cv2.putText(
            canvas,
            short_label,
            (inner_x, row_center + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (214, 218, 223),
            1,
            cv2.LINE_AA,
        )
        separator_y = int(heatmap_y + row_index * heatmap_h / label_count)
        cv2.line(canvas, (heatmap_x, separator_y), (heatmap_x + heatmap_w, separator_y), (52, 56, 62), 1)

    bottom_y = heatmap_y + heatmap_h + 18
    draw_text_lines(
        canvas,
        [
            "time -> left to right",
            f"min={format_compact_number(value_min)}  max={format_compact_number(value_max)}",
        ],
        inner_x,
        bottom_y,
        line_height=18,
        color=(160, 168, 176),
        font_scale=0.47,
    )


def draw_line_plot_section(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    timestamps: np.ndarray,
    series: Sequence[tuple[str, np.ndarray, tuple[int, int, int]]],
) -> None:
    inner_x, inner_y, inner_w, inner_h = draw_panel(canvas, x, y, width, height, title)
    plot_x, plot_y, plot_w, plot_h = pad_rect(inner_x, inner_y, inner_w, inner_h, 8)
    left_margin = 52
    bottom_margin = 32
    top_margin = 8
    right_margin = 10
    chart_x = plot_x + left_margin
    chart_y = plot_y + top_margin
    chart_w = plot_w - left_margin - right_margin
    chart_h = plot_h - top_margin - bottom_margin

    all_values = np.concatenate([np.asarray(values, dtype=np.float32) for _, values, _ in series])
    value_min = float(np.min(all_values)) if all_values.size else 0.0
    value_max = float(np.max(all_values)) if all_values.size else 1.0
    if math.isclose(value_min, value_max):
        value_min -= 1.0
        value_max += 1.0
    padding = (value_max - value_min) * 0.08
    value_min -= padding
    value_max += padding

    cv2.rectangle(canvas, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), (88, 95, 101), 1)

    for tick in range(5):
        ratio = tick / 4.0
        line_y = int(chart_y + ratio * chart_h)
        cv2.line(canvas, (chart_x, line_y), (chart_x + chart_w, line_y), (48, 52, 58), 1)
        tick_value = value_max - ratio * (value_max - value_min)
        cv2.putText(
            canvas,
            format_compact_number(tick_value),
            (plot_x, line_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (166, 172, 178),
            1,
            cv2.LINE_AA,
        )

    time_min = float(timestamps[0]) if timestamps.size else 0.0
    time_max = float(timestamps[-1]) if timestamps.size else 1.0
    if math.isclose(time_min, time_max):
        time_max = time_min + 1.0

    for label, values, color in series:
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            continue
        step_count = min(chart_w, values.shape[0])
        indices = np.linspace(0, values.shape[0] - 1, num=step_count)
        indices = np.round(indices).astype(np.int64)
        sampled_values = values[indices]
        sampled_times = timestamps[indices]
        points: list[tuple[int, int]] = []
        for sampled_time, sampled_value in zip(sampled_times, sampled_values, strict=True):
            x_ratio = (float(sampled_time) - time_min) / (time_max - time_min)
            y_ratio = (float(sampled_value) - value_min) / (value_max - value_min)
            point_x = int(chart_x + x_ratio * chart_w)
            point_y = int(chart_y + chart_h - y_ratio * chart_h)
            points.append((point_x, point_y))
        if len(points) >= 2:
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"{time_min:.1f}s",
        (chart_x - 6, chart_y + chart_h + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (166, 172, 178),
        1,
        cv2.LINE_AA,
    )
    right_label = f"{time_max:.1f}s"
    right_width, _ = cv2.getTextSize(right_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(
        canvas,
        right_label,
        (chart_x + chart_w - right_width[0], chart_y + chart_h + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (166, 172, 178),
        1,
        cv2.LINE_AA,
    )

    legend_x = chart_x
    legend_y = inner_y + inner_h - 4
    cursor_x = legend_x
    for label, _, color in series:
        cv2.line(canvas, (cursor_x, legend_y), (cursor_x + 18, legend_y), color, 3)
        cv2.putText(
            canvas,
            label,
            (cursor_x + 26, legend_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (210, 214, 220),
            1,
            cv2.LINE_AA,
        )
        cursor_x += 150


def resize_to_fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = image.shape[:2]
    if source_h == 0 or source_w == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / source_w, height / source_h)
    resized = cv2.resize(image, (max(1, int(source_w * scale)), max(1, int(source_h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    offset_x = (width - resized.shape[1]) // 2
    offset_y = (height - resized.shape[0]) // 2
    canvas[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized
    return canvas


def draw_frame_strip(
    canvas: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    frames: Sequence[np.ndarray],
    captions: Sequence[str],
) -> None:
    inner_x, inner_y, inner_w, inner_h = draw_panel(canvas, x, y, width, height, title)
    if not frames:
        cv2.putText(canvas, "No frames available", (inner_x, inner_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return

    gap = 12
    frame_w = max(1, (inner_w - gap * (len(frames) - 1)) // len(frames))
    frame_h = inner_h - 34
    cursor_x = inner_x
    for frame, caption in zip(frames, captions, strict=True):
        fitted = resize_to_fit(frame, frame_w, frame_h)
        canvas[inner_y : inner_y + frame_h, cursor_x : cursor_x + frame_w] = fitted
        cv2.rectangle(canvas, (cursor_x, inner_y), (cursor_x + frame_w, inner_y + frame_h), (92, 98, 104), 1)
        cv2.putText(
            canvas,
            caption,
            (cursor_x, inner_y + frame_h + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (212, 216, 222),
            1,
            cv2.LINE_AA,
        )
        cursor_x += frame_w + gap


def default_output_path(dataset: LeRobotDataset, episode_index: int) -> Path:
    safe_name = dataset.dataset_name.replace("/", "__")
    return Path("outputs") / "lerobot_vis" / safe_name / f"episode_{episode_index:06d}.png"


def build_episode_visualization(
    dataset: LeRobotDataset,
    episode: EpisodeData,
    output_path: Path,
    video_key: str,
    num_frames: int,
) -> Path:
    width = 1800
    outer_margin = 40
    gap = 24
    title_block_h = 96
    header_h = 172
    line_plot_h = 250
    heatmap_h = 300
    frame_strip_h = 340
    height = (
        outer_margin * 2
        + title_block_h
        + header_h
        + line_plot_h
        + heatmap_h * 2
        + frame_strip_h
        + gap * 4
    )
    canvas = np.full((height, width, 3), (19, 22, 27), dtype=np.uint8)

    cv2.putText(
        canvas,
        f"LeRobot Dataset Episode Summary: {dataset.dataset_name}",
        (outer_margin, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (245, 248, 250),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"episode={episode.episode_index:06d}  len={episode.length}  duration={episode.duration_seconds:.1f}s  fps={dataset.fps}",
        (outer_margin, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (190, 198, 206),
        1,
        cv2.LINE_AA,
    )

    header_panel_y = outer_margin + title_block_h
    header_x, header_y, header_w, header_h_inner = draw_panel(canvas, outer_margin, header_panel_y, width - 2 * outer_margin, header_h, "Dataset Metadata")
    left_lines = [
        f"root: {dataset.root}",
        f"robot_type: {dataset.robot_type}",
        f"state_dim: {dataset.state_dim}  action_dim: {dataset.action_dim}",
        f"video_keys: {', '.join(dataset.video_keys) if dataset.video_keys else 'none'}",
    ]
    right_lines = [
        f"episode_tasks: {', '.join(episode.tasks) if episode.tasks else 'n/a'}",
        f"parquet: {dataset.data_path_for_episode(episode.episode_index).name}",
        f"video: {dataset.video_path_for_episode(episode.episode_index, video_key).name}",
        f"split_spec: {json.dumps(dataset.split_spec)}",
    ]
    draw_text_lines(canvas, left_lines, header_x, header_y + 4)
    draw_text_lines(canvas, right_lines, header_x + header_w // 2, header_y + 4)

    section_y = header_panel_y + header_h + gap
    state_norm = np.linalg.norm(episode.state, axis=1)
    action_norm = np.linalg.norm(episode.action, axis=1)
    shared_dim = min(episode.state.shape[1], episode.action.shape[1])
    motion_norm = np.linalg.norm(episode.action[:, :shared_dim] - episode.state[:, :shared_dim], axis=1)
    draw_line_plot_section(
        canvas,
        outer_margin,
        section_y,
        width - 2 * outer_margin,
        line_plot_h,
        "Temporal Trends",
        episode.timestamp,
        [
            ("||state||", state_norm, (99, 221, 144)),
            ("||action||", action_norm, (70, 157, 247)),
            ("||action-state||", motion_norm, (255, 179, 71)),
        ],
    )

    section_y += line_plot_h + gap
    draw_heatmap_section(
        canvas,
        outer_margin,
        section_y,
        width - 2 * outer_margin,
        heatmap_h,
        "Observation State Heatmap",
        episode.state,
        dataset.state_names or [f"state_{index}" for index in range(episode.state.shape[1])],
    )

    section_y += heatmap_h + gap
    draw_heatmap_section(
        canvas,
        outer_margin,
        section_y,
        width - 2 * outer_margin,
        heatmap_h,
        "Action Heatmap",
        episode.action,
        dataset.action_names or [f"action_{index}" for index in range(episode.action.shape[1])],
    )

    section_y += heatmap_h + gap
    if episode.length > 0:
        sample_rows = np.linspace(0, episode.length - 1, num=max(num_frames, 1))
        sample_rows = np.round(sample_rows).astype(np.int64)
        frame_numbers = [int(episode.frame_index[row]) for row in sample_rows]
        frames = dataset.read_video_frames(episode.episode_index, frame_numbers, video_key=video_key)
        captions = [
            f"row={int(row)} frame={int(episode.frame_index[row])} t={episode.timestamp[row]:.1f}s"
            for row in sample_rows
        ]
    else:
        frames = []
        captions = []
    draw_frame_strip(canvas, outer_margin, section_y, width - 2 * outer_margin, frame_strip_h, "Sampled Video Frames", frames, captions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def cmd_scan(args: argparse.Namespace) -> int:
    roots = discover_dataset_roots(Path(args.search_root).expanduser())
    datasets = [LeRobotDataset.open(root) for root in roots]

    if args.json:
        payload = []
        for dataset in datasets:
            payload.append(dataset.summary())
        print(json.dumps(payload, indent=2))
        return 0

    print(
        f"{'dataset':58} {'episodes':>8} {'frames':>8} {'mins':>8} "
        f"{'state':>6} {'action':>6} {'video':>10} {'empty':>6}"
    )
    print("-" * 118)
    for dataset in datasets:
        minutes = dataset.total_frames / max(dataset.fps, 1) / 60.0
        video_desc = ",".join(dataset.video_keys) if dataset.video_keys else "-"
        empty = "yes" if dataset.total_episodes == 0 else "no"
        print(
            f"{dataset.dataset_name[:58]:58} {dataset.total_episodes:8d} {dataset.total_frames:8d} "
            f"{minutes:8.2f} {dataset.state_dim:6d} {dataset.action_dim:6d} "
            f"{video_desc[:10]:10} {empty:>6}"
        )
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    dataset = LeRobotDataset.open(args.dataset_root)
    print(json.dumps(dataset.summary(), indent=2))
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    dataset = LeRobotDataset.open(args.dataset_root)
    if dataset.total_episodes == 0:
        raise ValueError(f"Dataset is empty: {dataset.root}")

    episode_index = int(args.episode_index)
    episode = dataset.load_episode(episode_index)
    video_key = args.video_key or (dataset.video_keys[0] if dataset.video_keys else "")
    if not video_key:
        raise ValueError(f"No video feature found in {dataset.root}")

    output_path = Path(args.output).expanduser() if args.output else default_output_path(dataset, episode_index)
    written_path = build_episode_visualization(
        dataset=dataset,
        episode=episode,
        output_path=output_path,
        video_key=video_key,
        num_frames=int(args.frames),
    )
    print(json.dumps({"output": str(written_path.resolve()), "episode": episode_index, "video_key": video_key}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and visualize LeRobot datasets stored on disk.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Find all dataset roots below a search directory.")
    scan_parser.add_argument("--search-root", default="temp_test", help="Directory to scan for meta/info.json files.")
    scan_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a table.")
    scan_parser.set_defaults(func=cmd_scan)

    summary_parser = subparsers.add_parser("summary", help="Print a detailed summary for one dataset root.")
    summary_parser.add_argument("--dataset-root", required=True, help="Dataset root that contains meta/, data/, and videos/.")
    summary_parser.set_defaults(func=cmd_summary)

    vis_parser = subparsers.add_parser("visualize", help="Render a PNG summary for one episode.")
    vis_parser.add_argument("--dataset-root", required=True, help="Dataset root that contains meta/, data/, and videos/.")
    vis_parser.add_argument("--episode-index", type=int, default=0, help="Episode index to visualize.")
    vis_parser.add_argument("--video-key", default="", help="Video feature key to use, e.g. observation.camera_0.rgb.")
    vis_parser.add_argument("--frames", type=int, default=6, help="How many evenly spaced video frames to sample.")
    vis_parser.add_argument("--output", default="", help="Optional output PNG path.")
    vis_parser.set_defaults(func=cmd_visualize)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
