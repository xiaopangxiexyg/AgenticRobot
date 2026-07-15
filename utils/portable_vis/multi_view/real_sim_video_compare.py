#!/usr/bin/env python3
"""Export same-time real/sim camera comparison videos."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

try:
    import cv2
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("OpenCV is required to run this script. Install python-opencv/cv2 first.") from exc

import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
PREFERRED_CAMERA_ORDER = ("front", "left_wrist", "right_wrist")


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    episode_id: str
    frame_count: int | None
    video_paths: Mapping[str, str]
    start_timestamp: float | None = None
    end_timestamp: float | None = None
    duration_sec: float | None = None


@dataclass
class VideoStream:
    dataset_label: str
    camera_name: str
    path: Path
    capture: cv2.VideoCapture
    frame_count: int
    fps: float
    width: int
    height: int
    offset_s: float = 0.0
    cached_index: int = -1
    cached_frame: np.ndarray | None = None

    @property
    def duration_s(self) -> float:
        if self.frame_count <= 0 or self.fps <= 0.0:
            return 0.0
        return max(0.0, (self.frame_count - 1) / self.fps)

    def frame_index_for_time(self, global_time_s: float) -> int | None:
        local_time_s = global_time_s + self.offset_s
        if local_time_s < 0.0:
            return None
        index = int(math.floor(local_time_s * self.fps + 1e-9))
        if index >= self.frame_count:
            return None
        return max(0, index)

    def read_frame(self, frame_index: int) -> np.ndarray:
        index = max(0, min(int(frame_index), self.frame_count - 1))
        if index == self.cached_index and self.cached_frame is not None:
            return self.cached_frame
        if index != self.cached_index + 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            frame = _message_frame(max(self.width, 320), max(self.height, 240), "frame unavailable")
        else:
            frame = _ensure_bgr(frame)
        self.cached_index = index
        self.cached_frame = frame
        return frame

    def close(self) -> None:
        self.capture.release()


@dataclass
class CameraPair:
    camera_name: str
    real: VideoStream
    sim: VideoStream


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a side-by-side real/sim video for same-time, same-camera "
            "dataset inspection. Each camera is one row; real and sim are columns."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--real-dataset", type=Path, required=True, help="Real dataset root.")
    parser.add_argument("--sim-dataset", type=Path, required=True, help="Sim replay dataset root.")
    parser.add_argument("--episode", default="0", help="Episode index or episode_id in both datasets.")
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=[],
        help="Camera names. Accepts space-separated or comma-separated values. Defaults to real/sim intersection.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output comparison video path.")
    parser.add_argument(
        "--cell-size",
        default="640x480",
        help="Rendered size of each real/sim cell as WIDTHxHEIGHT.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Output FPS. 0 uses the minimum FPS across selected videos.",
    )
    parser.add_argument("--start-time", type=float, default=0.0, help="Global relative start time in seconds.")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Duration to export. 0 exports the full common valid time range.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum output frames. 0 means no limit; useful for smoke checks.",
    )
    parser.add_argument(
        "--real-offset",
        type=float,
        default=0.0,
        help="Seconds added to the real video local sample time.",
    )
    parser.add_argument(
        "--sim-offset",
        type=float,
        default=0.0,
        help="Seconds added to the sim video local sample time.",
    )
    parser.add_argument("--real-label", default="real", help="Column label for the real dataset.")
    parser.add_argument("--sim-label", default="sim", help="Column label for the sim dataset.")
    parser.add_argument("--codec", default="mp4v", help="FourCC codec for cv2.VideoWriter.")
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not draw labels/timestamps over the output video.",
    )
    return parser.parse_args(argv)


def _split_csv_like(values: Sequence[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in str(value).split(",") if part.strip())
    return items


def _parse_size(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX,]\s*(\d+)\s*", str(value))
    if not match:
        raise ValueError(f"Expected WIDTHxHEIGHT, got: {value}")
    width = int(match.group(1))
    height = int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError("Size dimensions must be positive.")
    return width, height


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _episode_id_from_selector(selector: str) -> str:
    text = str(selector).strip()
    if text.startswith("episode_"):
        return text
    if text.isdigit():
        return f"episode_{int(text):06d}"
    return text


def load_episode_record(dataset_root: Path, episode: str | int = "0") -> EpisodeRecord:
    root = dataset_root.expanduser().resolve()
    episodes_path = root / "meta" / "episodes.jsonl"
    selector = str(episode)
    requested_index = int(selector) if selector.isdigit() else None
    requested_id = _episode_id_from_selector(selector)

    if episodes_path.exists():
        matches: list[EpisodeRecord] = []
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            episode_index = _int_or_none(raw.get("episode_index"))
            episode_id = str(raw.get("episode_id") or f"episode_{episode_index or 0:06d}")
            if (requested_index is not None and episode_index == requested_index) or episode_id == requested_id:
                video_paths = raw.get("video_paths") or {}
                if not isinstance(video_paths, Mapping):
                    video_paths = {}
                matches.append(
                    EpisodeRecord(
                        episode_index=int(episode_index if episode_index is not None else requested_index or 0),
                        episode_id=episode_id,
                        frame_count=_int_or_none(raw.get("frame_count")),
                        video_paths={str(k): str(v) for k, v in video_paths.items()},
                        start_timestamp=_float_or_none(raw.get("start_timestamp")),
                        end_timestamp=_float_or_none(raw.get("end_timestamp")),
                        duration_sec=_float_or_none(raw.get("duration_sec")),
                    )
                )
        if matches:
            return matches[0]
        raise ValueError(f"Episode {episode!r} not found in {episodes_path}.")

    episode_index = requested_index if requested_index is not None else 0
    episode_id = _episode_id_from_selector(selector)
    return EpisodeRecord(
        episode_index=episode_index,
        episode_id=episode_id,
        frame_count=None,
        video_paths=_discover_conventional_video_paths(root, episode_id),
    )


def _camera_name_from_video_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        if part.startswith("observation.images."):
            return part.removeprefix("observation.images.")
    return None


def _discover_conventional_video_paths(dataset_root: Path, episode_id: str) -> dict[str, str]:
    paths: dict[str, str] = {}
    videos_root = dataset_root / "videos"
    if not videos_root.exists():
        return paths
    for path in videos_root.rglob(f"{episode_id}.*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        camera_name = _camera_name_from_video_path(path)
        if camera_name is None:
            continue
        paths.setdefault(camera_name, str(path.relative_to(dataset_root)))
    return paths


def _resolve_video_path(dataset_root: Path, episode: EpisodeRecord, camera_name: str) -> Path:
    root = dataset_root.expanduser().resolve()
    rel = episode.video_paths.get(camera_name)
    if rel:
        path = root / rel
        if path.exists():
            return path

    discovered = _discover_conventional_video_paths(root, episode.episode_id)
    rel = discovered.get(camera_name)
    if rel:
        path = root / rel
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Camera {camera_name!r} video was not found for {root} episode {episode.episode_id}."
    )


def _ordered_camera_names(names: set[str]) -> list[str]:
    ordered = [name for name in PREFERRED_CAMERA_ORDER if name in names]
    ordered.extend(sorted(name for name in names if name not in set(ordered)))
    return ordered


def _video_exists(dataset_root: Path, episode: EpisodeRecord, camera_name: str) -> bool:
    try:
        _resolve_video_path(dataset_root, episode, camera_name)
    except FileNotFoundError:
        return False
    return True


def _ordered_available_camera_intersection(
    real_dataset: Path,
    sim_dataset: Path,
    real_record: EpisodeRecord,
    sim_record: EpisodeRecord,
) -> list[str]:
    common = set(real_record.video_paths).intersection(sim_record.video_paths)
    available = {
        name
        for name in common
        if _video_exists(real_dataset, real_record, name)
        and _video_exists(sim_dataset, sim_record, name)
    }
    return _ordered_camera_names(available)


def _open_stream(
    *,
    dataset_label: str,
    dataset_root: Path,
    episode: EpisodeRecord,
    camera_name: str,
    offset_s: float,
) -> VideoStream:
    path = _resolve_video_path(dataset_root, episode, camera_name)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Could not read frame count from video: {path}")
    if fps <= 0.0:
        fps = 30.0
    return VideoStream(
        dataset_label=dataset_label,
        camera_name=camera_name,
        path=path,
        capture=capture,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        offset_s=offset_s,
    )


def open_camera_pairs(
    *,
    real_dataset: Path,
    sim_dataset: Path,
    episode: str | int,
    cameras: Sequence[str],
    real_label: str,
    sim_label: str,
    real_offset_s: float = 0.0,
    sim_offset_s: float = 0.0,
) -> list[CameraPair]:
    real_record = load_episode_record(real_dataset, episode)
    sim_record = load_episode_record(sim_dataset, episode)
    camera_names = (
        list(cameras)
        if cameras
        else _ordered_available_camera_intersection(real_dataset, sim_dataset, real_record, sim_record)
    )
    if not camera_names:
        raise ValueError("No common cameras were found between real and sim datasets.")

    pairs: list[CameraPair] = []
    try:
        for camera_name in camera_names:
            pairs.append(
                CameraPair(
                    camera_name=camera_name,
                    real=_open_stream(
                        dataset_label=real_label,
                        dataset_root=real_dataset,
                        episode=real_record,
                        camera_name=camera_name,
                        offset_s=real_offset_s,
                    ),
                    sim=_open_stream(
                        dataset_label=sim_label,
                        dataset_root=sim_dataset,
                        episode=sim_record,
                        camera_name=camera_name,
                        offset_s=sim_offset_s,
                    ),
                )
            )
    except Exception:
        close_camera_pairs(pairs)
        raise
    return pairs


def close_camera_pairs(pairs: Sequence[CameraPair]) -> None:
    for pair in pairs:
        pair.real.close()
        pair.sim.close()


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _message_frame(width: int, height: int, message: str) -> np.ndarray:
    frame = np.full((max(1, height), max(1, width), 3), (30, 35, 44), dtype=np.uint8)
    cv2.putText(
        frame,
        message,
        (16, max(32, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (226, 232, 242),
        2,
        cv2.LINE_AA,
    )
    return frame


def _resize_letterbox(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    source_h, source_w = frame.shape[:2]
    if source_h <= 0 or source_w <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / source_w, target_h / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(
        frame,
        (resized_w, resized_h),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.full((target_h, target_w, 3), (14, 16, 20), dtype=np.uint8)
    x = (target_w - resized_w) // 2
    y = (target_h - resized_h) // 2
    canvas[y : y + resized_h, x : x + resized_w] = resized
    return canvas


def _draw_label(
    frame: np.ndarray,
    text: str,
    *,
    origin: tuple[int, int] = (10, 26),
    scale: float = 0.58,
) -> None:
    x, y = origin
    thickness = 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (246, 248, 252), thickness, cv2.LINE_AA)


def _frame_for_stream(stream: VideoStream, time_s: float, cell_size: tuple[int, int], *, draw_labels: bool) -> np.ndarray:
    frame_index = stream.frame_index_for_time(time_s)
    if frame_index is None:
        local_time = time_s + stream.offset_s
        message = "before start" if local_time < 0.0 else "after end"
        frame = _message_frame(max(stream.width, cell_size[0]), max(stream.height, cell_size[1]), message)
    else:
        frame = stream.read_frame(frame_index)
    cell = _resize_letterbox(frame, cell_size)
    if draw_labels:
        index_text = "--" if frame_index is None else str(frame_index)
        _draw_label(
            cell,
            f"{stream.dataset_label} | {stream.camera_name} | frame {index_text}",
        )
    return cell


def compose_comparison_frame(
    pairs: Sequence[CameraPair],
    *,
    time_s: float,
    cell_size: tuple[int, int],
    draw_labels: bool = True,
) -> np.ndarray:
    if not pairs:
        raise ValueError("At least one camera pair is required.")

    cell_w, cell_h = cell_size
    header_h = 48 if draw_labels else 0
    canvas_h = header_h + cell_h * len(pairs)
    canvas_w = cell_w * 2
    canvas = np.full((canvas_h, canvas_w, 3), (10, 12, 16), dtype=np.uint8)

    if draw_labels:
        cv2.rectangle(canvas, (0, 0), (canvas_w, header_h), (18, 22, 30), -1)
        _draw_label(canvas, f"real/sim camera comparison | t={time_s:.3f}s", origin=(14, 32), scale=0.68)
        cv2.line(canvas, (cell_w, 0), (cell_w, canvas_h), (88, 96, 110), 1)

    for row, pair in enumerate(pairs):
        y = header_h + row * cell_h
        real_frame = _frame_for_stream(pair.real, time_s, cell_size, draw_labels=draw_labels)
        sim_frame = _frame_for_stream(pair.sim, time_s, cell_size, draw_labels=draw_labels)
        canvas[y : y + cell_h, 0:cell_w] = real_frame
        canvas[y : y + cell_h, cell_w : cell_w * 2] = sim_frame
        if draw_labels:
            cv2.line(canvas, (0, y), (canvas_w, y), (68, 76, 88), 1)
            _draw_label(canvas, pair.camera_name, origin=(14, y + cell_h - 14), scale=0.54)

    return canvas


def _common_valid_range(pairs: Sequence[CameraPair]) -> tuple[float, float]:
    starts = [0.0]
    ends: list[float] = []
    for pair in pairs:
        for stream in (pair.real, pair.sim):
            starts.append(max(0.0, -stream.offset_s))
            ends.append(stream.duration_s - stream.offset_s)
    end = min(ends) if ends else 0.0
    start = max(starts)
    if end < start:
        raise ValueError(f"No overlapping timeline range after offsets: start={start:.3f}, end={end:.3f}")
    return start, end


def _output_fps(pairs: Sequence[CameraPair], requested_fps: float) -> float:
    if requested_fps > 0.0:
        return float(requested_fps)
    fps_values = [stream.fps for pair in pairs for stream in (pair.real, pair.sim) if stream.fps > 0.0]
    if not fps_values:
        return 30.0
    return min(fps_values)


def export_real_sim_comparison(
    pairs: Sequence[CameraPair],
    *,
    output_path: Path,
    cell_size: tuple[int, int],
    fps: float,
    start_time_s: float = 0.0,
    duration_s: float = 0.0,
    max_frames: int = 0,
    codec: str = "mp4v",
    draw_labels: bool = True,
) -> dict[str, object]:
    if not pairs:
        raise ValueError("At least one camera pair is required.")
    common_start, common_end = _common_valid_range(pairs)
    start = max(common_start, float(start_time_s))
    end = common_end if duration_s <= 0.0 else min(common_end, start + float(duration_s))
    if end < start:
        raise ValueError(f"Requested range is outside common timeline: start={start:.3f}, end={end:.3f}")

    out_fps = _output_fps(pairs, fps)
    frame_count = int(math.floor((end - start) * out_fps + 1e-9)) + 1
    if max_frames > 0:
        frame_count = min(frame_count, int(max_frames))
    if frame_count <= 0:
        raise ValueError("No frames selected for export.")

    sample = compose_comparison_frame(pairs, time_s=start, cell_size=cell_size, draw_labels=draw_labels)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*str(codec)[:4]),
        out_fps,
        (sample.shape[1], sample.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video for writing: {output_path}")

    try:
        for frame_i in range(frame_count):
            time_s = start + frame_i / out_fps
            frame = sample if frame_i == 0 else compose_comparison_frame(
                pairs,
                time_s=time_s,
                cell_size=cell_size,
                draw_labels=draw_labels,
            )
            writer.write(frame)
    finally:
        writer.release()

    return {
        "output_path": str(output_path),
        "frame_count": frame_count,
        "fps": out_fps,
        "start_time_s": start,
        "end_time_s": start + (frame_count - 1) / out_fps,
        "cell_size": list(cell_size),
        "resolution": [int(sample.shape[1]), int(sample.shape[0])],
        "cameras": [pair.camera_name for pair in pairs],
        "real_paths": {pair.camera_name: str(pair.real.path) for pair in pairs},
        "sim_paths": {pair.camera_name: str(pair.sim.path) for pair in pairs},
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    pairs: list[CameraPair] = []
    try:
        cell_size = _parse_size(str(args.cell_size))
        cameras = _split_csv_like(args.cameras)
        pairs = open_camera_pairs(
            real_dataset=args.real_dataset,
            sim_dataset=args.sim_dataset,
            episode=args.episode,
            cameras=cameras,
            real_label=str(args.real_label),
            sim_label=str(args.sim_label),
            real_offset_s=float(args.real_offset),
            sim_offset_s=float(args.sim_offset),
        )
        summary = export_real_sim_comparison(
            pairs,
            output_path=args.output,
            cell_size=cell_size,
            fps=float(args.fps),
            start_time_s=float(args.start_time),
            duration_s=float(args.duration),
            max_frames=int(args.max_frames),
            codec=str(args.codec),
            draw_labels=not bool(args.no_labels),
        )
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        close_camera_pairs(pairs)


if __name__ == "__main__":
    raise SystemExit(main())
