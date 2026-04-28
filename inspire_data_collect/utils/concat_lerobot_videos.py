from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


DEFAULT_INPUT_DIR = Path(
    "/home/zc/HandMotion/real_robot_dataset/datasets/2026-04-27_chips_changed/"
    "videos/observation.images.front/chunk-000"
)


def natural_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def find_videos(input_dir: Path, pattern: str) -> list[Path]:
    videos = sorted(input_dir.glob(pattern), key=natural_key)
    return [path for path in videos if path.is_file()]


def _video_prop(capture: cv2.VideoCapture, prop: int, default: float) -> float:
    value = float(capture.get(prop) or 0.0)
    if not np.isfinite(value) or value <= 0.0:
        return float(default)
    return value


def probe_reference_video(path: Path, fallback_fps: float) -> tuple[int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    try:
        width = int(_video_prop(capture, cv2.CAP_PROP_FRAME_WIDTH, 0))
        height = int(_video_prop(capture, cv2.CAP_PROP_FRAME_HEIGHT, 0))
        fps = _video_prop(capture, cv2.CAP_PROP_FPS, fallback_fps)
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read frame size from video: {path}")
    return width, height, fps


def draw_source_label(
    frame: np.ndarray,
    *,
    text: str,
    index: int,
    total: int,
    font_scale: float,
    thickness: int,
) -> np.ndarray:
    output = frame.copy()
    lines = [f"source: {text}", f"{index + 1}/{total}"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = 10
    line_gap = 8
    sizes = [
        cv2.getTextSize(line, font, float(font_scale), int(thickness))[0]
        for line in lines
    ]
    box_width = max(width for width, _ in sizes) + padding * 2
    box_height = sum(height for _, height in sizes) + line_gap * (len(lines) - 1) + padding * 2
    box_width = min(int(box_width), output.shape[1])
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (box_width, box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, output, 0.45, 0.0, dst=output)

    y = padding + sizes[0][1]
    for line, (_, height) in zip(lines, sizes):
        cv2.putText(
            output,
            line,
            (padding, y),
            font,
            float(font_scale),
            (255, 255, 255),
            int(thickness),
            cv2.LINE_AA,
        )
        y += height + line_gap
    return output


def concat_videos(
    *,
    input_dir: Path,
    output_path: Path,
    pattern: str,
    fps: float | None,
    codec: str,
    font_scale: float,
    thickness: int,
    dry_run: bool,
) -> None:
    videos = find_videos(input_dir, pattern)
    if not videos:
        raise FileNotFoundError(f"No videos matched {pattern!r} in {input_dir}")

    width, height, source_fps = probe_reference_video(videos[0], fallback_fps=30.0)
    output_fps = float(source_fps if fps is None else fps)
    print(
        f"[ConcatVideos] videos={len(videos)} size={width}x{height} "
        f"fps={output_fps:.3f} output={output_path}"
    )
    if dry_run:
        for path in videos:
            print(path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*str(codec)),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video writer: {output_path}")

    total_frames = 0
    try:
        for video_index, video_path in enumerate(videos):
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            source_name = video_path.name
            frame_count = 0
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    frame = draw_source_label(
                        frame,
                        text=source_name,
                        index=video_index,
                        total=len(videos),
                        font_scale=font_scale,
                        thickness=thickness,
                    )
                    writer.write(frame)
                    frame_count += 1
                    total_frames += 1
            finally:
                capture.release()
            print(f"[ConcatVideos] appended {source_name}: {frame_count} frames")
    finally:
        writer.release()
    print(f"[ConcatVideos] wrote {total_frames} frames to {output_path}")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INPUT_DIR / "chunk-000_concat_with_source.mp4",
    )
    parser.add_argument("--pattern", type=str, default="*.mp4")
    parser.add_argument("--fps", type=float, default=None, help="Default uses the first video's FPS.")
    parser.add_argument("--codec", type=str, default="mp4v")
    parser.add_argument("--font-scale", type=float, default=0.85)
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Concatenate LeRobot episode videos and overlay the source filename at top-left."
    )
    add_args(parser)
    args = parser.parse_args(argv)
    concat_videos(
        input_dir=Path(args.input_dir).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
        pattern=str(args.pattern),
        fps=args.fps,
        codec=str(args.codec),
        font_scale=float(args.font_scale),
        thickness=int(args.thickness),
        dry_run=bool(args.dry_run),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
