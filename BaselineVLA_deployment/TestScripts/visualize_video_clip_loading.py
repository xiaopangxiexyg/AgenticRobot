from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from Datasets.video_utils import load_video_frames_ffmpeg_uint8


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ffmpeg clip loading for a local video.")
    parser.add_argument("--video-path", type=str, required=True, help="Path to a local video file.")
    parser.add_argument(
        "--frame-indices",
        type=str,
        required=True,
        help="Comma-separated frame indices, for example: 0,8,16,24",
    )
    parser.add_argument("--resize-height", type=int, default=256)
    parser.add_argument("--resize-width", type=int, default=256)
    parser.add_argument("--output", type=str, default=None, help="Optional output PNG path.")
    return parser.parse_args()


def _parse_frame_indices(text: str) -> list[int]:
    indices = [chunk.strip() for chunk in str(text).split(",")]
    resolved = [int(chunk) for chunk in indices if chunk]
    if not resolved:
        raise ValueError("frame_indices must contain at least one integer.")
    return resolved


def _build_contact_sheet(frames: np.ndarray, frame_indices: list[int]) -> np.ndarray:
    if frames.ndim != 4:
        raise ValueError(f"frames must have shape [T, H, W, C], got {frames.shape}")
    frame_count, frame_height, frame_width, _ = frames.shape
    columns = min(4, max(1, frame_count))
    rows = int(math.ceil(frame_count / float(columns)))
    caption_height = 34
    gap = 12
    canvas_height = rows * (frame_height + caption_height) + max(0, rows - 1) * gap
    canvas_width = columns * frame_width + max(0, columns - 1) * gap
    canvas = np.full((canvas_height, canvas_width, 3), 20, dtype=np.uint8)

    for index, frame in enumerate(frames):
        row = index // columns
        column = index % columns
        y0 = row * (frame_height + caption_height + gap)
        x0 = column * (frame_width + gap)
        canvas[y0 : y0 + frame_height, x0 : x0 + frame_width] = frame
        label = f"frame={int(frame_indices[index])}"
        cv2.putText(
            canvas,
            label,
            (x0 + 10, y0 + frame_height + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (238, 242, 247),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> None:
    args = _parse_args()
    video_path = Path(args.video_path).expanduser().resolve()
    frame_indices = _parse_frame_indices(args.frame_indices)
    frames = load_video_frames_ffmpeg_uint8(
        video_path=video_path,
        frame_indices=frame_indices,
        resize_height=int(args.resize_height),
        resize_width=int(args.resize_width),
        color_order="bgr",
    )
    contact_sheet = _build_contact_sheet(frames, frame_indices)

    if args.output is None:
        output_path = PROJECT_ROOT / "Outputs" / "video_clip_debug" / f"{video_path.stem}_clip.png"
    else:
        output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), contact_sheet)
    print(f"saved visualization to {output_path}")


if __name__ == "__main__":
    main()
