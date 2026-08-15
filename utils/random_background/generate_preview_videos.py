from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dataset_interface.random_background.background import (
    apply_random_background,
    generate_background,
    temporal_background_from_episode,
)


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("preview video generation requires opencv-python.") from exc
    return cv2


def _writer(path: Path, *, width: int, height: int, fps: float):
    cv2 = _require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    return writer


def _demo_foreground(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _require_cv2()
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    table_y = int(height * 0.72)
    cv2.rectangle(frame, (0, table_y), (width - 1, height - 1), (86, 96, 102), thickness=-1)
    cv2.rectangle(mask, (0, table_y), (width - 1, height - 1), 255, thickness=-1)

    palm_center = (int(width * 0.52), int(height * 0.48))
    cv2.ellipse(frame, palm_center, (int(width * 0.12), int(height * 0.10)), -8, 0, 360, (55, 64, 72), -1)
    cv2.ellipse(mask, palm_center, (int(width * 0.12), int(height * 0.10)), -8, 0, 360, 255, -1)

    for index, x_offset in enumerate((-90, -50, -10, 30, 70)):
        base_x = int(width * 0.52 + x_offset * width / 640.0)
        base_y = int(height * 0.42)
        tip_x = int(base_x + (index - 2) * 8 * width / 640.0)
        tip_y = int(height * (0.18 + 0.015 * abs(index - 2)))
        cv2.line(frame, (base_x, base_y), (tip_x, tip_y), (66, 76, 84), thickness=max(6, width // 70))
        cv2.line(mask, (base_x, base_y), (tip_x, tip_y), 255, thickness=max(6, width // 70))
        cv2.circle(frame, (tip_x, tip_y), max(5, width // 80), (72, 82, 90), -1)
        cv2.circle(mask, (tip_x, tip_y), max(5, width // 80), 255, -1)

    cv2.rectangle(frame, (int(width * 0.04), int(height * 0.64)), (int(width * 0.22), int(height * 0.74)), (110, 120, 130), -1)
    cv2.rectangle(mask, (int(width * 0.04), int(height * 0.64)), (int(width * 0.22), int(height * 0.74)), 255, -1)
    return frame, mask


def _moving_foreground(base_frame: np.ndarray, base_mask: np.ndarray, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    cv2 = _require_cv2()
    height, width = base_frame.shape[:2]
    dx = int(round(np.sin(frame_index * 0.08) * width * 0.035))
    dy = int(round(np.cos(frame_index * 0.06) * height * 0.018))
    matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    frame = cv2.warpAffine(base_frame, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpAffine(base_mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return frame, mask


def generate_previews(
    output_dir: Path,
    *,
    width: int = 640,
    height: int = 480,
    frames: int = 120,
    fps: float = 20.0,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_frame, base_mask = _demo_foreground(width, height)

    outputs: dict[str, str] = {}
    streams = {
        "temporal_block_shuffle": temporal_background_from_episode(
            width=width,
            height=height,
            episode_index=0,
            camera_key="right_wrist",
            mode="block_shuffle",
        ),
        "temporal_patch_mosaic": temporal_background_from_episode(
            width=width,
            height=height,
            episode_index=1,
            camera_key="right_wrist",
            mode="patch_mosaic",
        ),
        "temporal_lowfreq_noise": temporal_background_from_episode(
            width=width,
            height=height,
            episode_index=2,
            camera_key="left_wrist",
            mode="lowfreq_noise",
        ),
    }

    for name, stream in streams.items():
        path = output_dir / f"{name}_composite.mp4"
        writer = _writer(path, width=width, height=height, fps=fps)
        try:
            for frame_index in range(frames):
                frame, mask = _moving_foreground(base_frame, base_mask, frame_index)
                writer.write(stream.apply(frame, mask, frame_index, edge_blur_px=2))
        finally:
            writer.release()
        outputs[name] = str(path)

    path = output_dir / "background_only_temporal_motion.mp4"
    stream = temporal_background_from_episode(width=width, height=height, episode_index=9, camera_key="right_wrist")
    writer = _writer(path, width=width, height=height, fps=fps)
    try:
        for frame_index in range(frames):
            writer.write(stream.frame(frame_index))
    finally:
        writer.release()
    outputs["background_only"] = str(path)

    path = output_dir / "bad_frame_independent_random_reference.mp4"
    writer = _writer(path, width=width, height=height, fps=fps)
    try:
        for frame_index in range(frames):
            frame, mask = _moving_foreground(base_frame, base_mask, frame_index)
            background = generate_background(width, height, seed=900000 + frame_index, mode="embodied_mix")
            writer.write(apply_random_background(frame, mask, background=background, edge_blur_px=2))
    finally:
        writer.release()
    outputs["bad_frame_independent_reference"] = str(path)

    manifest_path = output_dir / "preview_manifest.json"
    manifest_path.write_text(json.dumps(outputs, indent=2, ensure_ascii=True), encoding="utf-8")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate random-background augmentation preview videos.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "samples")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=20.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outputs = generate_previews(
        args.output_dir,
        width=args.width,
        height=args.height,
        frames=args.frames,
        fps=args.fps,
    )
    print(json.dumps(outputs, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
