from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from .intrinsics import CameraIntrinsics, build_undistort_maps, require_cv2


SizePolicy = Literal["strict", "scale"]


@dataclass(frozen=True)
class VideoProcessResult:
    input_path: Path
    output_path: Path
    frame_count: int
    fps: float
    width: int
    height: int
    elapsed_sec: float
    output_intrinsics: CameraIntrinsics

    @property
    def processing_fps(self) -> float:
        if self.elapsed_sec <= 0.0:
            return 0.0
        return float(self.frame_count) / float(self.elapsed_sec)

    def to_dict(self) -> dict[str, object]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "frame_count": int(self.frame_count),
            "fps": float(self.fps),
            "width": int(self.width),
            "height": int(self.height),
            "elapsed_sec": float(self.elapsed_sec),
            "processing_fps": float(self.processing_fps),
            "output_intrinsics": self.output_intrinsics.to_dict(),
        }


def _valid_fps(value: float) -> float:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return 30.0
    if fps > 0.0 and fps < 1000.0:
        return fps
    return 30.0


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
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    intrinsics: CameraIntrinsics,
    *,
    alpha: float = 0.0,
    codec: str = "mp4v",
    size_policy: SizePolicy = "strict",
    max_frames: int = 0,
    expected_frame_count: int | None = None,
    overwrite: bool = False,
    progress: Callable[[int], None] | None = None,
) -> VideoProcessResult:
    """Undistort a video with precomputed maps and preserve frame count/timing."""

    cv2 = require_cv2()
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input video does not exist: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output video already exists: {output_path}")
    if expected_frame_count is not None and int(expected_frame_count) <= 0:
        raise ValueError(
            f"expected_frame_count must be positive when provided, got {expected_frame_count}"
        )
    temporary_output_path = output_path.with_name(
        f".{output_path.stem}.partial-{os.getpid()}-{uuid.uuid4().hex}{output_path.suffix}"
    )

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")
    writer = None
    started = time.perf_counter()
    frame_count = 0
    processing_succeeded = False
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

        effective_intrinsics = intrinsics
        if (width, height) != (intrinsics.width, intrinsics.height):
            if size_policy == "scale":
                effective_intrinsics = intrinsics.scaled_to(width, height)
            else:
                raise ValueError(
                    f"video size {width}x{height} does not match intrinsics "
                    f"{intrinsics.width}x{intrinsics.height}; use --size-policy scale "
                    "only if the video was uniformly resized from the recorded profile."
                )

        maps = build_undistort_maps(effective_intrinsics, alpha=alpha)
        writer = _open_writer(
            temporary_output_path,
            codec=codec,
            fps=fps,
            width=width,
            height=height,
        )

        while True:
            if max_frames > 0 and frame_count >= int(max_frames):
                break
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(maps.remap(frame))
            frame_count += 1
            if progress is not None:
                progress(frame_count)
        processing_succeeded = True
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not processing_succeeded:
            temporary_output_path.unlink(missing_ok=True)
    try:
        if expected_frame_count is not None and frame_count != int(expected_frame_count):
            raise RuntimeError(
                "decoded frame count does not satisfy the dataset contract: "
                f"input={input_path}, expected={int(expected_frame_count)}, decoded={frame_count}"
            )
        os.replace(temporary_output_path, output_path)
    except BaseException:
        temporary_output_path.unlink(missing_ok=True)
        raise
    elapsed = time.perf_counter() - started
    return VideoProcessResult(
        input_path=input_path,
        output_path=output_path,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        elapsed_sec=elapsed,
        output_intrinsics=maps.output,
    )
