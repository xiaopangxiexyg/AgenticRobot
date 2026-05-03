import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


DEFAULT_VIDEO_CANDIDATES = (
    "color_aligned.mp4",
    "rgb.mp4",
    "color.mp4",
    "rgb.mkv",
)


@dataclass
class VideoProperties:
    path: str
    file_name: str
    width: int
    height: int
    fps: float
    frame_count: int


class PortableVideoWriter:
    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        self.output_path = os.path.abspath(output_path)
        self.fps = float(fps)
        self.width = int(frame_size[0])
        self.height = int(frame_size[1])
        self.backend = ""
        self._released = False
        self._cv2_writer = None
        self._ffmpeg_process = None
        self._ffmpeg_stderr = None
        self._temp_output_path = ""

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            self._start_ffmpeg_writer(ffmpeg_bin)
            self.backend = "ffmpeg"
            return

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(self.output_path, fourcc, float(self.fps), (self.width, self.height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open output video writer: {self.output_path}")
        self._cv2_writer = writer
        self.backend = "cv2"

    def _build_temp_output_path(self) -> str:
        path = Path(self.output_path)
        suffix = path.suffix
        if suffix:
            return str(path.with_name(f"{path.stem}.tmp{suffix}"))
        return str(path.with_name(f"{path.name}.tmp"))

    def _start_ffmpeg_writer(self, ffmpeg_bin: str) -> None:
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        self._temp_output_path = self._build_temp_output_path()
        if os.path.exists(self._temp_output_path):
            os.remove(self._temp_output_path)
        command = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.12f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            self._temp_output_path,
        ]
        self._ffmpeg_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
        )

    def isOpened(self) -> bool:
        if self.backend == "ffmpeg":
            return self._ffmpeg_process is not None and self._ffmpeg_process.stdin is not None
        return self._cv2_writer is not None and bool(self._cv2_writer.isOpened())

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"Expected frame shape {(self.height, self.width, 3)!r}, got {array.shape!r}")
        if array.shape[0] != self.height or array.shape[1] != self.width:
            raise ValueError(f"Expected frame shape {(self.height, self.width, 3)!r}, got {array.shape!r}")
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating):
                array = np.clip(np.round(array), 0.0, 255.0).astype(np.uint8)
            else:
                array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    def write(self, frame: np.ndarray) -> None:
        safe_frame = self._prepare_frame(frame)
        if self.backend == "ffmpeg":
            if self._ffmpeg_process is None or self._ffmpeg_process.stdin is None:
                raise RuntimeError("ffmpeg writer is not open")
            try:
                self._ffmpeg_process.stdin.write(safe_frame.tobytes())
            except BrokenPipeError as exc:
                stderr = b""
                if self._ffmpeg_process.stderr is not None:
                    stderr = self._ffmpeg_process.stderr.read()
                message = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ffmpeg video writer pipe broke: {message}") from exc
            return
        if self._cv2_writer is None:
            raise RuntimeError("cv2 writer is not open")
        self._cv2_writer.write(safe_frame)

    def release(self) -> None:
        if self._released:
            return
        self._released = True

        if self.backend == "ffmpeg":
            process = self._ffmpeg_process
            if process is None:
                return
            if process.stdin is not None:
                process.stdin.close()
            stderr = b""
            if process.stderr is not None:
                stderr = process.stderr.read()
            return_code = process.wait()
            if return_code != 0:
                if self._temp_output_path and os.path.exists(self._temp_output_path):
                    os.remove(self._temp_output_path)
                message = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ffmpeg video writer failed for {self.output_path}: {message}")
            os.replace(self._temp_output_path, self.output_path)
            return

        if self._cv2_writer is not None:
            self._cv2_writer.release()


class RobustVideoReader:
    def __init__(
        self,
        video_path: str,
        expected_frame_count: Optional[int] = None,
        scan_ahead_limit: int = 8,
    ) -> None:
        self.video_path = os.path.abspath(video_path)
        self.expected_frame_count = None if expected_frame_count is None else max(0, int(expected_frame_count))
        self.scan_ahead_limit = max(1, int(scan_ahead_limit))
        self.capture = self._open_capture()
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.probed_frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.next_frame_index = 0
        self.last_frame = None
        self.decode_recovery_count = 0
        self.fallback_copy_count = 0
        self.scan_ahead_recovery_count = 0
        self.scan_ahead_frames = 0
        self.failed_recovery_count = 0
        self.finished = False

    def _open_capture(self):
        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        return capture

    def _reopen_and_seek(self, frame_index: int) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = self._open_capture()
        if frame_index > 0:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

    def _read_frame_at(self, frame_index: int):
        self._reopen_and_seek(frame_index)
        ret, frame = self.capture.read()
        if not ret:
            return None
        return frame

    def _frame_limit(self) -> Optional[int]:
        if self.expected_frame_count is not None:
            return int(self.expected_frame_count)
        if self.probed_frame_count > 0:
            return int(self.probed_frame_count)
        return None

    def _recover_current_frame(self):
        current_index = int(self.next_frame_index)
        recovered = self._read_frame_at(current_index)
        if recovered is not None:
            self.decode_recovery_count += 1
            self.last_frame = recovered
            self.next_frame_index += 1
            return True, recovered

        if self.last_frame is not None:
            self.decode_recovery_count += 1
            self.fallback_copy_count += 1
            self.next_frame_index += 1
            return True, self.last_frame.copy()

        frame_limit = self._frame_limit()
        max_index = current_index + self.scan_ahead_limit
        if frame_limit is not None:
            max_index = min(max_index, max(0, frame_limit - 1))
        for candidate_index in range(current_index + 1, max_index + 1):
            recovered = self._read_frame_at(candidate_index)
            if recovered is None:
                continue
            self.decode_recovery_count += 1
            self.scan_ahead_recovery_count += 1
            self.scan_ahead_frames += int(candidate_index - current_index)
            self.last_frame = recovered
            self.next_frame_index = candidate_index + 1
            return True, recovered

        self.failed_recovery_count += 1
        self.finished = True
        return False, None

    def read(self):
        if self.finished:
            return False, None

        frame_limit = self._frame_limit()
        if frame_limit is not None and self.next_frame_index >= frame_limit:
            self.finished = True
            return False, None

        ret, frame = self.capture.read()
        if ret:
            self.last_frame = frame
            self.next_frame_index += 1
            return True, frame

        return self._recover_current_frame()

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.finished = True

    def stats(self) -> dict:
        return {
            "decode_recovery_count": int(self.decode_recovery_count),
            "fallback_copy_count": int(self.fallback_copy_count),
            "scan_ahead_recovery_count": int(self.scan_ahead_recovery_count),
            "scan_ahead_frames": int(self.scan_ahead_frames),
            "failed_recovery_count": int(self.failed_recovery_count),
        }


def resolve_video_source(
    camera_dir: str,
    candidates: Sequence[str] = DEFAULT_VIDEO_CANDIDATES,
) -> Optional[str]:
    for file_name in candidates:
        candidate = os.path.join(camera_dir, file_name)
        if os.path.isfile(candidate):
            return candidate
    return None


def probe_video_file(video_path: str) -> Optional[VideoProperties]:
    if not os.path.isfile(video_path):
        return None
    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            return None
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return VideoProperties(
        path=os.path.abspath(video_path),
        file_name=os.path.basename(video_path),
        width=width,
        height=height,
        fps=fps if fps > 0.0 else 30.0,
        frame_count=max(0, frame_count),
    )


def resolve_and_probe_video_source(
    camera_dir: str,
    candidates: Sequence[str] = DEFAULT_VIDEO_CANDIDATES,
) -> Optional[VideoProperties]:
    video_path = resolve_video_source(camera_dir, candidates=candidates)
    if video_path is None:
        return None
    return probe_video_file(video_path)


def transcode_video_to_mp4(
    input_path: str,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_output_path = output_path + ".tmp.mp4"
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-g",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        temp_output_path,
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        stderr = result.stderr.strip()
        raise RuntimeError(f"ffmpeg transcode failed for {input_path}: {stderr}")
    os.replace(temp_output_path, output_path)


def copy_video_to_mp4(
    input_path: str,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_output_path = output_path + ".tmp.mp4"
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)
    shutil.copy2(input_path, temp_output_path)
    os.replace(temp_output_path, output_path)


def export_video_to_mp4(
    input_path: str,
    output_path: str,
) -> None:
    input_ext = os.path.splitext(input_path)[1].lower()
    if input_ext == ".mp4":
        copy_video_to_mp4(input_path, output_path)
        return
    transcode_video_to_mp4(input_path, output_path)
