from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("dataset_interface video reading requires opencv-python.") from exc
    return cv2


class CachedVideoReader:
    def __init__(self, path: str | Path):
        cv2 = _require_cv2()
        self.path = Path(path).expanduser().resolve()
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise RuntimeError(f"failed to open video: {self.path}")
        self._next_index: int | None = None

    def close(self) -> None:
        self.capture.release()

    @property
    def frame_count(self) -> int:
        cv2 = _require_cv2()
        return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def read(self, frame_index: int) -> np.ndarray:
        cv2 = _require_cv2()
        index = int(frame_index)
        if index < 0:
            raise IndexError(f"frame_index must be non-negative, got {frame_index}")
        if self._next_index != index:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise EOFError(f"failed to read frame {index} from {self.path}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"expected HxWx3 frame from {self.path}, got {frame.shape!r}")
        self._next_index = index + 1
        return frame

    def read_clip(self, start_frame: int, length: int, stride: int = 1) -> np.ndarray:
        indices = [int(start_frame) + offset * int(stride) for offset in range(int(length))]
        frames = [self.read(index) for index in indices]
        return np.stack(frames, axis=0)


class VideoReaderCache:
    """LRU cache for OpenCV video readers inside one DataLoader worker."""

    def __init__(self, max_size: int = 16):
        self.max_size = max(1, int(max_size))
        self.items: OrderedDict[Path, CachedVideoReader] = OrderedDict()

    def close(self) -> None:
        for reader in self.items.values():
            reader.close()
        self.items.clear()

    def reader(self, path: str | Path) -> CachedVideoReader:
        key = Path(path).expanduser().resolve()
        reader = self.items.get(key)
        if reader is not None:
            self.items.move_to_end(key)
            return reader
        reader = CachedVideoReader(key)
        self.items[key] = reader
        if len(self.items) > self.max_size:
            _, evicted = self.items.popitem(last=False)
            evicted.close()
        return reader

    def read(self, path: str | Path, frame_index: int) -> np.ndarray:
        return self.reader(path).read(frame_index)

    def read_clip(self, path: str | Path, start_frame: int, length: int, stride: int = 1) -> np.ndarray:
        return self.reader(path).read_clip(start_frame, length, stride)
