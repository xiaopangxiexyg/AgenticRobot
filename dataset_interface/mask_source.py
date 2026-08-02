from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from .metadata import DexTeleopDatasetIndex
from utils.random_background.mask_io import BitpackMaskReader


def episode_tag(episode_index: int) -> str:
    return f"episode_{int(episode_index):06d}"


class BitpackMaskReaderCache:
    """LRU cache for final keep-mask bitpack readers."""

    def __init__(self, max_size: int = 64):
        self.max_size = max(1, int(max_size))
        self.items: OrderedDict[Path, BitpackMaskReader] = OrderedDict()

    def close(self) -> None:
        for reader in self.items.values():
            reader.close()
        self.items.clear()

    def reader(self, manifest_path: str | Path) -> BitpackMaskReader:
        path = Path(manifest_path).expanduser().resolve()
        reader = self.items.get(path)
        if reader is not None:
            self.items.move_to_end(path)
            return reader
        reader = BitpackMaskReader(path)
        self.items[path] = reader
        if len(self.items) > self.max_size:
            _, evicted = self.items.popitem(last=False)
            evicted.close()
        return reader

    def read(self, manifest_path: str | Path, frame_index: int) -> np.ndarray:
        return self.reader(manifest_path).read(frame_index)


class FinalMaskSource:
    """Resolve final keep-mask manifests without a depth-mask fallback."""

    def __init__(
        self,
        *,
        dataset_index: DexTeleopDatasetIndex,
        final_mask_root: str | Path,
        cache_size: int = 64,
        manifest_name: str = "manifest.json",
    ):
        self.dataset_index = dataset_index
        self.final_mask_root = Path(final_mask_root).expanduser().resolve()
        self.manifest_name = str(manifest_name)
        self.cache = BitpackMaskReaderCache(max_size=cache_size)

    def close(self) -> None:
        self.cache.close()

    def manifest_path(self, episode_index: int, camera_key: str) -> Path:
        explicit = self.dataset_index.explicit_mask_path(episode_index, camera_key)
        path = (
            explicit
            if explicit is not None
            else (
                self.final_mask_root
                / self.dataset_index.dataset_name
                / episode_tag(episode_index)
                / str(camera_key)
                / self.manifest_name
            )
        )
        if not path.is_file():
            raise FileNotFoundError(
                "Final keep-mask manifest is missing: "
                f"{path}. Capture-time wrist_depth_mask_paths are not an "
                "implicit training fallback; generate the final keep-mask or "
                "provide final_mask_paths/wrist_final_mask_paths/"
                "wrist_keep_mask_paths."
            )
        return path.resolve()

    def read(self, episode_index: int, camera_key: str, frame_index: int) -> tuple[np.ndarray, Path]:
        manifest = self.manifest_path(episode_index, camera_key)
        return self.cache.read(manifest, frame_index), manifest

    def read_clip(
        self,
        episode_index: int,
        camera_key: str,
        start_frame: int,
        length: int,
        stride: int = 1,
    ) -> tuple[np.ndarray, Path]:
        manifest = self.manifest_path(episode_index, camera_key)
        frames = [
            self.cache.read(manifest, int(start_frame) + offset * int(stride))
            for offset in range(int(length))
        ]
        return np.stack(frames, axis=0), manifest
