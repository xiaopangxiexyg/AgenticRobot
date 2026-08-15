from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from dataset_contracts.wrist_depth import WRIST_DEPTH_MASK_FORMAT, unpack_binary_mask


MaskRole = Literal["keep", "replace"]


def _resolve_stream_path(manifest_path: Path, payload: Mapping[str, Any]) -> Path:
    stream = payload.get("stream")
    if not isinstance(stream, Mapping):
        raise ValueError(f"mask manifest has no stream object: {manifest_path}")
    raw_path = stream.get("path")
    if not raw_path:
        raise ValueError(f"mask manifest stream has no path: {manifest_path}")
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    candidates = [
        (manifest_path.parent / path).resolve(),
        (manifest_path.parent / path.name).resolve(),
    ]
    relative_dir = payload.get("relative_dir")
    if isinstance(relative_dir, str) and relative_dir:
        rel = Path(relative_dir)
        if rel.is_absolute():
            candidates.append((rel / path.name).resolve())
            candidates.append((rel.parent / path).resolve())
        else:
            rel_parts = rel.parts
            manifest_parts = manifest_path.parent.parts
            if len(manifest_parts) >= len(rel_parts) and manifest_parts[-len(rel_parts) :] == rel_parts:
                root = Path(*manifest_parts[: -len(rel_parts)])
                candidates.append((root / path).resolve())
                candidates.append((root / rel / path.name).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@dataclass(frozen=True)
class BitpackMaskManifest:
    manifest_path: Path
    stream_path: Path
    camera_key: str
    width: int
    height: int
    frame_count: int
    packed_bytes_per_frame: int
    semantics: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "BitpackMaskManifest":
        manifest_path = Path(path).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"mask manifest must contain an object: {manifest_path}")
        fmt = payload.get("format")
        if fmt != WRIST_DEPTH_MASK_FORMAT:
            raise ValueError(f"unsupported mask manifest format {fmt!r}: {manifest_path}")
        stream = payload.get("stream")
        if not isinstance(stream, Mapping):
            raise ValueError(f"mask manifest has no stream object: {manifest_path}")
        shape = stream.get("shape")
        if not isinstance(shape, list | tuple) or len(shape) != 2:
            raise ValueError(f"mask stream shape must be [height, width]: {manifest_path}")
        height, width = int(shape[0]), int(shape[1])
        packed = int(stream.get("packed_bytes_per_frame") or ((height * width + 7) // 8))
        return cls(
            manifest_path=manifest_path,
            stream_path=_resolve_stream_path(manifest_path, payload),
            camera_key=str(payload.get("camera_key") or ""),
            width=width,
            height=height,
            frame_count=int(stream.get("frame_count") or 0),
            packed_bytes_per_frame=packed,
            semantics=dict(payload.get("semantics") or {}),
        )


class BitpackMaskReader:
    """Random-access reader for DexTeleop bit-packed 0/1 mask streams."""

    def __init__(self, manifest: str | Path | BitpackMaskManifest):
        self.manifest = manifest if isinstance(manifest, BitpackMaskManifest) else BitpackMaskManifest.load(manifest)
        self._file = self.manifest.stream_path.open("rb")

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "BitpackMaskReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read(self, frame_index: int) -> np.ndarray:
        index = int(frame_index)
        if index < 0:
            raise IndexError(f"frame_index must be non-negative, got {frame_index}")
        if self.manifest.frame_count > 0 and index >= self.manifest.frame_count:
            raise IndexError(f"frame_index {index} is outside mask frame_count {self.manifest.frame_count}")
        offset = index * int(self.manifest.packed_bytes_per_frame)
        self._file.seek(offset)
        data = self._file.read(int(self.manifest.packed_bytes_per_frame))
        if len(data) != int(self.manifest.packed_bytes_per_frame):
            raise EOFError(f"failed to read packed mask frame {index} from {self.manifest.stream_path}")
        return unpack_binary_mask(data, height=self.manifest.height, width=self.manifest.width)


@lru_cache(maxsize=128)
def cached_manifest(path: str) -> BitpackMaskManifest:
    return BitpackMaskManifest.load(path)


def normalize_keep_mask(mask: Any, *, mask_role: MaskRole = "keep") -> np.ndarray:
    """Return uint8 0/255 keep mask from a 0/1 or 0/255 input mask."""

    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D or single-channel image, got shape={array.shape!r}")
    binary = array > 0
    if mask_role == "replace":
        binary = ~binary
    elif mask_role != "keep":
        raise ValueError(f"mask_role must be 'keep' or 'replace', got {mask_role!r}")
    return np.where(binary, 255, 0).astype(np.uint8)
