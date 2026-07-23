"""Readers for stable external tactile stream encodings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _resolve_dataset_path(root: Path, path_text: Any) -> Path | None:
    if path_text is None or not str(path_text):
        return None
    path = Path(str(path_text)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def tactile_stream_compression(spec: Mapping[str, Any]) -> str:
    compression = str(spec.get("compression") or "").strip().lower().replace("-", "_")
    if compression in {"", "raw", "uncompressed"}:
        encoding = str(spec.get("encoding") or "").strip().lower().replace("-", "_")
        if encoding.startswith("chunked_"):
            compression = encoding.removeprefix("chunked_")
            if compression == "delta_zstd":
                compression = "zstd"
        else:
            compression = "none"
    if compression == "raw_c_order":
        compression = "none"
    return compression


def _decompress_chunk(payload: bytes, *, compression: str, raw_bytes: int) -> bytes:
    if compression == "none":
        return payload
    if compression == "lz4":
        try:
            import lz4.frame
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading lz4-compressed tactile streams requires the lz4 Python package."
            ) from exc
        return lz4.frame.decompress(payload)
    if compression == "zstd":
        try:
            import zstandard as zstd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading zstd-compressed tactile streams requires the zstandard Python package."
            ) from exc
        return zstd.ZstdDecompressor().decompress(payload, max_output_size=int(raw_bytes))
    raise ValueError(f"Unsupported tactile stream compression '{compression}'.")


class ChunkedTactileStream:
    """Random-access reader for the writer's chunked tactile stream contract."""

    def __init__(self, *, dataset_root: Path, spec: Mapping[str, Any]):
        self.dataset_root = Path(dataset_root)
        self.spec = dict(spec)
        path = _resolve_dataset_path(self.dataset_root, self.spec.get("path"))
        if path is None:
            raise ValueError("Compressed tactile stream is missing a path.")
        self.path = path
        self.shape = tuple(int(dim) for dim in self.spec.get("shape", ()) or ())
        self.frame_count = int(self.spec.get("frame_count") or 0)
        self.dtype = np.dtype(str(self.spec.get("dtype") or "float32"))
        self.storage_dtype = np.dtype(
            str(self.spec.get("storage_dtype") or self.spec.get("dtype") or "float32")
        )
        self.compression = tactile_stream_compression(self.spec)
        self.chunks = [
            dict(chunk)
            for chunk in self.spec.get("chunks", []) or []
            if isinstance(chunk, Mapping)
        ]
        self._cached_chunk_index: int | None = None
        self._cached_chunk_array: np.ndarray | None = None

    def _chunk_index_for_frame(self, frame_index: int) -> int | None:
        index = int(frame_index)
        for chunk_index, chunk in enumerate(self.chunks):
            start = int(chunk.get("frame_start") or 0)
            count = int(chunk.get("frame_count") or 0)
            if start <= index < start + count:
                return int(chunk_index)
        return None

    def _load_chunk(self, chunk_index: int) -> np.ndarray:
        if self._cached_chunk_index == int(chunk_index) and self._cached_chunk_array is not None:
            return self._cached_chunk_array
        chunk = self.chunks[int(chunk_index)]
        offset = int(chunk.get("offset") or 0)
        compressed_bytes = int(chunk.get("compressed_bytes") or 0)
        raw_bytes = int(chunk.get("raw_bytes") or 0)
        frame_count = int(chunk.get("frame_count") or 0)
        if compressed_bytes <= 0 or raw_bytes <= 0 or frame_count <= 0:
            raise ValueError(f"Invalid tactile chunk metadata in {self.path}.")
        with self.path.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read(compressed_bytes)
        if len(payload) != compressed_bytes:
            raise ValueError(f"Short tactile chunk read from {self.path}.")
        raw = _decompress_chunk(payload, compression=self.compression, raw_bytes=raw_bytes)
        if len(raw) != raw_bytes:
            raise ValueError(
                f"Tactile chunk decompressed to {len(raw)} bytes, expected {raw_bytes}."
            )
        array = np.frombuffer(raw, dtype=self.storage_dtype).reshape((frame_count, *self.shape))
        encoding = str(self.spec.get("encoding") or "").strip().lower().replace("-", "_")
        if encoding == "chunked_delta_zstd":
            predictor = str(chunk.get("predictor") or self.spec.get("predictor") or "")
            if predictor != "delta_mod_uint8_bias_128":
                raise ValueError(f"Unsupported tactile delta predictor '{predictor}'.")
            if self.storage_dtype != np.dtype(np.uint8):
                raise ValueError("delta_mod_uint8_bias_128 requires uint8 storage.")
            restored = np.empty_like(array)
            restored[0] = array[0]
            for index in range(1, frame_count):
                restored[index] = (
                    restored[index - 1].astype(np.int16)
                    + array[index].astype(np.int16)
                    - 128
                ).astype(np.uint8)
            array = restored
        quantization = chunk.get("quantization")
        if isinstance(quantization, Mapping):
            scheme = str(quantization.get("scheme") or "")
            if scheme != "symmetric_per_last_axis_int16":
                raise ValueError(f"Unsupported tactile quantization scheme '{scheme}'.")
            scales = np.asarray(quantization.get("scales") or (), dtype=np.float32)
            if self.storage_dtype != np.dtype(np.int16):
                raise ValueError("symmetric_per_last_axis_int16 requires int16 storage.")
            if not self.shape or scales.shape != (self.shape[-1],):
                raise ValueError(
                    f"Invalid tactile quantization scales {scales.shape}; expected {(self.shape[-1],)}."
                )
            array = array.astype(np.float32) * scales.reshape(
                (1,) * (array.ndim - 1) + (self.shape[-1],)
            )
        elif self.storage_dtype != self.dtype:
            raise ValueError(
                f"Tactile storage dtype {self.storage_dtype} differs from logical dtype "
                f"{self.dtype} without a quantization contract."
            )
        self._cached_chunk_index = int(chunk_index)
        self._cached_chunk_array = array
        return array

    def frame(self, frame_index: int) -> np.ndarray | None:
        index = int(frame_index)
        if index < 0 or index >= int(self.frame_count):
            return None
        chunk_index = self._chunk_index_for_frame(index)
        if chunk_index is None:
            return None
        chunk = self.chunks[chunk_index]
        start = int(chunk.get("frame_start") or 0)
        array = self._load_chunk(chunk_index)
        offset = index - start
        if offset < 0 or offset >= int(array.shape[0]):
            return None
        return np.asarray(array[offset])


__all__ = ["ChunkedTactileStream", "tactile_stream_compression"]
