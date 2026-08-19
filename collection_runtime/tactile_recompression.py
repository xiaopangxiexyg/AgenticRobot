"""Device-neutral tactile persistence and verified lossless recompression.

The collection runtime owns the mutation mechanism and its integrity
receipts.  Offline tooling and RealTeleop bind this capability without
duplicating the transform or owning its commit protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from dataset_interface.modality_artifacts import PAYLOAD_HASH_PREFER_SEALED

from .modality_inventory import publish_tracked_modality_inventory
from .process_quiescence import (
    active_formal_collection_processes as _active_collection_processes,
)


DEFAULT_CHUNK_FRAMES = 8
DEFAULT_MIN_FREE_BYTES = 20 * 1024**3
COLLECTION_RECHECK_CHUNKS = 32
LOSSLESS_PROFILE_V1 = "dmtac_raw_temporal_delta_zstd_v1"
LOSSLESS_PREDICTOR_V1 = "delta_mod_uint8"


def _publish_updated_modality_inventory(dataset_root: Path) -> str | None:
    return publish_tracked_modality_inventory(
        dataset_root,
        payload_hash_mode=PAYLOAD_HASH_PREFER_SEALED,
    )


class TactileRecompressionCancelled(RuntimeError):
    """Raised at a verified chunk boundary when idle maintenance is revoked."""


@dataclass(frozen=True)
class TactileRecompressionPlan:
    dataset_root: str
    manifest_count: int
    stream_count: int
    raw_bytes: int
    largest_manifest_raw_bytes: int
    free_bytes: int
    min_free_bytes: int
    skipped_compressed_streams: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TactileRecompressionResult:
    dataset_root: str
    manifest_count: int
    stream_count: int
    raw_bytes: int
    compressed_bytes: int
    removed_original_bytes: int
    compression: str = "zstd"
    profile: str = LOSSLESS_PROFILE_V1
    receipt_path: str | None = None

    @property
    def compression_ratio(self) -> float | None:
        if self.raw_bytes <= 0:
            return None
        return float(self.compressed_bytes) / float(self.raw_bytes)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["compression_ratio"] = self.compression_ratio
        return payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, ensure_ascii=True, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_dataset_path(root: Path, value: Any) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Tactile stream path is empty.")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Tactile stream escapes dataset root: {resolved}") from exc
    return resolved


def _relative_dataset_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _compression_name(spec: Mapping[str, Any]) -> str:
    compression = str(spec.get("compression") or "").strip().lower().replace("-", "_")
    if compression:
        return "none" if compression in {"raw", "uncompressed", "raw_c_order"} else compression
    encoding = str(spec.get("encoding") or "").strip().lower().replace("-", "_")
    if encoding.startswith("chunked_"):
        return encoding.removeprefix("chunked_")
    return "none"


def _finalized_manifest_paths(dataset_root: Path) -> list[Path]:
    episode_rows = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    if not episode_rows:
        raise ValueError(
            f"No finalized episodes found in {dataset_root / 'meta' / 'episodes.jsonl'}."
        )
    relative_paths: set[str] = set()
    for row in episode_rows:
        tactile_paths = row.get("tactile_paths")
        if not isinstance(tactile_paths, Mapping):
            continue
        for value in tactile_paths.values():
            if value:
                relative_paths.add(str(value))
    manifests = [_resolve_dataset_path(dataset_root, value) for value in sorted(relative_paths)]
    missing = [path for path in manifests if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Finalized tactile manifest is missing: {missing[0]}")
    return manifests


def _raw_stream_size(spec: Mapping[str, Any]) -> int:
    frame_count = int(spec.get("frame_count") or 0)
    shape = tuple(int(value) for value in spec.get("shape", ()) or ())
    dtype = np.dtype(str(spec.get("dtype") or ""))
    if frame_count <= 0 or not shape or any(value <= 0 for value in shape):
        raise ValueError(f"Invalid tactile stream frame_count/shape: {dict(spec)}")
    return int(frame_count * math.prod(shape) * dtype.itemsize)


def _raw_manifest_streams(
    dataset_root: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any], Path]]]:
    manifest = _read_json(manifest_path)
    if manifest.get("format") != "binary_stream_v1":
        raise ValueError(f"Unsupported tactile manifest format in {manifest_path}")
    streams = manifest.get("streams")
    if not isinstance(streams, Mapping):
        raise ValueError(f"Tactile manifest has no streams mapping: {manifest_path}")
    raw: list[tuple[str, dict[str, Any], Path]] = []
    for sensor_key, value in sorted(streams.items()):
        if not isinstance(value, Mapping):
            raise ValueError(f"Invalid stream spec for {sensor_key} in {manifest_path}")
        spec = dict(value)
        if _compression_name(spec) != "none":
            continue
        source = _resolve_dataset_path(dataset_root, spec.get("path"))
        if not source.is_file():
            raise FileNotFoundError(f"Raw tactile stream is missing: {source}")
        expected = _raw_stream_size(spec)
        actual = source.stat().st_size
        if actual != expected:
            raise ValueError(
                f"Raw tactile stream size mismatch for {source}: {actual} != {expected}"
            )
        raw.append((str(sensor_key), spec, source))
    return manifest, raw


def plan_tactile_zstd_recompression(
    dataset_root: str | Path,
    *,
    max_manifests: int | None = None,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
) -> TactileRecompressionPlan:
    root = Path(dataset_root).expanduser().resolve()
    manifests = _finalized_manifest_paths(root)
    if max_manifests is not None and int(max_manifests) <= 0:
        raise ValueError("max_manifests must be positive when provided.")
    manifest_count = 0
    stream_count = 0
    raw_bytes = 0
    largest_manifest = 0
    skipped = 0
    for manifest_path in manifests:
        manifest, raw_streams = _raw_manifest_streams(root, manifest_path)
        if not raw_streams:
            skipped += len(manifest.get("streams", {}))
            continue
        if max_manifests is not None and manifest_count >= int(max_manifests):
            break
        manifest_raw = sum(source.stat().st_size for _, _, source in raw_streams)
        raw_bytes += manifest_raw
        stream_count += len(raw_streams)
        manifest_count += 1
        largest_manifest = max(largest_manifest, manifest_raw)
        streams = manifest.get("streams", {})
        skipped += max(0, len(streams) - len(raw_streams))
    free_bytes = shutil.disk_usage(root).free
    return TactileRecompressionPlan(
        dataset_root=str(root),
        manifest_count=manifest_count,
        stream_count=stream_count,
        raw_bytes=raw_bytes,
        largest_manifest_raw_bytes=largest_manifest,
        free_bytes=free_bytes,
        min_free_bytes=int(min_free_bytes),
        skipped_compressed_streams=skipped,
    )


class _ZstdCodec:
    def __init__(self, *, level: int, backend: str = "auto") -> None:
        self.level = int(level)
        if self.level < 1 or self.level > 19:
            raise ValueError("zstd level must be in [1, 19].")
        requested = str(backend).strip().lower()
        if requested not in {"auto", "python", "cli"}:
            raise ValueError("zstd backend must be auto, python, or cli.")
        self._zstandard: Any | None = None
        if requested in {"auto", "python"}:
            try:
                import zstandard
            except ModuleNotFoundError:
                if requested == "python":
                    raise
            else:
                self._zstandard = zstandard
        self._cli = shutil.which("zstd") if self._zstandard is None else None
        if self._zstandard is None and not self._cli:
            raise ModuleNotFoundError(
                "Tactile Zstd recompression requires the zstandard Python package or zstd CLI."
            )

    @property
    def backend(self) -> str:
        return "python" if self._zstandard is not None else "cli"

    def compress(self, payload: bytes) -> bytes:
        if self._zstandard is not None:
            return self._zstandard.ZstdCompressor(level=self.level).compress(payload)
        assert self._cli is not None
        return subprocess.run(
            [self._cli, "-q", f"-{self.level}", "-c"],
            input=payload,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout

    def decompress(self, payload: bytes, *, raw_bytes: int) -> bytes:
        if self._zstandard is not None:
            return self._zstandard.ZstdDecompressor().decompress(
                payload,
                max_output_size=int(raw_bytes),
            )
        assert self._cli is not None
        return subprocess.run(
            [self._cli, "-q", "-d", "-c"],
            input=payload,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout


def _compressed_path(source: Path, *, chunk_frames: int) -> Path:
    return source.with_name(f"{source.name}.tdelta{int(chunk_frames)}.zst")


def clean_uncommitted_tactile_recompression_artifacts(
    dataset_root: str | Path,
    *,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
) -> int:
    """Remove only deterministic compressed targets not referenced by manifests.

    A forced process stop can occur after the verified target inode was
    installed but before the raw manifest was atomically replaced.  In that
    state the manifest still names an intact raw source, so the compressed
    target is an uncommitted derivative and must be removed before retrying.
    This function never removes a manifest-referenced artifact or a raw source.
    """

    root = Path(dataset_root).expanduser().resolve()
    removed = 0
    touched_directories: set[Path] = set()
    for manifest_path in _finalized_manifest_paths(root):
        _manifest, raw_streams = _raw_manifest_streams(root, manifest_path)
        for _sensor_key, _spec, source in raw_streams:
            target = _compressed_path(source, chunk_frames=int(chunk_frames))
            if not target.exists():
                continue
            source_stat = source.stat()
            target_stat = target.stat()
            if target_stat.st_dev != source_stat.st_dev:
                raise RuntimeError(
                    f"Refusing cross-device uncommitted tactile target: {target}"
                )
            if target_stat.st_ino == source_stat.st_ino or target_stat.st_nlink != 1:
                raise RuntimeError(
                    f"Refusing unsafe uncommitted tactile target inode: {target}"
                )
            target.unlink()
            touched_directories.add(target.parent)
            removed += 1
    for directory in sorted(touched_directories):
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return removed


def _delta_encode(raw: bytes, *, frame_count: int, bytes_per_frame: int) -> bytes:
    values = np.frombuffer(raw, dtype=np.uint8).reshape((int(frame_count), int(bytes_per_frame)))
    encoded = np.empty_like(values)
    encoded[0] = values[0]
    if int(frame_count) > 1:
        encoded[1:] = (
            values[1:].astype(np.uint16) - values[:-1].astype(np.uint16)
        ).astype(np.uint8)
    return encoded.tobytes(order="C")


def _delta_decode(encoded: bytes, *, frame_count: int, bytes_per_frame: int) -> bytes:
    values = np.frombuffer(encoded, dtype=np.uint8).reshape(
        (int(frame_count), int(bytes_per_frame))
    )
    restored = np.empty_like(values)
    restored[0] = values[0]
    for index in range(1, int(frame_count)):
        restored[index] = (
            restored[index - 1].astype(np.int16) + values[index].astype(np.int16)
        ).astype(np.uint8)
    return restored.tobytes(order="C")


def _validate_compressed_stream(
    target: Path,
    chunks: Iterable[Mapping[str, Any]],
    *,
    codec: _ZstdCodec,
    bytes_per_frame: int,
    expected_sha256: str,
    cancel_requested: Callable[[], bool] | None = None,
) -> None:
    decoded_hash = hashlib.sha256()
    expected_offset = 0
    with target.open("rb") as handle:
        for chunk in chunks:
            if cancel_requested is not None and cancel_requested():
                raise TactileRecompressionCancelled(
                    f"Tactile recompression cancelled while validating {target}"
                )
            offset = int(chunk["offset"])
            compressed_bytes = int(chunk["compressed_bytes"])
            raw_bytes = int(chunk["raw_bytes"])
            frame_count = int(chunk["frame_count"])
            if offset != expected_offset:
                raise ValueError(f"Non-contiguous tactile chunk layout in {target}")
            handle.seek(offset)
            payload = handle.read(compressed_bytes)
            if len(payload) != compressed_bytes:
                raise ValueError(f"Short compressed tactile chunk read from {target}")
            encoded = codec.decompress(payload, raw_bytes=raw_bytes)
            if len(encoded) != raw_bytes:
                raise ValueError(f"Wrong decompressed tactile chunk size in {target}")
            decoded = _delta_decode(
                encoded,
                frame_count=frame_count,
                bytes_per_frame=bytes_per_frame,
            )
            decoded_hash.update(decoded)
            expected_offset += compressed_bytes
        if handle.read(1):
            raise ValueError(f"Compressed tactile stream has trailing bytes: {target}")
    if decoded_hash.hexdigest() != str(expected_sha256):
        raise ValueError(f"Decoded tactile SHA-256 mismatch for {target}")


def _compress_stream(
    source: Path,
    target: Path,
    spec: Mapping[str, Any],
    *,
    codec: _ZstdCodec,
    chunk_frames: int,
    cancel_requested: Callable[[], bool] | None = None,
    enforce_global_collection_exclusion: bool = True,
) -> tuple[list[dict[str, Any]], str, str, int]:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing compressed stream: {target}")
    if np.dtype(str(spec.get("dtype") or "")) != np.dtype(np.uint8):
        raise ValueError(
            f"{LOSSLESS_PROFILE_V1} only accepts logical uint8 raw tactile streams: {source}"
        )
    bytes_per_frame = _raw_stream_size({**dict(spec), "frame_count": 1})
    frame_count = int(spec["frame_count"])
    source_hash = hashlib.sha256()
    stored_hash = hashlib.sha256()
    chunks: list[dict[str, Any]] = []
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    compressed_total = 0
    try:
        with source.open("rb") as input_handle, os.fdopen(fd, "wb") as output_handle:
            frame_start = 0
            chunk_index = 0
            while frame_start < frame_count:
                if cancel_requested is not None and cancel_requested():
                    raise TactileRecompressionCancelled(
                        f"Tactile recompression cancelled before frame {frame_start} of {source}"
                    )
                if (
                    enforce_global_collection_exclusion
                    and chunk_index
                    and chunk_index % COLLECTION_RECHECK_CHUNKS == 0
                ):
                    active_collection = _active_collection_processes()
                    if active_collection:
                        raise RuntimeError(
                            "Formal collection started during tactile recompression: "
                            + "; ".join(active_collection[:3])
                        )
                count = min(int(chunk_frames), frame_count - frame_start)
                raw_bytes = count * bytes_per_frame
                raw = input_handle.read(raw_bytes)
                if len(raw) != raw_bytes:
                    raise ValueError(f"Short raw tactile chunk read from {source}")
                source_hash.update(raw)
                encoded = _delta_encode(
                    raw,
                    frame_count=count,
                    bytes_per_frame=bytes_per_frame,
                )
                compressed = codec.compress(encoded)
                restored_encoded = codec.decompress(compressed, raw_bytes=raw_bytes)
                restored = _delta_decode(
                    restored_encoded,
                    frame_count=count,
                    bytes_per_frame=bytes_per_frame,
                )
                if restored != raw:
                    raise ValueError(
                        f"Delta+Zstd round-trip mismatch for {source} at frame {frame_start}"
                    )
                offset = output_handle.tell()
                output_handle.write(compressed)
                stored_hash.update(compressed)
                chunks.append(
                    {
                        "frame_start": frame_start,
                        "frame_count": count,
                        "offset": int(offset),
                        "compressed_bytes": len(compressed),
                        "raw_bytes": raw_bytes,
                        "predictor": LOSSLESS_PREDICTOR_V1,
                    }
                )
                compressed_total += len(compressed)
                frame_start += count
                chunk_index += 1
            if input_handle.read(1):
                raise ValueError(f"Raw tactile stream has trailing bytes: {source}")
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        target_stat = target.stat()
        source_stat = source.stat()
        if target_stat.st_dev != source_stat.st_dev or target_stat.st_ino == source_stat.st_ino:
            raise ValueError(f"Tactile staging target is not an independent inode: {target}")
        if target_stat.st_nlink != 1:
            raise ValueError(f"Tactile staging target unexpectedly has links: {target}")
        digest = source_hash.hexdigest()
        _validate_compressed_stream(
            target,
            chunks,
            codec=codec,
            bytes_per_frame=bytes_per_frame,
            expected_sha256=digest,
            cancel_requested=cancel_requested,
        )
        return chunks, digest, stored_hash.hexdigest(), compressed_total
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def recompress_tactile_zstd_in_place(
    dataset_root: str | Path,
    *,
    level: int = 1,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    max_manifests: int | None = None,
    delete_originals: bool = False,
    backend: str = "auto",
    cancel_requested: Callable[[], bool] | None = None,
    enforce_global_collection_exclusion: bool = True,
) -> TactileRecompressionResult:
    if not delete_originals:
        raise ValueError(
            "In-place recompression requires delete_originals=True after verified manifest commit."
        )
    active_collection = (
        _active_collection_processes()
        if enforce_global_collection_exclusion
        else ()
    )
    if active_collection:
        raise RuntimeError(
            "Refusing offline tactile recompression while formal collection is active: "
            + "; ".join(active_collection[:3])
        )
    if int(chunk_frames) <= 0:
        raise ValueError("chunk_frames must be positive.")
    root = Path(dataset_root).expanduser().resolve()
    committed_v24 = sorted(
        (root / "meta" / "v24" / "control_trace" / "episodes").glob(
            "*/capture_commit.cjson"
        )
    )
    if committed_v24:
        raise RuntimeError(
            "Refusing in-place tactile recompression for a committed v2.4 capture; "
            "create a copy-on-write derived dataset instead."
        )
    manifests = _finalized_manifest_paths(root)
    if max_manifests is not None and int(max_manifests) <= 0:
        raise ValueError("max_manifests must be positive when provided.")
    codec = _ZstdCodec(level=int(level), backend=backend)
    transaction_id = uuid.uuid4().hex
    started_ns = time.time_ns()
    receipt_path = root / "meta" / "tactile_recompression" / f"{transaction_id}.json"
    processed_manifests = 0
    processed_streams = 0
    raw_total = 0
    compressed_total = 0
    removed_total = 0
    published_inventory_hash: str | None = None
    def current_result() -> TactileRecompressionResult:
        return TactileRecompressionResult(
            dataset_root=str(root),
            manifest_count=processed_manifests,
            stream_count=processed_streams,
            raw_bytes=raw_total,
            compressed_bytes=compressed_total,
            removed_original_bytes=removed_total,
            receipt_path=_relative_dataset_path(root, receipt_path),
        )

    def write_receipt(*, status: str, error: BaseException | None = None) -> None:
        payload: dict[str, Any] = {
            "schema": "dexteleop.tactile_lossless_recompression_receipt.v1",
            "transaction_id": transaction_id,
            "started_unix_ns": started_ns,
            "completed_unix_ns": time.time_ns(),
            "status": str(status),
            **current_result().to_dict(),
            "codec": {
                "compression": "zstd",
                "level": int(level),
                "chunk_frame_count": int(chunk_frames),
                "predictor": LOSSLESS_PREDICTOR_V1,
                "backend": codec.backend,
            },
        }
        if error is not None:
            payload["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "resume_safe": True,
            }
        if published_inventory_hash is not None:
            payload["published_modality_inventory_hash"] = published_inventory_hash
        _atomic_write_json(receipt_path, payload)

    try:
        for manifest_path in manifests:
            if cancel_requested is not None and cancel_requested():
                raise TactileRecompressionCancelled(
                    "Tactile recompression cancelled before the next manifest"
                )
            manifest, raw_streams = _raw_manifest_streams(root, manifest_path)
            if not raw_streams:
                continue
            if max_manifests is not None and processed_manifests >= int(max_manifests):
                break
            required = sum(source.stat().st_size for _, _, source in raw_streams)
            free = shutil.disk_usage(root).free
            if free - required < int(min_free_bytes):
                raise OSError(
                    f"Insufficient free space for atomic manifest conversion: free={free}, "
                    f"required<={required}, reserve={int(min_free_bytes)}"
                )
            created: list[Path] = []
            replacements: dict[str, dict[str, Any]] = {}
            source_stats: dict[Path, os.stat_result] = {}
            original_manifest_bytes = manifest_path.read_bytes()
            manifest_committed = False
            manifest_raw_bytes = 0
            manifest_compressed_bytes = 0
            try:
                for sensor_key, spec, source in raw_streams:
                    if cancel_requested is not None and cancel_requested():
                        raise TactileRecompressionCancelled(
                            "Tactile recompression cancelled before the next stream"
                        )
                    initial_stat = source.stat()
                    if initial_stat.st_nlink != 1:
                        raise RuntimeError(
                            f"Refusing tactile source with link_count={initial_stat.st_nlink}: "
                            f"{source}"
                        )
                    source_stats[source] = initial_stat
                    target = _compressed_path(source, chunk_frames=int(chunk_frames))
                    (
                        chunks,
                        raw_sha256,
                        stored_sha256,
                        compressed_bytes,
                    ) = _compress_stream(
                        source,
                        target,
                        spec,
                        codec=codec,
                        chunk_frames=int(chunk_frames),
                        cancel_requested=cancel_requested,
                        enforce_global_collection_exclusion=(
                            enforce_global_collection_exclusion
                        ),
                    )
                    created.append(target)
                    updated = dict(spec)
                    updated.update(
                        {
                            "path": _relative_dataset_path(root, target),
                            "encoding": "chunked_delta_zstd",
                            "compression": "zstd",
                            "storage_dtype": "uint8",
                            "predictor": LOSSLESS_PREDICTOR_V1,
                            "chunk_frame_count": int(chunk_frames),
                            "chunks": chunks,
                            "stored_size_bytes": int(compressed_bytes),
                            "stored_sha256": stored_sha256,
                            "stored_hash_profile": "stored_bytes_sha256_v1",
                            "recompressed_from": {
                                "path": _relative_dataset_path(root, source),
                                "encoding": str(spec.get("encoding") or "raw_c_order"),
                                "compression": "none",
                                "raw_sha256": raw_sha256,
                                "device": int(initial_stat.st_dev),
                                "inode": int(initial_stat.st_ino),
                                "link_count": int(initial_stat.st_nlink),
                            },
                        }
                    )
                    replacements[sensor_key] = updated
                    manifest_raw_bytes += int(initial_stat.st_size)
                    manifest_compressed_bytes += compressed_bytes
                updated_manifest = dict(manifest)
                updated_streams = dict(manifest["streams"])
                updated_streams.update(replacements)
                updated_manifest["streams"] = updated_streams
                updated_manifest["tactile_recompression"] = {
                    "profile": LOSSLESS_PROFILE_V1,
                    "compression": "zstd",
                    "level": int(level),
                    "chunk_frame_count": int(chunk_frames),
                    "predictor": LOSSLESS_PREDICTOR_V1,
                    "backend": codec.backend,
                    "transaction_id": transaction_id,
                }
                for source, initial_stat in source_stats.items():
                    current_stat = source.stat()
                    stable_fields = (
                        current_stat.st_dev,
                        current_stat.st_ino,
                        current_stat.st_nlink,
                        current_stat.st_size,
                        current_stat.st_mtime_ns,
                    )
                    initial_fields = (
                        initial_stat.st_dev,
                        initial_stat.st_ino,
                        initial_stat.st_nlink,
                        initial_stat.st_size,
                        initial_stat.st_mtime_ns,
                    )
                    if stable_fields != initial_fields:
                        raise RuntimeError(
                            f"Tactile source changed during recompression: {source}"
                        )
                if cancel_requested is not None and cancel_requested():
                    raise TactileRecompressionCancelled(
                        "Tactile recompression cancelled before manifest commit"
                    )
                active_collection = (
                    _active_collection_processes()
                    if enforce_global_collection_exclusion
                    else ()
                )
                if active_collection:
                    raise RuntimeError(
                        "Formal collection started before tactile manifest commit: "
                        + "; ".join(active_collection[:3])
                    )
                _atomic_write_json(manifest_path, updated_manifest)
                manifest_committed = True
                committed_manifest = _read_json(manifest_path)
                if committed_manifest.get(
                    "tactile_recompression"
                ) != updated_manifest.get("tactile_recompression"):
                    raise ValueError(
                        f"Committed tactile manifest verification failed: {manifest_path}"
                    )
            except BaseException:
                if manifest_committed:
                    _atomic_write_bytes(manifest_path, original_manifest_bytes)
                for target in created:
                    target.unlink(missing_ok=True)
                raise
            processed_manifests += 1
            processed_streams += len(raw_streams)
            raw_total += manifest_raw_bytes
            compressed_total += manifest_compressed_bytes
            for _, _, source in raw_streams:
                size = source.stat().st_size
                source.unlink()
                removed_total += size
    except BaseException as exc:
        # Earlier manifests may already be durably committed before a
        # cooperative stop is observed at the next manifest boundary.  Keep
        # an opted-in artifact-integrity chain current for that valid partial
        # prefix before publishing the cancellation/failure receipt.  The
        # original exception remains authoritative if inventory publication
        # itself fails; the receipt then cannot claim an old revision is
        # current, and the next capture admission/seal must validate and
        # republish from the actually referenced graph.
        if processed_manifests > 0:
            try:
                published_inventory_hash = _publish_updated_modality_inventory(root)
            except BaseException:
                published_inventory_hash = None
        try:
            write_receipt(
                status=(
                    "cancelled_partial"
                    if isinstance(exc, TactileRecompressionCancelled)
                    else "failed_partial"
                ),
                error=exc,
            )
        except BaseException:
            pass
        raise
    try:
        published_inventory_hash = _publish_updated_modality_inventory(root)
    except BaseException as exc:
        try:
            write_receipt(status="failed_partial", error=exc)
        except BaseException:
            pass
        raise
    write_receipt(status="completed")
    return current_result()


def run_idle_tactile_recompression_worker(
    dataset_root: str,
    stop_event: Any,
    tactile_bundle_root: str | Path | None = None,
    native_min_free_bytes: int | None = None,
    native_require_distinct_device: bool = False,
) -> Mapping[str, Any]:
    """Run verified recompression under external idle-lifecycle admission.

    The worker owns no admission decision. Its caller must establish that the
    dataset is idle and revoke that admission through ``stop_event``.
    """

    if tactile_bundle_root is not None and str(tactile_bundle_root).strip():
        # Keep the historical worker entry point usable by process controllers
        # while routing native two-root maintenance to its dedicated owner.
        from .tactile_bundle_recompression import (
            run_idle_native_tactile_recompression_worker,
        )

        return run_idle_native_tactile_recompression_worker(
            dataset_root,
            stop_event,
            str(tactile_bundle_root),
            native_min_free_bytes,
            native_require_distinct_device,
        )
    root = Path(dataset_root).expanduser().resolve()
    if stop_event.is_set():
        return {"outcome": "cancelled", "dataset_root": str(root)}
    orphan_count = clean_uncommitted_tactile_recompression_artifacts(
        root,
        chunk_frames=DEFAULT_CHUNK_FRAMES,
    )
    if stop_event.is_set():
        return {
            "outcome": "cancelled",
            "dataset_root": str(root),
            "uncommitted_artifacts_removed": int(orphan_count),
        }
    try:
        plan = plan_tactile_zstd_recompression(
            root,
            min_free_bytes=DEFAULT_MIN_FREE_BYTES,
        )
    except ValueError as exc:
        if "No finalized episodes found" in str(exc):
            return {"outcome": "no_work", "dataset_root": str(root)}
        raise
    if int(plan.stream_count) <= 0:
        published_inventory_hash = _publish_updated_modality_inventory(root)
        return {
            "outcome": "no_work",
            "dataset_root": str(root),
            "manifest_count": int(plan.manifest_count),
            "uncommitted_artifacts_removed": int(orphan_count),
            "published_modality_inventory_hash": published_inventory_hash,
        }
    try:
        result = recompress_tactile_zstd_in_place(
            root,
            level=1,
            chunk_frames=DEFAULT_CHUNK_FRAMES,
            min_free_bytes=DEFAULT_MIN_FREE_BYTES,
            delete_originals=True,
            backend="auto",
            cancel_requested=stop_event.is_set,
            # RealTeleop owns stricter state admission. The legacy /proc check
            # would reject merely because its idle parent process exists.
            enforce_global_collection_exclusion=False,
        )
    except TactileRecompressionCancelled:
        return {"outcome": "cancelled", "dataset_root": str(root)}
    return {
        "outcome": "completed",
        "uncommitted_artifacts_removed": int(orphan_count),
        **result.to_dict(),
    }


__all__ = [
    "DEFAULT_CHUNK_FRAMES",
    "DEFAULT_MIN_FREE_BYTES",
    "COLLECTION_RECHECK_CHUNKS",
    "LOSSLESS_PROFILE_V1",
    "LOSSLESS_PREDICTOR_V1",
    "TactileRecompressionCancelled",
    "TactileRecompressionPlan",
    "TactileRecompressionResult",
    "clean_uncommitted_tactile_recompression_artifacts",
    "plan_tactile_zstd_recompression",
    "recompress_tactile_zstd_in_place",
    "run_idle_tactile_recompression_worker",
]
