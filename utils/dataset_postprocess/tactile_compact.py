"""Versioned, in-place compaction for finalized DM-Tac stream manifests.

The migration preserves tactile frame count, ordering, timestamps, and frame
indices.  Raw camera tactile is spatially compacted; derived distributed force
keeps its logical shape and dtype but uses a per-chunk int16 representation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dataset_contracts.tactile import ChunkedTactileStream, tactile_stream_compression

from .tactile_recompress import _ZstdCodec, _active_collection_processes, _atomic_write_json


TACTILE_COMPACTION_SCHEMA_V1 = "dexteleop_tactile_compaction_v1"
RAW_PROFILE_V1 = "dmtac_raw_120x160_q6_delta_zstd_v1"
FORCE_PROFILE_V1 = "dmtac_force_qint16_zstd_v1"
RAW_SOURCE_SHAPE = (480, 640)
RAW_TARGET_SHAPE = (120, 160)
FORCE_SHAPE = (288, 384, 3)
DEFAULT_CHUNK_FRAMES = 32
DEFAULT_MIN_FREE_BYTES = 20 * 1024**3


@dataclass(frozen=True)
class TactileCompactionPlan:
    dataset_root: str
    manifest_count: int
    stream_count: int
    raw_stream_count: int
    force_stream_count: int
    already_compact_stream_count: int
    unsupported_streams: tuple[str, ...]
    source_bytes: int
    maximum_manifest_staging_bytes: int
    free_bytes: int
    min_free_bytes: int

    @property
    def ready(self) -> bool:
        return not self.unsupported_streams

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ready"] = self.ready
        return payload


@dataclass(frozen=True)
class TactileCompactionResult:
    dataset_root: str
    manifest_count: int
    stream_count: int
    raw_stream_count: int
    force_stream_count: int
    skipped_stream_count: int
    source_bytes: int
    compacted_bytes: int
    removed_source_bytes: int
    complete: bool

    @property
    def stored_ratio(self) -> float | None:
        return None if self.source_bytes <= 0 else self.compacted_bytes / self.source_bytes

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stored_ratio"] = self.stored_ratio
        return payload


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _resolve(root: Path, value: Any) -> Path:
    path = Path(str(value or "")).expanduser()
    if not str(path):
        raise ValueError("Tactile stream path is empty.")
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Tactile stream escapes dataset root: {resolved}") from exc
    return resolved


def tactile_manifest_paths(dataset_root: str | Path) -> tuple[Path, ...]:
    root = Path(dataset_root).expanduser().resolve()
    paths: set[Path] = set()
    for prefix in ("tactile", "tactile_force"):
        candidate = root / prefix
        if candidate.is_dir():
            paths.update(path.resolve() for path in candidate.rglob("manifest.json"))
    return tuple(sorted(paths))


def _profile(spec: Mapping[str, Any]) -> str:
    transform = spec.get("tactile_compaction")
    return str(transform.get("profile") or "") if isinstance(transform, Mapping) else ""


def _stream_kind(spec: Mapping[str, Any]) -> str:
    if _profile(spec) in {RAW_PROFILE_V1, FORCE_PROFILE_V1}:
        return "compact"
    dtype = str(spec.get("dtype") or "")
    shape = tuple(int(value) for value in spec.get("shape", ()) or ())
    if dtype == "uint8" and shape == RAW_SOURCE_SHAPE:
        return "raw"
    if dtype == "float32" and shape == FORCE_SHAPE:
        return "force"
    return "unsupported"


def _source_size(root: Path, spec: Mapping[str, Any]) -> int:
    path = _resolve(root, spec.get("path"))
    if not path.is_file():
        raise FileNotFoundError(path)
    return int(path.stat().st_size)


def _staging_bytes(spec: Mapping[str, Any], kind: str) -> int:
    frames = int(spec.get("frame_count") or 0)
    if kind == "raw":
        return frames * math.prod(RAW_TARGET_SHAPE)
    if kind == "force":
        return frames * math.prod(FORCE_SHAPE) * np.dtype(np.int16).itemsize
    return 0


def plan_tactile_compaction(
    dataset_root: str | Path,
    *,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
) -> TactileCompactionPlan:
    root = Path(dataset_root).expanduser().resolve()
    manifests = tactile_manifest_paths(root)
    stream_count = raw_count = force_count = compact_count = source_bytes = 0
    unsupported: list[str] = []
    maximum_staging = 0
    for manifest_path in manifests:
        manifest = _read_json(manifest_path)
        if manifest.get("format") != "binary_stream_v1":
            unsupported.append(f"{_relative(root, manifest_path)}: format={manifest.get('format')!r}")
            continue
        streams = manifest.get("streams")
        if not isinstance(streams, Mapping):
            unsupported.append(f"{_relative(root, manifest_path)}: streams is not an object")
            continue
        manifest_staging = 0
        for sensor_key, value in streams.items():
            stream_count += 1
            if not isinstance(value, Mapping):
                unsupported.append(f"{_relative(root, manifest_path)}:{sensor_key}: invalid spec")
                continue
            kind = _stream_kind(value)
            if kind == "compact":
                compact_count += 1
                continue
            if kind == "unsupported":
                unsupported.append(
                    f"{_relative(root, manifest_path)}:{sensor_key}: "
                    f"dtype={value.get('dtype')!r}, shape={value.get('shape')!r}, "
                    f"encoding={value.get('encoding')!r}"
                )
                continue
            size = _source_size(root, value)
            source_bytes += size
            manifest_staging += _staging_bytes(value, kind)
            if kind == "raw":
                raw_count += 1
            else:
                force_count += 1
        maximum_staging = max(maximum_staging, manifest_staging)
    return TactileCompactionPlan(
        dataset_root=str(root),
        manifest_count=len(manifests),
        stream_count=stream_count,
        raw_stream_count=raw_count,
        force_stream_count=force_count,
        already_compact_stream_count=compact_count,
        unsupported_streams=tuple(unsupported),
        source_bytes=source_bytes,
        maximum_manifest_staging_bytes=maximum_staging,
        free_bytes=shutil.disk_usage(root).free,
        min_free_bytes=int(min_free_bytes),
    )


class _SourceStream:
    def __init__(self, root: Path, spec: Mapping[str, Any]) -> None:
        self.root = root
        self.spec = dict(spec)
        self.shape = tuple(int(value) for value in self.spec.get("shape", ()) or ())
        self.dtype = np.dtype(str(self.spec.get("dtype") or ""))
        self.frame_count = int(self.spec.get("frame_count") or 0)
        if self.frame_count <= 0 or not self.shape:
            raise ValueError(f"Invalid tactile frame count/shape: {self.spec}")
        self.path = _resolve(root, self.spec.get("path"))
        compression = tactile_stream_compression(self.spec)
        if compression == "none":
            expected = self.frame_count * math.prod(self.shape) * self.dtype.itemsize
            if self.path.stat().st_size != expected:
                raise ValueError(
                    f"Raw tactile stream size mismatch for {self.path}: "
                    f"{self.path.stat().st_size} != {expected}"
                )
            self._array: np.ndarray | None = np.memmap(
                self.path,
                mode="r",
                dtype=self.dtype,
                shape=(self.frame_count, *self.shape),
            )
            self._chunked: ChunkedTactileStream | None = None
        else:
            self._array = None
            self._chunked = ChunkedTactileStream(dataset_root=root, spec=self.spec)

    def frames(self, start: int, count: int) -> np.ndarray:
        if self._array is not None:
            return np.asarray(self._array[start : start + count]).copy()
        assert self._chunked is not None
        values = [self._chunked.frame(index) for index in range(start, start + count)]
        if any(value is None for value in values):
            raise ValueError(f"Missing tactile frame in {self.path} at {start}:{start + count}")
        return np.stack(values)


def _compact_raw(frames: np.ndarray) -> np.ndarray:
    source = np.asarray(frames, dtype=np.uint8)
    if source.ndim != 3 or tuple(source.shape[1:]) != RAW_SOURCE_SHAPE:
        raise ValueError(f"Expected raw tactile chunk (*,{RAW_SOURCE_SHAPE}), got {source.shape}")
    count = source.shape[0]
    sums = source.astype(np.uint32).reshape(count, 120, 4, 160, 4).sum(axis=(2, 4))
    area_mean = ((sums + 8) // 16).astype(np.uint8)
    codes = area_mean >> np.uint8(2)
    return ((codes.astype(np.uint16) << 2) + 2).clip(0, 255).astype(np.uint8)


def _delta_encode(frames: np.ndarray) -> np.ndarray:
    values = np.asarray(frames, dtype=np.uint8)
    encoded = np.empty_like(values)
    encoded[0] = values[0]
    if len(values) > 1:
        encoded[1:] = (
            values[1:].astype(np.int16) - values[:-1].astype(np.int16) + 128
        ).astype(np.uint8)
    return encoded


def _delta_decode(encoded: np.ndarray) -> np.ndarray:
    values = np.asarray(encoded, dtype=np.uint8)
    restored = np.empty_like(values)
    restored[0] = values[0]
    for index in range(1, len(values)):
        restored[index] = (
            restored[index - 1].astype(np.int16)
            + values[index].astype(np.int16)
            - 128
        ).astype(np.uint8)
    return restored


def _target_path(source: Path, profile: str) -> Path:
    token = "raw120x160-q6-delta" if profile == RAW_PROFILE_V1 else "force-qint16"
    return source.with_name(f"{source.name}.{token}.zst")


def _write_compact_stream(
    root: Path,
    spec: Mapping[str, Any],
    *,
    kind: str,
    codec: _ZstdCodec,
    chunk_frames: int,
) -> tuple[Path, dict[str, Any], int, int]:
    source = _SourceStream(root, spec)
    profile = RAW_PROFILE_V1 if kind == "raw" else FORCE_PROFILE_V1
    target = _target_path(source.path, profile)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite tactile staging target: {target}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    chunks: list[dict[str, Any]] = []
    source_hash = hashlib.sha256()
    logical_hash = hashlib.sha256()
    written = 0
    try:
        with os.fdopen(fd, "wb") as output:
            for start in range(0, source.frame_count, int(chunk_frames)):
                count = min(int(chunk_frames), source.frame_count - start)
                original = np.ascontiguousarray(source.frames(start, count))
                source_hash.update(original.tobytes(order="C"))
                chunk_meta: dict[str, Any]
                if kind == "raw":
                    logical = np.ascontiguousarray(_compact_raw(original))
                    storage = _delta_encode(logical)
                    chunk_meta = {"predictor": "delta_mod_uint8_bias_128"}
                else:
                    if not np.all(np.isfinite(original)):
                        raise ValueError(f"Non-finite distributed_force values in {source.path}")
                    maxima = np.max(np.abs(original), axis=(0, 1, 2)).astype(np.float64)
                    scales = np.where(maxima > 0.0, maxima / 32767.0, 1.0).astype(np.float32)
                    storage = np.rint(original / scales.reshape(1, 1, 1, -1)).clip(
                        -32767, 32767
                    ).astype(np.int16)
                    logical = storage.astype(np.float32) * scales.reshape(1, 1, 1, -1)
                    chunk_meta = {
                        "quantization": {
                            "scheme": "symmetric_per_last_axis_int16",
                            "scales": scales.tolist(),
                        }
                    }
                logical_hash.update(np.ascontiguousarray(logical).tobytes(order="C"))
                raw_storage = np.ascontiguousarray(storage).tobytes(order="C")
                compressed = codec.compress(raw_storage)
                restored_bytes = codec.decompress(compressed, raw_bytes=len(raw_storage))
                if restored_bytes != raw_storage:
                    raise ValueError(f"Zstd round-trip mismatch in {source.path} at frame {start}")
                if kind == "raw":
                    decoded = _delta_decode(
                        np.frombuffer(restored_bytes, dtype=np.uint8).reshape(storage.shape)
                    )
                    if not np.array_equal(decoded, logical):
                        raise ValueError(f"Delta round-trip mismatch in {source.path} at {start}")
                offset = output.tell()
                output.write(compressed)
                chunks.append(
                    {
                        "frame_start": start,
                        "frame_count": count,
                        "offset": int(offset),
                        "compressed_bytes": len(compressed),
                        "raw_bytes": len(raw_storage),
                        **chunk_meta,
                    }
                )
                written += len(compressed)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    updated = dict(spec)
    if kind == "raw":
        updated.update(
            {
                "dtype": "uint8",
                "storage_dtype": "uint8",
                "shape": list(RAW_TARGET_SHAPE),
                "encoding": "chunked_delta_zstd",
                "predictor": "delta_mod_uint8_bias_128",
            }
        )
        detail = {
            "source_shape": list(RAW_SOURCE_SHAPE),
            "target_shape": list(RAW_TARGET_SHAPE),
            "spatial_downsample": "4x4_area_mean_integer_half_up",
            "quantization_bits": 6,
            "quantization_reconstruction": "uint8_bin_midpoint_step_4",
            "frequency_preserved": True,
        }
    else:
        updated.update(
            {
                "dtype": "float32",
                "storage_dtype": "int16",
                "shape": list(FORCE_SHAPE),
                "encoding": "chunked_zstd",
            }
        )
        detail = {
            "source_shape": list(FORCE_SHAPE),
            "target_shape": list(FORCE_SHAPE),
            "quantization": "symmetric_per_chunk_per_channel_int16",
            "frequency_preserved": True,
        }
    updated.update(
        {
            "path": _relative(root, target),
            "compression": "zstd",
            "chunk_frame_count": int(chunk_frames),
            "chunks": chunks,
            "tactile_compaction": {
                "schema": TACTILE_COMPACTION_SCHEMA_V1,
                "profile": profile,
                **detail,
                "source": {
                    "path": _relative(root, source.path),
                    "dtype": str(spec.get("dtype") or ""),
                    "shape": list(spec.get("shape", ()) or ()),
                    "encoding": str(spec.get("encoding") or "raw_c_order"),
                    "compression": tactile_stream_compression(spec),
                    "logical_sha256": source_hash.hexdigest(),
                },
                "logical_sha256": logical_hash.hexdigest(),
            },
        }
    )
    return target, updated, source.path.stat().st_size, written


def _updated_manifest_frames(
    manifest: Mapping[str, Any], replacements: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for value in manifest.get("frames", ()) or ():
        if not isinstance(value, Mapping):
            continue
        record = dict(value)
        shapes = dict(record.get("shapes") or {})
        dtypes = dict(record.get("dtypes") or {})
        source_shapes = dict(record.get("source_shapes") or shapes)
        source_dtypes = dict(record.get("source_dtypes") or dtypes)
        for sensor_key, spec in replacements.items():
            shapes[sensor_key] = list(spec.get("shape", ()) or ())
            dtypes[sensor_key] = str(spec.get("dtype") or "")
        record["source_shapes"] = source_shapes
        record["source_dtypes"] = source_dtypes
        record["shapes"] = shapes
        record["dtypes"] = dtypes
        output.append(record)
    return output


def _rollback_path(source: Path) -> Path:
    return source.with_name(f".{source.name}.tactile-compact-rollback")


def _validate_committed_manifest(root: Path, manifest_path: Path) -> None:
    manifest = _read_json(manifest_path)
    streams = manifest.get("streams")
    if not isinstance(streams, Mapping):
        raise ValueError(f"Committed tactile manifest has no streams: {manifest_path}")
    for sensor_key, value in streams.items():
        if not isinstance(value, Mapping) or _profile(value) not in {
            RAW_PROFILE_V1,
            FORCE_PROFILE_V1,
        }:
            continue
        reader = ChunkedTactileStream(dataset_root=root, spec=value)
        expected_shape = tuple(int(item) for item in value.get("shape", ()) or ())
        expected_dtype = np.dtype(str(value.get("dtype") or ""))
        indices = sorted({0, reader.frame_count // 2, reader.frame_count - 1})
        for index in indices:
            frame = reader.frame(index)
            if frame is None:
                raise ValueError(
                    f"Committed tactile reader returned no frame for {sensor_key}[{index}]"
                )
            if tuple(frame.shape) != expected_shape or frame.dtype != expected_dtype:
                raise ValueError(
                    f"Committed tactile reader contract mismatch for {sensor_key}[{index}]: "
                    f"shape/dtype={frame.shape}/{frame.dtype} != "
                    f"{expected_shape}/{expected_dtype}"
                )


def _write_receipt(root: Path, result: TactileCompactionResult) -> None:
    payload = {
        "schema": TACTILE_COMPACTION_SCHEMA_V1,
        "raw_profile": RAW_PROFILE_V1,
        "force_profile": FORCE_PROFILE_V1,
        "frequency_preserved": True,
        "result": result.to_dict(),
    }
    _atomic_write_json(root / "meta" / "tactile_compaction.json", payload)
    info_path = root / "meta" / "info.json"
    if info_path.is_file():
        info = _read_json(info_path)
        info["tactile_storage_profile"] = {
            "schema": TACTILE_COMPACTION_SCHEMA_V1,
            "raw_profile": RAW_PROFILE_V1,
            "force_profile": FORCE_PROFILE_V1,
            "frequency_preserved": True,
        }
        _atomic_write_json(info_path, info)


def compact_tactile_dataset_in_place(
    dataset_root: str | Path,
    *,
    level: int = 1,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    max_manifests: int | None = None,
    delete_originals: bool = False,
    backend: str = "auto",
) -> TactileCompactionResult:
    if not delete_originals:
        raise ValueError("In-place tactile compaction requires delete_originals=True.")
    active = _active_collection_processes()
    if active:
        raise RuntimeError(
            "Refusing tactile compaction while formal collection is active: "
            + "; ".join(active[:3])
        )
    if int(chunk_frames) <= 0:
        raise ValueError("chunk_frames must be positive.")
    if max_manifests is not None and int(max_manifests) <= 0:
        raise ValueError("max_manifests must be positive when provided.")
    root = Path(dataset_root).expanduser().resolve()
    plan = plan_tactile_compaction(root, min_free_bytes=min_free_bytes)
    if plan.unsupported_streams:
        raise ValueError(
            "Unsupported tactile streams prevent a uniform migration: "
            + "; ".join(plan.unsupported_streams[:10])
        )
    codec = _ZstdCodec(level=int(level), backend=backend)
    processed_manifests = processed_streams = raw_count = force_count = skipped = 0
    source_total = compact_total = removed_total = 0
    for manifest_path in tactile_manifest_paths(root):
        manifest = _read_json(manifest_path)
        streams = manifest.get("streams") or {}
        pending = [
            (str(key), dict(value), _stream_kind(value))
            for key, value in streams.items()
            if isinstance(value, Mapping) and _stream_kind(value) != "compact"
        ]
        skipped += sum(
            1
            for value in streams.values()
            if isinstance(value, Mapping) and _stream_kind(value) == "compact"
        )
        if not pending:
            continue
        if max_manifests is not None and processed_manifests >= int(max_manifests):
            break
        staging = sum(_staging_bytes(spec, kind) for _, spec, kind in pending)
        free = shutil.disk_usage(root).free
        if free - staging < int(min_free_bytes):
            raise OSError(
                f"Insufficient free space for manifest staging: free={free}, "
                f"required<={staging}, reserve={int(min_free_bytes)}"
            )
        created: list[Path] = []
        replacements: dict[str, dict[str, Any]] = {}
        sources: list[Path] = []
        source_sizes: list[int] = []
        backups: list[tuple[Path, Path]] = []
        committed = False
        try:
            for sensor_key, spec, kind in pending:
                target, updated, source_bytes, compacted_bytes = _write_compact_stream(
                    root,
                    spec,
                    kind=kind,
                    codec=codec,
                    chunk_frames=int(chunk_frames),
                )
                created.append(target)
                source = _resolve(root, spec.get("path"))
                sources.append(source)
                source_sizes.append(source_bytes)
                replacements[sensor_key] = updated
                source_total += source_bytes
                compact_total += compacted_bytes
                processed_streams += 1
                if kind == "raw":
                    raw_count += 1
                else:
                    force_count += 1
            updated_manifest = dict(manifest)
            updated_streams = dict(streams)
            updated_streams.update(replacements)
            updated_manifest["streams"] = updated_streams
            updated_manifest["frames"] = _updated_manifest_frames(manifest, replacements)
            updated_manifest["tactile_compaction"] = {
                "schema": TACTILE_COMPACTION_SCHEMA_V1,
                "raw_profile": RAW_PROFILE_V1,
                "force_profile": FORCE_PROFILE_V1,
                "frequency_preserved": True,
                "chunk_frame_count": int(chunk_frames),
                "zstd_level": int(level),
                "zstd_backend": codec.backend,
            }
            for source in sources:
                backup = _rollback_path(source)
                if backup.exists():
                    raise FileExistsError(f"Stale tactile rollback link exists: {backup}")
                os.link(source, backup)
                backups.append((source, backup))
            _atomic_write_json(manifest_path, updated_manifest)
            committed = True
            for source in sources:
                source.unlink()
            _validate_committed_manifest(root, manifest_path)
        except BaseException:
            if committed:
                _atomic_write_json(manifest_path, manifest)
                for source, backup in backups:
                    if not source.exists() and backup.exists():
                        os.replace(backup, source)
                    else:
                        backup.unlink(missing_ok=True)
            else:
                for _, backup in backups:
                    backup.unlink(missing_ok=True)
            for path in created:
                path.unlink(missing_ok=True)
            raise
        for _, backup in backups:
            backup.unlink(missing_ok=True)
        removed_total += sum(source_sizes)
        processed_manifests += 1
    remaining = plan_tactile_compaction(root, min_free_bytes=min_free_bytes)
    complete = remaining.raw_stream_count == 0 and remaining.force_stream_count == 0
    result = TactileCompactionResult(
        dataset_root=str(root),
        manifest_count=processed_manifests,
        stream_count=processed_streams,
        raw_stream_count=raw_count,
        force_stream_count=force_count,
        skipped_stream_count=skipped,
        source_bytes=source_total,
        compacted_bytes=compact_total,
        removed_source_bytes=removed_total,
        complete=complete,
    )
    if complete:
        _write_receipt(root, result)
    return result


__all__ = [
    "DEFAULT_CHUNK_FRAMES",
    "DEFAULT_MIN_FREE_BYTES",
    "FORCE_PROFILE_V1",
    "RAW_PROFILE_V1",
    "TACTILE_COMPACTION_SCHEMA_V1",
    "TactileCompactionPlan",
    "TactileCompactionResult",
    "compact_tactile_dataset_in_place",
    "plan_tactile_compaction",
    "tactile_manifest_paths",
]
