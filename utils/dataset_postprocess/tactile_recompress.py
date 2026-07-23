from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


DEFAULT_CHUNK_FRAMES = 32
DEFAULT_MIN_FREE_BYTES = 20 * 1024**3
COLLECTION_PROCESS_MARKERS = (
    "teleop_runtime.collection",
    "teleop_runtime.debug.run_vive_rm65_real_teleop",
)


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


def _active_collection_processes() -> tuple[str, ...]:
    """Return active formal-collection command lines on Linux.

    Offline compression competes for CPU, memory bandwidth, and disk even when
    it targets another dataset, so apply mode refuses globally while formal
    collection is alive.
    """

    proc = Path("/proc")
    if not proc.is_dir():
        return ()
    current_pid = os.getpid()
    matches: list[str] = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == current_pid:
            continue
        try:
            payload = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = payload.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if not command or "utils.dataset_postprocess" in command:
            continue
        if any(marker in command for marker in COLLECTION_PROCESS_MARKERS):
            matches.append(f"pid={entry.name} {command}")
    return tuple(sorted(matches))


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


def _compressed_path(source: Path) -> Path:
    return source.with_name(f"{source.name}.zst")


def _compress_stream(
    source: Path,
    target: Path,
    spec: Mapping[str, Any],
    *,
    codec: _ZstdCodec,
    chunk_frames: int,
) -> tuple[list[dict[str, int]], str, int]:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing compressed stream: {target}")
    bytes_per_frame = _raw_stream_size({**dict(spec), "frame_count": 1})
    frame_count = int(spec["frame_count"])
    source_hash = hashlib.sha256()
    chunks: list[dict[str, int]] = []
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    compressed_total = 0
    try:
        with source.open("rb") as input_handle, os.fdopen(fd, "wb") as output_handle:
            frame_start = 0
            while frame_start < frame_count:
                count = min(int(chunk_frames), frame_count - frame_start)
                raw_bytes = count * bytes_per_frame
                raw = input_handle.read(raw_bytes)
                if len(raw) != raw_bytes:
                    raise ValueError(f"Short raw tactile chunk read from {source}")
                source_hash.update(raw)
                compressed = codec.compress(raw)
                restored = codec.decompress(compressed, raw_bytes=raw_bytes)
                if restored != raw:
                    raise ValueError(f"Zstd round-trip mismatch for {source} at frame {frame_start}")
                offset = output_handle.tell()
                output_handle.write(compressed)
                chunks.append(
                    {
                        "frame_start": frame_start,
                        "frame_count": count,
                        "offset": int(offset),
                        "compressed_bytes": len(compressed),
                        "raw_bytes": raw_bytes,
                    }
                )
                compressed_total += len(compressed)
                frame_start += count
            if input_handle.read(1):
                raise ValueError(f"Raw tactile stream has trailing bytes: {source}")
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, target)
        return chunks, source_hash.hexdigest(), compressed_total
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
) -> TactileRecompressionResult:
    if not delete_originals:
        raise ValueError(
            "In-place recompression requires delete_originals=True after verified manifest commit."
        )
    active_collection = _active_collection_processes()
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
    processed_manifests = 0
    processed_streams = 0
    raw_total = 0
    compressed_total = 0
    removed_total = 0
    for manifest_path in manifests:
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
        try:
            for sensor_key, spec, source in raw_streams:
                target = _compressed_path(source)
                chunks, raw_sha256, compressed_bytes = _compress_stream(
                    source,
                    target,
                    spec,
                    codec=codec,
                    chunk_frames=int(chunk_frames),
                )
                created.append(target)
                updated = dict(spec)
                updated.update(
                    {
                        "path": _relative_dataset_path(root, target),
                        "encoding": "chunked_zstd",
                        "compression": "zstd",
                        "chunk_frame_count": int(chunk_frames),
                        "chunks": chunks,
                        "recompressed_from": {
                            "path": _relative_dataset_path(root, source),
                            "encoding": str(spec.get("encoding") or "raw_c_order"),
                            "compression": "none",
                            "raw_sha256": raw_sha256,
                        },
                    }
                )
                replacements[sensor_key] = updated
                raw_total += source.stat().st_size
                compressed_total += compressed_bytes
            updated_manifest = dict(manifest)
            updated_streams = dict(manifest["streams"])
            updated_streams.update(replacements)
            updated_manifest["streams"] = updated_streams
            updated_manifest["tactile_recompression"] = {
                "compression": "zstd",
                "level": int(level),
                "chunk_frame_count": int(chunk_frames),
                "backend": codec.backend,
            }
            _atomic_write_json(manifest_path, updated_manifest)
        except BaseException:
            for target in created:
                target.unlink(missing_ok=True)
            raise
        for _, _, source in raw_streams:
            size = source.stat().st_size
            source.unlink()
            removed_total += size
        processed_manifests += 1
        processed_streams += len(raw_streams)
    return TactileRecompressionResult(
        dataset_root=str(root),
        manifest_count=processed_manifests,
        stream_count=processed_streams,
        raw_bytes=raw_total,
        compressed_bytes=compressed_total,
        removed_original_bytes=removed_total,
    )


__all__ = [
    "DEFAULT_CHUNK_FRAMES",
    "DEFAULT_MIN_FREE_BYTES",
    "TactileRecompressionPlan",
    "TactileRecompressionResult",
    "plan_tactile_zstd_recompression",
    "recompress_tactile_zstd_in_place",
]
