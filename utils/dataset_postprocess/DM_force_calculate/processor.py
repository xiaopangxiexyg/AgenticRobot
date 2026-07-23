from __future__ import annotations

import json
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from dataset_contracts.tactile_calibration import (
    load_tactile_calibration,
    normalize_calibration_policy,
    save_tactile_calibration,
)
from utils.dataset_postprocess.io import read_json, read_jsonl, write_json


DM_FORCE_FORMAT = "dexteleop_dmtac_distributed_force_v1"
DM_FORCE_PROCESSOR_REVISION = "sdk_offline_force_fixed_calibration_zstd_v4"
TACTILE_STREAM_FORMAT = "binary_stream_v1"
RAW_DTYPE = "uint8"
RAW_SHAPE = (480, 640)
FORCE_DTYPE = "float32"
FORCE_SHAPE = (288, 384, 3)
DEFAULT_FORCE_COMPRESSION = "zstd"
DEFAULT_FORCE_COMPRESSION_LEVEL = 3
DEFAULT_FORCE_CHUNK_FRAMES = 16


def default_calibration_dir() -> Path:
    """Return the legacy DM-Tac calibration location without importing hardware."""

    return (
        Path(__file__).resolve().parents[3]
        / "hardware/DMTAC_gripper/configs/tactile_calibration"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _episode_id(index: int) -> str:
    return f"episode_{int(index):06d}"


def _episode_index_from_path(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("episode_"):
            try:
                return int(part.split("_", 1)[1].split(".", 1)[0])
            except (IndexError, ValueError):
                return None
    return None


def _resolve_dataset_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


def _safe_path_token(value: Any) -> str:
    import re

    text = str(value).strip() or "unnamed"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "unnamed"


def _timestamp_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, ensure_ascii=True, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _manifest_stream_paths(root: Path, manifest: Mapping[str, Any]) -> set[Path]:
    streams = manifest.get("streams")
    if not isinstance(streams, Mapping):
        return set()
    paths: set[Path] = set()
    for spec in streams.values():
        if not isinstance(spec, Mapping) or not spec.get("path"):
            continue
        candidate = _resolve_dataset_path(root, spec.get("path")).resolve()
        try:
            candidate.relative_to((root / "tactile_force").resolve())
        except ValueError:
            continue
        paths.add(candidate)
    return paths


@dataclass(frozen=True)
class RawTactileStreamTask:
    dataset_root: Path
    output_root: Path
    source_manifest_path: Path
    output_manifest_path: Path
    side: str
    episode_index: int
    sensor_key: str
    source_stream_path: Path
    output_stream_path: Path
    frame_count: int
    source_shape: tuple[int, ...]
    source_dtype: str
    source_stream_spec: dict[str, Any] = field(default_factory=dict)
    source_frames: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        source_spec = dict(self.source_stream_spec)
        source_encoding = source_spec.get("encoding")
        source_compression = str(source_spec.get("compression") or "").strip().lower().replace("-", "_")
        if not source_compression:
            text = str(source_encoding or "").strip().lower().replace("-", "_")
            source_compression = text.removeprefix("chunked_") if text.startswith("chunked_") else "none"
        return {
            "dataset_root": str(self.dataset_root),
            "output_root": str(self.output_root),
            "source_manifest_path": str(self.source_manifest_path),
            "output_manifest_path": str(self.output_manifest_path),
            "side": self.side,
            "episode_index": int(self.episode_index),
            "sensor_key": self.sensor_key,
            "source_stream_path": str(self.source_stream_path),
            "output_stream_path": str(self.output_stream_path),
            "frame_count": int(self.frame_count),
            "source_shape": list(self.source_shape),
            "source_dtype": self.source_dtype,
            "source_encoding": source_encoding,
            "source_compression": source_compression,
        }


@dataclass(frozen=True)
class ForceProcessConfig:
    dataset_root: Path
    output_root: Path | None = None
    episodes: tuple[int, ...] = ()
    sides: tuple[str, ...] = ()
    sensors: tuple[str, ...] = ()
    jobs: int = 0
    backend: str = "cpu"
    mode: str = "standard"
    cuda_id: int = 0
    remote_addr: str = "192.168.127.10:50051"
    pc_host: str = "0.0.0.0"
    pc_port: int = 60000
    base_frame_index: int | None = None
    calibration_policy: str = "auto"
    calibration_dir: Path | None = None
    save_calibration: bool = False
    compression: str = DEFAULT_FORCE_COMPRESSION
    compression_level: int = DEFAULT_FORCE_COMPRESSION_LEVEL
    chunk_frames: int = DEFAULT_FORCE_CHUNK_FRAMES
    max_frames: int = 0
    sdk_root: Path | None = None
    overwrite: bool = False
    scan_all_tactile: bool = True
    dry_run: bool = False
    progress: bool = True

    def normalized(self) -> "ForceProcessConfig":
        dataset_root = Path(self.dataset_root).expanduser().resolve()
        output_root = Path(self.output_root or dataset_root).expanduser().resolve()
        sides = tuple(str(side).strip().lower() for side in self.sides if str(side).strip())
        unknown_sides = [side for side in sides if side not in {"left", "right"}]
        if unknown_sides:
            raise ValueError(f"unsupported side(s): {unknown_sides}")
        calibration_policy = normalize_calibration_policy(self.calibration_policy)
        if self.save_calibration and self.base_frame_index is None:
            raise ValueError("save_calibration requires an explicit base_frame_index")
        if self.save_calibration and len(self.episodes) != 1:
            raise ValueError("save_calibration requires exactly one episode")
        compression = str(self.compression or DEFAULT_FORCE_COMPRESSION).strip().lower()
        if compression not in {"zstd", "none"}:
            raise ValueError("compression must be zstd or none")
        compression_level = int(self.compression_level)
        if not 1 <= compression_level <= 19:
            raise ValueError("compression_level must be in [1, 19]")
        chunk_frames = int(self.chunk_frames)
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        return ForceProcessConfig(
            dataset_root=dataset_root,
            output_root=output_root,
            episodes=tuple(int(value) for value in self.episodes),
            sides=sides,
            sensors=tuple(str(value) for value in self.sensors if str(value)),
            jobs=int(self.jobs),
            backend=str(self.backend),
            mode=str(self.mode),
            cuda_id=int(self.cuda_id),
            remote_addr=str(self.remote_addr),
            pc_host=str(self.pc_host),
            pc_port=int(self.pc_port),
            base_frame_index=(
                None if self.base_frame_index is None else int(self.base_frame_index)
            ),
            calibration_policy=calibration_policy,
            calibration_dir=Path(self.calibration_dir or default_calibration_dir()).expanduser().resolve(),
            save_calibration=bool(self.save_calibration),
            compression=compression,
            compression_level=compression_level,
            chunk_frames=chunk_frames,
            max_frames=max(0, int(self.max_frames)),
            sdk_root=None if self.sdk_root is None else Path(self.sdk_root).expanduser().resolve(),
            overwrite=bool(self.overwrite),
            scan_all_tactile=bool(self.scan_all_tactile),
            dry_run=bool(self.dry_run),
            progress=bool(self.progress),
        )

    def to_worker_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "mode": self.mode,
            "cuda_id": int(self.cuda_id),
            "remote_addr": self.remote_addr,
            "pc_host": self.pc_host,
            "pc_port": int(self.pc_port),
            "base_frame_index": (
                None if self.base_frame_index is None else int(self.base_frame_index)
            ),
            "calibration_policy": self.calibration_policy,
            "calibration_dir": str(self.calibration_dir or default_calibration_dir()),
            "save_calibration": bool(self.save_calibration),
            "compression": self.compression,
            "compression_level": int(self.compression_level),
            "chunk_frames": int(self.chunk_frames),
            "max_frames": int(self.max_frames),
            "sdk_root": None if self.sdk_root is None else str(self.sdk_root),
            "overwrite": bool(self.overwrite),
        }


@dataclass(frozen=True)
class StreamProcessResult:
    task: dict[str, Any]
    status: str
    frame_count: int = 0
    elapsed_sec: float = 0.0
    output_bytes: int = 0
    error: str | None = None
    calibration: dict[str, Any] = field(default_factory=dict)
    stream_spec: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "status": self.status,
            "frame_count": int(self.frame_count),
            "elapsed_sec": float(self.elapsed_sec),
            "output_bytes": int(self.output_bytes),
            "error": self.error,
            "calibration": _jsonable(self.calibration),
            "stream_spec": _jsonable(self.stream_spec),
        }


@dataclass(frozen=True)
class ForceProcessResult:
    dataset_root: Path
    output_root: Path
    discovered: int
    processed: int
    skipped: int
    failed: int
    output_manifest_count: int
    elapsed_sec: float
    summary_path: Path | None
    stream_results: tuple[StreamProcessResult, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "output_root": str(self.output_root),
            "discovered": int(self.discovered),
            "processed": int(self.processed),
            "skipped": int(self.skipped),
            "failed": int(self.failed),
            "output_manifest_count": int(self.output_manifest_count),
            "elapsed_sec": float(self.elapsed_sec),
            "summary_path": None if self.summary_path is None else str(self.summary_path),
            "stream_results": [item.to_dict() for item in self.stream_results],
        }


def _candidate_manifest_paths(dataset_root: Path, *, scan_all_tactile: bool) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    for record in read_jsonl(episodes_path):
        tactile_paths = record.get("tactile_paths")
        if not isinstance(tactile_paths, Mapping):
            continue
        for value in tactile_paths.values():
            if not value:
                continue
            path = _resolve_dataset_path(dataset_root, value).resolve()
            if path not in seen:
                seen.add(path)
                out.append(path)
    if scan_all_tactile:
        for path in sorted((dataset_root / "tactile").glob("observation.hand.*.tactile/*/episode_*/manifest.json")):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                out.append(resolved)
    return out


def _output_relative_dir(side: str, source_manifest: Mapping[str, Any], episode_index: int) -> Path:
    relative_dir = Path(str(source_manifest.get("relative_dir") or ""))
    parts = list(relative_dir.parts)
    if len(parts) >= 4 and parts[0] == "tactile":
        parts[0] = "tactile_force"
        parts[1] = f"observation.hand.{side}.distributed_force"
        return Path(*parts)
    chunk = "chunk-000"
    if len(parts) >= 3:
        chunk = parts[-2]
    return Path("tactile_force") / f"observation.hand.{side}.distributed_force" / chunk / _episode_id(episode_index)


def _is_raw_stream(spec: Mapping[str, Any]) -> bool:
    dtype = str(spec.get("dtype") or "")
    shape = tuple(int(dim) for dim in spec.get("shape", ()) or ())
    return dtype == RAW_DTYPE and shape == RAW_SHAPE


def discover_raw_tactile_tasks(config: ForceProcessConfig) -> list[RawTactileStreamTask]:
    cfg = config.normalized()
    episode_filter = set(cfg.episodes) if cfg.episodes else None
    side_filter = set(cfg.sides) if cfg.sides else None
    sensor_filter = set(cfg.sensors) if cfg.sensors else None
    tasks: list[RawTactileStreamTask] = []
    for manifest_path in _candidate_manifest_paths(cfg.dataset_root, scan_all_tactile=cfg.scan_all_tactile):
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path)
        if str(manifest.get("format") or "") != TACTILE_STREAM_FORMAT:
            continue
        side = str(manifest.get("side") or "")
        if side_filter is not None and side not in side_filter:
            continue
        episode_index = _episode_index_from_path(manifest_path)
        if episode_index is None:
            continue
        if episode_filter is not None and episode_index not in episode_filter:
            continue
        frames = tuple(
            dict(record)
            for record in manifest.get("frames", []) or []
            if isinstance(record, Mapping)
        )
        streams = manifest.get("streams")
        if not isinstance(streams, Mapping):
            continue
        output_relative_dir = _output_relative_dir(side, manifest, episode_index)
        output_manifest_path = cfg.output_root / output_relative_dir / "manifest.json"
        for sensor_key, raw_spec in streams.items():
            sensor = str(sensor_key)
            if sensor_filter is not None and sensor not in sensor_filter:
                continue
            if not isinstance(raw_spec, Mapping) or not _is_raw_stream(raw_spec):
                continue
            source_stream_path = _resolve_dataset_path(cfg.dataset_root, raw_spec.get("path")).resolve()
            if not source_stream_path.exists():
                continue
            frame_count = int(raw_spec.get("frame_count") or manifest.get("frame_count") or 0)
            if frame_count <= 0:
                continue
            safe_sensor = _safe_path_token(sensor)
            suffix = ".float32.bin.zst" if cfg.compression == "zstd" else ".float32.bin"
            output_stream_path = cfg.output_root / output_relative_dir / f"{safe_sensor}{suffix}"
            tasks.append(
                RawTactileStreamTask(
                    dataset_root=cfg.dataset_root,
                    output_root=cfg.output_root,
                    source_manifest_path=manifest_path.resolve(),
                    output_manifest_path=output_manifest_path,
                    side=side,
                    episode_index=episode_index,
                    sensor_key=sensor,
                    source_stream_path=source_stream_path,
                    output_stream_path=output_stream_path,
                    frame_count=frame_count,
                    source_shape=RAW_SHAPE,
                    source_dtype=RAW_DTYPE,
                    source_stream_spec=dict(raw_spec),
                    source_frames=frames,
                )
            )
    return tasks


def _ensure_dmrobotics_import(sdk_root: str | Path | None = None) -> Any:
    try:
        import dmrobotics  # type: ignore  # noqa: PLC0415

        return dmrobotics
    except ModuleNotFoundError:
        root = Path(sdk_root).expanduser().resolve() if sdk_root is not None else _default_sdk_root()
        if root.exists():
            sys.path.insert(0, str(root))
            try:
                import dmrobotics  # type: ignore  # noqa: PLC0415

                return dmrobotics
            except Exception as exc:  # noqa: BLE001
                raise ModuleNotFoundError(
                    "Failed to import dmrobotics from "
                    f"{root}. Use a Python environment supported by the DM-Tac SDK."
                ) from exc
        raise ModuleNotFoundError(
            "DM-Tac force calculation requires dmrobotics. Install the SDK or pass --sdk-root."
        )


def _default_sdk_root() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "DM_Tac_SDK_Publish_1.2.10"


class DMForceCalculator:
    def __init__(
        self,
        *,
        sensor_key: str,
        backend: str,
        mode: str,
        cuda_id: int,
        remote_addr: str,
        pc_host: str,
        pc_port: int,
        sdk_root: str | Path | None,
    ) -> None:
        dmrobotics = _ensure_dmrobotics_import(sdk_root)
        mode_value = dmrobotics.Mode.HIGH if str(mode).lower() == "high" else dmrobotics.Mode.STANDARD
        options = dmrobotics.SensorOptions(
            dev_id=str(sensor_key),
            backend=str(backend),
            mode=mode_value,
            show_fps=False,
            enable_raw=False,
            enable_deformation=True,
            enable_shear=True,
            enable_depth=True,
            enable_force=True,
            cuda_id=int(cuda_id),
            max_fps=120,
            remote_addr=str(remote_addr),
            pc_host=str(pc_host),
            pc_port=int(pc_port),
        )
        self.dmrobotics = dmrobotics
        self.sensor = dmrobotics.Sensor(options)
        from dmrobotics.src.dmSDK import DMTacImage  # type: ignore  # noqa: PLC0415

        self._image_cls = DMTacImage
        self.sensor_key = str(sensor_key)

    def close(self) -> None:
        try:
            self.sensor.disconnect()
        except Exception:
            pass

    def image(self, raw: np.ndarray) -> Any:
        return self._image_cls(img=np.asarray(raw, dtype=np.uint8), serial=self.sensor_key)

    def set_base_frame(self, raw: np.ndarray) -> None:
        self.sensor.setBaseFrame(self.image(raw))

    def distributed_force(self, raw: np.ndarray) -> np.ndarray:
        image = self.image(raw)
        process_result = self.sensor.process(
            image,
            getdepth=True,
            getshear=True,
            getforce=True,
            getdistforce=True,
        )
        errors: list[str] = []
        # Newer SDK builds may return distributed_force directly from the
        # offline process call. Prefer that result over a live getter, whose
        # disconnected placeholder is a correctly-shaped all-zero array.
        try:
            return _validate_distributed_force(_extract_distributed_force(process_result))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"process: {exc!r}")

        # DM-Tac SDK 1.2.10 computes non-zero offline depth, shear, and global
        # force, but leaves the corresponding DMV1 getter state untouched.
        # getDistributeForce() is the SDK-owned composition step, so mirror the
        # process outputs into that state before calling it. This does not set
        # or synthesize a tactile base frame.
        try:
            self._sync_offline_force_state(process_result)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"offline state sync: {exc!r}")
        try:
            getter_result = self.sensor.getDistributeForce()
        except Exception as exc:  # noqa: BLE001
            getter_result = None
            errors.append(f"getDistributeForce call: {exc!r}")
        try:
            return _validate_distributed_force(_extract_distributed_force(getter_result))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"getDistributeForce: {exc!r}")
        raise ValueError("failed to obtain distributed_force; " + "; ".join(errors))

    def _sync_offline_force_state(self, process_result: Any) -> None:
        if not isinstance(process_result, tuple | list) or len(process_result) < 5:
            raise ValueError("offline process result does not contain depth/shear/force")
        depth = np.asarray(process_result[2], dtype=np.float32)
        shear = np.asarray(process_result[3], dtype=np.float32)
        force = np.asarray(process_result[4], dtype=np.float32)
        if depth.shape != FORCE_SHAPE[:2]:
            raise ValueError(f"offline depth shape {depth.shape!r}")
        if shear.shape != (*FORCE_SHAPE[:2], 2):
            raise ValueError(f"offline shear shape {shear.shape!r}")
        if force.size < 3:
            raise ValueError(f"offline force shape {force.shape!r}")
        hardware = getattr(self.sensor, "hardware", None)
        required = ("_DMV1__s", "_DMV1__d", "_DMV1__n")
        missing = [name for name in required if hardware is None or not hasattr(hardware, name)]
        if missing:
            raise RuntimeError(
                "SDK does not expose the DM-Tac 1.2.10 offline force state: "
                + ", ".join(missing)
            )
        setattr(hardware, "_DMV1__s", np.ascontiguousarray(shear))
        setattr(hardware, "_DMV1__d", np.ascontiguousarray(depth))
        setattr(hardware, "_DMV1__n", np.ascontiguousarray(force.reshape(-1)))


def _validate_distributed_force(distforce: Any) -> np.ndarray:
    if distforce is None:
        raise ValueError("distributed_force is None")
    force = np.asarray(distforce, dtype=np.float32)
    if tuple(force.shape) != FORCE_SHAPE:
        raise ValueError(f"distributed_force shape {force.shape!r}, expected {FORCE_SHAPE!r}")
    return np.ascontiguousarray(force)


def _extract_distributed_force(result: Any) -> Any:
    if isinstance(result, Mapping):
        for key in ("distributed_force", "distforce", "distribute_force"):
            if key in result:
                return result[key]
    if isinstance(result, tuple | list):
        if len(result) >= 6:
            return result[5]
        if len(result) == 2:
            return result[1]
        if len(result) >= 1:
            return result[-1]
    raise ValueError(f"Sensor returned unsupported value for distributed_force: {type(result).__name__}")


def _raw_tactile_frame_reader(task: RawTactileStreamTask) -> Callable[[int], np.ndarray]:
    spec = dict(task.source_stream_spec)
    compression = str(spec.get("compression") or "").strip().lower().replace("-", "_")
    encoding = str(spec.get("encoding") or "").strip().lower().replace("-", "_")
    if not compression and encoding.startswith("chunked_"):
        compression = encoding.removeprefix("chunked_")
    if compression in {"", "none", "raw", "uncompressed"}:
        source = np.memmap(
            task.source_stream_path,
            dtype=np.dtype(task.source_dtype),
            mode="r",
            shape=(int(task.frame_count), *task.source_shape),
        )

        def read_raw_frame(index: int) -> np.ndarray:
            return np.asarray(source[int(index)], dtype=np.uint8)

        return read_raw_frame

    from dataset_contracts.tactile import ChunkedTactileStream

    spec.setdefault("path", str(task.source_stream_path))
    spec.setdefault("dtype", task.source_dtype)
    spec.setdefault("shape", list(task.source_shape))
    spec.setdefault("frame_count", int(task.frame_count))
    stream = ChunkedTactileStream(dataset_root=task.dataset_root, spec=spec)

    def read_compressed_frame(index: int) -> np.ndarray:
        frame = stream.frame(int(index))
        if frame is None:
            raise ValueError(f"missing compressed tactile frame {index} in {task.source_stream_path}")
        return np.asarray(frame, dtype=np.uint8)

    return read_compressed_frame


def _frame_records_for_tasks(tasks: Sequence[RawTactileStreamTask], frame_count: int) -> list[dict[str, Any]]:
    if not tasks:
        return []
    first_task = tasks[0]
    sensor_keys = [task.sensor_key for task in sorted(tasks, key=lambda item: item.sensor_key)]
    records: list[dict[str, Any]] = []
    by_tactile_index: dict[int, Mapping[str, Any]] = {}
    for record in first_task.source_frames:
        try:
            by_tactile_index[int(record.get("tactile_frame_index", len(by_tactile_index)))] = record
        except Exception:
            continue
    for index in range(frame_count):
        source = dict(by_tactile_index.get(index) or {})
        source_shapes = source.get("shapes") if isinstance(source.get("shapes"), Mapping) else {}
        source_dtypes = source.get("dtypes") if isinstance(source.get("dtypes"), Mapping) else {}
        source_field_by_key = source.get("field_by_key") if isinstance(source.get("field_by_key"), Mapping) else {}
        source["tactile_frame_index"] = int(index)
        source.setdefault("sample_frame_index", int(index))
        source["source_shapes"] = dict(source_shapes)
        source["source_dtypes"] = dict(source_dtypes)
        source["source_field_by_key"] = dict(source_field_by_key)
        source["shapes"] = {sensor_key: list(FORCE_SHAPE) for sensor_key in sensor_keys}
        source["dtypes"] = {sensor_key: FORCE_DTYPE for sensor_key in sensor_keys}
        source["field_by_key"] = {sensor_key: "distributed_force" for sensor_key in sensor_keys}
        source["fields"] = ["distributed_force"]
        records.append(source)
    return records


def _base_frame_descriptor(
    task: RawTactileStreamTask,
    worker_config: Mapping[str, Any],
    read_source_frame: Callable[[int], np.ndarray],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    policy = normalize_calibration_policy(str(worker_config.get("calibration_policy") or "auto"))
    calibration_dir = Path(
        worker_config.get("calibration_dir") or default_calibration_dir()
    ).expanduser().resolve()
    base_frame_value = worker_config.get("base_frame_index")
    if base_frame_value is not None:
        index = int(base_frame_value)
        if not 0 <= index < int(task.frame_count):
            raise ValueError(f"base_frame_index {index} out of range for {task.frame_count} frames")
        frame = np.ascontiguousarray(read_source_frame(index), dtype=np.uint8)
        source = {
            "kind": "dataset_raw_frame",
            "dataset_root": str(task.dataset_root),
            "source_manifest_path": str(task.source_manifest_path),
            "episode_index": int(task.episode_index),
            "frame_index": index,
        }
        if bool(worker_config.get("save_calibration", False)):
            saved = save_tactile_calibration(
                calibration_dir,
                task.sensor_key,
                frame,
                source=source,
            )
            descriptor = saved.descriptor()
            descriptor["status"] = "captured_and_saved"
        else:
            descriptor = {
                "status": "explicit_episode_frame",
                "sensor_key": task.sensor_key,
                "sha256": hashlib.sha256(memoryview(frame)).hexdigest(),
                "source": source,
            }
        return frame, descriptor
    if policy == "off":
        return None, {"status": "off", "sensor_key": task.sensor_key}
    calibration = load_tactile_calibration(
        calibration_dir,
        task.sensor_key,
        required=(policy == "required"),
    )
    if calibration is None:
        return None, {
            "status": "missing_raw_solution",
            "sensor_key": task.sensor_key,
            "calibration_dir": str(calibration_dir),
        }
    descriptor = calibration.descriptor()
    descriptor["status"] = "loaded_fixed"
    return calibration.frame, descriptor


def _existing_stream_spec(task: RawTactileStreamTask) -> Mapping[str, Any] | None:
    if not task.output_manifest_path.is_file():
        return None
    try:
        manifest = read_json(task.output_manifest_path)
        streams = manifest.get("streams")
        if not isinstance(streams, Mapping):
            return None
        spec = streams.get(task.sensor_key)
        if not isinstance(spec, Mapping):
            return None
        return spec
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _force_output_settings(worker_config: Mapping[str, Any]) -> tuple[str, int, int]:
    compression = str(worker_config.get("compression") or DEFAULT_FORCE_COMPRESSION).strip().lower()
    level = int(worker_config.get("compression_level") or DEFAULT_FORCE_COMPRESSION_LEVEL)
    chunk_frames = int(worker_config.get("chunk_frames") or DEFAULT_FORCE_CHUNK_FRAMES)
    if compression not in {"zstd", "none"}:
        raise ValueError("compression must be zstd or none")
    if not 1 <= level <= 19:
        raise ValueError("compression_level must be in [1, 19]")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    return compression, level, chunk_frames


def _existing_force_stream_matches(
    task: RawTactileStreamTask,
    spec: Mapping[str, Any] | None,
    *,
    frame_count: int,
    calibration: Mapping[str, Any],
    compression: str,
    compression_level: int,
    chunk_frames: int,
) -> bool:
    if not isinstance(spec, Mapping):
        return False
    if int(spec.get("frame_count") or 0) != int(frame_count):
        return False
    if dict(spec.get("calibration") or {}) != dict(calibration):
        return False
    actual_compression = str(spec.get("compression") or "none").strip().lower()
    if actual_compression != compression:
        return False
    expected_raw_bytes = int(frame_count) * int(np.prod(FORCE_SHAPE)) * np.dtype(FORCE_DTYPE).itemsize
    if compression == "none":
        return task.output_stream_path.stat().st_size == expected_raw_bytes
    if int(spec.get("compression_level") or 0) != int(compression_level):
        return False
    if int(spec.get("chunk_frame_count") or 0) != int(chunk_frames):
        return False
    chunks = spec.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return False
    if sum(int(item.get("raw_bytes") or 0) for item in chunks if isinstance(item, Mapping)) != expected_raw_bytes:
        return False
    return sum(
        int(item.get("compressed_bytes") or 0)
        for item in chunks
        if isinstance(item, Mapping)
    ) == task.output_stream_path.stat().st_size


class _ForceStreamWriter:
    def __init__(
        self,
        handle: Any,
        *,
        compression: str,
        compression_level: int,
        chunk_frames: int,
    ) -> None:
        self.handle = handle
        self.compression = compression
        self.compression_level = int(compression_level)
        self.chunk_frames = int(chunk_frames)
        self.frames: list[np.ndarray] = []
        self.frame_start = 0
        self.chunks: list[dict[str, int]] = []
        self.raw_sha256 = hashlib.sha256()
        self._compressor: Any | None = None
        self._decompressor: Any | None = None
        if compression == "zstd":
            try:
                import zstandard
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Default DM-Tac force compression requires the zstandard Python package. "
                    "Install the project environment or pass --compression none."
                ) from exc
            self._compressor = zstandard.ZstdCompressor(level=self.compression_level)
            self._decompressor = zstandard.ZstdDecompressor()

    def write(self, force: np.ndarray) -> None:
        array = _validate_distributed_force(force)
        if self.compression == "none":
            raw = array.tobytes(order="C")
            self.raw_sha256.update(raw)
            self.handle.write(raw)
            return
        self.frames.append(array.copy())
        if len(self.frames) >= self.chunk_frames:
            self._flush_chunk()

    def finish(self) -> dict[str, Any]:
        if self.frames:
            self._flush_chunk()
        spec: dict[str, Any] = {
            "encoding": "raw_c_order",
            "compression": "none",
            "raw_sha256": self.raw_sha256.hexdigest(),
        }
        if self.compression == "zstd":
            spec.update(
                {
                    "encoding": "chunked_zstd",
                    "compression": "zstd",
                    "compression_level": self.compression_level,
                    "chunk_frame_count": self.chunk_frames,
                    "chunks": list(self.chunks),
                }
            )
        return spec

    def _flush_chunk(self) -> None:
        raw = np.stack(self.frames).astype(np.float32, copy=False).tobytes(order="C")
        self.raw_sha256.update(raw)
        assert self._compressor is not None and self._decompressor is not None
        compressed = self._compressor.compress(raw)
        restored = self._decompressor.decompress(compressed, max_output_size=len(raw))
        if restored != raw:
            raise ValueError(f"Zstd round-trip mismatch at force frame {self.frame_start}")
        offset = int(self.handle.tell())
        self.handle.write(compressed)
        self.chunks.append(
            {
                "frame_start": self.frame_start,
                "frame_count": len(self.frames),
                "offset": offset,
                "compressed_bytes": len(compressed),
                "raw_bytes": len(raw),
            }
        )
        self.frame_start += len(self.frames)
        self.frames.clear()


def process_raw_tactile_stream(
    task: RawTactileStreamTask,
    worker_config: Mapping[str, Any],
    *,
    calculator_factory: Callable[..., DMForceCalculator] = DMForceCalculator,
) -> StreamProcessResult:
    started = time.perf_counter()
    max_frames = int(worker_config.get("max_frames") or 0)
    target_count = int(task.frame_count if max_frames <= 0 else min(task.frame_count, max_frames))
    task_payload = task.to_dict()
    if target_count <= 0:
        return StreamProcessResult(task=task_payload, status="skipped_empty")
    try:
        read_source_frame = _raw_tactile_frame_reader(task)
        base_frame, calibration = _base_frame_descriptor(task, worker_config, read_source_frame)
        compression, compression_level, chunk_frames = _force_output_settings(worker_config)
    except Exception as exc:  # noqa: BLE001
        return StreamProcessResult(
            task=task_payload,
            status="failed",
            elapsed_sec=time.perf_counter() - started,
            error=repr(exc),
        )
    if task.output_stream_path.exists() and not bool(worker_config.get("overwrite", False)):
        size = task.output_stream_path.stat().st_size
        manifest_revision = None
        if task.output_manifest_path.is_file():
            try:
                manifest_payload = read_json(task.output_manifest_path)
                processing = manifest_payload.get("processing")
                if isinstance(processing, Mapping):
                    manifest_revision = processing.get("processor_revision")
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                manifest_revision = None
        existing_spec = _existing_stream_spec(task)
        if (
            manifest_revision == DM_FORCE_PROCESSOR_REVISION
            and _existing_force_stream_matches(
                task,
                existing_spec,
                frame_count=target_count,
                calibration=calibration,
                compression=compression,
                compression_level=compression_level,
                chunk_frames=chunk_frames,
            )
        ):
            return StreamProcessResult(
                task=task_payload,
                status="skipped_exists",
                frame_count=target_count,
                output_bytes=size,
                elapsed_sec=time.perf_counter() - started,
                calibration=calibration,
                stream_spec=dict(existing_spec or {}),
            )
    task.output_stream_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = task.output_stream_path.with_suffix(task.output_stream_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    calculator = calculator_factory(
        sensor_key=task.sensor_key,
        backend=str(worker_config.get("backend", "cpu")),
        mode=str(worker_config.get("mode", "standard")),
        cuda_id=int(worker_config.get("cuda_id", 0)),
        remote_addr=str(worker_config.get("remote_addr", "192.168.127.10:50051")),
        pc_host=str(worker_config.get("pc_host", "0.0.0.0")),
        pc_port=int(worker_config.get("pc_port", 60000)),
        sdk_root=worker_config.get("sdk_root"),
    )
    try:
        if base_frame is not None:
            calculator.set_base_frame(base_frame)
        nonzero_force_frames = 0
        with tmp_path.open("wb") as handle:
            writer = _ForceStreamWriter(
                handle,
                compression=compression,
                compression_level=compression_level,
                chunk_frames=chunk_frames,
            )
            for index in range(target_count):
                force = calculator.distributed_force(read_source_frame(index))
                if np.any(force != 0.0):
                    nonzero_force_frames += 1
                writer.write(force)
            stream_spec = writer.finish()
            handle.flush()
            os.fsync(handle.fileno())
        if nonzero_force_frames == 0:
            raise ValueError(
                "SDK returned all-zero distributed_force for every frame; refusing to publish "
                "a disconnected getter placeholder as a processed stream"
            )
        tmp_path.replace(task.output_stream_path)
    except Exception as exc:  # noqa: BLE001
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return StreamProcessResult(
            task=task_payload,
            status="failed",
            frame_count=0,
            elapsed_sec=time.perf_counter() - started,
            error=repr(exc),
            calibration=calibration,
        )
    finally:
        calculator.close()
    return StreamProcessResult(
        task=task_payload,
        status="processed",
        frame_count=target_count,
        output_bytes=task.output_stream_path.stat().st_size,
        elapsed_sec=time.perf_counter() - started,
        calibration=calibration,
        stream_spec=stream_spec,
    )


def _merge_stream_results_into_manifests(
    tasks: Sequence[RawTactileStreamTask],
    results: Sequence[StreamProcessResult],
    *,
    config: ForceProcessConfig,
) -> list[Path]:
    processed_by_key = {
        (
            Path(result.task["output_manifest_path"]).resolve(),
            str(result.task["sensor_key"]),
        ): result
        for result in results
        if result.status in {"processed", "skipped_exists"}
    }
    manifests: list[Path] = []
    tasks_by_manifest: dict[Path, list[RawTactileStreamTask]] = {}
    for task in tasks:
        key = (task.output_manifest_path.resolve(), task.sensor_key)
        if key not in processed_by_key:
            continue
        tasks_by_manifest.setdefault(task.output_manifest_path, []).append(task)
    for manifest_path, manifest_tasks in sorted(tasks_by_manifest.items(), key=lambda item: str(item[0])):
        previous_manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        previous_stream_paths = _manifest_stream_paths(config.output_root, previous_manifest)
        side = manifest_tasks[0].side
        episode_index = manifest_tasks[0].episode_index
        relative_dir = str(manifest_path.parent.relative_to(config.output_root))
        streams: dict[str, dict[str, Any]] = {}
        frame_count = 0
        first_task = manifest_tasks[0]
        for task in sorted(manifest_tasks, key=lambda item: item.sensor_key):
            result = processed_by_key[(task.output_manifest_path.resolve(), task.sensor_key)]
            written_frames = int(result.frame_count)
            frame_count = max(frame_count, written_frames)
            stream_spec = dict(result.stream_spec)
            stream_spec.update({
                "path": str(task.output_stream_path.relative_to(config.output_root)),
                "dtype": FORCE_DTYPE,
                "shape": list(FORCE_SHAPE),
                "frame_count": written_frames,
                "source_path": str(task.source_stream_path.relative_to(config.dataset_root)),
                "source_dtype": task.source_dtype,
                "source_shape": list(task.source_shape),
                "source_compression": task.to_dict().get("source_compression", "none"),
                "calibration": dict(result.calibration),
            })
            streams[task.sensor_key] = stream_spec
        payload = {
            "format": TACTILE_STREAM_FORMAT,
            "force_format": DM_FORCE_FORMAT,
            "side": side,
            "relative_dir": relative_dir,
            "frame_count": int(frame_count),
            "sensor_keys": sorted(streams),
            "streams": streams,
            "frames": _frame_records_for_tasks(manifest_tasks, frame_count),
            "processing": {
                "created_at": _timestamp_now(),
                "source_dataset_root": str(config.dataset_root),
                "source_manifest_path": str(first_task.source_manifest_path.relative_to(config.dataset_root)),
                "backend": config.backend,
                "mode": config.mode,
                "processor_revision": DM_FORCE_PROCESSOR_REVISION,
                "base_frame_index": (
                    None if config.base_frame_index is None else int(config.base_frame_index)
                ),
                "calibration_policy": config.calibration_policy,
                "calibration_dir": str(config.calibration_dir or default_calibration_dir()),
                "save_calibration": bool(config.save_calibration),
                "compression": config.compression,
                "compression_level": int(config.compression_level),
                "chunk_frame_count": int(config.chunk_frames),
                "max_frames": int(config.max_frames),
                "sdk_root": None if config.sdk_root is None else str(config.sdk_root),
            },
        }
        _atomic_write_json(manifest_path, payload)
        active_stream_paths = _manifest_stream_paths(config.output_root, payload)
        for obsolete_path in sorted(previous_stream_paths - active_stream_paths):
            obsolete_path.unlink(missing_ok=True)
        manifests.append(manifest_path)
    return manifests


def _write_summary(
    *,
    config: ForceProcessConfig,
    tasks: Sequence[RawTactileStreamTask],
    results: Sequence[StreamProcessResult],
    manifests: Sequence[Path],
    elapsed_sec: float,
) -> Path:
    summary_path = config.output_root / "meta" / "dm_force_calculate" / "summary.json"
    payload = {
        "format": DM_FORCE_FORMAT,
        "created_at": _timestamp_now(),
        "dataset_root": str(config.dataset_root),
        "output_root": str(config.output_root),
        "task_count": len(tasks),
        "processed": sum(1 for result in results if result.status == "processed"),
        "skipped": sum(1 for result in results if result.status.startswith("skipped")),
        "failed": sum(1 for result in results if result.status == "failed"),
        "output_manifests": [str(path.relative_to(config.output_root)) for path in manifests],
        "elapsed_sec": float(elapsed_sec),
        "worker_config": config.to_worker_dict(),
        "stream_results": [result.to_dict() for result in results],
    }
    write_json(summary_path, _jsonable(payload))
    return summary_path


def _auto_jobs(requested: int, task_count: int, backend: str) -> int:
    if requested > 0:
        return max(1, int(requested))
    if task_count <= 0:
        return 1
    cpu = os.cpu_count() or 1
    if str(backend).lower() in {"cuda", "flux"}:
        return 1
    return max(1, min(task_count, 4, max(1, cpu // 2)))


def _print_progress(config: ForceProcessConfig, text: str) -> None:
    if config.progress:
        print(text, file=sys.stderr, flush=True)


def process_dataset(config: ForceProcessConfig) -> ForceProcessResult:
    cfg = config.normalized()
    started = time.perf_counter()
    tasks = discover_raw_tactile_tasks(cfg)
    if cfg.dry_run:
        return ForceProcessResult(
            dataset_root=cfg.dataset_root,
            output_root=cfg.output_root,
            discovered=len(tasks),
            processed=0,
            skipped=0,
            failed=0,
            output_manifest_count=0,
            elapsed_sec=time.perf_counter() - started,
            summary_path=None,
            stream_results=(),
        )
    jobs = _auto_jobs(cfg.jobs, len(tasks), cfg.backend)
    _print_progress(cfg, f"[dm-force] discovered {len(tasks)} raw streams; jobs={jobs}")
    results: list[StreamProcessResult] = []
    worker_config = cfg.to_worker_dict()
    if tasks and jobs == 1:
        for index, task in enumerate(tasks, start=1):
            _print_progress(cfg, f"[dm-force] {index}/{len(tasks)} {task.side} {task.sensor_key} ep={task.episode_index}")
            results.append(process_raw_tactile_stream(task, worker_config))
    elif tasks:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            future_to_task = {
                executor.submit(process_raw_tactile_stream, task, worker_config): task
                for task in tasks
            }
            done = 0
            for future in as_completed(future_to_task):
                done += 1
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = StreamProcessResult(
                        task=task.to_dict(),
                        status="failed",
                        error=repr(exc),
                    )
                results.append(result)
                _print_progress(
                    cfg,
                    f"[dm-force] {done}/{len(tasks)} {result.status} "
                    f"{task.side} {task.sensor_key} ep={task.episode_index}",
                )
    manifests = _merge_stream_results_into_manifests(tasks, results, config=cfg)
    elapsed = time.perf_counter() - started
    summary_path = _write_summary(
        config=cfg,
        tasks=tasks,
        results=results,
        manifests=manifests,
        elapsed_sec=elapsed,
    )
    processed = sum(1 for result in results if result.status == "processed")
    skipped = sum(1 for result in results if result.status.startswith("skipped"))
    failed = sum(1 for result in results if result.status == "failed")
    return ForceProcessResult(
        dataset_root=cfg.dataset_root,
        output_root=cfg.output_root,
        discovered=len(tasks),
        processed=processed,
        skipped=skipped,
        failed=failed,
        output_manifest_count=len(manifests),
        elapsed_sec=elapsed,
        summary_path=summary_path,
        stream_results=tuple(results),
    )
