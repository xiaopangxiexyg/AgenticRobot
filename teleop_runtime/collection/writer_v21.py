from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dataset_contracts.causal_snapshot import DEFAULT_CAUSAL_SNAPSHOT_CONTRACT
from dataset_contracts.v23_validation import validate_v23_row_causal_snapshots
from dataset_contracts.pose_frames import (
    POSE_FRAME_CONTRACT_METADATA_KEY,
    POSE_STORAGE_FRAME_DESK,
    PoseFrameContract,
    PoseFrameContractError,
)
from dataset_contracts.stats import (
    FORCE_COLLISION_LABEL_CLEAN,
    FORCE_COLLISION_LABEL_DETECTED,
    EpisodeStatsAccumulator,
    summary_has_force_collision,
)
from dataset_contracts.v23 import (
    DEXTELEOP_DATASET_SCHEMA_V23,
    V23_POSE_ACTION_MODE_METADATA_KEY,
    V23_POSE_STORAGE_FRAME_METADATA_KEY,
    V23_RELATIVE_ACTION_SUPPORTED_METADATA_KEY,
)
from dataset_contracts.v22 import (
    DEXTELEOP_DATASET_SCHEMA_V22,
    LEROBOT_V21_STYLE_FORMAT,
    DatasetLayoutBindingError,
    PolicyIOLayout,
    bind_policy_io_layout,
    policy_io_layout_from_metadata,
)
from dataset_contracts.wrist_depth import (
    DepthRangeMaskConfig,
    depth_range_mask,
    mask_manifest_payload,
    pack_binary_mask,
    raw_depth_manifest_payload,
)

from ..action import (
    ArmActionSpec,
    arm_action_specs_metadata,
    hand_action_specs_metadata,
    normalize_arm_action_specs,
    normalize_hand_action_specs,
)
from ..recording import _to_serializable
from .arm_mode_labels import ARM_MODE_IDS, ARM_MODE_SCHEMA
from .async_writer import WriteJob
from .hand_state_polling import hand_state_tactile_arrays
from .pose_storage import (
    bind_collection_row_pose_storage,
    validate_v23_collection_row_actual_tcp,
)
from .row_encoder import CollectionRowEncoder
from .sample_assembler import CollectorSample
from .schema_v21 import (
    camera_video_channel,
    chunk_id_from_episode_index,
    collector_sample_to_row,
    episode_id_from_index,
    write_rows_to_parquet,
)
from .schema_v22 import (
    LAST_ACTION_INITIALIZATION_METADATA_KEY,
    STRICT_LAST_ACTION_INITIALIZATION_METADATA,
    collector_sample_to_v22_row,
)


ROW_JOB_KIND = "lerobot_v21.row"
VIDEO_FRAME_JOB_KIND = "lerobot_v21.video_frame"
WRIST_DEPTH_FRAME_JOB_KIND = "lerobot_v21.wrist_depth_frame"
TACTILE_FRAME_JOB_KIND = "lerobot_v21.tactile_frame"
FINALIZE_EPISODE_JOB_KIND = "lerobot_v21.finalize_episode"
TACTILE_STORAGE_EXTERNAL = "external"
TACTILE_STORAGE_PARQUET = "parquet"
TACTILE_STREAM_FORMAT = "binary_stream_v1"
WRIST_DEPTH_STREAM_FORMAT = "binary_stream_v1"
TACTILE_COMPRESSION_NONE = "none"
TACTILE_COMPRESSION_LZ4 = "lz4"
TACTILE_COMPRESSION_ZSTD = "zstd"
TACTILE_CHUNK_FRAME_COUNT = 32
WRIST_DEPTH_STALE_REPEAT_FAILURE_STREAK = 8


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _fsync_file_and_parent(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    parent_fd = os.open(path.parent, flags)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


@dataclass(frozen=True)
class LegacyFinalizedVideoArtifact:
    """Durable MP4 artifact facts emitted only by the opt-in finalize API."""

    camera_key: str
    artifact_relative_path: str
    artifact_sha256: str
    artifact_byte_count: int
    encoding: str
    compression: str
    video_frame_index_by_sample_frame_index: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class LegacyEpisodeArtifactReport:
    episode_index: int
    video_artifacts: tuple[LegacyFinalizedVideoArtifact, ...] = ()


def _normalize_tactile_compression(value: Any) -> str:
    text = str(value or TACTILE_COMPRESSION_NONE).strip().lower().replace("-", "_")
    if text in {"", "raw", "uncompressed"}:
        text = TACTILE_COMPRESSION_NONE
    if text not in {
        TACTILE_COMPRESSION_NONE,
        TACTILE_COMPRESSION_LZ4,
        TACTILE_COMPRESSION_ZSTD,
    }:
        raise ValueError("tactile_compression must be one of: none, lz4, zstd.")
    return text


def _compress_tactile_chunk(payload: bytes, *, compression: str) -> bytes:
    if compression == TACTILE_COMPRESSION_NONE:
        return payload
    if compression == TACTILE_COMPRESSION_LZ4:
        try:
            import lz4.frame
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "tactile_compression='lz4' requires the lz4 Python package."
            ) from exc
        return lz4.frame.compress(payload, compression_level=0)
    if compression == TACTILE_COMPRESSION_ZSTD:
        try:
            import zstandard as zstd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "tactile_compression='zstd' requires the zstandard Python package."
            ) from exc
        return zstd.ZstdCompressor(level=1).compress(payload)
    raise ValueError(f"Unsupported tactile compression '{compression}'.")


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("LeRobot video export requires opencv-python.") from exc
    return cv2


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_serializable(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_serializable(payload), ensure_ascii=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _coerce_video_frame(image: Any, *, color_space: str = "rgb") -> np.ndarray:
    cv2 = _require_cv2()
    frame = np.asarray(image)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame.astype(np.uint8, copy=False), cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        frame = frame.astype(np.uint8, copy=False)
        if str(color_space).lower() == "rgb":
            frame = frame[..., ::-1]
        elif str(color_space).lower() != "bgr":
            raise ValueError(f"Unsupported video color_space '{color_space}'.")
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame.astype(np.uint8, copy=False)
        if str(color_space).lower() == "rgba":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif str(color_space).lower() == "bgra":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(f"Unsupported 4-channel video color_space '{color_space}'.")
    else:
        raise ValueError(f"Unsupported video frame shape {frame.shape}.")
    return np.ascontiguousarray(frame)


def _safe_path_token(value: Any) -> str:
    text = str(value).strip() or "unnamed"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "unnamed"


@dataclass
class LeRobotV21WriterConfig:
    root_dir: str | Path
    chunk_size: int = 1000
    nominal_fps: float = 15.0
    parquet_compression: str = "zstd"
    video_codec: str = "mp4v"
    task_id: str = "default"
    task_description: str = ""
    dataset_name: str = ""
    format_name: str = "lerobot_v2.1_style"
    write_parquet: bool = True
    write_video: bool = True
    tactile_storage: str = TACTILE_STORAGE_EXTERNAL
    tactile_compression: str = TACTILE_COMPRESSION_NONE
    allow_missing_video_frames: bool = False
    arm_action_specs: Mapping[str, Any] | None = None
    hand_action_specs: Mapping[str, Any] | None = None
    write_wrist_depth_mask: bool = False
    write_wrist_depth_raw: bool = False
    wrist_depth_camera_keys: tuple[str, ...] = ("left_wrist", "right_wrist")
    wrist_depth_min_m: float = 0.05
    wrist_depth_max_m: float = 0.5
    policy_io_layout: PolicyIOLayout | Mapping[str, Any] | None = None
    policy_io_layout_provenance: Mapping[str, Any] | None = None
    strict_v22_policy_io: bool = False
    pose_frame_contract: PoseFrameContract | Mapping[str, Any] | None = None
    row_encoder: CollectionRowEncoder | None = None
    dataset_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        if int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive.")
        if float(self.nominal_fps) <= 0.0:
            raise ValueError("nominal_fps must be positive.")
        self.write_wrist_depth_mask = bool(self.write_wrist_depth_mask)
        self.write_wrist_depth_raw = bool(self.write_wrist_depth_raw)
        self.wrist_depth_camera_keys = tuple(str(key) for key in self.wrist_depth_camera_keys)
        self.wrist_depth_mask_config = DepthRangeMaskConfig(
            depth_min_m=float(self.wrist_depth_min_m),
            depth_max_m=float(self.wrist_depth_max_m),
        )
        storage = str(self.tactile_storage).strip().lower().replace("-", "_")
        if storage not in {TACTILE_STORAGE_PARQUET, TACTILE_STORAGE_EXTERNAL}:
            raise ValueError("tactile_storage must be parquet or external.")
        self.tactile_storage = storage
        self.tactile_compression = _normalize_tactile_compression(self.tactile_compression)
        requested_arm_action_specs = self.arm_action_specs
        self.hand_action_specs = normalize_hand_action_specs(self.hand_action_specs)
        self.policy_io_layout_provenance = dict(self.policy_io_layout_provenance or {})
        self.strict_v22_policy_io = bool(self.strict_v22_policy_io)
        self.dataset_metadata = dict(self.dataset_metadata or {})
        if self.strict_v22_policy_io and self.row_encoder is None:
            configured_last_action = self.dataset_metadata.get(
                LAST_ACTION_INITIALIZATION_METADATA_KEY
            )
            if (
                configured_last_action is not None
                and configured_last_action != STRICT_LAST_ACTION_INITIALIZATION_METADATA
            ):
                raise ValueError(
                    "dataset_metadata last_action_initialization conflicts with the "
                    "strict standard row encoder contract."
                )
            self.dataset_metadata[LAST_ACTION_INITIALIZATION_METADATA_KEY] = dict(
                STRICT_LAST_ACTION_INITIALIZATION_METADATA
            )
        if self.policy_io_layout is not None:
            if isinstance(self.policy_io_layout, Mapping):
                self.policy_io_layout = PolicyIOLayout.from_dict(self.policy_io_layout)
            elif not isinstance(self.policy_io_layout, PolicyIOLayout):
                raise TypeError("policy_io_layout must be a PolicyIOLayout, mapping, or None.")
            if str(self.format_name) != LEROBOT_V21_STYLE_FORMAT:
                raise ValueError(
                    "DexTeleop v2.2 policy I/O keeps the LeRobot-v2.1-style storage format."
                )
        elif self.strict_v22_policy_io:
            raise ValueError("strict_v22_policy_io requires policy_io_layout.")
        if requested_arm_action_specs is None and self.policy_io_layout is not None:
            self.arm_action_specs = {
                side: ArmActionSpec(
                    side=side,
                    controlled_frame=self.policy_io_layout.arm_controlled_frame_by_side[side],
                    reference_frame=self.policy_io_layout.arm_reference_frame_by_side[side],
                )
                for side in ("left", "right")
            }
        else:
            self.arm_action_specs = normalize_arm_action_specs(requested_arm_action_specs)
        if self.policy_io_layout is not None:
            for side, spec in self.arm_action_specs.items():
                layout_controlled = self.policy_io_layout.arm_controlled_frame_by_side[side]
                layout_reference = self.policy_io_layout.arm_reference_frame_by_side[side]
                if (
                    spec.controlled_frame != layout_controlled
                    or spec.reference_frame != layout_reference
                ):
                    raise ValueError(
                        "arm_action_specs disagree with policy_io_layout for "
                        f"{side}: controlled/reference="
                        f"{spec.controlled_frame!r}/{spec.reference_frame!r} != "
                        f"{layout_controlled!r}/{layout_reference!r}."
                    )
        if self.pose_frame_contract is not None:
            if isinstance(self.pose_frame_contract, Mapping):
                self.pose_frame_contract = PoseFrameContract.from_dict(
                    self.pose_frame_contract
                )
            elif not isinstance(self.pose_frame_contract, PoseFrameContract):
                raise TypeError(
                    "pose_frame_contract must be a PoseFrameContract, mapping, or None."
                )
            if self.policy_io_layout is None:
                raise ValueError(
                    "v2.3 pose_frame_contract requires the v2.2-compatible policy_io_layout."
                )
            for side in self.pose_frame_contract.source.active_sides:
                expected_reference = (
                    self.pose_frame_contract.canonical_absolute_frame
                    if self.pose_frame_contract.pose_storage_frame == POSE_STORAGE_FRAME_DESK
                    else self.pose_frame_contract.source.raw_frame(side)
                )
                actual_reference = self.policy_io_layout.arm_reference_frame_by_side[side]
                if actual_reference != expected_reference:
                    raise ValueError(
                        "v2.3 policy_io_layout arm reference frame disagrees with pose storage "
                        f"for {side}: {actual_reference!r} != {expected_reference!r}."
                    )


@dataclass
class _VideoState:
    camera_key: str
    relative_path: str
    path: Path
    expected_by_sample: dict[int, bool] = field(default_factory=dict)
    pending_frames: dict[int, np.ndarray] = field(default_factory=dict)
    writer: Any | None = None
    frame_size: tuple[int, int] | None = None
    next_sample_index: int = 0
    next_video_frame_index: int = 0
    expected_frame_count: int = 0
    written_frame_count: int = 0
    missing_frame_count: int = 0
    video_frame_index_by_sample_frame_index: dict[int, int] = field(
        default_factory=dict
    )


@dataclass
class _TactileSensorStream:
    sensor_key: str
    relative_path: str
    path: Path
    compression: str = TACTILE_COMPRESSION_NONE
    chunk_frame_count: int = TACTILE_CHUNK_FRAME_COUNT
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    frame_count: int = 0
    handle: Any | None = None
    chunks: list[dict[str, Any]] = field(default_factory=list)
    _chunk_start_frame: int | None = None
    _chunk_frame_count: int = 0
    _chunk_raw_bytes: bytearray = field(default_factory=bytearray)

    @property
    def encoding(self) -> str:
        if self.compression == TACTILE_COMPRESSION_NONE:
            return "raw_c_order"
        return f"chunked_{self.compression}"

    def _ensure_handle(self) -> None:
        if self.handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("ab")

    def _flush_chunk(self) -> None:
        if self._chunk_frame_count <= 0:
            return
        self._ensure_handle()
        assert self.handle is not None
        raw_payload = bytes(self._chunk_raw_bytes)
        compressed = _compress_tactile_chunk(raw_payload, compression=self.compression)
        offset = int(self.handle.tell())
        self.handle.write(compressed)
        self.chunks.append(
            {
                "frame_start": int(self._chunk_start_frame or 0),
                "frame_count": int(self._chunk_frame_count),
                "offset": offset,
                "compressed_bytes": len(compressed),
                "raw_bytes": len(raw_payload),
            }
        )
        self._chunk_raw_bytes.clear()
        self._chunk_start_frame = None
        self._chunk_frame_count = 0

    def append(self, value: Any) -> None:
        array = np.asarray(value)
        if array.dtype == np.dtype("O"):
            raise ValueError(f"Tactile sensor '{self.sensor_key}' has unsupported object dtype.")
        dtype = str(array.dtype)
        if self.dtype is None:
            self.dtype = dtype
        elif str(self.dtype) != dtype:
            raise ValueError(
                f"Tactile sensor '{self.sensor_key}' dtype changed from "
                f"{self.dtype} to {dtype}."
            )
        if self.shape is None:
            self.shape = tuple(int(dim) for dim in array.shape)
        elif tuple(int(dim) for dim in array.shape) != self.shape:
            raise ValueError(
                f"Tactile sensor '{self.sensor_key}' shape changed from "
                f"{self.shape} to {tuple(array.shape)}."
            )
        contiguous = np.ascontiguousarray(array)
        payload = contiguous.tobytes(order="C")
        if self.compression == TACTILE_COMPRESSION_NONE:
            self._ensure_handle()
            assert self.handle is not None
            self.handle.write(payload)
        else:
            if self._chunk_start_frame is None:
                self._chunk_start_frame = int(self.frame_count)
            self._chunk_raw_bytes.extend(payload)
            self._chunk_frame_count += 1
            if self._chunk_frame_count >= int(self.chunk_frame_count):
                self._flush_chunk()
        self.frame_count += 1

    def close(self) -> None:
        self._flush_chunk()
        if self.handle is not None:
            self.handle.close()
            self.handle = None


@dataclass
class _TactileSideState:
    side: str
    relative_dir: str
    path_dir: Path
    manifest_relative_path: str
    manifest_path: Path
    expected_by_sample: dict[int, bool] = field(default_factory=dict)
    pending_frames: dict[int, Mapping[str, Any]] = field(default_factory=dict)
    streams: dict[str, _TactileSensorStream] = field(default_factory=dict)
    sensor_key_order: tuple[str, ...] | None = None
    frame_records: list[dict[str, Any]] = field(default_factory=list)
    next_sample_index: int = 0
    next_tactile_frame_index: int = 0


@dataclass
class _WristDepthStreamState:
    camera_key: str
    stream_kind: str
    relative_dir: str
    path_dir: Path
    manifest_relative_path: str
    manifest_path: Path
    stream_relative_path: str
    stream_path: Path
    expected_by_sample: dict[int, bool] = field(default_factory=dict)
    pending_frames: dict[int, Mapping[str, Any]] = field(default_factory=dict)
    next_sample_index: int = 0
    next_depth_frame_index: int = 0
    missing_frame_count: int = 0
    handle: Any | None = None
    shape: tuple[int, int] | None = None
    frame_records: list[dict[str, Any]] = field(default_factory=list)
    foreground_fraction_sum: float = 0.0
    valid_fraction_sum: float = 0.0

    def close_handle(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


@dataclass
class _EpisodeState:
    episode_index: int
    episode_id: str
    chunk_id: str
    metadata: dict[str, Any]
    rows_by_frame: dict[int, dict[str, Any]] = field(default_factory=dict)
    stats: EpisodeStatsAccumulator | None = None
    video_states: dict[str, _VideoState] = field(default_factory=dict)
    tactile_states: dict[str, _TactileSideState] = field(default_factory=dict)
    wrist_depth_states: dict[tuple[str, str], _WristDepthStreamState] = field(default_factory=dict)
    wrist_depth_integrity: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stats is None:
            self.stats = EpisodeStatsAccumulator(self.episode_index)


class LeRobotV21Writer:
    def __init__(self, config: LeRobotV21WriterConfig):
        self.config = config
        self.root_dir = Path(config.root_dir)
        self.meta_dir = self.root_dir / "meta"
        self.data_dir = self.root_dir / "data"
        self.videos_dir = self.root_dir / "videos"
        self._lock = threading.RLock()
        self._episodes: dict[int, _EpisodeState] = {}
        self._next_episode_index = 0
        self._finalized_episode_count = 0
        self._finalized_frame_count = 0
        self._planned_tactile_snapshot_refs: dict[
            tuple[int, str], tuple[int, int]
        ] = {}
        self._prepare_dataset_root()

    @property
    def info_path(self) -> Path:
        return self.meta_dir / "info.json"

    @property
    def episodes_path(self) -> Path:
        return self.meta_dir / "episodes.jsonl"

    @property
    def tasks_path(self) -> Path:
        return self.meta_dir / "tasks.jsonl"

    @property
    def episode_stats_path(self) -> Path:
        return self.meta_dir / "episodes_stats.jsonl"

    @property
    def next_episode_index(self) -> int:
        with self._lock:
            return int(self._next_episode_index)

    @property
    def next_sample_index(self) -> int:
        with self._lock:
            return int(self._finalized_frame_count)

    def reload_finalized_dataset_state(self) -> None:
        """Refresh counters after an idle, out-of-band dataset maintenance operation."""
        with self._lock:
            if self._episodes:
                raise RuntimeError("Cannot reload dataset state while an episode is active.")
            self._next_episode_index = 0
            self._finalized_episode_count = 0
            self._finalized_frame_count = 0
            # The complete dataset directory may have been removed while this
            # idle writer remained alive. Recreate the required structure and
            # then discover persisted counters from the current filesystem.
            self._prepare_dataset_root()

    @property
    def policy_io_layout(self) -> PolicyIOLayout | None:
        layout = self.config.policy_io_layout
        return layout if isinstance(layout, PolicyIOLayout) else None

    @property
    def is_v22(self) -> bool:
        return self.policy_io_layout is not None

    @property
    def pose_frame_contract(self) -> PoseFrameContract | None:
        contract = self.config.pose_frame_contract
        return contract if isinstance(contract, PoseFrameContract) else None

    def _existing_info_payload(self) -> dict[str, Any]:
        if not self.info_path.exists():
            return {}
        try:
            payload = json.loads(self.info_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cannot read existing dataset info at {self.info_path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Existing dataset info at {self.info_path} is not a JSON object.")
        return payload

    def _validate_existing_policy_io_layout(self) -> None:
        existing = self._existing_info_payload()
        if not existing:
            return
        existing_schema = existing.get("dexteleop_dataset_schema")
        layout = self.policy_io_layout
        has_existing_frames = bool(existing.get("frame_count", 0) or existing.get("episode_count", 0))
        if layout is None:
            if existing_schema in {
                DEXTELEOP_DATASET_SCHEMA_V22,
                DEXTELEOP_DATASET_SCHEMA_V23,
            }:
                raise DatasetLayoutBindingError(
                    "Refusing to append a v2.1 writer to a v2.2/v2.3 policy-I/O dataset. "
                    "Pass the matching policy_io_layout explicitly."
                )
            return
        accepted_schemas = {DEXTELEOP_DATASET_SCHEMA_V22}
        if self.pose_frame_contract is not None:
            accepted_schemas.add(DEXTELEOP_DATASET_SCHEMA_V23)
        if has_existing_frames and existing_schema not in accepted_schemas:
            raise DatasetLayoutBindingError(
                "Refusing to append with a dataset contract that differs from the populated "
                "dataset schema. Use an explicit conversion path instead."
            )
        if existing_schema in {
            DEXTELEOP_DATASET_SCHEMA_V22,
            DEXTELEOP_DATASET_SCHEMA_V23,
        }:
            existing_layout = policy_io_layout_from_metadata(existing, allow_legacy_v21=False)
            if existing_layout.layout_hash != layout.layout_hash:
                raise DatasetLayoutBindingError(
                    "Dataset policy_io_layout_hash differs from the writer configuration."
                )
        self._validate_existing_pose_frame_contract(existing)
        self._validate_existing_dataset_metadata(existing)

    def _validate_existing_dataset_metadata(self, existing: Mapping[str, Any]) -> None:
        if not self.config.dataset_metadata:
            return
        populated = bool(existing.get("frame_count", 0) or existing.get("episode_count", 0))
        for key, expected in self.config.dataset_metadata.items():
            if key not in existing:
                if populated:
                    raise ValueError(
                        f"Refusing to append without immutable dataset metadata {key!r}."
                    )
                continue
            if existing[key] != _to_serializable(expected):
                raise ValueError(
                    f"Existing dataset metadata {key!r} differs from writer configuration."
                )

    def _validate_existing_pose_frame_contract(self, existing: Mapping[str, Any]) -> None:
        configured = self.pose_frame_contract
        populated = bool(existing.get("frame_count", 0) or existing.get("episode_count", 0))
        existing_payload = existing.get(POSE_FRAME_CONTRACT_METADATA_KEY)
        if configured is None:
            if existing_payload is not None:
                raise PoseFrameContractError(
                    "Refusing to append without the existing dataset pose_frame_contract."
                )
            return
        if existing_payload is None:
            if populated:
                raise PoseFrameContractError(
                    "Refusing to append a v2.3 pose contract to a populated dataset without one."
                )
            return
        if not isinstance(existing_payload, Mapping):
            raise PoseFrameContractError(
                "Existing dataset pose_frame_contract must be a mapping."
            )
        existing_contract = PoseFrameContract.from_dict(existing_payload)
        if existing_contract.to_dict() != configured.to_dict():
            raise PoseFrameContractError(
                "Existing dataset pose_frame_contract differs from the writer configuration."
            )
        expected_patch = self._pose_frame_metadata_patch()
        for key in (
            "pose_frame_contract_hash",
            V23_POSE_STORAGE_FRAME_METADATA_KEY,
            V23_RELATIVE_ACTION_SUPPORTED_METADATA_KEY,
            V23_POSE_ACTION_MODE_METADATA_KEY,
            "causal_snapshot_contract",
            "causal_snapshot_contract_detail",
            "causal_snapshot_contract_hash",
        ):
            if key not in existing:
                if populated:
                    raise PoseFrameContractError(
                        f"Refusing to append without immutable v2.3 metadata {key!r}."
                    )
                continue
            if existing[key] != expected_patch[key]:
                raise PoseFrameContractError(
                    f"Existing dataset v2.3 metadata {key!r} differs from writer configuration."
                )

    def _prepare_dataset_root(self) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.episodes_path, self.tasks_path, self.episode_stats_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        self._validate_existing_policy_io_layout()
        self._load_existing_dataset_state()
        self._ensure_task_manifest()
        self._write_info_locked()

    def _load_existing_dataset_state(self) -> None:
        episode_rows = _read_jsonl(self.episodes_path)
        stats_rows = _read_jsonl(self.episode_stats_path)
        self._finalized_episode_count = len(episode_rows)
        if stats_rows:
            self._finalized_frame_count = sum(
                int(row.get("frame_count", 0))
                for row in stats_rows
            )
        else:
            self._finalized_frame_count = sum(
                int(row.get("frame_count", 0))
                for row in episode_rows
            )
        max_episode_index = -1
        for row in episode_rows:
            try:
                max_episode_index = max(max_episode_index, int(row.get("episode_index", -1)))
            except (TypeError, ValueError):
                continue
        self._next_episode_index = max(self._next_episode_index, max_episode_index + 1)

    def _ensure_task_manifest(self) -> None:
        if not self.config.task_id:
            return
        existing = self.tasks_path.read_text(encoding="utf-8").splitlines()
        task_prefix = f"\"task_id\": \"{self.config.task_id}\""
        if any(task_prefix in line for line in existing):
            return
        _append_jsonl(
            self.tasks_path,
            {
                "task_id": self.config.task_id,
                "description": self.config.task_description,
            },
        )

    def _write_info_locked(self) -> None:
        arm_specs = arm_action_specs_metadata(self.config.arm_action_specs)
        hand_specs = hand_action_specs_metadata(self.config.hand_action_specs)
        layout = self.policy_io_layout
        features = (
            layout.feature_metadata()
            if layout is not None
            else {
                "action": {"shape": [26], "dtype": "float32"},
                "observation.state": {"shape": [38], "dtype": "float32"},
                "observation.last_action": {"shape": [26], "dtype": "float32"},
            }
        )
        info: dict[str, Any] = {
                "format": self.config.format_name,
                "dataset_name": self.config.dataset_name,
                "nominal_fps": float(self.config.nominal_fps),
                "chunk_size": int(self.config.chunk_size),
                "episode_count": int(self._finalized_episode_count),
                "frame_count": int(self._finalized_frame_count),
                "write_parquet": bool(self.config.write_parquet),
                "write_video": bool(self.config.write_video),
                "tactile_storage": str(self.config.tactile_storage),
                "tactile_compression": (
                    str(self.config.tactile_compression)
                    if self.config.tactile_storage == TACTILE_STORAGE_EXTERNAL
                    else None
                ),
                "tactile_stream_format": (
                    TACTILE_STREAM_FORMAT
                    if self.config.tactile_storage == TACTILE_STORAGE_EXTERNAL
                    else None
                ),
                "wrist_depth_mask": bool(self.config.write_wrist_depth_mask),
                "wrist_depth_raw": bool(self.config.write_wrist_depth_raw),
                "wrist_depth_camera_keys": list(self.config.wrist_depth_camera_keys),
                "wrist_depth_stream_format": (
                    WRIST_DEPTH_STREAM_FORMAT
                    if self.config.write_wrist_depth_mask or self.config.write_wrist_depth_raw
                    else None
                ),
                "wrist_depth_mask_config": self.config.wrist_depth_mask_config.to_dict(),
                "task_id": self.config.task_id,
                "task_description": self.config.task_description,
                "arm_action_specs_by_side": arm_specs,
                "arm_controlled_frame_by_side": {
                    side: spec["controlled_frame"] for side, spec in arm_specs.items()
                },
                "arm_reference_frame_by_side": {
                    side: spec["reference_frame"] for side, spec in arm_specs.items()
                },
                "arm_spec_required": True,
                "hand_action_specs_by_side": hand_specs,
                "hand_active_mask_by_side": {
                    side: list(spec["active_mask"]) for side, spec in hand_specs.items()
                },
                "hand_spec_required": True,
                "label_specs": {
                    "teleop.arm_mode": {
                        "schema": ARM_MODE_SCHEMA,
                        "columns_optional": True,
                        "sides": ["left", "right"],
                        "mode_ids": dict(ARM_MODE_IDS),
                        "legacy_default_policy": "missing_columns_running_rows_are_teleop_else_unknown",
                        "json_column": "teleop.arm_mode_json",
                        "model_columns": [
                            "teleop.arm_mode.<side>.id",
                            "teleop.arm_mode.<side>.label",
                        ],
                        "audit_only_columns": [
                            "teleop.arm_mode.<side>.scope",
                            "teleop.arm_mode.<side>.init_phase",
                            "teleop.arm_mode.<side>.init_phase_valid",
                            "teleop.arm_mode.<side>.init_target_position_error_m",
                            "teleop.arm_mode.<side>.init_target_rotation_error_deg",
                            "teleop.arm_mode.<side>.init_translation_speed_m_s",
                            "teleop.arm_mode.<side>.init_rotation_speed_deg_s",
                            "teleop.arm_mode.<side>.init_dwell_sec",
                        ],
                    }
                },
                "features": features,
        }
        if layout is not None:
            info["dexteleop_dataset_schema"] = (
                DEXTELEOP_DATASET_SCHEMA_V23
                if self.pose_frame_contract is not None
                else DEXTELEOP_DATASET_SCHEMA_V22
            )
            info.update(bind_policy_io_layout(None, layout))
            info["policy_io_layout_collection"] = {
                **dict(self.config.policy_io_layout_provenance or {}),
                "strict": bool(self.config.strict_v22_policy_io),
            }
            info["policy_io_layout_binding"] = {
                "required_for_training_checkpoint": True,
                "required_for_deploy_runtime": True,
                "legacy_v21_requires_explicit_conversion": True,
            }
        if self.pose_frame_contract is not None:
            info.update(self._pose_frame_metadata_patch())
        for key, value in self.config.dataset_metadata.items():
            if key in info and info[key] != value:
                raise ValueError(
                    f"dataset_metadata cannot override writer-owned field {key!r}"
                )
            info[key] = _to_serializable(value)
        _write_json(self.info_path, info)

    def _pose_frame_metadata_patch(self) -> dict[str, Any]:
        contract = self.pose_frame_contract
        if contract is None:
            return {}
        return {
            "dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23,
            POSE_FRAME_CONTRACT_METADATA_KEY: contract.to_dict(),
            "pose_frame_contract_hash": contract.contract_hash,
            V23_POSE_STORAGE_FRAME_METADATA_KEY: contract.pose_storage_frame,
            V23_RELATIVE_ACTION_SUPPORTED_METADATA_KEY: True,
            V23_POSE_ACTION_MODE_METADATA_KEY: "absolute",
            **DEFAULT_CAUSAL_SNAPSHOT_CONTRACT.metadata_patch(),
        }

    def write_calibration_snapshot(self, name: str, payload: Any) -> Path:
        path = self.meta_dir / "calibration" / f"{name}.json"
        _write_json(path, payload)
        return path

    def write_device_config(self, name: str, payload: Any) -> Path:
        path = self.meta_dir / "devices" / f"{name}.json"
        _write_json(path, payload)
        return path

    def start_episode(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        episode_index: int | None = None,
    ) -> int:
        with self._lock:
            resolved_index = (
                int(self._next_episode_index)
                if episode_index is None
                else int(episode_index)
            )
            self._ensure_episode_state(resolved_index, metadata=metadata)
            return resolved_index

    def tactile_frame_counts(self, episode_index: int) -> dict[str, int]:
        with self._lock:
            episode = self._episodes.get(int(episode_index))
            if episode is None:
                return {}
            return {
                side: int(state.next_tactile_frame_index)
                for side, state in episode.tactile_states.items()
                if int(state.next_tactile_frame_index) > 0
            }

    def _ensure_episode_state(
        self,
        episode_index: int,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> _EpisodeState:
        if episode_index in self._episodes:
            state = self._episodes[episode_index]
            if metadata:
                state.metadata.update(_to_serializable(dict(metadata)))
            self._bind_episode_policy_io_layout(state.metadata)
            self._bind_episode_pose_frame_contract(state.metadata)
            state.metadata.setdefault(
                "arm_action_specs_by_side",
                arm_action_specs_metadata(self.config.arm_action_specs),
            )
            state.metadata.setdefault(
                "hand_action_specs_by_side",
                hand_action_specs_metadata(self.config.hand_action_specs),
            )
            return state
        episode_id = episode_id_from_index(episode_index)
        chunk_id = chunk_id_from_episode_index(
            episode_index,
            chunk_size=self.config.chunk_size,
        )
        episode_metadata = {} if metadata is None else dict(_to_serializable(dict(metadata)))
        self._bind_episode_policy_io_layout(episode_metadata)
        self._bind_episode_pose_frame_contract(episode_metadata)
        episode_metadata.setdefault(
            "arm_action_specs_by_side",
            arm_action_specs_metadata(self.config.arm_action_specs),
        )
        episode_metadata.setdefault(
            "arm_controlled_frame_by_side",
            {
                side: spec.controlled_frame
                for side, spec in self.config.arm_action_specs.items()
            },
        )
        episode_metadata.setdefault(
            "arm_reference_frame_by_side",
            {
                side: spec.reference_frame
                for side, spec in self.config.arm_action_specs.items()
            },
        )
        episode_metadata.setdefault("arm_spec_required", True)
        episode_metadata.setdefault(
            "hand_action_specs_by_side",
            hand_action_specs_metadata(self.config.hand_action_specs),
        )
        episode_metadata.setdefault(
            "hand_active_mask_by_side",
            {
                side: list(spec.active_mask or ())
                for side, spec in self.config.hand_action_specs.items()
            },
        )
        episode_metadata.setdefault("hand_spec_required", True)
        state = _EpisodeState(
            episode_index=int(episode_index),
            episode_id=episode_id,
            chunk_id=chunk_id,
            metadata=episode_metadata,
        )
        self._episodes[episode_index] = state
        self._next_episode_index = max(self._next_episode_index, episode_index + 1)
        return state

    def _bind_episode_policy_io_layout(self, metadata: dict[str, Any]) -> None:
        layout = self.policy_io_layout
        if layout is None:
            return
        binding = bind_policy_io_layout(None, layout)
        existing_hash = metadata.get("policy_io_layout_hash")
        if existing_hash is not None and str(existing_hash) != layout.layout_hash:
            raise DatasetLayoutBindingError(
                "Episode metadata policy_io_layout_hash differs from the writer configuration."
            )
        metadata.update(binding)
        if self.config.policy_io_layout_provenance:
            metadata.setdefault(
                "policy_io_layout_collection",
                {
                    **dict(self.config.policy_io_layout_provenance),
                    "strict": bool(self.config.strict_v22_policy_io),
                },
            )

    def _bind_episode_pose_frame_contract(self, metadata: dict[str, Any]) -> None:
        contract = self.pose_frame_contract
        if contract is None:
            return
        existing = metadata.get(POSE_FRAME_CONTRACT_METADATA_KEY)
        if existing is not None:
            if not isinstance(existing, Mapping):
                raise PoseFrameContractError(
                    "Episode pose_frame_contract must be a mapping."
                )
            if PoseFrameContract.from_dict(existing).to_dict() != contract.to_dict():
                raise PoseFrameContractError(
                    "Episode pose_frame_contract differs from the dataset contract."
                )
        patch = self._pose_frame_metadata_patch()
        for key, expected in patch.items():
            if key in metadata and metadata[key] != expected:
                raise PoseFrameContractError(
                    f"Episode v2.3 metadata {key!r} differs from the dataset contract."
                )
        metadata.update(patch)

    def _episode_parquet_relative_path(self, episode_index: int) -> str:
        chunk_id = chunk_id_from_episode_index(
            episode_index,
            chunk_size=self.config.chunk_size,
        )
        return str(Path("data") / chunk_id / f"{episode_id_from_index(episode_index)}.parquet")

    def _episode_video_relative_path(self, episode_index: int, camera_key: str) -> str:
        chunk_id = chunk_id_from_episode_index(
            episode_index,
            chunk_size=self.config.chunk_size,
        )
        return str(
            Path("videos")
            / camera_video_channel(camera_key)
            / chunk_id
            / f"{episode_id_from_index(episode_index)}.mp4"
        )

    def _episode_tactile_relative_dir(self, episode_index: int, side: str) -> str:
        chunk_id = chunk_id_from_episode_index(
            episode_index,
            chunk_size=self.config.chunk_size,
        )
        return str(
            Path("tactile")
            / f"observation.hand.{str(side)}.tactile"
            / chunk_id
            / episode_id_from_index(episode_index)
        )

    def _episode_tactile_manifest_relative_path(self, episode_index: int, side: str) -> str:
        return str(Path(self._episode_tactile_relative_dir(episode_index, side)) / "manifest.json")

    def _episode_wrist_depth_relative_dir(self, episode_index: int, camera_key: str, stream_kind: str) -> str:
        chunk_id = chunk_id_from_episode_index(
            episode_index,
            chunk_size=self.config.chunk_size,
        )
        return str(
            Path("wrist_depth")
            / str(stream_kind)
            / camera_video_channel(camera_key)
            / chunk_id
            / episode_id_from_index(episode_index)
        )

    def _episode_wrist_depth_manifest_relative_path(
        self,
        episode_index: int,
        camera_key: str,
        stream_kind: str,
    ) -> str:
        return str(Path(self._episode_wrist_depth_relative_dir(episode_index, camera_key, stream_kind)) / "manifest.json")

    def _episode_wrist_depth_stream_relative_path(
        self,
        episode_index: int,
        camera_key: str,
        stream_kind: str,
    ) -> str:
        suffix = "bitpack.bin" if stream_kind == "mask" else "uint16.bin"
        return str(
            Path(self._episode_wrist_depth_relative_dir(episode_index, camera_key, stream_kind))
            / f"stream.{suffix}"
        )

    def _ensure_video_state(self, episode: _EpisodeState, camera_key: str) -> _VideoState:
        if camera_key in episode.video_states:
            return episode.video_states[camera_key]
        relative_path = self._episode_video_relative_path(episode.episode_index, camera_key)
        state = _VideoState(
            camera_key=str(camera_key),
            relative_path=relative_path,
            path=self.root_dir / relative_path,
        )
        episode.video_states[camera_key] = state
        return state

    def _ensure_tactile_state(self, episode: _EpisodeState, side: str) -> _TactileSideState:
        side = str(side)
        if side in episode.tactile_states:
            return episode.tactile_states[side]
        relative_dir = self._episode_tactile_relative_dir(episode.episode_index, side)
        manifest_relative_path = self._episode_tactile_manifest_relative_path(
            episode.episode_index,
            side,
        )
        state = _TactileSideState(
            side=side,
            relative_dir=relative_dir,
            path_dir=self.root_dir / relative_dir,
            manifest_relative_path=manifest_relative_path,
            manifest_path=self.root_dir / manifest_relative_path,
        )
        episode.tactile_states[side] = state
        return state

    def _ensure_wrist_depth_state(
        self,
        episode: _EpisodeState,
        camera_key: str,
        stream_kind: str,
    ) -> _WristDepthStreamState:
        key = (str(camera_key), str(stream_kind))
        if key in episode.wrist_depth_states:
            return episode.wrist_depth_states[key]
        relative_dir = self._episode_wrist_depth_relative_dir(
            episode.episode_index,
            key[0],
            key[1],
        )
        manifest_relative_path = self._episode_wrist_depth_manifest_relative_path(
            episode.episode_index,
            key[0],
            key[1],
        )
        stream_relative_path = self._episode_wrist_depth_stream_relative_path(
            episode.episode_index,
            key[0],
            key[1],
        )
        state = _WristDepthStreamState(
            camera_key=key[0],
            stream_kind=key[1],
            relative_dir=relative_dir,
            path_dir=self.root_dir / relative_dir,
            manifest_relative_path=manifest_relative_path,
            manifest_path=self.root_dir / manifest_relative_path,
            stream_relative_path=stream_relative_path,
            stream_path=self.root_dir / stream_relative_path,
        )
        episode.wrist_depth_states[key] = state
        return state

    def _tactile_arrays_from_hand_state(
        self,
        hand_state: Mapping[str, Any],
    ) -> dict[str, np.ndarray]:
        return hand_state_tactile_arrays(hand_state)

    def _plan_external_tactile_jobs(
        self,
        sample: CollectorSample,
    ) -> tuple[list[WriteJob], dict[str, dict[str, Any]], dict[str, bool]]:
        jobs: list[WriteJob] = []
        refs: dict[str, dict[str, Any]] = {}
        expectations: dict[str, bool] = {}
        for side in ("left", "right"):
            hand_state = sample.hand_states.get(side) or {}
            arrays = self._tactile_arrays_from_hand_state(hand_state)
            has_tactile = bool(arrays)
            snapshot_sequence_raw = hand_state.get("tactile_snapshot_sequence")
            snapshot_sequence = None
            if not isinstance(snapshot_sequence_raw, (bool, np.bool_)):
                try:
                    candidate = int(snapshot_sequence_raw)
                except (TypeError, ValueError):
                    candidate = 0
                if candidate > 0 and candidate == snapshot_sequence_raw:
                    snapshot_sequence = candidate
            expectations[side] = has_tactile
            if not has_tactile:
                continue
            manifest_relative_path = self._episode_tactile_manifest_relative_path(
                sample.episode_index,
                side,
            )
            sensor_keys = sorted(arrays)
            planned_frame_index: int | None = None
            is_new_snapshot = True
            if snapshot_sequence is not None:
                planner_key = (int(sample.episode_index), side)
                with self._lock:
                    previous = self._planned_tactile_snapshot_refs.get(planner_key)
                    if previous is not None and previous[0] == snapshot_sequence:
                        planned_frame_index = int(previous[1])
                        is_new_snapshot = False
                    else:
                        planned_frame_index = 0 if previous is None else int(previous[1]) + 1
            refs[side] = {
                "format": TACTILE_STREAM_FORMAT,
                "path": manifest_relative_path,
                "frame_index": planned_frame_index,
                "sensor_keys": sensor_keys,
            }
            if not is_new_snapshot:
                expectations[side] = False
                continue
            jobs.append(
                WriteJob(
                    kind=TACTILE_FRAME_JOB_KIND,
                    payload={
                        "episode_index": int(sample.episode_index),
                        "frame_index": int(sample.frame_index),
                        "side": side,
                        "arrays": arrays,
                        "timestamp": hand_state.get("tactile_timestamp"),
                        "telemetry_timestamp": hand_state.get("telemetry_timestamp"),
                        "regions": list(
                            hand_state.get("tactile_regions", hand_state.get("tactile_devices", ()))
                        ),
                        "frame_ids": dict(hand_state.get("tactile_frame_ids", {}) or {}),
                        "shapes": dict(hand_state.get("tactile_shapes", {}) or {}),
                        "field_by_key": dict(hand_state.get("tactile_field_by_key", {}) or {}),
                        "fields": list(hand_state.get("tactile_fields", ()) or ()),
                        "version": hand_state.get("tactile_version"),
                        "snapshot_sequence": snapshot_sequence,
                        "planned_frame_index": planned_frame_index,
                        "snapshot_source_timestamp": hand_state.get(
                            "tactile_snapshot_source_timestamp"
                        ),
                        "snapshot_available_timestamp": hand_state.get(
                            "tactile_snapshot_available_timestamp"
                        ),
                        "snapshot_source_clock": hand_state.get(
                            "tactile_snapshot_source_clock"
                        ),
                        "snapshot_available_clock": hand_state.get(
                            "tactile_snapshot_available_clock"
                        ),
                    },
                )
            )
        return jobs, refs, expectations

    def _wrist_depth_enabled(self) -> bool:
        return bool(self.config.write_wrist_depth_mask or self.config.write_wrist_depth_raw)

    def _wrist_depth_camera_selected(self, camera_key: str) -> bool:
        return str(camera_key) in set(self.config.wrist_depth_camera_keys)

    @staticmethod
    def _depth_scale_from_metadata(metadata: Mapping[str, Any]) -> float | None:
        value = metadata.get("depth_scale_m_per_unit")
        try:
            scale = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(scale) or scale <= 0.0:
            return None
        return float(scale)

    def _plan_wrist_depth_jobs(
        self,
        sample: CollectorSample,
    ) -> tuple[list[WriteJob], dict[str, dict[str, bool]], dict[str, dict[str, Any]]]:
        if not self._wrist_depth_enabled():
            return [], {}, {}
        jobs: list[WriteJob] = []
        expectations: dict[str, dict[str, bool]] = {}
        status_by_camera: dict[str, dict[str, Any]] = {}
        for camera_key, match in sample.camera_frames.items():
            camera_name = str(camera_key)
            if not self._wrist_depth_camera_selected(camera_name):
                continue
            frame = match.frame
            has_frame = frame is not None
            has_depth = has_frame and frame.depth_image is not None
            depth_metadata = {} if frame is None else dict(frame.depth_metadata)
            has_scale = self._depth_scale_from_metadata(depth_metadata) is not None
            failure_reasons: list[str] = []
            if not has_frame:
                failure_reasons.append("missing_frame")
            elif not has_depth:
                failure_reasons.append("missing_depth_image")
            elif self.config.write_wrist_depth_mask and not has_scale:
                failure_reasons.append("missing_depth_scale")
            expectations[camera_name] = {
                "mask": bool(self.config.write_wrist_depth_mask and has_depth and has_scale),
                "raw": bool(self.config.write_wrist_depth_raw and has_depth),
            }
            status_by_camera[camera_name] = {
                "frame_present": bool(has_frame),
                "depth_present": bool(has_depth),
                "depth_scale_present": bool(has_scale),
                "source_timestamp": match.selected_timestamp if frame is not None else None,
                "source_frame_index": frame.frame_index if frame is not None else None,
                "depth_camera_timestamp_sec": depth_metadata.get("camera_timestamp_sec"),
                "depth_sensor_timestamp_ms": depth_metadata.get("sensor_timestamp_ms"),
                "depth_frame_number": depth_metadata.get("frame_number"),
                "mask_requested": bool(self.config.write_wrist_depth_mask),
                "raw_requested": bool(self.config.write_wrist_depth_raw),
                "mask_ready": bool(self.config.write_wrist_depth_mask and has_depth and has_scale),
                "raw_ready": bool(self.config.write_wrist_depth_raw and has_depth),
                "failure_reasons": list(failure_reasons),
            }
            if not has_depth:
                continue
            assert frame is not None
            jobs.append(
                WriteJob(
                    kind=WRIST_DEPTH_FRAME_JOB_KIND,
                    payload={
                        "episode_index": int(sample.episode_index),
                        "frame_index": int(sample.frame_index),
                        "camera_key": camera_name,
                        "depth_image": frame.depth_image,
                        "depth_metadata": depth_metadata,
                        "source_timestamp": match.selected_timestamp,
                        "receive_timestamp": frame.receive_timestamp,
                        "available_timestamp": frame.available_timestamp,
                        "source_frame_index": frame.frame_index,
                    },
                    critical=False,
                    metadata={
                        "storage_queue": "wrist_depth",
                        "camera_key": camera_name,
                    },
                )
            )
        return jobs, expectations, status_by_camera

    def plan_sample_jobs(self, sample: CollectorSample) -> list[WriteJob]:
        with self._lock:
            self._ensure_episode_state(sample.episode_index)
        camera_video_paths = (
            {
                camera_key: self._episode_video_relative_path(sample.episode_index, camera_key)
                for camera_key in sample.camera_frames
            }
            if self.config.write_video
            else None
        )
        tactile_jobs: list[WriteJob] = []
        tactile_refs: dict[str, dict[str, Any]] | None = None
        tactile_expectations: dict[str, bool] = {}
        inline_tactile = self.config.tactile_storage != TACTILE_STORAGE_EXTERNAL
        if self.config.tactile_storage == TACTILE_STORAGE_EXTERNAL:
            tactile_jobs, tactile_refs, tactile_expectations = self._plan_external_tactile_jobs(
                sample
            )
        wrist_depth_jobs, wrist_depth_expectations, wrist_depth_status = self._plan_wrist_depth_jobs(sample)
        layout = self.policy_io_layout
        if self.config.row_encoder is not None:
            row, camera_expectations = self.config.row_encoder.encode_row(
                sample,
                layout=layout,
                camera_video_paths=camera_video_paths,
                tactile_storage_refs=tactile_refs,
                inline_tactile=inline_tactile,
            )
        elif layout is None:
            row, camera_expectations = collector_sample_to_row(
                sample,
                camera_video_paths=camera_video_paths,
                tactile_storage_refs=tactile_refs,
                inline_tactile=inline_tactile,
            )
        else:
            row, camera_expectations = collector_sample_to_v22_row(
                sample,
                layout=layout,
                camera_video_paths=camera_video_paths,
                tactile_storage_refs=tactile_refs,
                inline_tactile=inline_tactile,
                strict=bool(self.config.strict_v22_policy_io),
            )
        if self.pose_frame_contract is not None:
            validate_v23_collection_row_actual_tcp(
                row,
                contract=self.pose_frame_contract,
            )
            arm_source_clocks = ["local_wall_time"]
            if self.pose_frame_contract.source.source_family == "vive_umi":
                arm_source_clocks.extend(
                    [
                        "vive_sender_monotonic_calibrated_to_local_wall",
                        "vive_receiver_wall_fallback",
                    ]
                )
            validate_v23_row_causal_snapshots(
                row,
                active_sides=self.pose_frame_contract.source.active_sides,
                arm_source_clocks=arm_source_clocks,
            )
            row = bind_collection_row_pose_storage(
                row,
                contract=self.pose_frame_contract,
            )
            row["arm_action_specs_json"] = json.dumps(
                arm_action_specs_metadata(self.config.arm_action_specs),
                ensure_ascii=True,
                sort_keys=True,
            )
        if not self.config.write_video:
            camera_expectations = {camera_key: False for camera_key in camera_expectations}
        # Commit tactile deduplication only after row encoding and every v2.3
        # fail-closed check succeeded.  A rejected row must not make a later
        # valid row believe that its large immutable snapshot was written.
        with self._lock:
            for tactile_job in tactile_jobs:
                sequence = tactile_job.payload.get("snapshot_sequence")
                frame_index = tactile_job.payload.get("planned_frame_index")
                if sequence is None or frame_index is None:
                    continue
                planner_key = (
                    int(tactile_job.payload["episode_index"]),
                    str(tactile_job.payload["side"]),
                )
                self._planned_tactile_snapshot_refs[planner_key] = (
                    int(sequence),
                    int(frame_index),
                )
        jobs = [
            WriteJob(
                kind=ROW_JOB_KIND,
                payload={
                    "episode_index": int(sample.episode_index),
                    "frame_index": int(sample.frame_index),
                    "row": row,
                    "camera_expectations": dict(camera_expectations),
                    "tactile_expectations": dict(tactile_expectations),
                    "wrist_depth_expectations": dict(wrist_depth_expectations),
                    "wrist_depth_status": dict(wrist_depth_status),
                },
            )
        ]
        jobs.extend(tactile_jobs)
        jobs.extend(wrist_depth_jobs)
        if self.config.write_video:
            for camera_key, match in sample.camera_frames.items():
                if match.frame is None:
                    continue
                jobs.append(
                    WriteJob(
                        kind=VIDEO_FRAME_JOB_KIND,
                        payload={
                            "episode_index": int(sample.episode_index),
                            "frame_index": int(sample.frame_index),
                            "camera_key": str(camera_key),
                            "image": match.frame.image,
                            "color_space": match.frame.metadata.get("color_space", "rgb"),
                        },
                        critical=False,
                        metadata={
                            "storage_queue": "video",
                            "camera_key": str(camera_key),
                        },
                    )
                )
        return jobs

    def plan_finalize_episode(
        self,
        episode_index: int,
        *,
        summary: Mapping[str, Any] | None = None,
    ) -> list[WriteJob]:
        return [
            WriteJob(
                kind=FINALIZE_EPISODE_JOB_KIND,
                payload={
                    "episode_index": int(episode_index),
                    "summary": {} if summary is None else dict(_to_serializable(dict(summary))),
                },
            )
        ]

    def handle_job(self, job: WriteJob) -> None:
        with self._lock:
            if job.kind == ROW_JOB_KIND:
                self._handle_row_job(job.payload)
                return
            if job.kind == VIDEO_FRAME_JOB_KIND:
                self._handle_video_frame_job(job.payload)
                return
            if job.kind == WRIST_DEPTH_FRAME_JOB_KIND:
                self._handle_wrist_depth_frame_job(job.payload)
                return
            if job.kind == TACTILE_FRAME_JOB_KIND:
                self._handle_tactile_frame_job(job.payload)
                return
            if job.kind == FINALIZE_EPISODE_JOB_KIND:
                self._finalize_episode_locked(
                    int(job.payload["episode_index"]),
                    summary=job.payload.get("summary"),
                )
                return
        raise ValueError(f"Unsupported LeRobotV21Writer job kind '{job.kind}'.")

    def _handle_row_job(self, payload: Mapping[str, Any]) -> None:
        episode = self._ensure_episode_state(int(payload["episode_index"]))
        frame_index = int(payload["frame_index"])
        row = dict(payload["row"])
        episode.rows_by_frame[frame_index] = row
        assert episode.stats is not None
        episode.stats.update(row)
        self._record_wrist_depth_status(
            episode,
            frame_index=frame_index,
            status_by_camera=payload.get("wrist_depth_status", {}),
        )
        for camera_key, expects_frame in dict(payload.get("camera_expectations", {})).items():
            video_state = self._ensure_video_state(episode, str(camera_key))
            video_state.expected_by_sample[frame_index] = bool(expects_frame)
            if bool(expects_frame):
                video_state.expected_frame_count += 1
            self._drain_video_state(episode, video_state)
        for side, expects_frame in dict(payload.get("tactile_expectations", {})).items():
            tactile_state = self._ensure_tactile_state(episode, str(side))
            tactile_state.expected_by_sample[frame_index] = bool(expects_frame)
            self._drain_tactile_state(episode, tactile_state)
            self._apply_latest_tactile_ref_to_row(row, tactile_state)
        for camera_key, expectations in dict(payload.get("wrist_depth_expectations", {})).items():
            if not isinstance(expectations, Mapping):
                continue
            for stream_kind in ("mask", "raw"):
                if stream_kind == "mask" and not self.config.write_wrist_depth_mask:
                    continue
                if stream_kind == "raw" and not self.config.write_wrist_depth_raw:
                    continue
                expects_frame = bool(expectations.get(stream_kind))
                depth_state = self._ensure_wrist_depth_state(
                    episode,
                    str(camera_key),
                    stream_kind,
                )
                depth_state.expected_by_sample[frame_index] = expects_frame
                self._drain_wrist_depth_state(episode, depth_state)

    @staticmethod
    def _new_wrist_depth_integrity_stats(camera_key: str) -> dict[str, Any]:
        return {
            "camera_key": str(camera_key),
            "samples": 0,
            "ok_samples": 0,
            "failure_count": 0,
            "frame_present_count": 0,
            "depth_present_count": 0,
            "depth_scale_present_count": 0,
            "mask_requested": False,
            "raw_requested": False,
            "mask_ready_count": 0,
            "raw_ready_count": 0,
            "missing_frame_count": 0,
            "missing_depth_image_count": 0,
            "missing_depth_scale_count": 0,
            "depth_frame_number_present_count": 0,
            "depth_camera_timestamp_present_count": 0,
            "repeated_depth_frame_count": 0,
            "stale_depth_frame_count": 0,
            "repeated_depth_frame_streak": 0,
            "max_repeated_depth_frame_streak": 0,
            "last_depth_frame_number": None,
            "last_depth_camera_timestamp_sec": None,
            "last_depth_sensor_timestamp_ms": None,
            "last_depth_source_frame_index": None,
            "failure_reason_counts": {},
            "first_failure_frame_index": None,
            "first_failure_reasons": [],
            "last_failure_frame_index": None,
            "last_failure_reasons": [],
            "failure_frame_indices_first": [],
            "failure_frame_indices_truncated": False,
        }

    def _record_wrist_depth_status(
        self,
        episode: _EpisodeState,
        *,
        frame_index: int,
        status_by_camera: object,
    ) -> None:
        if not self._wrist_depth_enabled() or not isinstance(status_by_camera, Mapping):
            return
        for camera_key, raw_status in dict(status_by_camera).items():
            if not isinstance(raw_status, Mapping):
                continue
            camera_name = str(camera_key)
            stats = episode.wrist_depth_integrity.setdefault(
                camera_name,
                self._new_wrist_depth_integrity_stats(camera_name),
            )
            stats["samples"] = int(stats.get("samples", 0)) + 1
            frame_present = bool(raw_status.get("frame_present"))
            depth_present = bool(raw_status.get("depth_present"))
            depth_scale_present = bool(raw_status.get("depth_scale_present"))
            mask_requested = bool(raw_status.get("mask_requested"))
            raw_requested = bool(raw_status.get("raw_requested"))
            mask_ready = bool(raw_status.get("mask_ready"))
            raw_ready = bool(raw_status.get("raw_ready"))
            stats["mask_requested"] = bool(stats.get("mask_requested")) or mask_requested
            stats["raw_requested"] = bool(stats.get("raw_requested")) or raw_requested
            if frame_present:
                stats["frame_present_count"] = int(stats.get("frame_present_count", 0)) + 1
            if depth_present:
                stats["depth_present_count"] = int(stats.get("depth_present_count", 0)) + 1
            if depth_scale_present:
                stats["depth_scale_present_count"] = int(stats.get("depth_scale_present_count", 0)) + 1
            if mask_ready:
                stats["mask_ready_count"] = int(stats.get("mask_ready_count", 0)) + 1
            if raw_ready:
                stats["raw_ready_count"] = int(stats.get("raw_ready_count", 0)) + 1

            raw_reasons = raw_status.get("failure_reasons", ())
            if isinstance(raw_reasons, str):
                reasons = [raw_reasons]
            elif isinstance(raw_reasons, (list, tuple)):
                reasons = [str(reason) for reason in raw_reasons if str(reason)]
            else:
                reasons = []
            if depth_present:
                depth_frame_number = raw_status.get("depth_frame_number")
                depth_camera_timestamp_sec = raw_status.get("depth_camera_timestamp_sec")
                depth_sensor_timestamp_ms = raw_status.get("depth_sensor_timestamp_ms")
                source_frame_index = raw_status.get("source_frame_index")
                if depth_frame_number is not None:
                    stats["depth_frame_number_present_count"] = (
                        int(stats.get("depth_frame_number_present_count", 0)) + 1
                    )
                if depth_camera_timestamp_sec is not None:
                    stats["depth_camera_timestamp_present_count"] = (
                        int(stats.get("depth_camera_timestamp_present_count", 0)) + 1
                    )
                previous_signature = (
                    stats.get("last_depth_frame_number"),
                    stats.get("last_depth_camera_timestamp_sec"),
                    stats.get("last_depth_sensor_timestamp_ms"),
                )
                current_signature = (
                    depth_frame_number,
                    depth_camera_timestamp_sec,
                    depth_sensor_timestamp_ms,
                )
                signature_present = any(value is not None for value in current_signature)
                previous_present = any(value is not None for value in previous_signature)
                source_changed = (
                    source_frame_index is None
                    or stats.get("last_depth_source_frame_index") is None
                    or source_frame_index != stats.get("last_depth_source_frame_index")
                )
                repeated = bool(
                    signature_present
                    and previous_present
                    and source_changed
                    and current_signature == previous_signature
                )
                if repeated:
                    repeat_streak = int(stats.get("repeated_depth_frame_streak", 0)) + 1
                    stats["repeated_depth_frame_count"] = (
                        int(stats.get("repeated_depth_frame_count", 0)) + 1
                    )
                else:
                    repeat_streak = 0
                stats["repeated_depth_frame_streak"] = int(repeat_streak)
                stats["max_repeated_depth_frame_streak"] = max(
                    int(stats.get("max_repeated_depth_frame_streak", 0)),
                    int(repeat_streak),
                )
                if repeat_streak >= WRIST_DEPTH_STALE_REPEAT_FAILURE_STREAK:
                    reasons.append("stale_depth_frame")
                stats["last_depth_frame_number"] = depth_frame_number
                stats["last_depth_camera_timestamp_sec"] = depth_camera_timestamp_sec
                stats["last_depth_sensor_timestamp_ms"] = depth_sensor_timestamp_ms
                stats["last_depth_source_frame_index"] = source_frame_index
            if not reasons:
                stats["ok_samples"] = int(stats.get("ok_samples", 0)) + 1
                continue

            stats["failure_count"] = int(stats.get("failure_count", 0)) + 1
            reason_counts = stats.setdefault("failure_reason_counts", {})
            if not isinstance(reason_counts, dict):
                reason_counts = {}
                stats["failure_reason_counts"] = reason_counts
            for reason in reasons:
                reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
            if "missing_frame" in reasons:
                stats["missing_frame_count"] = int(stats.get("missing_frame_count", 0)) + 1
            if "missing_depth_image" in reasons:
                stats["missing_depth_image_count"] = int(stats.get("missing_depth_image_count", 0)) + 1
            if "missing_depth_scale" in reasons:
                stats["missing_depth_scale_count"] = int(stats.get("missing_depth_scale_count", 0)) + 1
            if "stale_depth_frame" in reasons:
                stats["stale_depth_frame_count"] = int(stats.get("stale_depth_frame_count", 0)) + 1
            if stats.get("first_failure_frame_index") is None:
                stats["first_failure_frame_index"] = int(frame_index)
                stats["first_failure_reasons"] = list(reasons)
            stats["last_failure_frame_index"] = int(frame_index)
            stats["last_failure_reasons"] = list(reasons)
            failure_indices = stats.setdefault("failure_frame_indices_first", [])
            if isinstance(failure_indices, list) and len(failure_indices) < 64:
                failure_indices.append(int(frame_index))
            else:
                stats["failure_frame_indices_truncated"] = True

    def _wrist_depth_integrity_payload(self, episode: _EpisodeState) -> dict[str, Any] | None:
        if not self._wrist_depth_enabled() and not episode.wrist_depth_integrity:
            return None
        cameras: dict[str, dict[str, Any]] = {}
        for camera_key, raw_stats in sorted(episode.wrist_depth_integrity.items()):
            stats = dict(raw_stats)
            stats["failure_reason_counts"] = dict(stats.get("failure_reason_counts", {}) or {})
            stats["ok"] = int(stats.get("failure_count", 0)) == 0
            cameras[str(camera_key)] = stats
        configured_keys = [str(key) for key in self.config.wrist_depth_camera_keys]
        total_samples = sum(int(stats.get("samples", 0)) for stats in cameras.values())
        total_failures = sum(int(stats.get("failure_count", 0)) for stats in cameras.values())
        return {
            "enabled": bool(self._wrist_depth_enabled()),
            "mask_requested": bool(self.config.write_wrist_depth_mask),
            "raw_requested": bool(self.config.write_wrist_depth_raw),
            "camera_keys": list(configured_keys),
            "unseen_camera_keys": [key for key in configured_keys if key not in cameras],
            "sample_observations": int(total_samples),
            "failure_count": int(total_failures),
            "ok": int(total_failures) == 0,
            "cameras": cameras,
        }

    def _handle_video_frame_job(self, payload: Mapping[str, Any]) -> None:
        episode = self._ensure_episode_state(int(payload["episode_index"]))
        camera_key = str(payload["camera_key"])
        video_state = self._ensure_video_state(episode, camera_key)
        video_state.pending_frames[int(payload["frame_index"])] = _coerce_video_frame(
            payload["image"],
            color_space=str(payload.get("color_space", "rgb")),
        )
        self._drain_video_state(episode, video_state)

    def _handle_tactile_frame_job(self, payload: Mapping[str, Any]) -> None:
        episode = self._ensure_episode_state(int(payload["episode_index"]))
        side = str(payload["side"])
        tactile_state = self._ensure_tactile_state(episode, side)
        tactile_state.pending_frames[int(payload["frame_index"])] = payload
        self._drain_tactile_state(episode, tactile_state)

    def _handle_wrist_depth_frame_job(self, payload: Mapping[str, Any]) -> None:
        episode = self._ensure_episode_state(int(payload["episode_index"]))
        camera_key = str(payload["camera_key"])
        if not self._wrist_depth_camera_selected(camera_key):
            return
        if self.config.write_wrist_depth_mask and self._depth_scale_from_metadata(
            dict(payload.get("depth_metadata", {}) or {})
        ) is not None:
            mask_state = self._ensure_wrist_depth_state(episode, camera_key, "mask")
            mask_state.pending_frames[int(payload["frame_index"])] = payload
            self._drain_wrist_depth_state(episode, mask_state)
        if self.config.write_wrist_depth_raw:
            raw_state = self._ensure_wrist_depth_state(episode, camera_key, "raw")
            raw_state.pending_frames[int(payload["frame_index"])] = payload
            self._drain_wrist_depth_state(episode, raw_state)

    def _open_video_writer(self, state: _VideoState, frame: np.ndarray) -> None:
        cv2 = _require_cv2()
        height, width = frame.shape[:2]
        state.path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(state.path),
            cv2.VideoWriter_fourcc(*self.config.video_codec),
            float(self.config.nominal_fps),
            (int(width), int(height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for '{state.path}'.")
        state.writer = writer
        state.frame_size = (int(width), int(height))

    def _drain_video_state(self, episode: _EpisodeState, state: _VideoState) -> None:
        while state.next_sample_index in state.expected_by_sample:
            sample_index = int(state.next_sample_index)
            expects_frame = bool(state.expected_by_sample[sample_index])
            if not expects_frame:
                del state.expected_by_sample[sample_index]
                state.next_sample_index += 1
                continue
            if sample_index not in state.pending_frames:
                return
            frame = state.pending_frames.pop(sample_index)
            if state.writer is None:
                self._open_video_writer(state, frame)
            assert state.frame_size is not None
            expected_width, expected_height = state.frame_size
            if frame.shape[1] != expected_width or frame.shape[0] != expected_height:
                raise ValueError(
                    f"Video frame for camera '{state.camera_key}' has shape {frame.shape[:2]}, "
                    f"expected {(expected_height, expected_width)}."
                )
            assert state.writer is not None
            state.writer.write(frame)
            state.written_frame_count += 1
            state.video_frame_index_by_sample_frame_index[sample_index] = int(
                state.next_video_frame_index
            )
            row = episode.rows_by_frame.get(sample_index)
            if row is not None:
                row[f"{camera_video_channel(state.camera_key)}.video_frame_index"] = int(
                    state.next_video_frame_index
                )
            del state.expected_by_sample[sample_index]
            state.next_video_frame_index += 1
            state.next_sample_index += 1

    def _close_video_state(self, state: _VideoState) -> None:
        if state.writer is not None:
            state.writer.release()
            state.writer = None

    def _sensor_stream_for(
        self,
        state: _TactileSideState,
        sensor_key: str,
        dtype: str,
    ) -> _TactileSensorStream:
        if sensor_key in state.streams:
            return state.streams[sensor_key]
        safe_name = _safe_path_token(sensor_key)
        used_names: set[str] = set()
        for stream in state.streams.values():
            name = Path(stream.relative_path).name
            for suffix_text in (".bin.lz4", ".bin.zst", ".bin"):
                if name.endswith(suffix_text):
                    name = name[: -len(suffix_text)]
                    break
            used_names.add(name.rsplit(".", 1)[0])
        base_name = safe_name
        suffix = 1
        while safe_name in used_names:
            suffix += 1
            safe_name = f"{base_name}_{suffix}"
        safe_dtype = _safe_path_token(dtype)
        compression = str(self.config.tactile_compression)
        extension = "bin"
        if compression == TACTILE_COMPRESSION_LZ4:
            extension = "bin.lz4"
        elif compression == TACTILE_COMPRESSION_ZSTD:
            extension = "bin.zst"
        relative_path = str(Path(state.relative_dir) / f"{safe_name}.{safe_dtype}.{extension}")
        stream = _TactileSensorStream(
            sensor_key=str(sensor_key),
            relative_path=relative_path,
            path=self.root_dir / relative_path,
            compression=compression,
            chunk_frame_count=TACTILE_CHUNK_FRAME_COUNT,
            dtype=str(dtype),
        )
        state.streams[sensor_key] = stream
        return stream

    def _append_tactile_payload(
        self,
        state: _TactileSideState,
        *,
        sample_index: int,
        payload: Mapping[str, Any],
    ) -> tuple[int, tuple[str, ...]]:
        raw_arrays = payload.get("arrays", {})
        if not isinstance(raw_arrays, Mapping):
            raise ValueError("Tactile frame payload arrays must be a mapping.")
        arrays = {
            str(key): np.asarray(value)
            for key, value in raw_arrays.items()
            if value is not None and np.asarray(value).size > 0
        }
        if not arrays:
            raise ValueError("Tactile frame payload does not contain any arrays.")
        sensor_keys = tuple(sorted(arrays))
        if state.sensor_key_order is None:
            state.sensor_key_order = sensor_keys
        elif sensor_keys != state.sensor_key_order:
            raise ValueError(
                f"Tactile sensor set for side '{state.side}' changed from "
                f"{state.sensor_key_order} to {sensor_keys}."
            )

        tactile_frame_index = int(state.next_tactile_frame_index)
        for sensor_key in sensor_keys:
            array = arrays[sensor_key]
            if array.dtype == np.dtype("O"):
                raise ValueError(f"Tactile sensor '{sensor_key}' has unsupported object dtype.")
            self._sensor_stream_for(state, sensor_key, str(array.dtype)).append(array)
        dtypes = {sensor_key: str(arrays[sensor_key].dtype) for sensor_key in sensor_keys}
        state.frame_records.append(
            {
                "sample_frame_index": int(sample_index),
                "tactile_frame_index": tactile_frame_index,
                "timestamp": payload.get("timestamp"),
                "telemetry_timestamp": payload.get("telemetry_timestamp"),
                "regions": list(payload.get("regions", ())),
                "frame_ids": dict(payload.get("frame_ids", {}) or {}),
                "shapes": dict(payload.get("shapes", {}) or {}),
                "dtypes": dtypes,
                "field_by_key": dict(payload.get("field_by_key", {}) or {}),
                "fields": list(payload.get("fields", ()) or ()),
                "version": payload.get("version"),
                "snapshot_sequence": payload.get("snapshot_sequence"),
                "snapshot_source_timestamp": payload.get("snapshot_source_timestamp"),
                "snapshot_available_timestamp": payload.get("snapshot_available_timestamp"),
                "snapshot_source_clock": payload.get("snapshot_source_clock"),
                "snapshot_available_clock": payload.get("snapshot_available_clock"),
            }
        )
        return tactile_frame_index, sensor_keys

    def _apply_latest_tactile_ref_to_row(
        self,
        row: dict[str, Any],
        state: _TactileSideState,
    ) -> None:
        latest_frame_index = int(state.next_tactile_frame_index) - 1
        if latest_frame_index < 0:
            return
        prefix = f"observation.hand.{state.side}.tactile"
        row[f"{prefix}_storage_format"] = TACTILE_STREAM_FORMAT
        row[f"{prefix}_path"] = state.manifest_relative_path
        row[f"{prefix}_frame_index"] = latest_frame_index
        if state.sensor_key_order is not None:
            row[f"{prefix}_sensor_keys"] = list(state.sensor_key_order)

    def _drain_tactile_state(self, episode: _EpisodeState, state: _TactileSideState) -> None:
        while state.next_sample_index in state.expected_by_sample:
            sample_index = int(state.next_sample_index)
            expects_frame = bool(state.expected_by_sample[sample_index])
            if not expects_frame:
                del state.expected_by_sample[sample_index]
                state.next_sample_index += 1
                continue
            if sample_index not in state.pending_frames:
                return
            payload = state.pending_frames.pop(sample_index)
            tactile_frame_index, sensor_keys = self._append_tactile_payload(
                state,
                sample_index=sample_index,
                payload=payload,
            )
            row = episode.rows_by_frame.get(sample_index)
            if row is not None:
                prefix = f"observation.hand.{state.side}.tactile"
                row[f"{prefix}_storage_format"] = TACTILE_STREAM_FORMAT
                row[f"{prefix}_path"] = state.manifest_relative_path
                row[f"{prefix}_frame_index"] = int(tactile_frame_index)
                row[f"{prefix}_sensor_keys"] = list(sensor_keys)
            del state.expected_by_sample[sample_index]
            state.next_tactile_frame_index += 1
            state.next_sample_index += 1

    def _append_wrist_depth_payload(
        self,
        state: _WristDepthStreamState,
        *,
        sample_index: int,
        payload: Mapping[str, Any],
    ) -> tuple[int, dict[str, Any]]:
        depth = np.asarray(payload.get("depth_image"))
        if depth.ndim != 2:
            raise ValueError(
                f"Wrist depth frame for camera '{state.camera_key}' must be 2D, got {depth.shape!r}."
            )
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16, copy=False)
        depth = np.ascontiguousarray(depth)
        height, width = (int(depth.shape[0]), int(depth.shape[1]))
        shape = (height, width)
        if state.shape is None:
            state.shape = shape
        elif state.shape != shape:
            raise ValueError(
                f"Wrist depth frame for camera '{state.camera_key}' changed shape "
                f"from {state.shape} to {shape}."
            )
        if state.handle is None:
            state.stream_path.parent.mkdir(parents=True, exist_ok=True)
            state.handle = state.stream_path.open("ab")

        depth_metadata = dict(payload.get("depth_metadata", {}) or {})
        depth_frame_index = int(state.next_depth_frame_index)
        record: dict[str, Any] = {
            "sample_frame_index": int(sample_index),
            "depth_frame_index": depth_frame_index,
            "source_timestamp": payload.get("source_timestamp"),
            "receive_timestamp": payload.get("receive_timestamp"),
            "available_timestamp": payload.get("available_timestamp"),
            "source_frame_index": payload.get("source_frame_index"),
            "depth_camera_timestamp_sec": depth_metadata.get("camera_timestamp_sec"),
            "depth_sensor_timestamp_ms": depth_metadata.get("sensor_timestamp_ms"),
            "depth_frame_number": depth_metadata.get("frame_number"),
            "depth_scale_m_per_unit": depth_metadata.get("depth_scale_m_per_unit"),
            "shape": [height, width],
        }
        if state.stream_kind == "mask":
            depth_scale = self._depth_scale_from_metadata(depth_metadata)
            if depth_scale is None:
                raise ValueError(
                    f"Wrist depth mask for camera '{state.camera_key}' is missing a valid depth scale."
                )
            result = depth_range_mask(
                depth,
                depth_scale_m_per_unit=depth_scale,
                config=self.config.wrist_depth_mask_config,
            )
            assert state.handle is not None
            state.handle.write(pack_binary_mask(result.mask))
            record.update(result.stats_dict())
            state.valid_fraction_sum += float(result.valid_fraction)
            state.foreground_fraction_sum += float(result.foreground_fraction)
        elif state.stream_kind == "raw":
            assert state.handle is not None
            state.handle.write(depth.tobytes(order="C"))
            total = int(depth.size)
            valid_count = int(np.count_nonzero(depth))
            valid_fraction = 0.0 if total <= 0 else float(valid_count) / float(total)
            record["valid_fraction"] = valid_fraction
            state.valid_fraction_sum += float(valid_fraction)
        else:
            raise ValueError(f"Unsupported wrist depth stream kind '{state.stream_kind}'.")

        state.frame_records.append(record)
        return depth_frame_index, record

    def _apply_wrist_depth_ref_to_row(
        self,
        row: dict[str, Any],
        state: _WristDepthStreamState,
        *,
        depth_frame_index: int,
        record: Mapping[str, Any],
    ) -> None:
        prefix = f"{camera_video_channel(state.camera_key)}.wrist_depth_{state.stream_kind}"
        row[f"{prefix}_storage_format"] = WRIST_DEPTH_STREAM_FORMAT
        row[f"{prefix}_path"] = state.manifest_relative_path
        row[f"{prefix}_frame_index"] = int(depth_frame_index)
        row[f"{prefix}_source_timestamp"] = record.get("source_timestamp")
        row[f"{prefix}_receive_timestamp"] = record.get("receive_timestamp")
        row[f"{prefix}_available_timestamp"] = record.get("available_timestamp")
        row[f"{prefix}_source_frame_index"] = record.get("source_frame_index")
        row[f"{prefix}_depth_camera_timestamp_sec"] = record.get("depth_camera_timestamp_sec")
        row[f"{prefix}_depth_sensor_timestamp_ms"] = record.get("depth_sensor_timestamp_ms")
        row[f"{prefix}_depth_frame_number"] = record.get("depth_frame_number")
        row[f"{prefix}_valid_fraction"] = record.get("valid_fraction")
        if state.stream_kind == "mask":
            row[f"{prefix}_foreground_fraction"] = record.get("foreground_fraction")

    def _drain_wrist_depth_state(self, episode: _EpisodeState, state: _WristDepthStreamState) -> None:
        while state.next_sample_index in state.expected_by_sample:
            sample_index = int(state.next_sample_index)
            expects_frame = bool(state.expected_by_sample[sample_index])
            if not expects_frame:
                del state.expected_by_sample[sample_index]
                state.pending_frames.pop(sample_index, None)
                state.next_sample_index += 1
                continue
            if sample_index not in state.pending_frames:
                return
            payload = state.pending_frames.pop(sample_index)
            depth_frame_index, record = self._append_wrist_depth_payload(
                state,
                sample_index=sample_index,
                payload=payload,
            )
            row = episode.rows_by_frame.get(sample_index)
            if row is not None:
                self._apply_wrist_depth_ref_to_row(
                    row,
                    state,
                    depth_frame_index=depth_frame_index,
                    record=record,
                )
            del state.expected_by_sample[sample_index]
            state.next_depth_frame_index += 1
            state.next_sample_index += 1

    def _close_tactile_state(self, state: _TactileSideState) -> str | None:
        for stream in state.streams.values():
            stream.close()
        if not state.frame_records:
            return None
        streams = {
            sensor_key: {
                "path": stream.relative_path,
                "dtype": stream.dtype,
                "shape": [] if stream.shape is None else list(stream.shape),
                "frame_count": int(stream.frame_count),
                "encoding": stream.encoding,
                "compression": stream.compression,
                "chunk_frame_count": (
                    int(stream.chunk_frame_count)
                    if stream.compression != TACTILE_COMPRESSION_NONE
                    else None
                ),
                "chunks": (
                    list(stream.chunks)
                    if stream.compression != TACTILE_COMPRESSION_NONE
                    else []
                ),
            }
            for sensor_key, stream in state.streams.items()
        }
        _write_json(
            state.manifest_path,
            {
                "format": TACTILE_STREAM_FORMAT,
                "side": state.side,
                "relative_dir": state.relative_dir,
                "frame_count": len(state.frame_records),
                "sensor_keys": list(state.sensor_key_order or ()),
                "streams": streams,
                "frames": list(state.frame_records),
            },
        )
        return state.manifest_relative_path

    def _close_wrist_depth_state(self, state: _WristDepthStreamState) -> str | None:
        state.close_handle()
        if not state.frame_records or state.shape is None:
            return None
        height, width = state.shape
        if state.stream_kind == "mask":
            payload = mask_manifest_payload(
                camera_key=state.camera_key,
                relative_dir=state.relative_dir,
                stream_path=state.stream_relative_path,
                frame_count=len(state.frame_records),
                height=height,
                width=width,
                records=state.frame_records,
                config=self.config.wrist_depth_mask_config,
            )
        elif state.stream_kind == "raw":
            payload = raw_depth_manifest_payload(
                camera_key=state.camera_key,
                relative_dir=state.relative_dir,
                stream_path=state.stream_relative_path,
                frame_count=len(state.frame_records),
                height=height,
                width=width,
                records=state.frame_records,
            )
        else:
            raise ValueError(f"Unsupported wrist depth stream kind '{state.stream_kind}'.")
        _write_json(state.manifest_path, payload)
        return state.manifest_relative_path

    def _recompute_next_episode_index_locked(self) -> None:
        max_episode_index = -1
        for row in _read_jsonl(self.episodes_path):
            try:
                max_episode_index = max(max_episode_index, int(row.get("episode_index", -1)))
            except (TypeError, ValueError):
                continue
        if self._episodes:
            max_episode_index = max(max_episode_index, max(self._episodes))
        self._next_episode_index = max_episode_index + 1

    def discard_episode(self, episode_index: int) -> bool:
        with self._lock:
            episode = self._episodes.pop(int(episode_index), None)
            if episode is None:
                self._recompute_next_episode_index_locked()
                return False
            for state in episode.video_states.values():
                self._close_video_state(state)
                try:
                    state.path.unlink(missing_ok=True)
                except OSError:
                    pass
            for state in episode.tactile_states.values():
                self._close_tactile_state(state)
                shutil.rmtree(state.path_dir, ignore_errors=True)
            for state in episode.wrist_depth_states.values():
                state.close_handle()
                shutil.rmtree(state.path_dir, ignore_errors=True)
            for key in tuple(self._planned_tactile_snapshot_refs):
                if key[0] == int(episode_index):
                    self._planned_tactile_snapshot_refs.pop(key, None)
            self._recompute_next_episode_index_locked()
            return True

    def _finalize_episode_locked(
        self,
        episode_index: int,
        *,
        summary: Mapping[str, Any] | None = None,
        collect_artifact_report: bool = False,
    ) -> LegacyEpisodeArtifactReport | None:
        episode = self._ensure_episode_state(int(episode_index))
        for state in episode.video_states.values():
            self._drain_video_state(episode, state)
            unresolved = [
                sample_index
                for sample_index, expects_frame in sorted(state.expected_by_sample.items())
                if expects_frame
            ]
            if unresolved:
                state.missing_frame_count += len(unresolved)
                for sample_index in unresolved:
                    state.expected_by_sample.pop(sample_index, None)
                if not bool(self.config.allow_missing_video_frames):
                    raise RuntimeError(
                        f"Episode {episode.episode_id} camera '{state.camera_key}' is missing "
                        f"selected frames for sample indices {unresolved[:8]}."
                    )
            self._close_video_state(state)
        tactile_paths: dict[str, str] = {}
        for state in episode.tactile_states.values():
            self._drain_tactile_state(episode, state)
            unresolved = [
                sample_index
                for sample_index, expects_frame in sorted(state.expected_by_sample.items())
                if expects_frame
            ]
            if unresolved:
                raise RuntimeError(
                    f"Episode {episode.episode_id} tactile side '{state.side}' is missing "
                    f"frames for sample indices {unresolved[:8]}."
                )
            manifest_path = self._close_tactile_state(state)
            if manifest_path is not None:
                tactile_paths[state.side] = manifest_path
        wrist_depth_mask_paths: dict[str, str] = {}
        wrist_depth_raw_paths: dict[str, str] = {}
        wrist_depth_frame_stats: dict[str, dict[str, Any]] = {}
        for state in episode.wrist_depth_states.values():
            self._drain_wrist_depth_state(episode, state)
            unresolved = [
                sample_index
                for sample_index, expects_frame in sorted(state.expected_by_sample.items())
                if expects_frame
            ]
            if unresolved:
                state.missing_frame_count += len(unresolved)
                for sample_index in unresolved:
                    state.expected_by_sample.pop(sample_index, None)
            manifest_path = self._close_wrist_depth_state(state)
            if manifest_path is not None:
                if state.stream_kind == "mask":
                    wrist_depth_mask_paths[state.camera_key] = manifest_path
                elif state.stream_kind == "raw":
                    wrist_depth_raw_paths[state.camera_key] = manifest_path
            key = f"{state.camera_key}.{state.stream_kind}"
            frame_count = int(state.next_depth_frame_index)
            wrist_depth_frame_stats[key] = {
                "expected": int(frame_count + state.missing_frame_count),
                "written": frame_count,
                "missing": int(state.missing_frame_count),
                "path": manifest_path,
                "valid_fraction_mean": (
                    0.0 if frame_count <= 0 else float(state.valid_fraction_sum) / float(frame_count)
                ),
            }
            if state.stream_kind == "mask":
                wrist_depth_frame_stats[key]["foreground_fraction_mean"] = (
                    0.0 if frame_count <= 0 else float(state.foreground_fraction_sum) / float(frame_count)
                )

        artifact_report = None
        if collect_artifact_report:
            video_artifacts: list[LegacyFinalizedVideoArtifact] = []
            for camera_key, state in sorted(episode.video_states.items()):
                if state.next_video_frame_index <= 0:
                    continue
                if not state.path.is_file():
                    raise RuntimeError(
                        "Finalized video artifact is missing for "
                        f"camera {camera_key!r}."
                    )
                _fsync_file_and_parent(state.path)
                stat = state.path.stat()
                video_artifacts.append(
                    LegacyFinalizedVideoArtifact(
                        camera_key=camera_key,
                        artifact_relative_path=state.relative_path,
                        artifact_sha256=_file_sha256(state.path),
                        artifact_byte_count=int(stat.st_size),
                        encoding=f"mp4/{self.config.video_codec}",
                        compression="none",
                        video_frame_index_by_sample_frame_index=tuple(
                            sorted(
                                state.video_frame_index_by_sample_frame_index.items()
                            )
                        ),
                    )
                )
            artifact_report = LegacyEpisodeArtifactReport(
                episode_index=int(episode.episode_index),
                video_artifacts=tuple(video_artifacts),
            )

        rows = [episode.rows_by_frame[index] for index in sorted(episode.rows_by_frame)]
        if not rows:
            raise RuntimeError(f"Episode {episode.episode_id} does not contain any rows.")
        for offset, row in enumerate(rows):
            row["index"] = int(self._finalized_frame_count + offset)
        parquet_relative_path = self._episode_parquet_relative_path(episode.episode_index)
        if self.config.write_parquet:
            write_rows_to_parquet(
                rows,
                self.root_dir / parquet_relative_path,
                compression=self.config.parquet_compression,
                fixed_size_list_columns=(
                    None
                    if self.policy_io_layout is None
                    else self.policy_io_layout.parquet_fixed_list_columns()
                ),
            )

        assert episode.stats is not None
        stats = episode.stats.finalize(nominal_fps=self.config.nominal_fps)
        video_paths = {
            camera_key: state.relative_path
            for camera_key, state in episode.video_states.items()
            if state.next_video_frame_index > 0
        }
        summary_payload = {} if summary is None else dict(_to_serializable(dict(summary)))
        storage_queue_stats = summary_payload.get("storage_queue_stats")
        storage_routes = (
            storage_queue_stats.get("routes", {})
            if isinstance(storage_queue_stats, Mapping)
            else {}
        )
        if wrist_depth_frame_stats:
            for stat_key, stat_payload in wrist_depth_frame_stats.items():
                camera_key = stat_key.rsplit(".", 1)[0]
                route_stats = storage_routes.get(f"wrist_depth:{camera_key}", {})
                stat_payload["dropped"] = (
                    int(route_stats.get("dropped", 0) or 0)
                    if isinstance(route_stats, Mapping)
                    else 0
                )
        video_frame_stats = {}
        for camera_key, state in episode.video_states.items():
            route_stats = storage_routes.get(camera_key, {})
            if not isinstance(route_stats, Mapping):
                route_stats = storage_routes.get(f"video:{camera_key}", {})
            dropped = (
                int(route_stats.get("dropped", 0) or 0)
                if isinstance(route_stats, Mapping)
                else 0
            )
            video_frame_stats[camera_key] = {
                "expected": int(state.expected_frame_count),
                "written": int(state.written_frame_count),
                "dropped": dropped,
                "missing": int(state.missing_frame_count),
                "path": state.relative_path if state.next_video_frame_index > 0 else None,
            }
        if video_frame_stats:
            summary_payload.setdefault("video_frame_stats", video_frame_stats)
        if wrist_depth_frame_stats:
            summary_payload.setdefault("wrist_depth_frame_stats", wrist_depth_frame_stats)
        wrist_depth_integrity = self._wrist_depth_integrity_payload(episode)
        if wrist_depth_integrity is not None:
            summary_payload["wrist_depth_integrity"] = wrist_depth_integrity
            episode.metadata["wrist_depth_integrity"] = wrist_depth_integrity
        force_collision_detected = bool(
            stats.force_collision_detected
            or summary_has_force_collision(summary_payload)
        )
        force_collision_label = (
            FORCE_COLLISION_LABEL_DETECTED
            if force_collision_detected
            else FORCE_COLLISION_LABEL_CLEAN
        )
        trajectory_quality = {
            "pose_quality_status_counts": dict(stats.pose_quality_status_counts),
            "pose_quality_bad_count": int(stats.pose_quality_bad_count),
            "pose_quality_warning_count": int(stats.pose_quality_warning_count),
            "pose_quality_special_count": int(stats.pose_quality_special_count),
            "pose_quality_singularity_count": int(stats.pose_quality_singularity_count),
            "pose_quality_valid_score_count": int(stats.pose_quality_valid_score_count),
            "pose_quality_min_score": stats.pose_quality_min_score,
            "pose_quality_mean_score": stats.pose_quality_mean_score,
            "force_collision_detected": force_collision_detected,
            "force_collision_label": force_collision_label,
        }
        summary_trajectory_quality = summary_payload.get("trajectory_quality")
        if isinstance(summary_trajectory_quality, dict):
            summary_trajectory_quality.update(trajectory_quality)
        else:
            summary_trajectory_quality = dict(trajectory_quality)
        summary_payload["trajectory_quality"] = summary_trajectory_quality
        summary_payload["force_collision_detected"] = force_collision_detected
        summary_payload["force_collision_label"] = force_collision_label
        episode_trajectory_quality = episode.metadata.get("trajectory_quality")
        if isinstance(episode_trajectory_quality, dict):
            episode_trajectory_quality.update(trajectory_quality)
        else:
            episode_trajectory_quality = dict(trajectory_quality)
        episode.metadata["trajectory_quality"] = episode_trajectory_quality
        episode.metadata["force_collision_detected"] = force_collision_detected
        episode.metadata["force_collision_label"] = force_collision_label
        episode_record = {
            "episode_index": int(episode.episode_index),
            "episode_id": episode.episode_id,
            "frame_count": int(stats.frame_count),
            "start_timestamp": stats.start_timestamp,
            "end_timestamp": stats.end_timestamp,
            "duration_sec": float(stats.duration_sec),
            "parquet_path": parquet_relative_path if self.config.write_parquet else None,
            "video_paths": video_paths,
            "tactile_paths": tactile_paths,
            "wrist_depth_mask_paths": wrist_depth_mask_paths,
            "wrist_depth_raw_paths": wrist_depth_raw_paths,
            "metadata": dict(episode.metadata),
            "summary": summary_payload,
        }
        episode_stats_record = {
            "episode_index": int(stats.episode_index),
            "frame_count": int(stats.frame_count),
            "start_timestamp": stats.start_timestamp,
            "end_timestamp": stats.end_timestamp,
            "duration_sec": float(stats.duration_sec),
            "average_dt_sec": stats.average_dt_sec,
            "average_hz": stats.average_hz,
            "mean_control_hz": stats.mean_control_hz,
            "nominal_fps": float(stats.nominal_fps),
            "active_sides": list(stats.active_sides),
            "arm_state_source": stats.arm_state_source,
            "camera_keys": list(stats.camera_keys),
            "fault_count": int(stats.fault_count),
            "pose_quality_status_counts": dict(stats.pose_quality_status_counts),
            "pose_quality_bad_count": int(stats.pose_quality_bad_count),
            "pose_quality_warning_count": int(stats.pose_quality_warning_count),
            "pose_quality_special_count": int(stats.pose_quality_special_count),
            "pose_quality_singularity_count": int(stats.pose_quality_singularity_count),
            "pose_quality_valid_score_count": int(stats.pose_quality_valid_score_count),
            "pose_quality_min_score": stats.pose_quality_min_score,
            "pose_quality_mean_score": stats.pose_quality_mean_score,
            "force_collision_detected": force_collision_detected,
            "force_collision_label": force_collision_label,
        }
        _append_jsonl(self.episodes_path, episode_record)
        _append_jsonl(self.episode_stats_path, episode_stats_record)
        self._finalized_episode_count += 1
        self._finalized_frame_count += int(stats.frame_count)
        self._write_info_locked()
        for key in tuple(self._planned_tactile_snapshot_refs):
            if key[0] == int(episode_index):
                self._planned_tactile_snapshot_refs.pop(key, None)
        del self._episodes[episode_index]
        return artifact_report

    def finalize_episode(
        self,
        episode_index: int,
        *,
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._finalize_episode_locked(episode_index, summary=summary)

    def finalize_episode_with_artifact_report(
        self,
        episode_index: int,
        *,
        summary: Mapping[str, Any] | None = None,
    ) -> LegacyEpisodeArtifactReport:
        """Finalize normally and return durable specialized-artifact facts."""

        with self._lock:
            report = self._finalize_episode_locked(
                episode_index,
                summary=summary,
                collect_artifact_report=True,
            )
        assert report is not None
        return report
