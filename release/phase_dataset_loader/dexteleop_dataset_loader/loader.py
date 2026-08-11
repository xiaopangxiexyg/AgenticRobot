"""Portable read-only access to the released 0804 DexTeleop dataset."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .canonical import CanonicalDocument, load_canonical_file, load_canonical_lines


EXPECTED_DATASET_SCHEMA = "dexteleop_dataset_v2.3"
EXPECTED_MODALITY_SCHEMA = "dexteleop.modality_validity.v1"

PROGRESS_PROFILE_DOMAIN = "dexteleop.progress_profile.file.v1"
PROGRESS_BINDING_DOMAIN = "dexteleop.progress_binding.file.v1"
PROGRESS_FACT_DOMAIN = "dexteleop.progress_fact.file.v1"
PROGRESS_MANIFEST_DOMAIN = "dexteleop.progress_manifest.file.v1"
STAGE_PROFILE_DOMAIN = "dexteleop.task_stage_profile.file.v1"
STAGE_ENTRY_DOMAIN = "dexteleop.progress_stage_entry_binding.file.v1"
STAGE_ANNOTATION_BINDING_DOMAIN = "dexteleop.stage_annotation_binding.file.v1"
STAGE_ANNOTATION_BUNDLE_DOMAIN = "dexteleop.stage_annotation_bundle.file.v1"
STAGE_ANNOTATION_MANIFEST_DOMAIN = "dexteleop.stage_annotation_manifest.file.v1"
FAILURE_TRIGGER_DOMAIN = "dexteleop.failure_trigger_annotation.file.v1"

DEFAULT_MODEL_COLUMNS = (
    "episode_index",
    "frame_index",
    "timestamp",
    "action",
    "observation.state",
    "observation.last_action",
    "observation.arm.qpos",
)


class DatasetFormatError(ValueError):
    """Raised when the released directory is absent, unsafe, or inconsistent."""


class SelectionRequiredError(DatasetFormatError):
    """Raised rather than silently choosing one of multiple annotation sets."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetFormatError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetFormatError(f"expected JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetFormatError(f"cannot read JSON lines {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetFormatError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise DatasetFormatError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return tuple(rows)


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DatasetFormatError(f"{label} must be an object")
    return value


@dataclass(frozen=True)
class EpisodeRecord:
    dataset_root: Path
    payload: Mapping[str, Any]

    @property
    def episode_index(self) -> int:
        return int(self.payload["episode_index"])

    @property
    def episode_id(self) -> str:
        return str(self.payload.get("episode_id") or f"episode_{self.episode_index:06d}")

    @property
    def frame_count(self) -> int:
        return int(self.payload.get("frame_count", 0) or 0)

    @property
    def progress_reference(self) -> Mapping[str, Any] | None:
        metadata = self.payload.get("metadata")
        if not isinstance(metadata, Mapping):
            return None
        value = metadata.get("progress_episode")
        return value if isinstance(value, Mapping) else None


@dataclass(frozen=True)
class StageAnnotationSet:
    annotation_set_id: str
    annotation_dir: Path
    binding: Mapping[str, Any]
    annotations: Mapping[str, Any]
    manifest: Mapping[str, Any]
    failure_triggers: Mapping[str, Any] | None


class DexTeleopReleaseDataset:
    """Read one complete dataset directory without modifying it.

    Paths embedded by Episode metadata must be relative and must resolve under
    ``dataset_root``. This keeps a copied release portable and rejects path
    traversal or a symlink that escapes the release root.
    """

    def __init__(self, dataset_root: str | Path):
        root = Path(dataset_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise DatasetFormatError(f"dataset root is not a directory: {root}")
        self.root = root
        self.info = _read_json(root / "meta" / "info.json")
        schema = self.info.get("dexteleop_dataset_schema")
        if schema != EXPECTED_DATASET_SCHEMA:
            raise DatasetFormatError(
                f"expected {EXPECTED_DATASET_SCHEMA!r}, found {schema!r}"
            )
        validity_path = root / "meta" / "modality_validity.json"
        self.modality_validity = _read_json(validity_path) if validity_path.is_file() else {}
        validity_schema = self.modality_validity.get("schema_version")
        if validity_schema not in {None, EXPECTED_MODALITY_SCHEMA}:
            raise DatasetFormatError(f"unsupported modality validity schema {validity_schema!r}")
        rows = _read_jsonl(root / "meta" / "episodes.jsonl")
        episodes = tuple(EpisodeRecord(root, row) for row in rows)
        self._episodes = {episode.episode_index: episode for episode in episodes}
        if len(self._episodes) != len(episodes):
            raise DatasetFormatError("duplicate episode_index in meta/episodes.jsonl")
        declared_count = int(self.info.get("episode_count", len(episodes)))
        if declared_count != len(episodes):
            raise DatasetFormatError(
                f"episode count mismatch: info={declared_count} records={len(episodes)}"
            )

    @property
    def episode_indices(self) -> tuple[int, ...]:
        return tuple(sorted(self._episodes))

    @property
    def camera_keys(self) -> tuple[str, ...]:
        keys: set[str] = set()
        for episode in self._episodes.values():
            paths = episode.payload.get("video_paths")
            if isinstance(paths, Mapping):
                keys.update(str(key) for key, value in paths.items() if value)
        return tuple(sorted(keys))

    def episode(self, episode_index: int) -> EpisodeRecord:
        try:
            return self._episodes[int(episode_index)]
        except KeyError as exc:
            raise KeyError(f"unknown episode_index {episode_index}") from exc

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_name": self.info.get("dataset_name", self.root.name),
            "dataset_schema": self.info.get("dexteleop_dataset_schema"),
            "episode_count": len(self._episodes),
            "frame_count": int(self.info.get("frame_count", 0) or 0),
            "nominal_fps": float(self.info.get("nominal_fps", 0.0) or 0.0),
            "camera_keys": self.camera_keys,
            "policy_io_layout_hash": self.info.get("policy_io_layout_hash"),
            "pose_frame_contract_hash": self.info.get("pose_frame_contract_hash"),
            "causal_snapshot_contract_hash": self.info.get("causal_snapshot_contract_hash"),
        }

    def _resolve_relative(self, relative: Any, *, label: str) -> Path:
        if not isinstance(relative, str) or not relative.strip():
            raise DatasetFormatError(f"{label} must be a non-empty relative path")
        value = Path(relative)
        if value.is_absolute():
            raise DatasetFormatError(f"{label} must be relative to the dataset root")
        try:
            candidate = (self.root / value).resolve(strict=True)
            candidate.relative_to(self.root)
        except (OSError, ValueError) as exc:
            raise DatasetFormatError(f"{label} escapes or is missing from dataset root") from exc
        return candidate

    def _episode_path(self, episode_index: int, key: str, *, label: str) -> Path:
        value = self.episode(episode_index).payload.get(key)
        return self._resolve_relative(value, label=label)

    def parquet_path(self, episode_index: int) -> Path:
        return self._episode_path(episode_index, "parquet_path", label="Parquet path")

    def video_path(self, episode_index: int, camera_key: str) -> Path:
        episode = self.episode(episode_index)
        paths = _require_mapping(episode.payload.get("video_paths"), label="video_paths")
        try:
            relative = paths[str(camera_key)]
        except KeyError as exc:
            raise KeyError(
                f"episode {episode_index} has no camera {camera_key!r}"
            ) from exc
        return self._resolve_relative(relative, label=f"video {camera_key}")

    def _manifest_path(self, episode_index: int, key: str, slot: str) -> Path:
        episode = self.episode(episode_index)
        paths = _require_mapping(episode.payload.get(key), label=key)
        try:
            relative = paths[str(slot)]
        except KeyError as exc:
            raise KeyError(f"episode {episode_index} has no {key} slot {slot!r}") from exc
        return self._resolve_relative(relative, label=f"{key}.{slot}")

    def load_parquet(
        self,
        episode_index: int,
        *,
        columns: Sequence[str] | None = DEFAULT_MODEL_COLUMNS,
    ):
        """Return a ``pyarrow.Table`` for one Episode.

        Pass ``columns=None`` only when all audit columns are genuinely needed.
        """

        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Parquet loading requires pyarrow") from exc
        selected = None if columns is None else [str(item) for item in columns]
        try:
            table = pq.read_table(self.parquet_path(episode_index), columns=selected)
        except Exception as exc:
            raise DatasetFormatError(f"cannot read Episode {episode_index} Parquet: {exc}") from exc
        expected = self.episode(episode_index).frame_count
        if table.num_rows != expected:
            raise DatasetFormatError(
                f"Episode {episode_index} row mismatch: metadata={expected} parquet={table.num_rows}"
            )
        return table

    def _parquet_scalar(self, episode_index: int, column: str, frame_index: int) -> Any:
        frame = int(frame_index)
        episode = self.episode(episode_index)
        if frame < 0 or frame >= episode.frame_count:
            raise IndexError(
                f"frame_index {frame} outside Episode {episode_index} [0,{episode.frame_count})"
            )
        table = self.load_parquet(episode_index, columns=(column,))
        return table.column(column)[frame].as_py()

    def load_rgb_frame(
        self,
        episode_index: int,
        camera_key: str,
        frame_index: int,
        *,
        output: str = "rgb",
    ) -> np.ndarray:
        """Load the RGB frame referenced by one formal sample.

        ``output`` is ``"rgb"`` by default or ``"bgr"`` for OpenCV-native
        channel order. The Parquet video-frame reference is authoritative.
        """

        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("RGB loading requires opencv-python-headless") from exc
        if output not in {"rgb", "bgr"}:
            raise ValueError("output must be 'rgb' or 'bgr'")
        column = f"observation.images.{camera_key}.video_frame_index"
        video_frame_index = self._parquet_scalar(episode_index, column, frame_index)
        if video_frame_index is None or int(video_frame_index) < 0:
            raise DatasetFormatError(
                f"Episode {episode_index} sample {frame_index} has no valid {camera_key} frame"
            )
        path = self.video_path(episode_index, camera_key)
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise DatasetFormatError(f"cannot open video {path}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(video_frame_index))
            ok, image = capture.read()
        finally:
            capture.release()
        if not ok or image is None:
            raise DatasetFormatError(
                f"cannot decode {camera_key} video frame {video_frame_index} from {path}"
            )
        if output == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.asarray(image)

    @staticmethod
    def _manifest_frame(
        manifest: Mapping[str, Any],
        sample_frame_index: int,
    ) -> Mapping[str, Any] | None:
        frames = manifest.get("frames")
        if not isinstance(frames, list):
            raise DatasetFormatError("manifest frames must be a list")
        matches = [
            item
            for item in frames
            if isinstance(item, Mapping)
            and int(item.get("sample_frame_index", -1)) == int(sample_frame_index)
        ]
        if len(matches) > 1:
            raise DatasetFormatError(
                f"manifest has duplicate sample_frame_index {sample_frame_index}"
            )
        return matches[0] if matches else None

    def load_tactile_frame(
        self,
        episode_index: int,
        side: str,
        frame_index: int,
    ) -> dict[str, np.ndarray] | None:
        """Load all tactile sensor arrays mapped to one formal sample.

        Returns ``None`` when that sample has no tactile frame. It never repeats
        a previous frame to conceal missing data.
        """

        path = self._manifest_path(episode_index, "tactile_paths", side)
        manifest = _read_json(path)
        if manifest.get("format") != "binary_stream_v1":
            raise DatasetFormatError(f"unsupported tactile format in {path}")
        frame = self._manifest_frame(manifest, frame_index)
        if frame is None:
            return None
        tactile_index = int(frame.get("tactile_frame_index", -1))
        streams = _require_mapping(manifest.get("streams"), label="tactile streams")
        declared_keys = manifest.get("sensor_keys")
        if not isinstance(declared_keys, list):
            raise DatasetFormatError("tactile sensor_keys must be a list")
        output: dict[str, np.ndarray] = {}
        for raw_key in declared_keys:
            sensor_key = str(raw_key)
            spec = _require_mapping(streams.get(sensor_key), label=f"tactile stream {sensor_key}")
            if spec.get("encoding") != "raw_c_order" or spec.get("compression") != "none":
                raise DatasetFormatError(
                    f"unsupported tactile encoding/compression for {sensor_key}"
                )
            dtype = np.dtype(str(spec.get("dtype")))
            shape_value = spec.get("shape")
            if not isinstance(shape_value, list):
                raise DatasetFormatError(f"tactile shape missing for {sensor_key}")
            shape = tuple(int(size) for size in shape_value)
            stream_count = int(spec.get("frame_count", -1))
            if tactile_index < 0 or tactile_index >= stream_count:
                raise DatasetFormatError(
                    f"tactile frame {tactile_index} outside stream {sensor_key}"
                )
            stream_path = self._resolve_relative(
                spec.get("path"), label=f"tactile stream {sensor_key} path"
            )
            frame_values = int(np.prod(shape, dtype=np.int64))
            offset = tactile_index * frame_values * dtype.itemsize
            array: np.ndarray = np.memmap(
                stream_path,
                dtype=dtype,
                mode="r",
                offset=offset,
                shape=shape,
                order="C",
            )
            output[sensor_key] = np.asarray(array).copy()
        return output

    def load_wrist_depth_mask(
        self,
        episode_index: int,
        camera_key: str,
        frame_index: int,
    ) -> np.ndarray | None:
        """Return one ``bool[H,W]`` depth-range mask or ``None`` if absent."""

        path = self._manifest_path(
            episode_index,
            "wrist_depth_mask_paths",
            camera_key,
        )
        manifest = _read_json(path)
        if manifest.get("format") != "dexteleop_wrist_depth_mask_bitpack_v1":
            raise DatasetFormatError(f"unsupported wrist depth mask format in {path}")
        frame = self._manifest_frame(manifest, frame_index)
        if frame is None:
            return None
        depth_index = int(frame.get("depth_frame_index", -1))
        stream = _require_mapping(manifest.get("stream"), label="depth mask stream")
        shape_value = stream.get("shape")
        if not isinstance(shape_value, list) or len(shape_value) != 2:
            raise DatasetFormatError("depth mask shape must be [height,width]")
        shape = tuple(int(size) for size in shape_value)
        bytes_per_frame = int(stream.get("packed_bytes_per_frame", 0) or 0)
        stream_count = int(stream.get("frame_count", -1))
        expected_bytes = (int(np.prod(shape, dtype=np.int64)) + 7) // 8
        if bytes_per_frame != expected_bytes:
            raise DatasetFormatError("depth mask packed byte count does not match shape")
        if depth_index < 0 or depth_index >= stream_count:
            raise DatasetFormatError(f"depth mask frame {depth_index} outside stream")
        stream_path = self._resolve_relative(stream.get("path"), label="depth mask stream path")
        with stream_path.open("rb") as handle:
            handle.seek(depth_index * bytes_per_frame)
            packed = handle.read(bytes_per_frame)
        if len(packed) != bytes_per_frame:
            raise DatasetFormatError("depth mask stream ends before requested frame")
        bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="big")
        return bits[: int(np.prod(shape))].reshape(shape).astype(bool, copy=False)

    def load_progress_profile(self) -> CanonicalDocument:
        return load_canonical_file(
            self.root / "meta" / "progress" / "profile.cjson",
            expected_domain=PROGRESS_PROFILE_DOMAIN,
        )

    def load_stage_profile(self) -> CanonicalDocument:
        return load_canonical_file(
            self.root / "meta" / "progress" / "stages.cjson",
            expected_domain=STAGE_PROFILE_DOMAIN,
        )

    def _progress_episode_dir(self, episode_index: int) -> Path:
        reference = self.episode(episode_index).progress_reference
        if reference is None:
            raise DatasetFormatError(f"Episode {episode_index} has no progress reference")
        episode_uuid = reference.get("episode_uuid")
        if not isinstance(episode_uuid, str) or not episode_uuid:
            raise DatasetFormatError(f"Episode {episode_index} has invalid progress UUID")
        root = self.root / "meta" / "progress" / "episodes"
        return self._resolve_relative(
            (root.relative_to(self.root) / episode_uuid).as_posix(),
            label="progress Episode directory",
        )

    def load_progress_episode(self, episode_index: int) -> dict[str, Any]:
        """Decode one sealed Progress Episode and its facts.

        This portable reader checks required files and canonical domains. The
        in-repository p2 validator remains authoritative for semantic hashes and
        full timeline validation.
        """

        root = self._progress_episode_dir(episode_index)
        required = {
            "profile": ("profile.cjson", PROGRESS_PROFILE_DOMAIN),
            "binding": ("binding.cjson", PROGRESS_BINDING_DOMAIN),
            "manifest": ("manifest.cjson", PROGRESS_MANIFEST_DOMAIN),
        }
        result: dict[str, Any] = {}
        for key, (name, domain) in required.items():
            result[key] = load_canonical_file(root / name, expected_domain=domain).payload
        result["facts"] = tuple(
            item.payload
            for item in load_canonical_lines(
                root / "facts.cjsonl",
                expected_domain=PROGRESS_FACT_DOMAIN,
            )
        )
        optional = {
            "stage_profile": ("stages.cjson", STAGE_PROFILE_DOMAIN),
            "stage_entry": ("stage_entry.cjson", STAGE_ENTRY_DOMAIN),
        }
        for key, (name, domain) in optional.items():
            path = root / name
            result[key] = (
                load_canonical_file(path, expected_domain=domain).payload
                if path.is_file()
                else None
            )
        result["episode_dir"] = root
        return result

    def stage_annotation_set_ids(self, episode_index: int) -> tuple[str, ...]:
        reference = self.episode(episode_index).progress_reference
        if reference is None:
            return ()
        episode_uuid = str(reference.get("episode_uuid") or "")
        if not episode_uuid:
            return ()
        relative = Path("meta/progress/stage_annotations") / episode_uuid
        root = self.root / relative
        if not root.is_dir():
            return ()
        ids = [
            path.name
            for path in root.iterdir()
            if path.is_dir()
            and (path / "binding.cjson").is_file()
            and (path / "annotations.cjson").is_file()
            and (path / "manifest.cjson").is_file()
        ]
        return tuple(sorted(ids))

    def load_stage_annotation(
        self,
        episode_index: int,
        *,
        annotation_set_id: str | None = None,
    ) -> StageAnnotationSet:
        ids = self.stage_annotation_set_ids(episode_index)
        selected = annotation_set_id
        if selected is None:
            if not ids:
                raise DatasetFormatError(f"Episode {episode_index} has no sealed Stage annotation")
            if len(ids) != 1:
                raise SelectionRequiredError(
                    f"Episode {episode_index} has multiple Stage annotation sets; "
                    f"choose one explicitly from {ids}"
                )
            selected = ids[0]
        if selected not in ids:
            raise DatasetFormatError(
                f"unknown or incomplete Stage annotation set {selected!r}; available={ids}"
            )
        reference = self.episode(episode_index).progress_reference
        assert reference is not None
        episode_uuid = str(reference["episode_uuid"])
        relative = Path("meta/progress/stage_annotations") / episode_uuid / selected
        root = self._resolve_relative(relative.as_posix(), label="Stage annotation directory")
        binding = load_canonical_file(
            root / "binding.cjson",
            expected_domain=STAGE_ANNOTATION_BINDING_DOMAIN,
        ).payload
        annotations = load_canonical_file(
            root / "annotations.cjson",
            expected_domain=STAGE_ANNOTATION_BUNDLE_DOMAIN,
        ).payload
        manifest = load_canonical_file(
            root / "manifest.cjson",
            expected_domain=STAGE_ANNOTATION_MANIFEST_DOMAIN,
        ).payload
        for payload, label in (
            (binding, "Stage annotation binding"),
            (annotations, "Stage annotations"),
            (manifest, "Stage annotation manifest"),
        ):
            if not isinstance(payload, Mapping):
                raise DatasetFormatError(f"{label} payload must be an object")
        if binding.get("annotation_set_id") != selected:
            raise DatasetFormatError("Stage annotation directory and binding ID disagree")
        if manifest.get("annotation_set_binding_hash") != binding.get("semantic_content_hash"):
            raise DatasetFormatError("Stage annotation manifest does not bind binding hash")
        if manifest.get("annotation_bundle_hash") != annotations.get("semantic_content_hash"):
            raise DatasetFormatError("Stage annotation manifest does not bind annotation hash")
        failure_path = root / "failure_triggers.cjson"
        failure_triggers = (
            load_canonical_file(
                failure_path,
                expected_domain=FAILURE_TRIGGER_DOMAIN,
            ).payload
            if failure_path.is_file()
            else None
        )
        return StageAnnotationSet(
            annotation_set_id=selected,
            annotation_dir=root,
            binding=binding,
            annotations=annotations,
            manifest=manifest,
            failure_triggers=failure_triggers,
        )


__all__ = [
    "DEFAULT_MODEL_COLUMNS",
    "DatasetFormatError",
    "DexTeleopReleaseDataset",
    "EpisodeRecord",
    "SelectionRequiredError",
    "StageAnnotationSet",
]
