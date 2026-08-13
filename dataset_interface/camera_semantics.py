"""Read-only dataset camera semantic resolution from RealSense serials."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.camera_semantics import (
    CAMERA_SEMANTIC_MATERIALIZATION_HASH_DOMAIN_V1,
    CAMERA_SEMANTIC_MATERIALIZATION_SCHEMA_V1,
    CameraRoleBinding,
    CameraSemanticContractError,
    CameraSemanticResolution,
    ResolvedCameraRole,
)
from dataset_contracts.canonical_encoding import canonical_content_hash


REALSENSE_SNAPSHOT_HASH_DOMAIN_V1 = (
    "dexteleop.realsense_device_snapshot.v1"
)
CAMERA_ROLE_REGISTRY_ENV = "DEXTELEOP_CAMERA_ROLE_REGISTRY"
CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH = (
    Path("meta") / "camera_semantics" / "materialization.json"
)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CameraSemanticContractError(
            f"RealSense device snapshot is missing: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CameraSemanticContractError(
            f"Cannot read RealSense device snapshot {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CameraSemanticContractError(
            f"RealSense device snapshot must be an object: {path}"
        )
    return payload


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        return ()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        text = line.strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CameraSemanticContractError(
                f"Invalid provenance JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise CameraSemanticContractError(
                f"Provenance row must be an object: {path}:{line_number}"
            )
        rows.append(value)
    return tuple(rows)


def camera_role_binding_from_registry(
    registry: Mapping[str, Any],
    *,
    canonical_roles: tuple[str, ...] = ("left_wrist", "right_wrist"),
) -> CameraRoleBinding:
    raw_cameras = registry.get("cameras")
    if not isinstance(raw_cameras, Mapping):
        raise CameraSemanticContractError(
            "Camera role registry must contain a cameras object."
        )
    roles: dict[str, dict[str, str]] = {}
    for role in canonical_roles:
        entry = raw_cameras.get(role)
        if not isinstance(entry, Mapping):
            raise CameraSemanticContractError(
                f"Camera role registry has no {role!r} entry."
            )
        serial = str(entry.get("serial") or "").strip()
        if not serial:
            raise CameraSemanticContractError(
                f"Camera role registry entry {role!r} has no serial."
            )
        roles[role] = {"serial": serial}
    return CameraRoleBinding.from_dict(
        {
            "schema": "dexteleop.camera_role_binding.v1",
            "roles": roles,
        }
    )


def load_camera_role_binding(
    registry_path: str | Path,
    *,
    canonical_roles: tuple[str, ...] = ("left_wrist", "right_wrist"),
) -> CameraRoleBinding:
    return camera_role_binding_from_registry(
        _read_json_object(Path(registry_path).expanduser().resolve()),
        canonical_roles=canonical_roles,
    )


def camera_role_registry_from_environment() -> Path | None:
    value = str(os.environ.get(CAMERA_ROLE_REGISTRY_ENV) or "").strip()
    return None if not value else Path(value).expanduser().resolve()


def _camera_entries(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    cameras = snapshot.get("cameras")
    return cameras if isinstance(cameras, Mapping) else snapshot


def _serial_evidence(entry: Mapping[str, Any], *, stored_key: str) -> str:
    runtime_device = entry.get("runtime_device")
    nested = runtime_device if isinstance(runtime_device, Mapping) else {}
    observed_values = {
        str(value).strip()
        for value in (
            entry.get("runtime_serial"),
            entry.get("serial_number"),
            nested.get("serial_number"),
        )
        if value not in (None, "") and str(value).strip()
    }
    if len(observed_values) > 1:
        raise CameraSemanticContractError(
            f"Camera {stored_key!r} has conflicting observed serials: "
            f"{sorted(observed_values)}."
        )
    configured_values = {
        str(value).strip()
        for value in (
            entry.get("serial"),
            entry.get("requested_serial"),
            nested.get("serial"),
            nested.get("requested_serial"),
        )
        if value not in (None, "") and str(value).strip()
    }
    if len(configured_values) > 1:
        raise CameraSemanticContractError(
            f"Camera {stored_key!r} has conflicting configured serials: "
            f"{sorted(configured_values)}."
        )
    observed = next(iter(observed_values), None)
    configured = next(iter(configured_values), None)
    if observed is not None and configured is not None and observed != configured:
        raise CameraSemanticContractError(
            f"Camera {stored_key!r} configured serial {configured!r} differs "
            f"from observed serial {observed!r}."
        )
    serial = observed or configured
    if serial is None:
        raise CameraSemanticContractError(
            f"Camera {stored_key!r} has no configured or observed serial."
        )
    return serial


def resolve_camera_snapshot(
    snapshot: Mapping[str, Any],
    binding: CameraRoleBinding,
    *,
    source_id: str,
) -> CameraSemanticResolution:
    entries = _camera_entries(snapshot)
    stored_by_serial: dict[str, str] = {}
    for raw_key, raw_entry in entries.items():
        if not isinstance(raw_entry, Mapping):
            continue
        stored_key = str(raw_key).strip()
        if not stored_key:
            continue
        try:
            serial = _serial_evidence(raw_entry, stored_key=stored_key)
        except CameraSemanticContractError:
            if stored_key in binding.canonical_roles:
                raise
            continue
        if serial in stored_by_serial:
            raise CameraSemanticContractError(
                f"RealSense serial {serial!r} appears under both "
                f"{stored_by_serial[serial]!r} and {stored_key!r}."
            )
        stored_by_serial[serial] = stored_key

    resolved: list[ResolvedCameraRole] = []
    for role in binding.roles:
        stored_key = stored_by_serial.get(role.serial)
        if stored_key is None:
            raise CameraSemanticContractError(
                f"Source {source_id!r} does not contain serial {role.serial!r} "
                f"required for canonical role {role.canonical_role!r}."
            )
        resolved.append(
            ResolvedCameraRole(
                canonical_role=role.canonical_role,
                stored_key=stored_key,
                serial=role.serial,
            )
        )

    identity = all(
        item.canonical_role == item.stored_key for item in resolved
    )
    exact_wrist_swap = (
        {item.canonical_role for item in resolved}
        == {"left_wrist", "right_wrist"}
        and {
            item.canonical_role: item.stored_key for item in resolved
        }
        == {
            "left_wrist": "right_wrist",
            "right_wrist": "left_wrist",
        }
    )
    status = "identity" if identity else "swapped" if exact_wrist_swap else "remapped"
    snapshot_hash = canonical_content_hash(
        dict(snapshot),
        domain=REALSENSE_SNAPSHOT_HASH_DOMAIN_V1,
    )
    return CameraSemanticResolution(
        source_id=str(source_id),
        source_device_snapshot_hash=snapshot_hash,
        binding_hash=binding.contract_hash,
        roles=tuple(resolved),
        status=status,
    )


class DatasetCameraSemanticResolver:
    """Resolve canonical camera roles for direct or organized datasets.

    Construction and lookup are read-only. No cache, sidecar, or corrected
    metadata is written below ``dataset_root``.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        binding: CameraRoleBinding,
    ):
        if not isinstance(binding, CameraRoleBinding):
            raise TypeError("binding must be a CameraRoleBinding.")
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.binding = binding
        self._direct: CameraSemanticResolution | None = None
        self._by_episode: dict[int, CameraSemanticResolution] = {}
        self._snapshot_path_by_episode: dict[int, Path] = {}
        self._load()

    def _load_resolution(
        self,
        path: Path,
        *,
        source_id: str,
    ) -> CameraSemanticResolution:
        return resolve_camera_snapshot(
            _read_json_object(path),
            self.binding,
            source_id=source_id,
        )

    def _load(self) -> None:
        meta = self.dataset_root / "meta"
        materialization_path = (
            self.dataset_root / CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH
        )
        if materialization_path.is_file():
            materialization = _read_json_object(materialization_path)
            schema = str(materialization.get("schema") or "")
            if schema != CAMERA_SEMANTIC_MATERIALIZATION_SCHEMA_V1:
                raise CameraSemanticContractError(
                    "Unsupported camera semantic materialization schema "
                    f"{schema!r}: {materialization_path}"
                )
            receipt_hash = str(materialization.get("receipt_hash") or "")
            hash_payload = dict(materialization)
            hash_payload.pop("receipt_hash", None)
            expected_receipt_hash = canonical_content_hash(
                hash_payload,
                domain=CAMERA_SEMANTIC_MATERIALIZATION_HASH_DOMAIN_V1,
            )
            if receipt_hash != expected_receipt_hash:
                raise CameraSemanticContractError(
                    "Camera semantic materialization receipt hash mismatch: "
                    f"{materialization_path}"
                )
            if not bool(materialization.get("complete", False)):
                raise CameraSemanticContractError(
                    f"Camera semantic materialization is incomplete: {materialization_path}"
                )
            if str(materialization.get("binding_hash") or "") != self.binding.contract_hash:
                raise CameraSemanticContractError(
                    "Camera semantic materialization binding differs from the "
                    "requested physical camera registry."
                )
            snapshot_path = meta / "devices" / "realsense.json"
            resolution = self._load_resolution(
                snapshot_path,
                source_id=self.dataset_root.name,
            )
            if resolution.status != "identity":
                raise CameraSemanticContractError(
                    "A materialized dataset must expose canonical camera keys in "
                    "its output RealSense snapshot."
                )
            if (
                str(materialization.get("output_snapshot_hash") or "")
                != resolution.source_device_snapshot_hash
            ):
                raise CameraSemanticContractError(
                    "Camera semantic materialization output snapshot hash mismatch."
                )
            episode_rows = _read_jsonl(meta / "episodes.jsonl")
            try:
                actual_episode_indices = sorted(
                    int(row["episode_index"]) for row in episode_rows
                )
                declared_episode_indices = sorted(
                    int(value)
                    for value in materialization.get("episode_indices", ())
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise CameraSemanticContractError(
                    "Camera semantic materialization has invalid episode coverage."
                ) from exc
            if actual_episode_indices != declared_episode_indices:
                raise CameraSemanticContractError(
                    "Camera semantic materialization episode coverage differs "
                    "from meta/episodes.jsonl."
                )
            self._direct = resolution
            return
        provenance_path = meta / "provenance.jsonl"
        provenance = _read_jsonl(provenance_path)
        if not provenance:
            snapshot_path = meta / "devices" / "realsense.json"
            self._direct = self._load_resolution(
                snapshot_path,
                source_id=self.dataset_root.name,
            )
            return

        cache: dict[Path, CameraSemanticResolution] = {}
        for row_number, row in enumerate(provenance, start=1):
            try:
                episode_index = int(row["output_episode_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CameraSemanticContractError(
                    f"Provenance row {row_number} in {provenance_path} has no "
                    "integer output_episode_index."
                ) from exc
            source_root_text = str(row.get("source_dataset_root") or "").strip()
            if not source_root_text:
                raise CameraSemanticContractError(
                    f"Provenance row {row_number} in {provenance_path} has no "
                    "source_dataset_root."
                )
            source_root = Path(source_root_text).expanduser()
            source_name = source_root.name
            snapshot_path = (
                meta / "source_devices" / source_name / "realsense.json"
            )
            if not snapshot_path.is_file():
                direct_source_snapshot = (
                    source_root / "meta" / "devices" / "realsense.json"
                )
                if direct_source_snapshot.is_file():
                    snapshot_path = direct_source_snapshot
                else:
                    raise CameraSemanticContractError(
                        f"Episode {episode_index} source {source_root_text!r} has "
                        "no copied or directly accessible RealSense snapshot."
                    )
            resolved_path = snapshot_path.resolve()
            resolution = cache.get(resolved_path)
            if resolution is None:
                resolution = self._load_resolution(
                    resolved_path,
                    source_id=source_root_text,
                )
                cache[resolved_path] = resolution
            if episode_index in self._by_episode:
                raise CameraSemanticContractError(
                    f"Duplicate provenance for output episode {episode_index}."
                )
            self._by_episode[episode_index] = resolution
            self._snapshot_path_by_episode[episode_index] = resolved_path

    def resolution_for_episode(
        self,
        episode_index: int,
    ) -> CameraSemanticResolution:
        if self._direct is not None:
            return self._direct
        try:
            return self._by_episode[int(episode_index)]
        except KeyError as exc:
            raise CameraSemanticContractError(
                f"Dataset {self.dataset_root} has no camera provenance for "
                f"episode {int(episode_index)}."
            ) from exc

    def stored_key_for(self, episode_index: int, canonical_role: str) -> str:
        return self.resolution_for_episode(episode_index).stored_key_for(
            canonical_role
        )

    def audit_summary(self) -> dict[str, Any]:
        if self._direct is not None:
            resolutions = (self._direct,)
            episode_counts = {self._direct.semantic_content_hash: None}
        else:
            unique: dict[str, CameraSemanticResolution] = {}
            counts: dict[str, int] = {}
            for resolution in self._by_episode.values():
                key = resolution.semantic_content_hash
                unique[key] = resolution
                counts[key] = counts.get(key, 0) + 1
            resolutions = tuple(unique[key] for key in sorted(unique))
            episode_counts = counts
        return {
            "schema": "dexteleop.camera_semantic_dataset_audit.v1",
            "dataset_root": str(self.dataset_root),
            "binding_hash": self.binding.contract_hash,
            "episode_count": (
                None if self._direct is not None else len(self._by_episode)
            ),
            "resolutions": [
                {
                    **resolution.to_dict(),
                    "resolution_hash": resolution.semantic_content_hash,
                    "episode_count": episode_counts[
                        resolution.semantic_content_hash
                    ],
                }
                for resolution in resolutions
            ],
        }

    def require_frame_index_alignment(self) -> dict[str, Any]:
        """Require remapped views to share row-level video frame indices.

        OpenPI's ordinary dataset selects the per-camera frame index before the
        shared media pipeline remaps a canonical role to its historical stored
        key. Identity mappings need no check. A remapped episode is admitted
        only when every affected canonical/stored column is exactly aligned.
        """

        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:  # pragma: no cover - env contract
            raise CameraSemanticContractError(
                "Camera frame-index alignment audit requires pyarrow."
            ) from exc

        episodes_path = self.dataset_root / "meta" / "episodes.jsonl"
        episodes = _read_jsonl(episodes_path)
        if not episodes:
            raise CameraSemanticContractError(
                f"Camera frame-index audit has no episodes: {episodes_path}"
            )
        checked_episodes = 0
        checked_rows = 0
        checked_pairs: set[tuple[str, str]] = set()
        for row_number, episode in enumerate(episodes, start=1):
            try:
                episode_index = int(episode["episode_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CameraSemanticContractError(
                    f"Episode row {row_number} in {episodes_path} has no "
                    "integer episode_index."
                ) from exc
            resolution = self.resolution_for_episode(episode_index)
            pairs = tuple(
                (item.canonical_role, item.stored_key)
                for item in resolution.roles
                if item.canonical_role != item.stored_key
            )
            if not pairs:
                continue
            parquet_text = str(episode.get("parquet_path") or "").strip()
            if not parquet_text:
                raise CameraSemanticContractError(
                    f"Episode {episode_index} has no parquet_path for camera "
                    "frame-index alignment."
                )
            parquet_path = Path(parquet_text)
            if not parquet_path.is_absolute():
                parquet_path = self.dataset_root / parquet_path
            columns = tuple(
                dict.fromkeys(
                    f"observation.images.{key}.video_frame_index"
                    for pair in pairs
                    for key in pair
                )
            )
            try:
                table = pq.ParquetFile(parquet_path).read(columns=list(columns))
            except Exception as exc:
                raise CameraSemanticContractError(
                    f"Cannot read camera frame-index columns for episode "
                    f"{episode_index} from {parquet_path}: {exc}"
                ) from exc
            for canonical_role, stored_key in pairs:
                canonical_column = (
                    f"observation.images.{canonical_role}.video_frame_index"
                )
                stored_column = (
                    f"observation.images.{stored_key}.video_frame_index"
                )
                if not table.column(canonical_column).combine_chunks().equals(
                    table.column(stored_column).combine_chunks()
                ):
                    raise CameraSemanticContractError(
                        "Remapped wrist frame indices are not row-aligned: "
                        f"episode={episode_index} canonical_role={canonical_role!r} "
                        f"stored_key={stored_key!r}."
                    )
                checked_pairs.add((canonical_role, stored_key))
            checked_episodes += 1
            checked_rows += int(table.num_rows)
        return {
            "schema": "dexteleop.camera_frame_index_alignment_audit.v1",
            "dataset_root": str(self.dataset_root),
            "binding_hash": self.binding.contract_hash,
            "remapped_episode_count": checked_episodes,
            "remapped_row_count": checked_rows,
            "checked_pairs": [
                {"canonical_role": canonical, "stored_key": stored}
                for canonical, stored in sorted(checked_pairs)
            ],
            "aligned": True,
        }


__all__ = [
    "CAMERA_ROLE_REGISTRY_ENV",
    "CAMERA_SEMANTIC_MATERIALIZATION_RELATIVE_PATH",
    "DatasetCameraSemanticResolver",
    "REALSENSE_SNAPSHOT_HASH_DOMAIN_V1",
    "camera_role_binding_from_registry",
    "camera_role_registry_from_environment",
    "load_camera_role_binding",
    "resolve_camera_snapshot",
]
