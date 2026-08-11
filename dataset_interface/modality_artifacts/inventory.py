"""Build and verify immutable artifact graphs for detachable modalities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

from dataset_contracts.modality_artifacts import (
    ModalityArtifactInventoryV1,
    ModalityArtifactRecordV1,
)
from dataset_interface.episode_curation import source_dataset_base_fingerprint


class ModalityArtifactInventoryError(ValueError):
    """Raised when an on-disk modality graph is incomplete or ambiguous."""


PAYLOAD_HASH_SCAN = "scan"
PAYLOAD_HASH_PREFER_SEALED = "prefer_sealed"
PAYLOAD_HASH_REQUIRE_SEALED = "require_sealed"
PAYLOAD_HASH_MODES = (
    PAYLOAD_HASH_SCAN,
    PAYLOAD_HASH_PREFER_SEALED,
    PAYLOAD_HASH_REQUIRE_SEALED,
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STORED_HASH_PROFILE_V1 = "stored_bytes_sha256_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: Any, *, field: str, prefix: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(text)
    normalized = path.as_posix()
    if (
        not text
        or path.is_absolute()
        or ".." in path.parts
        or text != normalized
        or not normalized.startswith(prefix.rstrip("/") + "/")
    ):
        raise ModalityArtifactInventoryError(
            f"{field} must be a normalized path below {prefix!r}: {text!r}"
        )
    return normalized


def _resolve_existing_file(
    root: Path,
    relative: str,
    *,
    field: str,
    require_single_link: bool = True,
) -> Path:
    lexical_path = root / relative
    cursor = lexical_path
    while cursor != root:
        if cursor.is_symlink():
            raise ModalityArtifactInventoryError(
                f"{field} must not traverse a symlink: {relative!r}"
            )
        cursor = cursor.parent
    path = lexical_path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ModalityArtifactInventoryError(
            f"{field} escapes dataset root: {relative!r}"
        ) from exc
    if not path.is_file():
        raise ModalityArtifactInventoryError(
            f"{field} is missing or not a file: {relative!r}"
        )
    if require_single_link and int(path.stat().st_nlink) != 1:
        raise ModalityArtifactInventoryError(
            f"{field} must have link_count=1 before detachable storage planning: "
            f"{relative!r}"
        )
    return path


def _read_json_object(path: Path, *, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModalityArtifactInventoryError(f"invalid {field}: {path}") from exc
    if not isinstance(value, Mapping):
        raise ModalityArtifactInventoryError(f"{field} must be an object: {path}")
    return value


def _load_episode_records(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        raise ModalityArtifactInventoryError("missing meta/episodes.jsonl")
    records: list[Mapping[str, Any]] = []
    seen_indices: set[int] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ModalityArtifactInventoryError(
            f"cannot read episode catalog: {path}"
        ) from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ModalityArtifactInventoryError(
                f"invalid JSON in {path}:{line_number}"
            ) from exc
        if not isinstance(value, Mapping):
            raise ModalityArtifactInventoryError(
                f"episode record must be an object in {path}:{line_number}"
            )
        try:
            index = int(value["episode_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ModalityArtifactInventoryError(
                f"invalid episode_index in {path}:{line_number}"
            ) from exc
        if index < 0 or index in seen_indices:
            raise ModalityArtifactInventoryError(
                f"duplicate or negative episode_index {index}"
            )
        seen_indices.add(index)
        records.append(value)
    return tuple(records)


def _record_for_file(
    path: Path,
    *,
    logical_path: str,
    role: str,
    episode_index: int | None,
    side: str | None,
    sensor_key: str | None,
    frame_count: int | None,
    sealed_size_bytes: int | None = None,
    sealed_sha256: str | None = None,
) -> ModalityArtifactRecordV1:
    stat = path.stat()
    if sealed_size_bytes is not None or sealed_sha256 is not None:
        if sealed_size_bytes is None or sealed_sha256 is None:
            raise ModalityArtifactInventoryError(
                f"incomplete sealed integrity metadata for {logical_path}"
            )
        size_bytes = int(sealed_size_bytes)
        sha256 = str(sealed_sha256)
        if size_bytes < 0 or not _SHA256_RE.fullmatch(sha256):
            raise ModalityArtifactInventoryError(
                f"invalid sealed integrity metadata for {logical_path}"
            )
        if int(stat.st_size) != size_bytes:
            raise ModalityArtifactInventoryError(
                f"sealed byte count differs from file size for {logical_path}"
            )
    else:
        size_bytes = int(stat.st_size)
        sha256 = _sha256_file(path)
    return ModalityArtifactRecordV1(
        logical_path=logical_path,
        role=role,
        size_bytes=size_bytes,
        sha256=sha256,
        episode_index=episode_index,
        side=side,
        sensor_key=sensor_key,
        frame_count=frame_count,
    )


def build_modality_artifact_inventory(
    dataset_root: str | Path,
    *,
    modality: str = "tactile",
    payload_hash_mode: str = PAYLOAD_HASH_SCAN,
    require_single_link: bool = True,
) -> ModalityArtifactInventoryV1:
    """Build a read-only inventory from a self-contained finalized dataset.

    M1 intentionally supports tactile only.  The persisted contract is generic;
    modality-specific graph discovery stays here in ``dataset_interface``.
    """

    if str(modality).strip() != "tactile":
        raise ModalityArtifactInventoryError(
            "M1 inventory discovery currently supports modality='tactile' only"
        )
    hash_mode = str(payload_hash_mode or "").strip().lower()
    if hash_mode not in PAYLOAD_HASH_MODES:
        raise ModalityArtifactInventoryError(
            f"unsupported payload_hash_mode: {payload_hash_mode!r}"
        )
    root = Path(dataset_root).expanduser().resolve()
    catalog_path = root / "meta" / "episodes.jsonl"
    episodes = _load_episode_records(catalog_path)
    artifacts: dict[str, ModalityArtifactRecordV1] = {}

    def add(record: ModalityArtifactRecordV1) -> None:
        if record.logical_path in artifacts:
            raise ModalityArtifactInventoryError(
                f"duplicate tactile artifact reference: {record.logical_path}"
            )
        artifacts[record.logical_path] = record

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        tactile_paths = episode.get("tactile_paths", {})
        if not isinstance(tactile_paths, Mapping):
            raise ModalityArtifactInventoryError(
                f"episode {episode_index} tactile_paths must be an object"
            )
        for side, manifest_value in sorted(tactile_paths.items(), key=lambda item: str(item[0])):
            manifest_relative = _safe_relative_path(
                manifest_value,
                field=f"episode {episode_index} tactile_paths.{side}",
                prefix="tactile",
            )
            manifest_path = _resolve_existing_file(
                root,
                manifest_relative,
                field=f"episode {episode_index} tactile manifest",
                require_single_link=bool(require_single_link),
            )
            manifest = _read_json_object(manifest_path, field="tactile manifest")
            try:
                manifest_frame_count = int(manifest.get("frame_count", -1))
            except (TypeError, ValueError) as exc:
                raise ModalityArtifactInventoryError(
                    f"invalid frame_count in {manifest_relative}"
                ) from exc
            if manifest_frame_count < 0:
                raise ModalityArtifactInventoryError(
                    f"invalid frame_count in {manifest_relative}"
                )
            add(
                _record_for_file(
                    manifest_path,
                    logical_path=manifest_relative,
                    role="manifest",
                    episode_index=episode_index,
                    side=str(side),
                    sensor_key=None,
                    frame_count=manifest_frame_count,
                )
            )
            streams = manifest.get("streams", {})
            if not isinstance(streams, Mapping) or not streams:
                raise ModalityArtifactInventoryError(
                    f"tactile manifest has no stream map: {manifest_relative}"
                )
            for sensor_key, spec in sorted(streams.items(), key=lambda item: str(item[0])):
                if not isinstance(spec, Mapping):
                    raise ModalityArtifactInventoryError(
                        f"stream {sensor_key!r} in {manifest_relative} must be an object"
                    )
                payload_relative = _safe_relative_path(
                    spec.get("path"),
                    field=f"stream {sensor_key!r} path",
                    prefix="tactile",
                )
                payload_path = _resolve_existing_file(
                    root,
                    payload_relative,
                    field=f"stream {sensor_key!r} payload",
                    require_single_link=bool(require_single_link),
                )
                try:
                    stream_frame_count = int(spec.get("frame_count", -1))
                except (TypeError, ValueError) as exc:
                    raise ModalityArtifactInventoryError(
                        f"invalid stream frame_count for {sensor_key!r}"
                    ) from exc
                if stream_frame_count != manifest_frame_count:
                    raise ModalityArtifactInventoryError(
                        f"stream {sensor_key!r} frame_count {stream_frame_count} differs "
                        f"from manifest frame_count {manifest_frame_count}"
                    )
                sealed_size: int | None = None
                sealed_hash: str | None = None
                sealed_fields = {
                    "stored_size_bytes",
                    "stored_sha256",
                    "stored_hash_profile",
                }
                present_sealed_fields = sealed_fields.intersection(spec)
                if present_sealed_fields:
                    if present_sealed_fields != sealed_fields:
                        raise ModalityArtifactInventoryError(
                            f"stream {sensor_key!r} has incomplete stored integrity metadata"
                        )
                    if str(spec.get("stored_hash_profile")) != _STORED_HASH_PROFILE_V1:
                        raise ModalityArtifactInventoryError(
                            f"stream {sensor_key!r} has unsupported stored hash profile"
                        )
                    try:
                        sealed_size = int(spec.get("stored_size_bytes"))
                    except (TypeError, ValueError) as exc:
                        raise ModalityArtifactInventoryError(
                            f"stream {sensor_key!r} has invalid stored_size_bytes"
                        ) from exc
                    sealed_hash = str(spec.get("stored_sha256") or "")
                elif hash_mode == PAYLOAD_HASH_REQUIRE_SEALED:
                    raise ModalityArtifactInventoryError(
                        f"stream {sensor_key!r} has no sealed stored-byte hash"
                    )
                if hash_mode == PAYLOAD_HASH_SCAN:
                    sealed_size = None
                    sealed_hash = None
                add(
                    _record_for_file(
                        payload_path,
                        logical_path=payload_relative,
                        role="payload",
                        episode_index=episode_index,
                        side=str(side),
                        sensor_key=str(sensor_key),
                        frame_count=stream_frame_count,
                        sealed_size_bytes=sealed_size,
                        sealed_sha256=sealed_hash,
                    )
                )

    if not artifacts:
        raise ModalityArtifactInventoryError(
            "dataset has no referenced tactile artifacts"
        )
    tactile_root = root / "tactile"
    unsafe_symlinks = sorted(
        path.relative_to(root).as_posix()
        for path in tactile_root.rglob("*")
        if path.is_symlink()
    )
    if unsafe_symlinks:
        raise ModalityArtifactInventoryError(
            f"tactile tree contains symlink entries: {unsafe_symlinks[:5]}"
        )
    discovered = {
        path.relative_to(root).as_posix()
        for path in tactile_root.rglob("*")
        if path.is_file()
    }
    referenced = set(artifacts)
    extra = sorted(discovered - referenced)
    missing = sorted(referenced - discovered)
    if missing:
        raise ModalityArtifactInventoryError(
            f"referenced tactile artifacts are missing: {missing[:5]}"
        )
    if extra:
        raise ModalityArtifactInventoryError(
            f"unowned tactile files are not inventory-safe: {extra[:5]}"
        )
    ordered = tuple(artifacts[path] for path in sorted(artifacts))
    return ModalityArtifactInventoryV1(
        modality_key="tactile",
        source_dataset_base_fingerprint=source_dataset_base_fingerprint(root),
        source_episode_catalog_sha256=_sha256_file(catalog_path),
        source_episode_count=len(episodes),
        logical_prefixes=("tactile",),
        artifacts=ordered,
    )


def verify_modality_artifact_inventory(
    dataset_root: str | Path,
    inventory: ModalityArtifactInventoryV1,
    *,
    artifact_root: str | Path | None = None,
) -> None:
    """Verify catalog identity and every artifact byte without modifying data."""

    root = Path(dataset_root).expanduser().resolve()
    catalog_path = root / "meta" / "episodes.jsonl"
    episodes = _load_episode_records(catalog_path)
    if source_dataset_base_fingerprint(root) != inventory.source_dataset_base_fingerprint:
        raise ModalityArtifactInventoryError("source dataset base fingerprint changed")
    if _sha256_file(catalog_path) != inventory.source_episode_catalog_sha256:
        raise ModalityArtifactInventoryError("source episode catalog changed")
    if len(episodes) != inventory.source_episode_count:
        raise ModalityArtifactInventoryError("source episode count changed")
    storage_root = (
        root
        if artifact_root is None
        else Path(artifact_root).expanduser().resolve()
    )
    for artifact in inventory.artifacts:
        path = _resolve_existing_file(
            storage_root,
            artifact.logical_path,
            field=f"inventory artifact {artifact.logical_path}",
        )
        stat = path.stat()
        if int(stat.st_size) != artifact.size_bytes:
            raise ModalityArtifactInventoryError(
                f"artifact size changed: {artifact.logical_path}"
            )
        if _sha256_file(path) != artifact.sha256:
            raise ModalityArtifactInventoryError(
                f"artifact hash changed: {artifact.logical_path}"
            )


__all__ = [
    "ModalityArtifactInventoryError",
    "PAYLOAD_HASH_MODES",
    "PAYLOAD_HASH_PREFER_SEALED",
    "PAYLOAD_HASH_REQUIRE_SEALED",
    "PAYLOAD_HASH_SCAN",
    "build_modality_artifact_inventory",
    "verify_modality_artifact_inventory",
]
