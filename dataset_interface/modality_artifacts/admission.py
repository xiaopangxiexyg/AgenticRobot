"""Read-only admission for appending capture data to modality-tiered datasets."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

from dataset_contracts.modality_artifacts import ModalityArtifactInventoryV1
from dataset_interface.episode_curation import source_dataset_base_fingerprint

from .availability import (
    DETACHED_VERIFIED,
    INCOMPLETE_OR_CORRUPT,
    SELF_CONTAINED,
)
from .inventory import (
    _load_episode_records,
    _read_json_object,
    _resolve_existing_file,
    _safe_relative_path,
    _sha256_file,
)


MODALITY_ARTIFACT_SIDECAR_RELATIVE = Path("meta/modality_artifacts/tactile")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STORED_HASH_PROFILE_V1 = "stored_bytes_sha256_v1"


class DatasetCaptureWritableError(RuntimeError):
    """Raised when appending would invalidate a detached modality binding."""


class UnownedTactileFilesError(DatasetCaptureWritableError):
    """Raised when the hot tactile prefix contains files outside the catalog graph."""

    def __init__(self, paths: list[str] | tuple[str, ...]):
        self.paths = tuple(sorted(str(path) for path in paths))
        super().__init__(
            "cannot validate referenced hot tactile graph: tactile tree contains "
            f"unowned files: {list(self.paths[:5])}"
        )


def _discovered_tactile_files(root: Path) -> set[str]:
    tactile_root = root / "tactile"
    if not tactile_root.is_dir():
        return set()
    entries = tuple(tactile_root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise DatasetCaptureWritableError(
            "cannot validate referenced hot tactile graph: tactile tree contains "
            "symlink entries"
        )
    return {
        path.relative_to(root).as_posix()
        for path in entries
        if path.is_file()
    }


def _read_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetCaptureWritableError(
            f"cannot read modality sidecar {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise DatasetCaptureWritableError(
            f"modality sidecar is not an object: {path}"
        )
    return payload


def _catalog_has_tactile(catalog_path: Path) -> bool:
    if not catalog_path.is_file():
        return False
    try:
        lines = catalog_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError("episode record is not an object")
            tactile_paths = record.get("tactile_paths")
            if isinstance(tactile_paths, Mapping) and bool(tactile_paths):
                return True
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise DatasetCaptureWritableError(
            f"cannot inspect episode catalog before capture: {exc}"
        ) from exc
    return False


def _assert_referenced_hot_tactile_present(root: Path, catalog_path: Path) -> None:
    """Validate capture availability without applying detach-only inode rules."""

    try:
        referenced: set[str] = set()
        lines = catalog_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError(f"record {line_number} is not an object")
            tactile_paths = record.get("tactile_paths")
            if not isinstance(tactile_paths, Mapping):
                continue
            for side, value in tactile_paths.items():
                manifest_relative = _safe_relative_path(
                    value,
                    field=f"episode tactile_paths.{side}",
                    prefix="tactile",
                )
                manifest_path = _resolve_existing_file(
                    root,
                    manifest_relative,
                    field=f"tactile manifest {manifest_relative}",
                    require_single_link=False,
                )
                referenced.add(manifest_relative)
                manifest = _read_json_object(
                    manifest_path,
                    field="tactile manifest",
                )
                streams = manifest.get("streams")
                if streams is None:
                    # Compatibility manifests can be self-contained metadata-only
                    # artifacts. Their catalog-owned file is the complete graph.
                    continue
                if not isinstance(streams, Mapping) or not streams:
                    raise ValueError(
                        "tactile manifest has an invalid stream map: "
                        f"{manifest_relative}"
                    )
                for sensor_key, spec in streams.items():
                    if not isinstance(spec, Mapping):
                        raise ValueError(
                            f"stream {sensor_key!r} is not an object in {manifest_relative}"
                        )
                    payload_relative = _safe_relative_path(
                        spec.get("path"),
                        field=f"stream {sensor_key!r} path",
                        prefix="tactile",
                    )
                    _resolve_existing_file(
                        root,
                        payload_relative,
                        field=f"tactile payload {payload_relative}",
                        require_single_link=False,
                    )
                    referenced.add(payload_relative)
        extra = sorted(_discovered_tactile_files(root) - referenced)
        if extra:
            raise UnownedTactileFilesError(extra)
    except DatasetCaptureWritableError:
        raise
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise DatasetCaptureWritableError(
            f"cannot validate referenced hot tactile graph: {exc}"
        ) from exc


def _capture_inventory_hot_status(
    root: Path,
    inventory: ModalityArtifactInventoryV1,
    *,
    discovered_hot: set[str],
) -> tuple[str, str]:
    """Validate capture availability without re-hashing large payload files.

    A capture inventory produced with ``require_sealed`` binds every payload
    digest through its already-hashed tactile manifest.  Admission therefore
    verifies the manifest bytes, the complete logical graph, and every stored
    byte count.  Full payload-byte verification remains an explicit offline
    detach/integrity operation; doing it for the whole accumulated dataset on
    every writer start makes startup time proportional to historical tactile
    volume and can exceed the process-worker startup deadline.
    """

    inventory_paths = {item.logical_path for item in inventory.artifacts}
    unknown = sorted(discovered_hot - inventory_paths)
    if unknown:
        return (
            INCOMPLETE_OR_CORRUPT,
            f"hot modality tree contains unowned files: {unknown[:5]}",
        )
    missing = sorted(inventory_paths - discovered_hot)
    if missing:
        return (
            INCOMPLETE_OR_CORRUPT,
            f"hot modality artifacts are missing: {missing[:5]}",
        )

    for artifact in inventory.artifacts:
        try:
            path = _resolve_existing_file(
                root,
                artifact.logical_path,
                field=f"inventory artifact {artifact.logical_path}",
                # Capture append does not retire or rename historical files;
                # keep legacy hardlink-compatible admission behavior.
                require_single_link=False,
            )
            if int(path.stat().st_size) != int(artifact.size_bytes):
                return (
                    INCOMPLETE_OR_CORRUPT,
                    "hot modality artifact size changed: "
                    f"{artifact.logical_path}",
                )
            if artifact.role == "manifest" and _sha256_file(path) != artifact.sha256:
                return (
                    INCOMPLETE_OR_CORRUPT,
                    f"hot modality manifest changed: {artifact.logical_path}",
                )
        except (OSError, ValueError) as exc:
            return (
                INCOMPLETE_OR_CORRUPT,
                "cannot validate hot modality artifact "
                f"{artifact.logical_path}: {type(exc).__name__}: {exc}",
            )
    return SELF_CONTAINED, ""


def assert_dataset_capture_writable(dataset_root: str | Path) -> None:
    """Fail closed when append would invalidate detached tactile identity.

    Empty/new roots and complete non-tactile datasets remain compatible.  This
    function never writes sidecars or attempts transaction recovery.
    """

    root = Path(dataset_root).expanduser().resolve()
    sidecar_root = root / MODALITY_ARTIFACT_SIDECAR_RELATIVE
    pending = sidecar_root / "active_transaction.json"
    if pending.exists():
        raise DatasetCaptureWritableError(
            "dataset has an unresolved modality storage transaction; recover it offline"
        )
    catalog_path = root / "meta/episodes.jsonl"
    if not catalog_path.is_file():
        discovered = sorted(_discovered_tactile_files(root))
        if discovered:
            raise UnownedTactileFilesError(discovered)
        return
    inventory_dir = sidecar_root / "inventories"
    inventory_paths = (
        tuple(sorted(inventory_dir.glob("*.json")))
        if inventory_dir.is_dir()
        else ()
    )
    try:
        current_catalog_sha256 = _sha256_file(catalog_path)
        current_episode_count = len(_load_episode_records(catalog_path))
        current_base_fingerprint = source_dataset_base_fingerprint(root)
    except (OSError, ValueError) as exc:
        raise DatasetCaptureWritableError(
            f"cannot inspect current dataset identity before capture: {exc}"
        ) from exc

    identity_candidates: list[ModalityArtifactInventoryV1] = []
    for path in inventory_paths:
        try:
            inventory = ModalityArtifactInventoryV1.from_dict(_read_object(path))
        except Exception as exc:
            raise DatasetCaptureWritableError(
                f"invalid modality inventory sidecar {path}: {exc}"
            ) from exc
        if (
            inventory.source_episode_catalog_sha256 == current_catalog_sha256
            and inventory.source_episode_count == current_episode_count
            and inventory.source_dataset_base_fingerprint
            == current_base_fingerprint
        ):
            identity_candidates.append(inventory)

    discovered_hot = _discovered_tactile_files(root)
    inspected_current: list[tuple[ModalityArtifactInventoryV1, str, str]] = []
    for inventory in identity_candidates:
        status, reason = _capture_inventory_hot_status(
            root,
            inventory,
            discovered_hot=discovered_hot,
        )
        if status != SELF_CONTAINED:
            # Local import avoids an import cycle: evidence shares the sidecar
            # root constant from this module.  Only current, physically missing
            # candidates need receipt inspection; historical revisions do not.
            from .evidence import verified_detachment_scope

            if verified_detachment_scope(root, inventory.semantic_content_hash):
                status = DETACHED_VERIFIED
                reason = "current tactile inventory is intentionally detached"
        inspected_current.append((inventory, status, reason))

    current = [
        item
        for item in inspected_current
        if item[1] == SELF_CONTAINED
    ]
    if len(current) > 1:
        raise DatasetCaptureWritableError(
            "multiple current tactile inventories make capture admission ambiguous"
        )
    if len(current) == 1:
        _inventory, _status, _reason = current[0]
        _assert_referenced_hot_tactile_present(root, catalog_path)
        return
    detached = [
        item for item in inspected_current if item[1] == DETACHED_VERIFIED
    ]
    if detached:
        inventory, status, reason = detached[0]
        raise DatasetCaptureWritableError(
            "dataset is not physically self-contained for capture append: "
            f"{inventory.semantic_content_hash}:{status}:{reason}"
        )
    unresolved = list(inspected_current)
    if unresolved:
        # A current inventory sees a crash-left tail as an unknown hot file and
        # therefore reports INCOMPLETE_OR_CORRUPT before the catalog graph check
        # below. Surface the same typed, read-only evidence used by untracked
        # datasets, but only after the complete finalized graph itself validates.
        if all("hot modality tree contains unowned files" in item[2] for item in unresolved):
            _assert_referenced_hot_tactile_present(root, catalog_path)
        states = ", ".join(
            f"{item.semantic_content_hash}:{status}:{reason}"
            for item, status, reason in unresolved
        )
        raise DatasetCaptureWritableError(
            "dataset has modality inventory evidence but no complete current hot "
            f"artifact graph: {states}"
        )
    if not _catalog_has_tactile(catalog_path):
        discovered = sorted(_discovered_tactile_files(root))
        if discovered:
            raise UnownedTactileFilesError(discovered)
        return
    try:
        _assert_referenced_hot_tactile_present(root, catalog_path)
    except UnownedTactileFilesError:
        raise
    except Exception as exc:
        raise DatasetCaptureWritableError(
            "dataset has tactile capture references but no complete current hot "
            f"artifact graph: {exc}"
        ) from exc


def assert_dataset_capture_integrity_compatible(dataset_root: str | Path) -> bool:
    """Validate existing tactile streams for incremental integrity capture.

    Returns ``True`` when every existing stream is already capture-sealed and a
    complete inventory can be published without hashing historical payloads.
    Returns ``False`` for a valid legacy/unsealed graph.  That graph remains
    append-compatible: new Episodes may be sealed incrementally, while complete
    inventory publication stays deferred until an explicit offline full scan
    upgrades the historical manifests.

    Partially sealed, malformed, missing, or size-inconsistent streams remain
    hard failures.  Existing manifests and payloads are never rewritten here.
    """

    root = Path(dataset_root).expanduser().resolve()
    catalog_path = root / "meta/episodes.jsonl"
    if not catalog_path.is_file() or not _catalog_has_tactile(catalog_path):
        return True
    fully_sealed = True
    try:
        lines = catalog_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError(f"record {line_number} is not an object")
            tactile_paths = record.get("tactile_paths")
            if not isinstance(tactile_paths, Mapping):
                continue
            for side, value in tactile_paths.items():
                manifest_relative = _safe_relative_path(
                    value,
                    field=f"episode tactile_paths.{side}",
                    prefix="tactile",
                )
                manifest = _read_json_object(
                    _resolve_existing_file(
                        root,
                        manifest_relative,
                        field=f"tactile manifest {manifest_relative}",
                        require_single_link=False,
                    ),
                    field="tactile manifest",
                )
                streams = manifest.get("streams")
                if not isinstance(streams, Mapping) or not streams:
                    raise ValueError(
                        f"tactile manifest has no stream map: {manifest_relative}"
                    )
                for sensor_key, raw_spec in streams.items():
                    if not isinstance(raw_spec, Mapping):
                        raise ValueError(
                            f"stream {sensor_key!r} is not an object in {manifest_relative}"
                        )
                    required = {
                        "stored_size_bytes",
                        "stored_sha256",
                        "stored_hash_profile",
                    }
                    present = required.intersection(raw_spec)
                    if present and present != required:
                        raise ValueError(
                            f"stream {sensor_key!r} has partial stored-byte seal metadata"
                        )
                    payload_relative = _safe_relative_path(
                        raw_spec.get("path"),
                        field=f"stream {sensor_key!r} path",
                        prefix="tactile",
                    )
                    payload = _resolve_existing_file(
                        root,
                        payload_relative,
                        field=f"tactile payload {payload_relative}",
                        require_single_link=False,
                    )
                    if not present:
                        fully_sealed = False
                        continue
                    if str(raw_spec.get("stored_hash_profile")) != _STORED_HASH_PROFILE_V1:
                        raise ValueError(
                            f"stream {sensor_key!r} has unsupported stored hash profile"
                        )
                    stored_hash = str(raw_spec.get("stored_sha256") or "")
                    if not _SHA256_RE.fullmatch(stored_hash):
                        raise ValueError(
                            f"stream {sensor_key!r} has invalid stored SHA-256"
                        )
                    if int(raw_spec.get("stored_size_bytes", -1)) != payload.stat().st_size:
                        raise ValueError(
                            f"stream {sensor_key!r} stored size differs from payload"
                        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DatasetCaptureWritableError(
            "existing tactile graph is not compatible with incremental integrity "
            f"capture: {exc}"
        ) from exc
    return fully_sealed


__all__ = [
    "MODALITY_ARTIFACT_SIDECAR_RELATIVE",
    "DatasetCaptureWritableError",
    "UnownedTactileFilesError",
    "assert_dataset_capture_integrity_compatible",
    "assert_dataset_capture_writable",
]
