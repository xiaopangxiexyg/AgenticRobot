"""Strict reader for camera-corrected numeric/video prepared views."""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
from typing import Any

from dataset_contracts.camera_semantic_prepared_view import (
    CAMERA_SEMANTIC_PREPARED_VIEW_RELATIVE_PATH,
    CAMERA_SEMANTIC_VIDEO_INTEGRITY_EXCLUSIONS_RELATIVE_PATH,
    CameraSemanticPreparedViewContractError,
    CameraSemanticPreparedViewManifestV1,
    CameraSemanticVideoIntegrityExclusionsV1,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contains_tactile_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            # These two fields are camera-only provenance: they record that
            # tactile artifacts were not copied and tactile columns were
            # removed during materialization; neither is a tactile payload.
            if "tactile" in key_text and key_text not in {
                "tactile_artifacts_copied",
                "tactile_columns_removed",
            }:
                return True
            if key_text == "excluded_modalities" and item == ["tactile"]:
                continue
            if _contains_tactile_key(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_tactile_key(item) for item in value)
    if isinstance(value, str) and "tactile" in value.lower():
        # Source names/paths such as ``*_no_tactile`` are provenance labels,
        # not retained tactile content.
        return "no_tactile" not in value.lower()
    return False


def _validated_by_date_entry_sidecar(
    root: Path,
    manifest_hash: str,
) -> Path | None:
    """Validate the reserved by-date catalog sidecar outside payload inventory.

    ``ENTRY.json`` is written after a camera-only prepared view is sealed.  It
    binds the prepared-view manifest hash and includes its own semantic hash,
    so including it in the prepared-view manifest would create a circular
    identity.  Accept only the exact, self-consistent control-plane sidecar;
    every other undeclared file remains an inventory error.
    """

    path = root / "ENTRY.json"
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise CameraSemanticPreparedViewContractError(
            f"Prepared-view by-date entry must be a regular file: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CameraSemanticPreparedViewContractError(
            f"Cannot read prepared-view by-date entry: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CameraSemanticPreparedViewContractError(
            "Prepared-view by-date entry must be an object."
        )
    # Older direct by-date projections predate the hash-bound ENTRY sidecar.
    # Keep their catalog entry usable only when it is the exact non-tactile
    # direct-root schema; the prepared-view manifest still authenticates the
    # projected payload and inventory below.
    if (
        str(payload.get("schema") or "")
        == "realmandmtac.by_date_direct_entry.v1"
        and "entry_semantic_sha256" not in payload
    ):
        if payload.get("layout") != "direct_dataset_root":
            raise CameraSemanticPreparedViewContractError(
                "Legacy prepared-view by-date entry must identify a direct root."
            )
        if str(payload.get("directory_name") or "") != root.name:
            raise CameraSemanticPreparedViewContractError(
                "Legacy prepared-view by-date entry differs from its root."
            )
        if payload.get("modalities") != ["numeric", "rgb_video", "wrist_depth"]:
            raise CameraSemanticPreparedViewContractError(
                "Legacy prepared-view by-date entry has an unsupported modality set."
            )
        if payload.get("excluded_modalities") != ["tactile"]:
            raise CameraSemanticPreparedViewContractError(
                "Legacy prepared-view by-date entry must exclude tactile."
            )
        legacy_payload = dict(payload)
        legacy_payload.pop("excluded_modalities", None)
        if _contains_tactile_key(legacy_payload):
            raise CameraSemanticPreparedViewContractError(
                "Legacy prepared-view by-date entry contains tactile-specific metadata."
            )
        return path
    semantic_payload = dict(payload)
    supplied_hash = str(
        semantic_payload.pop("entry_semantic_sha256", "")
    ).strip().lower()
    if supplied_hash != _canonical_json_sha256(semantic_payload):
        raise CameraSemanticPreparedViewContractError(
            "Prepared-view by-date entry semantic hash mismatch."
        )
    if payload.get("direct_dataset_root") is not True:
        raise CameraSemanticPreparedViewContractError(
            "Prepared-view by-date entry must identify a direct dataset root."
        )
    for field in ("dataset_relative_path", "destination_relative_path"):
        if str(payload.get(field) or "") != root.name:
            raise CameraSemanticPreparedViewContractError(
                f"Prepared-view by-date entry {field} differs from its root."
            )
    if (
        str(payload.get("prepared_view_manifest_sha256") or "").strip().lower()
        != manifest_hash
    ):
        raise CameraSemanticPreparedViewContractError(
            "Prepared-view by-date entry binds a different manifest."
        )
    if _contains_tactile_key(payload):
        raise CameraSemanticPreparedViewContractError(
            "Prepared-view by-date entry contains tactile-specific metadata."
        )
    return path


def _validated_stage_annotation_writer_lock_sidecar(root: Path) -> Path | None:
    """Accept only an idle, payload-free annotation compatibility lock.

    The annotation store intentionally leaves this zero-byte flock inode in
    place after an editor exits.  It is runtime coordination state, not camera
    payload, so it cannot be part of the immutable prepared-view inventory.
    Episode-scoped lock directories and a legacy exclusive flock still mean
    that an annotation writer may be active and remain fail-closed.
    """

    path = root / "meta/progress/stage_annotations/.writer.lock"
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise CameraSemanticPreparedViewContractError(
            f"Prepared-view annotation writer lock must be a regular file: {path}"
        )
    if path.stat().st_size != 0:
        raise CameraSemanticPreparedViewContractError(
            "Prepared-view annotation writer lock must remain payload-free."
        )

    episode_lock_root = root / "meta/progress/stage_annotation_locks"
    if episode_lock_root.exists() or episode_lock_root.is_symlink():
        if episode_lock_root.is_symlink() or not episode_lock_root.is_dir():
            raise CameraSemanticPreparedViewContractError(
                "Prepared-view annotation episode lock root is invalid."
            )
        active_lock_entries: list[str] = []
        for entry in sorted(episode_lock_root.iterdir()):
            name = entry.name
            stale_marker = (
                name.startswith(".")
                and (".lock.stale-" in name or ".lock.releasing-" in name)
            )
            if not stale_marker:
                active_lock_entries.append(name)
                continue
            if entry.is_symlink() or not entry.is_dir():
                active_lock_entries.append(name)
                continue
            owner_path = entry / "owner.json"
            if owner_path.is_symlink() or not owner_path.is_file():
                active_lock_entries.append(name)
                continue
            try:
                owner = json.loads(owner_path.read_text(encoding="utf-8"))
                owner_pid = int(owner.get("pid", 0))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                active_lock_entries.append(name)
                continue
            if (
                owner.get("schema")
                != "dexteleop.stage_annotation_episode_lock.local.v1"
                or owner_pid <= 0
                or Path("/proc").joinpath(str(owner_pid)).exists()
            ):
                active_lock_entries.append(name)
        if active_lock_entries:
            raise CameraSemanticPreparedViewContractError(
                "Prepared-view annotations have active or unreconciled episode locks: "
                + ", ".join(active_lock_entries[:5])
            )

    try:
        with path.open("rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except BlockingIOError as exc:
        raise CameraSemanticPreparedViewContractError(
            "Prepared-view annotations are held by a legacy exclusive writer."
        ) from exc
    except OSError as exc:
        raise CameraSemanticPreparedViewContractError(
            f"Cannot validate prepared-view annotation writer lock: {exc}"
        ) from exc
    return path


def _accept_finalized_build_audit_size_drift(
    root: Path,
    relative_path: str,
    manifest: CameraSemanticPreparedViewManifestV1,
    *,
    verify_payload_hashes: bool,
) -> bool:
    """Accept a post-seal control-plane audit rewrite only in fast-read mode.

    The stage1 by-date materializer writes ``build_audit.json`` after sealing
    the camera-view manifest.  The audit binds the same manifest hash, but its
    final JSON formatting/fields can change the file size afterward.  This is
    not a training payload; keep the full verifier strict and allow only this
    explicitly proven, non-mutating audit drift for the training fast path.
    """

    if verify_payload_hashes or relative_path != "meta/build_audit.json":
        return False
    audit_path = root / relative_path
    entry_path = root / "ENTRY.json"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        entry = json.loads(entry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    verification = audit.get("post_publish_verification")
    return (
        audit.get("schema") == "realmandmtac.by_date_stage1_merge_audit.v1"
        and audit.get("status") == "PASS_VERIFIED"
        and audit.get("source_mutated") is False
        and isinstance(verification, dict)
        and verification.get("status") == "PASS"
        and int(verification.get("nonregular_count", -1)) == 0
        and isinstance(entry.get("copy_verification"), dict)
        and entry["copy_verification"].get("all_regular_files") is True
        and str(audit.get("prepared_view_manifest_hash") or "").lower()
        == manifest.manifest_hash
        and str(entry.get("prepared_view_manifest_sha256") or "").lower()
        == manifest.manifest_hash
    )


def camera_semantic_prepared_view_manifest_path(
    dataset_root: str | Path,
) -> Path:
    return (
        Path(dataset_root).expanduser().resolve()
        / CAMERA_SEMANTIC_PREPARED_VIEW_RELATIVE_PATH
    )


def is_camera_semantic_prepared_view(dataset_root: str | Path) -> bool:
    return camera_semantic_prepared_view_manifest_path(dataset_root).is_file()


def load_camera_semantic_video_integrity_exclusions(
    dataset_root: str | Path,
    *,
    verify_payload_hashes: bool = True,
) -> tuple[
    CameraSemanticPreparedViewManifestV1,
    CameraSemanticVideoIntegrityExclusionsV1 | None,
]:
    """Load a prepared view and its optional, hash-bound exclusion receipt."""

    root = Path(dataset_root).expanduser().resolve()
    manifest = load_camera_semantic_prepared_view(
        root,
        verify_payload_hashes=verify_payload_hashes,
    )
    if not manifest.excluded_episode_indices:
        return manifest, None
    payload = json.loads(
        (
            root
            / CAMERA_SEMANTIC_VIDEO_INTEGRITY_EXCLUSIONS_RELATIVE_PATH
        ).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise CameraSemanticPreparedViewContractError(
            "Video-integrity exclusions receipt must be an object."
        )
    receipt = CameraSemanticVideoIntegrityExclusionsV1.from_dict(payload)
    # load_camera_semantic_prepared_view already verifies this relationship;
    # keep the assertion local so callers receive a complete typed contract.
    if receipt.receipt_hash != manifest.video_integrity_exclusions_hash:
        raise CameraSemanticPreparedViewContractError(
            "Video-integrity exclusions receipt differs from the prepared-view manifest."
        )
    return manifest, receipt


def load_camera_semantic_prepared_view(
    dataset_root: str | Path,
    *,
    verify_payload_hashes: bool = True,
) -> CameraSemanticPreparedViewManifestV1:
    """Load and validate a self-contained non-tactile training projection."""

    root = Path(dataset_root).expanduser().resolve()
    manifest_path = root / CAMERA_SEMANTIC_PREPARED_VIEW_RELATIVE_PATH
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise CameraSemanticPreparedViewContractError(
            f"Camera-semantic prepared-view manifest is missing: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CameraSemanticPreparedViewContractError(
            f"Cannot read camera-semantic prepared-view manifest: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CameraSemanticPreparedViewContractError(
            "Camera-semantic prepared-view manifest must be an object."
        )
    manifest = CameraSemanticPreparedViewManifestV1.from_dict(payload)
    exclusions_path = (
        root / CAMERA_SEMANTIC_VIDEO_INTEGRITY_EXCLUSIONS_RELATIVE_PATH
    )
    exclusions = None
    if manifest.excluded_episode_indices:
        if not exclusions_path.is_file() or exclusions_path.is_symlink():
            raise CameraSemanticPreparedViewContractError(
                "Prepared view excludes corrupt-video episodes but its "
                "integrity receipt is missing."
            )
        try:
            exclusions_payload = json.loads(
                exclusions_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise CameraSemanticPreparedViewContractError(
                f"Cannot read video-integrity exclusions receipt: {exc}"
            ) from exc
        if not isinstance(exclusions_payload, dict):
            raise CameraSemanticPreparedViewContractError(
                "Video-integrity exclusions receipt must be an object."
            )
        exclusions = CameraSemanticVideoIntegrityExclusionsV1.from_dict(
            exclusions_payload
        )
        if (
            exclusions.receipt_hash
            != manifest.video_integrity_exclusions_hash
            or exclusions.source_dataset_name != manifest.source_dataset_name
            or exclusions.source_dataset_identity_hash
            != manifest.source_dataset_identity_hash
            or exclusions.selected_source_inventory_hash
            != manifest.selected_source_inventory_hash
            or exclusions.included_episode_indices != manifest.episode_indices
            or exclusions.excluded_episode_indices
            != manifest.excluded_episode_indices
        ):
            raise CameraSemanticPreparedViewContractError(
                "Video-integrity exclusions receipt differs from the prepared-view manifest."
            )
    elif exclusions_path.exists() or exclusions_path.is_symlink():
        raise CameraSemanticPreparedViewContractError(
            "Prepared view contains an undeclared video-integrity exclusions receipt."
        )

    by_date_entry_path = _validated_by_date_entry_sidecar(
        root,
        manifest.manifest_hash,
    )
    annotation_writer_lock_path = (
        _validated_stage_annotation_writer_lock_sidecar(root)
    )

    declared_files = {item.relative_path for item in manifest.files}
    entry_relative = (
        None
        if by_date_entry_path is None
        else by_date_entry_path.relative_to(root).as_posix()
    )
    actual_files: set[str] = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CameraSemanticPreparedViewContractError(
                f"Prepared view refuses symlinks: {path}"
            )
        relative = path.relative_to(root) if path.is_file() else None
        is_declared_entry_payload = (
            entry_relative is not None
            and relative is not None
            and relative.as_posix() == entry_relative
            and entry_relative in declared_files
        )
        is_mutable_annotation_sidecar = (
            relative is not None
            and len(relative.parts) >= 3
            and relative.parts[:3]
            in {
                ("meta", "progress", "stage_annotations"),
                ("meta", "progress", "stage_annotation_drafts"),
            }
            and relative.as_posix() not in declared_files
        )
        if (
            path.is_file()
            and path != manifest_path
            and path != annotation_writer_lock_path
            and (path != by_date_entry_path or is_declared_entry_payload)
            and not is_mutable_annotation_sidecar
        ):
            assert relative is not None
            if any(part.endswith(".lock") for part in relative.parts):
                raise CameraSemanticPreparedViewContractError(
                    "Camera-semantic prepared view contains a runtime lock file: "
                    f"{relative.as_posix()}"
                )
            if (
                relative.parts
                and relative.parts[0] == "meta"
                and any(
                    "tactile" in part.lower() for part in relative.parts[1:]
                )
            ):
                raise CameraSemanticPreparedViewContractError(
                    "Camera-semantic prepared view contains tactile-specific "
                    f"metadata: {relative.as_posix()}"
                )
            actual_files.add(relative.as_posix())
    if actual_files != declared_files:
        raise CameraSemanticPreparedViewContractError(
            "Prepared-view file inventory differs from the manifest: "
            f"missing={sorted(declared_files - actual_files)[:10]} "
            f"unexpected={sorted(actual_files - declared_files)[:10]}."
        )
    for item in manifest.files:
        path = root / item.relative_path
        stat = path.stat()
        if int(stat.st_size) != item.size_bytes:
            if _accept_finalized_build_audit_size_drift(
                root,
                item.relative_path,
                manifest,
                verify_payload_hashes=verify_payload_hashes,
            ):
                continue
            raise CameraSemanticPreparedViewContractError(
                f"Prepared-view file size mismatch: {item.relative_path}"
            )
        if verify_payload_hashes and _sha256_file(path) != item.sha256:
            raise CameraSemanticPreparedViewContractError(
                f"Prepared-view file hash mismatch: {item.relative_path}"
            )

    episodes_path = root / "meta/episodes.jsonl"
    episode_indices: list[int] = []
    for line_number, line in enumerate(
        episodes_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row: Any = json.loads(line)
            episode_indices.append(int(row["episode_index"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise CameraSemanticPreparedViewContractError(
                f"Invalid prepared-view episode row {line_number}."
            ) from exc
        if _contains_tactile_key(row):
            raise CameraSemanticPreparedViewContractError(
                "Prepared-view episode records cannot retain tactile fields."
            )
    if tuple(sorted(episode_indices)) != manifest.episode_indices:
        raise CameraSemanticPreparedViewContractError(
            "Prepared-view episode coverage differs from the manifest."
        )
    if exclusions is not None and set(episode_indices) & set(
        exclusions.excluded_episode_indices
    ):
        raise CameraSemanticPreparedViewContractError(
            "A video-integrity-excluded episode remains in the prepared-view catalog."
        )
    if (root / "tactile").exists():
        raise CameraSemanticPreparedViewContractError(
            "Camera-semantic prepared view unexpectedly contains tactile/."
        )
    for relative in (Path("meta/info.json"), Path("meta/modality_validity.json")):
        path = root / relative
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CameraSemanticPreparedViewContractError(
                f"Cannot read prepared-view metadata {relative}: {exc}"
            ) from exc
        if _contains_tactile_key(value):
            raise CameraSemanticPreparedViewContractError(
                "Camera-semantic prepared view retains tactile keys in "
                f"{relative.as_posix()}."
            )
    return manifest


__all__ = [
    "camera_semantic_prepared_view_manifest_path",
    "is_camera_semantic_prepared_view",
    "load_camera_semantic_prepared_view",
    "load_camera_semantic_video_integrity_exclusions",
]
