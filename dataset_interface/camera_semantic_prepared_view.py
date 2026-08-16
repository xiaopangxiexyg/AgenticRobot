"""Strict reader for camera-corrected numeric/video prepared views."""

from __future__ import annotations

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


def _contains_tactile_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            "tactile" in str(key).lower()
            or _contains_tactile_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_tactile_key(item) for item in value)
    return False


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

    actual_files: set[str] = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CameraSemanticPreparedViewContractError(
                f"Prepared view refuses symlinks: {path}"
            )
        if path.is_file() and path != manifest_path:
            relative = path.relative_to(root)
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
    declared_files = {item.relative_path for item in manifest.files}
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
