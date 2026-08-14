"""Strict reader for camera-corrected numeric/video prepared views."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dataset_contracts.camera_semantic_prepared_view import (
    CAMERA_SEMANTIC_PREPARED_VIEW_RELATIVE_PATH,
    CameraSemanticPreparedViewContractError,
    CameraSemanticPreparedViewManifestV1,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def camera_semantic_prepared_view_manifest_path(
    dataset_root: str | Path,
) -> Path:
    return (
        Path(dataset_root).expanduser().resolve()
        / CAMERA_SEMANTIC_PREPARED_VIEW_RELATIVE_PATH
    )


def is_camera_semantic_prepared_view(dataset_root: str | Path) -> bool:
    return camera_semantic_prepared_view_manifest_path(dataset_root).is_file()


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

    actual_files: set[str] = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CameraSemanticPreparedViewContractError(
                f"Prepared view refuses symlinks: {path}"
            )
        if path.is_file() and path != manifest_path:
            actual_files.add(path.relative_to(root).as_posix())
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
        if any("tactile" in str(key).lower() for key in row):
            raise CameraSemanticPreparedViewContractError(
                "Prepared-view episode records cannot retain tactile fields."
            )
    if tuple(sorted(episode_indices)) != manifest.episode_indices:
        raise CameraSemanticPreparedViewContractError(
            "Prepared-view episode coverage differs from the manifest."
        )
    if (root / "tactile").exists():
        raise CameraSemanticPreparedViewContractError(
            "Camera-semantic prepared view unexpectedly contains tactile/."
        )
    return manifest


__all__ = [
    "camera_semantic_prepared_view_manifest_path",
    "is_camera_semantic_prepared_view",
    "load_camera_semantic_prepared_view",
]
