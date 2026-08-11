"""Persistence mechanics for immutable detachable-modality inventories."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from dataset_contracts.modality_artifacts import ModalityArtifactInventoryV1
from dataset_interface.modality_artifacts import build_modality_artifact_inventory


MODALITY_INVENTORY_ROOT = Path("meta/modality_artifacts/tactile/inventories")


class ModalityInventoryPublishError(RuntimeError):
    """Raised when an immutable inventory cannot be published safely."""


def modality_inventory_path(
    dataset_root: str | Path,
    inventory_hash: str,
) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    return root / MODALITY_INVENTORY_ROOT / f"{str(inventory_hash)}.json"


def _read_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModalityInventoryPublishError(f"invalid modality inventory {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ModalityInventoryPublishError(f"modality inventory is not an object: {path}")
    return payload


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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


def publish_modality_inventory(
    dataset_root: str | Path,
    inventory: ModalityArtifactInventoryV1,
) -> Path:
    path = modality_inventory_path(dataset_root, inventory.semantic_content_hash)
    if path.exists():
        existing = ModalityArtifactInventoryV1.from_dict(_read_object(path))
        if existing != inventory:
            raise ModalityInventoryPublishError(f"inventory sidecar collision: {path}")
        return path
    _atomic_write(path, inventory.to_dict())
    return path


def build_and_publish_modality_inventory(
    dataset_root: str | Path,
    *,
    payload_hash_mode: str = "require_sealed",
) -> ModalityArtifactInventoryV1:
    inventory = build_modality_artifact_inventory(
        dataset_root,
        payload_hash_mode=payload_hash_mode,
    )
    publish_modality_inventory(dataset_root, inventory)
    return inventory


def modality_inventory_tracking_enabled(dataset_root: str | Path) -> bool:
    """Return whether this dataset opted into immutable modality revisions."""

    root = Path(dataset_root).expanduser().resolve()
    inventory_dir = root / MODALITY_INVENTORY_ROOT
    if inventory_dir.is_dir() and any(inventory_dir.glob("*.json")):
        return True
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        return False
    try:
        info = _read_object(info_path)
    except ModalityInventoryPublishError:
        return False
    return bool(info.get("tactile_artifact_integrity", False))


def publish_tracked_modality_inventory(
    dataset_root: str | Path,
    *,
    payload_hash_mode: str = "prefer_sealed",
) -> str | None:
    """Publish a new current revision only for datasets already tracking one."""

    if not modality_inventory_tracking_enabled(dataset_root):
        return None
    inventory = build_and_publish_modality_inventory(
        dataset_root,
        payload_hash_mode=payload_hash_mode,
    )
    return inventory.semantic_content_hash


def load_modality_inventory(
    dataset_root: str | Path,
    inventory_hash: str,
) -> ModalityArtifactInventoryV1:
    path = modality_inventory_path(dataset_root, str(inventory_hash))
    inventory = ModalityArtifactInventoryV1.from_dict(_read_object(path))
    if inventory.semantic_content_hash != str(inventory_hash):
        raise ModalityInventoryPublishError("inventory filename/hash mismatch")
    return inventory


__all__ = [
    "MODALITY_INVENTORY_ROOT",
    "ModalityInventoryPublishError",
    "build_and_publish_modality_inventory",
    "load_modality_inventory",
    "modality_inventory_tracking_enabled",
    "modality_inventory_path",
    "publish_modality_inventory",
    "publish_tracked_modality_inventory",
]
